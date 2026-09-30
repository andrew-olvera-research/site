#!/usr/bin/env python3
"""Bounded one-window-lookahead vision DAgger for the plant teacher."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from scripts.train_vision_student import (
    SCALE, collect_batch, evaluate, load_setup, local_path, parse_stage,
    wandb_evaluation_metrics,
)
from starscream.wandb import init_wandb
from starscream.student_learning import (make_readout_predictor,calibrate_readout,readout_losses,
    collection_beta,learning_rate_for_round,selection_key)


class Reservoir:
    def __init__(self, tracks, per_track, seed):
        self.tracks = list(tracks)
        self.limit = int(per_track)
        self.rng = np.random.default_rng(seed)
        self.rows = {track: [] for track in tracks}
        self.seen = defaultdict(int)

    def append(self, track, row):
        self.seen[track] += 1
        rows = self.rows[track]
        if len(rows) < self.limit:
            rows.append(row)
        else:
            index = int(self.rng.integers(self.seen[track]))
            if index < self.limit:
                rows[index] = row

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=False)
        rows = [(i, row) for i, track in enumerate(self.tracks) for row in self.rows[track]]
        if not rows:
            raise RuntimeError('collector produced no replay rows')
        fields = tuple(rows[0][1])
        for name in fields:
            np.save(directory / f'{name}.npy', np.stack([r[name] for _, r in rows]), allow_pickle=False)
        np.save(directory / 'track_id.npy', np.asarray([i for i, _ in rows], np.int16), allow_pickle=False)
        (directory / 'metadata.json').write_text(json.dumps(dict(
            tracks=self.tracks, rows=len(rows), seen=dict(self.seen), fields=fields)) + '\n')
        return len(rows)

    def finish_episode(self,track,seed,final_passed):
        for row in self.rows[track]:
            if row.get('episode_seed')==seed:
                row['gate_recovered']=int(row['gate_ordinal']<final_passed)


def collector_main(connection, config_path, device):
    try:
        torch.set_num_threads(1)
        config = json.loads(Path(config_path).read_text())
        _, settings, source, _, teacher, normalizer, student = load_setup(config)
        teacher.to(device).eval()
        student.to(device).eval()
        stage = parse_stage(source['curriculum'])
        tracks = list(source['curriculum']['tracks'])
        connection.send(('ready',))
        while True:
            request = connection.recv()
            if request[0] == 'close':
                break
            _, round_index, beta, snapshot, destination = request
            student.load_state_dict(torch.load(snapshot, map_location=device, weights_only=True))
            student.eval()
            replay = Reservoir(tracks, config['rows_per_track_per_round'], config['seed']+round_index)
            rng = np.random.default_rng(config['seed'] + 71*round_index)
            jobs = [(track, config['seed'] + round_index*100000 + i)
                    for i, track in enumerate(rng.permutation(tracks))]
            started = time.perf_counter()
            for begin in range(0, len(jobs), config['rollout_envs']):
                collect_batch(jobs[begin:begin+config['rollout_envs']], config, settings,
                              stage, teacher, normalizer, student, device, beta,
                              replay, teacher_only=(round_index == 0 or
                                  config.get('permanent_teacher_only',False) and round_index<config['permanent_rounds']))
            collection_seconds = time.perf_counter()-started
            rows = replay.save(Path(destination))
            connection.send(('done', round_index, rows, collection_seconds,
                             time.perf_counter()-started, dict(replay.seen)))
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


class AsyncCollector:
    def __init__(self, config_path, device, timeout):
        parent, child = mp.get_context('spawn').Pipe()
        self.connection = parent
        self.process = mp.get_context('spawn').Process(
            target=collector_main, args=(child, str(config_path), device), daemon=True)
        self.process.start()
        child.close()
        self.timeout = timeout
        self.pending = None
        if self.receive()[0] != 'ready':
            raise RuntimeError('collector startup failed')

    def receive(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError('student collector exceeded timeout')
        result = self.connection.recv()
        if result[0] == 'error':
            raise RuntimeError(result[1])
        return result

    def begin(self, round_index, beta, student, output):
        if self.pending is not None:
            raise RuntimeError('collector request already pending')
        snapshot = output / 'collector-actor.pt'
        torch.save({k:v.detach().cpu() for k,v in student.state_dict().items()}, snapshot)
        destination = output / 'replay' / f'round-{round_index:04d}'
        self.pending = (round_index, destination)
        self.connection.send(('collect', round_index, beta, str(snapshot), str(destination)))

    def wait(self):
        result = self.receive()
        if result[0] != 'done' or self.pending is None or result[1] != self.pending[0]:
            raise RuntimeError('student collector returned an unexpected window')
        destination = self.pending[1]
        self.pending = None
        return destination, result

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(('close',))
            except (BrokenPipeError, EOFError):
                pass
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=10)
        self.connection.close()


class ShardReplay:
    """File-backed, course-first, three-stratum sampling with bounded history."""
    def __init__(self, tracks, recent_rounds, fractions, seed, recovery_success_fraction=0.):
        self.tracks = list(tracks)
        self.recent_rounds = recent_rounds
        self.fractions = np.asarray(fractions, np.float64)
        if not np.isclose(self.fractions.sum(), 1):
            raise ValueError('replay strata must sum to one')
        self.rng = np.random.default_rng(seed)
        self.shards = []
        self.pools = {}
        self.recovered_pools={}
        self.recovery_success_fraction=float(recovery_success_fraction)
        if not 0<=self.recovery_success_fraction<=1:raise ValueError('invalid recovery fraction')

    def add(self, path, permanent):
        meta = json.loads((path/'metadata.json').read_text())
        if meta['tracks'] != self.tracks:
            raise ValueError('replay course order changed')
        arrays = {name: np.load(path/f'{name}.npy', mmap_mode='r', allow_pickle=False)
                  for name in (*meta['fields'], 'track_id')}
        self.shards.append(dict(path=path, permanent=permanent, round=int(path.name.split('-')[-1]),
                                arrays=arrays, rows=meta['rows']))
        recent = [s for s in self.shards if not s['permanent']]
        keep = {id(s) for s in recent[-self.recent_rounds:]}
        self.shards = [s for s in self.shards if s['permanent'] or id(s) in keep]
        self.pools = {}
        self.recovered_pools={}
        for shard in self.shards:
            bank = 'permanent' if shard['permanent'] else 'online'
            ids = shard['arrays']['track_id']
            strata = shard['arrays']['stratum']
            for course in np.unique(ids):
                for stratum in range(3):
                    indices = np.flatnonzero((ids == course) & (strata == stratum))
                    if len(indices):
                        self.pools.setdefault((bank, int(course), stratum), []).append((shard, indices))
                        if stratum==2 and 'gate_recovered' in shard['arrays']:
                            recovered=indices[shard['arrays']['gate_recovered'][indices]>0]
                            if len(recovered):self.recovered_pools.setdefault((bank,int(course),stratum),[]).append((shard,recovered))

    def sample(self, count, permanent_fraction):
        keys = ('features','masks','teacher_features','state','action','executed',
                'dynamics','valid','speed','readout')
        if self.shards and 'memory' in self.shards[0]['arrays']:keys+=('memory',)
        chosen = []
        available_courses = [i for i in range(len(self.tracks)) if any(
            (bank,i,s) in self.pools for bank in ('permanent','online') for s in range(3))]
        if not available_courses:
            raise RuntimeError('no student replay available')
        requested = np.zeros(3, np.int64)
        actual = np.zeros(3, np.int64)
        for _ in range(count):
            course = int(self.rng.choice(available_courses))
            stratum = int(self.rng.choice(3, p=self.fractions)); requested[stratum] += 1
            bank = 'permanent' if self.rng.random() < permanent_fraction else 'online'
            alternatives = [(bank, course, stratum), ('online' if bank=='permanent' else 'permanent',course,stratum)]
            alternatives += [(b,course,s) for s in (0,1,2) for b in ('online','permanent')]
            key = next((key for key in alternatives if key in self.pools), None)
            if key is None:
                continue
            actual[key[2]] += 1
            pool=self.pools[key]
            if stratum==2 and self.recovery_success_fraction>0 and self.rng.random()<self.recovery_success_fraction:
                recovery_key=next((k for k in alternatives[:2] if k in self.recovered_pools),None)
                if recovery_key is not None:pool=self.recovered_pools[recovery_key]
            shard, indices = pool[int(self.rng.integers(len(pool)))]
            chosen.append((shard, int(indices[self.rng.integers(len(indices))])))
        batch = {key: np.stack([shard['arrays'][key][index] for shard,index in chosen]) for key in keys}
        return batch, requested, actual

    @property
    def rows(self):
        return sum(s['rows'] for s in self.shards)


def update(student, predictor, optimizer, replay, config, device, teacher, normalizer, settings=None, readout_normalization=None):
    if config.get('readout_regression_weight',0) and readout_normalization is None:
        raise ValueError('readout regression requires frozen training-bank normalization')
    raw, requested, actual = replay.sample(config['batch_size'], config['permanent_fraction'])
    batch = {k: torch.as_tensor(v, device=device).float() for k,v in raw.items() if k!='masks'}
    masks = torch.as_tensor(np.unpackbits(raw['masks'],axis=-1,count=160),device=device).float()
    from starscream.vision_distillation import augment_masks
    if config['vision_augmentation']:
        masks = augment_masks(masks)
    student.train(); predictor.train()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
        output = student(batch['features'], masks, batch['speed'], batch['executed'], return_hidden=True,memory=batch.get('memory'))
        predicted_readout = predictor(output['action_readout'])
    action = F.smooth_l1_loss(output['action'].float(),batch['action'],beta=.05)
    delta = F.smooth_l1_loss(output['dynamics'].float(),batch['dynamics'],reduction='none',beta=.02).mean(-1)
    dynamics = (delta*batch['valid']).sum()/batch['valid'].sum().clamp_min(1)
    state = F.smooth_l1_loss(output['estimate'].float(),batch['state'],beta=.02)
    estimated = batch['teacher_features'].clone()
    estimated[:,-1,:19] = output['estimate'].float()*torch.as_tensor(SCALE,device=device)
    normalized = (estimated-torch.as_tensor(normalizer.mean,device=device))/torch.as_tensor(normalizer.std,device=device)
    reconstructed = teacher(normalized,batch['speed'])
    interface = F.smooth_l1_loss(reconstructed.float(),batch['action'],beta=.05)
    readout,readout_regression=readout_losses(predicted_readout,batch['readout'],readout_normalization)
    physical=action.new_zeros(())
    if config.get('physical_action_weight',0):
        if settings is None:raise ValueError('physical loss requires pinned teacher action settings')
        from scripts.train_privileged_racing import normalized_ctbr_tensor
        scales=torch.as_tensor(config.get('physical_action_scales',[10.,3.,3.,3.]),device=device)
        if scales.shape!=(4,) or not torch.isfinite(scales).all() or (scales<=0).any():
            raise ValueError('physical action scales must be four finite positive values')
        error=(normalized_ctbr_tensor(output['action'],settings)-normalized_ctbr_tensor(batch['action'],settings))/scales
        physical=F.smooth_l1_loss(error,torch.zeros_like(error),beta=.05)
    loss = (action+config['dynamics_weight']*dynamics+config['state_weight']*state+
            config['interface_weight']*interface+config['readout_weight']*readout+
            config.get('readout_regression_weight',0)*readout_regression+config.get('physical_action_weight',0)*physical)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(student.parameters())+list(predictor.parameters()),2)
    optimizer.step()
    return {k:float(v.detach()) for k,v in dict(loss=loss,action=action,dynamics=dynamics,
             state=state,interface=interface,readout=readout,readout_regression=readout_regression,physical_action=physical).items()},requested,actual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--launch',action='store_true',help='required to train prepared v2 experiments')
    parser.add_argument('--resume-bootstrap',action='store_true',
                        help='reuse completed teacher baseline and round-zero shard after startup failure')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config.get('schema')=='starscream-vision-distillation-v2' and not args.launch:
        args.preflight=True
    if config.get('evaluation_protocol'):
        from starscream.racing_evaluation import load_protocol
        load_protocol(config['evaluation_protocol'],config.get('evaluation_protocol_sha256'))
    if 'replay_capacity' in config and config.get('schema')=='starscream-vision-distillation-v2':
        raise ValueError('async v2 replay is bounded by shard windows, not replay_capacity')
    torch.set_num_threads(1)
    torch.manual_seed(config['seed']); np.random.seed(config['seed']);random.seed(config['seed'])
    checkpoint,settings,source,suite,teacher,normalizer,student=load_setup(config)
    if checkpoint['feature_dim']!=167 or config['observation']!='ekf' or not student.explicit_estimation:
        raise ValueError('async student requires plant teacher, EKF and explicit estimation')
    if config['model']['conditioning'] is not True or not 0 <= config['readout_weight'] <= .2:
        raise ValueError('invalid conditioned action-readout recipe')
    if args.preflight:
        from scripts.train_vision_student import environment,SensorHistory,sample_curriculum_spawn
        stage=parse_stage(source['curriculum']); env=environment(stage.tracks[0],config,settings,stage)
        try:
            spawn=sample_curriculum_spawn(env.track,stage,seed=17,episode_index=0)
            obs,_=env.reset(seed=17,options=dict(state=spawn.state,gate_index=spawn.gate_index,
                route_plan_total_gates=len(env.track.gates)))
            sensor=SensorHistory(17,'ekf',settings,memory_steps=config['model'].get('memory_steps',0),
                memory_stride=config['model'].get('memory_stride',2));sensor.append(obs)
            f,m,t=sensor.arrays()
            assert f.shape==(3,125) and t.shape==(3,167) and m.shape==(3,1,128,160)
            assert 'plant_settings' in obs['privileged']
            if student.memory_steps:assert sensor.memory_array().shape==(student.memory_steps,21)
            predictor=make_readout_predictor(student.width,teacher.hidden_dim,config)
            print(json.dumps(dict(status='preflight_pass',student=student.parameter_counts(),
                train_only_predictor_parameters=sum(p.numel() for p in predictor.parameters()),
                train_courses=len(stage.tracks),val_courses=len(suite['records']),
                teacher_sha256=hashlib.sha256(local_path(config['teacher_checkpoint']).read_bytes()).hexdigest())))
        finally:env.close()
        return
    output=local_path(config['output'])
    if args.resume_bootstrap:
        if ((output/'metrics.jsonl').exists() or not (output/'teacher_baseline.json').exists()
                or not (output/'replay/round-0000/metadata.json').exists()):
            raise ValueError('bootstrap resume requires baseline and round-zero shard, with no updates')
        if json.loads((output/'config.json').read_text()) != config:
            raise ValueError('bootstrap config changed')
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'replay').mkdir()
        (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    teacher.to(args.device).eval()
    student.to(args.device)
    predictor=make_readout_predictor(student.width,teacher.hidden_dim,config).to(args.device)
    optimizer=torch.optim.AdamW(list(student.parameters())+list(predictor.parameters()),
                                lr=config['learning_rate'],weight_decay=1e-5)
    logger=init_wandb(config)
    if args.resume_bootstrap:
        reference=json.loads((output/'teacher_baseline.json').read_text())
    else:
        print(json.dumps(dict(event='teacher_baseline_started')),flush=True)
        reference=evaluate(config,settings,parse_stage(source['evaluation_curriculum']),suite,
                           teacher,normalizer,student,args.device,True)
        (output/'teacher_baseline.json').write_text(json.dumps(reference,indent=2)+'\n')
    logger.log_eval(wandb_evaluation_metrics(reference,'teacher/'),0)
    print(json.dumps(dict(event='teacher_baseline_complete',score=reference['selection_score'])),flush=True)
    replay=ShardReplay(source['curriculum']['tracks'],config['recent_replay_rounds'],
                       config['replay_strata'],config['seed'],config.get('recovery_success_fraction',0))
    collector=AsyncCollector(args.config.resolve(),args.device,config['collector_timeout_seconds'])
    best=-1.;best_key=(-float('inf'),); stale=0; cumulative_collection_steps=0;readout_normalization=None
    try:
        if not args.resume_bootstrap:
            collector.begin(0,1.,student,output)
        for round_index in range(config['rounds']):
            started=time.perf_counter()
            if args.resume_bootstrap and round_index==0:
                path=output/'replay/round-0000'
                metadata=json.loads((path/'metadata.json').read_text())
                result=('done',0,metadata['rows'],0.,0.,metadata['seen'])
            else:
                path,result=collector.wait()
            cumulative_collection_steps += sum(result[5].values())
            replay.add(path,permanent=round_index<config['permanent_rounds'])
            if round_index==0 and config.get('readout_regression_weight',0):
                shard=replay.shards[0]['arrays']
                readout_normalization=calibrate_readout(shard['readout'],shard['track_id'])
                (output/'readout-normalization.json').write_text(json.dumps(readout_normalization,indent=2)+'\n')
            next_round=round_index+1
            if next_round<config['rounds']:
                beta=collection_beta(config,next_round)
                collector.begin(next_round,beta,student,output)
            updated=time.perf_counter();metrics=[];requested=np.zeros(3,np.int64);actual=np.zeros(3,np.int64)
            lr=learning_rate_for_round(config,round_index)
            for group in optimizer.param_groups:group['lr']=lr
            for _ in range(config['updates_per_round']):
                row,req,act=update(student,predictor,optimizer,replay,config,args.device,teacher,normalizer,settings,readout_normalization)
                metrics.append(row);requested+=req;actual+=act
            update_seconds=time.perf_counter()-updated
            evaluation={};eval_seconds=0.;improved=False
            if next_round%config['evaluation_interval']==0 or next_round==config['rounds']:
                evaluated=time.perf_counter()
                student.eval()
                evaluation=evaluate(config,settings,parse_stage(source['evaluation_curriculum']),suite,
                                    teacher,normalizer,student,args.device)
                eval_seconds=time.perf_counter()-evaluated
                score=evaluation['selection_score']
                wandb_eval=wandb_evaluation_metrics(evaluation)
                wandb_eval['teacher_success_rate']=reference['selection_score']
                if reference['selection_score']>0:wandb_eval['success_retention']=score/reference['selection_score']
                logger.log_eval(wandb_eval,next_round*config['updates_per_round'])
                key=selection_key(evaluation)
                if key>best_key:
                    best=score;best_key=key;stale=0;improved=True
                else:stale+=1
            payload=dict(model=student.state_dict(),predictor=predictor.state_dict(),
                         optimizer=optimizer.state_dict(),config=config,round=round_index,
                         evaluation=evaluation,teacher_checkpoint=config['teacher_checkpoint'],
                         readout_normalization=readout_normalization,
                         action_contract=checkpoint['action_contract'])
            temporary=output/'checkpoint.tmp'
            torch.save(payload,temporary);os.replace(temporary,output/'latest.pt')
            if improved:
                torch.save(payload,temporary);os.replace(temporary,output/'best.pt')
            record=dict(round=round_index,environment_steps_seen=cumulative_collection_steps,
                beta=collection_beta(config,round_index),learning_rate=lr,
                replay_rows=replay.rows,collection_seconds=result[3],collection_handoff_seconds=result[4]-result[3],
                collection_rows=result[2],collection_seen=sum(result[5].values()),
                update_seconds=update_seconds,evaluation_seconds=eval_seconds,
                round_seconds=time.perf_counter()-started,actor_staleness_rounds=int(next_round<config['rounds']),
                replay_requested_fraction=(requested/requested.sum()).tolist(),
                replay_actual_fraction=(actual/max(actual.sum(),1)).tolist(),
                **{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},**evaluation)
            with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            compact={k:v for k,v in record.items() if k!='tracks'}
            print(json.dumps(compact),flush=True)
            logger.log_train({k:v for k,v in compact.items() if not isinstance(v,list)},
                             next_round*config['updates_per_round'])
            if next_round>=config['early_stop_min_round'] and stale>=config['early_stop_patience_evals']:
                print(json.dumps(dict(event='early_stop',round=round_index,best=best)),flush=True)
                break
    finally:
        collector.close()
        logger.finish()


if __name__=='__main__':
    main()
