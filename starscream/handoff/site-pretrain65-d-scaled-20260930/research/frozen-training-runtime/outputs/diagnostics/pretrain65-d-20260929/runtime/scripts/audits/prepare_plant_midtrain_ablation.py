"""Freeze three short, matched DAgger continuations. Never launches training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from scripts.train_privileged_racing import dagger_teacher_beta,load_policy_checkpoint,load_track
from starscream.dagger_schedule import round_learning_rate_factor
from starscream.midtrain_quality import admission_probability,reference_steps
from starscream.racing_evaluation import load_protocol

ROOT=Path('outputs/diagnostics/plant-midtrain-v1')
SOURCE=Path('outputs/checkpoints/starscream-v6.21.1.1-plant-resume100-r318/latest.pt')
PINNED=Path('outputs/resume-inputs/plant-midtrain-v1/round-274-neutral.pt')
ARMS=('control','hard_timed','soft_time')
SOURCES=('scripts/train_privileged_racing.py','starscream/midtrain_quality.py',
         'starscream/dagger_schedule.py','starscream/dagger_async.py',
         'starscream/dagger_replay.py','starscream/timed_teacher_reporting.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    certificate=ROOT/'preflight.json'
    if args.verify:
        saved=json.loads(certificate.read_text())
        for path,fingerprint in saved['fingerprints'].items():
            if sha(path)!=fingerprint:raise ValueError(f'midtrain launch input changed: {path}')
        print('Midtrain ablation certificate verified.');return
    if certificate.exists():raise FileExistsError('Midtrain ablation already prepared')
    torch.set_num_threads(1)
    policy,normalizer,payload,_=load_policy_checkpoint(str(SOURCE),'cpu')
    assert payload['stage']=='dagger' and payload['round']==274
    assert payload['dagger_replay']['committed_round']==274
    assert payload['rng_state'] and payload['numpy_rng_state']
    assert payload['optimizer'] and payload['dagger_dynamic_sampler_state']
    pending=payload.get('dagger_async_pending')
    assert pending is None or int(pending['window'])==275
    payload['dagger_async_pending']=None # pending collection is uncommitted; all arms start fresh
    if PINNED.exists():
        existing=torch.load(PINNED,map_location='cpu',weights_only=False)
        assert existing['round']==274 and existing['dagger_async_pending'] is None
    else:
        PINNED.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,PINNED)
    base=copy.deepcopy(payload['training_config'])
    settings=base['dagger'];assert settings['rounds']==318
    manifest=json.loads(Path(settings['track_manifest']).read_text())
    train=[r for r in manifest['records'] if r.get('split')=='train']
    assert len(train)==65
    for record in train:
        cohort=record['qualification']['pace_cohorts']['randomized']
        assert cohort['successes']>=1 and cohort['median']>0
        gates=int(record.get('gate_count') or len(load_track(record['path']).gates))
        assert reference_steps(cohort['median'],gates,0)>0
    protocol='configs/eval/real100_v2_timed_protocol_v1.json';load_protocol(protocol)
    shared=dict(rounds=280,resume_checkpoint=str(PINNED.resolve()),
        dagger_resume_optimizer_state=True,dagger_resume_require_replay_state=True,
        dagger_resume_expert_refresh_rounds=0,dagger_successful_coverage_rounds=280,
        dagger_minimum_successful_episodes_per_track=0,
        dagger_coverage_retry_episodes_per_track=0,
        dagger_abort_on_underfilled_tracks=False,
        online_replay_capacity=200000,dagger_online_replay_rows_per_round=60000,
        episodes_per_round=260,updates_per_round=1024)
    assert all(abs(dagger_teacher_beta(r,dict(settings,**shared))-.35)<1e-12 for r in range(275,281))
    assert all(abs(round_learning_rate_factor(r,280,settings['dagger_learning_rate_schedule'])-.25)<1e-12 for r in range(275,281))
    assert admission_probability('hard_timed',True,150,100)==1
    assert admission_probability('hard_timed',True,491,100)==0
    plans={}
    for arm in ARMS:
        config=copy.deepcopy(base);s=config['dagger'];s.update(shared)
        run=f'starscream-plant-midtrain-v1-{arm}-r280'
        s['run_name']=run
        if arm=='control':
            s['dagger_require_successful_episodes']=False
            s.pop('midtrain_quality',None)
        else:
            s['dagger_require_successful_episodes']=True
            s['midtrain_quality']={'method':arm}
        s['timed_reporting']=dict(protocol=protocol,protocol_sha256=sha(protocol),
            suite='configs/eval/v6211_vision_selection25.json',episodes_per_course=4,
            interval=3,start_round=274,end_round=280,parent_sha256=sha(PINNED))
        s['tags']=['plant-privileged','midtrain-ablation-v1',arm,'r274-common-start']
        config['checkpoint']['run_name']=run
        config['wandb'].update(run_name=run,group='plant-midtrain-v1',tags=s['tags'],
            local_event_path=f'/workspace/outputs/logs/{run}.events.jsonl',
            local_full_event_path=f'/workspace/outputs/logs/{run}.full.events.jsonl')
        for key in ('id','run_id','resume'):config['wandb'].pop(key,None)
        config['experiment_notes']=dict(parent_checkpoint=str(PINNED),
            parent_sha256=sha(PINNED),purpose='short matched midtraining signal',
            arm=arm,common_rounds='275-280',baseline_timed_round=274)
        path=Path(f'configs/exp/v6.21.1.1/plant_midtrain_v1_{arm}.json')
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(config,indent=2)+'\n')
        plans[arm]=dict(run=run,config=str(path),method=s.get('midtrain_quality',{'method':'ordinary_dagger'}),
                        starting_round=274,ending_round=280)
    fingerprints={str(path):sha(path) for path in [PINNED,Path(protocol),
        Path('configs/eval/v6211_vision_selection25.json'),*map(Path,SOURCES),
        *(Path(v['config']) for v in plans.values())]}
    ROOT.mkdir(parents=True,exist_ok=True)
    certificate.write_text(json.dumps(dict(status='validated-not-launched',parent_round=274,
        parent_environment_steps=payload['environment_steps'],pending_discarded=pending is not None,
        parent_sha256=sha(SOURCE),pinned_sha256=sha(PINNED),arms=plans,
        common=dict(shared,learning_rate=3e-5,teacher_beta=.35),
        fingerprints=fingerprints),indent=2)+'\n')
    print(json.dumps(dict(parent_round=274,arms=plans,common_budget=dict(episodes_per_round=260,
        updates_per_round=1024,online_replay_capacity=200000),pending_discarded=pending is not None),indent=2))


if __name__=='__main__':main()
