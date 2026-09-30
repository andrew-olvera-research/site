#!/usr/bin/env python3
"""Route6 privileged-to-vision DAgger. Defaults to read-only preflight.

Use --launch explicitly to train. --verify-env only checks one reset/step and
mask agreement; it does not optimize or create experiment checkpoints.
"""
from __future__ import annotations
import argparse
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F

from scripts.eval_flight import ObservationHistory
from scripts.train_privileged_racing import (parse_stage, ppo_observation_features,
    ppo_normalized_to_ctbr, action_contract_metadata)
from starscream.env import FlightmareEnv
from starscream.privileged_racing import PrivilegedMLPPolicy, FeatureNormalizer
from starscream.racing_curriculum import sample_curriculum_spawn
from starscream.vision_distillation import VisionStudent, KinematicEKF, augment_masks
from starscream.wandb import init_wandb

ROOT = Path(__file__).resolve().parents[1]
SCALE = np.array([20]*3 + [30]*3 + [1]*6 + [6]*3 + [4000]*4, np.float32)


def local_path(value):
    path = Path(value)
    if str(path).startswith('/workspace/'):
        path = ROOT / path.relative_to('/workspace')
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_setup(config):
    path = local_path(config['teacher_checkpoint'])
    if sha256(path) != config['teacher_sha256']:
        raise ValueError('teacher fingerprint changed')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    settings = checkpoint['training_config'][checkpoint['stage']]
    if checkpoint['feature_dim'] not in (103, 167) or checkpoint['model_config']['context_steps'] != 3:
        raise ValueError('this experiment requires route6/h3 teacher')
    if checkpoint['model_config']['observation_contract'] not in ('starscream_route_v1', 'starscream_route_plant_v1'):
        raise ValueError('unsupported teacher contract')
    action_keys = ('collective_action_mapping','collective_action_maximum',
                   'collective_action_reference_thrust','collective_action_logit_scale')
    if not all(k in settings for k in action_keys):
        raise ValueError('teacher is missing explicit action-contract metadata')
    contract = action_contract_metadata(settings)
    if contract != checkpoint.get('action_contract', contract):
        raise ValueError('checkpoint action metadata mismatch')
    checkpoint['action_contract'] = contract
    if checkpoint['control_hz'] != 130:
        raise ValueError('expected 130Hz teacher')
    teacher = PrivilegedMLPPolicy(**checkpoint['model_config'])
    teacher.load_state_dict(checkpoint['model'])
    teacher.eval().requires_grad_(False)
    normalizer = FeatureNormalizer.from_state_dict(checkpoint['normalizer'])
    student = VisionStudent(**config['model'])
    if student.parameter_counts()['core'] > 3_000_000:
        raise ValueError('student core exceeds 3M INCLUDING auxiliary heads')
    if config['observation'] not in {'raw', 'ekf'}:
        raise ValueError('unknown observation arm')
    source_path = local_path(config['training_source'])
    if sha256(source_path) != config['training_source_sha256']:
        raise ValueError('pretraining config changed')
    source = json.loads(source_path.read_text())['dagger']
    suite_path = local_path(config['selection_manifest'])
    if sha256(suite_path) != config['selection_sha256']:
        raise ValueError('selection subset changed')
    suite = json.loads(suite_path.read_text())
    train = source['curriculum']['tracks']
    val = [record['path'] for record in suite['records']]
    if set(map(lambda p: local_path(p).resolve(), train)) & set(map(lambda p: local_path(p).resolve(), val)):
        raise ValueError('train/selection track overlap')
    missing = [str(local_path(p)) for p in train + val if not local_path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'missing tracks: {missing[:5]}')
    return checkpoint, settings, source, suite, teacher, normalizer, student


class SensorHistory:
    def __init__(self, seed, arm, settings, *, memory_steps=0, memory_stride=2):
        self.history = ObservationHistory(3, (128,160), estimate_dim=32,
                                          estimate_seed=seed, control_hz=130)
        self.ekf = KinematicEKF(1/130)
        self.arm, self.settings = arm, settings
        self.features, self.masks, self.truth = deque(maxlen=3), deque(maxlen=3), deque(maxlen=3)
        self.previous_vio = None
        self.previous_gate = None
        self.memory_steps,self.memory_stride=int(memory_steps),int(memory_stride)
        self.memory=deque(maxlen=max(1,1+(self.memory_steps-1)*self.memory_stride))

    def append(self, observation):
        self.history.append(observation)
        record = self.history.records[-1]
        raw = record['estimate'].copy()
        gate = int(record['gate_index'][0])
        # Historical proxy acceleration used truth velocity. Use only noisy
        # observations here, identically for BOTH arms, without extra truth.
        raw[21:24] = 0 if self.previous_vio is None or gate != self.previous_gate else np.clip(
            (raw[3:6] - self.previous_vio) * 130, -2, 2)
        self.previous_vio, self.previous_gate = raw[3:6].copy(), gate
        if self.arm == 'ekf':
            raw = self.ekf.update(raw, observation['measured']['body_rates'], gate)
        teacher_features = ppo_observation_features(observation, self.settings)
        # The plant teacher appends its 64 privileged settings after the
        # six control/timing features. The deployable student never sees them.
        trailer = teacher_features[97:103]
        feature = np.concatenate([raw, record['proprio'][:7], record['route'].ravel(),
                                  trailer,
                                  [float(observation['age']['camera'])*130/36,
                                   float(observation['valid']['camera'])]])
        self.features.append(feature.astype(np.float32))
        self.masks.append(record['mask'].astype(np.uint8))
        self.truth.append(teacher_features)
        if self.memory_steps:
            self.memory.append(self.memory_features(feature))

    @staticmethod
    def memory_features(feature):
        # Gate frame -> body velocity: invariant to a change of target gate.
        rotation=KinematicEKF.rotation(feature[6:12])
        body_velocity=rotation.T@feature[3:6]
        return np.concatenate([body_velocity,feature[32:39],feature[117:123],
                               feature[24:27],feature[[28,30]]]).astype(np.float32)

    def memory_array(self):
        if not self.memory_steps or not self.memory:
            raise ValueError('causal memory is not configured or initialized')
        values=list(self.memory)
        return np.stack([values[max(0,len(values)-1-age*self.memory_stride)]
                         for age in reversed(range(self.memory_steps))])

    def arrays(self):
        def padded(values):
            return np.stack([values[0]] * (3-len(values)) + list(values))
        return padded(self.features), padded(self.masks), padded(self.truth)


def environment(track, config, settings, stage):
    return FlightmareEnv(track=str(local_path(track)), next_gates=6,
        control_dt=1/130, image_size=(160,128), mask_size=(160,128),
        render_observations=False, mask_source='geometry', geometry_renderer='native_exact',
        image_delay=config['image_delay'], action_delay=stage.action_delay,
        dynamics_randomization=settings.get('dynamics_randomization', {'enabled': True}),
        plant_settings_observation=(settings.get('model', {}).get('observation_contract') == 'starscream_route_plant_v1'
                                    or settings.get('plant_settings_observation', False)),
        maximum_collective_thrust=40.0, policy_state_source='truth')


class BalancedReplay:
    """Per-course FIFO plus a retained teacher-only seed bank.

    Sampling first chooses a course uniformly, preventing long courses and the
    final collection batch from replacing all other behaviors in bounded replay.
    """
    def __init__(self, capacity, tracks):
        self.per_track = max(1, capacity // (2*len(tracks)))
        self.online = {track: deque(maxlen=self.per_track) for track in tracks}
        self.permanent = {track: deque(maxlen=self.per_track) for track in tracks}
        self.seed_collection = True
        self.seed_seen = {track:0 for track in tracks}
        self.rng = np.random.default_rng(48371)

    def append(self, track, row):
        bank = self.permanent if self.seed_collection else self.online
        if self.seed_collection:
            self.seed_seen[track] += 1
            if len(bank[track]) == self.per_track:
                index = int(self.rng.integers(self.seed_seen[track]))
                if index < self.per_track:
                    bank[track][index] = row
                return
        bank[track].append(row)

    def __len__(self):
        return sum(len(v) for bank in (self.permanent, self.online) for v in bank.values())

    def sample(self, count, rng):
        active = {name: [v for v in bank.values() if v]
                  for name,bank in [('permanent',self.permanent),('online',self.online)]}
        rows = []
        for _ in range(count):
            name = 'permanent' if rng.random()<.25 or not active['online'] else 'online'
            if not active[name]:
                name = 'online'
            course = active[name][int(rng.integers(len(active[name])))]
            rows.append(course[int(rng.integers(len(course)))])
        return rows


def update(student, optimizer, replay, config, rng, device, teacher, normalizer):
    rows = replay.sample(config['batch_size'], rng)
    batch = {key: torch.as_tensor(np.stack([r[key] for r in rows]), device=device).float()
             for key in rows[0] if key != 'masks'}
    masks = torch.as_tensor(np.stack([np.unpackbits(r['masks'], axis=-1, count=160) for r in rows]), device=device).float()
    student.train()
    if config['vision_augmentation']:
        masks = augment_masks(masks)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                        enabled=str(device).startswith('cuda')):
        result = student(batch['features'], masks, batch['speed'], batch['executed'],memory=batch.get('memory'))
    action = F.smooth_l1_loss(result['action'], batch['action'], beta=.05)
    delta = F.smooth_l1_loss(result['dynamics'], batch['dynamics'], reduction='none', beta=.02).mean(-1)
    dynamics = (delta * batch['valid']).sum() / batch['valid'].sum().clamp_min(1)
    loss = action + config['dynamics_weight']*dynamics
    state = interface = loss.new_zeros(())
    if student.explicit_estimation:
        state = F.smooth_l1_loss(result['estimate'], batch['state'], beta=.02)
        estimated = batch['teacher_features'].clone()
        estimated[:, -1, :19] = result['estimate'].float() * torch.as_tensor(SCALE, device=device)
        normalized = (estimated-torch.as_tensor(normalizer.mean, device=device))/torch.as_tensor(normalizer.std, device=device)
        # Frozen weights, differentiable inputs: do NOT put this under no_grad.
        reconstructed = teacher(normalized, batch['speed'])
        interface = F.smooth_l1_loss(reconstructed, batch['action'], beta=.05)
        loss = loss + config['state_weight']*state + config['interface_weight']*interface
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 2)
    optimizer.step()
    return dict(loss=float(loss.detach()), action=float(action.detach()),
                dynamics=float(dynamics.detach()), state=float(state.detach()), interface=float(interface.detach()))


def collect_batch(jobs, config, settings, stage, teacher, normalizer, student,
                  device, beta=0., replay=None, teacher_only=False):
    """One batched inference per tick; native geometry/plant steps on CPU workers.

    At most rollout_envs masks/states are live. Replay stores bit-packed masks;
    gradients never pass through the simulator. No optimizer work in evaluation.
    """
    slots = []
    results = [None] * len(jobs)
    mean = torch.as_tensor(normalizer.mean, device=device)
    std = torch.as_tensor(normalizer.std, device=device)
    student.eval()
    try:
        for index, (track, seed) in enumerate(jobs):
            env = environment(track, config, settings, stage)
            slots.append(dict(env=env, index=index))
            spawn = sample_curriculum_spawn(env.track, stage, seed=seed, episode_index=0)
            obs, _ = env.reset(seed=seed, options=dict(state=spawn.state, gate_index=spawn.gate_index,
                route_plan_total_gates=len(env.track.gates)))
            sensors = SensorHistory(seed, config['observation'], settings,
                memory_steps=config['model'].get('memory_steps',0),
                memory_stride=config['model'].get('memory_stride',2)); sensors.append(obs)
            slots[-1].update(obs=obs, sensor=sensors, track=track, start=env.tracker.passed_count,
                             rng=np.random.default_rng(seed ^ 73421), steps=0,seed=seed)
        with ThreadPoolExecutor(max_workers=min(config['rollout_envs'], len(jobs))) as pool:
            while slots:
                arrays = [s['sensor'].arrays() for s in slots]
                features, masks, truth = [np.stack(values) for values in zip(*arrays)]
                memory=(np.stack([s['sensor'].memory_array() for s in slots])
                        if config['model'].get('memory_steps',0) else None)
                speed = torch.full((len(slots),), float(config['speed_command']), device=device)
                with torch.inference_mode():
                    needs_teacher = teacher_only or replay is not None or beta>0
                    teacher_input = (torch.as_tensor(truth, device=device)-mean)/std if needs_teacher else None
                    expert = (teacher(teacher_input, speed).cpu().numpy()
                              if needs_teacher else None)
                    readout = None
                    if replay is not None and config.get('readout_weight', 0) > 0:
                        base = teacher.encode(teacher_input)
                        decision = teacher._conditioned_encoding(base, speed)
                        if teacher.topology_head is not None:
                            decision = decision + teacher.topology_adapter(teacher.topology_head(decision))
                        readout = decision.float().cpu().numpy()
                    if teacher_only or beta >= 1:
                        prediction = expert
                    else:
                        prediction = student(torch.as_tensor(features, device=device),
                            torch.as_tensor(masks, device=device).float(), speed,
                            memory=torch.as_tensor(memory,device=device) if memory is not None else None)['action'].cpu().numpy()
                executed = [expert[i] if teacher_only or s['rng'].random()<beta else prediction[i]
                            for i,s in enumerate(slots)]
                commands = [ppo_normalized_to_ctbr(a, settings) for a in executed]
                futures = [pool.submit(s['env'].step, command) for s,command in zip(slots, commands)]
                alive = []
                for i, (slot, future) in enumerate(zip(slots, futures)):
                    obs, _, terminated, truncated, info = future.result()
                    slot['steps'] += 1
                    previous = slot['obs']
                    if replay is not None:
                        gate_same = int(previous['flight_plan']['index'][0]) == int(obs['flight_plan']['index'][0])
                        current = features[i, -1]
                        offset = current[:3]*20 - current[39:42]*20
                        normal = current[42:45]
                        distance = float(np.linalg.norm(offset))
                        # Passing the active gate plane without a registered
                        # crossing is a recovery state; nearby approaches and
                        # crossings form the critical stratum.
                        stratum = (2 if gate_same and float(offset @ normal) > .5
                                   else 1 if not gate_same or distance < 5.0 else 0)
                        row=dict(features=features[i].copy(), masks=np.packbits(masks[i], axis=-1),
                            teacher_features=truth[i].copy(), state=np.asarray(previous['task_state'])/SCALE,
                            action=expert[i].copy(), executed=executed[i].copy(),
                            dynamics=(np.asarray(obs['task_state'])-np.asarray(previous['task_state']))/SCALE,
                            valid=float(gate_same and not terminated and not truncated), speed=config['speed_command'],
                            stratum=stratum, readout=(readout[i].copy() if readout is not None else np.zeros(teacher.hidden_dim, np.float32)))
                        if memory is not None:row['memory']=memory[i].copy()
                        if config.get('recovery_success_fraction',0):
                            row.update(episode_seed=slot['seed'],gate_ordinal=slot['env'].tracker.passed_count-int(info.get('gate_passed',False)),gate_recovered=0)
                        replay.append(slot['track'],row)
                    gates = slot['env'].tracker.passed_count-slot['start']
                    success = gates >= len(slot['env'].track.gates)
                    done = success or terminated or truncated or slot['steps']>=stage.max_steps
                    if done:
                        if config.get('recovery_success_fraction',0):
                            replay.finish_episode(slot['track'],slot['seed'],slot['env'].tracker.passed_count)
                        results[slot['index']] = dict(success=float(success), gates=gates, steps=slot['steps'],
                            lap_seconds=slot['steps']/130 if success else None)
                        slot['env'].close()
                    else:
                        slot['obs'] = obs; slot['sensor'].append(obs); alive.append(slot)
                slots = alive
        return results
    finally:
        for slot in slots:
            slot['env'].close()


def evaluate(config, settings, stage, suite, teacher, normalizer, student, device, teacher_only=False):
    if config.get('evaluation_protocol'):
        from starscream.racing_evaluation import evaluate_racing
        return evaluate_racing(config,settings,stage,suite,teacher,normalizer,student,device,teacher_only)
    rows = []
    jobs = []
    for record in suite['records']:
        track_seed = int(hashlib.sha256(record['slot'].encode()).hexdigest()[:8], 16) % 1_000_000
        jobs.extend((record['path'], config['evaluation_seed']+track_seed+i)
                    for i in range(config['evaluation_episodes']))
    results = []
    for start in range(0, len(jobs), config['rollout_envs']):
        results.extend(collect_batch(jobs[start:start+config['rollout_envs']], config, settings,
            stage, teacher, normalizer, student, device, teacher_only=teacher_only))
    for index, record in enumerate(suite['records']):
        episodes = results[index*config['evaluation_episodes']:(index+1)*config['evaluation_episodes']]
        rows.append(dict(slot=record['slot'], family=record['family'], episodes=episodes,
                         success=float(np.mean([e['success'] for e in episodes]))))
    # Population family weights counter unequal family quota rounding. This is
    # geometry-stratified development selection, not an unbiased test estimate.
    families = defaultdict(list)
    for row in rows:
        families[row['family']].append(row['success'])
    score = sum(suite['family_weights'][f] * np.mean(values) for f, values in families.items())
    return dict(selection_score=float(score), tracks=rows)


def wandb_evaluation_metrics(report, prefix=''):
    # Detailed course telemetry remains in local JSON only. Keep the remote
    # chart schema constant as the number of courses grows.
    episodes = [e for row in report['tracks'] for e in row['episodes']]
    metrics = {prefix+'success_rate': report['selection_score']}
    if episodes:
        metrics[prefix+'mean_episode_steps'] = float(np.mean([e['steps'] for e in episodes]))
    times = [e['lap_seconds'] for e in episodes if e['lap_seconds'] is not None]
    if times:
        metrics[prefix+'successful_lap_seconds'] = float(np.mean(times))
    for source,target in [('time_weighted_success','time_weighted_success'),('crashed','crash_rate'),
                          ('timeout','timeout_rate'),('clean_success','clean_success_rate')]:
        if source in report:metrics[prefix+target]=report[source]
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--launch', action='store_true')
    group.add_argument('--verify-env', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
    checkpoint, settings, source, suite, teacher, normalizer, student = load_setup(config)
    print(json.dumps(dict(status='preflight_pass', parameters=student.parameter_counts(),
        training_tracks=len(source['curriculum']['tracks']), validation_tracks=len(suite['records']),
        teacher=str(config['teacher_checkpoint']), observation=config['observation']), indent=2), flush=True)
    train_stage = parse_stage(source['curriculum'])
    val_stage = parse_stage(source['evaluation_curriculum'])
    teacher.to(args.device); student.to(args.device)
    if args.verify_env:
        env = environment(source['curriculum']['tracks'][0], config, settings, train_stage)
        try:
            spawn = sample_curriculum_spawn(env.track, train_stage, seed=17, episode_index=0)
            obs, _ = env.reset(seed=17, options=dict(state=spawn.state, gate_index=spawn.gate_index,
                route_plan_total_gates=len(env.track.gates)))
            sensor = SensorHistory(17, config['observation'], settings); sensor.append(obs)
            f, m, _ = sensor.arrays()
            with torch.no_grad():
                action = student(torch.tensor(f, device=args.device)[None],
                    torch.tensor(m, device=args.device).float()[None], torch.tensor([16.5], device=args.device))['action'][0].cpu().numpy()
            obs, *_ = env.step(ppo_normalized_to_ctbr(action, settings)); sensor.append(obs)
            print(json.dumps(dict(status='environment_reset_step_pass', mask_pixels=int(m.sum()))))
        finally:
            env.close()
        return
    if not args.launch:
        return
    if not config.get('wandb', {}).get('enabled') or config['wandb'].get('mode') != 'online':
        raise ValueError('vision experiments require enabled online W&B logging')
    output = local_path(config['output'])
    output.mkdir(parents=True, exist_ok=False)  # never overwrite or silently resume
    (output/'config.json').write_text(json.dumps(config, indent=2))
    logger = init_wandb(config)
    if logger.run is None or logger.run.settings.mode != 'online':
        raise RuntimeError('W&B did not initialize an online run')
    run_metadata = dict(run_id=logger.run.id, url=logger.run.url, mode=logger.run.settings.mode)
    (output/'wandb-run.json').write_text(json.dumps(run_metadata, indent=2)+'\n')
    print(json.dumps(dict(event='wandb_online', **run_metadata)), flush=True)
    logger.run.summary.update({f'parameters/{k}':v for k,v in student.parameter_counts().items()})
    logger.log_train(dict(started=1), 0)
    replay = BalancedReplay(config['replay_capacity'], source['curriculum']['tracks'])
    rng = np.random.default_rng(config['seed'])
    optimizer = torch.optim.AdamW(student.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    print(json.dumps(dict(event='teacher_baseline_started', episodes=25*config['evaluation_episodes'])), flush=True)
    reference = evaluate(config, settings, val_stage, suite, teacher, normalizer, student, args.device, True)
    (output/'teacher_baseline.json').write_text(json.dumps(reference, indent=2))
    logger.log_eval(wandb_evaluation_metrics(reference, 'teacher/'), 0)
    print(json.dumps(dict(event='teacher_baseline_complete', selection_score=reference['selection_score'])), flush=True)
    best = -1.0
    for round_index in range(config['rounds']):
        round_started = time.perf_counter()
        replay.seed_collection = round_index == 0
        beta = 1.0 if round_index == 0 else max(.05, 1-round_index/config['beta_decay_rounds'])
        print(json.dumps(dict(event='collection_started', round=round_index, beta=beta)), flush=True)
        tracks = list(source['curriculum']['tracks']); rng.shuffle(tracks)
        # Round-robin full coverage avoids silently dropping short/failing tracks.
        jobs = [(track, config['seed']+round_index*100000+index) for index,track in enumerate(tracks)]
        for start in range(0, len(jobs), config['rollout_envs']):
            collect_batch(jobs[start:start+config['rollout_envs']], config, settings, train_stage,
                teacher, normalizer, student, args.device, beta, replay)
        collection_seconds = time.perf_counter()-round_started
        update_started = time.perf_counter()
        print(json.dumps(dict(event='updates_started', round=round_index, replay=len(replay))), flush=True)
        metrics = []
        for update_index in range(config['updates_per_round']):
            metrics.append(update(student, optimizer, replay, config, rng, args.device, teacher, normalizer))
            if (update_index+1)%32 == 0 or update_index+1 == config['updates_per_round']:
                recent = metrics[-32:]
                logger.log_train(dict(round=round_index, beta=beta, replay=len(replay),
                    **{k:float(np.mean([m[k] for m in recent])) for k in recent[0]}),
                    round_index*config['updates_per_round']+update_index+1)
        update_seconds = time.perf_counter()-update_started
        evaluation = {}
        if (round_index+1)%config['evaluation_interval']==0 or round_index+1==config['rounds']:
            print(json.dumps(dict(event='selection_started', round=round_index)), flush=True)
            evaluation = evaluate(config, settings, val_stage, suite, teacher, normalizer, student, args.device)
            eval_metrics = wandb_evaluation_metrics(evaluation)
            eval_metrics['teacher_success_rate'] = reference['selection_score']
            if reference['selection_score']>0:
                eval_metrics['success_retention'] = evaluation['selection_score']/reference['selection_score']
            logger.log_eval(eval_metrics, (round_index+1)*config['updates_per_round'])
        score = evaluation.get('selection_score', -1.)
        payload = dict(model=student.state_dict(), config=config, optimizer=optimizer.state_dict(),
            round=round_index, evaluation=evaluation, action_contract=checkpoint['action_contract'],
            parameter_counts=student.parameter_counts())
        torch.save(payload, output/'latest.pt')
        if score > best:
            best = score; torch.save(payload, output/'best.pt')
        record = dict(round=round_index, beta=beta, replay=len(replay),
            collection_seconds=collection_seconds, update_seconds=update_seconds,
            updates_per_second=config['updates_per_round']/max(update_seconds,1e-9),
            learning_rate=optimizer.param_groups[0]['lr'],
            round_seconds=time.perf_counter()-round_started,
            **{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}, **evaluation)
        with open(output/'metrics.jsonl', 'a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k != 'tracks'}), flush=True)
        logger.log_train({k:v for k,v in record.items() if k != 'tracks'},
                         (round_index+1)*config['updates_per_round'])
    logger.finish()


if __name__ == '__main__':
    main()
