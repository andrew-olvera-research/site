#!/usr/bin/env python3
"""Replay pinned laps, save actor states, and probe the finite-route terminal cue.

The terminal probe changes only the actor's observation at the same simulator
state. It is an action sensitivity measurement, not a closed-loop time trial.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from train_privileged_racing import (CausalHistory, completion_gate_count,
    episode_target_speed, load_config, make_env, make_reward, parse_stage,
    ppo_normalized_to_ctbr, ppo_observation_features, reset_env,
    _normalize_dynamics_delta_for_control_rate)
from starscream.privileged_racing import load_policy_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, default=ROOT/'outputs/site-trajectories-v6211-18-20260923/selection.json')
    p.add_argument('--course', action='append', required=True)
    p.add_argument('--out', type=Path, default=ROOT/'outputs/evals/recovery-mode-probe')
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--terminal-spacing', type=float, default=5.)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.stride < 1 or args.terminal_spacing <= 0:
        p.error('stride and terminal-spacing must be positive')
    selection = json.loads(args.selection.read_text())
    config_path = Path(selection['config'].replace('/workspace/', str(ROOT)+'/'))
    checkpoint = Path(selection['checkpoint'].replace('/workspace/', str(ROOT)+'/'))
    settings = dict(load_config(config_path)['dagger'])
    stage = parse_stage(settings['evaluation_curriculum'])
    policy, normalizer, payload, _ = load_policy_checkpoint(checkpoint, args.device)
    policy.eval()
    dynamics_mean = np.asarray(payload['dynamics_target_mean'], np.float32)
    dynamics_std = np.asarray(payload['dynamics_target_std'], np.float32)
    by_name = {x['name']: x for x in selection['selections']}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': str(checkpoint), 'config': str(config_path),
              'stride': args.stride, 'terminal_spacing_m': args.terminal_spacing,
              'courses': []}
    for name in args.course:
        item = by_name[name]
        track_path = item['track_path'].replace('/workspace/', str(ROOT)+'/')
        seed = int(item['capture_seed'])
        episode_index = int(item['capture_episode_index'])
        target_speed = episode_target_speed(stage, seed)
        env = make_env(settings, track=track_path, reward=make_reward(settings, target_speed))
        try:
            obs, start_passed = reset_env(env, stage, seed=seed, episode_index=episode_index)
            history = CausalHistory(policy.context_steps)
            history.reset_feature(ppo_observation_features(obs, settings))
            target = completion_gate_count(env.track, stage.target_gates,
                enabled=bool(settings.get('allow_curriculum_completion_override', False))) * stage.rollout_laps
            rows = []
            latents = []
            dynamics_predictions = []
            dynamics_targets = []
            dynamics_valid = []
            misses = []
            recrossings = []
            for step in range(stage.max_steps):
                passed = int(env.tracker.passed_count-start_passed)
                route_indices = np.asarray(obs['flight_plan']['index'], np.int64)
                expected_indices = env.track.gate_indices(
                    env.tracker.index, len(route_indices), remaining=max(target-passed, 1))
                if not np.array_equal(route_indices, expected_indices):
                    raise AssertionError(f'route mismatch at step {step}: {route_indices} != {expected_indices}')
                expected_records = env.track.flight_plan(
                    env.tracker.index, len(route_indices), remaining=max(target-passed, 1))['records']
                if not np.array_equal(obs['flight_plan']['records'], expected_records):
                    raise AssertionError(f'route records mismatch at step {step}')
                state = np.asarray(obs['state'], np.float32)
                prior_task = np.asarray(obs['task_state'], np.float32).copy()
                gate = env.track.gates[env.tracker.index]
                side = float((state[:3]-gate.position)@gate.normal)
                local = gate.directed_rotation.T@(state[:3]-gate.position)
                normalized = normalizer.numpy(history.array()[None])
                tensor = torch.from_numpy(normalized).to(args.device)
                speed = torch.tensor([target_speed], device=args.device, dtype=torch.float32)
                with torch.no_grad():
                    action_norm = policy(tensor, speed)[0].float().cpu().numpy()
                    dynamics_prediction = policy.predict_dynamics(tensor)[0].float().cpu().numpy()
                    if step % args.stride == 0:
                        base = policy.encode(tensor)
                        conditioned = policy._conditioned_encoding(base, speed)
                        latents.append((step, base[0].float().cpu().numpy(),
                                        conditioned[0].float().cpu().numpy()))
                    alternate = np.full(4, np.nan, np.float32)
                    if passed == target-1:
                        route = np.asarray(obs['flight_plan']['records'], np.float32).copy()
                        # Route records are in the active gate's directed frame.
                        normal = route[0, 3:6].copy()
                        for slot in range(1, len(route)):
                            route[slot] = route[0]
                            route[slot, :3] = normal * (args.terminal_spacing * slot)
                        edited = tensor.clone()
                        raw = history.array()[None].copy()
                        raw[0,-1,19:19+len(route)*13] = route.reshape(-1)
                        edited = torch.from_numpy(normalizer.numpy(raw)).to(args.device)
                        alternate = policy(edited, speed)[0].float().cpu().numpy()
                action = ppo_normalized_to_ctbr(action_norm, settings)
                rows.append((step, float(obs['time']), passed, env.tracker.index,
                             *state[:3], *state[7:10], float(np.linalg.norm(state[7:10])),
                             side, *local[1:3], *action_norm, *alternate))
                old_passed = env.tracker.passed_count
                obs, _, terminated, _, info = env.step(action)
                dynamics_delta = _normalize_dynamics_delta_for_control_rate(
                    np.asarray(obs['task_state'], np.float32)-prior_task, settings)
                dynamics_predictions.append(dynamics_prediction)
                dynamics_targets.append((dynamics_delta-dynamics_mean)/dynamics_std)
                dynamics_valid.append(not bool(info.get('gate_passed', False)))
                new_state = np.asarray(obs['state'], np.float32)
                new_side = float((new_state[:3]-gate.position)@gate.normal)
                if side < 0 <= new_side and env.tracker.passed_count == old_passed:
                    misses.append({'step':step+1, 'gate':int(env.tracker.index), 'passed':passed})
                if side >= 0 > new_side and env.tracker.passed_count == old_passed:
                    recrossings.append({'step':step+1, 'gate':int(env.tracker.index), 'passed':passed})
                history.append_feature(ppo_observation_features(obs, settings))
                if terminated or env.tracker.passed_count-start_passed >= target:
                    break
            stem = name.replace('/', '_') + f'-e{episode_index}'
            columns = ['step','time','passed','active_gate','x','y','z','vx','vy','vz',
                       'speed','gate_side','gate_lateral','gate_vertical',
                       'action_collective','action_roll','action_pitch','action_yaw',
                       'straight_collective','straight_roll','straight_pitch','straight_yaw']
            data = np.asarray(rows, np.float32)
            np.savez_compressed(args.out/(stem+'.npz'), rows=data, columns=np.asarray(columns),
                latent_steps=np.asarray([x[0] for x in latents],np.int32),
                base_latents=np.stack([x[1] for x in latents]),
                conditioned_latents=np.stack([x[2] for x in latents]),
                dynamics_predictions=np.asarray(dynamics_predictions,np.float32),
                dynamics_targets=np.asarray(dynamics_targets,np.float32),
                dynamics_valid=np.asarray(dynamics_valid,np.bool_))
            terminal = data[data[:,2] == target-1]
            diffs = terminal[:,18:22]-terminal[:,14:18] if len(terminal) else np.empty((0,4))
            valid = np.asarray(dynamics_valid,np.bool_)
            prediction = np.asarray(dynamics_predictions,np.float32)[valid]
            target_dynamics = np.asarray(dynamics_targets,np.float32)[valid]
            dynamics_mse = float(np.mean((prediction-target_dynamics)**2))
            dynamics_zero_mse = float(np.mean(target_dynamics**2))
            dynamics_r2 = 1.0-dynamics_mse/float(np.mean(
                (target_dynamics-target_dynamics.mean(axis=0))**2))
            entry = {'course':name,'episode_index':episode_index,'seed':seed,
                'steps':len(rows),'passed':int(env.tracker.passed_count-start_passed),
                'target':int(target),'misses':misses,'backward_recrossings':recrossings,
                'terminal_frames':len(terminal),
                'terminal_action_delta_abs_mean':np.abs(diffs).mean(axis=0).tolist() if len(diffs) else None,
                'terminal_action_delta_mean':diffs.mean(axis=0).tolist() if len(diffs) else None,
                'valid_dynamics_steps':int(valid.sum()),
                'dynamics_normalized_mse':dynamics_mse,
                'dynamics_zero_baseline_mse':dynamics_zero_mse,
                'dynamics_pooled_r2':dynamics_r2,
                'data':stem+'.npz'}
            report['courses'].append(entry)
            print(json.dumps(entry), flush=True)
        finally:
            env.close()
    (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
