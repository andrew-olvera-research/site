"""Freeze RL7.4's bank, calibrate source baselines, prepare lap-time PPO."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train_privileged_racing import (load_config, stage_config, parse_stage,
    load_policy_checkpoint, ProcessRaceCollector, multitrack_metrics, lap_timing_metrics,
    validated_exploration_log_std)
from starscream.ppo_lap_time import succeeded

ROOT = Path('/workspace')
NAME = 'starscream-v6.21.rl.8.0-laptime-100m'
SOURCE = ROOT / 'outputs/checkpoints/starscream-v6.21.rl.7.4-unified-real50-100m/best-step-100208640-full_course_success-0.8125.pt'
ARTIFACT = ROOT / 'outputs/diagnostics/v621-rl8-laptime'
CONFIG = ROOT / 'configs/exp/v6.21.rl.8/laptime_100m.yaml'


def build():
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    c = stage_config(load_config(ROOT / 'configs/exp/v6.21.rl.7/unified_real50_100m.yaml'), 'ppo')
    s = c['ppo']
    tracks = source['ppo_adaptive_sampling_state']['tracks']
    assert len(tracks) == 200 and len({Path(t).stem for t in tracks}) == 200
    assert all(Path(t).is_file() for t in tracks)
    for key in ('actor_optimizer_initial_checkpoint', 'critic_initial_checkpoint',
                'ppo_saturation_mask_threshold', 'ppo_noise_bounds', 'track_manifest'):
        s.pop(key, None)
    s.update(run_name=NAME, track='rl74-frozen200-real50-validation', initial_checkpoint=str(SOURCE),
        restore_actor_optimizer=False, restore_critic_optimizer=False,
        initialize_online_course_bank_from_checkpoint=False, resume_online_course_bank=False,
        seed=2026091908, initial_environment_steps=0, target_environment_steps=100_000_000,
        cycles=math.ceil(100_000_000 / 204800), rollout_envs=400, episodes_per_cycle=400,
        ppo_envs_per_worker=20, ppo_rollout_window_steps=512, ppo_critic_warmup_cycles=10,
        discount=1., gae_lambda=.995, ppo_exploration_correlation=0.,
        actor_learning_rate=1e-6, actor_learning_rate_min=2e-7, actor_learning_rate_max=5e-6,
        evaluation_interval=10, evaluation_episodes=800, evaluation_seed=2036091908,
        reporting_evaluation_interval=20, reporting_evaluation_episodes=800,
        reporting_evaluation_seed=2036091908, monitor='lap_time_score',
        checkpoint_rank_curriculum_stages=[0], terminal_success_bonus=0., terminal_failure_penalty=0.,
        tags=['v6.21', 'rl8', 'lap-time', 'completion-constrained', 'iid-full-likelihood'])
    for key in ('ppo_online_course_bank', 'ppo_adaptive_level_replay', 'online_manifold_curriculum'):
        s[key] = {'enabled': False}
    for key in list(s['reward']):
        if key.endswith('_weight') or key == 'time_penalty_per_second':
            s['reward'][key] = 0.
    stage = copy.deepcopy(s['curriculum'][0])
    for key in ('track_manifest', 'track_split', 'qualified_tracks_only', 'target_speed_range'):
        stage.pop(key, None)
    stage.update(name='lap_time_with_completion', tracks=tracks, target_speed=16.5,
                 minimum_environment_steps=100_000_000)
    s['curriculum'] = [stage]
    c['wandb'].update(enabled=True, mode='online', group='v6.21.rl.8', run_name=NAME,
                      local_event_path=f'/workspace/outputs/logs/{NAME}.events.jsonl')
    c['experiment_notes'] = dict(
        objective='Undiscounted -elapsed/reference; failure total=-timeout/reference-1-dual. No progress/speed shaping.',
        constraints='Per-course source IID SR minus .05; EMA dual ascent. Validation aggregate SR minus .03 and each course minus .125; reject ranking on violations.',
        ranking='Equal-course successful mean lap-time improvement on frozen source-qualified real50 courses. Empirical guards, not safety guarantees.',
        data='Frozen 200-course source bank; no new tracks. Real50 is validation, not untouched test.',
        accounting='100M new RL transitions, rounded up to a complete 204800-transition window; calibration/evaluation excluded.',
        limitations='IID exploration and reward change together; this is an efficacy experiment, not an isolated reward ablation. Source baseline estimates have finite-sample uncertainty.')
    return stage_config(c, 'ppo')


def calibrate(c):
    s = c['ppo']
    policy, normalizer, _, _ = load_policy_checkpoint(SOURCE, 'cuda')
    policy.set_exact_likelihood_mode(True)
    with torch.no_grad():
        policy.log_std_parameter.copy_(validated_exploration_log_std(
            s['exploration_log_std'], policy.log_std_parameter, policy.minimum_log_std, policy.maximum_log_std))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    rows_by_split = {}
    for split, raw, count, stochastic in (
        ('training', s['curriculum'][0], 3200, True),
        ('validation', s['evaluation_curriculum'], 800, False)):
        path = ARTIFACT / f'{split}_source_episodes.json'
        if path.exists():
            rows = json.loads(path.read_text())
        else:
            print(f'calibrating {split}: {count} episodes stochastic={stochastic}', flush=True)
            np.random.seed(s['evaluation_seed']); torch.manual_seed(s['evaluation_seed'])
            settings = dict(s, compile_policy_backbone=False)
            collector = ProcessRaceCollector(policy, normalizer, settings, parse_stage(raw), 'cuda', workers=32)
            try:
                rows = collector.evaluate_rows(episodes=count, seed_base=s['evaluation_seed'], stochastic=stochastic)
            finally:
                collector.close()
            path.write_text(json.dumps(rows, indent=2) + '\n')
        rows_by_split[split] = rows
        print(f'{split} source SR={np.mean([succeeded(r) for r in rows]):.4f}', flush=True)
    courses = {}
    for track in s['curriculum'][0]['tracks']:
        name = Path(track).stem
        rows = [r for r in rows_by_split['training'] if Path(r['track']).stem == name]
        assert len(rows) == 16, (name, len(rows))
        times = [r['steps'] / s['control_hz'] for r in rows if succeeded(r)]
        courses[name] = dict(reference_seconds=float(np.mean(times)) if times else s['curriculum'][0]['max_steps'] / s['control_hz'],
                             success_rate=float(np.mean([succeeded(r) for r in rows])), episodes=len(rows),
                             reference_fallback=not bool(times))
    rows = rows_by_split['validation']
    validation = dict(success_floor=max(0., float(np.mean([succeeded(r) for r in rows])) - .03), courses={})
    for track in s['evaluation_curriculum']['tracks']:
        name = Path(track).stem
        subset = [r for r in rows if Path(r['track']).stem == name]
        assert len(subset) == 16
        times = [r['steps']/s['control_hz'] for r in subset if succeeded(r)]
        validation['courses'][name] = dict(success_floor=max(0., len(times)/16 - .125),
            source_success_rate=len(times)/16, pace_eligible=len(times)>=4,
            reference_seconds=float(np.mean(times)) if times else 0.)
    s['ppo_lap_time'] = dict(enabled=True, training_courses=courses, validation=validation,
        initial_dual=1., maximum_dual=20., dual_learning_rate=2., dual_minimum_episodes=16,
        training_success_tolerance=.05)
    provenance = dict(source=str(SOURCE), source_sha256=hashlib.file_digest(SOURCE.open('rb'), 'sha256').hexdigest(),
        course_sha256={t:hashlib.sha256(Path(t).read_bytes()).hexdigest() for t in s['curriculum'][0]['tracks']})
    (ARTIFACT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(c, indent=2)+'\n')
    print(f'prepared {CONFIG}; pace pool={sum(r["pace_eligible"] for r in validation["courses"].values())}', flush=True)


if __name__ == '__main__':
    calibrate(build())
