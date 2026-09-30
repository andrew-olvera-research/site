"""Bounded real-MPCC collection and replay validation; never trains or publishes."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import train_privileged_racing as t
from starscream.dagger_quality import QualityReplayPlan, quality_subsample, FIELDS

PILOT_QUALITY = dict(enabled=True, nominal_distance=.75, intervention_distance=1.5,
    maximum_distance=5., maximum_gate_seconds=6., recovery_seconds=3.,
    recovery_gate_passes=2, nominal_fraction=.70, corrective_fraction=.25,
    recovery_fraction=.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--courses', type=int, default=6)
    parser.add_argument('--config', type=Path, default=Path('configs/exp/v6.21.1.1/plant_dagger.yaml'))
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/checkpoints/starscream-v6.21.1.1-plant-selection25-dagger/best-step-150819355-selection_suite_success-0.57875.pt'))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--betas', default='1,.35')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synchronous', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--bounds', type=Path)
    parser.add_argument('--seed', type=int, default=2046092611)
    parser.add_argument('--track-list', type=Path)
    parser.add_argument('--speed-scale', type=float, default=1.)
    parser.add_argument('--dart', action='store_true')
    parser.add_argument('--prefix-fraction', type=float, default=0.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    config = t.load_config(args.config)
    settings = config['dagger'].copy()
    tracks = t.configured_tracks(settings)
    if not 1 <= args.courses <= len(tracks) or args.repeats < 1:
        parser.error('invalid course/repeat count')
    selected = tuple(tracks[i] for i in np.linspace(0, len(tracks)-1, args.courses, dtype=int))
    if args.track_list:
        selected = tuple(json.loads(args.track_list.read_text()))
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(tracks):
            parser.error('track list must contain distinct training courses')
    if not 0 < args.speed_scale <= 1:
        parser.error('speed scale must be in (0,1]')
    if not 0 <= args.prefix_fraction <= 1 or (args.dart and any(float(b) != 1 for b in args.betas.split(','))):
        parser.error('invalid prefix fraction or mixed-policy DART probe')
    quality = {**settings.get('dagger_trajectory_quality', PILOT_QUALITY), 'audit_only': args.audit_only}
    if args.bounds:
        quality['track_bounds'] = json.loads(args.bounds.read_text())['track_bounds']
    settings.update(dagger_trajectory_quality=quality,
        mpcc_build_root='/tmp/starscream-quality-validation',
        dagger_require_successful_episodes=False, rollout_envs=args.workers,
        dagger_async_collection=not args.synchronous,
        dagger_async_pipeline=False, dagger_transition_start_episode_fraction=args.prefix_fraction,
        dagger_host_inference_graph=False, dagger_event_inference=False,
        compile_policy_inference=False, cuda_graph_policy_inference=False)
    # Per-episode resets deliberately override the MPCC constructor's speed.
    # Change the DAgger command schedule, then verify executed episode labels.
    settings['dagger_teacher_speed_fractions'] = [args.speed_scale]
    stage = replace(t.parse_stage(settings['curriculum']), tracks=selected)
    checkpoint = args.checkpoint
    policy, normalizer, _, _ = t.load_policy_checkpoint(checkpoint, 'cuda')
    policy.eval()
    collector = t.ProcessDaggerCollector(policy, normalizer, settings, stage, 'cuda')
    results = []
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for beta in map(float, args.betas.split(',')):
            started = time.perf_counter()
            batch = collector.collect(episodes=len(selected)*args.repeats, beta=beta,
                seed_base=args.seed, dart=args.dart, tracks=selected)
            data = np.asarray(batch.trajectory)
            speeds = np.asarray(batch.speed_commands)
            for track, speed in zip(batch.tracks, speeds):
                expected = collector.track_target_speeds[str(Path(track).resolve())] * args.speed_scale
                if not np.isclose(speed, expected):
                    raise AssertionError(f'episode speed override ignored: {speed} != {expected}')
            names = sorted(set(batch.tracks))
            ids = np.asarray([names.index(x) for x in batch.tracks])
            plan = None
            keep = np.arange(len(data))
            if not args.audit_only:
                keep = quality_subsample(np.random.default_rng(13), ids, data, 60000, quality)
                assert len(keep) and not np.isin(data[keep, 0], [-1, 3]).any()
                assert np.mean(data[keep, 0] == 2) <= quality['recovery_fraction']
                plan = QualityReplayPlan(ids[keep], ids[keep], np.asarray(batch.active_gate_indices)[keep],
                    data[keep], 998, config=quality)
                for _ in range(100):
                    plan.sample(np.random.default_rng(_))
                assert plan.sample_counts[2]/plan.sample_counts.sum() <= quality['recovery_fraction']
            np.savez_compressed(args.output/f'beta-{beta}.npz', trajectory=data, tracks=ids,
                track_names=np.asarray(names), fields=np.asarray(FIELDS), retained_indices=keep,
                successful=np.asarray(batch.successful_labels), gate_indices=np.asarray(batch.active_gate_indices),
                speed_commands=speeds)
            per_track = []
            for i, name in enumerate(names):
                d = data[ids == i]
                per_track.append(dict(track=name, rows=len(d),
                    quality_counts={str(c): int(np.sum(d[:, 0] == c)) for c in (-1,0,1,2,3)},
                    distance_quantiles=np.quantile(d[np.isfinite(d[:,4]),4], [.5,.95,.99,1]).tolist(),
                    misses=int(d[:,1].sum()), interventions=int(np.sum((d[:,0]==3) & (d[:,15]>0))),
                    qualified_segments=int(np.sum(d[:,10]==1)), stopped=int(d[:,11].sum())))
            result = dict(beta=beta, seconds=time.perf_counter()-started, environment_steps=batch.environment_steps,
                labels=len(data), retained=len(keep), accepted_episodes=batch.accepted_episodes,
                rejected_episodes=batch.rejected_episodes, counters=batch.quality_counts,
                solver_failures=batch.solver_failures, audit_only=args.audit_only,
                observed_speed_commands=np.unique(speeds).tolist(),
                sampled_fractions=(plan.sample_counts/plan.sample_counts.sum()).tolist() if plan else None,
                quota_shortfalls=plan.shortfall.tolist() if plan else None, per_track=per_track)
            results.append(result)
            report = dict(checkpoint=str(checkpoint), quality=quality,
                synchronous=args.synchronous, canonical_starts=args.prefix_fraction == 0,
                prefix_fraction=args.prefix_fraction, dart=args.dart, training_courses_only=True,
                speed_scale=args.speed_scale,
                seed_base=args.seed, bounds_status='pilot bounds; not approved full-run calibration', results=results)
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k!='per_track'}), flush=True)
    finally:
        collector.close()


if __name__ == '__main__':
    main()
