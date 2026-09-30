"""Matched-seed nominal policy probe of MPCC-screened VAE proposals."""
import argparse
from copy import deepcopy
from pathlib import Path
import json
import math
from scripts.train_privileged_racing import load_config, stage_config, parse_stage, evaluate_policy
from starscream.privileged_racing import load_policy_checkpoint
from starscream.env.tracks import load_track
from starscream.course_model.training import atomic_json


def main():
    p = argparse.ArgumentParser()
    for key in ('config', 'checkpoint', 'manifest', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--episodes', type=int, default=16)
    p.add_argument('--seed', type=int, default=2026090614)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--reference-controls-only', action='store_true')
    p.add_argument('--section',choices=('dagger','ppo'),default='dagger')
    p.add_argument('--randomized-environment', action='store_true',
                   help='retain the training/reporting plant contract')
    args = p.parse_args()
    if args.episodes < 1 or args.workers < 1:
        p.error('positive episodes and workers required')
    settings = deepcopy(stage_config(load_config(args.config),args.section)[args.section])
    settings['evaluation_workers'] = args.workers
    if not args.randomized_environment:
        settings['dynamics_randomization'] = {'enabled': False}
        settings.pop('state_estimator_randomization', None)
        settings.pop('flight_plan_randomization', None)
        settings.pop('action_delay_range', None)
        settings['policy_state_source'] = 'truth'
        settings['action_delay'] = float(settings.get('reporting_action_delay', .011))
    policy, normalizer, _, resolved = load_policy_checkpoint(args.checkpoint, 'cuda')
    raw = deepcopy(settings['reporting_evaluation_curriculum'])
    if isinstance(raw, list):
        raw = raw[0]
    for key in ('track_manifest', 'real_course_suite'):
        raw.pop(key, None)
    rows = []
    records = json.loads(args.manifest.read_text())['records']
    if args.reference_controls_only:
        records = [{'path': name} for name in
                   ('swift_champion_2022_exact', 'multigp_cdra_2026_reconstructed')]
    for row in records:
        path = row['path'] if args.reference_controls_only else Path(row['path'])
        if not args.reference_controls_only and not path.is_absolute():
            path = args.manifest.parent / path
        track = load_track(path)
        # Preserve the established rollout_laps/target_gates contract. Changing
        # target_gates also changes terminal route-reference construction.
        stage = parse_stage(dict(raw, tracks=[str(path)], random_gate=False,
                                 fixed_start_gate_index=0))
        metrics = evaluate_policy(policy, normalizer, settings, stage,
                                  count=args.episodes, seed_base=args.seed, device='cuda')
        # Undefined successful-lap statistics are missing, not zero or a crash.
        metrics = {k: None if isinstance(v, float) and not math.isfinite(v) else v
                   for k, v in metrics.items()}
        rows.append(dict(name=track.name, path=str(path), metrics=metrics))
        atomic_json(args.output, dict(checkpoint=str(resolved), nominal=not args.randomized_environment,
                    episodes_per_track=args.episodes, seed=args.seed, rows=rows))
        print(track.name, metrics['full_course_success'], metrics['mean_gates'], flush=True)


if __name__ == '__main__':
    main()
