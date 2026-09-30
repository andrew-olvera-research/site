"""Validate the opt-in fresh-replay quality recipe without starting training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import train_privileged_racing as t
from starscream.dagger_quality import quality_config, track_quality_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/exp/v6.21.1.1/plant_quality_v1.yaml'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extra-rounds', type=int, default=0)
    args = p.parse_args()
    config = t.load_config(args.config)
    settings = config['dagger']
    base = t.load_config(Path('configs/exp/v6.21.1.1/plant_dagger.yaml'))['dagger']
    changed = sorted(k for k in set(base) | set(settings) if base.get(k) != settings.get(k))
    allowed = {'run_name', 'dagger_require_successful_episodes', 'dagger_successful_coverage_rounds',
               'dagger_trajectory_quality'}
    if args.extra_rounds < 0 or settings['rounds'] != base['rounds'] + args.extra_rounds:
        raise ValueError('round budget does not match the explicitly requested extension')
    if args.extra_rounds:
        allowed.update({'rounds', 'tags'})
        if settings.get('tags') != base.get('tags', []) + ['recovery-fix']:
            raise ValueError('extended recovery run must preserve original tags and add recovery-fix')
    if set(changed) - allowed:
        raise ValueError(f'unexpected change to the matched base recipe: {set(changed)-allowed}')
    if settings.get('resume_checkpoint') or settings.get('initial_checkpoint'):
        raise ValueError('scratch rerun must not inherit old weights or replay')
    quality = quality_config(settings)
    if not quality or quality.get('audit_only'):
        raise ValueError('training recipe must enable quality enforcement')
    tracks = t.configured_tracks(settings)
    if set(map(lambda x: str(Path(x).resolve()), tracks)) != set(quality['track_bounds']):
        raise ValueError('calibration must cover exactly the training courses')
    for track in tracks:
        track_quality_config(quality, track)
    if 'quality/*' not in config['wandb']['train_metric_allowlist']:
        raise ValueError('actual sampled quality metrics must be visible')
    sources = ['scripts/train_privileged_racing.py', 'starscream/dagger_quality.py',
               'starscream/dagger_replay.py', 'starscream/dagger_async.py',
               'starscream/dagger_transport.py', 'starscream/mpcc/controller.py', str(args.config)]
    result = dict(passed=True, training_courses=len(tracks), changed_dagger_keys=changed,
        run_name=settings['run_name'], rounds=settings['rounds'], fresh_weights=True, fresh_replay=True,
        extra_rounds=args.extra_rounds,
        teacher_profiles_and_speeds_unchanged=True, teacher_beta_end=settings['teacher_beta_end'],
        recovery_fraction_online=quality['recovery_fraction'],
        recovery_fraction_combined_upper_bound=quality['recovery_fraction']*settings['online_fraction'],
        calibration_sha256=quality['calibration_sha256'],
        source_sha256={str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources},
        limitation='Configuration/contract preflight; does not certify improved learned racing behavior.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
