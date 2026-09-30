"""Audit actor publication/commit boundaries and prepare an isolated resume trial."""
import argparse
import json
from pathlib import Path
import torch


def equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='starscream-dagger-async-pilot-v2')
    parser.add_argument('--resume-name', default='starscream-dagger-async-resume-v1')
    parser.add_argument('--force-rollback', action='store_true',
                        help='Exercise rollback/discard only in the isolated resume trial')
    parser.add_argument('--micro-rollback', action='store_true',
                        help='Create a tiny controlled repeated-rollback regression fixture')
    args = parser.parse_args()
    root = Path('outputs/checkpoints') / args.run
    production = torch.load('outputs/checkpoints/starscream-v6.21.1.update-fix-pretrain65-dagger/latest.pt',
                            map_location='cpu', weights_only=False)
    checkpoints = {}
    paths = {}
    for path in root.glob('best-step-*.pt'):
        state = torch.load(path, map_location='cpu', weights_only=False)
        checkpoints[int(state['round'])] = state
        paths[int(state['round'])] = path
    a, b = checkpoints[219], checkpoints[220]
    final = torch.load(root/'latest.pt', map_location='cpu', weights_only=False)
    checks = dict(
        first_pending_window=a['dagger_async_pending']['window'] == 220,
        first_pending_actor_version=a['dagger_async_pending']['actor_version'] == 218,
        first_pending_actor_exact=equal(a['dagger_async_pending']['actor'], production['model']),
        second_pending_window=b['dagger_async_pending']['window'] == 221,
        second_pending_actor_version=b['dagger_async_pending']['actor_version'] == 219,
        second_pending_actor_exact=equal(b['dagger_async_pending']['actor'], a['model']),
        final_pending_empty=final['dagger_async_pending'] is None,
        final_round=final['round'] == 221)
    if not all(checks.values()):
        raise AssertionError(checks)
    cfg = json.loads((Path('outputs/dagger-throughput') / f'{args.run}.json').read_text())
    if (Path('outputs/checkpoints')/args.resume_name).exists():
        raise FileExistsError('Resume namespace exists')
    for section in ('dagger', 'checkpoint', 'wandb'):
        cfg[section]['run_name'] = args.resume_name
    cfg['dagger']['resume_checkpoint'] = str(paths[219].resolve())
    if args.micro_rollback:
        fixture = dict(a, dagger_async_pending=None)
        fixture_path = Path('outputs/dagger-throughput') / f'{args.resume_name}-source.pt'
        torch.save(fixture, fixture_path)
        cfg['dagger'].update(resume_checkpoint=str(fixture_path.resolve()),
            episodes_per_round=65, updates_per_round=2, evaluation_episodes=2,
            rollout_envs=16, evaluation_workers=2, evaluation_envs_per_worker=1,
            reporting_evaluation_interval=0)
    if args.force_rollback or args.micro_rollback:
        cfg['dagger'].update(dagger_rollback_on_selection_regression=True,
                            dagger_rollback_selection_tolerance=-1000000.,
                            dagger_rollback_start_round=220)
        cfg['experiment_notes']['forced_rollback'] = 'Failure-path test only; intentionally reject every candidate.'
    cfg['wandb']['event_path'] = f'/workspace/outputs/logs/{args.resume_name}.events.jsonl'
    dest = Path('outputs/dagger-throughput')
    (dest / f'{args.resume_name}.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (dest/'async-checkpoint-audit.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
