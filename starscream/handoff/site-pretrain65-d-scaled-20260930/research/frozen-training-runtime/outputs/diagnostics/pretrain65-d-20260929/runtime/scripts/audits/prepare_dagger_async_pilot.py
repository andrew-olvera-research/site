"""Create an isolated bounded pipeline pilot; never reuse the production run name."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--rounds', type=int, default=221)
    p.add_argument('--episodes', type=int, default=130)
    p.add_argument('--updates', type=int, default=1270)
    p.add_argument('--eval-episodes', type=int, default=65)
    p.add_argument('--serial', action='store_true')
    p.add_argument('--event-inference', action='store_true')
    args = p.parse_args()
    cfg = json.loads(Path('configs/exp/v6.21.1/update_fix_dagger.yaml').read_text())
    if args.name == cfg['dagger']['run_name'] or '/' in args.name or '\\' in args.name:
        raise ValueError('Use a distinct simple pilot run name')
    if (Path('outputs/checkpoints') / args.name).exists():
        raise FileExistsError('Pilot checkpoint namespace already exists')
    for section in ('dagger', 'checkpoint', 'wandb'):
        cfg[section]['run_name'] = args.name
    cfg['dagger'].update(rounds=args.rounds, episodes_per_round=args.episodes,
        updates_per_round=args.updates, evaluation_episodes=args.eval_episodes,
        dagger_async_pipeline=not args.serial,
        dagger_event_inference=args.event_inference,
        dagger_pipeline_max_handoff_bytes=2*1024**3)
    cfg['wandb'].update(enabled=False, train_metric_allowlist=[], eval_metric_allowlist=[],
        event_path=f'/workspace/outputs/logs/{args.name}.events.jsonl')
    cfg['experiment_notes']['async_pilot'] = dict(
        note='Bounded integration validation; production replay capacities and sampler retained.',
        asynchronous=not args.serial, episodes=args.episodes, updates=args.updates)
    path = Path('outputs/dagger-throughput') / f'{args.name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
