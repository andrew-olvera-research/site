"""Summarize saved pipeline pilots without adding overlapping phase durations."""
import argparse
import json
from pathlib import Path


def summarize(run, initial_steps):
    path = Path('outputs/logs') / f'{run}.events.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    trains = [r['metrics'] for r in rows if 'train/round' in r['metrics']]
    evaluations = {r['step']: r['metrics'] for r in rows if 'eval/full_course_success' in r['metrics']}
    windows = []
    for train in trains:
        step = int(train['train/environment_steps'])
        evaluation = evaluations.get(step, {})
        window = {key: train.get('train/'+key) for key in (
            'round', 'round_seconds', 'collection_wait_seconds', 'collection_seconds',
            'updates_seconds', 'evaluation_seconds', 'new_labels', 'safety_rollback',
            'memory_available_gib', 'swap_used_gib', 'memory_swap_in_bytes_per_second',
            'memory_swap_out_bytes_per_second')}
        window.update(full_course_success=evaluation.get('eval/full_course_success'),
                      crash_rate=evaluation.get('eval/crash_rate'))
        windows.append(window)
    seconds = sum(w['round_seconds'] for w in windows)
    steps = int(trains[-1]['train/environment_steps']) - initial_steps
    return dict(run=run, recorded_seconds=seconds, environment_steps=steps,
                steps_per_recorded_second=steps/seconds, windows=windows,
                note='Recorded round timers exclude startup and checkpoint writes; phases overlap.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--initial-steps', type=int, default=152711599)
    p.add_argument('--output', type=Path, default=Path('outputs/dagger-throughput/pipeline-comparison.json'))
    args = p.parse_args()
    results = [summarize(run, args.initial_steps) for run in args.runs]
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    for result in results:
        print({key: result[key] for key in ('run', 'recorded_seconds', 'environment_steps', 'steps_per_recorded_second')})


if __name__ == '__main__':
    main()
