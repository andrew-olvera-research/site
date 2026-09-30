"""Correlate sampled device/memory telemetry with saved DAgger phase intervals."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    samples = [json.loads(line) for line in (Path('outputs/logs')/f'{args.tag}.system.jsonl').read_text().splitlines()]
    rows = [json.loads(line)['metrics'] for line in (Path('outputs/logs')/f'{args.run}.events.jsonl').read_text().splitlines()]
    trains = [row for row in rows if 'train/round' in row]
    intervals = []
    for train in trains:
        child = 'train/collection_last_call_host/'
        intervals.append(('Collection', int(train['train/round']),
            train.get(child+'pipeline_started_unix', train['train/collection_started_unix']),
            train.get(child+'pipeline_collect_seconds', train['train/collection_seconds'])))
        for name, prefix in [('Updates', 'updates'), ('Validation', 'evaluation')]:
            intervals.append((name, int(train['train/round']), train[f'train/{prefix}_started_unix'], train[f'train/{prefix}_seconds']))
    groups = {}
    for sample in samples:
        active = sorted({name for name, _, start, duration in intervals if start <= sample['unix'] < start+duration})
        label = ' + '.join(active) or 'Other/startup'
        groups.setdefault(label, []).append(sample)
    result = dict(
        gpu_scope='Device-wide nvidia-smi samples, not per-process; means are sampled rather than exact busy-time integrals.',
        minimum_available_gib=min(s['memory_available_gib'] for s in samples),
        peak_swap_used_gib=max(s['swap_used_gib'] for s in samples),
        phases={label: dict(samples=len(group),
            gpu_mean_percent=statistics.mean(s['gpu_utilization_percent'] for s in group if 'gpu_utilization_percent' in s)
                if any('gpu_utilization_percent' in s for s in group) else None,
            gpu_peak_mib=max((s.get('gpu_memory_mib', 0) for s in group)),
            available_mean_gib=statistics.mean(s['memory_available_gib'] for s in group))
            for label, group in groups.items()})
    dest = Path('outputs/dagger-throughput')
    (dest/f'{args.tag}-system-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin = min(s['unix'] for s in samples)
    times = [(s['unix']-origin)/60 for s in samples]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 1.5]})
    axes[0].plot(times, [s.get('gpu_utilization_percent', float('nan')) for s in samples], color='#2878b5', lw=1.2)
    axes[0].set(ylabel='GPU utilization (%)', ylim=(0, 102))
    axes[1].plot(times, [s['memory_available_gib'] for s in samples], label='Available RAM', color='#24845b')
    axes[1].plot(times, [s['swap_used_gib'] for s in samples], label='Occupied swap', color='#be5b30')
    axes[1].set_ylabel('GiB')
    axes[1].legend(loc='upper right')
    colors = {'Collection': '#24845b', 'Updates': '#2878b5', 'Validation': '#9364a6'}
    lanes = {'Collection': 2, 'Updates': 1, 'Validation': 0}
    for name, window, start, duration in intervals:
        axes[2].broken_barh([((start-origin)/60, duration/60)], (lanes[name]-.3, .6), facecolors=colors[name])
        if duration > 12:
            axes[2].text((start-origin+duration/2)/60, lanes[name], str(window), color='white', ha='center', va='center', fontsize=9)
    axes[2].set_yticks([0, 1, 2], ['Validation', 'Updates', 'Collection'])
    axes[2].set_xlabel('Minutes since monitor start')
    for ax in axes:
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('DAgger pipeline: GPU use, memory and overlapping phases')
    fig.tight_layout()
    fig.savefig(dest/f'{args.tag}-usage.png', dpi=160)


if __name__ == '__main__':
    main()
