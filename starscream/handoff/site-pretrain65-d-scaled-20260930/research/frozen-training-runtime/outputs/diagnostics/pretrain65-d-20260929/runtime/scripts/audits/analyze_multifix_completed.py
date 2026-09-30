"""Summarize the completed augmented warm controls and scratch fix pilot."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/diagnostics/multifix-20260928/results'


def main():
    OUT.mkdir(exist_ok=True)
    summary = {}
    for kind in ('broad', 'bounded', 'fixes', 'scratch-broad', 'scratch-bounded'):
        path = ROOT / f'outputs/logs/starscream-v6.21.1.1-mini-slalom-aug2-{kind}-r24.full.events.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        train = [e for e in events if 'train/round' in e['metrics']]
        evaluation = [e for e in events if 'eval/selection_suite_success' in e['metrics']]
        assert len(train) == 24 and len(evaluation) == 25
        round_by_step = {e['step']: int(e['metrics']['train/round']) for e in train}
        metrics = ['selection_suite_success', 'selection_suite_timely_success',
                   'selection_suite_clean_timely_success', 'full_course_success',
                   'clean_timely_success', 'timely_success', 'crash_rate', 'recovered_success',
                   'reference_misses', 'genuine_miss_episode_rate']
        curve = [dict(round=round_by_step.get(e['step'], 0), steps=e['step'],
                      **{key: e['metrics'][f'eval/{key}'] for key in metrics}) for e in evaluation]
        train_keys = ['round', 'loss', 'action_loss', 'physical_action_loss', 'dynamics_loss',
                      'online_replay', 'permanent_expert_replay', 'new_online_replay_labels']
        train_keys += [key.removeprefix('train/') for key in train[-1]['metrics'] if key.startswith('train/fresh/')]
        training = [{key: e['metrics'][f'train/{key}'] for key in train_keys} for e in train]
        final = evaluation[-1]['metrics']
        tracks = {k.removeprefix('eval/track/').removesuffix('/full_course_success'): v
                  for k, v in final.items() if k.startswith('eval/track/')
                  and k.endswith('/full_course_success') and '/lap_count/' not in k}
        summary[kind] = dict(source=str(path.relative_to(ROOT)), final=curve[-1], initial=curve[0],
            last_six={key: float(np.mean([e[key] for e in curve[-6:]])) for key in metrics},
            peak_eventual=max(curve, key=lambda e: e['selection_suite_success']),
            course_success=tracks, evaluation=curve, training=training)
    for kind in ('scratch-broad', 'scratch-bounded', 'fixes'):
        rows = summary[kind]['training']
        summary[kind]['fresh_windows'] = {label: {key: float(np.mean([r[key] for r in window])) for key in rows[-1] if key.startswith('fresh/')} for label, window in [('rounds_3_to_8', rows[2:8]), ('rounds_19_to_24', rows[18:24])]}
    fix = summary['fixes']['training']
    summary['fresh_comparison'] = {
        label: {key: float(np.mean([r[key] for r in window])) for key in fix[-1] if key.startswith('fresh/')}
        for label, window in [('rounds_3_to_8', fix[2:8]), ('rounds_19_to_24', fix[18:24])]
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    colors = {'broad': '#2563eb', 'bounded': '#d97706', 'fixes': '#059669'}
    for kind, label in [('broad', 'Broad · warm start'), ('bounded', 'Bounded · warm start'), ('fixes', 'Fixes · scratch')]:
        rows = summary[kind]['evaluation']
        for ax, metric, title in zip(axes[:2], ['selection_suite_success', 'selection_suite_clean_timely_success'],
                                      ['Eventual success', 'Clean and timely success']):
            ax.plot([r['round'] for r in rows], [100*r[metric] for r in rows], label=label, color=colors[kind], lw=1.8)
            ax.set(title=title, xlabel='Training round', ylabel='Weighted validation success (%)', ylim=(0, 103))
            ax.grid(alpha=.18)
    axes[0].legend(loc='lower right', fontsize=8)
    axes[2].plot(range(1, 25), [r['fresh/action_loss_unweighted'] for r in fix], label='Fresh, unweighted', color='#0f172a')
    axes[2].plot(range(1, 25), [r['fresh/action_loss_weighted'] for r in fix], label='Fresh, weighted', color='#059669')
    axes[2].plot(range(1, 25), [r['action_loss'] for r in fix], label='Training replay, weighted', color='#a855f7')
    axes[2].set(title='Fix arm: fresh vs replay fitting', xlabel='Training round', ylabel='Action Huber loss', ylim=(0,.58))
    axes[2].grid(alpha=.18); axes[2].legend(fontsize=8)
    fig.suptitle('24-round pilot · 20 fixed validation episodes · different initialization prevents causal ranking', fontsize=12)
    fig.savefig(OUT / 'learning-curves.png', dpi=180)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    for kind, label, color in [('scratch-broad', 'Broad scratch', '#2563eb'), ('scratch-bounded', 'Bounded scratch', '#d97706'), ('fixes', 'Fixes scratch', '#059669')]:
        rows = summary[kind]['evaluation']
        for ax, metric, title in zip(axes[:2], ['selection_suite_success', 'selection_suite_clean_timely_success'], ['Eventual success', 'Clean and timely success']):
            ax.plot([r['round'] for r in rows], [100*r[metric] for r in rows], label=label, color=color)
            ax.set(title=title, xlabel='Training round', ylabel='Family-weighted success (%)', ylim=(0, 103))
            ax.grid(alpha=.18)
        axes[2].plot(range(1,25), [r['fresh/action_loss_unweighted'] for r in summary[kind]['training']], label=label, color=color)
    axes[0].legend(fontsize=8)
    axes[2].set(title='Fresh action error, unweighted', xlabel='Training round', ylabel='Action Huber loss')
    axes[2].grid(alpha=.18)
    fig.suptitle('Matched scratch pilots: 24 rounds / 15,024 updates; one seed, repeated 20-episode panel')
    fig.savefig(OUT / 'scratch-learning-curves.png', dpi=180)
    print(json.dumps({k: {n: summary[k][n] for n in ['final', 'last_six', 'course_success']} for k in ['scratch-broad', 'scratch-bounded', 'fixes']}, indent=2))


if __name__ == '__main__':
    main()
