"""Reproduce the matched scratch collection comparison from local event logs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/diagnostics/collection-ablation-20260929/results'
METRICS = ('full_course_success', 'timely_success', 'clean_timely_success',
           'crash_rate', 'recovered_success', 'reference_misses',
           'selection_suite_success', 'selection_suite_timely_success',
           'selection_suite_clean_timely_success')
FRESH = ('action_loss_unweighted', 'action_loss_weighted',
         'thrust_physical_mae', 'roll_physical_mae',
         'pitch_physical_mae', 'yaw_physical_mae')
COLLECT = ('active_steps', 'teacher_steps', 'learner_steps', 'prefix_steps',
           'recovery_steps', 'active_teacher_steps', 'learner_recovery_steps',
           'recovery_completed', 'recovery_failed', 'recovery_timeout',
           'invalid_state_stops', 'segment_completed', 'misses', 'backward',
           'planned_plane_crossings', 'stopped', 'valid_nominal_rows',
           'valid_corrective_rows', 'valid_recovery_rows',
           'retained_nominal_rows', 'retained_corrective_rows',
           'retained_recovery_rows')


def source(arm):
    name = ('starscream-v6.21.1.1-mini-slalom-aug2-fixes-r24' if arm == 'a'
            else f'starscream-v6.21.1.1-mini-slalom-aug2-collection-{arm}-r24')
    return ROOT / f'outputs/logs/{name}.full.events.jsonl'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arm in 'abcde':
        events = [json.loads(line) for line in source(arm).read_text().splitlines()]
        train = [e for e in events if 'train/round' in e['metrics']]
        evaluation = [e for e in events if 'eval/selection_suite_success' in e['metrics']]
        if len(train) != 24 or len(evaluation) != 25:
            raise ValueError(f'{arm} does not have 24 train and 25 eval records')
        rounds = {e['step']: int(e['metrics']['train/round']) for e in train}
        curve = [dict(round=rounds.get(e['step'], 0), steps=e['step'],
                      **{key: float(e['metrics'][f'eval/{key}']) for key in METRICS})
                 for e in evaluation]
        if [e['round'] for e in curve] != list(range(25)):
            raise ValueError(f'{arm} evaluation/round alignment changed')
        training = [{key.removeprefix('train/'):value for key,value in e['metrics'].items()}
                    for e in train]
        checkpoint = ROOT / ('outputs/checkpoints/starscream-v6.21.1.1-mini-slalom-aug2-fixes-r24/latest.pt'
            if arm == 'a' else f'outputs/checkpoints/starscream-v6.21.1.1-mini-slalom-aug2-collection-{arm}-r24/latest.pt')
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if arm != 'a':
            status = json.loads((ROOT / f'outputs/diagnostics/collection-ablation-20260929/{arm}.status.json').read_text())
            if status['state'] != 'complete' or status['exit_code'] != 0 or status['round'] != 24:
                raise ValueError(f'{arm} status is not successful')
        counts = {key: int(sum(float(t.get(f'collection/{key}', 0)) for t in training[1:]))
                  for key in COLLECT} if arm != 'a' else {}
        last = evaluation[-1]['metrics']
        courses = {key.removeprefix('eval/track/').removesuffix('/full_course_success'): float(value)
            for key,value in last.items() if key.startswith('eval/track/')
            and key.endswith('/full_course_success') and '/lap_count/' not in key}
        summary[arm] = dict(source=str(source(arm).relative_to(ROOT)),
            budget=dict(rounds=24, updates=24*626, episodes=24*64, environment_steps=curve[-1]['steps']),
            final=curve[-1], last_six={key:float(np.mean([e[key] for e in curve[-6:]])) for key in METRICS},
            fresh_final={key:float(training[-1][f'fresh/{key}']) for key in FRESH},
            fresh_late={key:float(np.mean([e[f'fresh/{key}'] for e in training[-6:]])) for key in FRESH},
            new_valid_labels=sum(float(e['new_labels']) for e in training),
            new_online_replay_labels=sum(float(e['new_online_replay_labels']) for e in training),
            underfilled_retention_rounds=[int(e['round']) for e in training[1:] if e['new_online_replay_labels'] < 25002],
            collection=counts, courses=courses, curve=curve)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    colors=dict(a='#0f766e',b='#dc2626',c='#d97706',d='#2563eb',e='#9333ea')
    for arm in 'abcde':
        rows=summary[arm]['curve']
        for ax,key in [(axes[0],'selection_suite_success'),(axes[1],'selection_suite_clean_timely_success')]:
            ax.plot([r['round'] for r in rows],[100*r[key] for r in rows],label=arm.upper(),color=colors[arm],lw=1.8)
            ax.set(xlabel='Round',ylabel='Family-weighted success (%)',ylim=(0,103));ax.grid(alpha=.2)
    axes[0].set_title('Eventual completion')
    axes[1].set_title('Clean and timely')
    for arm in 'abcde':
        x=summary[arm]['final']
        axes[2].scatter(100*x['full_course_success'],100*x['clean_timely_success'],
                        c=colors[arm],s=90,label=arm.upper())
        axes[2].annotate(arm.upper(),(100*x['full_course_success']+1,100*x['clean_timely_success']+1))
    axes[2].set(xlabel='Final raw eventual success (%)',ylabel='Final raw clean/timely success (%)',
                xlim=(0,60),ylim=(0,30),title='Final 20-episode panel')
    axes[2].grid(alpha=.2);axes[0].legend(title='Arm',fontsize=8)
    fig.suptitle('Collection pilot · equal updates, different collected steps · one seed')
    fig.savefig(OUT/'learning-curves.png',dpi=170)
    print(json.dumps({arm:{k:summary[arm][k] for k in ('budget','final','last_six','fresh_late','underfilled_retention_rounds','collection')}
                      for arm in 'abcde'},indent=2))


if __name__ == '__main__':
    main()
