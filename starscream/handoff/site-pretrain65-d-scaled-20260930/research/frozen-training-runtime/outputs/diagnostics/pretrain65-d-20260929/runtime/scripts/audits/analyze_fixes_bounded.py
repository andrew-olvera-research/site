"""Compare corrected scratch broad and full bounded-package controls."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/diagnostics/fixes-bounded-20260929/results'
METRICS = ('full_course_success','timely_success','clean_timely_success',
           'crash_rate','recovered_success','reference_misses',
           'selection_suite_success','selection_suite_timely_success',
           'selection_suite_clean_timely_success')
FRESH = ('action_loss_unweighted','action_loss_weighted',
         'thrust_physical_mae','roll_physical_mae','pitch_physical_mae','yaw_physical_mae')


def load(name):
    path = ROOT / f'outputs/logs/{name}.full.events.jsonl'
    events=[json.loads(line) for line in path.read_text().splitlines()]
    train=[e['metrics'] for e in events if 'train/round' in e['metrics']]
    evaluations=[e['metrics'] for e in events if 'eval/selection_suite_success' in e['metrics']]
    if len(train)!=24 or len(evaluations)!=25:
        raise ValueError(f'{name} incomplete')
    final=evaluations[-1]
    return dict(source=str(path.relative_to(ROOT)),
        final={k:float(final[f'eval/{k}']) for k in METRICS},
        last_six={k:float(np.mean([m[f'eval/{k}'] for m in evaluations[-6:]])) for k in METRICS},
        fresh_final={k:float(train[-1][f'train/fresh/{k}']) for k in FRESH},
        fresh_last_six={k:float(np.mean([m[f'train/fresh/{k}'] for m in train[-6:]])) for k in FRESH},
        environment_steps=int(train[-1]['train/environment_steps']),
        valid_labels=sum(int(m['train/new_labels']) for m in train),
        retained_online_labels=sum(int(m['train/new_online_replay_labels']) for m in train),
        per_course={key.removeprefix('eval/track/').removesuffix('/full_course_success'):float(value)
            for key,value in final.items() if key.startswith('eval/track/')
            and key.endswith('/full_course_success') and '/lap_count/' not in key},
        curve=[{k:float(m[f'eval/{k}']) for k in METRICS} for m in evaluations],
        quality_collection={key.removeprefix('train/quality/collection/'):int(sum(
            m.get(key,0) for m in train[1:])) for key in train[-1]
            if key.startswith('train/quality/collection/')},
        final_training={key.removeprefix('train/'):value for key,value in train[-1].items()
            if key in ('train/action_loss','train/physical_action_loss','train/dynamics_loss',
                       'train/online_replay','train/permanent_expert_replay','train/teacher_execution_fraction')
            or key.startswith('train/quality/combined/sampled_')})


def main():
    OUT.mkdir(exist_ok=True)
    status=json.loads((ROOT/'outputs/diagnostics/fixes-bounded-20260929/fixes_bounded.status.json').read_text())
    if status['state']!='complete' or status['exit_code']!=0 or status['round']!=24:
        raise ValueError('bounded arm status is not complete')
    summary={kind:load(name) for kind,name in {
        'fixes_broad':'starscream-v6.21.1.1-mini-slalom-aug2-fixes-r24',
        'fixes_bounded':'starscream-v6.21.1.1-mini-slalom-aug2-fixes-bounded-r24',
    }.items()}
    if summary['fixes_bounded']['environment_steps']!=status['environment_steps']:
        raise ValueError('bounded status and event steps differ')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for kind,label,color in [('fixes_broad','A: fixes, broad','#0f766e'),
                              ('fixes_bounded','Fixes, bounded','#dc2626')]:
        for ax,key,title in zip(axes,('selection_suite_success','selection_suite_clean_timely_success'),
                                ('Eventual completion','Clean and timely completion')):
            ax.plot(range(25),[100*m[key] for m in summary[kind]['curve']],label=label,color=color,lw=1.9)
            ax.set(title=title,xlabel='Round',ylabel='Family-weighted success (%)',ylim=(0,55))
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Matched corrected scratch arms · 24 rounds / 15,024 updates · same fixed 20-episode panel')
    fig.savefig(OUT/'learning-curves.png',dpi=170)
    print(json.dumps({k:{j:v[j] for j in ('final','last_six','fresh_final','environment_steps')}
                      for k,v in summary.items()},indent=2))


if __name__=='__main__':
    main()
