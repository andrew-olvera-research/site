"""Read-only learning-curve and reporting provenance follow-up."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/diagnostics/pretraining-mode-decision-20260928'
names = {
    'bounded': 'starscream-v6.21.1.1-quality-v1.full.events.jsonl',
    'scaled': 'starscream-v6.21.1.1-plant-selection25-dagger-recovery-fix-scale.events.jsonl',
    'v3': 'starscream-v6.21.1.1-plant-recovery-support-v3.full.events.jsonl',
}
result = {}
for name, filename in names.items():
    panels, quality = [], []
    for line in (ROOT/'outputs/logs'/filename).open():
        row = json.loads(line); m = row.get('metrics', {})
        if 'eval/full_course_success' in m:
            panels.append({'step': row['step'], **{k:v for k,v in m.items() if k in (
                'eval/full_course_success','eval/selection_suite_success',
                'eval/selection_suite_clean_timely_success','eval/mean_gates',
                'eval/successful_median_steps','eval/crash_rate','eval/dagger_policy_version')}})
        if 'train/round' in m:
            quality.append({'step':row['step'], **{k:v for k,v in m.items() if
                k in ('train/round','train/learning_rate','train/loss','train/action_loss','train/dynamics_loss')
                or k.startswith('train/quality/') and any(s in k for s in ('distinct_', 'draws_per_', 'sampled_', 'stored_'))}})
    windows = {}
    for low,high in [(20,30),(30,40),(35,44),(40,43),(60,80),(100,125),(130,150),(150,177)]:
        subset=[p for p in panels if low*1e6<=p['step']<high*1e6]
        if subset:
            windows[f'{low}-{high}M']={'n':len(subset),'mean_success':statistics.mean(p['eval/full_course_success'] for p in subset),
                'mean_gates':statistics.mean(p['eval/mean_gates'] for p in subset)}
    result[name]={'windows':windows, 'last_panel':panels[-1],
        'peak':max(panels,key=lambda p:p['eval/full_course_success']),
        'quality_snapshots':[min(quality,key=lambda p:abs(p['train/round']-r)) for r in (20,40,60,80)] if quality else [],
        'last_quality':quality[-1] if quality else None}
timed=[]
for path in sorted((ROOT/'outputs/checkpoints/starscream-v6.21.1.1-plant-recovery-support-v3/timed-evaluation').glob('round-*.json')):
    d=json.loads(path.read_text())
    timed.append({'file':str(path.relative_to(ROOT)), **{k:d[k] for k in ('round','environment_steps','timely_success','clean_success','crashed','timeout')},
        'courses':len(d['tracks']),'episodes':sum(len(t['episodes']) for t in d['tracks'])})
result['v3_timed']=timed
def category(r):
    if not r['success']: return 'failed'
    return ('clean' if r['missed_crossings']==0 else 'retry') + ('_timely' if r['timely_success'] else '_late')
plant=json.loads((OUT/'plant-e1.json').read_text())
scaled=json.loads((OUT/'bounded_scale-e1.json').read_text())
a={(r['slot'],r['seed']):r for r in plant['episodes']}
b={(r['slot'],r['seed']):r for r in scaled['episodes']}
assert a.keys()==b.keys()
from collections import Counter
result['plant_to_scaled_categories']=dict(Counter(category(a[k])+' -> '+category(b[k]) for k in a))
result['scaled_failure_profile']={
    'crash_without_prior_miss':sum(r['crashed'] and not r['missed_crossings'] for r in b.values()),
    'crash_after_miss':sum(r['crashed'] and r['missed_crossings']>0 for r in b.values()),
    'failed_with_miss':sum(not r['success'] and r['missed_crossings']>0 for r in b.values()),
    'failed_before_first_gate':sum(not r['success'] and r['gates']==0 for r in b.values()),
}
(OUT/'bounded-followup-history.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{a:b for a,b in v.items() if a not in ('quality_snapshots','last_quality')} if isinstance(v,dict) else v for k,v in result.items()},indent=2))
