import json
from pathlib import Path
import numpy as np
ROOT=Path('/workspace');OUT=ROOT/'outputs/diagnostics/pretrain65-d-20260929'
names={'D':'starscream-pretrain65-d-scaled-20260929','plant':'starscream-v6.21.1.1-plant-selection25-dagger','update_fix':'starscream-v6.21.1.update-fix-pretrain65-dagger'}
data={}
for label,name in names.items():
    p=ROOT/'outputs/logs'/f'{name}.full.events.jsonl'
    if not p.exists():p=ROOT/'outputs/logs'/f'{name}.events.jsonl'
    events=[json.loads(l) for l in p.read_text().splitlines()]
    trains={r['step']:r['metrics'] for r in events if 'train/round' in r.get('metrics',{})}
    rows=[]
    for e in events:
        if 'eval/full_course_success' not in e.get('metrics',{}) or e['step'] not in trains:continue
        t=trains[e['step']];m=e['metrics']
        rows.append(dict(round=int(t['train/round']),steps=e['step'],updates=int(t['train/round'])*5081,
            **{k.split('/',1)[1]:v for k,v in m.items() if k.startswith('eval/') and '/track/' not in k and '/family/' not in k},train=t))
    data[label]=rows
keys=['round','steps','selection_suite_success','selection_suite_timely_success','selection_suite_clean_success','selection_suite_clean_timely_success','full_course_success','timely_success','clean_success','clean_timely_success','recovered_success','reference_misses','genuine_miss_episode_rate','crash_rate']
print('D CURVE')
for r in data['D']:print({k:round(r[k],4) if isinstance(r[k],float) else r[k] for k in keys if k in r})
last=data['D'][-1]
for label,rows in data.items():
    print('COMPARISON',label)
    for r in [min(rows,key=lambda r:abs(r['round']-last['round'])),min(rows,key=lambda r:abs(r['steps']-last['steps'])),rows[-1]]:
        print({k:r[k] for k in ['round','steps','full_course_success','selection_suite_success'] if k in r})
    for threshold in [.4,.5,last['selection_suite_success']]:
        hit=next((r for r in rows if r.get('selection_suite_success',r['full_course_success'])>=threshold),None)
        print('first threshold',threshold, None if hit is None else {k:hit[k] for k in ['round','steps','full_course_success','selection_suite_success'] if k in hit})
print('D WINDOWS')
for lo,hi in [(1,4),(5,8),(9,13)]:
    rows=[r for r in data['D'] if lo<=r['round']<=hi]
    print(lo,hi,{k:float(np.mean([r[k] for r in rows])) for k in keys[2:]})
print('COURSES LATEST')
suite=json.loads((ROOT/'configs/eval/v6211_vision_selection25.json').read_text())
events=[json.loads(l) for l in (ROOT/'outputs/logs'/f"{names['D']}.full.events.jsonl").read_text().splitlines()]
m=next(e['metrics'] for e in reversed(events) if e.get('metrics',{}).get('eval/dagger_policy_version')==last['round'])
for row in suite['records']:
    print(row['slot'],row['family'],{k:m.get(f"eval/track/{row['name']}/{k}") for k in ['full_course_success','timely_success','clean_success','clean_timely_success','reference_misses']})
(OUT/'learning-analysis-snapshot.json').write_text(json.dumps(data,indent=2))
