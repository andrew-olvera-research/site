from pathlib import Path
import json,statistics,collections,hashlib
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'starscream/handoff/site-pretrain65-d-scaled-20260930/research/sources'
PATHS={'pretrain65':'outputs/course-pools/v6211-pretrain/manifest.json','eval100':'configs/eval/v6_22_real100_hard_v2.manifest.json','eval60':'configs/eval/v6_22_real60.manifest.json'}
out={'method':'Ranges and medians over manifest cyclic gate transitions, including the closing transition. Geometry is not flown trajectory length. No difficulty ranking is implied.','sets':{}}
for alias,path in PATHS.items():
    raw=(BASE/path).read_bytes(); records=json.loads(raw)['records']
    if alias=='pretrain65':
        records=[r for r in records if r.get('split')=='train' and r.get('qualified')]
        out['sets'][alias]={'courses':len(records),'source_path':str((BASE/path).relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(raw).hexdigest(),'selection':'split=train and qualified=true'}
        continue
    geo=[r.get('geometry',r) for r in records]
    ts=[t for g in geo for t in g['transitions']]
    vals={'gates_per_course':[g['gate_count'] for g in geo], 'spacing_m':[t['incoming_m'] for t in ts], 'turn_deg':[t['turn_deg'] for t in ts], 'absolute_height_change_m':[abs(t['height_change_m']) for t in ts]}
    out['sets'][alias]={'source_path':str((BASE/path).relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(raw).hexdigest(),'courses':len(records),'transitions':len(ts),'families':dict(collections.Counter(r['family'] for r in records)),'sources':dict(collections.Counter(r.get('source','unspecified') for r in records)),'ranges':{k:{'min':min(v),'median':statistics.median(v),'max':max(v)} for k,v in vals.items()},'course_counts':{'turn_at_least_120deg':sum(any(t['turn_deg']>=120 for t in g['transitions']) for g in geo),'height_step_at_least_2m':sum(any(abs(t['height_change_m'])>=2 for t in g['transitions']) for g in geo),'spacing_at_least_25m':sum(any(t['incoming_m']>=25 for t in g['transitions']) for g in geo),'reverse_entry':sum(any(t['reverse_entry'] for t in g['transitions']) for g in geo)}}
print(json.dumps(out,indent=2))
(ROOT/'starscream/assets/data/course-geometry.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
