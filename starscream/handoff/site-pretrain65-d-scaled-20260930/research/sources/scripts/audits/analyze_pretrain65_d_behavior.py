"""Raw episode/gate-event analysis; success times explicitly exclude failures."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/diagnostics/pretrain65-d-behavior-20260929'
OUT.mkdir(parents=True,exist_ok=True)

def stats(x):
    a=np.asarray(x,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),maximum=float(a.max())) if len(a) else dict(n=0)

def events(r):
    last=0; first=None; misses=0; recoveries=[]; gaps=[]
    for e in r['events']:
        if e['kind']=='miss':
            if first is None:first=e['step']
            misses+=1
        elif e['kind']=='pass':
            gaps.append((e['step']-last)/130)
            last=e['step']
            if first is not None:recoveries.append(dict(seconds=(e['step']-first)/130,misses=misses,gate=e['gate']))
            first=None;misses=0
    return dict(max_completed_gate_gap_s=max(gaps,default=0),max_no_pass_interval_s=max(gaps+[(r['steps']-last)/130]),
        max_resolved_recovery_s=max((x['seconds'] for x in recoveries),default=0),
        recovery_seconds=sum(x['seconds'] for x in recoveries),resolved_recoveries=len(recoveries),
        repeat_miss_recoveries=sum(x['misses']>1 for x in recoveries),
        unresolved_recovery_s=(r['steps']-first)/130 if first is not None else 0,recoveries=recoveries)

def summarize(rows):
    done=[r for r in rows if r['success']]
    n=len(rows)
    return dict(n=n,successes=len(done),sr=len(done)/n,timely=sum(r['timely_success'] for r in rows)/n,
        clean_timely=sum(r['clean_success'] and r['timely_success'] for r in rows)/n,
        clean=sum(r['clean_success'] for r in rows)/n,crashes=sum(r['crashed'] for r in rows)/n,
        successful_lap_s=stats([r['steps']/130 for r in done]),
        successful_reference_ratio=stats([r['steps']/130/r['reference_seconds'] for r in done]),
        successful_max_no_pass_s=stats([r['max_no_pass_interval_s'] for r in done]),
        successes_with_recovery_over_s={str(t):sum(r['max_resolved_recovery_s']>t for r in done) for t in [3,5,8,10,15,20]},
        successes_with_no_pass_over_s={str(t):sum(r['max_no_pass_interval_s']>t for r in done) for t in [3,5,8,10,15,20]},
        termination=dict(Counter(r['termination_reason'] for r in rows)))

def writecsv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    protocol=json.loads((ROOT/'configs/eval/real100_v2_timed_protocol_v1.json').read_text())
    records={r['slot']:r for r in protocol['records']}
    raw=json.loads((ROOT/'outputs/evals/pretrain65-d-scaled-final-20260929/real100-v2-e32.json').read_text())
    rows=[dict(r,**events(r)) for r in raw['episodes']]
    bycourse=defaultdict(list);byfamily=defaultdict(list)
    for r in rows:bycourse[r['slot']].append(r);byfamily[r['family']].append(r)
    courses={s:dict(summarize(rr),name=records[s]['name'],family=records[s]['family'],reference_s=records[s]['reference_seconds'],deadline_s=records[s]['deadline_steps']/130,
        failure_at_next_gate_1based=dict(Counter(r['gates']+1 for r in rr if not r['success'])),
        miss_at_gate_1based=dict(Counter(e['gate']+1 for r in rr for e in r['events'] if e['kind']=='miss'))) for s,rr in bycourse.items()}
    groups={}
    for label,fn in [('clean_timely',lambda r:r['clean_success'] and r['timely_success']),('clean_late',lambda r:r['clean_success'] and not r['timely_success']),('miss_timely',lambda r:r['success'] and not r['clean_success'] and r['timely_success']),('miss_late',lambda r:r['success'] and not r['clean_success'] and not r['timely_success']),('noncompletion',lambda r:not r['success'])]:
        groups[label]=summarize([r for r in rows if fn(r)])
    result=dict(aggregate=summarize(rows),groups=groups,
        course_medians_s=stats([v['successful_lap_s']['median'] for v in courses.values() if v['successful_lap_s']['n']]),
        course_p90s_s=stats([v['successful_lap_s']['p90'] for v in courses.values() if v['successful_lap_s']['n']]),
        saturation=dict(perfect_eventual_courses=sum(v['sr']==1 for v in courses.values()),perfect_clean_timely_courses=sum(v['clean_timely']==1 for v in courses.values()),zero_clean_timely_courses=sum(v['clean_timely']==0 for v in courses.values()),clean_timely_at_least_90pct=sum(v['clean_timely']>=.9 for v in courses.values())),
        event_lists_at_limit=sum(len(r['events'])>=256 for r in rows),
        successes_multiple_recovered_gates=sum(r['success'] and r['resolved_recoveries']>=2 for r in rows),
        successes_repeat_miss_recovery=sum(r['success'] and r['repeat_miss_recoveries']>0 for r in rows),
        courses=courses,families={s:summarize(rr) for s,rr in byfamily.items()})
    (OUT/'full-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    writecsv(OUT/'courses.csv',[dict(slot=s,name=v['name'],family=v['family'],n=v['n'],successes=v['successes'],sr=v['sr'],timely=v['timely'],clean_timely=v['clean_timely'],median_s=v['successful_lap_s'].get('median'),p90_s=v['successful_lap_s'].get('p90'),max_s=v['successful_lap_s'].get('maximum'),reference_s=v['reference_s'],deadline_s=v['deadline_s']) for s,v in courses.items()])
    writecsv(OUT/'episodes.csv',[{k:v for k,v in r.items() if k not in ['events','recoveries']} for r in rows])
    worst=sorted((r for r in rows if r['success']),key=lambda r:r['max_resolved_recovery_s'],reverse=True)[:30]
    (OUT/'long-recovery-examples.json').write_text(json.dumps(worst,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['courses','families','groups']},indent=2))
    print('WORST MEDIANS',[(s,round(v['successful_lap_s'].get('median',0),2),v['successes']) for s,v in sorted(courses.items(),key=lambda x:x[1]['successful_lap_s'].get('median',0),reverse=True)[:10]])
    print('A2RL',courses['a2rl_s2_2026_source_consistent_v2'])
if __name__=='__main__':main()
