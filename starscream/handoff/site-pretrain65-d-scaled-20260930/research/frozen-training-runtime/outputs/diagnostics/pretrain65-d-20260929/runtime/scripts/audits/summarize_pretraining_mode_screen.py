"""Derive observable behavior categories and paired course intervals from the screen."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/diagnostics/pretraining-mode-decision-20260928'

def main():
    reports = {}
    rows = []
    for path in sorted(OUT.glob('*-e1.json')):
        d=json.loads(path.read_text())
        if not isinstance(d,dict) or 'aggregate' not in d:
            continue
        name=path.name.removesuffix('-e1.json')
        episodes=d['episodes']
        assert len(episodes)==100 and len({(r['slot'],r['seed']) for r in episodes})==100
        assert d['miss_definition']=='ordered-reference-v1'
        assert d['hard_cap_steps']==6000 and d['retry_limit_seconds']>6000/130 and d['gate_dwell_limit_seconds']>6000/130
        successes=[r for r in episodes if r['success']]
        for r in episodes:
            r['clean_timely']=int(r['success'] and r['timely_success'] and r['missed_crossings']==0)
            r['retry_success']=int(r['success'] and r['missed_crossings']>0)
            assert r['success']==r['clean_success']+r['retry_success']
        reports[name]=d
        n=len(episodes)
        rows.append(dict(name=name, checkpoint_round=d['checkpoint_round'], collected_steps=d['checkpoint_steps'],
            episodes=n, success=sum(r['success'] for r in episodes), timely=sum(r['timely_success'] for r in episodes),
            clean_timely=sum(r['clean_timely'] for r in episodes), clean_anytime=sum(r['clean_success'] for r in episodes),
            recovered_timely=sum(bool(r['retry_success'] and r['timely_success']) for r in episodes),
            clean_late=sum(bool(r['clean_success'] and not r['timely_success']) for r in episodes),
            recovered_late=sum(bool(r['retry_success'] and not r['timely_success']) for r in episodes),
            retry_success=sum(r['retry_success'] for r in episodes),
            retry_fraction_of_successes=sum(r['retry_success'] for r in episodes)/len(successes) if successes else None,
            late_fraction_of_successes=sum(r['late_success'] for r in episodes)/len(successes) if successes else None,
            median_success_seconds=statistics.median(r['steps']/130 for r in successes) if successes else None,
            median_success_reference_ratio=statistics.median(r['steps']/130/r['reference_seconds'] for r in successes) if successes else None,
            timely_2x=sum(bool(r['success'] and r['steps']<=int(130*max(2*r['reference_seconds'],r['reference_seconds']+3))) for r in episodes),
            episodes_with_miss=sum(r['missed_crossings']>0 for r in episodes),
            crash_at_timely_horizon=sum(bool(r['crashed'] and r['steps']<=r['deadline_steps']) for r in episodes),
            crash_at_6000=sum(r['crashed'] for r in episodes)))
    intervals={}
    for first,second in [('v621','update_fix'),('pre_fix65','update_fix'),('update_fix','plant'),('plant','bounded'),('plant','bounded_scale')]:
        if first not in reports or second not in reports:
            continue
        a={(r['slot'],r['seed']):r for r in reports[first]['episodes']}
        b={(r['slot'],r['seed']):r for r in reports[second]['episodes']}
        assert a.keys()==b.keys()
        rng=np.random.default_rng(20260928)
        idx=rng.integers(0,len(a),(10000,len(a)))
        result={}
        for key in ['success','timely_success','clean_timely','retry_success']:
            dif=np.array([b[k][key]-a[k][key] for k in sorted(a)])
            result[key]=dict(difference_pp=100*float(dif.mean()),
                course_bootstrap_95_pp=(100*np.quantile(dif[idx].mean(axis=1),[.025,.975])).tolist())
        common=[k for k in sorted(a) if a[k]['success'] and b[k]['success']]
        result['joint_successes']=len(common)
        result['median_paired_success_time_ratio']=statistics.median(b[k]['steps']/a[k]['steps'] for k in common) if common else None
        def retry_ratios(panel):
            success=np.array([panel[k]['success'] for k in sorted(a)])
            retry=np.array([panel[k]['retry_success'] for k in sorted(a)])
            denominator=success[idx].sum(axis=1)
            return np.divide(retry[idx].sum(axis=1),denominator,out=np.full(len(idx),np.nan),where=denominator>0)
        difference=retry_ratios(b)-retry_ratios(a)
        result['retry_fraction_difference_course_bootstrap_95_pp']=(100*np.nanquantile(difference,[.025,.975])).tolist()
        intervals[f'{second}_minus_{first}']=result
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'scripts/audits/eval_privileged_real100_v2_dual.py',ROOT/'starscream/dagger_quality.py',
         ROOT/'starscream/racing_evaluation.py',ROOT/'configs/eval/real100_v2_timed_protocol_v1.json']}
    result=dict(scope='100 courses x one matched seed per checkpoint; descriptive development screen, not replicated training evidence',
        rows=rows,paired_intervals=intervals,source_sha256=provenance)
    (OUT/'mode-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if rows:
        with (OUT/'mode-summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
