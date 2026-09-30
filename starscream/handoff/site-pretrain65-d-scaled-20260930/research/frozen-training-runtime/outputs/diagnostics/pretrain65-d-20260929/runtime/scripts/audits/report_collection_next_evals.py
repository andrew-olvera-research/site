"""Summarize paired evaluation panels without treating repeated courses as IID."""
import json
from pathlib import Path
import numpy as np

OUT=Path('/workspace/outputs/diagnostics/collection-next-20260929')

def main():
    summary={}
    for arm in ('a','d','bounded'):
        path=OUT/f'eval-{arm}-related.json'
        if not path.exists(): continue
        rows=json.loads(path.read_text())['episodes']
        def metrics(rs):
            success=np.asarray([r['gates']>=r['target_gates'] for r in rs])
            timely=np.asarray([r['within_clean_deadline'] for r in rs])
            clean=np.asarray([r['reference_misses']==0 for r in rs])
            return dict(episodes=len(rs),success=int(success.sum()),timely=int((success&timely).sum()),
                clean_timely=int((success&timely&clean).sum()),recovered=int((success&~clean).sum()),
                crashes=sum(r['crashed'] for r in rs),mean_misses=float(np.mean([r['reference_misses'] for r in rs])))
        summary[arm]=dict(related=metrics(rows),courses={t:metrics([r for r in rows if r['track']==t]) for t in sorted({r['track'] for r in rows})})
        real=OUT/f'eval-{arm}-real100.json'
        if real.exists():
            data=json.loads(real.read_text()); summary[arm]['real100']=data['aggregate']
            durations=[]; all_steps=0; recovery_steps=0
            for r in data['episodes']:
                start=None; all_steps+=r['steps']
                for e in r['events']:
                    if e['kind']=='miss' and start is None: start=e['step']
                    if e['kind']=='pass' and start is not None:
                        durations.append((e['step']-start)/130); recovery_steps+=e['step']-start; start=None
                if start is not None: recovery_steps+=r['steps']-start
            summary[arm]['real100']['observed_post_miss_time_fraction']=recovery_steps/max(all_steps,1)
            summary[arm]['real100']['successful_gate_recovery_seconds_p90']=float(np.quantile(durations,.9)) if durations else None
    # Paired comparison by episode identity within each course.
    if 'a' in summary and 'd' in summary:
        sets={arm:json.loads((OUT/f'eval-{arm}-related.json').read_text())['episodes'] for arm in ('a','d')}
        maps={arm:{(r['track'],r['episode_seed']):r for r in rows} for arm,rows in sets.items()}
        assert maps['a'].keys()==maps['d'].keys()
        comparison={}
        for name in ('success','timely','clean_timely'):
            def outcome(r):
                ok=r['gates']>=r['target_gates']
                if name!='success': ok=ok and r['within_clean_deadline']
                if name=='clean_timely': ok=ok and r['reference_misses']==0
                return bool(ok)
            comparison[name]=dict(a_only=sum(outcome(a) and not outcome(maps['d'][k]) for k,a in maps['a'].items()),
                                  d_only=sum(outcome(maps['d'][k]) and not outcome(a) for k,a in maps['a'].items()))
        summary['paired_a_d']=comparison
    (OUT/'eval-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
