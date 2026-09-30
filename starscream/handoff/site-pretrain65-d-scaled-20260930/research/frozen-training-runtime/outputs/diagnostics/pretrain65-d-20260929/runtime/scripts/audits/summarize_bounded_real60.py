"""Compare bounded r287 to historical real60 under the eight-repeat protocol."""
import json
from pathlib import Path
import numpy as np
import math
import statistics
import yaml

ROOT=Path(__file__).resolve().parents[2]
PATHS={
    'v621':ROOT/'outputs/evals/v6222-crosscompare/v621-v6_22_real60-e8.json',
    'update_fix':ROOT/'outputs/evals/v6211-update-fix-final/update-fix-real60-e8.json',
    'plant':ROOT/'outputs/evals/bounded-r287-real60/plant-real60-historical-start-e8.json',
    'bounded_r287':ROOT/'outputs/evals/bounded-r287-real60/r287-real60-historical-start-e8.json',
}
reports={k:json.loads(v.read_text()) for k,v in PATHS.items()}
base=reports['bounded_r287']
suite=yaml.safe_load((ROOT/'configs/eval/v6_22_real60.yaml').read_text())
families={row['name']:row['family'] for row in suite['active']}
expected={k:base[k] for k in ('seed','episodes','matched_track_seeds','max_steps','nominal_environment')}
assert expected==dict(seed=2034091462,episodes=8,matched_track_seeds=True,max_steps=6000,nominal_environment=False)
assert base['track_suite']=='configs/eval/v6_22_real60.yaml'
assert len(base['tracks'])==60
for name,r in reports.items():
    assert {k:r[k] for k in expected}==expected,(name,'contract mismatch')
    assert r['tracks']==base['tracks'],(name,'track mismatch')

def course_success(report):
    return np.asarray([report['metrics'][f'track/{name}/full_course_success'] for name in base['tracks']],float)

rng=np.random.default_rng(20260928)
idx=rng.integers(0,60,size=(20000,60))
summary={}
for name,r in reports.items():
    scores=course_success(r)
    summary[name]=dict(source=str(PATHS[name].relative_to(ROOT)),
        checkpoint=r['checkpoint'],full_course_success=r['metrics']['full_course_success'],
        successful_laps=int(round(480*r['metrics']['full_course_success'])),
        courses_with_success=int(np.sum(scores>0)),
        mean_gates=r['metrics']['mean_gates'],crash_rate=r['metrics']['crash_rate'],
        mean_gate_speed_mps=r['metrics'].get('mean_gate_speed_mps'),
        family_success={family:float(np.mean([scores[i] for i,course in enumerate(base['tracks'])
                                             if families[course]==family])) for family in sorted(set(families.values()))},
        median_of_successful_course_medians_seconds=statistics.median(
            float(r['metrics'][f'track/{course}/successful_median_steps'])/130 for course in base['tracks']
            if math.isfinite(float(r['metrics'][f'track/{course}/successful_median_steps']))))

contrasts={}
for name in ('v621','update_fix','plant'):
    difference=course_success(base)-course_success(reports[name])
    contrasts[f'bounded_r287_minus_{name}']=dict(difference_pp=float(difference.mean()*100),
        course_bootstrap_95_pp=(np.quantile(difference[idx].mean(axis=1),[.025,.975])*100).tolist(),
        courses_improved=int(np.sum(difference>0)),courses_declined=int(np.sum(difference<0)),
        courses_tied=int(np.sum(difference==0)))
    ratios=[]
    for course in base['tracks']:
        k=f'track/{course}/successful_median_steps'
        b=float(base['metrics'][k]);a=float(reports[name]['metrics'][k])
        if math.isfinite(a) and math.isfinite(b):ratios.append(b/a)
    contrasts[f'bounded_r287_minus_{name}'].update(joint_successful_courses=len(ratios),
        median_course_success_time_ratio=statistics.median(ratios) if ratios else None)

result=dict(contract=expected,rows=summary,contrasts=contrasts,
    caveat='Per-course bootstrap describes this frozen 60-course set. The reports share seed and reset-index convention, but separately configured model observations and training budgets differ. Successful-time ratios condition on both actors succeeding on the course and are not paired episode times.')
out=ROOT/'outputs/evals/bounded-r287-real60/comparison.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
