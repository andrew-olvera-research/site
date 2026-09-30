#!/usr/bin/env python3
"""Freeze eight established showcase courses plus ten diverse real100-v2 courses."""
import json
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from render_research_gifs import suite_records

KNOWN=[
 'v622_hard_v2b_vertical_chain_008_06',
 'v622_hard_v2b_compound_reversal_021_03',
 'v622_hard_diving_hairpin_039_011_arena1',
 'v622_hard_slalom_028_029',
 'v622_hard_long_braking_023_005_arena1_mirror1',
 'v622_hard_v2b_radius_switch_029_21',
 'v622_hard_ordered_3d_042_003_mirror1',
 'v622_hard_v2b_wrong_side_incidence_015_15',
]


def build():
    suite=ROOT/'configs/eval/v6_22_real100_hard_v2.yaml'
    evaluation=ROOT/'outputs/evals/v6211-update-fix-final/update-fix-real100-hard-v2-e8.json'
    records=suite_records(suite); metric=json.loads(evaluation.read_text())['metrics']
    by_name={r['name']:r for r in records}
    missing=[n for n in KNOWN if n not in by_name]
    if missing: raise ValueError(f'known showcase tracks missing from real100-v2: {missing}')
    chosen=[by_name[n] for n in KNOWN]
    families={r.get('family','') for r in chosen}
    candidates=[]
    for record in records:
        name=record['name']
        success=float(metric.get(f'track/{name}/full_course_success',0.))
        fastest=float(metric.get(f'track/{name}/successful_minimum_steps',float('inf')))
        mean_gates=float(metric.get(f'track/{name}/mean_gates',0.))
        if record in chosen or success<.125 or not (fastest>0 and fastest<float('inf')): continue
        candidates.append((record,success,fastest,mean_gates))
    # Greedy geometry novelty with distinct families; prefer enough benchmark
    # completions and a known finite pace so all additions can yield real laps.
    geometry={}
    from starscream.env.tracks import load_track
    import numpy as np
    for record in chosen+[r[0] for r in candidates]:
        track=load_track(record['path']); xyz=np.stack([g.position for g in track.gates])
        delta=np.roll(xyz,-1,0)-xyz; length=np.linalg.norm(delta,axis=1)
        direction=delta/np.maximum(length[:,None],1e-6)
        turn=np.arccos(np.clip(np.sum(direction*np.roll(direction,1,axis=0),axis=1),-1,1))
        geometry[record['name']]=np.asarray([len(xyz),length.sum(),np.ptp(xyz[:,2]),turn.mean(),turn.max(),length.std()])
    values=np.stack(list(geometry.values())); z=(values-values.mean(0))/np.maximum(values.std(0),1e-6)
    zby={name:z[i] for i,name in enumerate(geometry)}
    additions=[]; used=set(families)
    while len(additions)<10:
        eligible=[row for row in candidates if row[0] not in chosen+additions]
        novel=[r for r in eligible if r[0].get('family','') not in used]
        if novel: eligible=novel
        def score(row):
            record,success,steps,gates=row
            novelty=min(float(np.linalg.norm(zby[record['name']]-zby[r['name']])) for r in chosen+additions) if chosen+additions else 0
            # Favor courses that both show competence and offer a quick readable lap.
            return novelty+.55*success-.18*steps/2000 + .08*min(gates/10,1)
        selected=max(eligible,key=score)
        additions.append(selected[0]); used.add(selected[0].get('family',''))
    selected=chosen+additions
    # Two difficult but viable courses are dedicated recovery searches.
    extras={r['name']:row[1:] for row in candidates for r in [row[0]]}
    recovery_pool=[r for r in additions if .125<=extras[r['name']][0]<=.75]
    recovery_pool.sort(key=lambda r:(extras[r['name']][0],-extras[r['name']][2],r['name']))
    if len(recovery_pool)<2: raise RuntimeError('could not find two viable hard recovery courses')
    recovery=[r['name'] for r in recovery_pool[:2]]
    active=[]
    for index,r in enumerate(selected):
        item={k:r[k] for k in ('name','family','exposure') if k in r}
        item['track']=str(Path(r['path']).resolve().relative_to(ROOT))
        item['role']='recovery_search' if r['name'] in recovery else ('known_showcase' if r['name'] in KNOWN else 'diverse_addition')
        item['benchmark_success_rate']=float(metric.get(f"track/{r['name']}/full_course_success",0.))
        item['benchmark_fastest_steps']=float(metric.get(f"track/{r['name']}/successful_minimum_steps",float('nan')))
        active.append(item)
    return dict(schema='starscream-site-trajectory-suite-v1',description='Eight established showcases plus ten diverse real100-v2 additions; two additional recovery searches.',source_manifest='configs/eval/v6_22_real100_hard_v2.manifest.json',benchmark='outputs/evals/v6211-update-fix-final/update-fix-real100-hard-v2-e8.json',episodes_per_course=32,seed=2034091462,control_hz=130,active=active,recovery_courses=recovery)


if __name__=='__main__':
    print(yaml.safe_dump(build(),sort_keys=False))
