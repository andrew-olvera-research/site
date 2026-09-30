"""Join v6.22 geometry closeness and frozen actor results."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/course-pools/v622-benchmarks'
EVAL = ROOT/'outputs/evals/v622-frozen'


def rank(values):
    order = np.argsort(values); result=np.empty(len(values),float); result[order]=np.arange(len(values)); return result


def corr(a,b):
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def main():
    close=json.loads((OUT/'split-closeness.json').read_text())
    split_data={}
    for split,stem in [('real60','real60'),('real100-hard','real100_hard')]:
        manifest=json.loads((ROOT/'configs/eval'/f'v6_22_{stem}.manifest.json').read_text())
        split_data[split]={r['name']:r for r in manifest['records'] if r['suite']==split}
    evals={}
    for path in EVAL.glob('*-8ep.json'):
        d=json.loads(path.read_text())
        if 'metrics' not in d: continue
        model='rl74' if path.name.startswith('rl74') else 'base'; split='real60' if 'real60' in path.name else 'real100-hard'
        m=d['metrics']; rows={}
        for name in split_data[split]:
            prefix=f'track/{name}/'; rows[name]=dict(success=m.get(prefix+'full_course_success',float('nan')),
                crash=m.get(prefix+'crash_rate',float('nan')),gates=m.get(prefix+'mean_gates',float('nan')),
                gate_speed=m.get(prefix+'episode_mean_gate_speed_mps_median',float('nan')))
        evals[(model,split)]=dict(aggregate={k:m[k] for k in ('full_course_success','crash_rate','mean_gates','mean_gate_speed_mps','mean_speed_mps','maximum_speed_mps')},rows=rows)
    summary={}
    for split in ('real60','real100-hard'):
        summary[split]={}
        for model in ('base','rl74'):
            data=evals[(model,split)]; generated=[r for n,r in data['rows'].items()]
            summary[split][model]=dict(aggregate=data['aggregate'],generated=dict(
                courses=len(generated),completion=float(np.nanmean([r['success'] for r in generated])),
                crash_rate=float(np.nanmean([r['crash'] for r in generated])),
                mean_gates=float(np.nanmean([r['gates'] for r in generated])),
                gate_speed=float(np.nanmean([r['gate_speed'] for r in generated]))))
    real_cells=set().union(*(set(r['cells']) for r in split_data['real60'].values()))
    hard_cells=set().union(*(set(r['cells']) for r in split_data['real100-hard'].values()))
    novel=hard_cells-real_cells
    hard_trans=close['real100_hard']['gates']; real_trans=close['real60']['gates']
    feature_delta={}
    for feature in close['real60']['special']:
        rv=close['real60']['special'][feature]/max(real_trans,1); hv=close['real100_hard']['special'][feature]/max(hard_trans,1)
        feature_delta[feature]=dict(real60_per_gate=rv,hard_per_gate=hv,ratio=hv/max(rv,1e-9))
    records=close['nearest_hard_to_real60']['records']
    distance=np.array([r['distance'] for r in records]); jaccard=np.array([r['cell_jaccard'] for r in records]);
    closeness={}
    for model in ('base','rl74'):
        success=np.array([evals[(model,'real100-hard')]['rows'][r['name']]['success'] for r in records])
        closeness[model]=dict(distance_success_pearson=corr(distance,success),distance_success_spearman=corr(rank(distance),rank(success)),
            cell_jaccard_success_pearson=corr(jaccard,success),success_close_quartile=float(np.mean(success[distance<=np.quantile(distance,.25)])),
            success_far_quartile=float(np.mean(success[distance>=np.quantile(distance,.75)])))
    family={}
    for model in ('base','rl74'):
        family[model]={}
        for split in ('real60','real100-hard'):
            groups=defaultdict(list)
            for name,row in evals[(model,split)]['rows'].items(): groups[split_data[split][name]['family']].append(row['success'])
            family[model][split]={f:float(np.mean(v)) for f,v in sorted(groups.items())}
    result=dict(schema='starscream-v622-transfer-explanation-v1',scope='Frozen geometry and 8-episode matched randomized evaluation; public references excluded from generated comparisons.',
        geometry_closeness=dict(nearest_distance=close['nearest_hard_to_real60']['distance'],nearest_cell_jaccard=close['nearest_hard_to_real60']['cell_jaccard'],
            same_family_fraction=close['nearest_hard_to_real60']['same_family_fraction'],hard_cells=len(hard_cells),real60_cells=len(real_cells),
            hard_cells_novel_to_real60=len(novel),hard_cells_novel_fraction=len(novel)/max(len(hard_cells),1)),
        geometry_difficulty=dict(real60_gates=real_trans,hard_gates=hard_trans,feature_per_gate=feature_delta,
            marginal_quantile_comparison={key:{'real60':close['real60']['transitions'][key],'real100-hard':close['real100_hard']['transitions'][key]}
                                          for key in ('incoming_m','turn_deg','height_change_m','gate_center_height_m','width_m')}),
        policy_results=summary,hard_closeness_transfer=closeness,family_completion=family,
        interpretation=[
            'Hard is geometrically independent but not a separate behavioral regime: nearest whole-course distance median is moderate and cell overlap is low-to-moderate.',
            'The hard suite is harder in aggregate, especially for base, but its marginal turn/height distributions are close to real60; most extra difficulty is course count, narrowness, longer chains and selected extremes.',
            'Persistent RL uplift on hard generated courses supports genuine policy generalization; it does not prove the suite is maximally challenging.',
            'Cell novelty is not executed-behavior novelty: factorized geometric cells and teacher feasibility do not certify braking, approach-speed or recovery demands.' ])
    path=OUT/'transfer-explanation.json'; path.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); print('WROTE',path)


if __name__=='__main__': main()
