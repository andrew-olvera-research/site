"""Test already qualified alternate teachers on held-out recovery stress seeds."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.audits.audit_dagger_teacher_labels import audit_track
from starscream.course_model.training import atomic_json

ROOT=Path('outputs/diagnostics/v62111-train65-teacher-v4/frozen')
OUTPUT=ROOT/'recovery-alternatives'


def qualifies(cohorts):
    return len(cohorts)==3 and all(c['usable']==c['episodes'] for c in cohorts.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completed-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    stress=json.loads((ROOT/'recovery-stress/report.json').read_text())
    if not stress['complete'] and not args.completed_only:raise ValueError('Wait for all recovery stress probes')
    search=json.loads((ROOT/'search-summary.json').read_text())
    if stress['protocol']['search_contract']!=search['contract']:raise ValueError('Stress belongs to another teacher frontier')
    settings=json.loads(Path('configs/exp/v6.21.1.1/plant_dagger.yaml').read_text())['dagger']
    records={r['name']:r for r in json.loads(Path(settings['track_manifest']).read_text())['records']}
    by_name={r['name']:r for r in search['records']}
    targets=[r for r in stress['courses'] if r['regression_seeds']]
    protocol=dict(search_contract=search['contract'],stress_contract=stress['contract'],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        acceptance='Qualified 2/2 nominal, 6/6 randomized, 2/2 DART and no paired regression at 2x action noise')
    contract=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    OUTPUT.mkdir(exist_ok=True)
    cache=[];rows=[]
    for target in targets:
        result=by_name[target['name']]
        old=json.loads((ROOT/'recovery-stress'/f'{target["name"]}-baseline.json').read_text())['episodes']
        candidates={result['baseline']['name']:result['baseline']}
        candidates.update({s['candidate']['name']:s['candidate'] for s in result['screens']})
        ordered=sorted((name for name,cohorts in result['confirmations'].items()
            if name!=result['selected']['name'] and qualifies(cohorts)),
            key=lambda name:result['confirmations'][name]['randomized']['median'])
        chosen=None;trials=[]
        for name in ordered:
            candidate=candidates[name]
            if name=='baseline':
                episodes=old
            else:
                path=OUTPUT/f'{target["name"]}-{name}.json'
                if path.exists():
                    data=json.loads(path.read_text());assert data['contract']==contract and data['candidate']==candidate
                else:
                    s=deepcopy(settings)
                    s.update(mpcc_use_manifest_teacher_profile=False,mpcc_use_manifest_planner_profile=False,
                        mpcc_use_manifest_speed=False,mpcc_speed_frontier_profile='',mpcc_nominal_speed=16.5,
                        mpcc_config=candidate['controller'],mpcc_planner_config=candidate['planner'],
                        mpcc_family_configs={},mpcc_family_planner_configs={},mpcc_family_nominal_speeds={},
                        mpcc_build_root='/tmp/starscream-train65-recovery-alternatives')
                    report=audit_track(s,s['curriculum'],records[target['name']]['path'],0,1,4,621113900,
                        start_mode='canonical',repeats_per_start=4,start_perturbation_scale=1.,
                        dart_action_noise_scale=2.,dart_episode_fraction=1.,speed_fractions=(1.,),
                        frontier_speed=candidate['speed'],backend_cache=cache)
                    data=dict(contract=contract,candidate=candidate,episodes=report['episodes'])
                    atomic_json(path,data)
                episodes=data['episodes']
            bad=[after['seed'] for before,after in zip(old,episodes)
                if before['seed']!=after['seed'] or (before['success'] and
                    (not after['success'] or after['solver_failure_fraction']>max(.06,before['solver_failure_fraction'])))]
            trials.append(dict(name=name,successes=sum(x['success'] for x in episodes),regression_seeds=bad,
                randomized_median=result['confirmations'][name]['randomized']['median']))
            if not bad:
                chosen=name;break
        if chosen is None:raise ValueError(f'No recovery-safe qualified teacher for {target["name"]}')
        row=dict(name=target['name'],rejected=result['selected']['name'],selected=chosen,
            selected_randomized_median=result['confirmations'][chosen]['randomized']['median'],trials=trials)
        rows.append(row)
        atomic_json(OUTPUT/'report.json',dict(contract=contract,protocol=protocol,
            complete=stress['complete'] and len(rows)==len(targets),courses=rows))
        print(json.dumps(row),flush=True)
    if not targets:atomic_json(OUTPUT/'report.json',dict(contract=contract,protocol=protocol,complete=stress['complete'],courses=[]))


if __name__=='__main__':main()
