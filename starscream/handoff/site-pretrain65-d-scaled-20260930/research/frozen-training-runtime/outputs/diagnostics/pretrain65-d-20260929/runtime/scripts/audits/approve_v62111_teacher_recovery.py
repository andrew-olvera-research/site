"""Publish the fastest qualified teachers without held-out recovery regression."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.audits.freeze_v62111_train65_teacher import publish
from starscream.course_model.training import atomic_json

ROOT=Path('outputs/diagnostics/v62111-train65-teacher-v4/frozen')
OUTPUT=ROOT/'approved'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original=json.loads((ROOT/'search-summary.json').read_text())
    stress_path=ROOT/'recovery-stress/report.json'
    stress=json.loads(stress_path.read_text())
    alternatives=json.loads((ROOT/'recovery-alternatives/report.json').read_text())
    if not stress['complete'] or not alternatives['complete']:
        raise ValueError('Recovery audits must complete before approval')
    if (stress['protocol']['search_contract']!=original['contract']
            or alternatives['protocol']['search_contract']!=original['contract']
            or alternatives['protocol']['stress_contract']!=stress['contract']):
        raise ValueError('Recovery evidence belongs to another teacher frontier')
    flagged={r['name'] for r in stress['courses'] if r['regression_seeds']}
    selected={r['name']:r for r in alternatives['courses']}
    if flagged!=set(selected):raise ValueError('Every recovery regression requires a qualified alternative')
    search=deepcopy(original)
    OUTPUT.mkdir(exist_ok=True)
    approval_contract=hashlib.sha256(json.dumps(dict(base=original['contract'],
        stress_sha256=digest(stress_path),alternatives_contract=alternatives['contract'],
        selected={name:row['selected'] for name,row in sorted(selected.items())}),
        sort_keys=True).encode()).hexdigest()
    search['component_contracts'] += [stress['contract'],alternatives['contract']]
    search['contract']=approval_contract
    for result in search['records']:
        if result['name'] not in selected:continue
        decision=selected[result['name']]
        candidate=({result['baseline']['name']:result['baseline']}
            | {s['candidate']['name']:s['candidate'] for s in result['screens']}).get(decision['selected'])
        if candidate is None:raise ValueError(f'Missing qualified candidate for {result["name"]}')
        cohorts=result['confirmations'].get(candidate['name'],{})
        if (len(cohorts)!=3 or any(v['usable']!=v['episodes'] for v in cohorts.values())
                or cohorts['randomized']['median']!=decision['selected_randomized_median']):
            raise ValueError(f'Alternative lacks original full qualification: {result["name"]}')
        if candidate['name']=='baseline' and not result['baseline_qualified']:
            raise ValueError(f'Unqualified baseline recovery fallback: {result["name"]}')
        source=Path(result['evidence_root'])/result['name']
        root=OUTPUT/'adjusted-evidence'/result['name']
        for name in ('baseline',candidate['name']):
            for kind in ('nominal','randomized','dart'):
                raw=json.loads((source/name/f'{kind}.json').read_text())
                if raw['contract']!=result['contract']:raise ValueError('Recovery source contract differs')
                raw['approved_copy_of']=str((source/name/f'{kind}.json').resolve())
                destination=root/name/f'{kind}.json'
                destination.parent.mkdir(parents=True,exist_ok=True)
                atomic_json(destination,raw)
        result['selected']=candidate
        result['selected_median']=decision['selected_randomized_median']
        result['selected_qualified']=True
        result['selection_reason']='Held-out 2x DART recovery regression; fastest fully qualified alternative without paired regression'
        result['recovery_stress_remediation']=dict(original=decision['rejected'],approved=candidate['name'],
            alternatives_contract=alternatives['contract'])
        result['evidence_root']=str((OUTPUT/'adjusted-evidence').resolve())
        atomic_json(root/'selection.json',result)
    config=json.loads(Path('configs/exp/v6.21.1.1/plant_dagger.yaml').read_text())
    source=json.loads(Path(config['dagger']['track_manifest']).read_text())
    report=publish(config,source,search,OUTPUT)
    report.update(recovery_stress=dict(raw_passed=stress['passed'],raw_report_sha256=digest(stress_path),
        alternatives_contract=alternatives['contract'],remediated=selected))
    atomic_json(OUTPUT/'report.json',report)
    atomic_json(OUTPUT/'search-summary.json',search)
    (OUTPUT/'recovery-stress').mkdir(exist_ok=True)
    atomic_json(OUTPUT/'recovery-stress/raw-report.json',stress)
    approved=deepcopy(stress)
    approved['raw_passed']=stress['passed']
    approved['passed']=True
    approved['approval_condition']='Each raw regression belongs to a rejected candidate; final selected alternatives passed the same paired stress seeds'
    for row in approved['courses']:
        row['status']='rejected' if row['name'] in selected else 'retained'
    approved['protocol']['source_stress_contract']=stress['protocol']['search_contract']
    approved['protocol']['search_contract']=approval_contract
    approved['raw_report_sha256']=digest(stress_path)
    approved['remediations']=[dict(name=name,original=selected[name]['rejected'],approved=selected[name]['selected']) for name in sorted(selected)]
    atomic_json(OUTPUT/'recovery-stress/report.json',approved)
    atomic_json(OUTPUT/'recovery-stress/alternative-report.json',alternatives)
    print(json.dumps({k:v for k,v in report.items() if k not in ('courses_comparison','recovery_stress')},indent=2))
    print('remediated='+json.dumps(approved['remediations']))


if __name__=='__main__':main()
