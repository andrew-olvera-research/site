"""Publish completed train65 pace evidence as additive manifests and profiles."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import statistics
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from scripts.audits.tune_v62111_train65_teacher import summary, usable


def publish(config,source,search,output):
    if not search.get('complete') or len(search['records'])!=65:
        raise ValueError('All 65 course searches must finish before publication')
    results={r['name']:r for r in search['records']}
    active=set(config['dagger']['curriculum']['tracks'])
    if set(results)!={r['name'] for r in source['records'] if r['path'] in active}:
        raise ValueError('Search and training course sets differ')
    manifest=deepcopy(source)
    controllers,planners={},{}
    rows=[]
    for record in manifest['records']:
        if record['path'] not in active:
            continue
        result=results[record['name']]
        selected=result['selected']
        if selected['name']!='baseline' and not result['selected_qualified']:
            raise ValueError(f'Unqualified teacher replacement: {record["name"]}')
        profile='v62111-pace-'+record['fingerprint'][:12]
        if profile in controllers and (controllers[profile]!=selected['controller'] or planners[profile]!=selected['planner']):
            raise ValueError(f'Teacher profile identifier collision: {profile}')
        controllers[profile]=selected['controller']
        planners[profile]=selected['planner']
        old_speed=record['qualified_speed_mps']
        record['qualified_speed_mps']=selected['speed']
        previous_qualification=record['qualification']
        record['source_teacher_qualification']=previous_qualification
        record['qualification']=dict(
            qualified=(True if selected['name']!='baseline' else previous_qualification.get('qualified',True)),
            selected={'teacher_profile':profile},
            teacher_overrides=dict(mpcc_nominal_speed=selected['speed'],mpcc_speed_frontier_profile='',
                mpcc_config=selected['controller'],mpcc_planner_config=selected['planner']),
            contract_sha256=result.get('contract',search['contract']),
            refreshed_qualified=result['selected_qualified'],
            pace_cohorts=result['confirmations'][selected['name']],
            evidence_scope='Full-course nominal/randomized/DART pace; original admission preserved in source_teacher_qualification')
        evidence_root=Path(result.get('evidence_root',output.parent))
        record['teacher_pace_evidence']=dict(contract=result.get('contract',search['contract']),
            selection=str((evidence_root/record['name']/'selection.json').resolve()),
            refreshed_qualification=result['selected_qualified'],retained_baseline=selected['name']=='baseline')
        before,after=result['baseline_median'],result['selected_median']
        trial_root=evidence_root/record['name']
        paired=[]
        old_solver_p95=new_solver_p95=None
        baseline_path=trial_root/'baseline'/'randomized.json'
        selected_path=trial_root/selected['name']/'randomized.json'
        if baseline_path.exists() and selected_path.exists():
            old_episodes=json.loads(baseline_path.read_text())['episodes']
            new_episodes=json.loads(selected_path.read_text())['episodes']
            old_solver_p95=statistics.median(r['solve_time_p95_ms'] for r in old_episodes)
            new_solver_p95=statistics.median(r['solve_time_p95_ms'] for r in new_episodes)
            old_rows={r['seed']:r for r in old_episodes if usable(r)}
            new_rows={r['seed']:r for r in new_episodes if usable(r)}
            paired=[100*(1-new_rows[seed]['lap_time_seconds']/old_rows[seed]['lap_time_seconds'])
                    for seed in sorted(set(old_rows)&set(new_rows))]
        rows.append(dict(name=record['name'],profile=selected['name'],
            selection_reason=result.get('selection_reason','Retained baseline or completion-qualified pace improvement'),
            old_command_mps=old_speed,new_command_mps=selected['speed'],
            baseline_randomized_median_s=before,selected_randomized_median_s=after,
            reduction_percent=100*(1-after/before) if before and after else None,
            baseline_usable=result['confirmations']['baseline']['randomized']['usable'],
            selected_usable=result['confirmations'][selected['name']]['randomized']['usable'],
            paired_successful_seeds=len(paired),
            paired_mean_time_reduction_percent=sum(paired)/len(paired) if paired else None,
            baseline_solver_p95_median_ms=old_solver_p95,selected_solver_p95_median_ms=new_solver_p95,
            selected_qualified=result['selected_qualified']))
    manifest['teacher_pace_frontier']=dict(contract=search['contract'],
        objective='shortest tested completion-qualified randomized median lap',
        search_ceiling_mps=36.,heldout_validation_untouched=True)
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'teacher-profiles.json',dict(contract=search['contract'],
        controller_profiles=controllers,planner_profiles=planners))
    with (output/'course-comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    gains=[r['reduction_percent'] for r in rows if r['reduction_percent'] is not None]
    report=dict(contract=search['contract'],courses=len(rows),
        improved=sum(r['reduction_percent'] is not None and r['reduction_percent']>=1. for r in rows),
        changed=sum(r['profile']!='baseline' for r in rows),
        refreshed_qualified=sum(r['selected_qualified'] for r in rows),
        equal_course_mean_reduction_percent=sum(gains)/len(gains) if gains else None,
        courses_comparison=rows)
    atomic_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search',type=Path,required=True)
    p.add_argument('--refinement',type=Path)
    p.add_argument('--repair',type=Path,help='Completion-qualified fallback commands for stale baseline qualifications')
    p.add_argument('--rescue',type=Path,help='Extended qualification for unresolved repairs')
    args=p.parse_args()
    search=json.loads((args.search/'summary.json').read_text())
    if not search.get('complete') or len(search['records'])!=65:
        raise ValueError('All 65 searches must complete before freezing')
    contract=json.loads((args.search/'contract.json').read_text())
    assert search['contract']==contract['contract']
    for path,digest in contract['sources'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Search source changed before freeze: {path}')
    for result in search['records']:
        result['evidence_root']=str(args.search.resolve())
    if args.refinement:
        refinement=json.loads((args.refinement/'summary.json').read_text())
        refinement_contract=json.loads((args.refinement/'contract.json').read_text())
        if refinement['contract']!=refinement_contract['contract']:
            raise ValueError('Refinement summary and source contract differ')
        if not refinement['complete'] or refinement['primary_contract']!=search['contract']:
            raise ValueError('Refinement must finish against this primary search')
        expected={r['name'] for r in search['records'] if r['selected']['name']=='baseline'}
        actual=[r['name'] for r in refinement['records']]
        if len(actual)!=len(set(actual)) or set(actual)!=expected:
            raise ValueError('Refinement must cover exactly the retained primary teachers')
        for path,digest in refinement_contract['sources'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Refinement source changed: {path}')
        by_name={r['name']:r for r in search['records']}
        for result in refinement['records']:
            previous=by_name[result['name']]
            if result['selected']['name']!='baseline' and result['selected_qualified']:
                if previous['selected_median'] and result['selected_median']>=.99*previous['selected_median']:
                    raise ValueError('Refinement replacement is not faster')
                result['evidence_root']=str(args.refinement.resolve())
                by_name[result['name']]=result
        search['records']=[by_name[r['name']] for r in search['records']]
        search['component_contracts']=[search['contract'],refinement['contract']]
        search['contract']=hashlib.sha256(json.dumps(search['component_contracts']).encode()).hexdigest()
    if args.repair:
        repair=json.loads((args.repair/'summary.json').read_text())
        repair_contract=json.loads((args.repair/'contract.json').read_text())
        if not repair['complete'] or repair['contract']!=repair_contract['contract']:
            raise ValueError('Reliability repair is incomplete or its contract differs')
        if (not args.refinement or repair_contract['primary_contract']!=contract['contract']
                or repair_contract['refinement_contract']!=refinement['contract']):
            raise ValueError('Reliability repair belongs to another search')
        for path,digest in repair_contract['sources'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Reliability repair source changed: {path}')
        expected={r['name'] for r in search['records'] if not r['selected_qualified']}
        actual=[r['name'] for r in repair['records']]
        if set(actual)!=expected or len(actual)!=len(set(actual)) or set(repair_contract['targets'])!=expected:
            raise ValueError('Reliability repair course set differs')
        for result in repair['records']:
            result['evidence_root']=str(args.repair.resolve())
        if args.rescue:
            rescue=json.loads((args.rescue/'summary.json').read_text())
            rescue_contract=json.loads((args.rescue/'contract.json').read_text())
            if (not rescue['complete'] or rescue['contract']!=rescue_contract['contract']
                    or rescue_contract['repair_contract']!=repair['contract']):
                raise ValueError('Extended reliability qualification contract differs or is incomplete')
            for path,digest in rescue_contract['sources'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                    raise ValueError(f'Extended reliability source changed: {path}')
            expected_rescue={r['name'] for r in repair['records'] if not r['selected_qualified']}
            actual_rescue=[r['name'] for r in rescue['records']]
            if (set(actual_rescue)!=expected_rescue or len(actual_rescue)!=len(set(actual_rescue))
                    or set(rescue_contract['targets'])!=expected_rescue):
                raise ValueError('Extended reliability course set differs')
            repaired={r['name']:r for r in repair['records']}
            for result in rescue['records']:
                result['evidence_root']=str(args.rescue.resolve())
                repaired[result['name']]=result
            repair['records']=[repaired[r['name']] for r in repair['records']]
        by_name={r['name']:r for r in search['records']}
        for result in repair['records']:
            if not result['selected_qualified']:
                raise ValueError(f'No qualified teacher found for {result["name"]}')
            by_name[result['name']]=result
        search['records']=[by_name[r['name']] for r in search['records']]
        search['component_contracts'].append(repair['contract'])
        if args.rescue:search['component_contracts'].append(rescue['contract'])
        search['contract']=hashlib.sha256(json.dumps(search['component_contracts']).encode()).hexdigest()
    config=json.loads(Path('configs/exp/v6.21.1.1/plant_dagger.yaml').read_text())
    source=json.loads(Path(config['dagger']['track_manifest']).read_text())
    for result in search['records']:
        for candidate in (result['baseline'],result['selected']):
            for cohort,count in (('nominal',2),('randomized',6),('dart',2)):
                raw=json.loads((Path(result['evidence_root'])/result['name']/candidate['name']/f'{cohort}.json').read_text())
                if (raw['contract']!=result['contract'] or raw['candidate']!=candidate
                        or len(raw['episodes'])!=count or summary(raw['episodes'])!=raw['summary']
                        or raw['summary']!=result['confirmations'][candidate['name']][cohort]):
                    raise ValueError(f'Inconsistent teacher evidence for {result["name"]}/{candidate["name"]}/{cohort}')
    result=publish(config,source,search,args.search/'frozen')
    atomic_json(args.search/'frozen'/'search-summary.json',search)
    print(json.dumps({k:v for k,v in result.items() if k!='courses_comparison'},indent=2))


if __name__=='__main__':main()
