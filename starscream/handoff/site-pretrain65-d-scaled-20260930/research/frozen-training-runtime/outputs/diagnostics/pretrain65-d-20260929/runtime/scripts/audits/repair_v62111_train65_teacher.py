"""Find completion-qualified commands when the old teacher itself fails DART.

Only unresolved, freshly unqualified courses enter this pass. A slower qualified
teacher is preferable to retaining an unqualified baseline. Numerical cohorts
and acceptance thresholds are identical to the pace search.
"""
import argparse
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.audits import tune_v62111_train65_teacher as tune
from starscream.course_model.training import atomic_json

PRIMARY=tune.ROOT/'outputs/diagnostics/v62111-train65-teacher-v4'
OUTPUT=PRIMARY/'reliability-repair'


def repair(payload):
    settings,row,prior,contract=payload
    import torch
    torch.set_num_threads(1)
    root=OUTPUT/row['name'];root.mkdir(parents=True,exist_ok=True)
    final=root/'selection.json'
    if final.exists():
        result=json.loads(final.read_text());assert result['contract']==contract
        return result
    started=time.perf_counter()
    base=tune.baseline_candidate(settings,row)
    for kind in ('screen','nominal','randomized','dart'):
        source=PRIMARY/row['name']/'baseline'/f'{kind}.json'
        destination=root/'baseline'/f'{kind}.json'
        if not destination.exists():
            raw=json.loads(source.read_text())
            raw.update(reused_from=dict(path=str(source),contract=raw['contract']),contract=contract)
            destination.parent.mkdir(exist_ok=True)
            atomic_json(destination,raw)
    screens=[]
    for fraction in (1.,.95,.9,.85,.8,.7,.6):
        for mode in ('original','modest'):
            if fraction==1. and mode=='original':continue
            c=deepcopy(base);speed=round(base['speed']*fraction,3)
            c.update(name=f'repair-{mode}-{speed:g}',speed=speed)
            if mode=='modest':
                for key in ('maximum_acceleration','maximum_longitudinal_acceleration','maximum_braking_acceleration'):
                    c['controller'][key]=min(float(c['controller'][key])*1.1,40.)
                for key in ('collective_slew_limit','body_rate_slew_limit'):
                    c['controller'][key]=float(c['controller'][key])*1.1
            screens.append(tune.cohort(settings,row,c,'screen',root,contract))
    confirmations={'baseline':{kind:json.loads((root/'baseline'/f'{kind}.json').read_text())['summary']
        for kind in ('nominal','randomized','dart')}}
    eligible=[]
    for screen in sorted((r for r in screens if r['summary']['usable_rate']==1.),key=lambda r:r['summary']['median']):
        candidate=screen['candidate'];reports={}
        for kind in ('nominal','randomized','dart'):
            reports[kind]=tune.cohort(settings,row,candidate,kind,root,contract)
            if not tune.qualifies(reports):break
        confirmations[candidate['name']]={k:r['summary'] for k,r in reports.items()}
        if len(reports)==3 and tune.qualifies(reports):
            eligible.append((reports['randomized']['summary']['median'],candidate))
        if len(eligible)>=2:break
    if eligible:
        lap,selected=min(eligible,key=lambda item:item[0])
    else:
        selected=base;lap=confirmations['baseline']['randomized']['median']
    result=dict(contract=contract,name=row['name'],selected=selected,baseline=base,
        baseline_qualified=False,selected_qualified=bool(eligible),
        baseline_median=confirmations['baseline']['randomized']['median'],selected_median=lap,
        accepted=[c['name'] for _,c in eligible],confirmations=confirmations,
        screens=[dict(candidate=r['candidate'],summary=r['summary']) for r in screens],
        selection_reason='Fresh baseline qualification failed; choose shortest tested usable teacher, including slower commands',
        seconds=time.perf_counter()-started)
    atomic_json(final,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--completed-only',action='store_true',help='Repair already final course results while other courses are tuning')
    args=parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    lock=(OUTPUT/'.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    primary=json.loads((PRIMARY/'summary.json').read_text())
    refined=json.loads((PRIMARY/'local-refinement/summary.json').read_text())
    all_complete=bool(primary['complete'] and refined['complete'])
    if not all_complete and not args.completed_only:
        raise ValueError('Finish primary and focused refinement first')
    prior={r['name']:r for r in primary['records']}
    prior.update({r['name']:r for r in refined['records'] if r['selected_qualified']})
    targets=[r for r in prior.values() if not r['selected_qualified']]
    if not all_complete:
        finished_refinement={r['name'] for r in refined['records']}
        targets=[r for r in targets if r['name'] in finished_refinement]
    settings=json.loads(tune.CONFIG.read_text())['dagger']
    rows={r['name']:r for r in json.loads(Path(settings['track_manifest']).read_text())['records']}
    sources={**json.loads((PRIMARY/'local-refinement/contract.json').read_text())['sources'],
        str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for path,digest in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed source {path}')
    contract=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT/'contract.json').exists():
        assert json.loads((OUTPUT/'contract.json').read_text())['contract']==contract
    atomic_json(OUTPUT/'contract.json',dict(contract=contract,sources=sources,primary_contract=primary['contract'],
        refinement_contract=refined['contract'],targets=sorted(r['name'] for r in targets)))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(repair,(settings,rows[r['name']],r,contract)) for r in targets]
        for future in as_completed(futures):
            r=future.result();results.append(r)
            atomic_json(OUTPUT/'summary.json',dict(contract=contract,complete=all_complete and len(results)==len(targets),
                records=sorted(results,key=lambda r:r['name'])))
            print(json.dumps(dict(done=len(results),total=len(targets),name=r['name'],qualified=r['selected_qualified'],
                before=r['baseline_median'],after=r['selected_median'])),flush=True)
    if not targets:atomic_json(OUTPUT/'summary.json',dict(contract=contract,complete=all_complete,records=[]))


if __name__=='__main__':main()
