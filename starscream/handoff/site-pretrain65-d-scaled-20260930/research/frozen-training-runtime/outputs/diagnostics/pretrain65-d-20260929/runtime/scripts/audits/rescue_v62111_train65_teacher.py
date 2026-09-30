"""Extended line-clearance/low-speed search only for unresolved qualifications."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.audits import tune_v62111_train65_teacher as tune
from starscream.course_model.training import atomic_json

PRIMARY=tune.ROOT/'outputs/diagnostics/v62111-train65-teacher-v4'
OUTPUT=PRIMARY/'reliability-rescue'


def rescue(payload):
    settings,row,contract=payload
    import torch
    torch.set_num_threads(1)
    root=OUTPUT/row['name'];root.mkdir(parents=True,exist_ok=True)
    if (root/'selection.json').exists():
        r=json.loads((root/'selection.json').read_text());assert r['contract']==contract;return r
    base=tune.baseline_candidate(settings,row)
    for kind in ('screen','nominal','randomized','dart'):
        source=PRIMARY/row['name']/'baseline'/f'{kind}.json'
        destination=root/'baseline'/f'{kind}.json';destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            raw=json.loads(source.read_text())
            raw.update(reused_from=dict(path=str(source),contract=raw['contract']),contract=contract)
            atomic_json(destination,raw)
    candidates=[]
    for mode in ('original','centered','centered_modest'):
        speeds=(4.,6.,8.) if mode=='original' else (8.,12.,16.5,20.)
        for speed in speeds:
            c=deepcopy(base);c.update(name=f'rescue-{mode}-{speed:g}',speed=speed)
            if mode!='original':c['planner'].update(aperture_fraction=0.,aperture_margin=.25)
            if mode=='centered_modest':
                for key in ('maximum_acceleration','maximum_longitudinal_acceleration','maximum_braking_acceleration'):
                    c['controller'][key]=min(float(c['controller'][key])*1.1,40.)
                for key in ('collective_slew_limit','body_rate_slew_limit'):
                    c['controller'][key]=float(c['controller'][key])*1.1
            candidates.append(c)
    screens=[tune.cohort(settings,row,c,'screen',root,contract) for c in candidates]
    confirmations={'baseline':{kind:json.loads((root/'baseline'/f'{kind}.json').read_text())['summary']
        for kind in ('nominal','randomized','dart')}}
    eligible=[]
    for screen in sorted((r for r in screens if r['summary']['usable_rate']==1.),key=lambda r:r['summary']['median']):
        c=screen['candidate'];reports={}
        for kind in ('nominal','randomized','dart'):
            reports[kind]=tune.cohort(settings,row,c,kind,root,contract)
            if not tune.qualifies(reports):break
        confirmations[c['name']]={k:r['summary'] for k,r in reports.items()}
        if len(reports)==3 and tune.qualifies(reports):eligible.append((reports['randomized']['summary']['median'],c))
        if len(eligible)>=2:break
    lap,selected=min(eligible,key=lambda item:item[0]) if eligible else (confirmations['baseline']['randomized']['median'],base)
    result=dict(contract=contract,name=row['name'],selected=selected,baseline=base,
        baseline_qualified=False,selected_qualified=bool(eligible),accepted=[c['name'] for _,c in eligible],
        baseline_median=confirmations['baseline']['randomized']['median'],selected_median=lap,
        confirmations=confirmations,screens=[dict(candidate=r['candidate'],summary=r['summary']) for r in screens],
        selection_reason='Fresh baseline and local command repair failed; line-clearance and extended low-speed qualification')
    atomic_json(root/'selection.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--completed-only',action='store_true')
    args=parser.parse_args();OUTPUT.mkdir(exist_ok=True)
    lock=(OUTPUT/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    prior=json.loads((PRIMARY/'reliability-repair/summary.json').read_text())
    if not prior['complete'] and not args.completed_only:raise ValueError('Finish reliability repair first')
    targets=[r for r in prior['records'] if not r['selected_qualified']]
    parent=json.loads((PRIMARY/'reliability-repair/contract.json').read_text())
    sources={**parent['sources'],str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for path,digest in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed source {path}')
    contract=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    if (OUTPUT/'contract.json').exists():assert json.loads((OUTPUT/'contract.json').read_text())['contract']==contract
    atomic_json(OUTPUT/'contract.json',dict(contract=contract,sources=sources,repair_contract=parent['contract'],targets=sorted(r['name'] for r in targets)))
    settings=json.loads(tune.CONFIG.read_text())['dagger']
    rows={r['name']:r for r in json.loads(Path(settings['track_manifest']).read_text())['records']}
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(rescue,(settings,rows[r['name']],contract)) for r in targets]
        for future in as_completed(futures):
            r=future.result();results.append(r)
            atomic_json(OUTPUT/'summary.json',dict(contract=contract,complete=prior['complete'] and len(results)==len(targets),records=results))
            print(json.dumps(dict(name=r['name'],qualified=r['selected_qualified'],selected=r['selected']['name'],before=r['baseline_median'],after=r['selected_median'])),flush=True)
    if not targets:atomic_json(OUTPUT/'summary.json',dict(contract=contract,complete=prior['complete'],records=[]))


if __name__=='__main__':main()
