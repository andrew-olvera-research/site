"""Refine unchanged courses without changing their admitted racing line."""
from concurrent.futures import ProcessPoolExecutor
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

ROOT=tune.ROOT
PRIMARY=ROOT/'outputs/diagnostics/v62111-train65-teacher-v4'
OUTPUT=PRIMARY/'local-refinement'


def local_candidates(settings,row):
    base=tune.baseline_candidate(settings,row)
    prior=json.loads((PRIMARY/row['name']/'selection.json').read_text())
    clean=[s for s in prior['screens'] if s['summary']['usable_rate']==1. and s['candidate']['name']!='baseline']
    best=min(clean,key=lambda s:s['summary']['median'])['candidate']['speed'] if clean else base['speed']+4
    speeds=sorted({base['speed'],base['speed']+1,base['speed']+2,min(best,36.)})
    candidates=[base]
    for mode in ('centered_balanced','modest'):
        for speed in speeds:
            if not base['speed']<=speed<=36:continue
            candidate=deepcopy(base)
            candidate.update(name=f'{mode}-{speed:g}',speed=speed)
            c=candidate['controller']
            if mode=='centered_balanced':
                c.update(max_progress_speed=41.,maximum_acceleration=34.,
                    maximum_longitudinal_acceleration=26.,maximum_braking_acceleration=28.,
                    maximum_collective_thrust=40.,collective_slew_limit=18.,body_rate_slew_limit=6.,
                    corridor_margin=.15)
            else:
                for key in ('maximum_acceleration','maximum_longitudinal_acceleration','maximum_braking_acceleration'):
                    c[key]=min(float(c[key])*1.1,40.)
                for key in ('collective_slew_limit','body_rate_slew_limit'):
                    c[key]=float(c[key])*1.1
                c['max_progress_speed']=max(float(c['max_progress_speed']),speed+2.)
            # Keep the original path/aperture for both authority probes.
            candidates.append(candidate)
    return candidates


def refine(payload):
    settings,row,contract=payload
    root=OUTPUT/row['name']
    (root/'baseline').mkdir(parents=True,exist_ok=True)
    for kind in ('screen','nominal','randomized','dart'):
        source=PRIMARY/row['name']/'baseline'/f'{kind}.json'
        destination=root/'baseline'/f'{kind}.json'
        if not destination.exists():
            old=json.loads(source.read_text())
            old.update(reused_from=dict(path=str(source),contract=old['contract']),contract=contract)
            atomic_json(destination,old)
    tune.candidates=local_candidates
    return tune.job((settings,row,str(OUTPUT),contract))


def main():
    settings=json.loads(tune.CONFIG.read_text())['dagger']
    rows={r['name']:r for r in json.loads(Path(settings['track_manifest']).read_text())['records']}
    primary_contract=json.loads((PRIMARY/'contract.json').read_text())
    sources={**primary_contract['sources'],str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for path,digest in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Changed experiment source: {path}')
    contract=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    OUTPUT.mkdir(exist_ok=True)
    contract_path=OUTPUT/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text())['contract']!=contract:
        raise ValueError('Existing refinement contract differs')
    atomic_json(contract_path,dict(contract=contract,sources=sources,primary_contract=primary_contract['contract'],
        search='Original racing line; +10% authority and centered balanced authority; original and nearby commands',
        acceptance='Same completion/solver/recovery requirements as primary search'))
    submitted=set();pending={};results=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        while True:
            main_summary=json.loads((PRIMARY/'summary.json').read_text())
            targets=[r for r in main_summary['records'] if r['selected']['name']=='baseline']
            for result in targets:
                if result['name'] not in submitted:
                    pending[pool.submit(refine,(settings,rows[result['name']],contract))]=result['name']
                    submitted.add(result['name'])
            for future in list(pending):
                if future.done():
                    result=future.result();results.append(result);pending.pop(future)
                    print(json.dumps(dict(done=len(results),queued=len(submitted),name=result['name'],
                        selected=result['selected']['name'],before=result['baseline_median'],after=result['selected_median'])),flush=True)
            complete=bool(main_summary['complete'] and not pending and len(results)==len(targets))
            atomic_json(OUTPUT/'summary.json',dict(contract=contract,complete=complete,
                primary_contract=primary_contract['contract'],records=sorted(results,key=lambda r:r['name'])))
            if complete:break
            time.sleep(5)


if __name__=='__main__':main()
