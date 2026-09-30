"""Resumable completion-constrained per-course MPCC pace search for train65.

Search actual usable lap times, not nominal command magnitude. The original
teacher is always measured and retained if no faster candidate qualifies.
No training, W&B, WSL changes or validation-course tuning occurs here.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from starscream.course_model.training import atomic_json
from scripts.audits.audit_dagger_teacher_labels import audit_track

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/exp/v6.21.1.1/plant_dagger.yaml'
OUT = ROOT/'outputs/diagnostics/v62111-train65-teacher'
BACKENDS = []


def usable(row):
    return (row['success'] and not row.get('prefix_failed',False)
            and row['solver_failure_fraction'] <= .06 and row['recovery_fraction'] <= .10)


def summary(rows):
    laps = [r['lap_time_seconds'] for r in rows if usable(r)]
    return dict(episodes=len(rows),usable=len(laps),successes=sum(r['success'] for r in rows),
        median=float(np.median(laps)) if laps else None,
        usable_rate=len(laps)/len(rows),success_rate=sum(r['success'] for r in rows)/len(rows))


def baseline_candidate(s,row):
    profile = row['qualification']['selected']['teacher_profile']
    from scripts.train_privileged_racing import _merge
    controller = _merge(s.get('mpcc_config',{}),s['mpcc_manifest_teacher_profile_controller_configs'][profile])
    controller = _merge(controller,s.get('mpcc_family_configs',{}).get(row['family'],{}))
    planner = _merge(s.get('mpcc_planner_config',{}),s.get('mpcc_family_planner_configs',{}).get(row['family'],{}))
    planner = _merge(planner,s['mpcc_manifest_teacher_profile_planner_configs'][profile])
    return dict(name='baseline',speed=float(row['qualified_speed_mps']),controller=controller,planner=planner)


def candidates(s,row):
    base = baseline_candidate(s,row)
    result = [base]
    for mode in ('command','balanced','frontier','time_optimal'):
        for speed in sorted({base['speed']+2.,base['speed']+4.,16.,20.,24.,28.,32.,36.}):
            if speed>36.:
                continue
            if speed <= base['speed']:
                continue
            item = deepcopy(base)
            item.update(name=f'{mode}-{speed:g}',speed=speed)
            c,p = item['controller'],item['planner']
            if mode != 'command':
                c.update(max_progress_speed=41.,maximum_acceleration=34. if mode=='balanced' else 40.,
                    maximum_longitudinal_acceleration=26. if mode=='balanced' else 34.,
                    maximum_braking_acceleration=28. if mode=='balanced' else 36.,
                    maximum_collective_thrust=40.,collective_slew_limit=18. if mode=='balanced' else 24.,
                    body_rate_slew_limit=6. if mode=='balanced' else 8.,corridor_margin=.15 if mode=='balanced' else .12)
                p.update(aperture_fraction=.28 if mode=='balanced' else .35,aperture_margin=.12,offset_iterations=40)
            if mode == 'time_optimal':
                c.update(progress_reference_mode='time_optimal',terminal_speed_envelope=True)
            result.append(item)
    return result


def cohort(s,row,candidate,kind,root,contract):
    path = root/candidate['name']/f'{kind}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text())
        if old['contract'] != contract or old['candidate'] != candidate:
            raise ValueError(f'Stale candidate evidence: {path}')
        return old
    configured = deepcopy(s)
    configured.update(mpcc_use_manifest_teacher_profile=False,mpcc_use_manifest_planner_profile=False,
        mpcc_use_manifest_speed=False,mpcc_speed_frontier_profile='',
        # Constructor reference stays fixed so identical immutable profiles can
        # reuse a backend. audit_track sets each runtime command before reset.
        mpcc_nominal_speed=16.5,mpcc_config=candidate['controller'],mpcc_planner_config=candidate['planner'],
        mpcc_family_configs={},mpcc_family_planner_configs={},mpcc_family_nominal_speeds={},
        mpcc_build_root=f'/tmp/starscream-train65-tuner-{os.getpid()}')
    baseline_path=root/'baseline'/'screen.json'
    if candidate['name']!='baseline' and baseline_path.exists():
        baseline_rows=json.loads(baseline_path.read_text())['episodes']
        clean_steps=[r['steps'] for r in baseline_rows if usable(r)]
        if clean_steps:
            # This is a pace search: a lap over twice the matched baseline is
            # ineligible. Bound failed solver loops without weakening success.
            configured['curriculum']['max_steps']=min(configured['curriculum']['max_steps'],2*max(clean_steps))
    specs = dict(screen=(2,621112400,True,False),nominal=(2,621112500,False,False),
                 randomized=(6,621112600,True,False),dart=(2,621112700,True,True))
    repeats,seed,randomized,dart = specs[kind]
    if not randomized:
        configured['dynamics_randomization'] = {'enabled':False}
    report = audit_track(configured,configured['curriculum'],row['path'],0,1,repeats,seed,
        start_mode='canonical',repeats_per_start=repeats,
        start_perturbation_scale=float(s.get('dagger_dart_start_perturbation_scale',1.)) if dart else 1.,
        dart_action_noise_scale=1. if dart else 0.,dart_episode_fraction=1. if dart else 0.,
        speed_fractions=(1.,),frontier_speed=candidate['speed'],backend_cache=BACKENDS)
    result = dict(contract=contract,candidate=candidate,cohort=kind,episodes=report['episodes'],summary=summary(report['episodes']))
    atomic_json(path,result)
    return result


def qualifies(cohorts):
    # Preserve the original all-trials completion requirement, including DART.
    return all(c['summary']['usable'] == c['summary']['episodes'] for c in cohorts.values())


def job(payload):
    s,row,output,contract = payload
    import torch
    torch.set_num_threads(1)
    root = Path(output)/row['name']
    final = root/'selection.json'
    if final.exists():
        result=json.loads(final.read_text())
        if result['contract'] != contract:
            raise ValueError('Existing selection contract differs')
        return result
    started=time.perf_counter()
    screens=[]
    for candidate in candidates(s,row):
        result=cohort(s,row,candidate,'screen',root,contract)
        screens.append(result)
    fast_screens=sorted((r for r in screens[1:] if r['summary']['usable_rate']==1.),
                        key=lambda r:r['summary']['median'])
    existing={r['candidate']['name'] for r in screens}
    for screen in fast_screens[:2]:
        for delta in (-2.,-1.,1.,2.):
            candidate=deepcopy(screen['candidate'])
            speed=candidate['speed']+delta
            if not screens[0]['candidate']['speed']<speed<=36.:
                continue
            mode=candidate['name'].rsplit('-',1)[0]
            candidate.update(name=f'{mode}-{speed:g}',speed=speed)
            if candidate['name'] not in existing:
                screens.append(cohort(s,row,candidate,'screen',root,contract));existing.add(candidate['name'])
    baseline=screens[0]
    # Deduplicate saturated command trajectories before costly confirmation.
    seen=set()
    ranked=[]
    for screen in sorted(screens[1:],key=lambda r:r['summary']['median'] or float('inf')):
        if screen['summary']['usable_rate'] != 1.:
            continue
        if baseline['summary']['median'] is not None and screen['summary']['median'] >= .99*baseline['summary']['median']:
            continue
        signature=tuple((r['steps'],round(r['maximum_speed_mps'],5)) for r in screen['episodes'])
        if signature not in seen:
            ranked.append(screen['candidate']);seen.add(signature)
    confirmed={}
    def confirm(candidate):
        reports={}
        for kind in ('nominal','randomized','dart'):
            reports[kind]=cohort(s,row,candidate,kind,root,contract)
            if candidate['name']!='baseline' and not qualifies(reports):
                break
        confirmed[candidate['name']]=reports
        return reports
    baseline_reports=confirm(baseline['candidate'])
    selected=baseline['candidate']
    best_time=baseline_reports['randomized']['summary']['median']
    accepted=[]
    for candidate in ranked:
        reports=confirm(candidate)
        lap=reports.get('randomized',{}).get('summary',{}).get('median')
        if len(reports)==3 and qualifies(reports):
            accepted.append(candidate['name'])
            if best_time is None or lap < .99*best_time:
                selected=candidate;best_time=lap
        if len(accepted)>=2:
            break
    result=dict(contract=contract,name=row['name'],selected=selected,baseline=baseline['candidate'],
        baseline_qualified=qualifies(baseline_reports),accepted=accepted,
        selected_qualified=qualifies(confirmed[selected['name']]),
        baseline_median=baseline_reports['randomized']['summary']['median'],selected_median=best_time,
        confirmations={k:{c:r['summary'] for c,r in v.items()} for k,v in confirmed.items()},
        screens=[dict(candidate=r['candidate'],summary=r['summary']) for r in screens],
        seconds=time.perf_counter()-started)
    atomic_json(final,result)
    return result


def reuse_evidence(previous,output,contract,sources):
    """Reuse trials only when their inputs and execution functions are identical."""
    old=json.loads((previous/'contract.json').read_text())
    this_script=str(Path(__file__))
    if {k:v for k,v in old['sources'].items() if k!=this_script}!={k:v for k,v in sources.items() if k!=this_script}:
        raise ValueError('Simulator or input contract changed; cannot reuse trials')
    def execution_code(path):
        tree=ast.parse(path.read_text())
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
                if isinstance(n,ast.FunctionDef) and n.name in ('cohort','summary','usable','baseline_candidate')}
    if execution_code(previous/'tuner-source.py')!=execution_code(Path(__file__)):
        raise ValueError('Trial execution code changed; cannot reuse trials')
    count=0
    for path in previous.glob('*/*/*.json'):
        if path.stem not in ('screen','nominal','randomized','dart'):
            continue
        destination=output/path.relative_to(previous)
        if destination.exists():
            continue
        result=json.loads(path.read_text())
        if result['contract']!=old['contract']:
            raise ValueError('Mixed evidence contracts')
        result.update(contract=contract,reused_from=dict(path=str(path.resolve()),contract=old['contract']))
        destination.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(destination,result);count+=1
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--reuse-evidence',type=Path)
    args=p.parse_args()
    s=json.loads(CONFIG.read_text())['dagger']
    manifest=Path(s['track_manifest'])
    rows=json.loads(manifest.read_text())['records']
    active=set(s['curriculum']['tracks'])
    rows=[r for r in rows if r['path'] in active]
    assert len(rows)==65
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (CONFIG,manifest,Path(__file__),ROOT/'scripts/audits/audit_dagger_teacher_labels.py',
         ROOT/'scripts/train_privileged_racing.py',ROOT/'starscream/mpcc/controller.py',ROOT/'starscream/mpcc/config.py')}
    contract=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    contract_path=args.output/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text())['contract']!=contract:
        raise ValueError('Refusing to overwrite evidence from another contract')
    atomic_json(contract_path,dict(contract=contract,sources=sources,
        acceptance='2/2 nominal, 6/6 randomized, 2/2 DART usable; <=6% solver failure and <=10% recovery per episode',
        objective='shortest qualified randomized median lap, >=1% improvement',ceiling_mps=36.,
        candidate_timeout='at most twice longest usable baseline screen lap',
        search='20/24/28/32/36 commands; command/balanced/frontier/time-optimal profiles; local +/-1,2 m/s'))
    if args.reuse_evidence:
        print('reused_cohorts='+str(reuse_evidence(args.reuse_evidence,args.output,contract,sources)),flush=True)
    results=[]
    chosen=rows[:args.limit or None]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(job,(s,row,str(args.output),contract)) for row in chosen]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            atomic_json(args.output/'summary.json',dict(contract=contract,complete=len(results)==65,
                records=sorted(results,key=lambda r:r['name'])))
            print(json.dumps(dict(done=len(results),total=len(chosen),name=result['name'],
                selected=result['selected']['name'],before=result['baseline_median'],after=result['selected_median'],
                qualified=result['selected_qualified'])),flush=True)


if __name__=='__main__':main()
