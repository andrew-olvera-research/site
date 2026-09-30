"""Bounded Green-inspired spline/codec pool, with explicit provenance gates.

Numeric paper benchmark files were not located. Never relabel local proxies.
Splits/slots precede expert qualification; all failures remain in the ledger.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import multiprocessing as mp
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.course_model.schema import static_reasons
from starscream.env.tracks import load_track
from starscream.env.procedural_tracks import save_track_yaml, geometry_fingerprint
from starscream.env.racing_distribution import RacingDistributionConfig
from starscream.env.racing_distribution.generator import RacingTaskGenerator

SPEC=Path('/workspace/configs/exp/v6.19/pool.yaml')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def strata():
    # Physical design strata, NOT claimed discrete maneuver families from paper.
    return [(f'{size}_{vertical}_{order}', size, vertical, order)
            for size in ['compact','open'] for vertical in ['low','high']
            for order in ['azimuth_sorted','sampled']]

def settings_for(cfg, path):
    from scripts.train_privileged_racing import load_config
    s=deepcopy(load_config(Path(cfg['teacher_config']))['dagger'])
    for k in list(s):
        if k.startswith(('mpcc_family_', 'dagger_transition_start_', 'track_sampling_')): s.pop(k)
    s.update(track_manifest=None,qualified_tracks_only=False,
        mpcc_use_manifest_speed=False,mpcc_use_manifest_teacher_profile=False,
        mpcc_use_manifest_planner_profile=False,
        mpcc_build_root=f'/tmp/starscream-v619-audit-{os.getpid()}',
        racing_line_cache=str(Path(cfg['output'])/'racing-lines'))
    stage=deepcopy(s['curriculum'])
    for k in ['track_manifest','track_split','track_families','qualified_tracks_only',
              'minimum_qualified_speed','real_course_suite','track_limit']:
        stage.pop(k,None)
    stage.update(tracks=[path],target_gates=len(load_track(path).gates),
        max_steps=cfg['max_steps'],rollout_laps=1,random_gate=False,
        fixed_start_gate_index=0,manifest_speed_scale_range=None,
        allow_archived_task_resets=False)
    return s,stage

def screen_geometry(track):
    p=np.array([g.position for g in track.gates]);d=np.linalg.norm(np.roll(p,-1,0)-p,axis=1)
    reasons=static_reasons(track)
    if not 6<=len(p)<=8: reasons.append('paper_gate_count')
    if not 30<=d.sum()<=105: reasons.append('length_envelope')
    if d.min()<2.5 or d.max()>28: reasons.append('spacing_envelope')
    for g in track.gates:
        if np.any(g.position<track.bounds[:,0]) or np.any(g.position>track.bounds[:,1]): reasons.append('bounds')
    return sorted(set(reasons))

def generate(cfg):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    prior=json.loads((root/'candidates.json').read_text()) if (root/'candidates.json').exists() else {}
    rows=list(prior.get('records',[]));rejects=list(prior.get('static_rejections',[]));base=json.loads(Path(cfg['generator_config']).read_text())
    qualified=json.loads((root/'parent_qualification.json').read_text()) if (root/'parent_qualification.json').exists() else {}
    done={r['parent_id'] for r in qualified.get('selected',[])}
    for f,(family,size,vertical,order) in enumerate(strata()):
        c=dict(base,bounds_xy_m=[10.,13.] if size=='compact' else [13.,17.],
            minimum_altitude_m=1.4,maximum_altitude_m=4.7 if vertical=='low' else 7.,
            spline_control_points=[6,9],spline_control_order_mode=order)
        generator=RacingTaskGenerator(RacingDistributionConfig(**c))
        for split,slots in [('train',cfg['train_parents_per_stratum']),('validation',cfg['validation_parents_per_stratum'])]:
            for slot in range(slots):
                slot_id=f'{family}_{split}_{slot}'
                existing=[r for r in rows if r['parent_id']==slot_id]
                accepted=len(existing)
                if slot_id in done or accepted>=cfg['candidates_per_slot']:continue
                first=max([r['attempt'] for r in existing],default=-1)+1
                for attempt in range(first,first+400):
                    seed=cfg['seed']+f*100000+(50000 if split=='validation' else 0)+slot*1000+attempt
                    name=f'v619_{slot_id}_{attempt}'
                    try:
                        t=generator.generate(backend='informed_spline',seed=seed,split=split,name=name)
                        reasons=screen_geometry(t)
                    except ValueError as e:
                        rejects.append(dict(slot=slot_id,seed=seed,reasons=[str(e)]));continue
                    if reasons:
                        rejects.append(dict(slot=slot_id,seed=seed,reasons=reasons));continue
                    p=save_track_yaml(t,root/'candidates'/f'{name}.yaml').resolve()
                    rows.append(dict(name=name,path=str(p),family=family,split=split,
                        parent_id=slot_id,seed=seed,attempt=attempt,origin='informed_spline',
                        fingerprint=geometry_fingerprint(t),generator_config=c))
                    accepted+=1
                    if accepted>=cfg['candidates_per_slot']: break
                if accepted<cfg['candidates_per_slot']: raise RuntimeError(f'insufficient static candidates: {slot_id}')
    atomic_json(root/'candidates.json',dict(records=rows,static_rejections=rejects,
        scope='Green-inspired physical envelope, not paper numeric reproduction',spec_sha256=digest(SPEC)))
    print('generated',len(rows),'candidates',flush=True)

def qualify_one(args):
    cfg,row=args
    import torch
    from scripts.audits.audit_dagger_teacher_labels import audit_track
    from starscream.course_navigation.behavior_pool import behavior_summary
    torch.set_num_threads(1)
    root=Path(cfg['output']);out=root/'qualification'/row['name'];out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
    fd=os.open(os.devnull,os.O_WRONLY);os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
    s,stage=settings_for(cfg,row['path']);cache=[];cohorts={};behavior=None
    for k,domain in enumerate(['nominal','randomized','dart']):
        ss=deepcopy(s)
        if domain=='nominal':ss['dynamics_randomization']={'enabled':False}
        result=audit_track(ss,stage,row['path'],0,1,1,cfg['seed']+1000000+1009*k,
            start_mode='canonical',repeats_per_start=1,start_perturbation_scale=1.,
            dart_action_noise_scale=1. if domain=='dart' else 0.,
            dart_episode_fraction=1. if domain=='dart' else 0.,speed_fractions=(1.,),
            frontier_speed=cfg['speed_command'],trace_directory=out/domain,backend_cache=cache,
            qualification_limits={'solver':cfg['maximum_solver_failure'],'recovery':cfg['maximum_recovery']})
        summary=result['summary'];cohorts[domain]=summary
        passed=summary['success_rate']==1 and summary['solver_failure_fraction']<=cfg['maximum_solver_failure'] and summary['recovery_fraction']<=cfg['maximum_recovery']
        if domain=='nominal' and passed:
            with np.load(out/domain/'episode-000.npz') as trace:behavior=behavior_summary(trace)
        if not passed:break
    passed=len(cohorts)==3 and all(x['success_rate']==1 and x['solver_failure_fraction']<=cfg['maximum_solver_failure'] and x['recovery_fraction']<=cfg['maximum_recovery'] for x in cohorts.values())
    r=dict(name=row['name'],qualified=passed,cohorts=cohorts,behavior=behavior)
    atomic_json(out/'result.json',r);return r

def qualify(cfg):
    root=Path(cfg['output']);rows=json.loads((root/'candidates.json').read_text())['records']
    slots={r['parent_id'] for r in rows};chosen={};results=[]
    # Keep a deterministic first-passing candidate per predeclared slot. No
    # policy/validation score or benchmark geometry is used for selection.
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
        for attempt in range(cfg['candidates_per_slot']):
            todo=[]
            for slot in sorted(slots-chosen.keys()):
                rr=[r for r in rows if r['parent_id']==slot]
                if attempt<len(rr):todo.append(rr[attempt])
            for row,result in zip(todo,pool.map(qualify_one,[(cfg,r) for r in todo])):
                results.append(result)
                if result['qualified']:chosen[row['parent_id']]=dict(row,qualification=result)
                atomic_json(root/'parent_qualification.json',dict(records=results,selected=list(chosen.values()),missing=sorted(slots-chosen.keys())))
                print('qualified',row['parent_id'],result['qualified'],len(chosen),'/',len(slots),flush=True)
            if len(chosen)==len(slots):break
    if slots-chosen.keys():raise RuntimeError(f'Unfilled strata: {sorted(slots-chosen.keys())}; do not dilute/remove silently')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['generate','qualify']);args=ap.parse_args()
    cfg=yaml.safe_load(SPEC.read_text())
    {'generate':generate,'qualify':qualify}[args.mode](cfg)

if __name__=='__main__':main()
