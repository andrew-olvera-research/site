"""Resumable whole-loop generation and MPCC qualification for v6.20.1."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing as mp
import os
import hashlib
from pathlib import Path
import sys
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import save_track_yaml,geometry_fingerprint
from starscream.env.tracks import load_track
from starscream.env.racing_manifold.transition_corpus import transition_metrics,shape_distance
from starscream.env.racing_manifold.transition_corpus_v201 import generate_course,screen
from scripts.audits.prepare_v620_transitions import pilot_job
from scripts.audits.prepare_v619_pool import digest
from scripts.audits.prepare_v619_pool import settings_for
from starscream.env.racing_manifold.transition_corpus import executed_transition_metrics
from scripts.audits.prepare_v620_transitions import finite_report
from starscream.env.racing_manifold.corpus_coverage import clone_distance

SPEC=Path('/workspace/configs/exp/v6.20.1/transition_pool.yaml')


def qualification_job(args):
    cfg,row=args
    import torch
    from scripts.audits.audit_dagger_teacher_labels import audit_track
    from starscream.course_navigation.behavior_pool import behavior_summary
    torch.set_num_threads(1)
    root=Path(cfg['output'])/'qualification'/row['name']/cfg['contract_sha256'][:12]
    root.mkdir(parents=True,exist_ok=True)
    if (root/'result.json').exists():return json.loads((root/'result.json').read_text())
    with (root/'solver.log').open('w') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    settings,stage=settings_for(cfg,row['path'])
    settings.update(deepcopy(cfg['teacher_overrides']))
    settings['mpcc_build_root']=f'/tmp/starscream-v6201-{os.getpid()}'
    cohorts={};behaviors={};transitions={};passed=True;backend_cache=[]
    for i,domain in enumerate(('nominal','randomized','dart','prefix')):
        s=deepcopy(settings)
        if domain=='nominal':s['dynamics_randomization']={'enabled':False}
        out=root/domain
        result=audit_track(s,stage,row['path'],0,1,1,cfg['seed']+9000000+100003*i,
            start_mode='expert-prefix-all' if domain=='prefix' else 'canonical',
            repeats_per_start=cfg['prefix_repeats'] if domain=='prefix' else cfg['canonical_repeats'],
            start_perturbation_scale=1.,dart_action_noise_scale=1. if domain=='dart' else 0.,
            dart_episode_fraction=1. if domain=='dart' else 0.,speed_fractions=(1.,),
            frontier_speed=cfg['speed_command'],trace_directory=out,backend_cache=backend_cache,
            qualification_limits={'solver':cfg['maximum_solver_failure'],'recovery':cfg['maximum_recovery']})
        cohorts[domain]=result
        behaviors[domain]=[];transitions[domain]=[]
        for path in sorted(out.glob('episode-*.npz')):
            with np.load(path) as trace:
                if len(trace['states'])>1:
                    behaviors[domain].append(behavior_summary(trace))
                    transitions[domain].append(executed_transition_metrics(trace))
        passed=bool(result['episodes']) and all(e['success'] and not e['prefix_failed'] and
            e['solver_failure_fraction']<=cfg['maximum_solver_failure'] and
            e['recovery_fraction']<=cfg['maximum_recovery'] for e in result['episodes'])
        if not passed:break
    report=finite_report(dict(name=row['name'],qualified=passed and len(cohorts)==4,
        contract_sha256=cfg['contract_sha256'],fingerprint=row['fingerprint'],
        cohorts=cohorts,behavior=behaviors,executed_transitions=transitions,
        teacher_overrides=cfg['teacher_overrides']))
    atomic_json(root/'result.json',report)
    return report


def full_qualify(cfg):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    if not (root/'candidates.json').exists() and cfg.get('candidate_ledger'):
        atomic_json(root/'candidates.json',json.loads(Path(cfg['candidate_ledger']).read_text()))
    payload=json.loads((root/'candidates.json').read_text())
    if payload['spec_sha256']!=digest(Path(cfg['generation_spec'])):raise ValueError('generation specification changed')
    rows=payload['records'];slots=sorted({r['slot'] for r in rows})
    if cfg.get('balanced_candidate_order',False):
        # First-fit over attempt-zero seeds chose almost one handedness.
        # Alternate preferred mirror per cell before observing policy outcomes;
        # admission still measures actual turn coverage after qualification.
        preferred={slot:i%2 for i,slot in enumerate(slots)}
        rows=sorted(rows,key=lambda r:(r['slot'],r['seed']%2!=preferred[r['slot']]))
    cfg=deepcopy(cfg)
    # Resolved teacher settings and source code hashes enter qualification cache
    # identity; no stale evidence survives a controller or contract change.
    from scripts.train_privileged_racing import load_config
    source_hashes={p:digest(Path(p)) for p in (
        'scripts/audits/prepare_v6201.py','scripts/audits/audit_dagger_teacher_labels.py',
        'scripts/train_privileged_racing.py','starscream/mpcc/controller.py',
        'starscream/mpcc/acados_backend.py','starscream/mpcc/config.py','starscream/mpcc/model.py',
        'starscream/mpcc/racing_line.py','starscream/env/racing_manifold/corpus_coverage.py')}
    cfg['contract_sha256']=hashlib.sha256(json.dumps(dict(config=cfg,
        resolved_teacher=load_config(Path(cfg['teacher_config']))['dagger'],sources=source_hashes),sort_keys=True).encode()).hexdigest()
    dest=root/'qualification.json';selected={};results=[]
    if dest.exists():
        old=json.loads(dest.read_text())
        if old['contract_sha256']!=cfg['contract_sha256']:raise ValueError('qualification contract changed: choose a new ledger')
        selected={r['slot']:r for r in old['selected']};results=old['records']
    def save(finished=False):
        atomic_json(dest,dict(contract_sha256=cfg['contract_sha256'],teacher_overrides=cfg['teacher_overrides'],
            source_hashes=source_hashes,qualification_config={k:v for k,v in cfg.items() if k!='contract_sha256'},
            selected=list(selected.values()),records=results,missing_slots=[s for s in slots if s not in selected],
            complete=len(selected)==len(slots),attempts_finished=finished))
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
        for attempt in range(cfg['candidates_per_slot']):
            pending=[]
            for slot in slots:
                rr=[r for r in rows if r['slot']==slot]
                if slot not in selected and attempt<len(rr):pending.append(rr[attempt])
            for row,result in zip(pending,pool.map(qualification_job,[(cfg,r) for r in pending])):
                if result['qualified']:
                    track=load_track(row['path']);collisions=[]
                    for other in selected.values():
                        if other['split']==row['split']:continue
                        distance=clone_distance(track,load_track(other['path']))
                        if distance is not None and distance<cfg['minimum_normalized_shape_distance']:
                            collisions.append(dict(name=other['name'],distance=distance))
                    result=dict(result,split_clone_conflicts=collisions,admitted=not collisions)
                    if not collisions:selected[row['slot']]=dict(row,status='qualified',qualification=result,qualified_speed_mps=cfg['speed_command'])
                results.append(result)
                save();print('qualified',row['slot'],result['qualified'],'admitted',result.get('admitted',False),len(selected),'/',len(slots),flush=True)
            if len(selected)==len(slots):break
    save(finished=True)
    print('remaining',len(slots)-len(selected),flush=True)


def generate(cfg):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    dest=root/'candidates.json'
    if dest.exists():raise FileExistsError('candidate ledger is immutable')
    rows=[];rejects=[]
    for split in ('train','validation'):
        for n in cfg['gate_counts']:
            for j,g in enumerate(cfg['grammars']):
                slot=f'{split}_{n}_{g}';accepted=0
                for k in range(100):
                    seed=cfg['seed']+n*100000+j*10000+(5000 if split=='validation' else 0)+k
                    name=f'v6201_{slot}_{k}'
                    t=generate_course(g,n,seed,name,revision=cfg.get('generator_revision',1));reasons=screen(t)
                    if reasons:rejects.append(dict(slot=slot,seed=seed,reasons=reasons));continue
                    path=save_track_yaml(t,root/'candidates'/f'{name}.yaml').resolve()
                    rows.append(dict(name=name,path=str(path),slot=slot,split=split,grammar=g,
                        family=f'{n}_{g}',source_family=f'{n}_{g}',source_stratum='canonical',
                        gate_count=n,parent_id=str(seed),seed=seed,fingerprint=geometry_fingerprint(t),
                        transitions=transition_metrics(t),status='unqualified'))
                    accepted+=1
                    if accepted==cfg['candidates_per_slot']:break
    atomic_json(dest,dict(records=rows,rejections=rejects,spec_sha256=digest(SPEC)))
    print('generated',len(rows),'rejected',len(rejects),flush=True)


def qualify(cfg, pilot=False):
    root=Path(cfg['output']);rows=json.loads((root/'candidates.json').read_text())['records']
    if pilot:rows=[r for r in rows if r['split']=='train' and r['gate_count'] in (4,7) and r['grammar'] in ('stacked_reversal','go_around')]
    slots=sorted({r['slot'] for r in rows});selected={};ledger=[]
    destination=root/('pilot.json' if pilot else 'qualification.json')
    if destination.exists():
        old=json.loads(destination.read_text());selected={r['slot']:r for r in old['selected']};ledger=old['records']
    def save():
        atomic_json(destination,dict(selected=list(selected.values()),records=ledger,
            missing_slots=[s for s in slots if s not in selected],complete=len(selected)==len(slots)))
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
        for attempt in range(cfg['candidates_per_slot']):
            pending=[]
            for slot in slots:
                rr=[r for r in rows if r['slot']==slot]
                if slot not in selected and attempt<len(rr):pending.append(rr[attempt])
            jobs=[(cfg,r,16.5,'nominal') for r in pending]
            clean=[]
            for row,result in zip(pending,pool.map(pilot_job,jobs)):
                ledger.append(result)
                if result['clean']:clean.append(dict(row,nominal=result))
                print(row['slot'],attempt,'nominal',result['clean'],flush=True);save()
            # Pilot measures nominal feasibility only, never freezes.
            if pilot:
                for row in clean:selected[row['slot']]=row
                save();continue
            for domain in ('randomized','dart','prefix'):
                survivors=[]
                for row,result in zip(clean,pool.map(pilot_job,[(cfg,r,16.5,domain) for r in clean])):
                    ledger.append(result)
                    if result['clean']:survivors.append(dict(row,**{domain:result}))
                    print(row['slot'],domain,result['clean'],flush=True);save()
                clean=survivors
            for row in clean:selected[row['slot']]=row
            save()
    print('selected',len(selected),'/',len(slots),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['generate','pilot','qualify','full-qualify']);ap.add_argument('--spec',type=Path,default=SPEC);a=ap.parse_args()
    SPEC=a.spec;c=yaml.safe_load(SPEC.read_text())
    if a.mode=='generate':generate(c)
    elif a.mode=='full-qualify':full_qualify(c)
    else:qualify(c,pilot=a.mode=='pilot')
