"""Build candidates, audit directed transitions, and run bounded teacher pilots.

No training launcher and no auto-freeze. Failed cells remain in the ledger.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.tracks import load_track
from starscream.env.procedural_tracks import save_track_yaml, geometry_fingerprint
from starscream.env.racing_distribution import RacingDistributionConfig
from starscream.env.racing_distribution.generator import RacingTaskGenerator
from starscream.env.racing_manifold.transition_corpus import (
    GRAMMARS, graft_transition, static_admission, transition_metrics, shape_distance,
    executed_transition_metrics)
from scripts.audits.prepare_v619_pool import strata, settings_for, digest

SPEC = Path('/workspace/configs/exp/v6.20/transition_pool.yaml')


def finite_report(value):
    """Undefined successful-lap metrics are null, never fake zero or NaN JSON."""
    if isinstance(value,dict):return {k:finite_report(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_report(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def generate(cfg):
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    if (root/'candidates.json').exists():
        raise FileExistsError('Immutable candidate ledger exists; use a new output for changed generation')
    base = json.loads(Path(cfg['generator_config']).read_text())
    records, rejected = [], []
    for f,(cell,size,vertical,order) in enumerate(strata()):
        settings = dict(base, bounds_xy_m=[10.,13.] if size=='compact' else [13.,17.],
            minimum_altitude_m=2., maximum_altitude_m=5.5 if vertical=='low' else 8.,
            spline_control_points=[6,9], spline_control_order_mode=order)
        generator = RacingTaskGenerator(RacingDistributionConfig(**settings))
        for split in ('train','validation'):
            # Checkerboard, covering both values of every physical factor.
            if split=='validation' and f not in (0,3,5,6): continue
            for g,grammar in enumerate(GRAMMARS):
                slot = f'{cell}_{grammar}_{split}'
                accepted = 0
                for attempt in range(100):
                    seed = cfg['seed']+f*100000+g*10000+(5000 if split=='validation' else 0)+attempt
                    try:
                        parent = generator.generate(backend='informed_spline', seed=seed, split=split, name=f'parent_{seed}')
                        name = f'v620_{slot}_{attempt}'
                        track = graft_transition(parent, grammar, seed+1, name)
                        reasons = static_admission(track)
                    except ValueError as e:
                        rejected.append(dict(slot=slot,seed=seed,reasons=[str(e)])); continue
                    if reasons:
                        rejected.append(dict(slot=slot,seed=seed,reasons=reasons)); continue
                    path = save_track_yaml(track,root/'candidates'/f'{name}.yaml').resolve()
                    records.append(dict(name=name,path=str(path),slot=slot,split=split,
                        family=f'{cell}::{grammar}',geometry_cell=cell,grammar=grammar,
                        parent_id=f'parent_{seed}',parent_fingerprint=geometry_fingerprint(parent),
                        fingerprint=geometry_fingerprint(track),seed=seed,
                        source_family=f'{cell}::{grammar}',source_stratum='canonical',
                        status='static_candidate_not_dynamically_qualified',transitions=transition_metrics(track)))
                    accepted += 1
                    if accepted == cfg['candidates_per_slot']: break
                if accepted < cfg['candidates_per_slot']:
                    rejected.append(dict(slot=slot,reasons=['unfilled_slot']))
    atomic_json(root/'candidates.json',dict(records=records,rejections=rejected,spec_sha256=digest(SPEC)))
    audit_static(cfg)
    print('candidates',len(records),'static rejections',len(rejected),flush=True)


def audit_static(cfg):
    root=Path(cfg['output']); rows=json.loads((root/'candidates.json').read_text())['records']
    loaded={r['name']:load_track(r['path']) for r in rows}
    train=[r for r in rows if r['split']=='train']; val=[r for r in rows if r['split']=='validation']
    nearest=[]
    for r in val:
        pairs=[(shape_distance(loaded[r['name']],loaded[t['name']]),t['name']) for t in train]
        pairs=[(d,n) for d,n in pairs if d is not None]
        d,n=min(pairs) if pairs else (None,None)
        nearest.append(dict(name=r['name'],nearest_train=n,distance=d,
            too_close=d is not None and d<cfg['minimum_normalized_shape_distance']))
    excluded={r['name'] for r in nearest if r['too_close']}
    eligible=[r for r in rows if r['name'] not in excluded]
    # Keep rejected proposals in the immutable source ledger, never silently
    # use a near-clone as held-out validation.
    atomic_json(root/'eligible_candidates.json',dict(records=eligible,excluded_near_clones=sorted(excluded),
        missing_slots=sorted({r['slot'] for r in rows}-{r['slot'] for r in eligible}),
        status='static_only_not_frozen'))
    atomic_json(root/'static_audit.json',dict(
        train_slots=len({r['slot'] for r in train}),validation_slots=len({r['slot'] for r in val}),
        parent_overlap=sorted({r['parent_id'] for r in train}&{r['parent_id'] for r in val}),
        fingerprint_overlap=sorted({r['fingerprint'] for r in train}&{r['fingerprint'] for r in val}),
        validation_nearest=nearest,
        references=[dict(path=p,transitions=transition_metrics(load_track(p))) for p in cfg['references']],
        caveat='Yaw/scale-invariant distance detects clones; it does not prove behavioral transfer.'))


def pilot_job(args):
    cfg,row,speed,domain=args
    import torch
    from scripts.audits.audit_dagger_teacher_labels import audit_track
    from starscream.course_navigation.behavior_pool import behavior_summary
    torch.set_num_threads(1)
    variant=cfg.get('variant','baseline')
    out=Path(cfg['output'])/'teacher-pilot'/row['name']/f'{speed:g}-{domain}-{variant}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): return json.loads((out/'result.json').read_text())
    # Suppress ACADOS chatter only inside this isolated worker.
    with (out/'solver.log').open('w') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    s,stage=settings_for(cfg,row['path'])
    s['mpcc_build_root']=f'/tmp/starscream-v620-{os.getpid()}'
    if variant!='baseline':
        # Runtime set_nominal_speed changes pace, but constructor profile
        # limits are chosen before that. Explicitly test both contracts.
        s['mpcc_nominal_speed']=speed
    if variant=='centered-frontier':
        s['mpcc_planner_config']=dict(sample_count=1400,aperture_fraction=.10,
            aperture_margin=.22,offset_iterations=30)
    if domain=='nominal': s['dynamics_randomization']={'enabled':False}
    result=audit_track(s,stage,row['path'],0,1,1,cfg['seed']+8000000,
        start_mode='expert-prefix-all' if domain=='prefix' else 'canonical',
        repeats_per_start=1,start_perturbation_scale=1.,
        dart_action_noise_scale=1. if domain=='dart' else 0.,
        dart_episode_fraction=1. if domain=='dart' else 0.,speed_fractions=(1.,),
        frontier_speed=speed,trace_directory=out,
        qualification_limits={'solver':cfg['maximum_solver_failure'],'recovery':cfg['maximum_recovery']})
    behavior=[]; executed=[]
    for p in sorted(out.glob('episode-*.npz')):
        with np.load(p) as trace:
            if len(trace['states'])>1:
                behavior.append(behavior_summary(trace))
                executed.append(executed_transition_metrics(trace))
    summary=result['summary']
    clean=summary['success_rate']==1 and summary['solver_failure_fraction']<=cfg['maximum_solver_failure'] and summary['recovery_fraction']<=cfg['maximum_recovery']
    report=dict(name=row['name'],path=row['path'],grammar=row.get('grammar','reference'),
        speed=speed,domain=domain,variant=variant,clean=clean,audit=result,behavior=behavior,
        constructor_nominal_speed=s['mpcc_nominal_speed'],executed_transitions=executed,
        scope='pilot only; nominal pass is not full admission')
    atomic_json(out/'result.json',report)
    return report


def pilot(cfg, domain):
    root=Path(cfg['output']);rows=json.loads((root/'candidates.json').read_text())['records']
    suffix='' if cfg.get('variant','baseline')=='baseline' else '-'+cfg['variant']
    # Train-only pilot selection independent of policy outcomes.
    selected=[[r for r in rows if r['split']=='train' and r['grammar']==g][cfg.get('candidate_index',0)] for g in GRAMMARS]
    if cfg.get('candidate_index',0):suffix+=f'-candidate{cfg["candidate_index"]}'
    selected += [dict(name=Path(p).stem,path=p) for p in cfg['references'][:2]]
    if cfg.get('technical_only'):
        selected=[r for r in selected if r.get('grammar') in ('stacked_reversal','go_around')]
        suffix+='-technical'
    if domain=='nominal':
        jobs=[(cfg,r,s,domain) for r in selected for s in cfg['speed_commands']]
    else:
        prior=json.loads((root/f'pilot-nominal{suffix}.json').read_text())
        if not prior['complete']:raise ValueError('finish nominal frontier first')
        jobs=[]
        for row in selected:
            clean=[r for r in prior['records'] if r['name']==row['name'] and r['clean']]
            if clean:
                # Fastest executed lap, NOT largest configured speed command.
                best=min(clean,key=lambda r:r['audit']['summary']['successful_lap_time_seconds'])
                jobs.append((cfg,row,best['speed'],domain))
    results=[]
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
        for r in pool.map(pilot_job,jobs):
            results.append(r)
            atomic_json(root/f'pilot-{domain}{suffix}.json',dict(records=results,complete=len(results)==len(jobs)))
            print(r['name'],r['speed'],'clean',r['clean'],flush=True)


def probe(cfg, checkpoint):
    import torch
    from dataclasses import replace
    from scripts.train_privileged_racing import load_config,parse_stage,ProcessRaceCollector,multitrack_metrics
    from starscream.privileged_racing import load_policy_checkpoint
    torch.set_num_threads(1)
    root=Path(cfg['output']);pilot=json.loads((root/'pilot-nominal.json').read_text())
    if not pilot['complete']:raise ValueError('finish teacher pilot first')
    # Never inspect validation for selection: only train proposals and already
    # declared development references. Keep all failures, including the floor.
    records=[]
    for file in root.glob('pilot-nominal*.json'):
        report=json.loads(file.read_text())
        if report['complete']:records.extend(report['records'])
    paths=sorted({r['path'] for r in records if r['clean']} | set(cfg['references'][:2]))
    policy,norm,payload,resolved=load_policy_checkpoint(Path(checkpoint),'cuda')
    policy.eval()
    settings=deepcopy(load_config(Path(cfg['teacher_config']))['dagger'])
    settings.update(dynamics_randomization={'enabled':False},evaluation_workers=2)
    raw=deepcopy(settings['reporting_evaluation_curriculum'])
    raw.update(tracks=paths,target_speed=16.5)
    outputs=[]
    if (root/'policy-pilot.json').exists():
        previous=json.loads((root/'policy-pilot.json').read_text())
        if previous['checkpoint']!=str(resolved):raise ValueError('Use a new output for a different policy probe')
        outputs=previous['records']
    for path in paths:
        if any(r['path']==path for r in outputs):continue
        stage=replace(parse_stage(raw),tracks=(path,),target_gates=len(load_track(path).gates))
        collector=ProcessRaceCollector(policy,norm,settings,stage,'cuda',workers=2)
        try:
            with torch.no_grad(): rows=collector.evaluate_rows(episodes=12,seed_base=2026091463)
        finally:collector.close()
        metrics=multitrack_metrics(rows,stage.target_gates)
        outputs.append(finite_report(dict(path=path,metrics=metrics,episodes=rows)))
        atomic_json(root/'policy-pilot.json',dict(checkpoint=str(resolved),round=payload.get('round'),
            status='development_diagnostic_not_admission',nominal_dynamics=True,
            complete=len(outputs)==len(paths),records=outputs))
        print('policy',Path(path).stem,'full',metrics['full_course_success'],'gates',metrics['mean_gates'],flush=True)


def summarize(cfg):
    root=Path(cfg['output']); results=[]
    for file in sorted((root/'teacher-pilot').glob('*/*/result.json')):
        r=json.loads(file.read_text());s=r['audit']['summary']
        executed=[]
        for trace_path in sorted(file.parent.glob('episode-*.npz')):
            with np.load(trace_path) as trace:
                if len(trace['states'])>1:executed.append(executed_transition_metrics(trace))
        results.append(dict(name=r['name'],grammar=r['grammar'],speed=r['speed'],
            variant=r.get('variant','baseline'),domain=r['domain'],clean=r['clean'],
            full=s['success_rate'],lap_seconds=s['successful_lap_time_seconds'],
            recovery=s['recovery_fraction'],solver_failure=s['solver_failure_fraction'],
            gates=s['mean_gates'],executed_transitions=executed,result_path=str(file)))
    qualified={g:sorted({r['name'] for r in results if r['grammar']==g and r['clean'] and r['domain']=='nominal'}) for g in GRAMMARS}
    fastest={}
    for r in results:
        if r['clean'] and r['domain']=='nominal' and (r['name'] not in fastest or r['lap_seconds']<fastest[r['name']]['lap_seconds']):
            fastest[r['name']]=r
    atomic_json(root/'development_summary.json',dict(status='not_frozen_not_training_ready',
        nominal_clean_by_grammar=qualified,missing_pilot_grammars=[g for g,n in qualified.items() if not n],
        fastest_clean_nominal=fastest,teacher_pilots=results,
        caveat='Single nominal flights do not establish reliability. Hard cells cannot be replaced by easy ones.'))
    print({g:len(n) for g,n in qualified.items()},flush=True)


def route_support(cfg):
    """Compare one-lap terminal-faithful route support, not transfer prediction."""
    root=Path(cfg['output'])
    prior=json.loads(Path('/workspace/outputs/course-pools/v619-green-inspired/manifest.json').read_text())['records']
    proposed=json.loads((root/'eligible_candidates.json').read_text())['records']
    summary=json.loads((root/'development_summary.json').read_text())
    clean=set(summary['fastest_clean_nominal'])
    pools={'v619_train':[r for r in prior if r['split']=='train'],
        'v620_proposed_train':[r for r in proposed if r['split']=='train'],
        'v620_nominal_pilot_clean_train':[r for r in proposed if r['split']=='train' and r['name'] in clean]}
    def records(t):
        a=np.stack([t.flight_plan(i,6,remaining=len(t.gates)-i)['records'] for i in range(len(t.gates))]).astype(float)
        a[:,:,:3]/=20.;a[:,:,9:11]/=3.
        return a.reshape(len(t.gates),-1)/np.sqrt(78)
    refs={p:records(load_track(p)) for p in cfg['references']}
    result={}
    for key,rows in pools.items():
        values=[];labels=[]
        for r in rows:
            a=records(load_track(r['path']));values.extend(a)
            labels.extend([dict(name=r['name'],active_gate=i) for i in range(len(a))])
        if not values:continue
        a=np.asarray(values);result[key]={}
        for path,ref in refs.items():
            distance=np.linalg.norm(ref[:,None]-a[None],axis=-1)
            result[key][Path(path).stem]=[dict(active_gate=i,nearest=labels[int(np.argmin(d))],distance=float(d.min())) for i,d in enumerate(distance)]
    atomic_json(root/'reference_route_support.json',dict(results=result,
        scope='development reference only; fixed physical scales; one-lap terminal repetition; route geometry not policy-state/behavior distance'))
    print('saved reference route support',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['generate','audit','pilot','probe','summarize','route-support']);ap.add_argument('--domain',choices=['nominal','randomized','dart','prefix'],default='nominal');ap.add_argument('--checkpoint');ap.add_argument('--variant',choices=['baseline','frontier','centered-frontier'],default='baseline');ap.add_argument('--speeds');ap.add_argument('--candidate-index',type=int,default=0);ap.add_argument('--technical-only',action='store_true')
    args=ap.parse_args();cfg=yaml.safe_load(SPEC.read_text())
    cfg['variant']=args.variant
    cfg['candidate_index']=args.candidate_index
    cfg['technical_only']=args.technical_only
    if args.speeds:cfg['speed_commands']=[float(x) for x in args.speeds.split(',')]
    if args.mode=='generate':generate(cfg)
    elif args.mode=='audit':audit_static(cfg)
    elif args.mode=='pilot':pilot(cfg,args.domain)
    elif args.mode=='summarize':summarize(cfg)
    elif args.mode=='route-support':route_support(cfg)
    elif not args.checkpoint:ap.error('probe requires --checkpoint')
    else:probe(cfg,args.checkpoint)
