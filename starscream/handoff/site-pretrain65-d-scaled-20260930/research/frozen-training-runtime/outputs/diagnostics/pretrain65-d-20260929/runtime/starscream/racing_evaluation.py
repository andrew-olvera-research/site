"""Versioned racing deadlines and fixed-shape inference for comparable evaluation.

Reference times are frozen empirical witnesses, not certified optimal times.
The deadline is an administrative truncation, never an absorbing crash state.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deadline_steps(reference_seconds, multiplier=1.5, recovery_seconds=3., control_hz=130):
    if not all(math.isfinite(v) for v in (reference_seconds, multiplier, recovery_seconds, control_hz)):
        raise ValueError('deadline values must be finite')
    if reference_seconds <= 0 or multiplier < 1 or recovery_seconds < 0 or control_hz <= 0:
        raise ValueError('invalid deadline parameters')
    # floor: never accept a completion later than the specified time limit.
    return math.floor(max(multiplier*reference_seconds, reference_seconds+recovery_seconds)*control_hz+1e-9)


def score_episode(success, steps, reference_seconds, cutoff_steps, control_hz=130, crashed=False):
    if steps < 1 or cutoff_steps < 1:
        raise ValueError('positive step counts required')
    timely = bool(success and not crashed and steps <= cutoff_steps)
    return dict(timely_success=float(timely),
                time_weighted_success=float(timely)*min(1., reference_seconds/(steps/control_hz)),
                completion_time_ratio=steps/control_hz/reference_seconds if success else None)


def load_protocol(path, expected_sha256=None):
    from starscream.evaluation_suite import repository_path
    path=repository_path(path)
    if expected_sha256 is not None and digest(path)!=expected_sha256:
        raise ValueError('evaluation protocol fingerprint changed')
    protocol=json.loads(path.read_text())
    if protocol['schema']!='starscream-racing-evaluation-v2':
        raise ValueError('unknown racing evaluation protocol')
    if protocol['control_hz'] != 130 or protocol['inference']['batch_size']!=16:
        raise ValueError('v2 protocol requires 130 Hz and canonical batch size 16')
    if protocol['inference']['precision']!='fp32_math_no_tf32':
        raise ValueError('unsupported evaluation precision')
    if digest(repository_path(protocol['source_manifest']))!=protocol['source_manifest_sha256']:
        raise ValueError('evaluation geometry manifest changed')
    records=protocol['records']
    if len({r['slot'] for r in records})!=len(records):
        raise ValueError('duplicate protocol slots')
    checked_evidence={}
    for r in records:
        evidence=r['evidence']
        if evidence not in checked_evidence:
            checked_evidence[evidence]=digest(repository_path(evidence))
        if checked_evidence[evidence]!=r['evidence_sha256']:
            raise ValueError('reference-time evidence changed')
        if digest(repository_path(r['path']))!=r['track_sha256']:
            raise ValueError(f'course geometry changed: {r["slot"]}')
        expected=deadline_steps(r['reference_seconds'],protocol['deadline_multiplier'],
                                protocol['recovery_allowance_seconds'],protocol['control_hz'])
        if r['deadline_steps']!=expected:
            raise ValueError('stored deadline differs from formula')
    return protocol


@contextmanager
def canonical_inference():
    """Pin FP32 SDPA and matmul settings and restore the caller's train settings."""
    old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        with torch.inference_mode(), torch.autocast('cuda',enabled=False), sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)=old


def fixed_shape_call(function, tensors, lane_ids, batch_size=16):
    """Live lanes always keep their original positions; unused lanes are zero."""
    lanes=list(lane_ids)
    if len(lanes)!=len(set(lanes)) or not lanes or min(lanes)<0 or max(lanes)>=batch_size:
        raise ValueError('invalid fixed inference lanes')
    padded=[]
    for tensor in tensors:
        if tensor.shape[0]!=len(lanes):
            raise ValueError('input batch and live lanes differ')
        out=tensor.new_zeros((batch_size,*tensor.shape[1:]))
        out[lanes]=tensor
        padded.append(out)
    result=function(*padded)
    return result[lanes]


def aggregate_report(rows, family_weights):
    metrics=('timely_success','time_weighted_success','crashed','timeout','clean_success')
    families={}
    for row in rows:
        eps=row['episodes']
        row.update({k:float(np.mean([e[k] for e in eps])) for k in metrics})
        row['success']=row['timely_success']
        times=[e['lap_seconds'] for e in eps if e['timely_success']]
        row['fastest_lap_seconds']=min(times) if times else None
        row['fastest_lap_ratio']=min(e['completion_time_ratio'] for e in eps if e['timely_success']) if times else None
        families.setdefault(row['family'],[]).append(row)
    if set(families)!=set(family_weights) or not np.isclose(sum(family_weights.values()),1):
        raise ValueError('family weights do not match evaluated courses')
    result={k:float(sum(family_weights[f]*np.mean([r[k] for r in group]) for f,group in families.items())) for k in metrics}
    result['selection_score']=result['timely_success']
    result['selection_key']=[result['timely_success'],result['time_weighted_success'],-result['crashed']]
    result['tracks']=rows
    result['observed_completion']=float(np.mean([e['success'] for r in rows for e in r['episodes']]))
    return result


def evaluate_racing(config, settings, stage, suite, teacher, normalizer, student, device, teacher_only=False):
    """Shared teacher/student evaluator. No simulator reward or learning is modified."""
    from concurrent.futures import ThreadPoolExecutor
    from scripts.train_vision_student import environment, SensorHistory, sample_curriculum_spawn, ppo_normalized_to_ctbr
    protocol=load_protocol(config['evaluation_protocol'],config.get('evaluation_protocol_sha256'))
    references={r['slot']:r for r in protocol['records']}
    batch_size=protocol['inference']['batch_size']
    workers=int(config.get('evaluation_workers',batch_size))
    if not 1<=workers<=batch_size:
        raise ValueError('evaluation_workers must be 1..16; inference always uses 16')
    if float(config['speed_command'])!=float(protocol['speed_command']):
        raise ValueError('policy speed command differs from evaluation protocol')
    if stage.random_gate or stage.mpcc_prefix_steps:
        raise ValueError('timed protocol requires gate-zero starts without expert prefixes')
    if int(config['evaluation_episodes'])<1:
        raise ValueError('positive evaluation episode count required')
    jobs=[]
    for r in suite['records']:
        ref=references[r['slot']]
        if r['path']!=ref['path']:
            raise ValueError('suite course differs from deadline reference')
        offset=int(hashlib.sha256(r['slot'].encode()).hexdigest()[:8],16)%1_000_000
        for repeat in range(config['evaluation_episodes']):
            jobs.append((r,ref,config['evaluation_seed']+offset+repeat,repeat))
    mean=torch.as_tensor(normalizer.mean,device=device)
    std=torch.as_tensor(normalizer.std,device=device)
    results={r['slot']:[] for r in suite['records']}
    teacher.eval();student.eval()
    model_config=config['model']
    with canonical_inference():
        for start in range(0,len(jobs),workers):
            slots=[]
            try:
                for lane,(r,ref,seed,repeat) in enumerate(jobs[start:start+workers]):
                    env=environment(r['path'],config,settings,stage)
                    slot=dict(env=env,lane=lane,record=r,reference=ref,seed=seed,repeat=repeat)
                    slots.append(slot)
                    spawn=sample_curriculum_spawn(env.track,stage,seed=seed,episode_index=0)
                    obs,_=env.reset(seed=seed,options=dict(state=spawn.state,gate_index=spawn.gate_index,route_plan_total_gates=len(env.track.gates)))
                    sensor=SensorHistory(seed,config['observation'],settings,
                        memory_steps=model_config.get('memory_steps',0),memory_stride=model_config.get('memory_stride',2))
                    sensor.append(obs)
                    from starscream.dagger_quality import ReferenceGateEvents
                    from scripts.train_privileged_racing import make_gate_reference
                    reference_events = None
                    if settings.get('ppo_reference_aware_gate_events', False):
                        line, _, _, _ = make_gate_reference(slot['env'], settings)
                        reference_events = ReferenceGateEvents(line, len(slot['env'].track.gates), obs['state'][:3],
                            float(settings.get('gate_event_reference_tolerance', .75)))
                    slot['gate_events'] = reference_events
                    slot.update(obs=obs,sensor=sensor,steps=0,misses=0,returns=0,recovered=0,
                                missed_gate=False,first_miss=None,recovery_seconds=[],last_pass=0,max_dwell=0,events=[],crashed=False)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    while slots:
                        arrays=[s['sensor'].arrays() for s in slots]
                        features,masks,truth=[np.stack(v) for v in zip(*arrays)]
                        speed=torch.full((len(slots),),float(config['speed_command']),device=device)
                        lanes=[s['lane'] for s in slots]
                        if teacher_only:
                            action=fixed_shape_call(teacher,[(torch.as_tensor(truth,device=device)-mean)/std,speed],lanes,batch_size)
                        else:
                            inputs=[torch.as_tensor(features,device=device),torch.as_tensor(masks,device=device).float(),speed]
                            if model_config.get('memory_steps',0):
                                inputs.append(torch.as_tensor(np.stack([s['sensor'].memory_array() for s in slots]),device=device))
                                call=lambda f,m,v,h:student(f,m,v,memory=h)['action']
                            else:call=lambda f,m,v:student(f,m,v)['action']
                            action=fixed_shape_call(call,inputs,lanes,batch_size)
                        actions=action.float().cpu().numpy()
                        before=[]
                        for s in slots:
                            gate=s['env'].track.gates[s['env'].tracker.index]
                            before.append((gate,s['env'].tracker.passed_count,float((s['obs']['state'][:3]-gate.position)@gate.normal),
                                           s['env'].tracker.index,s['obs']['state'][:3].copy()))
                        futures=[pool.submit(s['env'].step,ppo_normalized_to_ctbr(a,settings)) for s,a in zip(slots,actions)]
                        alive=[]
                        for s,(gate,passed,side,gate_index,previous),future in zip(slots,before,futures):
                            obs,_,terminated,truncated,info=future.result();s['steps']+=1
                            s['crashed'] |= bool(info.get('ground_contact',False) or info.get('unity_collision',False))
                            new_passed=s['env'].tracker.passed_count
                            new_side=float((obs['state'][:3]-gate.position)@gate.normal)
                            genuine_miss = (s['gate_events'].advance(previous, obs['state'][:3], gate, gate_index, new_passed>passed)
                                if s['gate_events'] else side<0<=new_side)
                            event=None
                            if new_passed>passed:
                                event='pass';s['max_dwell']=max(s['max_dwell'],s['steps']-s['last_pass']);s['last_pass']=s['steps']
                                if s['missed_gate']:
                                    s['recovered']+=1;s['recovery_seconds'].append((s['steps']-s['first_miss'])/130)
                                s['missed_gate']=False;s['first_miss']=None
                            elif genuine_miss:
                                event='miss';s['misses']+=1
                                if not s['missed_gate']:s['first_miss']=s['steps']
                                s['missed_gate']=True
                            elif side>=0>new_side:event='return';s['returns']+=1
                            if event and len(s['events'])<256:s['events'].append(dict(step=s['steps'],gate=passed,kind=event))
                            success=new_passed>=len(s['env'].track.gates) and not s['crashed']
                            cutoff=s['reference']['deadline_steps']
                            expired=s['steps']>=cutoff
                            if success or s['crashed'] or terminated or truncated or expired:
                                timeout=bool(expired and not success and not s['crashed'] and not terminated)
                                reason='success' if success else 'crash' if s['crashed'] else 'deadline' if timeout else 'environment_terminal'
                                result=dict(seed=s['seed'],repeat=s['repeat'],success=float(success),gates=new_passed,
                                    target_gates=len(s['env'].track.gates),steps=s['steps'],lap_seconds=s['steps']/130 if success else None,
                                    reference_seconds=s['reference']['reference_seconds'],deadline_steps=cutoff,
                                    crashed=float(s['crashed']),timeout=float(timeout),terminated=bool(terminated or s['crashed']),
                                    truncated=bool(truncated or timeout),termination_reason=reason,
                                    clean_success=float(success and s['misses']==0),missed_crossings=s['misses'],
                                    backward_crossings=s['returns'],recovered_gates=s['recovered'],recovery_seconds=s['recovery_seconds'],
                                    max_gate_dwell_seconds=max(s['max_dwell'],s['steps']-s['last_pass'])/130,events=s['events'])
                                result.update(score_episode(success,s['steps'],s['reference']['reference_seconds'],cutoff,crashed=s['crashed']))
                                results[s['record']['slot']].append(result);s['env'].close()
                            else:
                                s['obs']=obs;s['sensor'].append(obs);alive.append(s)
                        slots=alive
            finally:
                for s in slots:s['env'].close()
    rows=[dict(slot=r['slot'],family=r['family'],episodes=results[r['slot']]) for r in suite['records']]
    report=aggregate_report(rows,suite['family_weights'])
    report['miss_definition'] = 'ordered-reference-v1' if settings.get('ppo_reference_aware_gate_events', False) else 'raw-plane-v1'
    report['reference_groups']={}
    for kind in sorted({references[r['slot']]['reference_kind'] for r in rows}):
        group=[r for r in rows if references[r['slot']]['reference_kind']==kind]
        report['reference_groups'][kind]=dict(courses=len(group),aggregation='equal-course',
            timely_success=float(np.mean([r['timely_success'] for r in group])),
            time_weighted_success=float(np.mean([r['time_weighted_success'] for r in group])))
    report['protocol']=dict(path=config['evaluation_protocol'],sha256=config.get('evaluation_protocol_sha256'),
        schema=protocol['schema'],inference=protocol['inference'],workers=workers,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else str(device),
        deadline_is_truncation=True,reference_is_certified_optimal=False,
        evaluation_seed=config['evaluation_seed'],speed_command=config['speed_command'],
        teacher_sha256=config.get('teacher_sha256'),
        configuration_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
    return report
