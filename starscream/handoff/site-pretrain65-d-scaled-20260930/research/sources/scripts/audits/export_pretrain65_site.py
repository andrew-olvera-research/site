"""Replay frozen benchmark witnesses and render a provenance-checked site gallery."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import render_research_gifs as renderer
from scripts.train_privileged_racing import (CausalHistory, load_config, make_env,
    make_reward, make_gate_reference, parse_stage, ppo_normalized_to_ctbr,
    ppo_observation_features, reset_env)
from starscream.dagger_quality import ReferenceGateEvents
from starscream.privileged_racing import load_policy_checkpoint
from starscream.racing_evaluation import canonical_inference, fixed_shape_call, load_protocol
from starscream.env.tracks import load_track

PACKAGE = ROOT / 'outputs/site-pretrain65-d-scaled-20260930'
CONFIG = ROOT / 'outputs/diagnostics/pretrain65-d-20260929/production-config.json'
CHECKPOINT = ROOT / 'outputs/checkpoints/starscream-pretrain65-d-scaled-20260929/best-step-029465874-selection_suite_timely_success-0.7375.pt'
EVAL = ROOT / 'outputs/evals/pretrain65-d-scaled-final-20260929/real100-v2-e32.json'
PROTOCOL = ROOT / 'configs/eval/real100_v2_timed_protocol_v1.json'
EXPECTED_CHECKPOINT_HASH = 'cee73ce74fdfcfd64adb7b10d374ff143ef07980a61feff53112641d7a56eb3a'

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def plan_jobs(protocol, report):
    by_name = {r['name']: r for r in protocol['records']}
    by_slot = {r['slot']: r for r in protocol['records']}
    old = read(ROOT / 'outputs/site-trajectories-v6211-18-20260923/manifest.json')
    names = list(dict.fromkeys(a['name'] for a in old['assets']))
    jobs = []
    def add(record, row, role, kind):
        jobs.append(dict(record=record, expected=row, role=role, kind=kind,
                         lane=int(row['repeat']) % 16))
    for name in names:
        record = by_name[name]
        rows = [r for r in report['episodes'] if r['slot'] == record['slot'] and r['success']]
        clean = [r for r in rows if r['clean_success']]
        if rows:
            add(record, min(clean or rows, key=lambda r: (r['steps'], r['repeat'])),
                'established_gallery', 'clean' if clean else 'recovery_fallback')
        else:
            add(record, max((r for r in report['episodes'] if r['slot']==record['slot']),
                            key=lambda r: (r['gates'], -r['steps'])),
                'established_gallery', 'failure')
    a2rl = next(r for r in protocol['records'] if r['name'].startswith('a2rl'))
    successes = [r for r in report['episodes'] if r['slot']==a2rl['slot'] and r['success']]
    add(a2rl, min(successes, key=lambda r:r['steps']), 'difficult_public_reference', 'recovery')
    failures = [r for r in report['episodes'] if r['slot']==a2rl['slot'] and not r['success']]
    add(a2rl, max(failures, key=lambda r:(r['gates'], -r['steps'])),
        'difficult_public_reference', 'failure')
    long_row = next(r for r in report['episodes'] if r['slot']=='real100-hard-055' and r['seed']==2035006365)
    add(by_slot[long_row['slot']], long_row, 'long_recovery_witness', 'late_recovery')
    # A short missed-and-return flight offers a readable contrast to the tail.
    gallery_slots = {j['record']['slot'] for j in jobs[:18]}
    recoveries = [r for r in report['episodes'] if r['slot'] in gallery_slots
                  and r['success'] and not r['clean_success'] and r['timely_success']]
    short = min(recoveries, key=lambda r:r['steps'])
    add(by_slot[short['slot']], short, 'short_recovery_witness', 'recovery')
    return jobs

def capture(jobs, settings, policy, normalizer, stage, output):
    remaining = list(enumerate(jobs, 1))
    with canonical_inference():
        while remaining:
            batch_jobs, used = [], set()
            for item in remaining:
                if item[1]['lane'] not in used:
                    batch_jobs.append(item)
                    used.add(item[1]['lane'])
            remaining = [j for j in remaining if j not in batch_jobs]
            slots = []
            try:
                for ordinal, job in batch_jobs:
                    record, expected = job['record'], job['expected']
                    env = make_env(settings, track=str(ROOT/record['path']), reward=make_reward(settings,16.5))
                    obs, start_passed = reset_env(env,stage,seed=int(expected['seed']),episode_index=0)
                    history = CausalHistory(policy.context_steps)
                    history.reset_feature(ppo_observation_features(obs,settings))
                    line, _, _, _ = make_gate_reference(env,settings)
                    tracker = ReferenceGateEvents(line,len(env.track.gates),obs['state'][:3],.75)
                    state = np.asarray(obs['state'],np.float32).copy()
                    slots.append(dict(job=job,ordinal=ordinal,env=env,obs=obs,history=history,
                        tracker=tracker,start=start_passed,steps=0,misses=0,events=[],crashed=False,
                        done=False,frames=[dict(state=state,time=float(obs['time']),passed=0,
                                               speed=float(np.linalg.norm(state[7:10]))) ]))
                with ThreadPoolExecutor(max_workers=len(slots)) as pool:
                    while any(not s['done'] for s in slots):
                        active = [s for s in slots if not s['done']]
                        histories = np.stack([s['history'].array() for s in active])
                        tensor = torch.from_numpy(normalizer.numpy(histories)).to('cuda')
                        speeds = torch.full((len(active),),16.5,device='cuda')
                        actions = fixed_shape_call(policy,[tensor,speeds],[s['job']['lane'] for s in active],16).float().cpu().numpy()
                        before = [(s['env'].track.gates[s['env'].tracker.index],s['env'].tracker.index,
                                   s['env'].tracker.passed_count,s['obs']['state'][:3].copy()) for s in active]
                        futures = [pool.submit(s['env'].step,ppo_normalized_to_ctbr(a,settings)) for s,a in zip(active,actions)]
                        for s,(gate,index,passed,previous),future in zip(active,before,futures):
                            obs,_,terminated,truncated,info = future.result()
                            s['steps'] += 1
                            new_passed=s['env'].tracker.passed_count
                            miss=s['tracker'].advance(previous,obs['state'][:3],gate,index,new_passed>passed)
                            if new_passed>passed:
                                s['events'].append(dict(step=s['steps'],kind='pass',gate=passed))
                            elif miss:
                                s['misses'] += 1
                                s['events'].append(dict(step=s['steps'],kind='miss',gate=passed))
                            s['crashed'] |= bool(info.get('ground_contact') or info.get('unity_collision'))
                            s['obs']=obs
                            s['history'].append_feature(ppo_observation_features(obs,settings))
                            state=np.asarray(obs['state'],np.float32).copy()
                            s['frames'].append(dict(state=state,time=float(obs['time']),passed=int(new_passed-s['start']),
                                                    speed=float(np.linalg.norm(state[7:10]))))
                            success=new_passed-s['start']>=len(s['env'].track.gates)
                            if success or s['crashed'] or terminated or truncated or s['steps']>=6000:
                                expected=s['job']['expected']
                                actual=dict(success=int(success),steps=s['steps'],missed_crossings=s['misses'],
                                    crashed=int(s['crashed']),gates=int(new_passed-s['start']))
                                for key in actual:
                                    if actual[key] != expected[key]:
                                        raise RuntimeError(f"Replay mismatch ordinal={s['ordinal']} {key}: {actual[key]} vs {expected[key]}")
                                if s['events'] != expected['events']:
                                    raise RuntimeError(f"Event sequence mismatch ordinal={s['ordinal']}")
                                summary=dict(track=s['env'].track.name,track_fingerprint=s['env'].track.fingerprint,
                                    episode_index=int(expected['repeat']),seed=int(expected['seed']),steps=s['steps'],
                                    target_gates=len(s['env'].track.gates),passed_gates=actual['gates'],completed=bool(success),
                                    crashed=s['crashed'],missed_gate_crossings=s['misses'],target_speed_mps=16.5,
                                    duration_seconds=s['steps']/130,maximum_speed_mps=max(f['speed'] for f in s['frames']),
                                    control_hz=130,miss_definition='ordered-reference-v1',timely_success=bool(expected['timely_success']),
                                    clean_success=bool(expected['clean_success']),deadline_steps=expected['deadline_steps'],
                                    benchmark_slot=expected['slot'],benchmark_repeat=expected['repeat'],
                                    inference='fp32_math_no_tf32; batch 16; original stable lane; inactive lanes zero',
                                    original_lane=s['job']['lane'],reproduced_exact_steps_outcomes_and_events=True)
                                stem=output/f"capture-{s['ordinal']:02d}"
                                np.savez_compressed(stem.with_suffix('.npz'),states=np.stack([f['state'] for f in s['frames']]),
                                    times=np.array([f['time'] for f in s['frames']]),passed=np.array([f['passed'] for f in s['frames']]))
                                write(stem.with_suffix('.json'),summary)
                                s['done']=True
                                print(f"Captured {s['ordinal']}: {summary['track']} {s['steps']} steps; exact match",flush=True)
            finally:
                for s in slots:
                    s['env'].close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()!=EXPECTED_CHECKPOINT_HASH:
        raise ValueError('Pinned checkpoint changed')
    protocol=load_protocol(PROTOCOL)
    report=read(EVAL)
    jobs=plan_jobs(protocol,report)
    captures=PACKAGE/'captures'
    captures.mkdir(parents=True,exist_ok=True)
    write(captures/'jobs.json',jobs)
    if not args.render_only:
        policy,normalizer,_,resolved=load_policy_checkpoint(CHECKPOINT,'cuda')
        policy.eval()
        settings=dict(load_config(CONFIG)['dagger'])
        stage=replace(parse_stage(settings['evaluation_curriculum']),target_gates=64,max_steps=6000,
                      random_gate=False,fixed_start_gate_index=0,rollout_laps=1)
        settings['evaluation_fixed_start_gate_index']=0
        capture(jobs,settings,policy,normalizer,stage,captures)
        del policy,normalizer
        torch.cuda.empty_cache()
    output=PACKAGE/'generated'
    output.mkdir(parents=True,exist_ok=True)
    render_args=argparse.Namespace(output=output,fps=20,width=720,height=405,fov=72.,keep_frames=False)
    assets=[]
    for ordinal,job in enumerate(jobs,1):
        record=job['record']
        summary=read(captures/f'capture-{ordinal:02d}.json')
        archive=np.load(captures/f'capture-{ordinal:02d}.npz')
        frames=[dict(state=s,time=float(t),passed=int(p),speed=float(np.linalg.norm(s[7:10])))
                for s,t,p in zip(archive['states'],archive['times'],archive['passed'])]
        rows=[r for r in report['episodes'] if r['slot']==record['slot']]
        selection=dict(name=record['name'],family=record['family'],exposure='held-out geometry; inspected development benchmark',
            role=job['role'],capture_kind=job['kind'],track_path=record['path'],episodes_evaluated=32,
            successful_episodes=sum(r['success'] for r in rows),capture_episodes_replayed=1,
            capture_clean_successful_episodes=int(summary['completed'] and summary['clean_success']),
            benchmark_success_rate=sum(r['success'] for r in rows)/32,
            benchmark_fastest_steps=min((r['steps'] for r in rows if r['success']),default=None))
        sidecars=list(output.glob(f'{ordinal:02d}-*.json'))
        existing=[p for p in sidecars if not p.name.endswith('.demo.json')]
        if existing:
            asset=read(existing[0])
        else:
            print(f'Rendering {ordinal}/{len(jobs)} {record["name"]}',flush=True)
            asset=renderer.render_selection(load_track(ROOT/record['path']),frames,summary,selection,ordinal,render_args)
        assets.append(asset)
    manifest=dict(schema='starscream-pretrain65-site-gallery-v1',checkpoint=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=EXPECTED_CHECKPOINT_HASH,config=str(CONFIG.relative_to(ROOT)),
        benchmark=str(EVAL.relative_to(ROOT)),protocol=str(PROTOCOL.relative_to(ROOT)),
        selection_protocol='Curated frozen e32 benchmark witnesses: fastest clean completion when available; separate recovered, late and failure examples. Every capture exactly reproduced original steps, outcomes and gate event sequence.',
        assets=assets)
    write(output/'manifest.json',manifest)
    content=['<!doctype html><meta charset="utf-8"><title>Scaled D benchmark witnesses</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}figure{margin:30px 0}img{max-width:100%}figcaption{margin:8px 0}</style>',
        '<h1>Pretrain65 scaled D — frozen benchmark witnesses</h1>',
        '<p>Curated full episodes at simulation speed. These are demonstrations, not a success-rate estimate. Clean, recovered, late, and failed flights are labeled using the ordered-reference benchmark rule. Original geometry is preserved.</p>']
    for a in assets:
        content.append(f'<figure><img src="{html.escape(a["gif"])}"><figcaption>{html.escape(a["name"])} · {a["capture_kind"]} · {a["steps"]/130:.2f} s · seed {a["seed"]} · misses {a["missed_gate_crossings"]} · <a href="{a["mp4"]}">MP4</a> · <a href="{a["trajectory_json"]}">trajectory JSON</a></figcaption></figure>')
    (output/'index.html').write_text('\n'.join(content))
    print(f'Complete: {len(assets)} assets',flush=True)

if __name__=='__main__':
    main()
