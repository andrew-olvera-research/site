#!/usr/bin/env python3
"""Capped real100-v2 teacher evaluation with timely and bounded-total outcomes."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from scripts.train_privileged_racing import (
    CausalHistory, load_config, make_env, make_reward, parse_stage,
    ppo_normalized_to_ctbr, ppo_observation_features, reset_env, stage_config, make_gate_reference,
)
from starscream.privileged_racing import load_policy_checkpoint
from starscream.racing_evaluation import canonical_inference, fixed_shape_call, load_protocol

from starscream.dagger_quality import ReferenceGateEvents

ROOT=Path(__file__).resolve().parents[2]

def complete(success, steps, deadline, misses, recovered, reason):
    timely=bool(success and steps<=deadline)
    return dict(success=int(success), timely_success=int(timely),
                clean_success=int(success and misses==0),
                recovered_success=int(success and recovered>0),
                late_success=int(success and not timely),
                steps=int(steps), missed_crossings=int(misses),
                recovered_gates=int(recovered), termination_reason=reason)

def evaluate(args, *, loaded=None):
    torch.set_num_threads(1)
    speed_command = float(getattr(args, "speed_command", 16.5))
    if not np.isfinite(speed_command) or speed_command <= 0:
        raise ValueError("speed command must be finite and positive")
    config=stage_config(load_config(args.config), "dagger")
    settings=dict(config["dagger"])
    policy, normalizer, payload, resolved=(load_policy_checkpoint(args.checkpoint,args.device) if loaded is None else loaded)
    protocol=load_protocol(args.protocol)
    records=protocol["records"][:args.limit_courses or None]
    stage=replace(parse_stage(settings["evaluation_curriculum"]),
                  target_gates=64, max_steps=args.max_steps, random_gate=False,
                  fixed_start_gate_index=0, rollout_laps=1)
    settings["evaluation_fixed_start_gate_index"]=0
    policy.eval()
    jobs=[]
    for record in records:
        offset=int(hashlib.sha256(record["slot"].encode()).hexdigest()[:8],16)%1_000_000
        for repeat in range(args.episodes):
            jobs.append((record,args.seed+offset+repeat,repeat))
    results=[]
    sampled=[]
    with canonical_inference():
        for start in range(0,len(jobs),args.workers):
            slots=[]
            try:
                for lane,(record,seed,repeat) in enumerate(jobs[start:start+args.workers]):
                    track=str(ROOT/record["path"])
                    env=make_env(settings,track=track,reward=make_reward(settings,16.5))
                    observation,start_passed=reset_env(env,stage,seed=seed,episode_index=0)
                    history=CausalHistory(policy.context_steps)
                    history.reset_feature(ppo_observation_features(observation,settings))
                    events = None
                    if args.reference_aware:
                        line, _, _, _ = make_gate_reference(env, settings)
                        events = ReferenceGateEvents(line, len(env.track.gates), observation['state'][:3], args.reference_tolerance)
                    slots.append(dict(gate_events=events,lane=lane,env=env,record=record,seed=seed,repeat=repeat,
                                      observation=observation,history=history,start_passed=start_passed,
                                      steps=0,last_pass=0,misses=0,recovered=0,first_miss=None,
                                      crashed=False,done=False,events=[],path_m=0.,trajectory=[]))
                with ThreadPoolExecutor(max_workers=len(slots)) as pool:
                    while any(not s["done"] for s in slots):
                        active=[s for s in slots if not s["done"]]
                        histories=np.stack([s["history"].array() for s in active])
                        if args.save_states and len(sampled)<args.max_states:
                            for slot,h in zip(active,histories):
                                if slot["steps"]%20==0 and len(sampled)<args.max_states:
                                    sampled.append((h.copy(),slot["record"]["slot"],slot["seed"]))
                        batch=torch.from_numpy(normalizer.numpy(histories)).to(args.device)
                        speeds=torch.full((len(active),),speed_command,device=args.device)
                        lanes=[s["lane"] for s in active]
                        actions=fixed_shape_call(policy,[batch,speeds],lanes,16).float().cpu().numpy()
                        before=[]
                        for s in active:
                            gate=s["env"].track.gates[s["env"].tracker.index]
                            side=float((s["observation"]["state"][:3]-gate.position)@gate.normal)
                            before.append((gate,s["env"].tracker.passed_count,side,s["env"].tracker.index,s["observation"]["state"][:3].copy()))
                        futures=[pool.submit(s["env"].step,ppo_normalized_to_ctbr(action,settings))
                                 for s,action in zip(active,actions)]
                        for s,(gate,passed,side,gate_index,previous),future in zip(active,before,futures):
                            obs,_,terminated,truncated,info=future.result()
                            s["path_m"] += float(np.linalg.norm(obs["state"][:3]-previous))
                            s["steps"]+=1
                            if getattr(args, "trajectory_stride", 0) and s["steps"] % args.trajectory_stride == 0:
                                s["trajectory"].append([s["steps"], *map(float, obs["state"][:3]), int(s["env"].tracker.passed_count)])
                            s["observation"]=obs
                            s["history"].append_feature(ppo_observation_features(obs,settings))
                            s["crashed"] |= bool(info.get("ground_contact") or info.get("unity_collision"))
                            new_passed=s["env"].tracker.passed_count
                            new_side=float((obs["state"][:3]-gate.position)@gate.normal)
                            genuine_miss = (s['gate_events'].advance(previous, obs['state'][:3], gate, gate_index, new_passed>passed)
                                if s['gate_events'] else side<0<=new_side)
                            if new_passed>passed:
                                if s["first_miss"] is not None:s["recovered"]+=1
                                s["first_miss"]=None
                                s["last_pass"]=s["steps"]
                                s["events"].append(dict(step=s["steps"],kind="pass",gate=passed))
                            elif genuine_miss:
                                s["misses"]+=1
                                if s["first_miss"] is None:s["first_miss"]=s["steps"]
                                s["events"].append(dict(step=s["steps"],kind="miss",gate=passed))
                            success=new_passed-s["start_passed"]>=len(s["env"].track.gates)
                            miss_expired=(s["first_miss"] is not None and
                                          s["steps"]-s["first_miss"]>=args.retry_steps)
                            dwell_expired=s["steps"]-s["last_pass"]>=args.dwell_steps
                            hard_expired=s["steps"]>=args.max_steps
                            if success or s["crashed"] or terminated or truncated or miss_expired or dwell_expired or hard_expired:
                                reason=("success" if success else "crash" if s["crashed"] or terminated
                                        else "retry_limit" if miss_expired else "gate_dwell_limit" if dwell_expired
                                        else "hard_cap" if hard_expired else "environment_terminal")
                                out=complete(success,s["steps"],s["record"]["deadline_steps"],
                                             s["misses"],s["recovered"],reason)
                                out.update(slot=s["record"]["slot"],family=s["record"]["family"],
                                           seed=s["seed"],repeat=s["repeat"],gates=new_passed-s["start_passed"],
                                           target_gates=len(s["env"].track.gates),
                                           reference_seconds=s["record"]["reference_seconds"],
                                           deadline_steps=s["record"]["deadline_steps"],
                                           crashed=int(s["crashed"]),events=s["events"][:256],
                                           events_total=len(s["events"]), path_m=s["path_m"],
                                           mean_path_speed_mps=s["path_m"]/(s["steps"]/130),
                                           trajectory=s["trajectory"])
                                results.append(out)
                                s["done"]=True
            finally:
                for s in slots:s["env"].close()
            print(f"progress {len(results)}/{len(jobs)}",flush=True)
    def metrics(rows):
        n=len(rows)
        done=[r for r in rows if r["success"]]
        return dict(episodes=n,total_success=sum(r["success"] for r in rows)/n,
                    timely_success=sum(r["timely_success"] for r in rows)/n,
                    clean_success=sum(r["clean_success"] for r in rows)/n,
                    clean_timely_success=sum(bool(r["clean_success"] and r["timely_success"]) for r in rows)/n,
                    recovered_success=sum(r["recovered_success"] for r in rows)/n,
                    late_success=sum(r["late_success"] for r in rows)/n,
                    crash_rate=sum(r["crashed"] for r in rows)/n,
                    median_success_seconds=float(np.median([r["steps"]/130 for r in done])) if done else None,
                    p90_success_seconds=float(np.quantile([r["steps"]/130 for r in done],.9)) if done else None,
                    termination_reasons={k:sum(r["termination_reason"]==k for r in rows) for k in
                                         sorted(set(r["termination_reason"] for r in rows))})
    by_family=defaultdict(list)
    for r in results:by_family[r["family"]].append(r)
    report=dict(schema="starscream-real100-v2-dual-outcome-v2",miss_definition="ordered-reference-v1" if args.reference_aware else "raw-plane-v1",reference_tolerance=args.reference_tolerance,checkpoint=str(resolved),
                checkpoint_round=int(payload["round"]),checkpoint_steps=int(payload["environment_steps"]),
                protocol=str(args.protocol),episodes_per_course=args.episodes,courses=len(records),
                seed=args.seed,speed_command=speed_command,hard_cap_steps=args.max_steps,retry_limit_seconds=args.retry_steps/130,
                gate_dwell_limit_seconds=args.dwell_steps/130,
                aggregate=metrics(results),families={k:metrics(v) for k,v in by_family.items()},episodes=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    if args.save_states:
        np.savez_compressed(args.save_states,histories=np.asarray([x[0] for x in sampled],np.float32),slots=np.asarray([x[1] for x in sampled]),seeds=np.asarray([x[2] for x in sampled],np.int64))
    print(json.dumps(report["aggregate"],indent=2),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--reference-aware", action="store_true", help="Exempt locally planned reference crossings; use for new comparisons")
    p.add_argument("--reference-tolerance", type=float, default=.75)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--protocol",type=Path,default=ROOT/"configs/eval/real100_v2_timed_protocol_v1.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--episodes",type=int,default=4)
    p.add_argument("--seed",type=int,default=2034091462)
    p.add_argument("--speed-command",type=float,default=16.5,help="Actor command only; keep frozen environment and deadlines")
    p.add_argument("--trajectory-stride",type=int,default=0)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--device",default="cuda")
    p.add_argument("--max-steps",type=int,default=6000)
    p.add_argument("--retry-steps",type=int,default=390)
    p.add_argument("--dwell-steps",type=int,default=780)
    p.add_argument("--limit-courses",type=int,default=0)
    p.add_argument("--save-states",type=Path)
    p.add_argument("--max-states",type=int,default=20000)
    evaluate(p.parse_args())

