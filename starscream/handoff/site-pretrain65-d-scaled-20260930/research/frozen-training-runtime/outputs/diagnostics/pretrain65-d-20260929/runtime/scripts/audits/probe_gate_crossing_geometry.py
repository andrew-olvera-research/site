#!/usr/bin/env python3
"""Diagnostic: per-step reference deviation and gate-plane crossing geometry.

Same canonical rollout contract as eval_privileged_real100_v2_dual.py (matched
seeds, fixed-16 lanes, command 16.5, 6,000-step horizon, no retry/dwell stops).
Adds a per-step trace (position, velocity, contour/lag error to the reference
line, active gate, action) and, for each crossing of the active gate plane, the
hit point in the gate frame. Observation only; never alters the rollout.
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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

ROOT = Path(__file__).resolve().parents[2]


def main(args):
    torch.set_num_threads(1)
    config = stage_config(load_config(args.config), "dagger")
    settings = dict(config["dagger"])
    policy, normalizer, payload, resolved = load_policy_checkpoint(args.checkpoint, args.device)
    if args.train_manifest:
        manifest = json.loads(Path(args.train_manifest).read_text())
        records = [dict(path=r["path"], slot=r["name"], family=r.get("family", ""), deadline_steps=0,
                        speed=float(r["qualified_speed_mps"])) for r in manifest["records"]
                   if r.get("split") == args.split and r.get("qualified", True) and r.get("qualified_speed_mps")]
        records = records[:args.limit_courses or None]
    else:
        protocol = load_protocol(args.protocol)
        records = protocol["records"][:args.limit_courses or None]
    stage = replace(parse_stage(settings["evaluation_curriculum"]), target_gates=64, max_steps=args.max_steps,
                    random_gate=False, fixed_start_gate_index=0, rollout_laps=1)
    settings["evaluation_fixed_start_gate_index"] = 0
    policy.eval()
    jobs = []
    for record in records:
        offset = int(hashlib.sha256(record["slot"].encode()).hexdigest()[:8], 16) % 1_000_000
        jobs.append((record, args.seed + offset))
    traces, crossings, episodes = {}, [], []
    with canonical_inference():
        for start in range(0, len(jobs), args.workers):
            slots = []
            try:
                for lane, (record, seed) in enumerate(jobs[start:start + args.workers]):
                    env = make_env(settings, track=str(ROOT / record["path"]) if not str(record["path"]).startswith("/") else record["path"], reward=make_reward(settings, 16.5))
                    obs, start_passed = reset_env(env, stage, seed=seed, episode_index=0)
                    history = CausalHistory(policy.context_steps)
                    history.reset_feature(ppo_observation_features(obs, settings))
                    line, _, _, _ = make_gate_reference(env, settings)
                    events = ReferenceGateEvents(line, len(env.track.gates), obs["state"][:3], args.reference_tolerance)
                    slots.append(dict(lane=lane, env=env, record=record, seed=seed, obs=obs, history=history,
                                      line=line, events=events, start_passed=start_passed, steps=0, crashed=False,
                                      done=False, trace=[], misses=0))
                with ThreadPoolExecutor(max_workers=len(slots)) as pool:
                    while any(not s["done"] for s in slots):
                        active = [s for s in slots if not s["done"]]
                        batch = torch.from_numpy(normalizer.numpy(np.stack([s["history"].array() for s in active]))).to(args.device)
                        speeds = torch.as_tensor([s["record"].get("speed", 16.5) if args.qualified_speed else 16.5 for s in active], device=args.device, dtype=torch.float32)
                        actions = fixed_shape_call(policy, [batch, speeds], [s["lane"] for s in active], 16).float().cpu().numpy()
                        before = []
                        for s in active:
                            tr = s["env"].tracker
                            gate = s["env"].track.gates[tr.index]
                            before.append((gate, tr.passed_count, tr.index, s["obs"]["state"][:3].copy()))
                        futures = [pool.submit(s["env"].step, ppo_normalized_to_ctbr(a, settings)) for s, a in zip(active, actions)]
                        for s, a, (gate, passed, gidx, prev), fut in zip(active, actions, before, futures):
                            obs, _, terminated, truncated, info = fut.result()
                            s["steps"] += 1
                            s["obs"] = obs
                            s["history"].append_feature(ppo_observation_features(obs, settings))
                            s["crashed"] |= bool(info.get("ground_contact") or info.get("unity_collision"))
                            pos = np.asarray(obs["state"][:3], float)
                            vel = np.asarray(obs["state"][7:10] if len(obs["state"]) > 9 else np.zeros(3), float)
                            new_passed = s["env"].tracker.passed_count
                            miss = s["events"].advance(prev, pos, gate, gidx, new_passed > passed)
                            proj = s["line"].project(pos, hint_progress=s["events"].progress, search_radius=8.)
                            s["trace"].append([s["steps"], *pos, *vel, proj.distance, proj.lag_error, proj.progress,
                                               passed - s["start_passed"], *a])
                            a0 = (prev - gate.position) @ gate.normal
                            a1 = (pos - gate.position) @ gate.normal
                            if a0 < 0 <= a1:  # forward crossing of the active gate plane
                                frac = -a0 / max(a1 - a0, 1e-9)
                                hit = prev + frac * (pos - prev)
                                local = (hit - gate.position) @ gate.directed_rotation
                                # reference line's own crossing of this plane (nearest in progress)
                                ref_local = None
                                for q in np.linspace(proj.progress - 6, proj.progress + 6, 241):
                                    p = s["line"].evaluate(q)["position"]
                                    lp = (p - gate.position) @ gate.directed_rotation
                                    if ref_local is None or abs(lp[0]) < abs(ref_local[0]):
                                        ref_local = lp
                                crossings.append(dict(slot=s["record"]["slot"], family=s["record"]["family"],
                                    step=s["steps"], gate=int(passed - s["start_passed"]),
                                    kind="pass" if new_passed > passed else "miss" if miss else "planned_outside",
                                    prior_misses=s["misses"], half_width=float(gate.size[0] / 2),
                                    half_height=float(gate.size[1] / 2), hit_y=float(local[1]), hit_z=float(local[2]),
                                    ref_y=float(ref_local[1]), ref_z=float(ref_local[2]),
                                    speed=float(np.linalg.norm(pos - prev) * 130)))
                                s["misses"] += int(miss)
                            success = new_passed - s["start_passed"] >= len(s["env"].track.gates)
                            if success or s["crashed"] or terminated or truncated or s["steps"] >= args.max_steps:
                                traces[s["record"]["slot"]] = np.asarray(s["trace"], np.float32)
                                episodes.append(dict(slot=s["record"]["slot"], family=s["record"]["family"],
                                    success=bool(success), crashed=bool(s["crashed"]), steps=s["steps"],
                                    misses=s["misses"], deadline_steps=s["record"]["deadline_steps"]))
                                s["done"] = True
            finally:
                for s in slots:
                    s["env"].close()
            print(f"progress {len(episodes)}/{len(jobs)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "crossings.json").write_text(json.dumps(dict(checkpoint=str(resolved),
        round=int(payload["round"]), crossings=crossings, episodes=episodes), indent=1))
    np.savez_compressed(args.output / "traces.npz", **{k.replace("/", "_"): v for k, v in traces.items()})
    print(json.dumps(dict(success=np.mean([e["success"] for e in episodes]),
                          clean=np.mean([e["success"] and e["misses"] == 0 for e in episodes]))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=ROOT / "configs/eval/real100_v2_timed_protocol_v1.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2034091462)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--reference-tolerance", type=float, default=.75)
    p.add_argument("--limit-courses", type=int, default=0)
    p.add_argument("--train-manifest", type=Path, help="Evaluate the training courses of this manifest instead of the protocol")
    p.add_argument("--split", default="train")
    p.add_argument("--qualified-speed", action="store_true", help="Command each training course at its qualified speed")
    main(p.parse_args())
