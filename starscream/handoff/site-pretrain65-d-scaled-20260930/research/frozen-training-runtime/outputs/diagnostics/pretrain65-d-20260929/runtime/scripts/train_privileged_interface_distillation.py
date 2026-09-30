#!/usr/bin/env python3
"""Distil deployable observations through a frozen privileged racing policy."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_flight import ObservationHistory
from scripts.train_compact_observation_ablation import (
    compact_features, load_config, make_dataset, move,
)
from starscream.checkpoint_manager import CheckpointManager, capture_rng_state
from starscream.dataloader import stratified_episode_split
from starscream.env import FlightmareEnv
from starscream.privileged_distillation import (
    CausalPrivilegedStateAdapter, PrivilegedInterfacePolicy, task_to_physical,
)
from starscream.privileged_racing import (
    FeatureNormalizer, PrivilegedMLPPolicy, resolve_ranked_checkpoint,
)
from scripts.train_privileged_racing import (
    action_contract_metadata, episode_target_speed, parse_stage,
    ppo_normalized_to_ctbr,
)
from starscream.racing_curriculum import RacingCurriculumStage, sample_curriculum_spawn
from starscream.wandb import init_wandb


RAW_HISTORY = 36
POLICY_HISTORY = 18


def arguments() -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config = copy.deepcopy(config)
        config["distillation"].update(
            steps=2, batch_size=2, workers=0, validation_workers=0, cache_workers=0,
            validation_batches=1, validation_interval=1,
            evaluation_interval=1, evaluation_episodes=2,
            evaluation_workers=2, offline_max_windows=32,
            validation_max_windows=32,
        )
        config["distillation"]["evaluation"]["curriculum"]["max_steps"] = 18
        config["output_root"] = "/tmp/starscream-privileged-interface-smoke"
        config["checkpoint"] = {
            "run_name": "privileged-interface-smoke", "monitor": "selection_score",
            "mode": "max", "top_k": 1,
        }
        config["wandb"] = {"enabled": False, "run_name": "privileged-interface-smoke"}
    if args.disable_wandb:
        config["wandb"] = {"enabled": False}
    return args, config


def raw_record_feature(record: Mapping[str, np.ndarray]) -> np.ndarray:
    feature = np.concatenate([
        np.asarray(record["estimate"], np.float32),
        np.asarray(record["proprio"], np.float32)[:7],
        np.asarray(record["route"], np.float32).reshape(-1),
        np.asarray(record["previous_action"], np.float32),
        np.asarray(record["timing"], np.float32)[[3, 7]],
    ]).astype(np.float32)
    if feature.shape != (84,) or not np.all(np.isfinite(feature)):
        raise ValueError("interface feature must be finite and 84-dimensional")
    return feature


def raw_feature_sequence(history: ObservationHistory) -> np.ndarray:
    records = list(history.records)[-POLICY_HISTORY:]
    if not records:
        raise RuntimeError("empty deployment observation history")
    records = [records[0]] * (POLICY_HISTORY - len(records)) + records
    return np.stack([raw_record_feature(record) for record in records])


def exact_task_sequence(history: ObservationHistory) -> np.ndarray:
    records = list(history.records)[-POLICY_HISTORY:]
    if not records:
        raise RuntimeError("empty deployment observation history")
    records = [records[0]] * (POLICY_HISTORY - len(records)) + records
    return np.stack([
        np.asarray(record["deployable_task_state"], np.float32) for record in records
    ])


def build_policy(
    settings: Mapping[str, Any], device: str,
) -> tuple[PrivilegedInterfacePolicy, Path, Path]:
    teacher_path = resolve_ranked_checkpoint(settings["teacher_checkpoint"])
    raw_path = resolve_ranked_checkpoint(settings["raw_checkpoint"])
    teacher_state = torch.load(teacher_path, map_location="cpu", weights_only=False)
    raw_state = torch.load(raw_path, map_location="cpu", weights_only=False)
    if int(teacher_state.get("feature_dim", 64)) != 64:
        raise ValueError("privileged teacher does not use the 64-value contract")
    if raw_state.get("observation_source") != "raw_estimate":
        raise ValueError("raw checkpoint must use deployment estimates")
    teacher = PrivilegedMLPPolicy(**dict(teacher_state["model_config"]))
    teacher.load_state_dict(teacher_state["model"])
    teacher_normalizer = FeatureNormalizer.from_state_dict(teacher_state["normalizer"])
    legacy_normalizer = FeatureNormalizer.from_state_dict(raw_state["normalizer"])
    if len(legacy_normalizer.mean) != 77:
        raise ValueError("raw checkpoint normalizer must use the 77-value ablation contract")
    # Insert directly normalized measured body rates/motor speeds after the
    # 32D estimate while preserving all established normalization statistics.
    raw_normalizer = FeatureNormalizer(
        np.concatenate([legacy_normalizer.mean[:32], np.zeros(7,np.float32), legacy_normalizer.mean[32:]]),
        np.concatenate([legacy_normalizer.std[:32], np.ones(7,np.float32), legacy_normalizer.std[32:]]),
    )
    adapter_config = dict(settings["adapter"])
    adapter = CausalPrivilegedStateAdapter(
        **adapter_config, raw_normalizer=raw_normalizer
    )
    training = teacher_state.get("training_config", {})
    stage_name = str(teacher_state.get("stage", "ppo"))
    teacher_settings = training.get(stage_name, training.get("ppo", training.get("dagger", {})))
    action_contract = action_contract_metadata(
        teacher_settings if isinstance(teacher_settings, Mapping) else {}
    )
    configured_contract = action_contract_metadata(settings)
    if configured_contract != action_contract:
        raise ValueError(
            "distillation action contract does not match frozen teacher: "
            f"configured={configured_contract}, teacher={action_contract}"
        )
    policy = PrivilegedInterfacePolicy(
        adapter, teacher, teacher_normalizer, action_contract=action_contract
    ).to(device)
    return policy, teacher_path, raw_path


def make_datasets(settings: Mapping[str, Any]) -> tuple[Any, Any]:
    train_paths, validation_paths = stratified_episode_split(
        settings["data"], float(settings.get("validation_fraction", 0.12)),
        int(settings.get("validation_seed", 2026081501)), tracks=settings["track"],
    )
    dataset_settings = dict(settings)
    dataset_settings.update(
        observation_source="raw_estimate", raw_history=RAW_HISTORY,
        policy_history=POLICY_HISTORY,
    )
    train_dataset = make_dataset(dataset_settings, train_paths, None)
    validation_dataset = make_dataset(dataset_settings, validation_paths, None)
    def sampled_view(dataset: Any, maximum: int, seed: int) -> Any:
        if maximum <= 0 or len(dataset) <= maximum:
            return dataset
        generator = np.random.default_rng(seed)
        # Sorting retains near-sequential HDF5 reads while removing the old
        # first-file/early-trajectory bias.
        indices = np.sort(generator.choice(len(dataset), maximum, replace=False))
        return Subset(dataset, indices.tolist())

    maximum = int(settings.get("offline_max_windows", 0))
    validation_maximum = int(settings.get("validation_max_windows", maximum))
    subset_seed = int(settings.get("offline_subset_seed", settings.get("seed", 0)))
    train_view = sampled_view(train_dataset, maximum, subset_seed)
    validation_view = sampled_view(
        validation_dataset, validation_maximum, subset_seed + 1
    )
    return (train_dataset, train_view), (validation_dataset, validation_view)


def paired_batch(
    raw: Mapping[str, torch.Tensor], device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    batch = move(raw, device)
    compact, _, exact_task = compact_features(
        batch, source="raw_estimate", policy_history=POLICY_HISTORY,
        raw_history=RAW_HISTORY, encoder=None, amp=True, encoder_grad=False,
    )
    start=RAW_HISTORY-POLICY_HISTORY; stop=RAW_HISTORY
    features=torch.cat(
        [compact[...,:32],batch["proprio"][:,start:stop,:7].float(),compact[...,32:]],dim=-1
    )
    return batch, features.float(), exact_task.float()


def distillation_loss(
    policy: PrivilegedInterfacePolicy, gate: torch.Tensor,
    raw_features: torch.Tensor, exact_task: torch.Tensor,
    settings: Mapping[str, Any],
    speed_command: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = policy.paired_forward(raw_features, exact_task, speed_command)
    adapter = output["adapter"]
    predicted_task = adapter.task
    error = predicted_task - exact_task
    action = F.smooth_l1_loss(
        output["estimated_action"], output["exact_action"],
        beta=float(settings.get("action_huber_beta", 0.03)),
    )
    state = F.smooth_l1_loss(
        predicted_task, exact_task, beta=float(settings.get("state_huber_beta", 0.02))
    )
    temporal_valid = gate[:, 1:] == gate[:, :-1]
    predicted_delta = predicted_task[:, 1:] - predicted_task[:, :-1]
    exact_delta = exact_task[:, 1:] - exact_task[:, :-1]
    state_delta = (
        F.smooth_l1_loss(
            predicted_delta[temporal_valid], exact_delta[temporal_valid], beta=0.01
        ) if temporal_valid.any() else predicted_delta.sum() * 0.0
    )
    latent = 1.0 - F.cosine_similarity(
        output["estimated_latent"], output["exact_latent"], dim=-1
    ).mean()
    dynamics = F.smooth_l1_loss(
        output["estimated_dynamics"], output["exact_dynamics"], beta=0.02
    )
    log_scale = adapter.log_scale
    uncertainty = (0.5 * error.square() * torch.exp(-2.0 * log_scale) + log_scale).mean()
    correction = adapter.correction.square().mean()
    total = (
        float(settings.get("action_weight", 1.0)) * action
        + float(settings.get("state_weight", 0.50)) * state
        + float(settings.get("state_delta_weight", 0.20)) * state_delta
        + float(settings.get("latent_weight", 0.10)) * latent
        + float(settings.get("dynamics_weight", 0.10)) * dynamics
        + float(settings.get("uncertainty_weight", 0.01)) * uncertainty
        + float(settings.get("correction_weight", 0.002)) * correction
    )
    physical_error = task_to_physical(error.detach()).abs()
    return total, {
        "loss": total, "action_loss": action, "state_loss": state,
        "state_delta_loss": state_delta, "latent_loss": latent,
        "dynamics_loss": dynamics, "uncertainty_loss": uncertainty,
        "correction_loss": correction,
        "action_rmse": (output["estimated_action"]-output["exact_action"]).square().mean().sqrt(),
        "position_mae_m": physical_error[..., :3].mean(),
        "velocity_mae_mps": physical_error[..., 3:6].mean(),
        "attitude_mae": physical_error[..., 6:12].mean(),
        "body_rate_mae_radps": physical_error[..., 12:15].mean(),
        "motor_mae_radps": physical_error[..., 15:19].mean(),
        "gate_consistent_fraction": temporal_valid.float().mean(),
    }


def _cache_identity(settings: Mapping[str, Any], split: str, view: Any) -> str:
    indices = getattr(view, "indices", None)
    payload = {
        "version": 2, "split": split, "data": str(settings["data"]),
        "track": settings["track"], "raw_history": RAW_HISTORY,
        "policy_history": POLICY_HISTORY,
        "deployment_estimate_seed": int(settings.get("deployment_estimate_seed", 0)),
        "validation_seed": int(settings.get("validation_seed", 0)),
        "validation_fraction": float(settings.get("validation_fraction", 0.0)),
        "offline_max_windows": int(settings.get("offline_max_windows", 0)),
        "validation_max_windows": int(settings.get("validation_max_windows", 0)),
        "offline_subset_seed": int(settings.get("offline_subset_seed", settings.get("seed", 0))),
        "indices_hash": (
            hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest()[:16]
            if indices is not None else "full"
        ),
        "length": len(view),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def materialize_pairs(
    settings: Mapping[str, Any], split: str, view: Any, output_root: str,
) -> dict[str, torch.Tensor]:
    """Build the expensive overlapping HDF5 windows once and reuse them."""

    cache_dir = Path(output_root) / "cache" / "privileged-interface-distillation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}-{_cache_identity(settings, split, view)}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"loaded {split} paired cache: {len(payload['features'])} windows from {cache_path}", flush=True)
        return payload
    workers = int(settings.get("cache_workers", settings.get("workers", 8)))
    loader = DataLoader(
        view, batch_size=int(settings.get("cache_batch_size", 256)), shuffle=False,
        num_workers=workers, persistent_workers=False, prefetch_factor=2 if workers else None,
    )
    features: list[torch.Tensor] = []
    exact: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    started = time.perf_counter()
    for index, raw in enumerate(loader):
        batch, raw_features, exact_task = paired_batch(raw, "cpu")
        features.append(raw_features.to(torch.float16))
        exact.append(exact_task.to(torch.float16))
        gates.append(
            batch["gate_index"][:, RAW_HISTORY-POLICY_HISTORY:RAW_HISTORY, 0]
            .to(torch.int16)
        )
        if (index + 1) % 25 == 0:
            count = sum(len(item) for item in features)
            print(f"building {split} cache: {count}/{len(view)} windows", flush=True)
    payload = {
        "features": torch.cat(features), "exact_task": torch.cat(exact),
        "gate": torch.cat(gates),
    }
    temporary = cache_path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    elapsed = time.perf_counter() - started
    print(f"built {split} paired cache: {len(view)} windows in {elapsed:.1f}s at {cache_path}", flush=True)
    return payload


def cache_to_device(cache: Mapping[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in cache.items()}


def select_batch(
    cache: Mapping[str, torch.Tensor], indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        cache["gate"].index_select(0, indices),
        cache["features"].index_select(0, indices).float(),
        cache["exact_task"].index_select(0, indices).float(),
    )


@torch.no_grad()
def validate(
    policy: PrivilegedInterfacePolicy, cache: Mapping[str, torch.Tensor],
    settings: Mapping[str, Any], device: str,
) -> dict[str, float]:
    policy.eval(); totals: dict[str, list[float]] = {}
    batch_size = int(settings.get("batch_size", 1024))
    maximum = min(len(cache["features"]), batch_size * int(settings.get("validation_batches", 32)))
    speed_value = float(settings.get(
        "validation_speed_command", policy.teacher.default_speed_command
    ))
    for start in range(0, maximum, batch_size):
        indices = torch.arange(start, min(start + batch_size, maximum), device=device)
        gate, features, exact = select_batch(cache, indices)
        speed = torch.full((len(indices),), speed_value, device=device)
        _, metrics = distillation_loss(
            policy, gate, features, exact, settings, speed
        )
        for key, value in metrics.items():
            totals.setdefault(key, []).append(float(value))
    return {key: float(np.mean(value)) for key, value in totals.items()}


def _eval_worker(
    connection: Any, settings: dict[str, Any], stage_raw: dict[str, Any], index: int,
) -> None:
    env: FlightmareEnv | None = None
    try:
        stage = RacingCurriculumStage.from_mapping(stage_raw)
        env = FlightmareEnv(
            track=stage.tracks[index % len(stage.tracks)], next_gates=3,
            image_size=(160, 128), control_dt=1.0/90.0,
            render_observations=False, mask_source="geometry", mask_size=(160, 128),
            geometry_renderer=str(settings.get("geometry_renderer", "exact")),
            image_delay=float(settings.get("image_delay", 0.033)),
            action_delay=stage.action_delay, terminate_on_collision=True,
            dynamics_randomization=settings.get("dynamics_randomization"),
        )
        history: ObservationHistory | None = None
        start_passed = steps = 0
        speed_sum = maximum_speed = gate_speed_sum = 0.0
        gate_speed_count = 0
        collective_command_sum = maximum_collective_command = 0.0
        collective_saturation_steps = 0
        maximum_motor_utilization = 0.0
        target_speed = float(stage.target_speed)
        target_gates = min(stage.target_gates, len(env.track.gates)) * max(
            int(stage.rollout_laps), 1
        )
        connection.send(("ready", index))
        while True:
            request = connection.recv()
            if request[0] == "close": connection.send(("closed", index)); return
            if request[0] == "reset":
                episode, seed = int(request[1]), int(request[2])
                target_speed = episode_target_speed(stage, seed)
                spawn = sample_curriculum_spawn(env.track, stage, seed=seed, episode_index=episode)
                options: dict[str, Any] = {
                    "gate_index": spawn.gate_index, "state": spawn.state,
                    "spawn": dict(spawn.metadata),
                }
                if spawn.previous_action is not None: options["previous_action"] = spawn.previous_action
                observation, _ = env.reset(seed=seed, options=options)
                history = ObservationHistory(RAW_HISTORY, (128,160), estimate_dim=32, estimate_seed=seed)
                history.append(observation); start_passed = env.tracker.passed_count; steps = 0
                speed_sum = maximum_speed = gate_speed_sum = 0.0
                gate_speed_count = 0
                collective_command_sum = maximum_collective_command = 0.0
                collective_saturation_steps = 0
                maximum_motor_utilization = 0.0
                connection.send((
                    "reset", raw_feature_sequence(history), exact_task_sequence(history),
                    target_speed,
                ))
                continue
            if request[0] != "advance" or history is None: raise RuntimeError("invalid eval operation")
            normalized_action = np.asarray(request[1], np.float32)
            physical_action = ppo_normalized_to_ctbr(normalized_action, settings)
            observation, _, terminated, _, info = env.step(physical_action)
            history.append(observation); steps += 1
            state = np.asarray(observation["state"], np.float32)
            speed = float(np.linalg.norm(state[7:10]))
            speed_sum += speed
            maximum_speed = max(maximum_speed, speed)
            collective_command_sum += float(physical_action[0])
            maximum_collective_command = max(
                maximum_collective_command, float(physical_action[0])
            )
            collective_saturation_steps += int(abs(float(normalized_action[0])) >= 0.95)
            maximum_motor_utilization = max(
                maximum_motor_utilization,
                float(np.max(info.get("applied_motor_normalized", 0.0))),
            )
            if bool(info.get("gate_passed", False)):
                forward_speed = float(
                    dict(info.get("reward_components", {})).get("forward_speed", speed)
                )
                gate_speed_sum += forward_speed
                gate_speed_count += 1
            gates = env.tracker.passed_count - start_passed
            success = gates >= target_gates
            crash = bool(info.get("ground_contact") or info.get("unity_collision") or (terminated and not success))
            done = bool(success or crash or steps >= stage.max_steps)
            metrics = None if not done else {
                "success": float(success), "p1": float(gates>=1), "p2": float(gates>=2),
                "p3": float(gates>=3), "gates_passed": float(gates), "crash": float(crash),
                "steps": float(steps), "successful_lap_time": steps/90.0 if success else float("nan"),
                "mean_speed_mps": speed_sum/max(steps, 1),
                "maximum_speed_mps": maximum_speed,
                "mean_gate_speed_mps": gate_speed_sum/max(gate_speed_count, 1),
                "mean_collective_command_mps2": collective_command_sum/max(steps, 1),
                "maximum_collective_command_mps2": maximum_collective_command,
                "collective_saturation_fraction": collective_saturation_steps/max(steps, 1),
                "maximum_motor_utilization": maximum_motor_utilization,
            }
            connection.send((
                "step", raw_feature_sequence(history), exact_task_sequence(history),
                done, metrics,
            ))
    except BaseException:
        try: connection.send(("error", traceback.format_exc()))
        except BaseException: pass
        raise
    finally:
        if env is not None: env.close()
        connection.close()


@torch.no_grad()
def evaluate_closed_loop(
    policy: PrivilegedInterfacePolicy, settings: Mapping[str, Any], device: str,
    *, exact_teacher: bool = False,
) -> dict[str, float]:
    evaluation = settings["evaluation"]
    stage = parse_stage(evaluation["curriculum"])
    episodes = int(settings.get("evaluation_episodes", 96))
    workers = min(int(settings.get("evaluation_workers", 12)), episodes)
    context = mp.get_context("spawn"); slots = []
    stage_raw = asdict(stage); stage_raw["tracks"] = list(stage.tracks)
    worker_settings = dict(evaluation)
    for key in (
        "collective_action_mapping", "collective_action_maximum",
        "collective_action_reference_thrust", "collective_action_logit_scale",
    ):
        if key in settings:
            worker_settings[key] = settings[key]
    for index in range(workers):
        parent, child = context.Pipe()
        process = context.Process(
            target=_eval_worker,
            args=(child, worker_settings, stage_raw, index), daemon=True,
        )
        process.start(); child.close()
        response = parent.recv()
        if response[0] != "ready": raise RuntimeError(response)
        slots.append({
            "connection":parent,"process":process,"done":True,
            "feature":None,"exact":None,"speed":float(stage.target_speed),
        })
    scheduled=completed=0; rows=[]; policy.eval()
    disagreement_squared=disagreement_count=disagreement_max=0.0
    try:
        for slot in slots:
            seed = int(evaluation.get("seed",2026081850)) + 7919*scheduled
            slot["connection"].send(("reset",scheduled,seed)); response=slot["connection"].recv()
            slot["feature"],slot["exact"],slot["speed"]=response[1],response[2],float(response[3])
            slot["done"]=False; scheduled+=1
        while completed < episodes:
            active=[slot for slot in slots if not slot["done"]]
            tensor=torch.from_numpy(np.stack([slot["feature"] for slot in active])).to(device)
            exact=torch.from_numpy(np.stack([slot["exact"] for slot in active])).to(device)
            speed=torch.as_tensor(
                [slot["speed"] for slot in active], device=device, dtype=tensor.dtype
            )
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.startswith("cuda")):
                student_actions=policy(tensor, speed).float()
                exact_input=policy.exact_teacher_input(tensor,exact)
                teacher_actions=policy.teacher(exact_input, speed).float()
            difference=(student_actions-teacher_actions).float()
            disagreement_squared+=float(difference.square().sum())
            disagreement_count+=difference.numel()
            disagreement_max=max(disagreement_max,float(difference.abs().max()))
            actions=(teacher_actions if exact_teacher else student_actions).cpu().numpy()
            for slot,action in zip(active,actions): slot["connection"].send(("advance",action))
            for slot in active:
                response=slot["connection"].recv()
                if response[0]=="error": raise RuntimeError(response[1])
                slot["feature"],slot["exact"],slot["done"]=response[1],response[2],bool(response[3])
                if slot["done"]:
                    rows.append(response[4]); completed+=1
                    if scheduled < episodes:
                        seed=int(evaluation.get("seed",2026081850))+7919*scheduled
                        slot["connection"].send(("reset",scheduled,seed)); reset=slot["connection"].recv()
                        slot["feature"],slot["exact"],slot["speed"],slot["done"]=reset[1],reset[2],float(reset[3]),False
                        scheduled+=1
        output={}
        for key in rows[0]:
            values=np.asarray([row[key] for row in rows],np.float64)
            output[key]=float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        output["selection_score"]=(
            10.0*output["success"]+2.0*output["p3"]+0.75*output["p2"]
            +0.1*output["p1"]-0.5*output["crash"]
        )
        output["action_rmse_to_exact_teacher"]=(
            disagreement_squared/max(disagreement_count,1)
        )**0.5
        output["action_max_abs_to_exact_teacher"]=disagreement_max
        output["policy_source"]="exact_teacher" if exact_teacher else "distilled_student"
        return output
    finally:
        for slot in slots:
            if slot["process"].is_alive():
                try: slot["connection"].send(("close",))
                except (BrokenPipeError,EOFError): pass
        for slot in slots:
            slot["process"].join(timeout=5)
            if slot["process"].is_alive(): slot["process"].terminate(); slot["process"].join(timeout=2)
            slot["connection"].close()


def checkpoint_payload(
    policy: PrivilegedInterfacePolicy, optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any], teacher_path: Path, raw_path: Path, step: int,
) -> dict[str, Any]:
    payload=policy.checkpoint_payload()
    payload.update(
        optimizer=optimizer.state_dict(), training_config=dict(config),
        teacher_checkpoint=str(teacher_path), raw_checkpoint=str(raw_path),
        step=step, rng_state=capture_rng_state(),
        action_contract=dict(policy.action_contract),
    )
    return payload


def run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    settings=config["distillation"]; seed=int(settings.get("seed",2026081825))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if args.device.startswith("cuda"): torch.cuda.manual_seed_all(seed)
    policy,teacher_path,raw_path=build_policy(settings,args.device)
    (train_dataset,train_view),(validation_dataset,validation_view)=make_datasets(settings)
    train_cache=materialize_pairs(settings,"train",train_view,config["output_root"])
    validation_cache=materialize_pairs(settings,"validation",validation_view,config["output_root"])
    cache_device=(
        str(settings.get("cache_device",args.device))
        if args.device.startswith("cuda") else args.device
    )
    train_cache=cache_to_device(train_cache,cache_device)
    validation_cache=cache_to_device(validation_cache,cache_device)
    optimizer=torch.optim.AdamW(
        policy.adapter.parameters(),lr=float(settings.get("learning_rate",2e-4)),
        weight_decay=float(settings.get("weight_decay",1e-5)),fused=args.device.startswith("cuda"),
    )
    manager=CheckpointManager.from_config(config); logger=init_wandb(config)
    steps=int(settings.get("steps",5000)); batch_size=int(settings.get("batch_size",1024))
    train_started=time.perf_counter(); interval_started=train_started
    try:
        speed_interval = settings.get("speed_command_range", (
            policy.teacher.default_speed_command,
            policy.teacher.default_speed_command,
        ))
        speed_low, speed_high = (float(speed_interval[0]), float(speed_interval[1]))
        if speed_low < 0.0 or speed_high < speed_low:
            raise ValueError("invalid distillation speed_command_range")
        for step in range(1,steps+1):
            indices=torch.randint(len(train_cache["features"]),(batch_size,),device=cache_device)
            gate,features,exact=select_batch(train_cache,indices)
            speed = torch.empty(batch_size, device=cache_device).uniform_(
                speed_low, speed_high
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=args.device.startswith("cuda") and bool(settings.get("amp",True))):
                loss,metrics=distillation_loss(
                    policy, gate, features, exact, settings, speed
                )
            loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(policy.adapter.parameters(),float(settings.get("gradient_clip",2.0)))
            optimizer.step(); metrics["gradient_norm"]=gradient
            if step % int(settings.get("log_interval",25)) == 0:
                now=time.perf_counter(); interval=int(settings.get("log_interval",25))
                throughput=interval*batch_size/max(now-interval_started,1e-9); interval_started=now
                train_metrics={key:float(value.detach()) for key,value in metrics.items()}
                train_metrics.update(samples_per_second=throughput,elapsed_seconds=now-train_started)
                if args.device.startswith("cuda"):
                    train_metrics.update(
                        gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
                        gpu_reserved_gib=torch.cuda.memory_reserved()/2**30,
                    )
                logger.log_train(train_metrics,step)
                print(
                    f"step={step} loss={train_metrics['loss']:.4f} "
                    f"samples_s={throughput:.0f} gpu_reserved_gib={train_metrics.get('gpu_reserved_gib',0):.2f}",
                    flush=True,
                )
            if step % int(settings.get("validation_interval",250)) == 0 or step==steps:
                validation=validate(policy,validation_cache,settings,args.device)
                eval_metrics={f"offline_{key}":value for key,value in validation.items()}
                if step % int(settings.get("evaluation_interval",500)) == 0 or step==steps:
                    eval_started=time.perf_counter()
                    closed=evaluate_closed_loop(policy,settings,args.device); eval_metrics.update(closed)
                    eval_metrics["evaluation_seconds"]=time.perf_counter()-eval_started
                else:
                    # Closed-loop ranking is mandatory. Intermediate offline-only
                    # snapshots are intentionally not checkpoint candidates.
                    closed=None
                logger.log_eval(eval_metrics,step)
                if closed is not None:
                    manager.save_eval(eval_metrics,step=step,summary="Frozen privileged-interface distillation evaluation.",kind="closed-loop")
                    manager.save(checkpoint_payload(policy,optimizer,config,teacher_path,raw_path,step),step=step,metrics=eval_metrics)
                print(
                    f"step={step} offline_action_rmse={validation['action_rmse']:.4f} "
                    f"position_mae_m={validation['position_mae_m']:.3f} "
                    f"p1={eval_metrics.get('p1',float('nan')):.3f} "
                    f"p2={eval_metrics.get('p2',float('nan')):.3f} "
                    f"p3={eval_metrics.get('p3',float('nan')):.3f} "
                    f"full={eval_metrics.get('success',float('nan')):.3f}",flush=True,
                )
    finally:
        train_dataset.close(); validation_dataset.close(); logger.finish()


if __name__=="__main__":
    parsed,configuration=arguments(); run(parsed,configuration)
