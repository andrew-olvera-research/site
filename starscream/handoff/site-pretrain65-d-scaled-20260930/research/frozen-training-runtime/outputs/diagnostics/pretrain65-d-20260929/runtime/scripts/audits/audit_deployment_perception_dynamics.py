#!/usr/bin/env python3
"""Audit the frozen deployment representation at control-critical phases.

The report combines four checks that aggregate tokenizer validation hides:
offline/online preprocessing parity, delayed-mask pixel replay, phase-stratified
state/action accuracy, and modality/temporal ablations of the frozen stack.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import torch
import yaml

# Direct execution places scripts/audits rather than the repository root on
# sys.path; add the root so the audited production modules are imported exactly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_flight import ObservationHistory, ctbr_to_normalized, normalized_to_ctbr
from scripts.train_actor_bc import load_frozen_actor_stack
from starscream.actor_critic import GaussianActor, build_actor_state
from starscream.dataloader import (
    GATE_PHASE_NAMES, DreamerSequenceDataset, stratified_episode_split,
)
from starscream.env import FlightmareEnv
from starscream.env.types import Proprioception
from starscream.racing_curriculum import RacingCurriculumStage, sample_curriculum_spawn
from scripts.train_flow_policy_fpo import make_mpcc


STATE_SCALE = np.asarray(
    [20.0] * 3 + [30.0] * 3 + [1.0] * 6 + [6.0] * 3 + [4000.0] * 4,
    np.float32,
)
STATE_GROUPS = {
    "position": slice(0, 3), "velocity": slice(3, 6),
    "attitude_6d": slice(6, 12), "body_rates": slice(12, 15),
    "motor_omega": slice(15, 19),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", default="figure8")
    parser.add_argument("--history", type=int, default=18)
    parser.add_argument("--samples-per-phase", type=int, default=96)
    parser.add_argument("--mask-replay-samples", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026081705)
    parser.add_argument("--dagger-config", type=Path)
    parser.add_argument("--online-episodes", type=int, default=0)
    parser.add_argument("--online-max-steps", type=int, default=850)
    parser.add_argument("--online-execution", choices=("student", "mpcc"), default="student")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def normalized_state(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, np.float32) / STATE_SCALE


def normalize_action(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float32)
    return np.concatenate((value[..., :1] / 15.0 - 1.0, value[..., 1:] / 6.0), -1).clip(-1, 1)


def phase_indices(dataset: DreamerSequenceDataset, history: int, count: int, seed: int):
    pools: dict[int, list[int]] = defaultdict(list)
    for index, (path, start) in enumerate(dataset.windows):
        target = start + history - 1
        pools[int(dataset.episode_gate_phases[path][target])].append(index)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for phase in range(len(GATE_PHASE_NAMES)):
        pool = np.asarray(pools[phase], np.int64)
        if len(pool):
            selected.extend(rng.choice(pool, min(count, len(pool)), replace=False).tolist())
    rng.shuffle(selected)
    return selected


def batch_samples(dataset: DreamerSequenceDataset, indices: list[int]) -> dict[str, torch.Tensor]:
    samples = [dataset[index] for index in indices]
    keys = set.intersection(*(set(sample) for sample in samples))
    return {
        key: torch.stack([sample[key] for sample in samples])
        for key in keys if isinstance(samples[0][key], torch.Tensor)
    }


def ablated(batch: dict[str, torch.Tensor], encoder, kind: str) -> dict[str, torch.Tensor]:
    result = {key: value.clone() for key, value in batch.items()}
    history = result["mask"].shape[1] - 1
    sequence_keys = ["mask", "proprio", "route", "timing", "previous_action"]
    if "estimate" in result:
        sequence_keys.append("estimate")
    for key in sequence_keys:
        result[key] = result[key][:, :history]
    if kind == "zero_mask":
        result["mask"].zero_()
    elif kind == "center_proprio":
        result["proprio"][:] = encoder.proprio_center.detach().cpu()
    elif kind == "center_route":
        result["route"][:] = encoder.route_center.detach().cpu()
    elif kind == "center_estimate":
        if "estimate" in result:
            result["estimate"][:] = encoder.estimate_center.detach().cpu()
    elif kind == "repeat_current":
        for key in sequence_keys:
            result[key] = result[key][:, -1:].expand_as(result[key]).clone()
    elif kind == "reverse_prefix":
        order = torch.cat((torch.arange(history - 2, -1, -1), torch.tensor([history - 1])))
        for key in sequence_keys:
            result[key] = result[key][:, order]
    elif kind != "full":
        raise ValueError(kind)
    return result


@torch.no_grad()
def representation_report(actor, encoder, action_tokenizer, batch, device: str, history: int):
    phase = batch["gate_phase"][:, history - 1].numpy()
    target_state = batch["task_state"][:, history - 1].numpy()
    target_action = batch["commanded_action"][:, history - 1].numpy()
    reports: dict[str, Any] = {}
    baseline_action = None
    kinds = ["full", "zero_mask", "center_proprio", "center_route"]
    if "estimate" in batch:
        kinds.append("center_estimate")
    kinds.extend(("repeat_current", "reverse_prefix"))
    for kind in kinds:
        current = ablated(batch, encoder, kind)
        current = {key: value.to(device) for key, value in current.items()}
        features = build_actor_state(encoder, current, history, device, True)
        output = actor(features, action_tokenizer)
        encoder_inputs = {
            key: current[key]
            for key in ("mask", "proprio", "route", "timing", "estimate")
            if key in current
        }
        latents = encoder.encode(encoder_inputs)
        state_mean, _ = encoder.estimate_state(latents)
        predicted_state = state_mean[:, -1].float().cpu().numpy()
        predicted_action = output.mean[:, 0].float().cpu().numpy()
        if baseline_action is None:
            baseline_action = predicted_action.copy()
        rows = {}
        for index, name in enumerate(GATE_PHASE_NAMES):
            chosen = phase == index
            if not chosen.any():
                continue
            physical_error = np.abs(predicted_state[chosen] - target_state[chosen]) * STATE_SCALE
            rows[name] = {
                "count": int(chosen.sum()),
                "state_mae_by_group": {
                    group: float(physical_error[:, section].mean())
                    for group, section in STATE_GROUPS.items()
                },
                "action_rmse": float(np.sqrt(np.mean((predicted_action[chosen] - target_action[chosen]) ** 2))),
                "action_shift_from_full": float(np.sqrt(np.mean((predicted_action[chosen] - baseline_action[chosen]) ** 2))),
            }
        reports[kind] = rows
    return reports


@torch.no_grad()
def dynamics_report(encoder, batch, device: str, history: int):
    current = {
        key: value[:, :history].to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    latents = encoder.encode({
        key: current[key]
        for key in ("mask", "proprio", "route", "timing", "estimate")
        if key in current
    })
    decoded = encoder.decode(
        latents,
        normalized_action=current["normalized_action"],
        action_timing=current["action_timing"],
    )
    if decoded.action_state_delta_mean is None or decoded.state_delta_mean is None:
        raise RuntimeError("audit requires both tokenizer dynamics heads")
    target = current["task_state"][:, history - 1] - current["task_state"][:, history - 2]
    action_prediction = decoded.action_state_delta_mean[:, history - 2]
    observed_prediction = decoded.state_delta_mean[:, history - 1]
    phase = current["gate_phase"][:, history - 2].cpu().numpy()
    result = {}
    for index, name in enumerate(GATE_PHASE_NAMES):
        chosen = phase == index
        if not chosen.any():
            continue
        rows = {}
        for prediction_name, prediction in (
            ("action_conditioned", action_prediction),
            ("observed_delta", observed_prediction),
        ):
            physical = (
                prediction[chosen].float().cpu().numpy()
                - target[chosen].float().cpu().numpy()
            ) * STATE_SCALE
            rows[prediction_name] = {
                group: float(np.abs(physical[:, section]).mean())
                for group, section in STATE_GROUPS.items()
            }
        result[name] = {"count": int(chosen.sum()), **rows}
    return result


def sequence_and_delta_report(dataset: DreamerSequenceDataset, indices: list[int], history: int):
    by_phase: dict[str, list[np.ndarray]] = defaultdict(list)
    for index in indices:
        path, start = dataset.windows[index]
        with h5py.File(path, "r", swmr=True) as episode:
            target = start + history - 1
            previous = normalized_state(episode["observation/task_state"][target])
            following = normalized_state(episode["observation/task_state"][target + 1])
            phase = int(dataset.episode_gate_phases[path][target])
            by_phase[GATE_PHASE_NAMES[phase]].append(np.abs(following - previous) * STATE_SCALE)
    return {
        phase: {
            "count": len(values),
            "absolute_delta_by_group": {
                group: float(np.asarray(values)[:, section].mean())
                for group, section in STATE_GROUPS.items()
            },
        }
        for phase, values in by_phase.items()
    }


def raw_observation(episode, index: int) -> dict[str, Any]:
    def field(key):
        return np.asarray(episode[key][index])
    return {
        "gate_mask": field("observation/gate_mask"),
        "measured": {
            "body_rates": field("observation/measured/body_rates"),
            "motor_omega": field("observation/measured/motor_omega"),
        },
        "previous_action": field("observation/previous_action"),
        "flight_plan": {
            "records": field("observation/flight_plan/records"),
            "index": field("observation/flight_plan/index"),
        },
        "timestamp": {
            name: field(f"observation/timestamp/{name}")
            for name in ("sim", "camera")
        },
        "age": {
            name: field(f"observation/age/{name}")
            for name in ("camera", "body_rates", "motor_omega", "previous_action")
        },
        "valid": {
            name: field(f"observation/valid/{name}")
            for name in ("camera", "body_rates", "motor_omega", "previous_action")
        },
        "task_state": field("observation/task_state"),
    }


def preprocessing_parity(dataset: DreamerSequenceDataset, index: int, history: int):
    path, start = dataset.windows[index]
    expected = dataset[index]
    with h5py.File(path, "r", swmr=True) as episode:
        online = ObservationHistory(history, tuple(expected["mask"].shape[-2:]))
        if start:
            online.previous_camera_time = float(episode["observation/timestamp/camera"][start - 1])
            online.previous_sim_time = float(episode["observation/timestamp/sim"][start - 1])
        for offset in range(history):
            online.append(raw_observation(episode, start + offset))
        actual = online.batch("cpu")
    keys = ("mask", "proprio", "route", "timing", "previous_action", "deployable_task_state", "gate_index")
    comparison = {}
    for key in keys:
        reference_key = "task_state" if key == "deployable_task_state" else key
        reference = expected[reference_key][:history].unsqueeze(0)
        difference = (actual[key].float() - reference.float()).abs()
        comparison[key] = {"maximum_absolute_error": float(difference.max()), "exact": bool(torch.equal(actual[key], reference))}
    return {"episode": str(path), "start": start, "channels": comparison}


def mask_replay(paths: list[Path], track: str, samples: int, seed: int):
    candidates = []
    for path in paths:
        with h5py.File(path, "r", swmr=True) as episode:
            candidates.extend((path, index) for index in range(len(episode["action/ctbr"])))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), min(samples, len(candidates)), replace=False)
    env = FlightmareEnv(track=track, image_size=(320, 240), mask_size=(160, 128), mask_source="geometry", geometry_renderer="exact")
    metrics: dict[str, list[float]] = {
        "exact_iou": [], "fast_polygon_iou": [],
        "exact_pixel_disagreement": [], "fast_polygon_pixel_disagreement": [],
    }
    source_lags = []
    try:
        for selected in chosen:
            path, index = candidates[int(selected)]
            with h5py.File(path, "r", swmr=True) as episode:
                sim = np.asarray(episode["observation/timestamp/sim"][:], np.float64)
                camera_time = float(episode["observation/timestamp/camera"][index])
                source = int(np.argmin(np.abs(sim - camera_time)))
                state = np.asarray(episode["observation/state"][source], np.float32)
                thrust = np.asarray(episode["observation/motor_thrusts"][source], np.float32)
                omega = np.asarray(episode["observation/motor_omega"][source], np.float32)
                stored = np.asarray(episode["observation/gate_mask"][index]) > 0
                proprio = Proprioception(state, thrust, omega)
                for renderer in ("exact", "fast_polygon"):
                    env.geometry_renderer = renderer
                    rendered = env._nearest_resize(env._render_gate_mask(proprio), env.mask_size) > 0
                    intersection = np.count_nonzero(rendered & stored)
                    union = np.count_nonzero(rendered | stored)
                    metrics[f"{renderer}_iou"].append(
                        1.0 if union == 0 else intersection / union
                    )
                    metrics[f"{renderer}_pixel_disagreement"].append(
                        float(np.not_equal(rendered, stored).mean())
                    )
                source_lags.append(index - source)
    finally:
        env.close()
    return {
        "samples": len(source_lags),
        **{
            name: {
                "mean": float(np.mean(values)), "minimum": float(np.min(values)),
                "p05": float(np.quantile(values, 0.05)), "maximum": float(np.max(values)),
            }
            for name, values in metrics.items()
        },
        "source_lag_steps": {"mean": float(np.mean(source_lags)), "values": sorted(set(source_lags))},
    }


def mask_statistics(batch: dict[str, torch.Tensor], history: int):
    masks = batch["mask"][:, history - 1].numpy() >= 0.5
    phases = batch["gate_phase"][:, history - 1].numpy()
    result = {}
    for index, name in enumerate(GATE_PHASE_NAMES):
        chosen = masks[phases == index]
        if not len(chosen):
            continue
        fraction = chosen.mean(axis=(1, 2, 3))
        result[name] = {
            "count": len(chosen), "empty_fraction": float((fraction == 0).mean()),
            "foreground_mean": float(fraction.mean()), "foreground_p05": float(np.quantile(fraction, 0.05)),
            "foreground_p95": float(np.quantile(fraction, 0.95)),
        }
    return result


def online_phase(after_crossing: int, task_state: np.ndarray) -> str:
    if after_crossing == 0:
        return "crossing"
    if 0 < after_crossing <= 10:
        return "post_crossing"
    if 10 < after_crossing <= 45:
        return "next_gate_acquisition"
    if float(np.linalg.norm(task_state[:3])) <= 2.0:
        return "approach"
    return "ordinary"


def aggregate_online(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = {}
    for phase, values in rows.items():
        state = np.stack([row["state_error"] for row in values])
        estimate_input = np.stack([
            row["estimate_input_error"] for row in values
            if row.get("estimate_input_error") is not None
        ])
        dynamics = np.stack([row["dynamics_error"] for row in values if row.get("dynamics_error") is not None])
        result[phase] = {
            "count": len(values),
            "state_mae_by_group": {
                group: float(state[:, section].mean())
                for group, section in STATE_GROUPS.items()
            },
            "estimate_input_mae_by_group": {
                group: float(estimate_input[:, section].mean())
                for group, section in {
                    "position": slice(0, 3), "velocity": slice(3, 6),
                    "attitude_6d": slice(6, 12),
                }.items()
            } if len(estimate_input) else {},
            "dynamics_mae_by_group": {
                group: float(dynamics[:, section].mean())
                for group, section in STATE_GROUPS.items()
            } if len(dynamics) else {},
            "empty_mask_fraction": float(np.mean([row["mask_empty"] for row in values])),
            "startup_fraction": float(np.mean([row["startup"] for row in values])),
            "actor_teacher_action_rmse": float(np.mean([
                row["actor_teacher_action_rmse"]
                for row in values if row.get("actor_teacher_action_rmse") is not None
            ])) if any(row.get("actor_teacher_action_rmse") is not None for row in values) else None,
        }
    return result


@torch.no_grad()
def online_rollout_report(
    actor, encoder, action_tokenizer, config_path: Path, *, episodes: int,
    max_steps: int, device: str, seed: int,
    execution: str,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["actor_dagger"]
    stage = RacingCurriculumStage.from_mapping(settings["stage"])
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outcomes = []
    for episode_index in range(episodes):
        episode_seed = seed + 104729 * episode_index
        env = FlightmareEnv(
            track=stage.tracks[episode_index % len(stage.tracks)], next_gates=3,
            image_size=(160, 128), control_dt=1.0 / 90.0,
            render_observations=False, mask_source="geometry", mask_size=(160, 128),
            geometry_renderer=str(settings.get("geometry_renderer", "exact")),
            image_delay=float(settings.get("image_delay", 0.033)),
            action_delay=stage.action_delay, terminate_on_collision=True,
        )
        try:
            spawn = sample_curriculum_spawn(
                env.track, stage, seed=episode_seed, episode_index=episode_index
            )
            options: dict[str, Any] = {
                "gate_index": spawn.gate_index, "state": spawn.state,
                "spawn": dict(spawn.metadata),
            }
            if spawn.previous_action is not None:
                options["previous_action"] = spawn.previous_action
            observation, _ = env.reset(seed=episode_seed, options=options)
            history = ObservationHistory(
                actor.temporal_context, tuple(encoder.image_size),
                estimate_dim=int(getattr(encoder, "estimate_dim", 0)),
                estimate_seed=episode_seed,
            )
            history.append(observation)
            warmup = make_mpcc(env, settings, stage.action_delay)
            warmup_terminated = False
            for _ in range(int(settings.get("history_warmup_steps", actor.temporal_context - 1))):
                command = warmup(observation).action.as_array()
                observation, _, warmup_terminated, _, _ = env.step(command)
                warmup.observe_executed_action(command)
                history.append(observation)
                if warmup_terminated:
                    break
            passed = 0
            after_crossing = -1
            crashed = bool(warmup_terminated)
            if warmup_terminated:
                outcomes.append({"gates": 0, "crash": True, "steps": 0})
                continue
            for step in range(max_steps):
                batch = history.batch(device, repeat_first_padding=True)
                latents = encoder.encode({
                    key: batch[key]
                    for key in ("mask", "proprio", "route", "timing", "estimate")
                    if key in batch
                })
                state_mean, _ = encoder.estimate_state(latents)
                prediction = state_mean[:, -1].float().cpu().numpy()[0]
                target = normalized_state(observation["task_state"])
                phase = online_phase(after_crossing, np.asarray(observation["task_state"]))
                features = build_actor_state(encoder, batch, actor.temporal_context, device, True)
                action = actor(features, action_tokenizer).mean[0, 0].float().cpu().numpy()
                action_series = torch.zeros(
                    1, actor.temporal_context, 4, device=device, dtype=torch.float32
                )
                action_series[:, -1] = torch.from_numpy(action).to(device)
                action_timing = torch.ones(
                    1, actor.temporal_context, 3, device=device, dtype=torch.float32
                )
                action_timing[..., 1] = stage.action_delay * 90.0
                dynamics = encoder.decode(
                    latents, normalized_action=action_series, action_timing=action_timing
                ).action_state_delta_mean[:, -1].float().cpu().numpy()[0]
                record = {
                    "state_error": np.abs(prediction - target) * STATE_SCALE,
                    "dynamics_prediction": dynamics,
                    "dynamics_error": None,
                    "target_before": target,
                    "mask_empty": not bool(np.any(observation["gate_mask"])),
                    "startup": False,
                    "actor_teacher_action_rmse": None,
                    "estimate_input_error": (
                        np.abs(
                            batch["estimate"][0, -1, :12].float().cpu().numpy()
                            - target[:12]
                        ) * STATE_SCALE[:12]
                        if "estimate" in batch else None
                    ),
                }
                if execution == "mpcc":
                    teacher_command = warmup(observation).action.as_array()
                    teacher_action = ctbr_to_normalized(teacher_command)
                    record["actor_teacher_action_rmse"] = float(
                        np.sqrt(np.mean((action - teacher_action) ** 2))
                    )
                    executed_command = teacher_command
                else:
                    executed_command = normalized_to_ctbr(action)
                next_observation, _, terminated, _, info = env.step(executed_command)
                if execution == "mpcc":
                    warmup.observe_executed_action(executed_command)
                next_target = normalized_state(next_observation["task_state"])
                record["dynamics_error"] = np.abs(
                    dynamics - (next_target - target)
                ) * STATE_SCALE
                observation = next_observation
                history.append(observation)
                if bool(info.get("gate_passed")):
                    phase = "crossing"
                    passed += 1
                    after_crossing = 1
                elif after_crossing >= 0:
                    after_crossing += 1
                rows[phase].append(record)
                crashed = bool(
                    info.get("ground_contact") or info.get("unity_collision")
                    or (terminated and passed < stage.target_gates)
                )
                if crashed or passed >= stage.target_gates:
                    break
            outcomes.append({"gates": passed, "crash": crashed, "steps": step + 1})
        finally:
            env.close()
    return {
        "episodes": episodes,
        "execution": execution,
        "mean_gates": float(np.mean([row["gates"] for row in outcomes])),
        "p1": float(np.mean([row["gates"] >= 1 for row in outcomes])),
        "p2": float(np.mean([row["gates"] >= 2 for row in outcomes])),
        "crash": float(np.mean([row["crash"] for row in outcomes])),
        "phase_metrics": aggregate_online(rows),
    }


def main() -> None:
    args = arguments()
    _, validation_paths = stratified_episode_split(args.data, 0.12, 2026081850, tracks=args.track)
    initial = torch.load(args.actor, map_location="cpu", weights_only=False)
    actor = GaussianActor(**initial["model_config"]).to(args.device)
    actor.load_state_dict(initial["model"])
    actor.eval()
    _, encoder, action_tokenizer, _, _, _, _ = load_frozen_actor_stack(
        None, args.tokenizer, Path(initial["action_tokenizer"]), args.device
    )
    if "observation_encoder" in initial:
        encoder.load_state_dict(initial["observation_encoder"])
    dataset = DreamerSequenceDataset(
        args.data, paths=validation_paths, sequence_length=args.history,
        stride=1, mode="control", mask_size=tuple(encoder.image_size),
        validate_contents=False, cache_in_memory=False, include_privileged=False,
        route_source="flight_plan", route_target_source="body_relative_gates",
        deployment_estimate_source=(
            "proxy_v1" if int(getattr(encoder, "estimate_dim", 0)) else "none"
        ),
        deployment_estimate_seed=2026081718,
    )
    indices = phase_indices(dataset, args.history, args.samples_per_phase, args.seed)
    batch = batch_samples(dataset, indices)
    report = {
        "contract": {
            "history": args.history, "validation_episodes": len(validation_paths),
            "sampled_windows": len(indices), "actor": str(args.actor),
            "tokenizer": str(args.tokenizer),
            "actor_checkpoint_step": int(initial.get("step", 0)),
            "actor_checkpoint_round": int(initial.get("round", 0)),
            "actor_parameters": sum(parameter.numel() for parameter in actor.parameters()),
            "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        },
        "preprocessing_parity": preprocessing_parity(dataset, indices[0], args.history),
        "mask_replay": mask_replay(validation_paths, args.track, args.mask_replay_samples, args.seed),
        "mask_statistics": mask_statistics(batch, args.history),
        "task_state_delta_by_gate_phase": sequence_and_delta_report(dataset, indices, args.history),
        "dynamics_head_error_by_gate_phase": dynamics_report(
            encoder, batch, args.device, args.history
        ),
        "representation_ablations": representation_report(
            actor, encoder, action_tokenizer, batch, args.device, args.history
        ),
        "positional_parameters": {
            "tokenizer_macro_position_rms": float(encoder.macro_belief.position.float().square().mean().sqrt().cpu()),
            "tokenizer_macro_output_weight_rms": float(encoder.macro_belief.output.weight.float().square().mean().sqrt().cpu()),
            "actor_temporal_position_rms": float(actor.temporal_position.detach().float().square().mean().sqrt().cpu()),
        },
    }
    if args.online_episodes:
        if args.dagger_config is None:
            raise ValueError("--online-episodes requires --dagger-config")
        report["online_student_rollout"] = online_rollout_report(
            actor, encoder, action_tokenizer, args.dagger_config,
            episodes=args.online_episodes, max_steps=args.online_max_steps,
            device=args.device, seed=args.seed + 500000,
            execution=args.online_execution,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(jsonable(report), indent=2), flush=True)


if __name__ == "__main__":
    main()
