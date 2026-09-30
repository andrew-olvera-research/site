#!/usr/bin/env python3
"""Matched BC ablation for exact, raw-estimate, and learned-estimate observations.

The experiment deliberately shares the successful privileged causal Transformer
and direct one-step CTBR head.  Only the compact per-step state representation
changes.  Every run finishes with the same deterministic closed-loop suite so
offline fit cannot masquerade as a useful feedback policy.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

# Direct execution places ``scripts`` rather than the repository root on
# sys.path. Keep imports identical inside Docker and pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_flight import ObservationHistory
from starscream.checkpoint_manager import CheckpointManager
from starscream.dataloader import DreamerSequenceDataset, stratified_episode_split
from starscream.env import FlightmareEnv
from starscream.privileged_racing import (
    FeatureNormalizer,
    PrivilegedMLPPolicy,
    normalized_to_ctbr,
)
from starscream.racing_curriculum import RacingCurriculumStage, sample_curriculum_spawn
from starscream.tokenizer import MultiModalTokenizer
from starscream.wandb import init_wandb


STATE_DIM = 19
ROUTE_DIM = 39
COMMON_DIM = ROUTE_DIM + 4 + 2
SOURCE_DIMS = {"exact": STATE_DIM, "raw_estimate": 32, "learned_estimate": 38}


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError(f"cyclic config inheritance at {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("inherits", raw.pop("extends", None))
    if parent is None:
        return raw
    return _merge(load_config((path.parent / str(parent)).resolve(), seen), raw)


def parse_args() -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config = copy.deepcopy(config)
        config["training"].update(
            steps=2, batch_size=2, normalizer_samples=4,
            validation_interval=1, validation_batches=1, workers=0,
        )
        config["evaluation"].update(episodes=1, workers=1, max_steps=4)
        config["wandb"] = {"enabled": False}
        config["output_root"] = "/tmp/starscream-compact-observation-smoke"
        name = f"compact-observation-{config['observation']['source']}-smoke"
        config["checkpoint"].update(run_name=name)
    if args.disable_wandb:
        config["wandb"] = {"enabled": False}
    return args, config


def load_encoder(path: str | Path, device: str, trainable: bool) -> MultiModalTokenizer:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model_config = state.get("model_config", {})
    encoder = MultiModalTokenizer(**dict(model_config.get("encoder", model_config)))
    encoder.load_state_dict(state["encoder"] if "encoder" in state else state["model"])
    encoder.to(device)
    encoder.requires_grad_(trainable)
    encoder.train(trainable)
    return encoder


def move(batch: Mapping[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def encoder_inputs(batch: Mapping[str, torch.Tensor], raw_history: int) -> dict[str, torch.Tensor]:
    return {
        key: batch[key][:, :raw_history]
        for key in ("mask", "proprio", "route", "timing", "estimate")
    }


def compact_features(
    batch: Mapping[str, torch.Tensor],
    *,
    source: str,
    policy_history: int,
    raw_history: int,
    encoder: MultiModalTokenizer | None,
    amp: bool,
    encoder_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Build a compact sequence and return optional learned state predictions."""

    start = raw_history - policy_history
    stop = raw_history
    state_key = "task_state" if "task_state" in batch else "deployable_task_state"
    target_state = batch[state_key][:, start:stop]
    learned_state = None
    if source == "exact":
        state_features = target_state
    elif source == "raw_estimate":
        state_features = batch["estimate"][:, start:stop]
    elif source == "learned_estimate":
        if encoder is None:
            raise RuntimeError("learned_estimate requires an observation encoder")
        context = torch.enable_grad() if encoder_grad else torch.no_grad()
        with context, torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=batch["mask"].device.type == "cuda" and amp,
        ):
            latents = encoder.encode(encoder_inputs(batch, raw_history))
            mean, log_scale = encoder.estimate_state(latents)
        learned_state = mean[:, start:stop].float()
        state_features = torch.cat(
            [learned_state, log_scale[:, start:stop].float().clamp(-8.0, 4.0)], dim=-1
        )
    else:
        raise ValueError(f"unknown observation source {source!r}")
    route = batch["route"][:, start:stop].flatten(2)
    previous = batch["previous_action"][:, start:stop]
    timing = batch["timing"][:, start:stop]
    # Previous-command age and validity are the two timing fields used by the
    # successful 64D privileged contract.
    age_valid = timing[..., [3, 7]]
    features = torch.cat([state_features.float(), route.float(), previous.float(), age_valid.float()], -1)
    expected = SOURCE_DIMS[source] + COMMON_DIM
    if features.shape[-1] != expected:
        raise RuntimeError(f"compact source produced {features.shape[-1]} features, expected {expected}")
    return features, learned_state, target_state


class CompactDataset(Dataset):
    """Drop image-sized fields before collation when a source does not use them."""

    COMMON_KEYS = {
        "task_state", "route", "previous_action", "timing", "proprio",
        "commanded_action", "gate_index",
    }

    def __init__(self, dataset: DreamerSequenceDataset, source: str) -> None:
        self.dataset = dataset
        self.source = source

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        keys = set(self.COMMON_KEYS)
        if self.source in {"raw_estimate", "learned_estimate"}:
            keys.add("estimate")
        if self.source == "learned_estimate":
            keys.update(("mask", "proprio"))
        return {key: sample[key] for key in keys}

    def close(self) -> None:
        self.dataset.close()


def make_dataset(
    settings: Mapping[str, Any], paths: list[Path], encoder: MultiModalTokenizer | None
) -> CompactDataset:
    image_size = tuple(
        int(value) for value in (
            encoder.image_size if encoder is not None else settings.get("image_size", [128, 160])
        )
    )
    dataset = DreamerSequenceDataset(
        root=settings["data"], paths=paths,
        sequence_length=int(settings.get("raw_history", 36)), stride=1,
        mode="dynamics", mask_size=image_size,
        prefer_gatenet=bool(settings.get("prefer_gatenet", True)),
        require_controller_valid=True,
        validate_contents=bool(settings.get("validate_contents", False)),
        include_privileged=False, cache_in_memory=False,
        route_source="flight_plan", route_target_source="body_relative_gates",
        deployment_estimate_source="proxy_v1",
        deployment_estimate_seed=int(settings.get("deployment_estimate_seed", 2026081718)),
        max_open_files=int(settings.get("max_open_files", 8)),
    )
    return CompactDataset(dataset, str(settings["observation_source"]))


@torch.no_grad()
def fit_normalizer(
    dataset: Dataset,
    *,
    source: str,
    encoder: MultiModalTokenizer | None,
    settings: Mapping[str, Any],
    device: str,
    amp: bool,
) -> FeatureNormalizer:
    count = min(int(settings.get("normalizer_samples", 2048)), len(dataset))
    rng = np.random.default_rng(int(settings.get("seed", 0)) + 17)
    selected = rng.choice(len(dataset), count, replace=False).tolist()
    loader = DataLoader(
        Subset(dataset, selected), batch_size=min(int(settings.get("batch_size", 16)), count),
        shuffle=False, num_workers=0,
    )
    values: list[np.ndarray] = []
    for raw in loader:
        batch = move(raw, device)
        features, _, _ = compact_features(
            batch, source=source,
            policy_history=int(settings.get("policy_history", 18)),
            raw_history=int(settings.get("raw_history", 36)),
            encoder=encoder, amp=amp, encoder_grad=False,
        )
        values.append(features.float().cpu().numpy().reshape(-1, features.shape[-1]))
    return FeatureNormalizer.fit(np.concatenate(values))


def objective(
    policy: PrivilegedMLPPolicy,
    normalizer: FeatureNormalizer,
    features: torch.Tensor,
    learned_state: torch.Tensor | None,
    target_state: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw_history = int(settings.get("raw_history", 36))
    target_index = raw_history - 1
    normalized = normalizer.tensor(features)
    target_action = batch["commanded_action"][:, target_index].float()
    previous = batch["previous_action"][:, target_index].float()
    predicted = policy(normalized)
    action = F.smooth_l1_loss(
        predicted, target_action, beta=float(settings.get("huber_beta", 0.05))
    )
    delta = F.smooth_l1_loss(
        predicted - previous, target_action - previous,
        beta=float(settings.get("delta_huber_beta", 0.04)),
    )
    next_delta = batch["task_state"][:, raw_history] - batch["task_state"][:, target_index]
    dynamics_prediction = policy.predict_dynamics(normalized)
    gate = batch["gate_index"]
    gate_consistent = gate[:, target_index, 0] == gate[:, raw_history, 0]
    if gate_consistent.any():
        dynamics = F.smooth_l1_loss(
            dynamics_prediction[gate_consistent], next_delta[gate_consistent], beta=0.02
        )
    else:
        dynamics = dynamics_prediction.sum() * 0.0
    state = predicted.sum() * 0.0
    state_mae = predicted.new_zeros(())
    if learned_state is not None:
        state = F.smooth_l1_loss(learned_state, target_state, beta=0.02)
        state_mae = (learned_state[:, -1] - target_state[:, -1]).abs().mean()
    total = (
        action
        + float(settings.get("delta_weight", 0.35)) * delta
        + float(settings.get("dynamics_weight", 0.15)) * dynamics
        + float(settings.get("state_weight", 0.25)) * state
    )
    return total, {
        "loss": total, "action_loss": action, "delta_loss": delta,
        "dynamics_loss": dynamics, "state_loss": state,
        "state_mae": state_mae,
        "action_rmse": (predicted - target_action).square().mean().sqrt(),
        "gate_consistent_fraction": gate_consistent.float().mean(),
    }


@dataclass
class EvalSlot:
    env: FlightmareEnv
    history: ObservationHistory
    start_passed: int
    target_gates: int
    steps: int = 0
    gates: int = 0
    crashed: bool = False
    done: bool = False


def make_eval_slot(
    settings: Mapping[str, Any], stage: RacingCurriculumStage, episode_index: int,
    seed: int, raw_history: int,
) -> EvalSlot:
    env = FlightmareEnv(
        track=str(settings["track"]), next_gates=3, image_size=(160, 128),
        control_dt=1.0 / 90.0, render_observations=False,
        mask_source="geometry", mask_size=(160, 128),
        geometry_renderer=str(settings.get("geometry_renderer", "exact")),
        image_delay=float(settings.get("image_delay", 0.033)),
        action_delay=float(stage.action_delay), terminate_on_collision=True,
    )
    spawn = sample_curriculum_spawn(env.track, stage, seed=seed, episode_index=episode_index)
    options: dict[str, Any] = {
        "gate_index": spawn.gate_index, "state": spawn.state,
        "spawn": dict(spawn.metadata),
    }
    if spawn.previous_action is not None:
        options["previous_action"] = spawn.previous_action
    observation, _ = env.reset(seed=seed, options=options)
    history = ObservationHistory(
        raw_history, (128, 160), estimate_dim=32, estimate_seed=seed
    )
    history.append(observation)
    return EvalSlot(
        env, history, env.tracker.passed_count,
        min(int(stage.target_gates), len(env.track.gates)),
    )


@torch.no_grad()
def evaluate(
    policy: PrivilegedMLPPolicy,
    normalizer: FeatureNormalizer,
    encoder: MultiModalTokenizer | None,
    config: Mapping[str, Any],
    device: str,
) -> dict[str, float]:
    settings = config["evaluation"]
    training = config["training"]
    source = str(config["observation"]["source"])
    raw_history = int(training.get("raw_history", 36))
    policy_history = int(training.get("policy_history", 18))
    stage = RacingCurriculumStage.from_mapping(dict(settings["curriculum"]))
    count = int(settings.get("episodes", 36))
    workers = min(int(settings.get("workers", 4)), count)
    seed_base = int(settings.get("seed", 2026081850))
    policy.eval()
    if encoder is not None:
        encoder.eval()
    results: list[dict[str, float]] = []
    episode_index = 0
    while episode_index < count:
        batch_count = min(workers, count - episode_index)
        slots = [
            make_eval_slot(
                settings, stage, episode_index + offset,
                seed_base + 1009 * (episode_index + offset), raw_history,
            )
            for offset in range(batch_count)
        ]
        try:
            with ThreadPoolExecutor(max_workers=batch_count) as executor:
                while any(not slot.done for slot in slots):
                    active = [slot for slot in slots if not slot.done]
                    batches = [slot.history.batch(device, repeat_first_padding=True) for slot in active]
                    batch = {
                        key: torch.cat([item[key] for item in batches], 0)
                        for key in batches[0]
                    }
                    features, _, _ = compact_features(
                        batch, source=source, policy_history=policy_history,
                        raw_history=raw_history, encoder=encoder,
                        amp=bool(training.get("amp", True)), encoder_grad=False,
                    )
                    actions = policy(normalizer.tensor(features)).float().cpu().numpy()
                    futures = {
                        index: executor.submit(slot.env.step, normalized_to_ctbr(actions[index]))
                        for index, slot in enumerate(active)
                    }
                    for index, slot in enumerate(active):
                        observation, _, terminated, _, info = futures[index].result()
                        slot.history.append(observation)
                        slot.steps += 1
                        slot.gates = slot.env.tracker.passed_count - slot.start_passed
                        success = slot.gates >= slot.target_gates
                        slot.crashed = bool(
                            info.get("ground_contact") or info.get("unity_collision")
                            or (terminated and not success)
                        )
                        slot.done = bool(
                            success or slot.crashed
                            or slot.steps >= int(settings.get("max_steps", stage.max_steps))
                        )
                        if slot.done:
                            results.append({
                                "gates": float(slot.gates), "crash": float(slot.crashed),
                                "steps": float(slot.steps), "success": float(success),
                            })
        finally:
            for slot in slots:
                slot.env.close()
        episode_index += batch_count
    gates = np.asarray([row["gates"] for row in results])
    success_steps = [row["steps"] for row in results if row["success"]]
    return {
        "episodes": float(len(results)), "mean_gates": float(gates.mean()),
        "p1": float(np.mean(gates >= 1)), "p2": float(np.mean(gates >= 2)),
        "p3": float(np.mean(gates >= 3)),
        "full_course_success": float(np.mean([row["success"] for row in results])),
        "crash_rate": float(np.mean([row["crash"] for row in results])),
        "successful_lap_time": (
            float(np.mean(success_steps) / 90.0) if success_steps else float("nan")
        ),
    }


def checkpoint_payload(
    policy: PrivilegedMLPPolicy, normalizer: FeatureNormalizer,
    encoder: MultiModalTokenizer | None, config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "contract": "compact-observation-causal-direct-ctbr-v1",
        "observation_source": str(config["observation"]["source"]),
        "model": policy.state_dict(), "model_config": policy.model_config(),
        "normalizer": normalizer.state_dict(), "optimizer": optimizer.state_dict(),
        "training_config": dict(config), "action_horizon": 1, "control_hz": 90,
    }
    if encoder is not None:
        result.update(
            observation_encoder=encoder.state_dict(),
            observation_tokenizer=str(config["observation"]["tokenizer"]),
        )
    return result


def run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    settings = config["training"]
    source = str(config["observation"]["source"])
    if source not in SOURCE_DIMS:
        raise ValueError(f"observation.source must be one of {sorted(SOURCE_DIMS)}")
    seed = int(settings.get("seed", 2026081801))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    encoder_trainable = source == "learned_estimate" and float(
        settings.get("encoder_learning_rate", 0.0)
    ) > 0.0
    encoder = (
        load_encoder(config["observation"]["tokenizer"], args.device, encoder_trainable)
        if source == "learned_estimate" else None
    )
    train_paths, validation_paths = stratified_episode_split(
        settings["data"], float(settings.get("validation_fraction", 0.12)),
        int(settings.get("validation_seed", 2026081501)), tracks=str(settings["track"]),
    )
    dataset_settings = dict(settings); dataset_settings["observation_source"] = source
    train_dataset = make_dataset(dataset_settings, train_paths, encoder)
    validation_dataset = make_dataset(dataset_settings, validation_paths, encoder)
    amp = bool(settings.get("amp", True)) and args.device.startswith("cuda")
    normalizer = fit_normalizer(
        train_dataset, source=source, encoder=encoder, settings=settings,
        device=args.device, amp=amp,
    )
    input_dim = SOURCE_DIMS[source] + COMMON_DIM
    model_config = dict(settings.get("model", {})); model_config["input_dim"] = input_dim
    policy = PrivilegedMLPPolicy(**model_config).to(args.device)
    groups: list[dict[str, Any]] = [
        {"params": policy.parameters(), "lr": float(settings.get("learning_rate", 3e-4))}
    ]
    if encoder_trainable and encoder is not None:
        groups.append({
            "params": encoder.parameters(),
            "lr": float(settings.get("encoder_learning_rate", 3e-6)),
        })
    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(settings.get("weight_decay", 1e-5)),
        fused=args.device.startswith("cuda"),
    )
    loader_options = {
        "num_workers": int(settings.get("workers", 2)),
        "pin_memory": args.device.startswith("cuda"),
    }
    if loader_options["num_workers"]:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=int(settings.get("prefetch_factor", 2)),
            multiprocessing_context="spawn",
        )
    train_loader = DataLoader(
        train_dataset, batch_size=int(settings.get("batch_size", 16)),
        shuffle=True, drop_last=True, **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=int(settings.get("validation_batch_size", settings.get("batch_size", 16))),
        shuffle=False, drop_last=False, **loader_options,
    )
    manager = CheckpointManager.from_config(config)
    logger = init_wandb(config)
    print(
        f"compact_observation_bc source={source} input={input_dim} "
        f"policy_parameters={sum(p.numel() for p in policy.parameters()):,} "
        f"encoder_parameters={sum(p.numel() for p in encoder.parameters()) if encoder else 0:,} "
        f"encoder_trainable={encoder_trainable} train_windows={len(train_dataset)} "
        f"validation_windows={len(validation_dataset)}",
        flush=True,
    )
    iterator = iter(train_loader)
    best_score = -float("inf")
    best_policy: dict[str, torch.Tensor] | None = None
    best_encoder: dict[str, torch.Tensor] | None = None
    best_step = 0
    started = time.perf_counter()
    steps = int(settings.get("steps", 3000))
    accumulation = int(settings.get("gradient_accumulation", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        for _ in range(accumulation):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(train_loader); raw = next(iterator)
            batch = move(raw, args.device)
            features, learned, target_state = compact_features(
                batch, source=source,
                policy_history=int(settings.get("policy_history", 18)),
                raw_history=int(settings.get("raw_history", 36)), encoder=encoder,
                amp=amp, encoder_grad=encoder_trainable,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss, pieces = objective(
                    policy, normalizer, features, learned, target_state, batch, settings
                )
            (loss / accumulation).backward()
            for key, value in pieces.items():
                accumulated[key] = accumulated.get(key, 0.0) + float(value.detach()) / accumulation
        gradient = torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + (
                list(encoder.parameters()) if encoder_trainable and encoder is not None else []
            ),
            float(settings.get("gradient_clip", 2.0)),
        )
        optimizer.step()
        if step == 1 or step % int(settings.get("log_interval", 25)) == 0:
            metrics = {
                **accumulated,
                "gradient_norm": float(gradient),
                "samples_per_second": step * int(settings.get("batch_size", 16)) * accumulation /
                max(time.perf_counter() - started, 1e-6),
            }
            logger.log_train(metrics, step)
            print(
                f"step={step}/{steps} loss={metrics['loss']:.5f} "
                f"action_rmse={metrics['action_rmse']:.5f} "
                f"state_mae={metrics['state_mae']:.5f} samples_s={metrics['samples_per_second']:.1f}",
                flush=True,
            )
        if step % int(settings.get("validation_interval", 250)) == 0 or step == steps:
            policy.eval()
            if encoder is not None:
                encoder.eval()
            sums: dict[str, float] = {}; examples = 0
            with torch.no_grad():
                for batch_index, raw_validation in enumerate(validation_loader):
                    if batch_index >= int(settings.get("validation_batches", 32)):
                        break
                    validation = move(raw_validation, args.device)
                    features, learned, target_state = compact_features(
                        validation, source=source,
                        policy_history=int(settings.get("policy_history", 18)),
                        raw_history=int(settings.get("raw_history", 36)), encoder=encoder,
                        amp=amp, encoder_grad=False,
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        _, pieces = objective(
                            policy, normalizer, features, learned, target_state,
                            validation, settings,
                        )
                    count = int(features.shape[0]); examples += count
                    for key, value in pieces.items():
                        sums[key] = sums.get(key, 0.0) + float(value) * count
            metrics = {key: value / max(examples, 1) for key, value in sums.items()}
            metrics["selection_score"] = -metrics["action_rmse"]
            logger.log_eval(metrics, step)
            manager.save(
                checkpoint_payload(policy, normalizer, encoder, config, optimizer),
                step=step, metrics=metrics,
            )
            print(
                f"validation step={step} action_rmse={metrics['action_rmse']:.5f} "
                f"state_mae={metrics['state_mae']:.5f}", flush=True,
            )
            if metrics["selection_score"] > best_score:
                best_score = metrics["selection_score"]; best_step = step
                best_policy = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
                best_encoder = (
                    {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
                    if encoder is not None else None
                )
            policy.train()
            if encoder is not None:
                encoder.train(encoder_trainable)
    if best_policy is None:
        raise RuntimeError("training completed without a selected checkpoint")
    policy.load_state_dict(best_policy)
    if encoder is not None and best_encoder is not None:
        encoder.load_state_dict(best_encoder)
    # Persistent HDF5 workers otherwise survive into Flightmare construction
    # and can hang native-library shutdown. Evaluation no longer needs replay.
    for loader in (train_loader, validation_loader):
        worker_iterator = getattr(loader, "_iterator", None)
        if worker_iterator is not None:
            worker_iterator._shutdown_workers()
            loader._iterator = None
    train_dataset.close(); validation_dataset.close()
    closed_loop = evaluate(policy, normalizer, encoder, config, args.device)
    logger.log_eval({f"closed_loop_{key}": value for key, value in closed_loop.items()}, best_step)
    manager.save_eval(
        closed_loop, step=best_step,
        summary=f"{source} compact observation BC closed-loop evaluation",
        probes={"best_offline_selection_score": best_score},
        kind="closed-loop",
    )
    manager.save(
        checkpoint_payload(policy, normalizer, encoder, config, optimizer),
        step=best_step,
        metrics={**closed_loop, "offline_selection_score": best_score}, rank=False,
    )
    print(f"closed_loop source={source} {json.dumps(closed_loop, sort_keys=True)}", flush=True)
    logger.finish()


if __name__ == "__main__":
    arguments, configuration = parse_args()
    run(arguments, configuration)
