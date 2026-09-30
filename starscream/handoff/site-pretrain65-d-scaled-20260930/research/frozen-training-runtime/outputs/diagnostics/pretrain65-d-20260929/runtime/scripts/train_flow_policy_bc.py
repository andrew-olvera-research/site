#!/usr/bin/env python3
"""Behavior cloning for the direct tokenizer-conditioned shortcut-flow policy."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from starscream.checkpoint_manager import (
    CheckpointManager,
    capture_rng_state,
    restore_rng_state,
)
from starscream.dataloader import (
    DreamerSequenceDataset,
    GATE_PHASE_NAMES,
    LocalityAwareBatchSampler,
    stratified_episode_split,
)
from starscream.DiT import TokenizerFlowPolicy
from starscream.flow_policy import tokenizer_policy_tokens
from starscream.wandb import init_wandb
from train_actor_bc import (
    RecoveryAwareBatchSampler,
    load_frozen_actor_stack,
    move_batch,
)


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    seen = set() if seen is None else seen
    if resolved in seen:
        raise ValueError(f"cyclic experiment config inheritance at {resolved}")
    seen.add(resolved)
    with resolved.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    parent = config.get("inherits")
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return _merge_config(_load_config(parent_path, seen), config)


def parse_args() -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoke", action="store_true")
    preliminary, _ = parser.parse_known_args()
    config = _load_config(preliminary.config)
    parser.set_defaults(**config.get("flow_policy_bc", {}))
    args = parser.parse_args()
    if args.device is None:
        args.device = str(config.get("flow_policy_bc", {}).get("device", "cuda"))
    if args.resume is None:
        args.resume = bool(config.get("flow_policy_bc", {}).get("resume", False))
    if args.smoke:
        config = copy.deepcopy(config)
        settings = config["flow_policy_bc"]
        settings.update(
            steps=2,
            workers=0,
            batch_size=min(4, int(settings.get("batch_size", 32))),
            validation_batch_size=4,
            validation_batches=1,
            validation_interval=2,
            checkpoint_interval=2,
            sample_count=2,
            sampling_steps=2,
            resume=False,
        )
        config["output_root"] = "/tmp/starscream-flow-policy-bc-smoke"
        config["checkpoint"] = {
            "run_name": "flow-policy-bc-smoke",
            "monitor": "selection_score",
            "mode": "min",
            "top_k": 1,
        }
        config["wandb"] = {"enabled": False, "run_name": "flow-policy-bc-smoke"}
        args.steps = 2
        args.resume = False
    return args, config


def history_valid_steps(settings: dict, batch_size: int, device: torch.device) -> torch.Tensor:
    probabilities = torch.tensor(
        settings.get("history_step_probabilities", [0.15, 0.25, 0.60]),
        device=device,
        dtype=torch.float32,
    )
    if probabilities.shape != (3,) or torch.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("history_step_probabilities must contain three nonnegative values")
    return torch.multinomial(probabilities / probabilities.sum(), batch_size, replacement=True) + 1


def phase_weighted_metrics(
    values: torch.Tensor, phase: torch.Tensor, prefix: str, totals: dict[str, float]
) -> None:
    recovery = phase == 3
    critical = phase >= 2
    totals[f"{prefix}_sum"] += float(values.sum())
    totals[f"{prefix}_recovery_sum"] += float(values[recovery].sum())
    totals[f"{prefix}_critical_sum"] += float(values[critical].sum())


@torch.no_grad()
def validate(
    actor,
    observation_tokenizer,
    action_tokenizer,
    loader,
    settings,
    *,
    history_raw_steps: int,
    action_horizon: int,
    device: str,
    amp: bool,
) -> dict[str, float]:
    actor.eval()
    totals = {
        "count": 0.0,
        "recovery_count": 0.0,
        "critical_count": 0.0,
        "sample_sum": 0.0,
        "sample_recovery_sum": 0.0,
        "sample_critical_sum": 0.0,
        "best_sum": 0.0,
        "best_recovery_sum": 0.0,
        "best_critical_sum": 0.0,
        "mode_sum": 0.0,
        "mode_recovery_sum": 0.0,
        "mode_critical_sum": 0.0,
        "diversity": 0.0,
        "saturation": 0.0,
        "flow_loss": 0.0,
        "endpoint_mse": 0.0,
    }
    target_start = history_raw_steps - 1
    samples_per_input = int(settings.get("sample_count", 4))
    sampling_steps = int(settings.get("sampling_steps", 4))
    sampling_method = str(settings.get("sampling_method", "heun"))
    devices = [torch.cuda.current_device()] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(settings.get("validation_seed", 20262030)))
        for batch_index, raw in enumerate(loader):
            if batch_index >= int(settings.get("validation_batches", 50)):
                break
            batch = move_batch(raw, device)
            observation_tokens, action_tokens = tokenizer_policy_tokens(
                observation_tokenizer,
                action_tokenizer,
                batch,
                history_raw_steps=history_raw_steps,
                amp=amp,
            )
            target = batch[str(settings.get("target_field", "commanded_action"))][
                :, target_start : target_start + action_horizon
            ].float()
            phase = batch[str(settings.get("phase_field", "tail_phase"))][
                :, target_start : target_start + action_horizon
            ].amax(1)
            previous = batch["previous_action"][:, history_raw_steps - 1]
            # Held-out flow matching remains the likelihood-proxy diagnostic.
            generated = []
            with torch.autocast(
                "cuda", dtype=torch.bfloat16,
                enabled=device.startswith("cuda") and amp,
            ):
                _, flow = actor.shortcut_training_loss(
                    target,
                    observation_tokens,
                    action_tokens,
                    previous_action=previous,
                    direct_weight=float(settings.get("shortcut_direct_weight", 0.5)),
                    bootstrap_weight=float(settings.get("shortcut_bootstrap_weight", 1.0)),
                )
                mode = actor.sample(
                    observation_tokens,
                    action_tokens,
                    steps=sampling_steps,
                    method=sampling_method,
                    previous_action=previous,
                    deterministic=True,
                ).float()
                for _ in range(samples_per_input):
                    generated.append(actor.sample(
                        observation_tokens,
                        action_tokens,
                        steps=sampling_steps,
                        method=sampling_method,
                        previous_action=previous,
                        deterministic=False,
                    ).float())
            generated = torch.stack(generated, dim=1)
            sample_error = (generated - target[:, None]).square().mean((2, 3))
            expected = sample_error.mean(1)
            best = sample_error.min(1).values
            mode_error = (mode - target).square().mean((1, 2))
            totals["count"] += len(target)
            totals["recovery_count"] += int((phase == 3).sum())
            totals["critical_count"] += int((phase >= 2).sum())
            phase_weighted_metrics(expected, phase, "sample", totals)
            phase_weighted_metrics(best, phase, "best", totals)
            phase_weighted_metrics(mode_error, phase, "mode", totals)
            totals["diversity"] += float(generated.std(1, unbiased=False).mean() * len(target))
            totals["saturation"] += float(
                (generated.abs() >= 0.98).float().mean() * len(target)
            )
            totals["flow_loss"] += float(flow["flow_matching_loss"] * len(target))
            totals["endpoint_mse"] += float(flow["endpoint_mse"] * len(target))
    count = max(1.0, totals["count"])
    recovery = max(1.0, totals["recovery_count"])
    critical = max(1.0, totals["critical_count"])
    metrics = {
        "sample_mse": totals["sample_sum"] / count,
        "best_of_k_mse": totals["best_sum"] / count,
        "mode_mse": totals["mode_sum"] / count,
        "recovery_sample_mse": totals["sample_recovery_sum"] / recovery,
        "recovery_best_of_k_mse": totals["best_recovery_sum"] / recovery,
        "recovery_mode_mse": totals["mode_recovery_sum"] / recovery,
        "critical_sample_mse": totals["sample_critical_sum"] / critical,
        "critical_best_of_k_mse": totals["best_critical_sum"] / critical,
        "critical_mode_mse": totals["mode_critical_sum"] / critical,
        "sample_diversity": totals["diversity"] / count,
        "sample_saturation": totals["saturation"] / count,
        "flow_matching_loss": totals["flow_loss"] / count,
        "endpoint_mse": totals["endpoint_mse"] / count,
        "recovery_fraction": totals["recovery_count"] / count,
        "critical_fraction": totals["critical_count"] / count,
    }
    rw = float(settings.get("recovery_selection_weight", 0.5))
    cw = float(settings.get("critical_selection_weight", 0.25))
    ow = 1.0 - rw - cw
    if ow < 0:
        raise ValueError("recovery and critical selection weights must sum to at most one")
    sample_score = (
        ow * metrics["sample_mse"]
        + rw * metrics["recovery_sample_mse"]
        + cw * metrics["critical_sample_mse"]
    )
    best_score = (
        ow * metrics["best_of_k_mse"]
        + rw * metrics["recovery_best_of_k_mse"]
        + cw * metrics["critical_best_of_k_mse"]
    )
    metrics["selection_score"] = (
        float(settings.get("selection_sample_weight", 0.5)) * sample_score
        + float(settings.get("selection_best_of_k_weight", 0.5)) * best_score
        + float(settings.get("selection_flow_weight", 0.05))
        * metrics["flow_matching_loss"]
    )
    actor.train()
    return metrics


def main() -> None:
    args, config = parse_args()
    settings = config["flow_policy_bc"]
    device = str(args.device)
    steps = int(args.steps or settings.get("steps", 30000))
    seed = int(settings.get("seed", 20260814))
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(settings.get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", True))
    amp = bool(settings.get("amp", True))

    loaded = load_frozen_actor_stack(
        None,
        Path(settings["tokenizer"]),
        Path(settings["action_tokenizer"]),
        device,
    )
    _, observation_tokenizer, action_tokenizer, encoder_config, observation_path, action_path, _ = loaded
    observation_tokenizer.eval().requires_grad_(False).to(device)
    action_tokenizer.eval().requires_grad_(False).to(device)
    patch = int(observation_tokenizer.temporal_patch_size)
    history_steps = int(settings.get("history_steps", 3))
    history_raw_steps = history_steps * patch
    action_horizon = int(settings.get("action_horizon", patch))
    if history_steps != 3:
        raise ValueError("the prototype contract requires exactly three macro history steps")
    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")

    train_paths, validation_paths = stratified_episode_split(
        settings["data"],
        float(settings.get("validation_fraction", 0.1)),
        int(settings.get("validation_seed", seed)),
    )
    dataset_options = dict(
        root=settings["data"],
        sequence_length=history_raw_steps + action_horizon - 1,
        stride=int(settings.get("stride", patch)),
        mode="dynamics",
        mask_size=tuple(encoder_config.get("image_size", (128, 160))),
        max_open_files=int(settings.get("max_open_files", 12)),
        validate_contents=bool(settings.get("validate_contents", False)),
        include_privileged=False,
        cache_in_memory=False,
        require_controller_valid=False,
    )
    train_dataset = DreamerSequenceDataset(paths=train_paths, **dataset_options)
    validation_dataset = DreamerSequenceDataset(paths=validation_paths, **dataset_options)
    train_indices, train_phases = train_dataset.behavior_cloning_windows(
        history=history_raw_steps, action_horizon=action_horizon,
        require_solver_clean=bool(settings.get("require_solver_clean_targets", False)),
    )
    validation_indices, _ = validation_dataset.behavior_cloning_windows(
        history=history_raw_steps, action_horizon=action_horizon,
        require_solver_clean=bool(settings.get("require_solver_clean_targets", False)),
    )
    gate_phase_fractions = settings.get("gate_phase_fractions")
    if gate_phase_fractions is not None:
        one_step_indices, one_step_phases = train_dataset.one_step_behavior_cloning_windows(
            history=history_raw_steps
        )
        if action_horizon != 1 or one_step_indices != train_indices:
            raise ValueError("gate-phase balanced flow BC requires one-step targets")
        phase_labels = [0] * len(train_dataset)
        for index, phase in zip(one_step_indices, one_step_phases):
            phase_labels[index] = phase
        sampler = LocalityAwareBatchSampler(
            train_dataset,
            int(settings.get("batch_size", 48)),
            episodes_per_batch=int(settings.get("episodes_per_batch", 4)),
            cache_size=int(settings.get("max_open_files", 12)),
            windows_per_locality_block=int(settings.get("windows_per_locality_block", 64)),
            seed=seed, eligible_indices=train_indices, phase_labels=phase_labels,
            phase_names=GATE_PHASE_NAMES, phase_fractions=gate_phase_fractions,
        )
    else:
        sampler = RecoveryAwareBatchSampler(
            train_dataset,
            train_indices,
            train_phases,
            batch_size=int(settings.get("batch_size", 48)),
            recovery_fraction_start=float(settings.get("recovery_fraction_start", 0.5)),
            recovery_fraction_end=float(settings.get("recovery_fraction_end", 0.2)),
            tail_fraction_start=float(settings.get("tail_fraction_start", 0.2)),
            tail_fraction_end=float(settings.get("tail_fraction_end", 0.1)),
            curriculum_steps=int(settings.get("recovery_curriculum_steps", 5000)),
            episodes_per_batch=int(settings.get("episodes_per_batch", 4)),
            seed=seed,
        )
    workers = int(settings.get("workers", 3))
    loader_options = dict(
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=workers > 0,
    )
    if workers:
        loader_options.update(
            prefetch_factor=int(settings.get("prefetch_factor", 2)),
            multiprocessing_context="spawn",
        )
    loader = DataLoader(train_dataset, batch_sampler=sampler, **loader_options)
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=int(settings.get("validation_batch_size", 48)),
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    actor_config = {
        "observation_token_dim": int(observation_tokenizer.d_bottleneck),
        "action_token_dim": int(action_tokenizer.d_latent),
        "observation_tokens": int(observation_tokenizer.n_latents),
        "visual_tokens": int(settings.get("visual_tokens", 4)),
        "action_tokens": int(action_tokenizer.n_tokens),
        "history_steps": history_steps,
        "action_horizon": action_horizon,
        "action_dim": int(action_tokenizer.action_dim),
        "d_model": int(settings.get("d_model", 384)),
        "n_heads": int(settings.get("n_heads", 8)),
        "context_depth": int(settings.get("context_depth", 4)),
        "flow_depth": int(settings.get("flow_depth", 8)),
        "mlp_ratio": int(settings.get("mlp_ratio", 4)),
        "k_max": int(settings.get("k_max", 8)),
        "source_mode": str(settings.get("source_mode", "gaussian")),
        "source_noise": float(settings.get("source_noise", 1.0)),
    }
    actor = TokenizerFlowPolicy(**actor_config).to(device)
    teacher = None
    teacher_path = settings.get("teacher_actor")
    teacher_fraction = float(settings.get("teacher_fraction", 0.0))
    if not 0.0 <= teacher_fraction <= 1.0:
        raise ValueError("teacher_fraction must be in [0,1]")
    if teacher_path and teacher_fraction > 0.0:
        teacher_checkpoint = torch.load(
            Path(teacher_path), map_location=device, weights_only=False
        )
        teacher_config = teacher_checkpoint.get("model_config")
        if not isinstance(teacher_config, dict):
            raise ValueError("teacher checkpoint is missing model_config")
        contract_keys = (
            "observation_token_dim", "action_token_dim", "observation_tokens",
            "visual_tokens", "action_tokens", "history_steps", "action_horizon",
            "action_dim", "k_max", "source_mode",
        )
        mismatches = {
            key: (teacher_config.get(key), actor_config.get(key))
            for key in contract_keys
            if teacher_config.get(key) != actor_config.get(key)
        }
        if mismatches:
            raise ValueError(f"teacher/student policy contract mismatch: {mismatches}")
        teacher = TokenizerFlowPolicy(**teacher_config).to(device)
        teacher.load_state_dict(teacher_checkpoint["model"])
        teacher.eval().requires_grad_(False)
        teacher_steps = int(settings.get("teacher_sampling_steps", 1))
        teacher_method = str(settings.get("teacher_sampling_method", "euler"))
        if teacher_steps < 1 or teacher.k_max % teacher_steps:
            raise ValueError("teacher_sampling_steps must divide the teacher k_max")
        if teacher_method not in {"euler", "heun"}:
            raise ValueError("teacher_sampling_method must be euler or heun")
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(settings.get("learning_rate", 2e-4)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
        fused=bool(settings.get("fused_optimizer", True)) and device.startswith("cuda"),
    )
    warmup = int(settings.get("warmup_steps", 500))
    minimum = float(settings.get("min_lr_ratio", 0.1))

    def multiplier(step: int) -> float:
        if step < warmup:
            return max(1, step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    manager = CheckpointManager.from_config(config)
    logger = init_wandb(config)
    start = 0
    resumed = manager.resume(bool(args.resume), map_location=device)
    if resumed:
        actor.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        restore_rng_state(resumed.get("rng_state"))
        start = int(resumed.get("step", 0))

    print(
        f"flow_policy history_macros={history_steps} tokens_per_macro="
        f"{actor.observation_tokens}+{actor.action_tokens} context_tokens="
        f"{history_steps * actor.tokens_per_step} action_horizon={action_horizon} "
        f"trainable_parameters={sum(p.numel() for p in actor.parameters()):,} "
        f"frozen_observation_parameters={sum(p.numel() for p in observation_tokenizer.parameters()):,} "
        f"frozen_action_parameters={sum(p.numel() for p in action_tokenizer.parameters()):,} "
        f"eligible_train_windows={len(train_indices)} "
        f"eligible_validation_windows={len(validation_indices)} "
        f"teacher={teacher_path if teacher is not None else 'none'} "
        f"teacher_fraction={teacher_fraction if teacher is not None else 0.0:.2f}",
        flush=True,
    )
    iterator = iter(loader)
    target_start = history_raw_steps - 1
    last_log = time.perf_counter()
    for step in range(start, steps):
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        batch = move_batch(raw, device)
        valid_history = history_valid_steps(settings, len(batch["mask"]), batch["mask"].device)
        observation_tokens, action_tokens = tokenizer_policy_tokens(
            observation_tokenizer,
            action_tokenizer,
            batch,
            history_raw_steps=history_raw_steps,
            valid_history_steps=valid_history,
            amp=amp,
        )
        target = batch[str(settings.get("target_field", "commanded_action"))][
            :, target_start : target_start + action_horizon
        ]
        previous = batch["previous_action"][:, history_raw_steps - 1]
        phase = batch[str(settings.get("phase_field", "tail_phase"))][
            :, target_start : target_start + action_horizon
        ].amax(1)
        teacher_mask = None
        if teacher is not None:
            teacher_mask = torch.rand(target.shape[0], device=target.device) < teacher_fraction
            if torch.any(teacher_mask):
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=device.startswith("cuda") and amp,
                ):
                    teacher_target = teacher.sample(
                        observation_tokens[teacher_mask],
                        action_tokens[teacher_mask],
                        steps=teacher_steps,
                        method=teacher_method,
                        previous_action=previous[teacher_mask],
                        deterministic=True,
                    ).float()
                target = target.clone()
                target[teacher_mask] = teacher_target
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=device.startswith("cuda") and amp,
        ):
            loss, metrics = actor.shortcut_training_loss(
                target,
                observation_tokens,
                action_tokens,
                previous_action=previous,
                direct_weight=float(settings.get("shortcut_direct_weight", 0.5)),
                bootstrap_weight=float(settings.get("shortcut_bootstrap_weight", 1.0)),
            )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), float(settings.get("grad_clip", 5.0))
        )
        optimizer.step()
        scheduler.step()
        if step % int(settings.get("log_interval", 25)) == 0:
            now = time.perf_counter()
            logged = {
                **metrics,
                "gradient_norm": gradient.detach(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "history_steps": valid_history.float().mean(),
                "one_step_history_fraction": (valid_history == 1).float().mean(),
                "recovery_fraction": (phase == 3).float().mean(),
                "teacher_target_fraction": (
                    teacher_mask.float().mean() if teacher_mask is not None
                    else target.new_zeros(())
                ),
                "steps_per_second": int(settings.get("log_interval", 25))
                / max(now - last_log, 1e-6),
            }
            print(
                f"step={step + 1} "
                + " ".join(f"{key}={float(value):.5f}" for key, value in logged.items()),
                flush=True,
            )
            logger.log_train(logged, step + 1)
            last_log = now

        validation_metrics = None
        if (step + 1) % int(settings.get("validation_interval", 500)) == 0 or step + 1 == steps:
            validation_metrics = validate(
                actor,
                observation_tokenizer,
                action_tokenizer,
                validation_loader,
                settings,
                history_raw_steps=history_raw_steps,
                action_horizon=action_horizon,
                device=device,
                amp=amp,
            )
            print(
                f"step={step + 1} validation "
                + " ".join(f"{key}={value:.5f}" for key, value in validation_metrics.items()),
                flush=True,
            )
            logger.log_eval(validation_metrics, step + 1)
            manager.save_eval(
                validation_metrics,
                step=step + 1,
                summary=(
                    "Held-out expert-only shortcut-flow BC over frozen visual/fusion "
                    "observation tokens and causal applied-action tokens."
                ),
            )
        if validation_metrics is not None or (step + 1) % int(settings.get("checkpoint_interval", 500)) == 0:
            manager.save(
                {
                    "model": actor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng_state": capture_rng_state(),
                    "model_config": actor_config,
                    "training_config": config,
                    "observation_tokenizer": str(observation_path),
                    "action_tokenizer": str(action_path),
                    "teacher_actor": str(teacher_path) if teacher is not None else None,
                    "input_contract": {
                        "actor_state": "three_macro_frozen_observation_latents_plus_causal_applied_action_tokens",
                        "history_padding": "repeat_first_available_macro_step",
                        "observation_token_groups": "four_visual_segmentation_plus_three_fusion",
                        "action_conditioning": "two_applied_action_tokens_per_macro_no_expert_target",
                        "actor_decoder": "shortcut_flow_transformer_adaln_zero_rope",
                        "action": "normalized_commanded_ctbr",
                        "action_chunk_steps": action_horizon,
                        "target_requires_controller_valid": True,
                        "world_model": False,
                    },
                },
                step=step + 1,
                metrics=validation_metrics,
                rank=validation_metrics is not None,
            )
    logger.finish()


if __name__ == "__main__":
    main()
