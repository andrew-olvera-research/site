#!/usr/bin/env python3
"""Train and validate shortcut-forced latent dynamics with task heads."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from starscream.attention import set_attention_backend
from starscream.checkpoint_manager import CheckpointManager, capture_rng_state, restore_rng_state
from starscream.dataloader import (
    DreamerSequenceDataset, LocalityAwareBatchSampler, stratified_episode_split,
)
from starscream.dynamics import RaceDynamics
from starscream.loss import dynamics_loss, shortcut_forcing_objective
from starscream.tokenizer import ActionTokenizer, MultiModalTokenizer, macro_last
from starscream.wandb import init_wandb


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--action-tokenizer", type=Path)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=26)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("auto", "flash", "mem_efficient", "math"), default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-steps", type=int, default=0)
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        with preliminary.config.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        parser.set_defaults(**config.get("dynamics", {}))
    else:
        config = {
            "output_root": "outputs", "wandb": {"enabled": False, "run_name": "dynamics-cli"},
            "checkpoint": {"monitor": "loss", "mode": "min", "top_k": 3},
        }
    args = parser.parse_args()
    if args.data is None or args.tokenizer is None:
        parser.error("--data and --tokenizer are required (directly or via --config)")
    args.full_config = config
    if args.benchmark_steps:
        args.steps = int(args.benchmark_steps)
        args.full_config["output_root"] = "/tmp/starscream-dynamics-benchmark"
        args.full_config["wandb"] = {
            "enabled": False, "run_name": f"dynamics-benchmark-b{args.batch_size}"
        }
        args.full_config["checkpoint"] = {
            "run_name": f"dynamics-benchmark-b{args.batch_size}",
            "monitor": "loss", "mode": "min", "top_k": 1,
        }
        args.full_config.setdefault("dynamics", {})["validation_fraction"] = 0.0
        args.full_config["dynamics"]["checkpoint_interval"] = args.benchmark_steps
        args.full_config["dynamics"]["log_interval"] = 1
        args.full_config["dynamics"]["compile"] = False
    return args


def make_loader(dataset, settings, *, training: bool):
    workers = int(settings.get("workers", 6))
    batch_size = int(settings.get("batch_size", 16))
    options = dict(
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    if workers > 0:
        options["prefetch_factor"] = int(settings.get("prefetch_factor", 4))
        # Workers are first started after the GPU models are constructed. Spawn
        # avoids fork-inheriting the entire 48M dynamics/tokenizer address space.
        options["multiprocessing_context"] = "spawn"
    if training:
        batch_sampler = LocalityAwareBatchSampler(
            dataset, batch_size,
            episodes_per_batch=int(settings.get("episodes_per_batch", 4)),
            cache_size=int(settings.get("max_open_files", 8)),
            windows_per_locality_block=int(settings.get("windows_per_locality_block", 64)),
            phase_fractions=settings.get("phase_fractions"),
            terminal_fraction=float(settings.get("terminal_fraction", 0.0)),
            seed=int(settings.get("seed", 0)),
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, **options)
    # Build a fixed, coverage-spread subset from locality-ordered batches. A
    # global RandomSampler makes workers materialize a different full HDF5 file
    # for nearly every sample, saturating disk and defeating the episode cache.
    locality_sampler = LocalityAwareBatchSampler(
        dataset, batch_size,
        episodes_per_batch=int(settings.get("episodes_per_batch", 4)),
        cache_size=int(settings.get("max_open_files", 8)),
        windows_per_locality_block=int(settings.get("windows_per_locality_block", 64)),
        drop_last=False,
        seed=int(settings.get("seed", 0)) + 10000,
    )
    # Keep validation shapes identical to training. Variable final batches make
    # a compiled transformer recompile each forward variant (shortcut teacher,
    # causal transition, and action contrast), producing long utilization and
    # VRAM spikes that look like loader stalls.
    candidate_batches = [
        indices for indices in locality_sampler if len(indices) == batch_size
    ]
    count = min(len(candidate_batches), int(settings.get("validation_batches", 25)))
    positions = torch.linspace(0, len(candidate_batches) - 1, steps=count).round().long().tolist()
    fixed_batches = [candidate_batches[position] for position in positions]
    if bool(settings.get("terminal_validation", True)):
        # Append a distinct pass over every held-out episode ending. validate()
        # excludes these batches from the main loss and reports them separately.
        terminal_indices = [
            index for index, terminal in enumerate(dataset.terminal_windows) if terminal
        ]
        terminal_indices.sort(key=lambda index: (str(dataset.windows[index][0]), index))
        for start in range(0, len(terminal_indices), batch_size):
            indices = terminal_indices[start : start + batch_size]
            if len(indices) < batch_size:
                repeats = (batch_size - len(indices) + len(terminal_indices) - 1) // len(terminal_indices)
                indices = indices + (terminal_indices * repeats)[: batch_size - len(indices)]
            fixed_batches.append(indices)
    return DataLoader(dataset, batch_sampler=fixed_batches, **options)


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def encode_targets(batch, tokenizer, action_tokenizer, model, patch_size, *, amp=True):
    with torch.no_grad(), torch.autocast(
        device_type="cuda", enabled=batch["mask"].is_cuda and amp, dtype=torch.bfloat16
    ):
        observation_inputs = {
            key: batch[key] for key in ("mask", "proprio", "route", "timing")
        }
        latents = tokenizer.encode(observation_inputs)
        packed = model.pack(latents)
        initial_context, clean = packed[:, :1], packed[:, 1:]
        aligned_actions = batch["normalized_action"][:, patch_size - 1 :]
        aligned_timing = batch["action_timing"][:, patch_size - 1 :]
        if aligned_actions.shape[1] % patch_size:
            raise RuntimeError("aligned action transitions must form complete temporal patches")
        if action_tokenizer is not None:
            action_output = action_tokenizer(aligned_actions, aligned_timing)
            action_context = action_output.context
            macro_actions = action_output.mean.mean(dim=2)
        else:
            action_context = None
            macro_actions = aligned_actions.reshape(
                aligned_actions.shape[0], -1, patch_size, aligned_actions.shape[-1]
            ).mean(dim=2)
        macro_count = clean.shape[1]
        reward = batch["reward"][:, patch_size - 1 :].reshape(
            batch["reward"].shape[0], macro_count, patch_size
        ).sum(-1)
        # A race ends on completion or crash. The old collector encoded only
        # crashes in `continue`; `is_last` supplies successful finish boundaries.
        transition_continue = batch["continue"] * (~batch["is_last"]).to(batch["continue"].dtype)
        continuation = transition_continue[:, patch_size - 1 :].reshape(
            transition_continue.shape[0], macro_count, patch_size
        ).prod(-1)
        macro_state = macro_last(batch["task_state"], patch_size)
        targets = {"reward": reward, "continue": continuation, "task_state": macro_state}
        for key in ("privileged_state", "privileged_state_estimate", "privileged_state_estimate_std"):
            if key in batch:
                targets[key] = macro_last(batch[key], patch_size)
    return initial_context, clean, macro_actions, action_context, targets


def compute_objective(
    batch, model, tokenizer, action_tokenizer, patch_size, settings
):
    initial_context, clean, macro_actions, action_context, targets = encode_targets(
        batch, tokenizer, action_tokenizer, model, patch_size,
        amp=bool(settings.get("amp", True)),
    )
    _, shortcut_objective = shortcut_forcing_objective(
        model, clean, macro_actions, initial_context=initial_context,
        action_context=action_context,
        direct_weight=float(settings.get("shortcut_direct_weight", 0.25)),
        bootstrap_weight=float(settings.get("shortcut_bootstrap_weight", 1.0)),
    )
    # Train the deployable transition and every control head without access to
    # a corrupted version of the future. Each transition receives only the
    # previous clean latent, its action, and fresh source noise. This matches
    # recursive imagination and removes target leakage from reward/state heads.
    batch_size, horizon = clean.shape[:2]
    previous = torch.cat([initial_context, clean[:, :-1]], dim=1)
    flat_previous = previous.flatten(0, 1).unsqueeze(1)
    flat_target = clean.flatten(0, 1).unsqueeze(1)
    flat_actions = macro_actions.flatten(0, 1).unsqueeze(1)
    flat_action_context = (
        action_context.flatten(0, 1).unsqueeze(1) if action_context is not None else None
    )
    source = torch.randn_like(flat_target)
    one_jump_step = int(torch.tensor(model.k_max).log2().item())
    step_indices = torch.full(
        flat_target.shape[:2], one_jump_step, device=clean.device, dtype=torch.long
    )
    signal_indices = torch.zeros_like(step_indices)
    transition_flat = model(
        source, flat_actions, step_indices, signal_indices,
        initial_context=flat_previous, action_context=flat_action_context,
        causal_transition=bool(settings.get("causal_residual", False)),
    )

    def unflatten(value):
        if not isinstance(value, torch.Tensor):
            return value
        # torch.compile CUDA graphs reuse model output buffers. The optional
        # counterfactual-action forward below would otherwise overwrite these
        # views before reconstruction and head losses consume them (validation
        # reliably exposed this because inference mode enables CUDA graphs).
        return value.clone().reshape(batch_size, horizon, *value.shape[2:])

    output = replace(
        transition_flat,
        **{name: unflatten(value) for name, value in vars(transition_flat).items()},
    )
    transition_error = (
        transition_flat.predicted_packed_latents.float() - flat_target.float()
    ).square().flatten(2).mean(-1)
    persistence_error = (
        flat_previous.float() - flat_target.float()
    ).square().flatten(2).mean(-1)
    transition_direct = transition_error.mean()
    relative_floor = float(settings.get("transition_relative_floor", 0.005))
    transition_relative = (
        transition_error / persistence_error.detach().clamp_min(relative_floor)
    ).mean()
    persistence_margin = float(settings.get("persistence_margin", 0.10))
    persistence_loss = F.relu(
        transition_error - (1.0 - persistence_margin) * persistence_error
    ).mean()
    action_contrast = transition_direct.new_zeros(())
    wrong_action_ratio = transition_direct.new_zeros(())
    contrast_weight = float(settings.get("action_contrast_weight", 0.0))
    contrast_fraction = float(settings.get("action_contrast_fraction", 0.0))
    if contrast_weight > 0 and contrast_fraction > 0 and flat_target.shape[0] > 1:
        count = max(2, min(flat_target.shape[0], round(flat_target.shape[0] * contrast_fraction)))
        selected = torch.randperm(flat_target.shape[0], device=clean.device)[:count]
        wrong = selected.roll(1)
        wrong_output = model(
            source[selected], flat_actions[wrong], step_indices[selected], signal_indices[selected],
            initial_context=flat_previous[selected],
            action_context=flat_action_context[wrong] if flat_action_context is not None else None,
            causal_transition=bool(settings.get("causal_residual", False)),
        )
        wrong_error = (
            wrong_output.predicted_packed_latents.float() - flat_target[selected].float()
        ).square().flatten(2).mean(-1)
        correct_selected = transition_error[selected]
        action_margin = float(settings.get("action_contrast_margin", 0.10))
        action_contrast = F.relu(
            correct_selected.detach() * (1.0 + action_margin) - wrong_error
        ).mean()
        wrong_action_ratio = wrong_error.mean() / correct_selected.mean().detach().clamp_min(1e-8)
    latent_objective = (
        float(settings.get("shortcut_weight", 0.25)) * shortcut_objective
        + float(settings.get("transition_weight", 1.0)) * transition_direct
        + float(settings.get("transition_relative_weight", 0.0)) * transition_relative
        + float(settings.get("persistence_loss_weight", 1.0)) * persistence_loss
        + contrast_weight * action_contrast
    )
    predicted_latents = model.unpack(output.predicted_packed_latents)
    reconstruction = tokenizer.decode(predicted_latents)
    loss, metrics = dynamics_loss(
        output, clean, {**batch, **targets}, latent_loss=latent_objective,
        reconstruction=reconstruction,
        reconstruction_weight=float(settings.get("reconstruction_weight", 0.1)),
        continuation_negative_weight=float(settings.get("continuation_negative_weight", 10.0)),
        continuation_focal_gamma=float(settings.get("continuation_focal_gamma", 0.0)),
        continuation_loss_mode=str(settings.get("continuation_loss_mode", "weighted_bce")),
        continuation_brier_weight=float(settings.get("continuation_brier_weight", 0.0)),
        continuation_balanced_weight=float(settings.get("continuation_balanced_weight", 1.0)),
        continuation_calibration_weight=float(settings.get("continuation_calibration_weight", 0.0)),
        continuation_ranking_weight=float(settings.get("continuation_ranking_weight", 0.0)),
        continuation_ranking_margin=float(settings.get("continuation_ranking_margin", 0.0)),
        continuation_label_smoothing=float(settings.get("continuation_label_smoothing", 0.0)),
        latent_weight=float(settings.get("latent_weight", 1.0)),
        reward_weight=float(settings.get("reward_weight", 1.0)),
        continuation_weight=float(settings.get("continuation_weight", 1.0)),
        task_state_weight=float(settings.get("task_state_weight", 0.1)),
        task_state_objective=str(settings.get("task_state_objective", "absolute_uncertainty")),
        task_state_group_weights=settings.get("task_state_group_weights"),
        reward_tail_threshold=float(settings.get("reward_tail_threshold", 0.0)),
        reward_tail_weight=float(settings.get("reward_tail_weight", 0.0)),
        reward_decoded_weight=float(settings.get("reward_decoded_weight", 0.0)),
        reward_distribution_weight=float(settings.get("reward_distribution_weight", 1.0)),
        reward_direct_weight=float(settings.get("reward_direct_weight", 0.0)),
        reward_direct_tail_weight=float(settings.get("reward_direct_tail_weight", 0.0)),
        privileged_state_weight=float(settings.get("privileged_state_weight", 0.0)),
        estimator_state_weight=float(settings.get("estimator_state_weight", 0.0)),
        estimator_std_weight=float(settings.get("estimator_std_weight", 0.0)),
        privileged_state_center=settings.get("privileged_state_center"),
        privileged_state_scale=settings.get("privileged_state_scale"),
        estimator_state_center=settings.get("estimator_state_center"),
        estimator_state_scale=settings.get("estimator_state_scale"),
        estimator_std_center=settings.get("estimator_std_center"),
        estimator_std_scale=settings.get("estimator_std_scale"),
    )
    metrics.update({
        "shortcut": shortcut_objective.detach(),
        "transition_mse": transition_direct.detach(),
        "transition_relative": transition_relative.detach(),
        "persistence_mse": persistence_error.mean().detach(),
        "transition_to_persistence": (
            transition_direct / persistence_error.mean().clamp_min(1e-8)
        ).detach(),
        "persistence_margin_loss": persistence_loss.detach(),
        "action_contrast": action_contrast.detach(),
        "wrong_action_error_ratio": wrong_action_ratio.detach(),
    })
    return loss, metrics, reconstruction, targets


@torch.inference_mode()
def validate(model, tokenizer, action_tokenizer, loader, device, patch_size, settings):
    model.eval()
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    terminal_totals: dict[str, float] = {}
    terminal_weights: dict[str, float] = {}
    preview = None
    primary_batches = int(settings.get("validation_batches", 25))
    terminal_weighted = {"terminal_probability", "terminal_bce", "terminal_recall"}
    nonterminal_weighted = {
        "nonterminal_bce", "nonterminal_false_stop_probability"
    }

    def accumulate(target_totals, target_weights, metrics, batch_size):
        terminal_count = float(metrics.get("terminal_count", 0.0))
        nonterminal_count = float(metrics.get("nonterminal_count", 0.0))
        for key, value in metrics.items():
            if key in {"terminal_count", "nonterminal_count"}:
                target_totals[key] = target_totals.get(key, 0.0) + float(value)
                continue
            weight = (
                terminal_count if key in terminal_weighted
                else nonterminal_count if key in nonterminal_weighted
                else float(batch_size)
            )
            if weight <= 0:
                continue
            target_totals[key] = target_totals.get(key, 0.0) + float(value) * weight
            target_weights[key] = target_weights.get(key, 0.0) + weight

    devices = [torch.cuda.current_device()] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(settings.get("validation_seed", 20262030)))
        for batch_index, batch in enumerate(loader):
            batch_size = int(batch["mask"].shape[0])
            batch = move_batch(batch, device)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16,
                enabled=device.startswith("cuda") and bool(settings.get("amp", True)),
            ):
                _, metrics, reconstruction, _ = compute_objective(
                    batch, model, tokenizer, action_tokenizer, patch_size, settings
                )
            if batch_index < primary_batches:
                accumulate(totals, weights, metrics, batch_size)
                if preview is None:
                    target = macro_last(batch["mask"], patch_size)[:, 1:]
                    preview = {
                        "reconstruction_mask": reconstruction.mask_logits[0, -1, 0].sigmoid(),
                        "target_mask": target[0, -1, 0],
                    }
            else:
                accumulate(terminal_totals, terminal_weights, metrics, batch_size)
    model.train()
    result = {
        key: value if key.endswith("_count") else value / max(1.0, weights.get(key, 0.0))
        for key, value in totals.items()
    }
    for key in (
        "terminal_probability", "terminal_bce", "terminal_recall",
        "nonterminal_false_stop_probability", "nonterminal_bce",
        "terminal_count", "nonterminal_count",
    ):
        if key in terminal_totals:
            value = terminal_totals[key]
            result[f"terminal_eval_{key}"] = (
                value if key.endswith("_count")
                else value / max(1.0, terminal_weights.get(key, 0.0))
            )
    if bool(settings.get("pmpo_selection_score", False)):
        reward_term = result.get("reward_distribution_mae", result.get("reward_mae", 0.0))
        terminal_term = result.get("terminal_eval_terminal_bce", 0.0)
        nonterminal_term = result.get("terminal_eval_nonterminal_bce", 0.0)
        result["pmpo_head_score"] = (
            float(settings.get("pmpo_selection_reward_weight", 1.0)) * reward_term
            + float(settings.get("pmpo_selection_terminal_weight", 0.5)) * terminal_term
            + float(settings.get("pmpo_selection_nonterminal_weight", 0.5)) * nonterminal_term
        )
    return result, preview


def main() -> None:
    args = arguments()
    settings = args.full_config.get("dynamics", {})
    settings.update(
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        workers=args.workers,
        learning_rate=args.learning_rate,
        device=args.device,
        attention=args.attention,
        resume=args.resume,
        seed=args.seed,
    )
    manager = CheckpointManager.from_config(args.full_config)
    logger = init_wandb(args.full_config)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if str(settings.get("precision", "bf16")).lower() != "bf16":
        raise SystemExit("dynamics training currently requires precision: bf16")
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(settings.get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", True))
    set_attention_backend(args.attention)
    data = Path(args.data)
    train_paths, validation_paths = stratified_episode_split(
        data, float(settings.get("validation_fraction", 0.1)), args.seed
    )

    tokenizer_checkpoint = torch.load(args.tokenizer, map_location=args.device, weights_only=False)
    tokenizer_model_config = tokenizer_checkpoint.get("model_config", {})
    tokenizer_config = dict(tokenizer_model_config.get("encoder", tokenizer_model_config))
    tokenizer = MultiModalTokenizer(**tokenizer_config).to(args.device)
    tokenizer.load_state_dict(tokenizer_checkpoint.get("encoder", tokenizer_checkpoint["model"]))
    tokenizer.eval().requires_grad_(False)
    patch_size = tokenizer.temporal_patch_size
    if (args.sequence_length + 1) % patch_size:
        raise SystemExit("sequence_length + 1 observations must be divisible by temporal_patch_size")

    action_tokenizer = None
    if args.action_tokenizer:
        action_checkpoint = torch.load(args.action_tokenizer, map_location=args.device, weights_only=False)
        action_model_config = action_checkpoint.get("model_config", {})
        action_config = dict(action_model_config.get("tokenizer", action_model_config))
        action_tokenizer = ActionTokenizer(**action_config).to(args.device)
        action_tokenizer.load_state_dict(action_checkpoint.get("tokenizer", action_checkpoint["model"]))
        action_tokenizer.eval().requires_grad_(False)
        if action_tokenizer.temporal_patch_size != patch_size:
            raise SystemExit("observation and action tokenizer patch sizes must match")

    dataset_options = dict(
        root=data, sequence_length=args.sequence_length, stride=int(settings.get("stride", 1)),
        mode="dynamics", mask_size=tuple(tokenizer_config.get("image_size", (128, 160))),
        max_open_files=int(settings.get("max_open_files", 8)),
        validate_contents=bool(settings.get("validate_contents", False)),
        include_privileged=True,
        cache_in_memory=bool(settings.get("cache_in_memory", True)),
    )
    train_dataset = DreamerSequenceDataset(paths=train_paths, **dataset_options)
    validation_dataset = DreamerSequenceDataset(paths=validation_paths, **dataset_options) if validation_paths else None
    loader = make_loader(train_dataset, settings, training=True)
    validation_loader = make_loader(validation_dataset, settings, training=False) if validation_dataset else None

    model_config = {
        "d_model": 448, "d_bottleneck": tokenizer.d_bottleneck,
        "n_latents": tokenizer.n_latents, "n_spatial": 2, "n_heads": 7, "depth": 10,
    }
    model_config.update(settings.get("model", {}))
    model = RaceDynamics(**model_config).to(args.device)
    initialize_from = settings.get("initialize_from")
    if initialize_from:
        initialized = torch.load(initialize_from, map_location=args.device, weights_only=False)
        source_state = initialized["model"]
        target_state = model.state_dict()
        compatible = {
            key: value for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        incompatible = model.load_state_dict(compatible, strict=False)
        initialized_parameters = sum(value.numel() for value in compatible.values())
        print(
            f"initialized_from={initialize_from} compatible_parameters={initialized_parameters:,} "
            f"new_keys={len(incompatible.missing_keys)} skipped_keys={len(source_state) - len(compatible)}",
            flush=True,
        )
    if bool(settings.get("freeze_non_pmpo_parameters", False)):
        if not bool(model_config.get("pmpo_auxiliary_heads", False)):
            raise ValueError("freeze_non_pmpo_parameters requires pmpo_auxiliary_heads=true")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("pmpo_"))
        print("froze_non_pmpo_parameters=true", flush=True)
    if settings.get("compile", False) and not args.benchmark_steps:
        model.compile(mode=str(settings.get("compile_mode", "default")), dynamic=False)
    base_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (head_parameters if name.startswith("pmpo_") else base_parameters).append(parameter)
    optimizer_groups = []
    if base_parameters:
        optimizer_groups.append({
            "params": base_parameters,
            "lr": args.learning_rate,
            "weight_decay": float(settings.get("weight_decay", 1e-4)),
            "group_name": "backbone",
        })
    if head_parameters:
        optimizer_groups.append({
            "params": head_parameters,
            "lr": float(settings.get("pmpo_head_learning_rate", args.learning_rate)),
            "weight_decay": float(settings.get("pmpo_head_weight_decay", settings.get("weight_decay", 1e-4))),
            "group_name": "pmpo_heads",
        })
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        fused=bool(settings.get("fused_optimizer", True)) and args.device.startswith("cuda"),
    )
    warmup_steps = int(settings.get("warmup_steps", 0))
    min_lr_ratio = float(settings.get("min_lr_ratio", 1.0))

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        if min_lr_ratio >= 1.0:
            return 1.0
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    start = 0
    checkpoint = manager.resume(bool(args.resume), map_location=args.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start = int(checkpoint.get("step", 0))
        restore_rng_state(checkpoint.get("rng_state"))

    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"train_episodes={len(train_paths)} validation_episodes={len(validation_paths)} "
        f"train_windows={len(train_dataset)} trainable_parameters={parameters:,}", flush=True
    )
    iterator = iter(loader)
    log_interval = int(settings.get("log_interval", 25))
    validation_interval = int(settings.get("validation_interval", 250))
    checkpoint_interval = int(settings.get("checkpoint_interval", 250))
    last_log_time = time.perf_counter()
    early_stopping_patience = int(settings.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(settings.get("early_stopping_min_delta", 0.0))
    early_stopping_best = math.inf if manager.mode == "min" else -math.inf
    early_stopping_stale = 0
    for step in range(start, args.steps):
        data_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        data_seconds = time.perf_counter() - data_started
        batch = move_batch(batch, args.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=args.device.startswith("cuda") and bool(settings.get("amp", True)),
        ):
            loss, metrics, _, _ = compute_objective(
                batch, model, tokenizer, action_tokenizer, patch_size, settings
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings.get("grad_clip", 10.0))
        )
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-6)
            throughput = log_interval if step > start else 1
            logged = {
                **metrics,
                "gradient_norm": gradient_norm.detach(),
                "steps_per_second": throughput / elapsed,
                "samples_per_second": throughput * args.batch_size / elapsed,
                "data_time_ms": data_seconds * 1000.0,
                "gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith("cuda") else 0.0,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            for group in optimizer.param_groups:
                if group.get("group_name") == "pmpo_heads":
                    logged["pmpo_head_learning_rate"] = group["lr"]
            print(f"step={step + 1} " + " ".join(f"{key}={float(value):.5f}" for key, value in logged.items()), flush=True)
            logger.log_train(logged, step + 1)
            last_log_time = now

        validation_metrics = None
        should_stop = False
        if validation_loader and ((step + 1) % validation_interval == 0 or step + 1 == args.steps):
            validation_metrics, preview = validate(
                model, tokenizer, action_tokenizer, validation_loader,
                args.device, patch_size, settings,
            )
            print(
                f"step={step + 1} validation "
                + " ".join(f"{key}={value:.5f}" for key, value in validation_metrics.items()),
                flush=True,
            )
            logger.log_eval(validation_metrics, step + 1)
            if preview:
                logger.log_images(preview, step + 1)
            manager.save_eval(
                validation_metrics, step=step + 1,
                summary="Held-out shortcut-forcing dynamics, symlog two-hot reward, finish/crash continuation, task-state, and decoded segmentation reconstruction evaluation.",
            )
            if early_stopping_patience > 0 and manager.monitor in validation_metrics:
                score = float(validation_metrics[manager.monitor])
                improved = (
                    score < early_stopping_best - early_stopping_min_delta
                    if manager.mode == "min"
                    else score > early_stopping_best + early_stopping_min_delta
                )
                if improved:
                    early_stopping_best = score
                    early_stopping_stale = 0
                else:
                    early_stopping_stale += 1
                should_stop = early_stopping_stale >= early_stopping_patience

        if (step + 1) % checkpoint_interval == 0 or step + 1 == args.steps or validation_metrics is not None:
            manager.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": capture_rng_state(), "model_config": model_config,
                "training_config": args.full_config,
                "observation_tokenizer": str(args.tokenizer),
                "action_tokenizer": str(args.action_tokenizer) if args.action_tokenizer else None,
            }, step=step + 1, metrics=validation_metrics, rank=validation_metrics is not None)
        if should_stop:
            print(
                f"early_stopping step={step + 1} monitor={manager.monitor} "
                f"best={early_stopping_best:.6f} stale_validations={early_stopping_stale}",
                flush=True,
            )
            break
    logger.finish()


if __name__ == "__main__":
    main()
