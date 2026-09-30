#!/usr/bin/env python3
"""Behavior-clone a Gaussian CTBR actor on frozen Dreamer posterior latents."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset
import yaml

from starscream.actor_critic import GaussianActor, actor_state_dimension, build_actor_state
from starscream.checkpoint_manager import CheckpointManager, capture_rng_state, restore_rng_state
from starscream.dataloader import DreamerSequenceDataset, stratified_episode_split
from starscream.dynamics import RaceDynamics
from starscream.loss import gaussian_nll
from starscream.tokenizer import ActionTokenizer, MultiModalTokenizer
from starscream.wandb import init_wandb


class RecoveryAwareBatchSampler(Sampler[list[int]]):
    """Sample expert recovery targets from step one while retaining HDF5 locality."""

    def __init__(
        self,
        dataset: DreamerSequenceDataset,
        indices: list[int],
        phases: list[int],
        *,
        batch_size: int,
        recovery_fraction_start: float,
        recovery_fraction_end: float,
        tail_fraction_start: float,
        tail_fraction_end: float,
        curriculum_steps: int,
        episodes_per_batch: int,
        seed: int,
    ) -> None:
        if len(indices) != len(phases):
            raise ValueError("BC indices and phases must have equal lengths")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.recovery_fraction_start = float(recovery_fraction_start)
        self.recovery_fraction_end = float(recovery_fraction_end)
        self.tail_fraction_start = float(tail_fraction_start)
        self.tail_fraction_end = float(tail_fraction_end)
        self.curriculum_steps = int(curriculum_steps)
        self.episodes_per_batch = max(1, int(episodes_per_batch))
        self.seed = int(seed)
        self.batches_seen = 0
        self.epoch = 0
        self.recovery_by_path: dict[Path, list[int]] = {}
        self.tail_by_path: dict[Path, list[int]] = {}
        self.other_by_path: dict[Path, list[int]] = {}
        for index, phase in zip(indices, phases):
            path, _ = dataset.windows[index]
            destination = (
                self.recovery_by_path if phase == 3
                else self.tail_by_path if phase in {1, 2, 4}
                else self.other_by_path
            )
            destination.setdefault(path, []).append(index)
        if not self.recovery_by_path:
            raise ValueError("no valid expert recovery targets were found")
        if not self.other_by_path:
            raise ValueError("no non-recovery expert targets were found")
        if not self.tail_by_path:
            raise ValueError("no deviation-onset/near-crash/collision expert targets were found")
        self.recovery_paths = list(self.recovery_by_path)
        self.tail_paths = list(self.tail_by_path)
        self.other_paths = list(self.other_by_path)
        self.length = max(1, len(indices) // self.batch_size)

    def __len__(self) -> int:
        return self.length

    def _fractions(self) -> tuple[float, float]:
        if self.curriculum_steps <= 0:
            return self.recovery_fraction_end, self.tail_fraction_end
        progress = min(1.0, self.batches_seen / self.curriculum_steps)
        recovery = self.recovery_fraction_start + progress * (
            self.recovery_fraction_end - self.recovery_fraction_start
        )
        tail = self.tail_fraction_start + progress * (
            self.tail_fraction_end - self.tail_fraction_start
        )
        return recovery, tail

    @staticmethod
    def _draw(
        rng: random.Random,
        pools: dict[Path, list[int]],
        paths: list[Path],
        count: int,
        path_budget: int,
    ) -> list[int]:
        if count <= 0:
            return []
        chosen_paths = rng.sample(paths, k=min(path_budget, len(paths)))
        return [rng.choice(pools[chosen_paths[offset % len(chosen_paths)]]) for offset in range(count)]

    def __iter__(self):
        rng = random.Random(self.seed + 1009 * self.epoch)
        self.epoch += 1
        path_budget = max(1, self.episodes_per_batch // 3)
        for _ in range(self.length):
            recovery_fraction, tail_fraction = self._fractions()
            recovery_count = min(
                self.batch_size - 2,
                max(1, round(self.batch_size * recovery_fraction)),
            )
            tail_count = min(
                self.batch_size - recovery_count - 1,
                max(1, round(self.batch_size * tail_fraction)),
            )
            batch = self._draw(
                rng, self.recovery_by_path, self.recovery_paths,
                recovery_count, path_budget,
            )
            batch.extend(self._draw(
                rng, self.tail_by_path, self.tail_paths,
                tail_count, path_budget,
            ))
            batch.extend(self._draw(
                rng, self.other_by_path, self.other_paths,
                self.batch_size - recovery_count - tail_count, path_budget,
            ))
            rng.shuffle(batch)
            self.batches_seen += 1
            yield batch


def load_frozen_actor_stack(
    dynamics_path: Path | None,
    observation_path: Path | None,
    action_path: Path | None,
    device: str,
):
    """Load the deployable BC stack and an optional compatible world model."""

    dynamics_checkpoint = (
        torch.load(dynamics_path, map_location="cpu", weights_only=False)
        if dynamics_path is not None else None
    )
    if observation_path is None:
        if dynamics_checkpoint is None:
            raise ValueError("actor BC requires an observation tokenizer checkpoint")
        observation_path = Path(dynamics_checkpoint["observation_tokenizer"])
    action_reference = action_path or (
        Path(dynamics_checkpoint["action_tokenizer"])
        if dynamics_checkpoint and dynamics_checkpoint.get("action_tokenizer") else None
    )
    if action_reference is None:
        raise ValueError("actor BC requires an action tokenizer checkpoint")

    observation_checkpoint = torch.load(observation_path, map_location="cpu", weights_only=False)
    observation_model_config = observation_checkpoint.get("model_config", {})
    observation_config = dict(
        observation_model_config.get("encoder", observation_model_config)
    )
    observation = MultiModalTokenizer(**observation_config)
    observation.load_state_dict(
        observation_checkpoint.get("encoder", observation_checkpoint["model"])
    )
    observation.eval().requires_grad_(False).to(device)

    action_checkpoint = torch.load(action_reference, map_location="cpu", weights_only=False)
    action_model_config = action_checkpoint.get("model_config", {})
    action_config = dict(action_model_config.get("tokenizer", action_model_config))
    action = ActionTokenizer(**action_config)
    action.load_state_dict(action_checkpoint.get("tokenizer", action_checkpoint["model"]))
    action.eval().requires_grad_(False).to(device)

    dynamics = None
    if dynamics_checkpoint is not None:
        dynamics = RaceDynamics(**dynamics_checkpoint["model_config"])
        dynamics.load_state_dict(dynamics_checkpoint["model"])
        dynamics.eval().requires_grad_(False)
        if dynamics.n_latents != observation.n_latents or dynamics.d_bottleneck != observation.d_bottleneck:
            raise ValueError("dynamics and observation tokenizer latent shapes do not match")
    if action.temporal_patch_size != observation.temporal_patch_size:
        raise ValueError("observation and action tokenizer temporal patches do not match")
    return (
        dynamics, observation, action, observation_config,
        observation_path, action_reference, dynamics_checkpoint,
    )


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def actor_bc_objective(
    output,
    target: torch.Tensor,
    settings: dict,
    step: int,
    target_action_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean-first BC with independently calibrated policy uncertainty."""

    mean = output.mean.float()
    target = target.float()
    query_weights = mean.new_tensor(
        settings.get("query_weights", [1.0] * target.shape[1])
    )
    if query_weights.shape != (target.shape[1],) or (query_weights <= 0).any():
        raise ValueError("query_weights must contain one positive value per action query")
    query_weights = query_weights / query_weights.mean()
    broadcast_weights = query_weights.view(1, -1, 1)
    mean_objective = str(settings.get("mean_objective", "smooth_l1"))
    if mean_objective == "smooth_l1":
        mean_elementwise = F.smooth_l1_loss(
            mean,
            target,
            beta=float(settings.get("mean_huber_beta", 0.1)),
            reduction="none",
        )
    elif mean_objective == "mse":
        mean_elementwise = (mean - target).square()
    else:
        raise ValueError("mean_objective must be smooth_l1 or mse")
    mean_loss = (mean_elementwise * broadcast_weights).mean()

    variance_start = int(settings.get("variance_learning_start_step", 0))
    variance_active = step >= variance_start
    gnll_elementwise = gaussian_nll(
        mean, output.log_std.float(), target, reduction="none"
    )
    diagnostic_gnll = (gnll_elementwise * broadcast_weights).mean()
    if variance_active:
        # The variance tower calibrates against current residuals without
        # changing the mean or shared temporal representation.
        variance_nll = (
            gaussian_nll(
                mean.detach(), output.log_std.float(), target, reduction="none"
            ) * broadcast_weights
        ).mean()
    else:
        variance_nll = mean_loss.new_zeros(())
    loss = (
        float(settings.get("mean_loss_weight", 1.0)) * mean_loss
        + float(settings.get("variance_nll_weight", 0.1)) * variance_nll
    )
    token_loss = mean_loss.new_zeros(())
    if output.action_tokens is not None:
        if target_action_tokens is None:
            raise ValueError("action-token actor requires target action tokens")
        token_loss = F.smooth_l1_loss(
            output.action_tokens.float(),
            target_action_tokens.detach().float(),
            beta=float(settings.get("action_token_huber_beta", 0.05)),
        )
        loss = loss + float(settings.get("action_token_loss_weight", 0.1)) * token_loss
    metrics = {
        "loss": loss.detach(),
        "mean_loss": mean_loss.detach(),
        "mean_mse": F.mse_loss(mean, target).detach(),
        "gnll": diagnostic_gnll.detach(),
        "variance_nll": variance_nll.detach(),
        "mean_std": output.log_std.detach().float().exp().mean(),
        "variance_active": mean_loss.new_tensor(float(variance_active)),
        "action_token_loss": token_loss.detach(),
    }
    for query in range(target.shape[1]):
        metrics[f"query_{query}_mean_mse"] = F.mse_loss(
            mean[:, query], target[:, query]
        ).detach()
        metrics[f"query_{query}_gnll"] = gnll_elementwise[:, query].mean().detach()
    return loss, metrics


@torch.no_grad()
def validate(
    actor: GaussianActor,
    encoder: MultiModalTokenizer,
    action_tokenizer: ActionTokenizer,
    loader: DataLoader,
    *,
    history: int,
    action_horizon: int,
    device: str,
    amp: bool,
    max_batches: int,
    recovery_selection_weight: float,
    critical_selection_weight: float,
    selection_calibration_weight: float,
) -> dict[str, float]:
    actor.eval()
    totals = {
        "gnll": 0.0, "mse": 0.0, "count": 0.0,
        "recovery_gnll": 0.0, "recovery_mse": 0.0, "recovery_count": 0.0,
        "critical_gnll": 0.0, "critical_mse": 0.0, "critical_count": 0.0,
        "coverage_1": 0.0, "coverage_2": 0.0, "elements": 0.0,
        "recovery_coverage_1": 0.0, "recovery_coverage_2": 0.0,
        "recovery_elements": 0.0,
    }
    for query in range(action_horizon):
        totals.update({
            f"query_{query}_gnll": 0.0,
            f"query_{query}_mse": 0.0,
            f"query_{query}_coverage_1": 0.0,
            f"query_{query}_coverage_2": 0.0,
            f"recovery_query_{query}_mse": 0.0,
            f"critical_query_{query}_mse": 0.0,
        })
    target_start = history - 1
    for batch_index, raw_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        state = build_actor_state(encoder, batch, history, device, amp)
        target = batch["commanded_action"][:, target_start:target_start + action_horizon]
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=device.startswith("cuda") and amp,
        ):
            output = actor(state, action_tokenizer)
        elementwise = gaussian_nll(
            output.mean.float(), output.log_std.float(), target.float(), reduction="none"
        )
        per_sample_gnll = elementwise.mean((1, 2))
        per_sample_mse = (output.mean.float() - target.float()).square().mean((1, 2))
        absolute_error = (output.mean.float() - target.float()).abs()
        standard_deviation = output.log_std.float().exp()
        phase = batch["tail_phase"][:, target_start:target_start + action_horizon].amax(1)
        recovery = phase == 3
        critical = phase >= 2
        totals["gnll"] += float(per_sample_gnll.sum())
        totals["mse"] += float(per_sample_mse.sum())
        totals["count"] += len(per_sample_gnll)
        totals["recovery_gnll"] += float(per_sample_gnll[recovery].sum())
        totals["recovery_mse"] += float(per_sample_mse[recovery].sum())
        totals["recovery_count"] += int(recovery.sum())
        totals["critical_gnll"] += float(per_sample_gnll[critical].sum())
        totals["critical_mse"] += float(per_sample_mse[critical].sum())
        totals["critical_count"] += int(critical.sum())
        totals["coverage_1"] += int((absolute_error <= standard_deviation).sum())
        totals["coverage_2"] += int((absolute_error <= 2.0 * standard_deviation).sum())
        totals["elements"] += absolute_error.numel()
        totals["recovery_coverage_1"] += int(
            (absolute_error[recovery] <= standard_deviation[recovery]).sum()
        )
        totals["recovery_coverage_2"] += int(
            (absolute_error[recovery] <= 2.0 * standard_deviation[recovery]).sum()
        )
        totals["recovery_elements"] += absolute_error[recovery].numel()
        for query in range(action_horizon):
            query_error = (
                output.mean[:, query].float() - target[:, query].float()
            ).square().mean(1)
            totals[f"query_{query}_gnll"] += float(
                elementwise[:, query].mean(1).sum()
            )
            totals[f"query_{query}_mse"] += float(query_error.sum())
            totals[f"query_{query}_coverage_1"] += int(
                (absolute_error[:, query] <= standard_deviation[:, query]).sum()
            )
            totals[f"query_{query}_coverage_2"] += int(
                (absolute_error[:, query] <= 2.0 * standard_deviation[:, query]).sum()
            )
            totals[f"recovery_query_{query}_mse"] += float(query_error[recovery].sum())
            totals[f"critical_query_{query}_mse"] += float(query_error[critical].sum())
    actor.train()
    count = max(1.0, totals["count"])
    recovery_count = max(1.0, totals["recovery_count"])
    critical_count = max(1.0, totals["critical_count"])
    elements = max(1.0, totals["elements"])
    recovery_elements = max(1.0, totals["recovery_elements"])
    metrics = {
        "gnll": totals["gnll"] / count,
        "mean_mse": totals["mse"] / count,
        "recovery_gnll": totals["recovery_gnll"] / recovery_count,
        "recovery_mean_mse": totals["recovery_mse"] / recovery_count,
        "recovery_fraction": totals["recovery_count"] / count,
        "critical_gnll": totals["critical_gnll"] / critical_count,
        "critical_mean_mse": totals["critical_mse"] / critical_count,
        "critical_fraction": totals["critical_count"] / count,
        "coverage_1sigma": totals["coverage_1"] / elements,
        "coverage_2sigma": totals["coverage_2"] / elements,
        "recovery_coverage_1sigma": totals["recovery_coverage_1"] / recovery_elements,
        "recovery_coverage_2sigma": totals["recovery_coverage_2"] / recovery_elements,
    }
    metrics["recovery_calibration_error"] = (
        abs(metrics["recovery_coverage_1sigma"] - 0.682689)
        + abs(metrics["recovery_coverage_2sigma"] - 0.954500)
    )
    query_elements = count * 4.0
    for query in range(action_horizon):
        metrics.update({
            f"query_{query}_gnll": totals[f"query_{query}_gnll"] / count,
            f"query_{query}_mean_mse": totals[f"query_{query}_mse"] / count,
            f"query_{query}_coverage_1sigma": totals[f"query_{query}_coverage_1"] / query_elements,
            f"query_{query}_coverage_2sigma": totals[f"query_{query}_coverage_2"] / query_elements,
            f"recovery_query_{query}_mean_mse": totals[f"recovery_query_{query}_mse"] / recovery_count,
            f"critical_query_{query}_mean_mse": totals[f"critical_query_{query}_mse"] / critical_count,
        })
    recovery_weight = float(recovery_selection_weight)
    critical_weight = float(critical_selection_weight)
    overall_weight = 1.0 - recovery_weight - critical_weight
    if overall_weight < 0:
        raise ValueError("recovery and critical selection weights must sum to at most one")
    metrics["selection_score"] = (
        overall_weight * metrics["mean_mse"]
        + recovery_weight * metrics["recovery_mean_mse"]
        + critical_weight * metrics["critical_mean_mse"]
        + float(selection_calibration_weight) * metrics["recovery_calibration_error"]
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--dynamics", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--action-tokenizer", type=Path)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--action-horizon", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        with preliminary.config.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        parser.set_defaults(**config.get("actor_bc", {}))
    else:
        config = {
            "output_root": "outputs",
            "wandb": {"enabled": False, "run_name": "actor-bc-cli"},
            "checkpoint": {"monitor": "selection_score", "mode": "min", "top_k": 3},
        }
    args = parser.parse_args()
    if args.smoke:
        args.steps = 2
        args.batch_size = min(args.batch_size, 8)
        args.workers = 0
        args.checkpoint_interval = 1
        args.log_interval = 1
        config.setdefault("actor_bc", {})["validation_batches"] = 1
        config.setdefault("actor_bc", {})["validation_batch_size"] = min(
            int(config.get("actor_bc", {}).get("validation_batch_size", 8)), 8
        )
        run_name = str(
            config.get("checkpoint", {}).get(
                "run_name", config.get("wandb", {}).get("run_name", "actor-bc")
            )
        )
        config.setdefault("checkpoint", {})["run_name"] = f"{run_name}-smoke"
        config["wandb"] = {"enabled": False, "run_name": f"{run_name}-smoke"}
    if args.data is None:
        parser.error("--data is required (directly or via --config)")
    if args.tokenizer is None and args.dynamics is None:
        parser.error("--tokenizer is required when --dynamics is not configured")
    if args.action_tokenizer is None and args.dynamics is None:
        parser.error("--action-tokenizer is required when --dynamics is not configured")
    settings = config.get("actor_bc", {})
    manager = CheckpointManager.from_config(config)
    logger = init_wandb(config)
    torch.manual_seed(int(settings.get("seed", 0)))
    random.seed(int(settings.get("seed", 0)))
    torch.set_float32_matmul_precision("high")
    amp = bool(settings.get("amp", True))
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(settings.get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", True))

    (
        frozen_dynamics, encoder, action_tokenizer, encoder_config,
        observation_path, action_path, dynamics_checkpoint,
    ) = load_frozen_actor_stack(
        args.dynamics, args.tokenizer, args.action_tokenizer, args.device
    )
    patch = encoder.temporal_patch_size
    if args.history % patch:
        raise SystemExit("history must be divisible by the observation temporal patch size")
    if args.action_horizon != action_tokenizer.temporal_patch_size:
        raise SystemExit(
            "action_horizon must equal the frozen action tokenizer patch size; "
            "one actor decision must map to one world-model macro action"
        )

    validation_fraction = float(settings.get("validation_fraction", 0.1))
    split_seed = int(settings.get("validation_seed", settings.get("seed", 0)))
    tracks = settings.get("tracks", settings.get("track"))
    train_paths, validation_paths = stratified_episode_split(
        args.data, validation_fraction, split_seed, tracks=tracks
    )
    dataset_options = dict(
        root=args.data,
        sequence_length=args.history + args.action_horizon - 1,
        stride=int(settings.get("stride", patch)),
        mode="dynamics",
        mask_size=tuple(encoder_config.get("image_size", (128, 160))),
        max_open_files=int(settings.get("max_open_files", 12)),
        validate_contents=bool(settings.get("validate_contents", False)),
        include_privileged=False,
        cache_in_memory=False,
        require_controller_valid=False,
        route_source=str(settings.get("route_source", "flight_plan")),
        route_target_source=str(settings.get("route_target_source", "input")),
    )
    if int(getattr(encoder, "estimate_dim", 0)) > 0:
        dataset_options.update(
            deployment_estimate_source=str(
                settings.get("deployment_estimate_source", "proxy_v1")
            ),
            deployment_estimate_seed=int(
                settings.get("deployment_estimate_seed", 2026081718)
            ),
        )
    train_dataset = DreamerSequenceDataset(paths=train_paths, **dataset_options)
    validation_dataset = (
        DreamerSequenceDataset(paths=validation_paths, **dataset_options)
        if validation_paths else None
    )
    train_indices, train_phases = train_dataset.behavior_cloning_windows(
        history=args.history, action_horizon=args.action_horizon
    )
    if not train_indices:
        raise SystemExit("no valid expert BC targets were found")
    batch_sampler = RecoveryAwareBatchSampler(
        train_dataset, train_indices, train_phases,
        batch_size=args.batch_size,
        recovery_fraction_start=float(settings.get("recovery_fraction_start", 0.5)),
        recovery_fraction_end=float(settings.get("recovery_fraction_end", 0.3)),
        tail_fraction_start=float(settings.get("tail_fraction_start", 0.15)),
        tail_fraction_end=float(settings.get("tail_fraction_end", 0.10)),
        curriculum_steps=int(settings.get("recovery_curriculum_steps", 5000)),
        episodes_per_batch=int(settings.get("episodes_per_batch", 4)),
        seed=int(settings.get("seed", 0)),
    )
    loader_options = dict(
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    if args.workers > 0:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=int(settings.get("prefetch_factor", 2)),
            multiprocessing_context="spawn",
        )
    loader = DataLoader(train_dataset, batch_sampler=batch_sampler, **loader_options)
    validation_loader = None
    validation_indices: list[int] = []
    validation_phases: list[int] = []
    if validation_dataset:
        validation_indices, validation_phases = validation_dataset.behavior_cloning_windows(
            history=args.history, action_horizon=args.action_horizon
        )
        if validation_indices:
            validation_loader = DataLoader(
                Subset(validation_dataset, validation_indices),
                batch_size=int(settings.get("validation_batch_size", args.batch_size)),
                shuffle=False,
                drop_last=False,
                **loader_options,
            )

    input_dim = actor_state_dimension(encoder)
    actor_config = {
        "input_dim": input_dim,
        "hidden_dim": int(settings.get("hidden_dim", 512)),
        "action_dim": 4,
        "action_horizon": args.action_horizon,
        "min_log_std": float(settings.get("min_log_std", -5.0)),
        "max_log_std": float(settings.get("max_log_std", 1.0)),
        "temporal_depth": int(settings.get("temporal_depth", 3)),
        "temporal_heads": int(settings.get("temporal_heads", 8)),
        "temporal_context": args.history // patch,
        "temporal_dropout": float(settings.get("temporal_dropout", 0.0)),
        "separate_std_head": bool(settings.get("separate_std_head", True)),
        "std_hidden_dim": int(settings.get("std_hidden_dim", 256)),
        "std_backbone_gradient_scale": float(settings.get("std_backbone_gradient_scale", 0.0)),
        "initial_log_std": float(settings.get("initial_log_std", -1.5)),
        "decoder_type": str(settings.get("decoder_type", "monolithic")),
        "action_query_heads": int(settings.get("action_query_heads", settings.get("temporal_heads", 8))),
        "action_query_depth": int(settings.get("action_query_depth", 1)),
        "action_query_dropout": float(settings.get("action_query_dropout", 0.0)),
        "action_token_count": (
            action_tokenizer.n_tokens
            if bool(settings.get("use_action_tokenizer_decoder", False)) else 0
        ),
        "action_token_dim": (
            action_tokenizer.d_latent
            if bool(settings.get("use_action_tokenizer_decoder", False)) else 0
        ),
        "head_hidden_dim": int(
            settings.get("head_hidden_dim", settings.get("hidden_dim", 512))
        ),
        "head_depth": int(settings.get("head_depth", 2)),
    }
    actor = GaussianActor(**actor_config).to(args.device)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=float(settings.get("learning_rate", 3e-4)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
        fused=bool(settings.get("fused_optimizer", True)) and args.device.startswith("cuda"),
    )
    warmup_steps = int(settings.get("warmup_steps", 0))
    min_lr_ratio = float(settings.get("min_lr_ratio", 0.1))

    def lr_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    start = 0
    checkpoint = manager.resume(bool(args.resume), map_location=args.device)
    if checkpoint is not None:
        actor.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start = int(checkpoint.get("step", 0))
        restore_rng_state(checkpoint.get("rng_state"))

    phase_counts = {phase: train_phases.count(phase) for phase in range(5)}
    print(
        f"actor_input=temporal_latents_deltas_decoded_proprio_exact_actions_state_estimate input_dim={input_dim} "
        f"macro_action_raw_ctbr_steps={args.action_horizon} "
        f"decoder={actor.decoder_type} "
        f"frozen_dynamics_parameters={sum(p.numel() for p in frozen_dynamics.parameters()) if frozen_dynamics is not None else 0:,} "
        f"train_episodes={len(train_paths)} validation_episodes={len(validation_paths)} "
        f"eligible_train_windows={len(train_indices)} eligible_validation_windows={len(validation_indices)} "
        f"phase_counts={phase_counts}",
        flush=True,
    )

    iterator = iter(loader)
    target_start = args.history - 1
    validation_interval = int(settings.get("validation_interval", 500))
    early_stopping_patience = int(settings.get("early_stopping_patience", 0))
    early_stopping_start_step = int(settings.get("early_stopping_start_step", 0))
    early_stopping_min_delta = float(settings.get("early_stopping_min_delta", 0.0))
    early_stopping_best = math.inf
    early_stopping_stale = 0
    last_log_time = time.perf_counter()
    for step in range(start, args.steps):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)
        batch = move_batch(raw_batch, args.device)
        state = build_actor_state(encoder, batch, args.history, args.device, amp)
        target = batch["commanded_action"][:, target_start:target_start + args.action_horizon]
        phase = batch["tail_phase"][:, target_start:target_start + args.action_horizon].amax(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=args.device.startswith("cuda") and amp,
        ):
            output = actor(state, action_tokenizer)
            target_action_tokens = None
            if actor.uses_action_tokenizer:
                with torch.no_grad():
                    target_action_tokens, _ = action_tokenizer.encode(
                        target,
                        batch["action_timing"][
                            :, target_start:target_start + args.action_horizon
                        ],
                    )
            loss, metrics = actor_bc_objective(
                output, target, settings, step, target_action_tokens
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), float(settings.get("grad_clip", 5.0))
        )
        optimizer.step()
        scheduler.step()

        if step % args.log_interval == 0:
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-6)
            logged = {
                **metrics,
                "gradient_norm": gradient_norm.detach(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "recovery_fraction": (phase == 3).float().mean(),
                "tail_fraction": ((phase == 1) | (phase == 2) | (phase == 4)).float().mean(),
                "steps_per_second": (args.log_interval if step > start else 1) / elapsed,
            }
            print(
                f"step={step + 1} " + " ".join(
                    f"{key}={float(value):.5f}" for key, value in logged.items()
                ), flush=True,
            )
            logger.log_train(logged, step + 1)
            last_log_time = now

        validation_metrics = None
        should_stop = False
        if validation_loader and (
            (step + 1) % validation_interval == 0 or step + 1 == args.steps
        ):
            validation_metrics = validate(
                actor, encoder, action_tokenizer, validation_loader,
                history=args.history,
                action_horizon=args.action_horizon,
                device=args.device,
                amp=amp,
                max_batches=int(settings.get("validation_batches", 100)),
                recovery_selection_weight=float(settings.get("recovery_selection_weight", 0.5)),
                critical_selection_weight=float(settings.get("critical_selection_weight", 0.25)),
                selection_calibration_weight=float(settings.get("selection_calibration_weight", 0.02)),
            )
            print(
                f"step={step + 1} validation " + " ".join(
                    f"{key}={value:.5f}" for key, value in validation_metrics.items()
                ), flush=True,
            )
            logger.log_eval(validation_metrics, step + 1)
            manager.save_eval(
                validation_metrics,
                step=step + 1,
                summary="Held-out expert-only Gaussian-NLL CTBR behavior cloning, including recovery targets after invalid trajectory history.",
            )
            score = validation_metrics.get(manager.monitor)
            if (
                early_stopping_patience > 0
                and step + 1 >= early_stopping_start_step
                and score is not None
            ):
                if score < early_stopping_best - early_stopping_min_delta:
                    early_stopping_best = score
                    early_stopping_stale = 0
                else:
                    early_stopping_stale += 1
                should_stop = early_stopping_stale >= early_stopping_patience

        if (
            (step + 1) % args.checkpoint_interval == 0
            or step + 1 == args.steps
            or validation_metrics is not None
        ):
            checkpoint_payload = {
                "model": actor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": capture_rng_state(),
                "model_config": actor_config,
                "training_config": config,
                "observation_tokenizer": str(observation_path),
                "action_tokenizer": str(action_path),
                "input_contract": {
                    "actor_state": "rolling_posterior_latents_with_deltas_decoded_proprio_exact_applied_actions_timing_and_tokenizer_state_estimate",
                    "actor_decoder": actor.decoder_type,
                    "action": "normalized_commanded_ctbr",
                    "action_decoder": (
                        "frozen_action_tokenizer"
                        if actor.uses_action_tokenizer else "direct_mlp"
                    ),
                    "action_chunk_steps": args.action_horizon,
                    "target_requires_controller_valid": True,
                    "history_may_include_invalid_actions": True,
                },
            }
            if dynamics_checkpoint is not None and args.dynamics is not None:
                checkpoint_payload.update({
                    "world_model_checkpoint": str(args.dynamics),
                    "world_model_step": int(dynamics_checkpoint.get("step", 0)),
                })
                checkpoint_payload["input_contract"]["pmpo_state"] = (
                    "rolling_predicted_latents_with_matching_derived_features"
                )
            manager.save(
                checkpoint_payload,
                step=step + 1,
                metrics=validation_metrics,
                rank=validation_metrics is not None,
            )
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
