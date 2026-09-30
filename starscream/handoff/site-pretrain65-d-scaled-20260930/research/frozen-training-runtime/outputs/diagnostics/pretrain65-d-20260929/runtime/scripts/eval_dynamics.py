#!/usr/bin/env python3
"""Deterministic optimization and control probes for latent race dynamics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dreamerv4 import TwoHot
from starscream.attention import set_attention_backend
from starscream.checkpoint_manager import CheckpointManager
from starscream.dataloader import DreamerSequenceDataset, stratified_episode_split
from starscream.dynamics import RaceDynamics
from starscream.tokenizer import ActionTokenizer, MultiModalTokenizer, macro_last


class Mean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, value: torch.Tensor) -> None:
        value = value.detach().float()
        self.total += float(value.sum())
        self.count += value.numel()

    def value(self) -> float:
        return self.total / max(1, self.count)


def move(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def encode_targets(batch, observation, action, model, patch: int, amp: bool):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        latents = observation.encode(
            {key: batch[key] for key in ("mask", "proprio", "route", "timing")}
        )
        packed = model.pack(latents)
        initial, clean = packed[:, :1], packed[:, 1:]
        aligned_action = batch["normalized_action"][:, patch - 1 :]
        aligned_timing = batch["action_timing"][:, patch - 1 :]
        action_output = action(aligned_action, aligned_timing) if action is not None else None
        if action_output is None:
            macro_action = aligned_action.reshape(
                aligned_action.shape[0], -1, patch, aligned_action.shape[-1]
            ).mean(2)
            action_context = None
        else:
            macro_action = action_output.mean.mean(2)
            action_context = action_output.context
    macro_count = clean.shape[1]
    reward = batch["reward"][:, patch - 1 :].reshape(
        batch["reward"].shape[0], macro_count, patch
    ).sum(-1)
    transition_continue = batch["continue"] * (~batch["is_last"]).to(batch["continue"].dtype)
    continuation = transition_continue[:, patch - 1 :].reshape(
        transition_continue.shape[0], macro_count, patch
    ).prod(-1)
    targets = {
        "reward": reward,
        "continue": continuation,
        "task_state": macro_last(batch["task_state"], patch)[:, 1:],
    }
    if "privileged_state_estimate" in batch:
        targets["estimator_state"] = macro_last(
            batch["privileged_state_estimate"], patch
        )[:, 1:, :19]
    return initial, clean, macro_action, action_context, targets


def binary_report(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    scores, labels = scores.float().flatten(), labels.bool().flatten()
    positive, negative = scores[labels], scores[~labels]
    auc = (
        ((positive[:, None] > negative[None]).float()
         + 0.5 * (positive[:, None] == negative[None]).float()).mean()
        if len(positive) and len(negative) else scores.new_tensor(float("nan"))
    )
    order = scores.argsort(descending=True)
    ordered = labels[order].float()
    precision_curve = ordered.cumsum(0) / torch.arange(
        1, len(ordered) + 1, device=scores.device
    )
    average_precision = (
        precision_curve[ordered.bool()].mean()
        if ordered.any() else scores.new_tensor(float("nan"))
    )
    candidates = torch.linspace(0, 1, 201, device=scores.device)
    best = (0.0, 0.5, 0.0, 0.0)
    for threshold in candidates:
        prediction = scores >= threshold
        tp = (prediction & labels).sum().float()
        precision = tp / prediction.sum().clamp_min(1)
        recall = tp / labels.sum().clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
        if float(f1) > best[0]:
            best = (float(f1), float(threshold), float(precision), float(recall))
    prediction = scores >= 0.5
    tp = (prediction & labels).sum().float()
    precision = tp / prediction.sum().clamp_min(1)
    recall = tp / labels.sum().clamp_min(1)
    return {
        "count": len(scores), "positives": int(labels.sum()),
        "auc_roc": float(auc), "average_precision": float(average_precision),
        "brier": float((scores - labels.float()).square().mean()),
        "threshold_0_5_precision": float(precision),
        "threshold_0_5_recall": float(recall),
        "best_f1": best[0], "best_f1_threshold": best[1],
        "best_f1_precision": best[2], "best_f1_recall": best[3],
        "positive_score_mean": float(positive.mean()) if len(positive) else float("nan"),
        "negative_score_mean": float(negative.mean()) if len(negative) else float("nan"),
    }


def reward_tail_report(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    prediction, target = prediction.float().flatten(), target.float().flatten()
    magnitude = target.abs()
    edges = torch.quantile(magnitude, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]))
    bins = {}
    for index in range(len(edges) - 1):
        mask = (magnitude >= edges[index]) & (
            magnitude <= edges[index + 1] if index == len(edges) - 2 else magnitude < edges[index + 1]
        )
        bins[f"q{[0, 50, 90, 99][index]}_q{[50, 90, 99, 100][index]}"] = {
            "count": int(mask.sum()),
            "target_abs_mean": float(magnitude[mask].mean()),
            "mae": float((prediction[mask] - target[mask]).abs().mean()),
            "bias": float((prediction[mask] - target[mask]).mean()),
        }
    return {
        "mae": float((prediction - target).abs().mean()),
        "rmse": float((prediction - target).square().mean().sqrt()),
        "bias": float((prediction - target).mean()),
        "magnitude_bins": bins,
    }


def r2_per_dimension(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    residual = (prediction - target).square().sum((0, 1))
    centered = (target - target.mean((0, 1), keepdim=True)).square().sum((0, 1)).clamp_min(1e-8)
    return (1.0 - residual / centered).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split", choices=("validation", "all"), default="validation")
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--rollout-batches", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--seed", type=int, default=20262030)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("auto", "flash", "mem_efficient", "math"), default="flash")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    full_config = checkpoint["training_config"]
    settings = full_config.get("dynamics", full_config)
    set_attention_backend(args.attention)
    amp = args.device.startswith("cuda") and bool(settings.get("amp", True))

    observation_path = Path(checkpoint.get("observation_tokenizer", settings["tokenizer"]))
    observation_checkpoint = torch.load(observation_path, map_location=args.device, weights_only=False)
    observation_model_config = observation_checkpoint.get("model_config", {})
    observation_config = dict(observation_model_config.get("encoder", observation_model_config))
    observation = MultiModalTokenizer(**observation_config).to(args.device)
    observation.load_state_dict(observation_checkpoint.get("encoder", observation_checkpoint["model"]))
    observation.eval().requires_grad_(False)

    action_path = checkpoint.get("action_tokenizer", settings.get("action_tokenizer"))
    action = None
    if action_path:
        action_checkpoint = torch.load(action_path, map_location=args.device, weights_only=False)
        action_model_config = action_checkpoint.get("model_config", {})
        action_config = dict(action_model_config.get("tokenizer", action_model_config))
        action = ActionTokenizer(**action_config).to(args.device)
        action.load_state_dict(action_checkpoint.get("tokenizer", action_checkpoint["model"]))
        action.eval().requires_grad_(False)

    model = RaceDynamics(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    patch = observation.temporal_patch_size
    data = args.data or Path(settings["data"])
    if args.split == "validation":
        _, paths = stratified_episode_split(
            data, float(settings.get("validation_fraction", 0.1)), int(settings.get("seed", 0))
        )
    else:
        paths = sorted((*data.glob("*.h5"), *data.glob("*.hdf5")))
    sequence_length = int(settings.get("sequence_length", 26))
    dataset = DreamerSequenceDataset(
        data, sequence_length=sequence_length,
        stride=int(args.stride or sequence_length), mode="dynamics", paths=paths,
        mask_size=tuple(observation_config.get("image_size", (128, 160))),
        max_open_files=int(settings.get("max_open_files", 8)), validate_contents=False,
        include_privileged=True, cache_in_memory=False,
    )
    loader_options = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    if args.workers:
        loader_options.update(persistent_workers=True, multiprocessing_context="spawn")
    loader = DataLoader(dataset, **loader_options)
    terminal_indices = [index for index, value in enumerate(dataset.terminal_windows) if value]
    terminal_loader = DataLoader(Subset(dataset, terminal_indices), **loader_options)

    grid = defaultdict(Mean)
    rollout = defaultdict(Mean)
    reward_predictions, reward_targets = [], []
    estimator_predictions, estimator_targets = [], []
    terminal_scores, terminal_labels = [], []
    action_ablation, shuffled_action_ablation, context_ablation = Mean(), Mean(), Mean()
    baseline_ablation = Mean()
    reward_coder = TwoHot(model.reward_head.out_features).to(args.device)

    def reward_mean(output):
        return (
            output.reward_value.float()
            if output.reward_value is not None
            else reward_coder.mean(output.reward_logits.float())
        )

    def encoded(batch):
        return encode_targets(batch, observation, action, model, patch, amp)

    with torch.inference_mode(), torch.random.fork_rng(
        devices=[torch.cuda.current_device()] if args.device.startswith("cuda") else []
    ):
        torch.manual_seed(args.seed)
        for batch_index, raw_batch in enumerate(loader):
            batch = move(raw_batch, args.device)
            initial, clean, actions, action_context, targets = encoded(batch)
            source = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype)
            for step_index in range(int(torch.tensor(model.k_max).log2()) + 1):
                step_size = (2 ** step_index) / model.k_max
                for cell in range(round(1.0 / step_size)):
                    tau = cell * step_size
                    noisy = source + tau * (clean - source)
                    steps = torch.full(clean.shape[:2], step_index, device=clean.device, dtype=torch.long)
                    signals = torch.full(clean.shape[:2], round(tau * model.k_max), device=clean.device, dtype=torch.long)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        output = model(noisy, actions, steps, signals, initial, action_context)
                    key = f"step_{step_index}/tau_{tau:.3f}"
                    grid[f"{key}/latent_rmse"].add(
                        (output.predicted_packed_latents.float() - clean.float()).square().flatten(2).mean(-1).sqrt()
                    )
                    grid[f"{key}/input_rmse"].add(
                        (noisy.float() - clean.float()).square().flatten(2).mean(-1).sqrt()
                    )
                    grid[f"{key}/reward_mae"].add(
                        (reward_mean(output) - targets["reward"]).abs()
                    )
            # Match the deployable transition used by training and recursive
            # imagination: every target receives its immediately preceding
            # clean latent as a one-step context. Residual models must not be
            # scored through the auxiliary multi-step shortcut branch.
            batch_size, horizon = clean.shape[:2]
            previous = torch.cat([initial, clean[:, :-1]], dim=1)
            flat_previous = previous.flatten(0, 1).unsqueeze(1)
            flat_clean = clean.flatten(0, 1).unsqueeze(1)
            flat_source = source.flatten(0, 1).unsqueeze(1)
            flat_actions = actions.flatten(0, 1).unsqueeze(1)
            flat_action_context = (
                action_context.flatten(0, 1).unsqueeze(1)
                if action_context is not None else None
            )
            steps = torch.full(
                flat_clean.shape[:2], int(torch.tensor(model.k_max).log2()),
                device=clean.device, dtype=torch.long,
            )
            signals = torch.zeros_like(steps)
            causal_transition = bool(getattr(model, "causal_residual", False))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                output = model(
                    flat_source, flat_actions, steps, signals, flat_previous,
                    flat_action_context, causal_transition=causal_transition,
                )
                zero_output = model(
                    flat_source, torch.zeros_like(flat_actions), steps, signals,
                    flat_previous,
                    torch.zeros_like(flat_action_context) if flat_action_context is not None else None,
                    causal_transition=causal_transition,
                )
                permutation = torch.arange(
                    flat_actions.shape[0] - 1, -1, -1, device=actions.device
                )
                shuffled_output = model(
                    flat_source, flat_actions[permutation], steps, signals,
                    flat_previous,
                    flat_action_context[permutation] if flat_action_context is not None else None,
                    causal_transition=causal_transition,
                )
                no_context = model(
                    flat_source, flat_actions, steps, signals,
                    torch.zeros_like(flat_previous), flat_action_context,
                    causal_transition=causal_transition,
                )
            baseline_error = (
                output.predicted_packed_latents.float() - flat_clean.float()
            ).square().flatten(2).mean(-1)
            baseline_ablation.add(baseline_error)
            action_ablation.add(
                (zero_output.predicted_packed_latents.float() - flat_clean.float()).square().flatten(2).mean(-1)
            )
            shuffled_action_ablation.add(
                (shuffled_output.predicted_packed_latents.float() - flat_clean.float()).square().flatten(2).mean(-1)
            )
            context_ablation.add(
                (no_context.predicted_packed_latents.float() - flat_clean.float()).square().flatten(2).mean(-1)
            )
            reward_predictions.append(reward_mean(output).reshape(batch_size, horizon).cpu())
            reward_targets.append(targets["reward"].cpu())
            if output.estimator_state is not None and "estimator_state" in targets:
                center = output.estimator_state.new_tensor(settings["estimator_state_center"])
                scale = output.estimator_state.new_tensor(settings["estimator_state_scale"])
                estimator_predictions.append(
                    (output.estimator_state.float() * scale + center)
                    .reshape(batch_size, horizon, -1).cpu()
                )
                estimator_targets.append(targets["estimator_state"].float().cpu())
            if batch_index < args.rollout_batches:
                context = initial
                for horizon in range(clean.shape[1]):
                    rollout_steps = torch.full(
                        source[:, horizon : horizon + 1].shape[:2],
                        int(torch.tensor(model.k_max).log2()),
                        device=clean.device, dtype=torch.long,
                    )
                    rollout_signals = torch.zeros_like(rollout_steps)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        one = model(
                            source[:, horizon : horizon + 1], actions[:, horizon : horizon + 1],
                            rollout_steps, rollout_signals, context,
                            action_context[:, horizon : horizon + 1] if action_context is not None else None,
                            causal_transition=bool(getattr(model, "causal_residual", False)),
                        )
                    rollout[f"horizon_{horizon + 1}/latent_rmse"].add(
                        (one.predicted_packed_latents.float() - clean[:, horizon : horizon + 1].float())
                        .square().flatten(2).mean(-1).sqrt()
                    )
                    rollout[f"horizon_{horizon + 1}/persistence_rmse"].add(
                        (initial.float() - clean[:, horizon : horizon + 1].float())
                        .square().flatten(2).mean(-1).sqrt()
                    )
                    rollout[f"horizon_{horizon + 1}/reward_mae"].add(
                        (reward_mean(one) - targets["reward"][:, horizon : horizon + 1]).abs()
                    )
                    context = one.predicted_packed_latents
            if args.batches and batch_index + 1 >= args.batches:
                break

        # Every held-out ending is evaluated on an identical one-jump path.
        for raw_batch in terminal_loader:
            batch = move(raw_batch, args.device)
            initial, clean, actions, action_context, targets = encoded(batch)
            source = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype)
            previous = torch.cat([initial, clean[:, :-1]], dim=1)
            flat_previous = previous.flatten(0, 1).unsqueeze(1)
            flat_source = source.flatten(0, 1).unsqueeze(1)
            flat_actions = actions.flatten(0, 1).unsqueeze(1)
            flat_action_context = (
                action_context.flatten(0, 1).unsqueeze(1)
                if action_context is not None else None
            )
            steps = torch.full(
                flat_source.shape[:2], int(torch.tensor(model.k_max).log2()),
                device=clean.device, dtype=torch.long,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                output = model(
                    flat_source, flat_actions, steps, torch.zeros_like(steps),
                    flat_previous, flat_action_context,
                    causal_transition=bool(getattr(model, "causal_residual", False)),
                )
            terminal_scores.append(
                (1.0 - output.continue_logits.sigmoid()).reshape_as(targets["continue"]).float().cpu()
            )
            terminal_labels.append((targets["continue"] < 0.5).cpu())

    reward_prediction, reward_target = torch.cat(reward_predictions), torch.cat(reward_targets)
    report = {
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "evaluated_windows": min(len(dataset), args.batches * args.batch_size if args.batches else len(dataset)),
        "shortcut_grid": {key: value.value() for key, value in sorted(grid.items())},
        "one_jump_ablation": {
            "baseline_mse": baseline_ablation.value(),
            "zero_action_mse": action_ablation.value(),
            "zero_action_error_ratio": action_ablation.value() / max(1e-12, baseline_ablation.value()),
            "shuffled_action_mse": shuffled_action_ablation.value(),
            "shuffled_action_error_ratio": shuffled_action_ablation.value() / max(1e-12, baseline_ablation.value()),
            "zero_context_mse": context_ablation.value(),
            "zero_context_error_ratio": context_ablation.value() / max(1e-12, baseline_ablation.value()),
        },
        "open_loop_rollout": {key: value.value() for key, value in sorted(rollout.items())},
        "reward": reward_tail_report(reward_prediction, reward_target),
        "continuation": binary_report(torch.cat(terminal_scores), torch.cat(terminal_labels)),
        "evaluated_episodes": [path.name for path in paths],
    }
    if estimator_predictions:
        prediction, target = torch.cat(estimator_predictions), torch.cat(estimator_targets)
        report["estimator_state"] = {
            "mae": float((prediction - target).abs().mean()),
            "mae_per_dimension": (prediction - target).abs().mean((0, 1)).tolist(),
            "r2_per_dimension": r2_per_dimension(prediction, target),
        }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        run_name = full_config.get("wandb", {}).get("run_name", args.checkpoint.parent.name)
        manager = CheckpointManager(
            run_name, output_root=Path(full_config.get("output_root", "outputs"))
        )
        manager.save_eval(
            {"checkpoint_step": report["checkpoint_step"], "reward": report["reward"],
             "continuation": report["continuation"], "estimator_state": report.get("estimator_state")},
            probes={key: report[key] for key in ("shortcut_grid", "one_jump_ablation", "open_loop_rollout")},
            step=report["checkpoint_step"], kind="probes",
            summary="Deterministic shortcut grid, causal action/context ablations, reward tails, continuation discrimination, state decoding, and open-loop drift.",
        )


if __name__ == "__main__":
    main()
