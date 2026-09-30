#!/usr/bin/env python3
"""Evaluate action JEPA likelihood, ordering sensitivity, and latent geometry."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from starscream.dataloader import DreamerSequenceDataset, stratified_episode_split
from starscream.checkpoint_manager import CheckpointManager
from starscream.loss import action_tokenizer_loss
from starscream.tokenizer import ActionJEPA, ActionTokenizer


def ridge_probe(
    features: torch.Tensor, targets: torch.Tensor, ridge: float = 1e-3, seed: int = 0
) -> dict[str, float]:
    permutation = torch.randperm(features.shape[0], generator=torch.Generator().manual_seed(seed))
    features, targets = features[permutation], targets[permutation]
    split = max(1, int(0.7 * features.shape[0]))
    train_x, test_x = features[:split].float(), features[split:].float()
    train_y, test_y = targets[:split].float(), targets[split:].float()
    if not len(test_x):
        return {"mae": float("nan"), "r2": float("nan")}
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-5)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    train_x = torch.cat([train_x, torch.ones(len(train_x), 1)], -1)
    test_x = torch.cat([test_x, torch.ones(len(test_x), 1)], -1)
    weights = torch.linalg.solve(
        train_x.T @ train_x + ridge * torch.eye(train_x.shape[-1]), train_x.T @ train_y
    )
    prediction = test_x @ weights
    residual = (prediction - test_y).square().sum()
    total = (test_y - test_y.mean(0)).square().sum().clamp_min(1e-8)
    return {"mae": float((prediction - test_y).abs().mean()), "r2": float(1.0 - residual / total)}


def latent_geometry(features: torch.Tensor) -> dict[str, float]:
    features = features.float()
    centered = features - features.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    effective_rank = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12)
    return {
        "effective_rank": float(effective_rank),
        "effective_rank_fraction": float(effective_rank / features.shape[-1]),
        "feature_std_mean": float(features.std(0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("validation", "all"), default="validation")
    parser.add_argument("--track", help="evaluate only episodes whose filename starts with this track name")
    parser.add_argument("--stride", type=int, help="evaluation window stride; defaults to non-overlapping sequences")
    parser.add_argument("--split-seed", type=int, help="override the checkpoint seed used for the episode split")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    full_config = checkpoint["training_config"]
    settings = full_config.get("action_tokenizer", full_config)
    model_settings = settings["model"]
    tokenizer_config = dict(model_settings.get("tokenizer", model_settings))
    predictor_config = dict(model_settings.get("predictor", {}))
    objective = dict(settings.get("objective", {}))
    model = ActionJEPA(ActionTokenizer(**tokenizer_config), **predictor_config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data = args.data or Path(settings["data"])
    all_paths = sorted((*data.glob("*.h5"), *data.glob("*.hdf5")))
    if args.split == "validation" and len(all_paths) > 1:
        _, paths = stratified_episode_split(
            data, float(settings.get("validation_fraction", 0.1)),
            int(args.split_seed if args.split_seed is not None else settings.get("seed", 0)),
        )
    else:
        paths = all_paths
    if args.track:
        paths = [path for path in paths if path.name.startswith(f"{args.track}_")]
        if not paths:
            raise ValueError(f"no {args.split} episodes found for track {args.track!r}")
    sequence_length = int(settings["sequence_length"])
    dataset = DreamerSequenceDataset(
        data, sequence_length=sequence_length,
        stride=int(args.stride or sequence_length),
        mode="action_tokenizer", paths=paths,
        max_open_files=int(settings.get("max_open_files", 8)), validate_contents=False,
        cache_in_memory=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    features: list[torch.Tensor] = []
    geometry: list[torch.Tensor] = []
    order_delta: list[torch.Tensor] = []
    action_delta: list[torch.Tensor] = []
    timing_delta: list[torch.Tensor] = []
    totals: dict[str, float] = {}
    absolute_errors: list[torch.Tensor] = []
    one_sigma: list[torch.Tensor] = []
    state_effect_errors: list[torch.Tensor] = []
    evaluated_samples = 0
    uncertainty_enabled = bool(tokenizer_config.get("predict_uncertainty", True))
    patch = int(tokenizer_config.get("temporal_patch_size", 3))
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                output = model(batch)
                _, loss_metrics = action_tokenizer_loss(output, batch, **objective)
                tokens = output.reconstruction.tokens
                grouped = batch["applied_action"].reshape(batch["applied_action"].shape[0], -1, patch, 4)
                reverse_batch = dict(batch)
                reverse_batch["applied_action"] = grouped.flip(2).reshape_as(batch["applied_action"])
                reverse_tokens, _ = model.tokenizer.encode(reverse_batch["applied_action"], reverse_batch["action_timing"])
                mean_action = model.tokenizer.action_center.view(1, 1, -1).expand_as(batch["applied_action"])
                mean_timing = model.tokenizer.timing_center.view(1, 1, -1).expand_as(batch["action_timing"])
                zero_action_tokens, _ = model.tokenizer.encode(mean_action, batch["action_timing"])
                zero_timing_tokens, _ = model.tokenizer.encode(batch["applied_action"], mean_timing)
            batch_samples = int(batch["applied_action"].shape[0])
            evaluated_samples += batch_samples
            for key, value in loss_metrics.items():
                totals[key] = totals.get(key, 0.0) + batch_samples * float(value)
            mean = grouped.mean(2)
            endpoint_delta = grouped[:, :, -1] - grouped[:, :, 0]
            total_variation = (
                (grouped[:, :, 1:] - grouped[:, :, :-1])
                .abs().mean((2, 3), keepdim=False).unsqueeze(-1)
                if patch > 1
                else grouped.new_zeros(*grouped.shape[:2], 1)
            )
            features.append(tokens.float().flatten(2).cpu())
            geometry.append(torch.cat([mean, endpoint_delta, total_variation], dim=-1).float().cpu())
            residual = output.reconstruction.mean.float() - grouped.float()
            absolute_errors.append(residual.abs().mean((1, 2)).cpu())
            if uncertainty_enabled:
                one_sigma.append(
                    (residual.abs() <= output.reconstruction.log_std.float().exp())
                    .float().mean((1, 2)).cpu()
                )
            if output.reconstruction.state_delta_mean is not None:
                state_effect_errors.append(
                    (output.reconstruction.state_delta_mean.float()
                     - (batch["task_state"][:, 1:] - batch["task_state"][:, :-1]).float())
                    .abs().cpu()
                )
            order_delta.append((tokens.float() - reverse_tokens.float()).square().mean((-1, -2)).sqrt().cpu())
            action_delta.append((tokens.float() - zero_action_tokens.float()).square().mean((-1, -2)).sqrt().cpu())
            timing_delta.append((tokens.float() - zero_timing_tokens.float()).square().mean((-1, -2)).sqrt().cpu())
            if args.batches and index + 1 >= args.batches:
                break
    flat_features = torch.cat(features).flatten(0, 1)
    report = {key: value / evaluated_samples for key, value in totals.items()}
    report["within_patch_order_rms"] = float(torch.cat(order_delta).mean())
    report["action_conditioning_rms"] = float(torch.cat(action_delta).mean())
    report["timing_conditioning_rms"] = float(torch.cat(timing_delta).mean())
    report["action_geometry_probe"] = ridge_probe(
        flat_features, torch.cat(geometry).flatten(0, 1),
        seed=int(args.split_seed if args.split_seed is not None else settings.get("seed", 0)),
    )
    report["action_mae_per_channel"] = torch.cat(absolute_errors).mean(0).tolist()
    normalized_action_mae = torch.cat(absolute_errors).mean(0)
    report["action_mae_physical"] = {
        "collective_thrust": float(normalized_action_mae[0] * 15.0),
        "body_rate_radps": float(normalized_action_mae[1:].mean() * 6.0),
    }
    if state_effect_errors:
        state_effect_error = torch.cat(state_effect_errors).flatten(0, 1)
        physical_scale = state_effect_error.new_tensor(
            [20.0] * 3 + [30.0] * 3 + [1.0] * 6
            + [6.0] * 3 + [4000.0] * 4
        )
        state_effect_error = state_effect_error * physical_scale
        report["state_effect_mae_physical"] = {
            "position_m": float(state_effect_error[:, 0:3].mean()),
            "velocity_mps": float(state_effect_error[:, 3:6].mean()),
            "attitude_6d": float(state_effect_error[:, 6:12].mean()),
            "body_rates_radps": float(state_effect_error[:, 12:15].mean()),
            "motor_omega_radps": float(state_effect_error[:, 15:19].mean()),
        }
    report["uncertainty_enabled"] = uncertainty_enabled
    report["one_sigma_coverage_per_channel"] = (
        torch.cat(one_sigma).mean(0).tolist() if one_sigma else None
    )
    report["latent_geometry"] = latent_geometry(flat_features)
    report["evaluated_episodes"] = [path.name for path in paths]
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        run_name = full_config.get("wandb", {}).get("run_name", args.checkpoint.parent.name or args.checkpoint.stem)
        manager = CheckpointManager(run_name, output_root=Path(__file__).resolve().parents[1] / "outputs")
        probe_keys = ("within_patch_order_rms", "action_conditioning_rms", "timing_conditioning_rms", "action_geometry_probe", "latent_geometry", "evaluated_episodes")
        manager.save_eval(
            {key: value for key, value in report.items() if key not in probe_keys},
            probes={key: report[key] for key in probe_keys},
            step=int(checkpoint.get("step", 0)), kind="probes",
            summary="Action-tokenizer audit of JEPA/GNLL quality, ordered-patch sensitivity, and linear latent action geometry.",
        )


if __name__ == "__main__":
    main()
