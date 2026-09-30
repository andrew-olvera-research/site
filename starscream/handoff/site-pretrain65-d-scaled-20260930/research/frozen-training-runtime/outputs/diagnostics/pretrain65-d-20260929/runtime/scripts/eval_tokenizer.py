#!/usr/bin/env python3
"""Evaluate observation JEPA reconstruction, conditioning, and latent geometry probes."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from starscream.dataloader import (
    DreamerSequenceDataset, TAIL_PHASE_NAMES, stratified_episode_split,
)
from starscream.checkpoint_manager import CheckpointManager
from starscream.loss import binary_segmentation_metrics, observation_jepa_loss
from starscream.tokenizer import MultiModalTokenizer, ObservationJEPA, macro_last


def ridge_probe(
    features: torch.Tensor, targets: torch.Tensor, ridge: float | None = None, seed: int = 0
) -> dict[str, float]:
    permutation = torch.randperm(features.shape[0], generator=torch.Generator().manual_seed(seed))
    features, targets = features[permutation], targets[permutation]
    split = max(1, int(0.7 * features.shape[0]))
    train_x, test_x = features[:split].float(), features[split:].float()
    train_y, test_y = targets[:split].float(), targets[split:].float()
    if not len(test_x):
        return {"mae": float("nan"), "r2": float("nan")}

    def fit(source_x, source_y, alpha):
        x_mean, x_std = source_x.mean(0), source_x.std(0).clamp_min(1e-5)
        y_mean, y_std = source_y.mean(0), source_y.std(0).clamp_min(1e-5)
        design = torch.cat(
            [(source_x - x_mean) / x_std, torch.ones(len(source_x), 1)], dim=-1
        )
        penalty = torch.eye(design.shape[-1])
        penalty[-1, -1] = 0.0
        weights = torch.linalg.solve(
            design.T @ design + float(alpha) * penalty, design.T @ ((source_y - y_mean) / y_std)
        )
        return weights, x_mean, x_std, y_mean, y_std

    def predict(value, fitted):
        weights, x_mean, x_std, y_mean, y_std = fitted
        design = torch.cat(
            [(value - x_mean) / x_std, torch.ones(len(value), 1)], dim=-1
        )
        return design @ weights * y_std + y_mean

    if ridge is None:
        if len(train_x) < 2:
            ridge = 1.0
        else:
            inner = min(len(train_x) - 1, max(1, int(0.85 * len(train_x))))
            fit_x, validation_x = train_x[:inner], train_x[inner:]
            fit_y, validation_y = train_y[:inner], train_y[inner:]
            candidates = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
            ridge = min(
                candidates,
                key=lambda alpha: float(
                    ((predict(validation_x, fit(fit_x, fit_y, alpha)) - validation_y)
                     / fit_y.std(0).clamp_min(1e-5)).square().mean()
                ),
            )
    prediction = predict(test_x, fit(train_x, train_y, ridge))
    residual = (prediction - test_y).square().sum()
    total = (test_y - test_y.mean(0)).square().sum().clamp_min(1e-8)
    return {
        "mae": float((prediction - test_y).abs().mean()),
        "r2": float(1.0 - residual / total),
        "ridge": float(ridge),
    }


def mlp_probe(
    features: torch.Tensor, targets: torch.Tensor, seed: int = 0, steps: int = 300
) -> dict[str, float]:
    """Measure recoverable nonlinear information without updating the tokenizer."""

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=generator)
    features, targets = features[permutation].float(), targets[permutation].float()
    split = max(1, int(0.7 * features.shape[0]))
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = targets[:split], targets[split:]
    if not len(test_x):
        return {"mae": float("nan"), "r2": float("nan")}
    x_mean, x_std = train_x.mean(0), train_x.std(0).clamp_min(1e-5)
    y_mean, y_std = train_y.mean(0), train_y.std(0).clamp_min(1e-5)
    train_x = (train_x - x_mean) / x_std
    test_x = (test_x - x_mean) / x_std
    normalized_y = (train_y - y_mean) / y_std
    torch.manual_seed(seed)
    hidden = min(256, max(64, features.shape[-1] // 2))
    probe = torch.nn.Sequential(
        torch.nn.Linear(features.shape[-1], hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, targets.shape[-1]),
    )
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(int(steps)):
        indices = torch.randint(
            len(train_x), (min(512, len(train_x)),), generator=generator
        )
        prediction = probe(train_x[indices])
        loss = torch.nn.functional.smooth_l1_loss(prediction, normalized_y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.inference_mode():
        prediction = probe(test_x) * y_std + y_mean
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


def episode_metadata(path: Path, name: str, fallback: str = "unknown") -> str:
    """Read a scalar episode metadata field without loading episode arrays."""

    with h5py.File(path, "r", swmr=True) as episode:
        key = f"metadata/{name}"
        if key not in episode:
            return fallback
        value = np.asarray(episode[key]).item()
    return value.decode() if isinstance(value, bytes) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("validation", "all"), default="validation")
    parser.add_argument("--track", help="evaluate only episodes whose filename starts with this track name")
    parser.add_argument("--regime", help="evaluate only episodes with this distribution_regime metadata")
    parser.add_argument(
        "--phase", choices=TAIL_PHASE_NAMES,
        help="evaluate only windows whose highest-priority temporal tail phase matches",
    )
    parser.add_argument("--stride", type=int, help="evaluation window stride; defaults to non-overlapping sequences")
    parser.add_argument("--split-seed", type=int, help="override the checkpoint seed used for the episode split")
    parser.add_argument("--nonlinear-probe", action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    full_config = checkpoint["training_config"]
    settings = full_config.get("tokenizer", full_config)
    model_settings = settings["model"]
    encoder_config = dict(model_settings.get("encoder", model_settings))
    predictor_config = dict(model_settings.get("predictor", {}))
    objective = dict(settings.get("objective", {}))
    # Warmup keys belong to the trainer's schedule, not the loss signature.
    objective.pop("auxiliary_warmup_steps", None)
    model = ObservationJEPA(MultiModalTokenizer(**encoder_config), **predictor_config).to(args.device)
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
    if args.regime:
        paths = [
            path for path in paths
            if episode_metadata(path, "distribution_regime") == args.regime
        ]
        if not paths:
            raise ValueError(
                f"no {args.split} episodes found for distribution regime {args.regime!r}"
            )
    sequence_length = int(settings["sequence_length"])
    dataset = DreamerSequenceDataset(
        data, sequence_length=sequence_length,
        stride=int(args.stride or sequence_length),
        mode="tokenizer", mask_size=tuple(encoder_config.get("image_size", (128, 160))), paths=paths,
        max_open_files=int(settings.get("max_open_files", 8)), validate_contents=False,
        include_privileged=bool(settings.get("include_privileged", False)),
        cache_in_memory=True,
        include_actions=(
            bool(settings.get("include_actions", False))
            or bool(predictor_config.get("action_conditioned", False))
        ),
        route_source=str(settings.get("route_source", "flight_plan")),
        route_target_source=str(settings.get("route_target_source", "input")),
        deployment_estimate_source=str(
            settings.get("deployment_estimate_source", "none")
        ),
        deployment_estimate_seed=int(
            settings.get("deployment_estimate_seed", settings.get("seed", 0))
        ),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    metrics_total: dict[str, float] = {}
    evaluated_samples = 0
    features: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    routes: list[torch.Tensor] = []
    proprios: list[torch.Tensor] = []
    camera_deltas: list[torch.Tensor] = []
    proprio_deltas: list[torch.Tensor] = []
    route_deltas: list[torch.Tensor] = []
    estimate_deltas: list[torch.Tensor] = []
    order_deltas: list[torch.Tensor] = []
    state_absolute_errors: list[torch.Tensor] = []
    route_position_errors: list[torch.Tensor] = []
    route_normal_cosines: list[torch.Tensor] = []
    route_up_cosines: list[torch.Tensor] = []
    mask_visibility: list[torch.Tensor] = []
    estimate_input_state_errors: list[torch.Tensor] = []
    estimate_input_gate_position_errors: list[torch.Tensor] = []
    estimate_ablation_state_errors: list[torch.Tensor] = []
    estimate_ablation_gate_position_errors: list[torch.Tensor] = []
    patch = int(encoder_config.get("temporal_patch_size", 3))
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if args.phase:
                phase_index = TAIL_PHASE_NAMES.index(args.phase)
                selected = batch["tail_phase"].amax(dim=1) == phase_index
                if not selected.any():
                    continue
                batch = {key: value[selected] for key, value in batch.items()}
            batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                output = model(batch)
                _, loss_metrics = observation_jepa_loss(output, batch, **objective)
                mask_target = macro_last(batch["mask"], patch)
                mask_metrics = binary_segmentation_metrics(
                    output.reconstruction.mask_logits.float(), mask_target.float()
                )
                latent = output.reconstruction.latents
                no_camera = dict(batch)
                no_camera["mask"] = torch.zeros_like(batch["mask"])
                camera_latent = model.encoder.encode(no_camera)
                no_proprio = dict(batch)
                no_proprio["proprio"] = model.encoder.proprio_center.view(1, 1, -1).expand_as(batch["proprio"])
                proprio_latent = model.encoder.encode(no_proprio)
                no_route = dict(batch)
                no_route["route"] = model.encoder.route_center.view(1, 1, 1, -1).expand_as(batch["route"])
                route_latent = model.encoder.encode(no_route)
                if "estimate" in batch:
                    no_estimate = dict(batch)
                    no_estimate["estimate"] = model.encoder.estimate_center.view(
                        1, 1, -1
                    ).expand_as(batch["estimate"])
                    estimate_latent = model.encoder.encode(no_estimate)
                    estimate_ablation = model.encoder.decode(estimate_latent)
                else:
                    estimate_latent = latent
                    estimate_ablation = None
                reversed_proprio = dict(batch)
                grouped = batch["proprio"].reshape(batch["proprio"].shape[0], -1, patch, batch["proprio"].shape[-1])
                reversed_proprio["proprio"] = grouped.flip(2).reshape_as(batch["proprio"])
                order_latent = model.encoder.encode(reversed_proprio)
            batch_samples = int(batch["mask"].shape[0])
            evaluated_samples += batch_samples
            for key, value in loss_metrics.items():
                metrics_total[key] = metrics_total.get(key, 0.0) + batch_samples * float(value)
            for key, value in mask_metrics.items():
                report_key = f"mask_{key}"
                # observation_jepa_loss already reports the foreground metrics.
                # Do not add them twice under the same name.
                if report_key not in loss_metrics:
                    metrics_total[report_key] = (
                        metrics_total.get(report_key, 0.0) + batch_samples * float(value)
                    )
            features.append(latent.float().flatten(2).cpu())
            states.append(macro_last(batch["task_state"], patch).float().cpu())
            route_target = macro_last(
                batch.get("route_target", batch["route"]), patch
            ).float()
            routes.append(route_target[..., :3].flatten(2).cpu())
            proprios.append(macro_last(batch["proprio"], patch).float().cpu())
            state_absolute_errors.append(
                (output.reconstruction.state_mean.float()
                 - macro_last(batch["task_state"], patch).float()).abs().cpu()
            )
            route_prediction = output.reconstruction.route.float()
            route_position_errors.append(
                ((route_prediction[..., :3] - route_target[..., :3]).abs() * 20.0).cpu()
            )
            route_normal_cosines.append(
                torch.nn.functional.cosine_similarity(
                    route_prediction[..., 3:6], route_target[..., 3:6], dim=-1
                ).clamp(-1.0, 1.0).cpu()
            )
            route_up_cosines.append(
                torch.nn.functional.cosine_similarity(
                    route_prediction[..., 6:9], route_target[..., 6:9], dim=-1
                ).clamp(-1.0, 1.0).cpu()
            )
            mask_visibility.append(
                (macro_last(batch["mask"], patch).float().sum(dim=(-3, -2, -1)) > 0)
                .cpu()
            )
            if "estimate" in batch:
                state_target = macro_last(batch["task_state"], patch).float()
                route_target_macro = macro_last(
                    batch.get("route_target", batch["route"]), patch
                ).float()
                estimate_input = macro_last(batch["estimate"], patch).float()
                estimate_input_state_errors.append(
                    (estimate_input[..., :12] - state_target[..., :12]).abs().cpu()
                )
                estimate_input_gate_position_errors.append(
                    ((estimate_input[..., 12:15] - route_target_macro[..., 0, :3])
                     .abs() * 20.0).cpu()
                )
                estimate_ablation_state_errors.append(
                    (estimate_ablation.state_mean.float() - state_target).abs().cpu()
                )
                estimate_ablation_gate_position_errors.append(
                    ((estimate_ablation.route[..., 0, :3]
                      - route_target_macro[..., 0, :3]).abs() * 20.0).cpu()
                )
            camera_deltas.append((latent.float() - camera_latent.float()).square().mean(dim=(-1, -2)).sqrt().cpu())
            proprio_deltas.append((latent.float() - proprio_latent.float()).square().mean(dim=(-1, -2)).sqrt().cpu())
            route_deltas.append((latent.float() - route_latent.float()).square().mean(dim=(-1, -2)).sqrt().cpu())
            estimate_deltas.append(
                (latent.float() - estimate_latent.float())
                .square().mean(dim=(-1, -2)).sqrt().cpu()
            )
            order_deltas.append((latent.float() - order_latent.float()).square().mean(dim=(-1, -2)).sqrt().cpu())
            if args.batches and index + 1 >= args.batches:
                break
    if not features:
        raise ValueError("no evaluation windows matched the requested filters")
    flat_features = torch.cat(features).flatten(0, 1)
    report = {key: value / evaluated_samples for key, value in metrics_total.items()}
    report["camera_conditioning_rms"] = float(torch.cat(camera_deltas).mean())
    report["proprio_conditioning_rms"] = float(torch.cat(proprio_deltas).mean())
    report["route_conditioning_rms"] = float(torch.cat(route_deltas).mean())
    report["estimate_conditioning_rms"] = float(torch.cat(estimate_deltas).mean())
    report["within_patch_order_rms"] = float(torch.cat(order_deltas).mean())
    probe_seed = int(args.split_seed if args.split_seed is not None else settings.get("seed", 0))
    report["task_state_probe"] = ridge_probe(flat_features, torch.cat(states).flatten(0, 1), seed=probe_seed)
    report["route_position_probe"] = ridge_probe(flat_features, torch.cat(routes).flatten(0, 1), seed=probe_seed)
    report["proprio_probe"] = ridge_probe(flat_features, torch.cat(proprios).flatten(0, 1), seed=probe_seed)
    report["latent_geometry"] = latent_geometry(flat_features)
    state_error_sequences = torch.cat(state_absolute_errors)
    state_error = state_error_sequences.flatten(0, 1)
    physical_state_scale = state_error.new_tensor(
        [20.0] * 3 + [30.0] * 3 + [1.0] * 6 + [6.0] * 3 + [4000.0] * 4
    )
    physical_state_error = state_error * physical_state_scale
    report["state_mae_physical"] = {
        "position_m": float(physical_state_error[:, 0:3].mean()),
        "velocity_mps": float(physical_state_error[:, 3:6].mean()),
        "attitude_6d": float(physical_state_error[:, 6:12].mean()),
        "body_rates_radps": float(physical_state_error[:, 12:15].mean()),
        "motor_omega_radps": float(physical_state_error[:, 15:19].mean()),
    }
    route_position_error_sequences = torch.cat(route_position_errors)
    route_normal_cosine_sequences = torch.cat(route_normal_cosines)
    route_up_cosine_sequences = torch.cat(route_up_cosines)
    position_error = route_position_error_sequences.flatten(0, 1)
    normal_cosine = route_normal_cosine_sequences.flatten()
    up_cosine = route_up_cosine_sequences.flatten()
    report["route_target_kind"] = str(settings.get("route_target_source", "input"))
    report["route_pose_error"] = {
        "position_mae_m": float(position_error.mean()),
        "active_gate_position_mae_m": float(position_error[:, 0].mean()),
        "normal_angle_deg": float(torch.rad2deg(torch.acos(normal_cosine)).mean()),
        "up_angle_deg": float(torch.rad2deg(torch.acos(up_cosine)).mean()),
    }
    last_state_error = state_error_sequences[:, -1] * physical_state_scale
    report["last_step_state_mae_physical"] = {
        "position_m": float(last_state_error[:, 0:3].mean()),
        "velocity_mps": float(last_state_error[:, 3:6].mean()),
        "attitude_6d": float(last_state_error[:, 6:12].mean()),
        "body_rates_radps": float(last_state_error[:, 12:15].mean()),
        "motor_omega_radps": float(last_state_error[:, 15:19].mean()),
    }
    last_position_error = route_position_error_sequences[:, -1]
    last_normal_cosine = route_normal_cosine_sequences[:, -1].flatten()
    last_up_cosine = route_up_cosine_sequences[:, -1].flatten()
    report["last_step_route_pose_error"] = {
        "position_mae_m": float(last_position_error.mean()),
        "active_gate_position_mae_m": float(last_position_error[:, 0].mean()),
        "normal_angle_deg": float(
            torch.rad2deg(torch.acos(last_normal_cosine)).mean()
        ),
        "up_angle_deg": float(torch.rad2deg(torch.acos(last_up_cosine)).mean()),
    }
    visible = torch.cat(mask_visibility).flatten()
    state_position_error = physical_state_error[:, 0:3].mean(dim=-1)
    state_velocity_error = physical_state_error[:, 3:6].mean(dim=-1)
    active_gate_position_error = position_error[:, 0].mean(dim=-1)
    visibility_report = {"empty_fraction": float((~visible).float().mean())}
    for name, selected in (("visible", visible), ("empty", ~visible)):
        if selected.any():
            visibility_report[name] = {
                "samples": int(selected.sum()),
                "active_gate_position_mae_m": float(
                    active_gate_position_error[selected].mean()
                ),
                "state_position_mae_m": float(state_position_error[selected].mean()),
                "state_velocity_mae_mps": float(state_velocity_error[selected].mean()),
            }
    report["error_by_mask_visibility"] = visibility_report
    if estimate_input_state_errors:
        proxy_state_error = torch.cat(estimate_input_state_errors).flatten(0, 1)
        proxy_scale = proxy_state_error.new_tensor(
            [20.0] * 3 + [30.0] * 3 + [1.0] * 6
        )
        proxy_state_error = proxy_state_error * proxy_scale
        proxy_gate_error = torch.cat(
            estimate_input_gate_position_errors
        ).flatten(0, 1)
        report["deployment_estimate_input_error"] = {
            "position_m": float(proxy_state_error[:, 0:3].mean()),
            "velocity_mps": float(proxy_state_error[:, 3:6].mean()),
            "attitude_6d": float(proxy_state_error[:, 6:12].mean()),
            "active_gate_position_mae_m": float(proxy_gate_error.mean()),
        }
        ablated_state_error = torch.cat(
            estimate_ablation_state_errors
        ).flatten(0, 1) * physical_state_scale
        ablated_gate_error = torch.cat(
            estimate_ablation_gate_position_errors
        ).flatten(0, 1)
        report["without_deployment_estimate_error"] = {
            "position_m": float(ablated_state_error[:, 0:3].mean()),
            "velocity_mps": float(ablated_state_error[:, 3:6].mean()),
            "attitude_6d": float(ablated_state_error[:, 6:12].mean()),
            "active_gate_position_mae_m": float(ablated_gate_error.mean()),
        }
    if args.nonlinear_probe:
        report["task_state_mlp_probe"] = mlp_probe(
            flat_features, torch.cat(states).flatten(0, 1), seed=probe_seed
        )
    if model.encoder.latent_group_sizes:
        state_targets = torch.cat(states).flatten(0, 1)
        route_targets = torch.cat(routes).flatten(0, 1)
        proprio_targets = torch.cat(proprios).flatten(0, 1)
        offset = 0
        group_report = {}
        for name, tokens in zip(
            model.encoder.latent_group_names, model.encoder.latent_group_sizes
        ):
            width = int(tokens) * model.encoder.d_bottleneck
            group_features = flat_features[:, offset : offset + width]
            group_report[name] = {
                "latent_geometry": latent_geometry(group_features),
                "task_state_probe": ridge_probe(
                    group_features, state_targets, seed=probe_seed
                ),
                "route_position_probe": ridge_probe(
                    group_features, route_targets, seed=probe_seed
                ),
                "proprio_probe": ridge_probe(
                    group_features, proprio_targets, seed=probe_seed
                ),
            }
            offset += width
        report["latent_groups"] = group_report
    report["evaluated_episodes"] = [path.name for path in paths]
    report["evaluated_windows"] = evaluated_samples
    report["tail_phase"] = args.phase
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        run_name = full_config.get("wandb", {}).get("run_name", args.checkpoint.parent.name or args.checkpoint.stem)
        manager = CheckpointManager(run_name, output_root=Path(__file__).resolve().parents[1] / "outputs")
        probe_keys = ("camera_conditioning_rms", "proprio_conditioning_rms", "route_conditioning_rms", "within_patch_order_rms", "task_state_probe", "route_position_probe", "proprio_probe", "latent_geometry", "evaluated_episodes")
        manager.save_eval(
            {key: value for key, value in report.items() if key not in probe_keys},
            probes={key: report[key] for key in probe_keys},
            step=int(checkpoint.get("step", 0)), kind="probes",
            summary="Observation-tokenizer audit of reconstruction/JEPA quality, modality conditioning, temporal-order sensitivity, and linear latent geometry.",
        )


if __name__ == "__main__":
    main()
