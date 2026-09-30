#!/usr/bin/env python3
"""Pretrain the multi-rate observation tokenizer with an EMA-target JEPA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from starscream.attention import set_attention_backend
from starscream.checkpoint_manager import CheckpointManager, capture_rng_state, restore_rng_state
from starscream.dataloader import (
    DreamerSequenceDataset,
    LocalityAwareBatchSampler,
    stratified_episode_split,
)
from starscream.loss import observation_jepa_loss
from starscream.tokenizer import MultiModalTokenizer, ObservationJEPA
from starscream.wandb import init_wandb


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict:
    """Load a YAML config with an optional relative ``extends`` parent."""

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(load_config(parent_path.resolve()), config)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    full_config = load_config(args.config)
    config = full_config.get("tokenizer", {})
    for name in ("steps", "batch_size", "device"):
        override = getattr(args, name)
        if override is not None:
            config[name] = override
    if args.smoke_test:
        config.update(steps=2, validation_batches=1, checkpoint_interval=2, log_interval=1)
        full_config["output_root"] = "/tmp/starscream-tokenizer-smoke"
        full_config["checkpoint"] = {
            **full_config.get("checkpoint", {}), "run_name": "observation-tokenizer-config-smoke"
        }
        full_config["wandb"] = {"enabled": False, "run_name": "observation-tokenizer-config-smoke"}
    args.config_data = full_config
    args.settings = config
    return args


def move_batch(batch: dict, device: str) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def initialization_path(settings: dict) -> Path | None:
    """Resolve either an explicit checkpoint or the best ranked manifest entry."""

    direct = settings.get("initialize_from")
    manifest_value = settings.get("initialize_from_manifest")
    if direct and manifest_value:
        raise ValueError("set only one of initialize_from and initialize_from_manifest")
    if not manifest_value:
        return Path(direct) if direct else None
    manifest = Path(manifest_value)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoints = payload.get("checkpoints", [])
    if not checkpoints:
        raise RuntimeError(f"initialization manifest has no ranked checkpoints: {manifest}")
    selected = Path(checkpoints[0]["path"])
    return selected if selected.is_absolute() else manifest.parent / selected


def objective_at_step(settings: dict, step: int) -> dict:
    """Warm predictive pressure in only after content reconstruction has stabilized."""

    objective = dict(settings)
    warmup = int(objective.pop("prediction_warmup_steps", 0))
    if warmup > 0:
        final_weight = float(objective.get("prediction_weight", 1.0))
        objective["prediction_weight"] = final_weight * min(1.0, max(0.0, step / warmup))
    auxiliary_warmup = int(objective.pop("auxiliary_warmup_steps", 0))
    if auxiliary_warmup > 0:
        fraction = min(1.0, max(0.0, step / auxiliary_warmup))
        for name in (
            "state_delta_weight", "future_state_weight",
            "inverse_action_weight", "progress_weight",
            "action_state_delta_weight",
        ):
            if name in objective:
                objective[name] = float(objective[name]) * fraction
    return objective


@torch.inference_mode()
def validate(model, loader, device: str, max_batches: int, objective: dict, amp: bool) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            _, metrics = observation_jepa_loss(model(batch), batch, **objective)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batches += 1
        if max_batches and batches >= max_batches:
            break
    model.train()
    return {key: value / max(1, batches) for key, value in totals.items()}


def main() -> None:
    parsed = arguments()
    settings = parsed.settings
    full_config = parsed.config_data
    seed = int(settings.get("seed", 0))
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")
    device = str(settings.get("device", "cuda"))
    amp = bool(settings.get("amp", True)) and device.startswith("cuda")
    if str(settings.get("precision", "bf16")).lower() != "bf16":
        raise SystemExit("tokenizer training currently requires precision: bf16")
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", True))
        torch.backends.cuda.matmul.allow_tf32 = bool(settings.get("allow_tf32", True))
    set_attention_backend(settings.get("attention", "auto"))
    data = Path(settings["data"])
    manager = CheckpointManager.from_config(full_config)
    logger = init_wandb(full_config)
    sequence_length = int(settings.get("sequence_length", 24))
    encoder_config = dict(settings.get("model", {}).get("encoder", settings.get("model", {})))
    predictor_config = dict(settings.get("model", {}).get("predictor", {}))
    patch_size = int(encoder_config.get("temporal_patch_size", 3))
    if sequence_length % patch_size:
        raise SystemExit("sequence_length must be divisible by temporal_patch_size")
    train_paths, validation_paths = stratified_episode_split(
        data, float(settings.get("validation_fraction", 0.1)), seed,
        tracks=settings.get("tracks", settings.get("track")),
    )
    dataset_options = dict(
        root=data,
        sequence_length=sequence_length,
        stride=int(settings.get("stride", patch_size)),
        mode="tokenizer",
        mask_size=tuple(encoder_config.get("image_size", (128, 160))),
        max_open_files=int(settings.get("max_open_files", 8)),
        validate_contents=bool(settings.get("validate_contents", False)),
        include_privileged=bool(settings.get("include_privileged", False)),
        cache_in_memory=bool(settings.get("cache_in_memory", True)),
        include_actions=bool(settings.get("include_actions", False)),
        route_source=str(settings.get("route_source", "flight_plan")),
        route_target_source=str(settings.get("route_target_source", "input")),
        deployment_estimate_source=str(
            settings.get("deployment_estimate_source", "none")
        ),
        deployment_estimate_seed=int(
            settings.get("deployment_estimate_seed", seed)
        ),
    )
    train = DreamerSequenceDataset(paths=train_paths, **dataset_options)
    validation = DreamerSequenceDataset(paths=validation_paths, **dataset_options) if validation_paths else None
    batch_size = int(settings.get("batch_size", 8))
    loader_options = dict(
        num_workers=int(settings.get("workers", 4)),
        pin_memory=device.startswith("cuda"),
        persistent_workers=int(settings.get("workers", 4)) > 0,
    )
    if int(settings.get("workers", 4)) > 0:
        loader_options["prefetch_factor"] = int(settings.get("prefetch_factor", 2))
    batch_sampler = LocalityAwareBatchSampler(
        train,
        batch_size,
        episodes_per_batch=int(settings.get("episodes_per_batch", 4)),
        cache_size=int(settings.get("max_open_files", 8)),
        windows_per_locality_block=int(settings.get("windows_per_locality_block", 64)),
        seed=seed,
        phase_fractions=settings.get("phase_fractions"),
    )
    loader = DataLoader(train, batch_sampler=batch_sampler, **loader_options)
    validation_loader = (
        DataLoader(validation, batch_size=batch_size, shuffle=False, **loader_options)
        if validation else None
    )
    model = ObservationJEPA(MultiModalTokenizer(**encoder_config), **predictor_config).to(device)
    initialize_from = initialization_path(settings)
    if initialize_from:
        initialized = torch.load(initialize_from, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(initialized["model"], strict=False)
        allowed_missing = {
            "encoder.fusion_state_head.weight", "encoder.fusion_state_head.bias",
            "target_encoder.fusion_state_head.weight", "target_encoder.fusion_state_head.bias",
        }
        allowed_missing_prefixes = tuple(settings.get("initialize_allow_missing_prefixes", ()))
        allowed_unexpected_prefixes = tuple(settings.get("initialize_allow_unexpected_prefixes", ()))
        unexpected_missing = {
            key for key in incompatible.missing_keys
            if key not in allowed_missing
            and not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        }
        unexpected_keys = {
            key for key in incompatible.unexpected_keys
            if not any(key.startswith(prefix) for prefix in allowed_unexpected_prefixes)
        }
        if unexpected_keys or unexpected_missing:
            raise RuntimeError(
                "initialization checkpoint is incompatible: "
                f"missing={sorted(unexpected_missing)} "
                f"unexpected={sorted(unexpected_keys)}"
            )
        print(f"initialized_from={initialize_from}", flush=True)
    if settings.get("freeze_visual_path", False):
        visual_modules = [
            model.encoder.visual,
            model.encoder.modality_bottlenecks["visual"],
            model.encoder.mask_decoder_trunk,
            model.encoder.mask_head,
        ]
        for module in visual_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        model.encoder.modality_embedding.requires_grad_(False)
        print("froze_visual_path=true", flush=True)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"train_episodes={len(train_paths)} train_windows={len(train)} "
        f"validation_episodes={len(validation_paths)} trainable_parameters={parameters:,}",
        flush=True,
    )
    if settings.get("compile", False):
        model.compile(mode=str(settings.get("compile_mode", "default")), dynamic=False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(settings.get("learning_rate", 3e-4)),
        weight_decay=float(settings.get("weight_decay", 0.05)),
        fused=bool(settings.get("fused_optimizer", True)) and device.startswith("cuda"),
    )
    start = 0
    checkpoint = manager.resume(bool(settings.get("resume", False)), map_location=device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = int(checkpoint.get("step", 0))
        restore_rng_state(checkpoint.get("rng_state"))
    iterator = iter(loader)
    steps = int(settings.get("steps", 30000))
    validation_interval = int(settings.get("validation_interval", 500))
    checkpoint_interval = int(settings.get("checkpoint_interval", 1000))
    ema_rate = float(settings.get("ema_rate", 0.005))
    objective_settings = dict(settings.get("objective", {}))
    for step in range(start, steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        objective = objective_at_step(objective_settings, step + 1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss, metrics = observation_jepa_loss(model(batch), batch, **objective)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("grad_clip", 5.0)))
        optimizer.step()
        model.update_target(ema_rate)
        if step % int(settings.get("log_interval", 25)) == 0:
            print(f"step={step} " + " ".join(f"{key}={float(value):.5f}" for key, value in metrics.items()), flush=True)
            logger.log_train(metrics, step + 1)
        validation_metrics = None
        if validation_loader and ((step + 1) % validation_interval == 0 or step + 1 == steps):
            validation_metrics = validate(
                model, validation_loader, device, int(settings.get("validation_batches", 25)),
                objective, amp,
            )
            print(
                f"step={step + 1} validation "
                + " ".join(f"{key}={value:.5f}" for key, value in validation_metrics.items()),
                flush=True,
            )
            logger.log_eval(validation_metrics, step + 1)
            manager.save_eval(
                validation_metrics, step=step + 1,
                summary="Held-out observation reconstruction, JEPA prediction, and latent-variance evaluation.",
            )
        if (step + 1) % checkpoint_interval == 0 or step + 1 == steps or validation_metrics is not None:
            checkpoint = {
                "model": model.state_dict(),
                "encoder": model.encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": capture_rng_state(),
                "model_config": settings.get("model", {}),
                "training_config": full_config,
                "input_contract": {
                    "route_source": str(settings.get("route_source", "flight_plan")),
                    "route_target_source": str(
                        settings.get("route_target_source", "input")
                    ),
                    "deployment_estimate_source": str(
                        settings.get("deployment_estimate_source", "none")
                    ),
                    "gate_frame_safe_dynamics": bool(
                        objective_settings.get("gate_frame_safe_dynamics", False)
                    ),
                    "gate_consistent_prediction": bool(
                        objective_settings.get("gate_consistent_prediction", False)
                    ),
                },
            }
            manager.save(checkpoint, step=step + 1, metrics=validation_metrics, rank=validation_metrics is not None)
    logger.finish()


if __name__ == "__main__":
    main()
