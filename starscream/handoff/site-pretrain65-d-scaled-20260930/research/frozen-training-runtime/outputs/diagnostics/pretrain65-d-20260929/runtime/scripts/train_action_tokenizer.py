#!/usr/bin/env python3
"""Pretrain the ordered applied-action tokenizer with JEPA and Gaussian NLL."""

from __future__ import annotations

import argparse
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
from starscream.loss import action_tokenizer_loss
from starscream.tokenizer import ActionJEPA, ActionTokenizer
from starscream.wandb import init_wandb


def move_batch(batch: dict, device: str) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        full_config = yaml.safe_load(stream) or {}
        settings = full_config.get("action_tokenizer", {})
    if args.steps is not None:
        settings["steps"] = args.steps
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.smoke_test:
        settings.update(steps=2, validation_batches=1, checkpoint_interval=2, log_interval=1)
        full_config["output_root"] = "/tmp/starscream-tokenizer-smoke"
        full_config["checkpoint"] = {
            **full_config.get("checkpoint", {}), "run_name": "action-tokenizer-config-smoke"
        }
        full_config["wandb"] = {"enabled": False, "run_name": "action-tokenizer-config-smoke"}
    return settings, full_config


@torch.inference_mode()
def validate(model, loader, device, max_batches, objective, amp):
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            _, metrics = action_tokenizer_loss(model(batch), batch, **objective)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
        if max_batches and count >= max_batches:
            break
    model.train()
    return {key: value / max(1, count) for key, value in totals.items()}


def main() -> None:
    settings, full_config = arguments()
    seed = int(settings.get("seed", 0))
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")
    device = str(settings.get("device", "cuda"))
    amp = bool(settings.get("amp", True)) and device.startswith("cuda")
    if str(settings.get("precision", "bf16")).lower() != "bf16":
        raise SystemExit("action-tokenizer training currently requires precision: bf16")
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", True))
        torch.backends.cuda.matmul.allow_tf32 = bool(settings.get("allow_tf32", True))
    set_attention_backend(settings.get("attention", "auto"))
    model_settings = settings.get("model", {})
    tokenizer_config = dict(model_settings.get("tokenizer", model_settings))
    predictor_config = dict(model_settings.get("predictor", {}))
    patch = int(tokenizer_config.get("temporal_patch_size", 3))
    sequence_length = int(settings.get("sequence_length", 48))
    if sequence_length % patch:
        raise SystemExit("sequence_length must be divisible by temporal_patch_size")
    data = Path(settings["data"])
    train_paths, validation_paths = stratified_episode_split(
        data, float(settings.get("validation_fraction", 0.1)), seed,
        tracks=settings.get("tracks", settings.get("track")),
    )
    options = dict(
        root=data, sequence_length=sequence_length, stride=int(settings.get("stride", patch)),
        mode="action_tokenizer", max_open_files=int(settings.get("max_open_files", 8)),
        validate_contents=bool(settings.get("validate_contents", False)),
        cache_in_memory=bool(settings.get("cache_in_memory", True)),
    )
    train = DreamerSequenceDataset(paths=train_paths, **options)
    validation = DreamerSequenceDataset(paths=validation_paths, **options) if validation_paths else None
    workers = int(settings.get("workers", 4))
    batch_size = int(settings.get("batch_size", 64))
    loader_options = dict(
        num_workers=workers,
        pin_memory=device.startswith("cuda"), persistent_workers=workers > 0,
    )
    if workers > 0:
        loader_options["prefetch_factor"] = int(settings.get("prefetch_factor", 2))
    batch_sampler = LocalityAwareBatchSampler(
        train,
        batch_size,
        episodes_per_batch=int(settings.get("episodes_per_batch", 4)),
        cache_size=int(settings.get("max_open_files", 8)),
        windows_per_locality_block=int(settings.get("windows_per_locality_block", 64)),
        seed=seed,
    )
    loader = DataLoader(train, batch_sampler=batch_sampler, **loader_options)
    validation_loader = (
        DataLoader(validation, batch_size=batch_size, shuffle=False, **loader_options)
        if validation else None
    )
    model = ActionJEPA(ActionTokenizer(**tokenizer_config), **predictor_config).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"train_episodes={len(train_paths)} train_windows={len(train)} "
        f"validation_episodes={len(validation_paths)} trainable_parameters={parameters:,}", flush=True
    )
    if settings.get("compile", False):
        model.compile(mode=str(settings.get("compile_mode", "default")), dynamic=False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(settings.get("learning_rate", 3e-4)),
        weight_decay=float(settings.get("weight_decay", 0.05)),
        fused=bool(settings.get("fused_optimizer", True)) and device.startswith("cuda"),
    )
    manager = CheckpointManager.from_config(full_config)
    logger = init_wandb(full_config)
    iterator = iter(loader)
    steps = int(settings.get("steps", 30000))
    start = 0
    checkpoint = manager.resume(bool(settings.get("resume", False)), map_location=device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = int(checkpoint.get("step", 0))
        restore_rng_state(checkpoint.get("rng_state"))
    objective = dict(settings.get("objective", {}))
    for step in range(start, steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss, metrics = action_tokenizer_loss(model(batch), batch, **objective)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("grad_clip", 5.0)))
        optimizer.step()
        model.update_target(float(settings.get("ema_rate", 0.005)))
        if step % int(settings.get("log_interval", 25)) == 0:
            print(f"step={step} " + " ".join(f"{key}={float(value):.5f}" for key, value in metrics.items()), flush=True)
            logger.log_train(metrics, step + 1)
        validation_metrics = None
        validation_interval = int(settings.get("validation_interval", 500))
        if validation_loader and ((step + 1) % validation_interval == 0 or step + 1 == steps):
            validation_metrics = validate(
                model, validation_loader, device,
                int(settings.get("validation_batches", 25)), objective, amp,
            )
            print(
                f"step={step + 1} validation "
                + " ".join(f"{key}={value:.5f}" for key, value in validation_metrics.items()), flush=True
            )
            logger.log_eval(validation_metrics, step + 1)
            manager.save_eval(
                validation_metrics, step=step + 1,
                summary="Held-out ordered-action JEPA, Gaussian NLL, temporal-delta, and uncertainty evaluation.",
            )
        if (step + 1) % int(settings.get("checkpoint_interval", 1000)) == 0 or step + 1 == steps or validation_metrics is not None:
            checkpoint = {
                "model": model.state_dict(), "tokenizer": model.tokenizer.state_dict(),
                "optimizer": optimizer.state_dict(), "rng_state": capture_rng_state(),
                "model_config": model_settings, "training_config": full_config,
                "input_contract": {
                    "gate_frame_safe_state_effect": bool(
                        objective.get("gate_frame_safe_state_effect", False)
                    )
                },
            }
            manager.save(checkpoint, step=step + 1, metrics=validation_metrics, rank=validation_metrics is not None)
    logger.finish()


if __name__ == "__main__":
    main()
