#!/usr/bin/env python3
"""Train SkyDreamer-style GateNet on simulator and labeled real images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from starscream.checkpoint_manager import CheckpointManager, capture_rng_state, restore_rng_state
from starscream.gatenet import CameraCalibration, CameraRectifier, build_gatenet
from starscream.gatenet_data import (
    FolderGateDataset,
    HDF5GateDataset,
    IMAGE_SUFFIXES,
    build_gatenet_loaders,
)
from starscream.loss import binary_segmentation_metrics, gatenet_loss
from starscream.wandb import init_wandb


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sim-data", type=Path, help="directory of rendered Starscream HDF5 episodes")
    parser.add_argument(
        "--real-data", type=Path, action="append",
        help="repeatable directory containing train/images, train/masks, val/images, val/masks",
    )
    parser.add_argument("--gate-type", choices=("orange", "mavlab"), default="orange")
    parser.add_argument("--architecture", choices=("efficientnet_b0", "paper"), default="efficientnet_b0")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--real-fraction", type=float, default=0.5)
    parser.add_argument("--camera-calibration", type=Path, help="YAML intrinsics/distortion for real images")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--initialize", type=Path, help="load model weights only for a new fine-tuning run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        with preliminary.config.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        values = config.get("gatenet", config)
        parser.set_defaults(**values)
    else:
        config = {
            "output_root": "outputs", "gatenet": {},
            "wandb": {"enabled": False, "run_name": "gatenet-cli"},
            "checkpoint": {"monitor": "iou", "mode": "max", "top_k": 3},
        }
    args = parser.parse_args()
    args.sim_data = Path(args.sim_data) if args.sim_data else None
    args.real_data = [Path(path) for path in (args.real_data or [])]
    args.initialize = Path(args.initialize) if args.initialize else None
    args.camera_calibration = Path(args.camera_calibration) if args.camera_calibration else None
    args.full_config = config
    return args


def split(items: list, seed: int) -> tuple[list, list]:
    items = list(items)
    random.Random(seed).shuffle(items)
    if len(items) <= 1:
        return items, items
    validation = max(1, round(0.1 * len(items)))
    return items[validation:], items[:validation]


def make_datasets(args, size):
    train_sets, validation_sets = [], []
    rectifier = (
        CameraRectifier(CameraCalibration.from_yaml(args.camera_calibration), (size[1], size[0]))
        if args.camera_calibration
        else None
    )
    if args.sim_data:
        paths = sorted((*args.sim_data.glob("*.h5"), *args.sim_data.glob("*.hdf5")))
        train_paths, validation_paths = split(paths, args.seed)
        train_sets.append(HDF5GateDataset(args.sim_data, size=size, augment=True, paths=train_paths))
        validation_sets.append(HDF5GateDataset(args.sim_data, size=size, paths=validation_paths))
    for real_root in args.real_data or []:
        if (real_root / "train" / "images").exists():
            train_sets.append(
                FolderGateDataset(real_root / "train", size=size, augment=True, rectifier=rectifier)
            )
            validation_sets.append(
                FolderGateDataset(real_root / "val", size=size, rectifier=rectifier)
            )
        else:
            image_dir = real_root / "images"
            files = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
            train_files, validation_files = split(files, args.seed + 1)
            print("warning: randomly splitting real frames; train/ and val/ recording-session splits are safer")
            train_sets.append(FolderGateDataset(real_root, size=size, augment=True, files=train_files, rectifier=rectifier))
            validation_sets.append(FolderGateDataset(real_root, size=size, files=validation_files, rectifier=rectifier))
    train_sets = [dataset for dataset in train_sets if len(dataset)]
    validation_sets = [dataset for dataset in validation_sets if len(dataset)]
    if not train_sets:
        raise SystemExit("provide non-empty --sim-data and/or --real-data")
    return train_sets, validation_sets


@torch.no_grad()
def validate(model, loader, device, max_batches=None):
    model.eval()
    totals = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "empty_false_positive": 0.0}
    batches = 0
    for batch in loader:
        image, mask = batch["image"].to(device), batch["mask"].to(device)
        metrics = binary_segmentation_metrics(model(image), mask)
        for key in totals:
            totals[key] += float(metrics[key])
        batches += 1
        if max_batches and batches >= max_batches:
            break
    return {key: value / max(batches, 1) for key, value in totals.items()}


def main() -> None:
    args = arguments()
    manager = CheckpointManager.from_config(args.full_config)
    logger = init_wandb(args.full_config)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    size = (384, 384) if args.gate_type == "orange" else (196, 196)
    base_channels = 16 if args.gate_type == "orange" else 32
    train_sets, validation_sets = make_datasets(args, size)
    loaders = build_gatenet_loaders(
        train_sets,
        validation_sets,
        batch_size=args.batch_size,
        workers=args.workers,
        real_fraction=args.real_fraction,
        seed=args.seed,
    )
    train_loader, validation_loader = loaders.train, loaders.validation
    print(f"training_samples={loaders.train_samples} validation_samples={loaders.validation_samples}")
    model = build_gatenet(
        args.architecture, pretrained=not args.no_pretrained, base_channels=base_channels
    ).to(args.device)
    if hasattr(model, "encoder"):
        backbone_parameters = list(model.encoder.parameters())
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        decoder_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
        optimizer_groups = [
            {"params": backbone_parameters, "lr": args.learning_rate * 0.1},
            {"params": decoder_parameters, "lr": args.learning_rate},
        ]
    else:
        optimizer_groups = model.parameters()
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch, best_iou = 0, -1.0
    if args.initialize and args.resume:
        raise SystemExit("use either --initialize or --resume, not both")
    if args.initialize:
        checkpoint = torch.load(args.initialize, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
    checkpoint = manager.resume(bool(args.resume), map_location=args.device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint.get("model", checkpoint.get("state_dict")))
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        restore_rng_state(checkpoint.get("rng_state"))
    use_amp = args.device.startswith("cuda")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if hasattr(model, "encoder"):
            train_backbone = epoch >= args.freeze_backbone_epochs
            model.encoder.requires_grad_(train_backbone)
            if not train_backbone:
                model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        trained_batches = 0
        for index, batch in enumerate(train_loader):
            image = batch["image"].to(args.device, non_blocking=True)
            mask = batch["mask"].to(args.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, _ = gatenet_loss(model.forward_multiscale(image), mask)
                scaled_loss = loss / args.accumulate
            scaled_loss.backward()
            at_debug_limit = bool(args.max_train_batches and index + 1 >= args.max_train_batches)
            if (index + 1) % args.accumulate == 0 or index + 1 == len(train_loader) or at_debug_limit:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach())
            trained_batches += 1
            if args.max_train_batches and index + 1 >= args.max_train_batches:
                break
        scheduler.step()
        metrics = validate(
            model, validation_loader, args.device, args.max_validation_batches
        )
        print(
            f"epoch={epoch:03d} loss={running / max(trained_batches, 1):.4f} "
            + " ".join(f"val_{key}={value:.4f}" for key, value in metrics.items())
        )
        train_metrics = {"loss": running / max(trained_batches, 1), "learning_rate": optimizer.param_groups[-1]["lr"]}
        logger.log_train(train_metrics, epoch + 1)
        logger.log_eval(metrics, epoch + 1)
        manager.save_eval(
            metrics, step=epoch + 1,
            summary="Held-out GateNet segmentation quality including IoU, precision, recall, and empty-frame false positives.",
        )
        manager.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": capture_rng_state(),
                "epoch": epoch,
                "gate_type": args.gate_type,
                "architecture": args.architecture,
                "base_channels": base_channels,
                "input_size": size,
                "training_config": args.full_config,
            },
            step=epoch + 1, metrics=metrics,
        )
    logger.finish()


if __name__ == "__main__":
    main()
