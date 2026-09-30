"""Paired RGB/gate-mask datasets and sim-to-real augmentations for GateNet."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from .gatenet import CameraRectifier


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class GateNetAugment:
    """Joint geometry plus camera degradations used for sim-to-real training."""

    def __init__(self, *, strong: bool = True) -> None:
        self.strong = bool(strong)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(()) < 0.5:
            image = image.flip(-1)
            mask = mask.flip(-1)
        angle = (torch.rand(()) * 2 - 1) * (12.0 if self.strong else 5.0)
        scale = 1.0 + (torch.rand(()) * 2 - 1) * (0.12 if self.strong else 0.05)
        translate = (torch.rand(2) * 2 - 1) * (0.08 if self.strong else 0.03)
        radians = angle * math.pi / 180.0
        cosine, sine = torch.cos(radians) / scale, torch.sin(radians) / scale
        theta = image.new_tensor(
            [[cosine, -sine, translate[0]], [sine, cosine, translate[1]]]
        ).unsqueeze(0)
        grid = F.affine_grid(theta, (1, *image.shape), align_corners=False)
        image = F.grid_sample(
            image[None], grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )[0]
        mask = F.grid_sample(mask[None], grid, mode="nearest", align_corners=False)[0]

        gain = torch.empty(3, 1, 1).uniform_(0.65 if self.strong else 0.85, 1.35 if self.strong else 1.15)
        contrast = float(torch.empty(()).uniform_(0.65 if self.strong else 0.85, 1.35 if self.strong else 1.15))
        gamma = float(torch.empty(()).uniform_(0.65 if self.strong else 0.85, 1.5 if self.strong else 1.15))
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = ((image - mean) * contrast + mean).clamp(0, 1)
        image = (image * gain).clamp(0, 1).pow(gamma)
        noise_limit = 40.0 / 255.0 if self.strong else 12.0 / 255.0
        image = (image + torch.randn_like(image) * float(torch.rand(()) * noise_limit)).clamp(0, 1)
        if torch.rand(()) < (0.35 if self.strong else 0.1):
            image = F.avg_pool2d(image[None], 3, stride=1, padding=1)[0]
        if self.strong and torch.rand(()) < 0.25:
            height, width = image.shape[-2:]
            box_h = int(torch.randint(max(2, height // 20), max(3, height // 5), ()))
            box_w = int(torch.randint(max(2, width // 20), max(3, width // 5), ()))
            top = int(torch.randint(0, max(1, height - box_h), ()))
            left = int(torch.randint(0, max(1, width - box_w), ()))
            image[:, top : top + box_h, left : left + box_w] = torch.rand(3, 1, 1)
            mask[:, top : top + box_h, left : left + box_w] = 0
        return image, mask


def _resize(image: torch.Tensor, mask: torch.Tensor, size: tuple[int, int]):
    image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]
    mask = F.interpolate(mask[None], size=size, mode="nearest")[0]
    return image, mask


class HDF5GateDataset(Dataset):
    """Simulator RGB and object-segmentation pairs from Starscream episodes."""

    def __init__(
        self,
        root: str | Path,
        *,
        size: tuple[int, int] = (384, 384),
        augment: bool = False,
        paths: list[Path] | None = None,
    ) -> None:
        root = Path(root)
        self.paths = paths if paths is not None else sorted((*root.glob("*.h5"), *root.glob("*.hdf5")))
        self.size = size
        self.augment = GateNetAugment(strong=True) if augment else None
        self.samples: list[tuple[Path, int]] = []
        for path in self.paths:
            with h5py.File(path, "r", swmr=True) as episode:
                if "observation/rgb" not in episode:
                    continue
                if not any(key in episode for key in ("observation/sim_gate_mask", "observation/gate_mask")):
                    continue
                self.samples.extend((path, index) for index in range(episode["observation/rgb"].shape[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path, frame = self.samples[index]
        with h5py.File(path, "r", swmr=True) as episode:
            rgb = np.asarray(episode["observation/rgb"][frame])
            if "observation/sim_gate_mask" in episode:
                raw_mask = np.asarray(episode["observation/sim_gate_mask"][frame])
            else:
                raw_mask = np.asarray(episode["observation/gate_mask"][frame])
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.ascontiguousarray(raw_mask))[None].float() / 255.0
        image, mask = _resize(image, mask, self.size)
        if self.augment:
            image, mask = self.augment(image, mask)
        return {"image": image, "mask": (mask >= 0.5).float(), "source": "sim"}


class FolderGateDataset(Dataset):
    """Real labeled data in ``images/`` and ``masks/`` with matching stems."""

    def __init__(
        self,
        root: str | Path,
        *,
        size: tuple[int, int] = (384, 384),
        augment: bool = False,
        files: list[Path] | None = None,
        rectifier: CameraRectifier | None = None,
    ) -> None:
        self.root = Path(root)
        image_dir, mask_dir = self.root / "images", self.root / "masks"
        candidates = files if files is not None else sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        mask_by_stem = {path.stem: path for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
        self.samples = [(path, mask_by_stem[path.stem]) for path in candidates if path.stem in mask_by_stem]
        self.size = size
        self.augment = GateNetAugment(strong=False) if augment else None
        self.rectifier = rectifier

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.samples[index]
        rgb = np.array(Image.open(image_path).convert("RGB"), copy=True)
        raw_mask = np.array(Image.open(mask_path).convert("L"), copy=True)
        if self.rectifier is not None:
            rgb = self.rectifier(rgb)
            raw_mask = self.rectifier(raw_mask, is_mask=True)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.ascontiguousarray(raw_mask))[None].float() / 255.0
        image, mask = _resize(image, mask, self.size)
        if self.augment:
            image, mask = self.augment(image, mask)
        return {"image": image, "mask": (mask >= 0.5).float(), "source": "real"}


@dataclass(frozen=True)
class GateNetLoaders:
    train: DataLoader
    validation: DataLoader
    train_samples: int
    validation_samples: int


def build_gatenet_loaders(
    train_datasets: list[Dataset],
    validation_datasets: list[Dataset],
    *,
    batch_size: int,
    workers: int = 4,
    real_fraction: float = 0.5,
    seed: int = 42,
) -> GateNetLoaders:
    """Build balanced SL loaders while preserving dataset-owned session splits."""

    train_datasets = [dataset for dataset in train_datasets if len(dataset)]
    validation_datasets = [dataset for dataset in validation_datasets if len(dataset)]
    if not train_datasets:
        raise ValueError("at least one non-empty GateNet training dataset is required")
    if not 0 < real_fraction < 1:
        raise ValueError("real_fraction must be between zero and one")
    train = ConcatDataset(train_datasets)
    validation = ConcatDataset(validation_datasets) if validation_datasets else train
    sampler = None
    if len(train_datasets) > 1:
        real_count = sum(isinstance(dataset, FolderGateDataset) for dataset in train_datasets)
        sim_count = len(train_datasets) - real_count
        weights = []
        for dataset in train_datasets:
            if isinstance(dataset, FolderGateDataset):
                fraction = real_fraction / max(real_count, 1)
            else:
                fraction = (1.0 - real_fraction) / max(sim_count, 1)
            weights.extend([fraction / len(dataset)] * len(dataset))
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=len(train) >= batch_size,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return GateNetLoaders(train_loader, validation_loader, len(train), len(validation))
