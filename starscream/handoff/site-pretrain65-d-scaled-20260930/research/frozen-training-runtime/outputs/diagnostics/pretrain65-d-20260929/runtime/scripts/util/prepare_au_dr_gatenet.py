#!/usr/bin/env python3
"""Convert AU-DR box annotations into conservative gate-ring pseudo-masks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="extracted au_dr directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--types", default="single,multiple,distant,close",
        help="comma-separated AU-DR scene types; add partial/occlusion only for later robustness training",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=200, help="keep adjacent video-like frames in one split")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--ring-width-fraction", type=float, default=0.13)
    parser.add_argument("--mask-mode", choices=("dark-ring", "box-ring"), default="dark-ring")
    parser.add_argument("--link", choices=("symlink", "hardlink", "copy"), default="symlink")
    return parser.parse_args()


def place_image(source: Path, destination: Path, method: str) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if method == "symlink":
        destination.symlink_to(source.resolve())
    elif method == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def pseudo_mask(
    rgb: np.ndarray,
    annotations: list[dict],
    *,
    width_fraction: float,
    mode: str,
) -> Image.Image:
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    for annotation in annotations:
        x0 = max(0, int(np.floor(annotation["xmin"])))
        y0 = max(0, int(np.floor(annotation["ymin"])))
        x1 = min(width - 1, int(np.ceil(annotation["xmax"])))
        y1 = min(height - 1, int(np.ceil(annotation["ymax"])))
        if x1 <= x0 or y1 <= y0:
            continue
        line_width = max(2, round(min(x1 - x0, y1 - y0) * width_fraction))
        ring_image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(ring_image).rectangle((x0, y0, x1, y1), outline=255, width=line_width)
        ring = np.asarray(ring_image) > 0
        if mode == "dark-ring":
            box_values = luminance[y0 : y1 + 1, x0 : x1 + 1]
            threshold = np.percentile(box_values, 60)
            selected = ring & (luminance <= threshold)
            # A poor threshold should not erase the localization supervision.
            if selected.sum() < 0.2 * ring.sum():
                selected = ring
        else:
            selected = ring
        mask[selected] = 255
    return Image.fromarray(mask).filter(ImageFilter.MaxFilter(3))


def validation_block(block: int, fraction: float) -> bool:
    if fraction <= 0:
        return False
    bucket = (block * 2654435761 % 10000) / 10000.0
    return bucket < fraction


def main() -> None:
    args = arguments()
    if not 0 <= args.validation_fraction < 1:
        raise SystemExit("validation-fraction must be in [0, 1)")
    annotation_path = args.source / "annotations.json"
    image_root = args.source / "images"
    records = json.loads(annotation_path.read_text(encoding="utf-8"))["annotations"]
    allowed_types = {item.strip() for item in args.types.split(",") if item.strip()}
    manifest = []
    counts = {"train": 0, "val": 0}
    for index, record in enumerate(records):
        if record["type"] not in allowed_types or index % args.frame_stride:
            continue
        source = image_root / record["image"]
        if not source.exists():
            continue
        split = "val" if validation_block(index // args.block_size, args.validation_fraction) else "train"
        image_dir, mask_dir = args.output / split / "images", args.output / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        destination = image_dir / record["image"]
        place_image(source, destination, args.link)
        with Image.open(source) as image:
            rgb = np.array(image.convert("RGB"), copy=True)
        mask = pseudo_mask(
            rgb,
            record["annotations"],
            width_fraction=args.ring_width_fraction,
            mode=args.mask_mode,
        )
        mask.save(mask_dir / f"{source.stem}.png")
        manifest.append((split, record["image"], record["type"], len(record["annotations"])))
        counts[split] += 1
    with (args.output / "au_dr_sources.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "image", "scene_type", "gate_count"))
        writer.writerows(manifest)
    metadata = {
        "source": "open-airlab/AU-DR",
        "annotation_kind": "bounding-box-derived pseudo-mask",
        "mask_mode": args.mask_mode,
        "types": sorted(allowed_types),
        "ring_width_fraction": args.ring_width_fraction,
        "block_size": args.block_size,
        "validation_fraction": args.validation_fraction,
        "counts": counts,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"prepared AU-DR pseudo-masks: train={counts['train']} val={counts['val']} output={args.output}")


if __name__ == "__main__":
    main()
