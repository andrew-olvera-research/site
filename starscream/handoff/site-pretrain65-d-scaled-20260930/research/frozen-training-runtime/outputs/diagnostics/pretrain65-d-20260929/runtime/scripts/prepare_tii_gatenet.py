#!/usr/bin/env python3
"""Convert TII Race Against the Machine corner labels into GateNet ring masks."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil

from PIL import Image, ImageDraw


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="extracted TII dataset root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--link", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--ring-width-fraction", type=float, default=0.08)
    return parser.parse_args()


def image_for_label(label: Path) -> Path | None:
    camera_name = label.parent.name.replace("labels_", "camera_", 1)
    camera_dir = label.parent.parent / camera_name
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        candidate = camera_dir / f"{label.stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = list(label.parent.parent.glob(f"camera_*/{label.stem}.*"))
    return matches[0] if matches else None


def mask_from_tii_label(label: Path, image_size: tuple[int, int], width_fraction: float) -> Image.Image:
    width, height = image_size
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for line in label.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 17:
            continue
        values = [float(value) for value in fields]
        box_width, box_height = values[3] * width, values[4] * height
        line_width = max(2, round(min(box_width, box_height) * width_fraction))
        points = [
            (values[5] * width, values[6] * height),
            (values[8] * width, values[9] * height),
            (values[11] * width, values[12] * height),
            (values[14] * width, values[15] * height),
        ]
        draw.line([*points, points[0]], fill=255, width=line_width, joint="curve")
    return mask


def place_image(source: Path, destination: Path, method: str) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if method == "symlink":
        destination.symlink_to(source.resolve())
    elif method == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def main() -> None:
    args = arguments()
    if not 0 <= args.validation_fraction < 1:
        raise SystemExit("validation-fraction must be in [0, 1)")
    labels = sorted(args.source.glob("**/labels_flight-*/*.txt"))
    if not labels:
        raise SystemExit(f"no TII labels_flight-* directories found under {args.source}")
    flights = sorted({label.parent.parent for label in labels})
    validation_count = max(1, round(len(flights) * args.validation_fraction)) if len(flights) > 1 else 0
    validation_flights = set(flights[-validation_count:]) if validation_count else set()
    manifest_rows = []
    written = 0
    for label_index, label in enumerate(labels):
        if label_index % args.frame_stride:
            continue
        image_path = image_for_label(label)
        if image_path is None:
            continue
        flight = label.parent.parent
        split = "val" if flight in validation_flights else "train"
        image_dir = args.output / split / "images"
        mask_dir = args.output / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        prefix = flight.name.replace("flight-", "")
        name = f"tii_{prefix}_{label.stem}"
        image_destination = image_dir / f"{name}{image_path.suffix.lower()}"
        mask_destination = mask_dir / f"{name}.png"
        with Image.open(image_path) as image:
            image_size = image.size
        place_image(image_path, image_destination, args.link)
        mask_from_tii_label(label, image_size, args.ring_width_fraction).save(mask_destination)
        manifest_rows.append((split, str(image_path), str(label), name))
        written += 1
    with (args.output / "tii_sources.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "image", "label", "name"))
        writer.writerows(manifest_rows)
    print(
        f"prepared {written} images from {len(flights)} flight sessions; "
        f"validation_sessions={len(validation_flights)} output={args.output}"
    )


if __name__ == "__main__":
    main()
