"""Backend-neutral distribution coverage and support comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..tracks import load_track
from .descriptors import CourseDescriptor, TRANSITION_COLUMNS
from .schema import RacingDistributionConfig
from .validator import RacingDistributionValidator


TRANSITION_EDGES: Mapping[str, tuple[float, ...]] = {
    "segment_length_m": (2.0, 4.0, 6.0, 9.0, 13.0, 18.0),
    "signed_yaw_turn_degrees": (-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0),
    "turn_3d_degrees": (15.0, 30.0, 50.0, 75.0, 110.0, 145.0),
    "elevation_change_m": (-3.0, -1.5, -0.5, 0.5, 1.5, 3.0),
    "absolute_slope": (0.08, 0.18, 0.30, 0.45, 0.65),
    "gate_normal_mismatch_degrees": (5.0, 12.0, 25.0, 45.0),
    "gate_up_change_degrees": (10.0, 30.0, 60.0, 100.0, 145.0),
    "curvature_proxy_m_inv": (0.10, 0.20, 0.40, 0.70, 1.10),
    "torsion_proxy": (-0.50, -0.20, -0.05, 0.05, 0.20, 0.50),
}

TRANSITION_SCALES = np.asarray(
    [10.0, 90.0, 90.0, 2.5, 0.5, 40.0, 90.0, 0.6, 0.4], np.float64
)


def _entropy(histogram: np.ndarray) -> float:
    probability = histogram.astype(np.float64)
    probability /= max(float(probability.sum()), 1.0)
    selected = probability[probability > 0.0]
    raw = float(-np.sum(selected * np.log(selected)))
    maximum = np.log(max(len(probability), 2))
    return raw / maximum


def _transition_metrics(descriptors: Sequence[CourseDescriptor]) -> dict[str, Any]:
    if not descriptors:
        return {"count": 0, "marginal": {}, "joint_demand_cells": 0}
    values = np.concatenate([item.transition_features[:, :9] for item in descriptors], axis=0)
    marginal: dict[str, Any] = {}
    bins_by_name: dict[str, np.ndarray] = {}
    for index, name in enumerate(TRANSITION_COLUMNS[:9]):
        edges = np.asarray(TRANSITION_EDGES[name], np.float64)
        bins = np.searchsorted(edges, values[:, index], side="right")
        histogram = np.bincount(bins, minlength=len(edges) + 1)
        bins_by_name[name] = bins
        marginal[name] = {
            "occupied_bins": int(np.count_nonzero(histogram)),
            "bin_count": len(histogram),
            "occupancy_fraction": float(np.count_nonzero(histogram) / len(histogram)),
            "normalized_entropy": _entropy(histogram),
            "histogram": histogram.tolist(),
        }
    demand_columns = (
        "segment_length_m", "turn_3d_degrees", "absolute_slope",
        "gate_up_change_degrees", "curvature_proxy_m_inv",
    )
    joint = set(zip(*(bins_by_name[name].tolist() for name in demand_columns)))
    return {
        "count": len(values),
        "marginal": marginal,
        "joint_demand_cells": len(joint),
        "mean_marginal_occupancy_fraction": float(np.mean([
            item["occupancy_fraction"] for item in marginal.values()
        ])),
        "mean_marginal_entropy": float(np.mean([
            item["normalized_entropy"] for item in marginal.values()
        ])),
    }


def _support_distance(
    train: Sequence[CourseDescriptor], validation: Sequence[CourseDescriptor],
) -> dict[str, float | None]:
    if not train or not validation:
        return {"p50": None, "p90": None, "p99": None, "maximum": None}
    reference = np.concatenate([item.transition_features[:, :9] for item in train], axis=0)
    query = np.concatenate([item.transition_features[:, :9] for item in validation], axis=0)
    reference = reference / TRANSITION_SCALES
    query = query / TRANSITION_SCALES
    nearest: list[np.ndarray] = []
    for start in range(0, len(query), 256):
        chunk = query[start : start + 256]
        squared = np.sum((chunk[:, None, :] - reference[None, :, :]) ** 2, axis=-1)
        nearest.append(np.sqrt(np.min(squared, axis=1)))
    values = np.concatenate(nearest)
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, np.float64)
    if not len(data):
        return {}
    return {
        "minimum": float(data.min()), "p10": float(np.quantile(data, 0.10)),
        "mean": float(data.mean()), "p90": float(np.quantile(data, 0.90)),
        "maximum": float(data.max()),
    }


def compare_distribution_manifests(
    manifests: Sequence[str | Path],
    *,
    output: str | Path | None = None,
    require_dynamic: bool = False,
) -> dict[str, Any]:
    """Compare generator outputs without opening any sealed benchmark track."""

    if len(manifests) < 1:
        raise ValueError("at least one distribution manifest is required")
    comparison: dict[str, Any] = {
        "schema": "starscream-racing-distribution-comparison-v1",
        "scientific_scope": "generated-train-to-generated-validation-only",
        "distributions": {},
    }
    for raw_path in manifests:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = RacingDistributionConfig(**dict(payload.get("config") or {}))
        validator = RacingDistributionValidator(config, cache_directory=path.parent / "racing-lines")
        report = validator.validate_manifest(path, require_dynamic=require_dynamic)
        split_descriptors: dict[str, list[CourseDescriptor]] = {
            "train": [], "validation": [], "composition_holdout": [],
        }
        # Successful validation produces reports in record order.  Manifest
        # generation itself makes artifact failures fatal, so this explicit
        # check prevents a misleading comparison if a file was later damaged.
        if len(report.track_reports) != len(payload.get("records", ())):
            raise ValueError(f"cannot compare damaged manifest {path}")
        for record, track_report in zip(payload["records"], report.track_reports):
            if track_report.descriptor is not None:
                split_descriptors[str(record["split"])].append(track_report.descriptor)
        all_descriptors = split_descriptors["train"] + split_descriptors["validation"]
        fields = (
            "length_m", "gate_density_per_100m", "vertical_excursion_m",
            "p95_turn_degrees", "p99_curvature_m_inv", "hard_transition_fraction",
            "qualified_speed_mps", "lateral_demand_ratio",
            "teacher_average_speed_mps",
        )
        name = str(payload.get("backend") or path.parent.name)
        if name in comparison["distributions"]:
            name = f"{name}:{path.parent.name}"
        comparison["distributions"][name] = {
            "manifest": str(path.resolve()),
            "validation": report.to_mapping(include_track_reports=False),
            "generation": payload.get("generation", {}),
            "course_metrics": {
                field: _summary([
                    item.values[field] for item in all_descriptors
                    if field not in {
                        "qualified_speed_mps", "lateral_demand_ratio",
                        "teacher_average_speed_mps",
                    }
                    or item.dynamic_available
                ])
                for field in fields
            },
            "transition_metrics": {
                split: _transition_metrics(items)
                for split, items in split_descriptors.items()
            },
            "train_to_validation_nearest_transition_distance": _support_distance(
                split_descriptors["train"], split_descriptors["validation"]
            ),
            "train_to_composition_holdout_nearest_transition_distance": _support_distance(
                split_descriptors["train"], split_descriptors["composition_holdout"]
            ),
        }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
    return comparison


def write_distribution_gallery(
    manifests: Sequence[str | Path], output: str | Path, *, per_split: int = 4,
) -> Path:
    """Render XY/XZ projections for fast human topology inspection."""

    import matplotlib.pyplot as plt

    rows: list[tuple[str, str, list[Any]]] = []
    for raw_path in manifests:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = str(payload.get("backend") or path.parent.name)
        for split in ("train", "validation", "composition_holdout"):
            records = [item for item in payload["records"] if item["split"] == split][:per_split]
            if records:
                rows.append((label, split, [load_track(path.parent / item["path"]) for item in records]))
    figure, axes = plt.subplots(len(rows), 2, figsize=(14, max(4.0 * len(rows), 5.0)), squeeze=False)
    for row, (backend, split, tracks) in enumerate(rows):
        for track in tracks:
            points = np.stack([gate.position for gate in track.gates])
            closed = np.concatenate([points, points[:1]], axis=0)
            axes[row, 0].plot(closed[:, 0], closed[:, 1], marker="o", alpha=0.75)
            axes[row, 1].plot(np.arange(len(points) + 1), np.r_[points[:, 2], points[0, 2]], marker="o", alpha=0.75)
        axes[row, 0].set_title(f"{backend} / {split}: XY")
        axes[row, 0].set_aspect("equal", adjustable="datalim")
        axes[row, 1].set_title(f"{backend} / {split}: altitude by gate")
        for axis in axes[row]:
            axis.grid(alpha=0.25)
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
