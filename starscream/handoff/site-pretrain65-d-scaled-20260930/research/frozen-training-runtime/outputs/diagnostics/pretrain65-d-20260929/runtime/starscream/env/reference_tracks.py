"""Reference-centered racing-course augmentation.

This generator preserves the ordered local maneuver sequence of an immutable
real course while perturbing its metric realization.  It is intentionally a
domain-adaptation distribution, not evidence of zero-shot generalization to
the source course.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .procedural_tracks import geometry_fingerprint, save_track_yaml
from .tracks import Gate, Track, forward_up_quaternion, load_track


@dataclass(frozen=True, slots=True)
class ReferenceAugmentationConfig:
    horizontal_scale_range: tuple[float, float] = (0.995, 1.005)
    horizontal_jitter_sigma_m: float = 0.020
    horizontal_jitter_limit_m: float = 0.05
    vertical_jitter_sigma_m: float = 0.015
    vertical_jitter_limit_m: float = 0.03
    global_vertical_offset_range_m: tuple[float, float] = (0.15, 0.30)
    orientation_jitter_degrees: float = 1.0
    aperture_range_m: tuple[float, float] = (1.44, 1.50)
    preserve_reference_aperture: bool = False
    aperture_scale_range: tuple[float, float] = (0.985, 1.015)
    translation_range_m: float = 0.80
    global_yaw_range_radians: tuple[float, float] = (-np.pi, np.pi)
    stacked_pair_xy_tolerance_m: float = 0.30
    stacked_pair_relative_jitter_m: float = 0.025
    minimum_rigid_deformation_rms_m: float = 0.015
    maximum_rigid_deformation_rms_m: float = 0.10
    minimum_gate_height_m: float = 0.98
    generation_attempts: int = 200


def _yaw_rotate(vectors: np.ndarray, yaw: float) -> np.ndarray:
    rotation = np.asarray([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)],
    ])
    result = np.asarray(vectors, np.float64).copy()
    result[..., :2] = result[..., :2] @ rotation.T
    return result


def _rigid_deformation_rms(reference: np.ndarray, candidate: np.ndarray) -> float:
    """RMS deformation after removing XY translation and rotation."""

    left = np.asarray(reference, np.float64).copy()
    right = np.asarray(candidate, np.float64).copy()
    left[:, :2] -= left[:, :2].mean(0)
    right[:, :2] -= right[:, :2].mean(0)
    covariance = right[:, :2].T @ left[:, :2]
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = right.copy()
    aligned[:, :2] = right[:, :2] @ rotation
    # A global altitude offset has no trajectory content either.
    aligned[:, 2] -= float(np.mean(aligned[:, 2] - left[:, 2]))
    return float(np.sqrt(np.mean(np.sum((aligned - left) ** 2, axis=1))))


def _turns(points: np.ndarray) -> np.ndarray:
    segments = np.roll(points, -1, axis=0) - points
    units = segments / np.maximum(np.linalg.norm(segments, axis=1)[:, None], 1e-9)
    return np.degrees(np.arccos(np.clip(np.sum(np.roll(units, 1, axis=0) * units, axis=1), -1, 1)))


def reference_track_audit(
    reference: Track, track: Track, *, cache_directory: str | Path | None = None,
    config: ReferenceAugmentationConfig | None = None,
) -> dict[str, Any]:
    from ..mpcc import RacingLinePlanner, RacingLinePlannerConfig

    cfg = config or ReferenceAugmentationConfig()
    reference_points = np.stack([gate.position for gate in reference.gates]).astype(np.float64)
    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    reference_spacing = np.linalg.norm(np.roll(reference_points, -1, axis=0) - reference_points, axis=1)
    spacing = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    deformation = _rigid_deformation_rms(reference_points, points)
    turn_error = np.abs(_turns(points) - _turns(reference_points))
    turns = _turns(points)
    line = RacingLinePlanner(RacingLinePlannerConfig(
        sample_count=1200, offset_iterations=30,
        cache_directory=str(cache_directory) if cache_directory is not None else None,
    )).plan(track)
    query = np.linspace(0.0, line.length, 2400, endpoint=False)
    sampled = line.evaluate(query)
    gate_frames = line.evaluate(np.asarray(line.gate_progress))
    normals = np.stack([gate.normal for gate in track.gates]).astype(np.float64)
    minimum_line_gate_alignment = float(np.min(np.sum(gate_frames["tangent"] * normals, axis=1)))
    global_yaw = float((track.metadata or {}).get("global_yaw_radians", 0.0))
    expected_normals = _yaw_rotate(
        np.stack([gate.normal for gate in reference.gates]).astype(np.float64),
        global_yaw,
    )
    directed_normal_alignment = np.sum(expected_normals * normals, axis=1)
    minimum_reference_normal_alignment = float(np.min(directed_normal_alignment))
    stacked = [
        index for index in range(len(points))
        if np.linalg.norm(reference_points[index, :2] - reference_points[(index + 1) % len(points), :2])
        < cfg.stacked_pair_xy_tolerance_m
        and abs(reference_points[index, 2] - reference_points[(index + 1) % len(points), 2]) > 1.0
    ]
    reasons: list[str] = []
    if len(track.gates) != len(reference.gates):
        reasons.append("gate-count")
    if np.any(spacing / reference_spacing < 0.82) or np.any(spacing / reference_spacing > 1.18):
        reasons.append("local-spacing-drift")
    if float(np.mean(turn_error)) > 9.0 or float(np.max(turn_error)) > 20.0:
        reasons.append("turn-sequence-drift")
    if deformation < cfg.minimum_rigid_deformation_rms_m:
        reasons.append("too-close-to-reference")
    if deformation > cfg.maximum_rigid_deformation_rms_m:
        reasons.append("too-far-from-reference")
    low_cluster_count = int(np.sum(points[:, 2] <= float(points[:, 2].min()) + 0.15))
    if low_cluster_count < 3:
        reasons.append("low-gate-cluster-lost")
    if float(points[:, 2].min()) < 0.98:
        reasons.append("gate-below-height-floor")
    if minimum_line_gate_alignment < 0.94:
        reasons.append("racing-line-gate-direction")
    if minimum_reference_normal_alignment < 0.94:
        reasons.append("reference-traversal-direction")
    for index in stacked:
        following = (index + 1) % len(points)
        reference_pair_delta = (
            reference_points[index, :2] - reference_points[following, :2]
        )
        candidate_pair_delta = points[index, :2] - points[following, :2]
        if abs(np.linalg.norm(candidate_pair_delta) - np.linalg.norm(reference_pair_delta)) > max(
            0.10, 2.5 * cfg.stacked_pair_relative_jitter_m,
        ):
            reasons.append("stacked-pair-broken")
        if abs(points[index, 2] - points[following, 2]) < 1.70:
            reasons.append("stacked-pair-compressed")
    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "track": track.name,
        "reference": reference.name,
        "gate_count": len(track.gates),
        "racing_line_length_m": float(line.length),
        "vertical_excursion_m": float(np.ptp(points[:, 2])),
        "mean_turn_degrees": float(np.mean(turns)),
        "p99_curvature_m_inv": float(np.quantile(sampled["curvature"], 0.99)),
        "minimum_line_gate_alignment": minimum_line_gate_alignment,
        "minimum_reference_normal_alignment": minimum_reference_normal_alignment,
        "rigid_deformation_rms_m": deformation,
        "mean_local_spacing_ratio": float(np.mean(spacing / reference_spacing)),
        "minimum_local_spacing_ratio": float(np.min(spacing / reference_spacing)),
        "maximum_local_spacing_ratio": float(np.max(spacing / reference_spacing)),
        "mean_turn_error_degrees": float(np.mean(turn_error)),
        "maximum_turn_error_degrees": float(np.max(turn_error)),
        "low_gate_count": int(np.sum(points[:, 2] <= 1.18)),
        "low_cluster_count": low_cluster_count,
        "geometry_fingerprint": geometry_fingerprint(track),
    }


def augment_reference_track(
    reference: Track, *, seed: int, name: str, split: str,
    config: ReferenceAugmentationConfig | None = None,
) -> Track:
    cfg = config or ReferenceAugmentationConfig()
    rng = np.random.default_rng(int(seed))
    original = np.stack([gate.position for gate in reference.gates]).astype(np.float64)
    center = original[:, :2].mean(0)
    scale = float(rng.uniform(*cfg.horizontal_scale_range))
    points = original.copy()
    points[:, :2] = center + scale * (points[:, :2] - center)
    jitter = np.column_stack([
        np.clip(rng.normal(0, cfg.horizontal_jitter_sigma_m, (len(points), 2)),
                -cfg.horizontal_jitter_limit_m, cfg.horizontal_jitter_limit_m),
        np.clip(rng.normal(0, cfg.vertical_jitter_sigma_m, len(points)),
                -cfg.vertical_jitter_limit_m, cfg.vertical_jitter_limit_m),
    ])
    # Preserve vertically stacked motifs as one object; only their heights and
    # a tiny relative XY displacement vary.
    for index in range(len(points)):
        following = (index + 1) % len(points)
        if (np.linalg.norm(original[index, :2] - original[following, :2])
                < cfg.stacked_pair_xy_tolerance_m
                and abs(original[index, 2] - original[following, 2]) > 1.0):
            joint = 0.5 * (jitter[index, :2] + jitter[following, :2])
            relative = rng.uniform(
                -cfg.stacked_pair_relative_jitter_m,
                cfg.stacked_pair_relative_jitter_m, size=2,
            )
            jitter[index, :2] = joint + relative
            jitter[following, :2] = joint - relative
    points += jitter
    points[:, 2] += float(rng.uniform(*cfg.global_vertical_offset_range_m))
    points[:, 2] = np.maximum(points[:, 2], cfg.minimum_gate_height_m)
    yaw = float(rng.uniform(*cfg.global_yaw_range_radians))
    centered = points.copy()
    centered[:, :2] -= center
    points = _yaw_rotate(centered, yaw)
    points[:, :2] += center + rng.uniform(-cfg.translation_range_m, cfg.translation_range_m, 2)

    gates: list[Gate] = []
    orientation_jitter: list[float] = []
    aperture = float(rng.uniform(*cfg.aperture_range_m))
    aperture_scale = float(rng.uniform(*cfg.aperture_scale_range))
    for index, (source, position) in enumerate(zip(reference.gates, points)):
        # Preserve the rendered gate frame here. ``Gate.normal`` is already
        # reversed for enter_from_opposite_side; feeding that directed normal
        # into a new Gate and then copying the flag reverses it a second time.
        # This previously corrupted Split-S/ladder topology while leaving gate
        # centers deceptively close to the reference course.
        normal = _yaw_rotate(source.physical_normal[None], yaw)[0]
        local_yaw = float(np.radians(rng.uniform(-cfg.orientation_jitter_degrees, cfg.orientation_jitter_degrees)))
        normal = _yaw_rotate(normal[None], local_yaw)[0]
        up = np.asarray([0.0, 0.0, 1.0])
        size = (
            np.asarray(source.size, np.float64) * aperture_scale
            if cfg.preserve_reference_aperture
            else aperture * rng.uniform(0.985, 1.015, 2)
        )
        gates.append(Gate(
            position=position.astype(np.float32),
            quaternion_wxyz=forward_up_quaternion(normal, up),
            size=size.astype(np.float32),
            name=f"gate_{index:02d}_{source.name}",
            enter_from_opposite_side=source.enter_from_opposite_side,
        ))
        orientation_jitter.append(float(np.degrees(local_yaw)))
    # World bounds must transform with the augmented course. Reusing the
    # source's axis-aligned rectangle made global yaw/translation alter reset
    # clipping and racing-line corridors despite identical local topology.
    # Reserve room for the 3.4 m approach spawn and aggressive overshoot.
    bounds = np.asarray([
        [float(points[:, 0].min() - 8.0), float(points[:, 0].max() + 8.0)],
        [float(points[:, 1].min() - 8.0), float(points[:, 1].max() + 8.0)],
        [0.0, max(float(points[:, 2].max() + 5.0), float(reference.bounds[2, 1]))],
    ], np.float32)
    return Track(
        name=name, gates=tuple(gates), bounds=bounds, loop=reference.loop,
        metadata={
            "source": "reference-centered-augmentation-v1",
            "reference_name": reference.name,
            "reference_geometry_fingerprint": geometry_fingerprint(reference),
            "split": split, "seed": int(seed), "horizontal_scale": scale,
            "global_yaw_radians": yaw,
            "orientation_jitter_degrees": orientation_jitter,
            "preserve_reference_aperture": cfg.preserve_reference_aperture,
            "aperture_scale": aperture_scale,
            "scientific_scope": "benchmark-adaptation-not-zero-shot",
        },
    )


def generate_reference_manifest(
    output: str | Path, *, references: list[str | Path], train_count: int,
    validation_count: int, seed: int,
    config: ReferenceAugmentationConfig | None = None,
    configs_by_reference: dict[str, ReferenceAugmentationConfig] | None = None,
) -> Path:
    cfg = config or ReferenceAugmentationConfig()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    loaded = [load_track(path) for path in references]
    if not loaded:
        raise ValueError("at least one reference course is required")
    records: list[dict[str, Any]] = []
    seen = {geometry_fingerprint(track) for track in loaded}
    for split, count, offset in (("train", train_count, 0), ("validation", validation_count, 10_000_000)):
        for slot in range(count):
            reference = loaded[slot % len(loaded)]
            reference_cfg = (configs_by_reference or {}).get(reference.name, cfg)
            accepted = None
            failures: list[str] = []
            for attempt in range(reference_cfg.generation_attempts):
                track_seed = int(seed + offset + slot * 1009 + attempt * 104729)
                family = f"{reference.name}_neighborhood"
                name = f"reference_{split}_{slot:03d}_{reference.name}_{track_seed}"
                candidate = augment_reference_track(
                    reference, seed=track_seed, name=name, split=split,
                    config=reference_cfg,
                )
                audit = reference_track_audit(
                    reference, candidate, cache_directory=root / "racing-lines",
                    config=reference_cfg,
                )
                fingerprint = geometry_fingerprint(candidate)
                failures.extend(audit["reasons"])
                if audit["valid"] and fingerprint not in seen:
                    accepted = candidate, audit, track_seed, family
                    break
            if accepted is None:
                raise RuntimeError(f"could not augment {reference.name}: {sorted(set(failures))}")
            track, audit, track_seed, family = accepted
            relative = Path("tracks") / split / f"{track.name}.yaml"
            save_track_yaml(track, root / relative)
            artifact = load_track(root / relative)
            fingerprint = geometry_fingerprint(artifact)
            seen.add(fingerprint)
            records.append({
                "name": artifact.name, "family": family, "split": split,
                "seed": track_seed, "path": str(relative),
                "track_fingerprint": artifact.fingerprint,
                "geometry_fingerprint": fingerprint,
                "reference_name": reference.name,
                "reference_geometry_fingerprint": geometry_fingerprint(reference),
                "static_audit": audit, "qualified_speed_mps": None,
                "qualification": None,
            })
    payload = {
        "schema": "starscream-procedural-track-manifest-v1",
        "generator": "reference-centered-augmentation-v1", "seed": int(seed),
        "config": asdict(cfg),
        "configs_by_reference": {
            name: asdict(reference_cfg)
            for name, reference_cfg in (configs_by_reference or {}).items()
        },
        "scientific_scope": "benchmark-adaptation-not-zero-shot",
        "family_weights": {
            f"{track.name}_neighborhood": 1.0 / len(loaded) for track in loaded
        },
        "references": [
            {"name": track.name, "geometry_fingerprint": geometry_fingerprint(track)}
            for track in loaded
        ],
        "records": records,
    }
    destination = root / "manifest.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
