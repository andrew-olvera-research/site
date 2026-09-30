"""Topology-preserving metric augmentation around named real courses.

Named courses remain the training tasks.  These augmentations only thicken
each task's local metric neighborhood; they never synthesize new topology and
must be independently labeled by MPCC before use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..procedural_tracks import geometry_fingerprint, save_track_yaml
from ..tracks import Gate, Track, forward_up_quaternion, load_track


@dataclass(frozen=True, slots=True)
class NamedCourseAugmentationConfig:
    horizontal_scale_range: tuple[float, float] = (0.99, 1.01)
    smooth_xy_sigma_m: float = 0.10
    smooth_xy_limit_m: float = 0.24
    smooth_z_sigma_m: float = 0.06
    smooth_z_limit_m: float = 0.14
    orientation_jitter_degrees: float = 2.0
    aperture_scale_range: tuple[float, float] = (0.98, 1.02)
    minimum_gate_height_m: float = 0.90
    stacked_pair_xy_tolerance_m: float = 0.30
    minimum_deformation_rms_m: float = 0.055
    maximum_deformation_rms_m: float = 0.65
    minimum_spacing_ratio: float = 0.88
    maximum_spacing_ratio: float = 1.12
    maximum_mean_turn_error_degrees: float = 6.0
    maximum_turn_error_degrees: float = 15.0
    maximum_mean_route_context_shift: float = 0.12
    maximum_route_context_shift: float = 0.25
    generation_attempts: int = 300


def _yaw_rotate(values: np.ndarray, yaw: float) -> np.ndarray:
    source = np.asarray(values, np.float64)
    result = source.copy()
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    result[..., 0] = c * source[..., 0] - s * source[..., 1]
    result[..., 1] = s * source[..., 0] + c * source[..., 1]
    return result


def _smooth_periodic_noise(
    rng: np.random.Generator, count: int, dimensions: int, sigma: float, limit: float,
) -> np.ndarray:
    values = rng.normal(size=(count, dimensions))
    # Low-pass in route phase.  Two passes retain local variation while
    # preventing independent gate jitter from creating impossible kinks.
    for _ in range(2):
        values = (
            np.roll(values, 1, axis=0) + 2.0 * values + np.roll(values, -1, axis=0)
        ) / 4.0
    values -= np.mean(values, axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(values**2)))
    if rms > 1.0e-9:
        values *= sigma / rms
    return np.clip(values, -limit, limit)


def _turns(points: np.ndarray) -> np.ndarray:
    segments = np.roll(points, -1, axis=0) - points
    units = segments / np.maximum(np.linalg.norm(segments, axis=1)[:, None], 1.0e-9)
    return np.degrees(np.arccos(np.clip(np.sum(np.roll(units, 1, axis=0) * units, axis=1), -1, 1)))


def _deformation_rms(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.asarray(reference, np.float64).copy()
    right = np.asarray(candidate, np.float64).copy()
    left[:, :2] -= np.mean(left[:, :2], axis=0)
    right[:, :2] -= np.mean(right[:, :2], axis=0)
    covariance = right[:, :2].T @ left[:, :2]
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    right[:, :2] = right[:, :2] @ rotation
    right[:, 2] -= float(np.mean(right[:, 2] - left[:, 2]))
    return float(np.sqrt(np.mean(np.sum((right - left) ** 2, axis=1))))


def route_contexts(track: Track, future_gates: int = 6) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index, anchor in enumerate(track.gates):
        rotation = anchor.directed_rotation.astype(np.float64)
        values: list[np.ndarray] = []
        for offset in range(1, future_gates + 1):
            gate = track.gates[(index + offset) % len(track.gates)]
            values.extend((
                rotation.T @ (gate.position.astype(np.float64) - anchor.position.astype(np.float64)) / 20.0,
                rotation.T @ gate.normal.astype(np.float64),
                gate.size.astype(np.float64) / 3.0,
            ))
        rows.append(np.concatenate(values))
    return np.stack(rows)


def augment_named_course(
    reference: Track, *, seed: int, name: str, split: str,
    config: NamedCourseAugmentationConfig | None = None,
) -> Track:
    cfg = config or NamedCourseAugmentationConfig()
    rng = np.random.default_rng(int(seed))
    original = np.stack([gate.position for gate in reference.gates]).astype(np.float64)
    points = original.copy()
    center = np.mean(points[:, :2], axis=0)
    scale = float(rng.uniform(*cfg.horizontal_scale_range))
    points[:, :2] = center + scale * (points[:, :2] - center)
    xy_noise = _smooth_periodic_noise(
        rng, len(points), 2, cfg.smooth_xy_sigma_m, cfg.smooth_xy_limit_m,
    )
    z_noise = _smooth_periodic_noise(
        rng, len(points), 1, cfg.smooth_z_sigma_m, cfg.smooth_z_limit_m,
    )[:, 0]
    # Vertically stacked gates are one maneuver object.  Moving their XY
    # centers independently destroys the split-S while passing coarse audits.
    for index in range(len(points)):
        following = (index + 1) % len(points)
        if (
            np.linalg.norm(original[index, :2] - original[following, :2])
            < cfg.stacked_pair_xy_tolerance_m
            and abs(float(original[index, 2] - original[following, 2])) > 1.0
        ):
            shared = 0.5 * (xy_noise[index] + xy_noise[following])
            xy_noise[index] = shared
            xy_noise[following] = shared
    points[:, :2] += xy_noise
    points[:, 2] = np.maximum(points[:, 2] + z_noise, cfg.minimum_gate_height_m)

    aperture_scale = float(rng.uniform(*cfg.aperture_scale_range))
    gates: list[Gate] = []
    orientation_jitter: list[float] = []
    for index, (source, position) in enumerate(zip(reference.gates, points)):
        local_yaw = float(np.radians(rng.uniform(
            -cfg.orientation_jitter_degrees, cfg.orientation_jitter_degrees,
        )))
        normal = _yaw_rotate(source.physical_normal[None], local_yaw)[0]
        up = _yaw_rotate(source.up[None], local_yaw)[0]
        gates.append(Gate(
            position=position.astype(np.float32),
            quaternion_wxyz=forward_up_quaternion(normal, up),
            size=(source.size.astype(np.float64) * aperture_scale).astype(np.float32),
            name=f"gate_{index:02d}_{source.name}",
            enter_from_opposite_side=source.enter_from_opposite_side,
            kind=source.kind,
            render=source.render,
        ))
        orientation_jitter.append(float(np.degrees(local_yaw)))
    bounds = np.asarray([
        [float(points[:, 0].min() - 8.0), float(points[:, 0].max() + 8.0)],
        [float(points[:, 1].min() - 8.0), float(points[:, 1].max() + 8.0)],
        [0.0, max(float(points[:, 2].max() + 5.0), float(reference.bounds[2, 1]))],
    ], np.float32)
    return Track(
        name=name, gates=tuple(gates), bounds=bounds, loop=reference.loop,
        metadata={
            "source": "named-course-smooth-augmentation-v1",
            "reference_name": reference.name,
            "reference_geometry_fingerprint": geometry_fingerprint(reference),
            "split": split,
            "seed": int(seed),
            "horizontal_scale": scale,
            "aperture_scale": aperture_scale,
            "orientation_jitter_degrees": orientation_jitter,
            "scientific_scope": "local-family-regularization-not-new-topology",
        },
    )


def audit_named_course_augmentation(
    reference: Track, candidate: Track, *,
    config: NamedCourseAugmentationConfig | None = None,
) -> dict[str, Any]:
    from ...mpcc import RacingLinePlanner, RacingLinePlannerConfig

    cfg = config or NamedCourseAugmentationConfig()
    original = np.stack([gate.position for gate in reference.gates]).astype(np.float64)
    points = np.stack([gate.position for gate in candidate.gates]).astype(np.float64)
    reference_spacing = np.linalg.norm(np.roll(original, -1, axis=0) - original, axis=1)
    spacing = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    ratio = spacing / np.maximum(reference_spacing, 1.0e-9)
    turn_error = np.abs(_turns(points) - _turns(original))
    context_shift = np.linalg.norm(
        route_contexts(candidate) - route_contexts(reference), axis=1,
    )
    normal_alignment = np.asarray([
        float(left.normal @ right.normal)
        for left, right in zip(reference.gates, candidate.gates)
    ])
    deformation = _deformation_rms(original, points)
    reasons: list[str] = []
    if len(candidate.gates) != len(reference.gates):
        reasons.append("gate-count")
    if [gate.kind for gate in candidate.gates] != [gate.kind for gate in reference.gates]:
        reasons.append("checkpoint-kind")
    if [gate.render for gate in candidate.gates] != [gate.render for gate in reference.gates]:
        reasons.append("render-contract")
    if not (cfg.minimum_deformation_rms_m <= deformation <= cfg.maximum_deformation_rms_m):
        reasons.append("deformation-range")
    if float(np.min(ratio)) < cfg.minimum_spacing_ratio or float(np.max(ratio)) > cfg.maximum_spacing_ratio:
        reasons.append("spacing-drift")
    if (
        float(np.mean(turn_error)) > cfg.maximum_mean_turn_error_degrees
        or float(np.max(turn_error)) > cfg.maximum_turn_error_degrees
    ):
        reasons.append("turn-drift")
    if float(np.min(normal_alignment)) < np.cos(np.radians(5.0)):
        reasons.append("orientation-drift")
    if (
        float(np.mean(context_shift)) > cfg.maximum_mean_route_context_shift
        or float(np.max(context_shift)) > cfg.maximum_route_context_shift
    ):
        reasons.append("route-context-drift")
    if float(np.min(points[:, 2])) < cfg.minimum_gate_height_m - 1.0e-5:
        reasons.append("height-floor")
    # Racing-line construction is the expensive check. Do not spend it on a
    # candidate already rejected by local geometry/observation contracts.
    try:
        if reasons:
            raise RuntimeError("cheap-audit-rejected")
        line = RacingLinePlanner(RacingLinePlannerConfig(
            sample_count=900, offset_iterations=20,
        )).plan(candidate)
        gate_frames = line.evaluate(np.asarray(line.gate_progress))
        line_alignment = float(np.min(np.sum(
            gate_frames["tangent"] * np.stack([gate.normal for gate in candidate.gates]), axis=1,
        )))
        # Hidden flag/route checkpoints in official maps can intentionally
        # constrain a side rather than align with the spline tangent (the GQ
        # canonical minimum is ~0.53).  Preserve that contract instead of
        # imposing rendered-gate semantics on every checkpoint kind.
        minimum_alignment = 0.90 if all(gate.kind == "gate" for gate in candidate.gates) else 0.50
        if line_alignment < minimum_alignment:
            reasons.append("racing-line-direction")
        line_length = float(line.length)
    except Exception as error:  # surfaced in the immutable audit record
        line_alignment = float("nan")
        line_length = float("nan")
        if str(error) != "cheap-audit-rejected":
            reasons.append(f"racing-line:{type(error).__name__}")
    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "reference": reference.name,
        "track": candidate.name,
        "deformation_rms_m": deformation,
        "minimum_spacing_ratio": float(np.min(ratio)),
        "maximum_spacing_ratio": float(np.max(ratio)),
        "mean_turn_error_degrees": float(np.mean(turn_error)),
        "maximum_turn_error_degrees": float(np.max(turn_error)),
        "mean_route_context_shift": float(np.mean(context_shift)),
        "maximum_route_context_shift": float(np.max(context_shift)),
        "minimum_normal_alignment": float(np.min(normal_alignment)),
        "minimum_line_gate_alignment": line_alignment,
        "racing_line_length_m": line_length,
        "geometry_fingerprint": geometry_fingerprint(candidate),
    }


def generate_named_course_augmentations(
    output: str | Path, *, parents: Sequence[Mapping[str, Any]],
    train_per_parent: int, validation_per_parent: int, seed: int,
    config: NamedCourseAugmentationConfig | None = None,
) -> Path:
    cfg = config or NamedCourseAugmentationConfig()
    root = Path(output)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parent_index, parent in enumerate(parents):
        reference_path = Path(str(parent["path"]))
        reference = load_track(reference_path)
        seen.add(geometry_fingerprint(reference))
        for split, count, split_offset in (
            ("train", train_per_parent, 0),
            ("family_validation", validation_per_parent, 10_000_000),
        ):
            for slot in range(count):
                accepted: tuple[Track, dict[str, Any], int] | None = None
                for attempt in range(cfg.generation_attempts):
                    track_seed = int(
                        seed + parent_index * 1_000_003 + split_offset
                        + slot * 1009 + attempt * 104729
                    )
                    candidate = augment_named_course(
                        reference, seed=track_seed,
                        name=f"named_aug_{split}_{parent['key']}_{slot:03d}_{track_seed}",
                        split=split, config=cfg,
                    )
                    audit = audit_named_course_augmentation(reference, candidate, config=cfg)
                    fingerprint = geometry_fingerprint(candidate)
                    if audit["valid"] and fingerprint not in seen:
                        accepted = candidate, audit, track_seed
                        break
                if accepted is None:
                    raise RuntimeError(f"unable to augment {reference.name} split={split} slot={slot}")
                candidate, audit, track_seed = accepted
                relative = Path("tracks") / split / f"{candidate.name}.yaml"
                save_track_yaml(candidate, root / relative)
                artifact = load_track(root / relative)
                fingerprint = geometry_fingerprint(artifact)
                seen.add(fingerprint)
                records.append({
                    "name": artifact.name,
                    "family": str(parent["key"]),
                    "reference_name": reference.name,
                    "split": split,
                    "seed": track_seed,
                    "path": str(relative),
                    "geometry_fingerprint": fingerprint,
                    "track_fingerprint": artifact.fingerprint,
                    "qualified_speed_mps": float(parent["qualified_speed_mps"]),
                    "static_audit": audit,
                    "dynamic_qualification": None,
                })
    payload = {
        "schema": "starscream-named-course-augmentation-manifest-v1",
        "generator": "named-course-smooth-augmentation-v1",
        "seed": int(seed),
        "config": asdict(cfg),
        "scientific_scope": "local-family-regularization-not-new-topology",
        "parents": [dict(parent) for parent in parents],
        "records": records,
    }
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "manifest.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
