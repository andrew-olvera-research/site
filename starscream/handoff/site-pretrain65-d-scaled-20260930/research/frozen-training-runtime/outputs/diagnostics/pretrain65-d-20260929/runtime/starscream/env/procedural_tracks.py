"""Deterministic, auditable procedural drone-racing track generation.

The generator deliberately materializes every course.  A manifest is the
contract between data collection, BC, DAgger, evaluation, and later PPO; no
training process is allowed to silently regenerate a different course from the
same human-readable experiment configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .tracks import Gate, Track, forward_up_quaternion


FAMILY_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("oval_sprint", 0.15),
    ("slalom", 0.20),
    ("figure_eight", 0.15),
    ("split_s", 0.20),
    ("vertical_wave", 0.15),
    ("mixed_3d", 0.15),
)


@dataclass(frozen=True, slots=True)
class ProceduralTrackConfig:
    """Course-distribution and static acceptance contract."""

    minimum_length_m: float = 64.0
    maximum_length_m: float = 88.0
    minimum_gates: int = 7
    maximum_gates: int = 10
    minimum_gate_spacing_m: float = 3.8
    maximum_gate_spacing_m: float = 17.0
    minimum_nonadjacent_spacing_m: float = 2.6
    maximum_curvature_m_inv: float = 1.50
    maximum_p99_curvature_m_inv: float = 0.85
    minimum_alignment: float = 0.30
    gate_width_range_m: tuple[float, float] = (2.45, 2.90)
    gate_height_range_m: tuple[float, float] = (2.45, 2.90)
    bounds: tuple[tuple[float, float], ...] = (
        (-18.0, 18.0), (-18.0, 18.0), (0.0, 9.0),
    )
    generation_attempts: int = 80
    planner_offset_iterations: int = 12
    planner_samples: int = 1200

    def __post_init__(self) -> None:
        if (
            self.minimum_length_m <= 0
            or self.maximum_length_m <= self.minimum_length_m
            or self.minimum_gates < 5
            or self.maximum_gates < self.minimum_gates
            or self.minimum_gate_spacing_m <= 0
            or self.maximum_gate_spacing_m <= self.minimum_gate_spacing_m
            or self.minimum_nonadjacent_spacing_m <= 0
            or self.maximum_curvature_m_inv <= 0
            or self.maximum_p99_curvature_m_inv <= 0
            or not 0 < self.minimum_alignment < 1
            or self.generation_attempts < 1
        ):
            raise ValueError("invalid procedural track configuration")
        bounds = np.asarray(self.bounds, np.float64)
        if bounds.shape != (3, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("procedural bounds must have shape (3,2)")


def geometry_fingerprint(track: Track) -> str:
    """Hash geometry without the human-readable name.

    ``Track.fingerprint`` intentionally includes the name.  Dataset split
    leakage needs the stronger invariant that renamed identical geometry is
    still detected.
    """

    digest = hashlib.sha256()
    digest.update(np.asarray(track.bounds, np.float32).tobytes())
    digest.update(bytes([int(track.loop)]))
    for gate in track.gates:
        digest.update(gate.position.tobytes())
        digest.update(gate.quaternion_wxyz.tobytes())
        digest.update(gate.size.tobytes())
        digest.update(bytes([int(gate.enter_from_opposite_side)]))
    return digest.hexdigest()


def _curve(family: str, theta: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized XY, metric Z offset, and gate roll profiles."""

    if family == "oval_sprint":
        ratio = rng.uniform(0.55, 0.76)
        radial = 1.0 + rng.uniform(0.04, 0.10) * np.cos(2.0 * theta + rng.uniform(-np.pi, np.pi))
        xy = np.stack([radial * np.cos(theta), ratio * radial * np.sin(theta)], axis=-1)
        z = rng.uniform(0.15, 0.45) * np.sin(2.0 * theta + rng.uniform(-np.pi, np.pi))
        roll = 0.12 * np.sin(theta)
    elif family == "slalom":
        ratio = rng.uniform(0.62, 0.82)
        radial = 1.0 + rng.uniform(0.13, 0.22) * np.sin(3.0 * theta + rng.uniform(-np.pi, np.pi))
        xy = np.stack([radial * np.cos(theta), ratio * radial * np.sin(theta)], axis=-1)
        z = rng.uniform(0.35, 0.80) * np.sin(2.0 * theta + rng.uniform(-np.pi, np.pi))
        roll = 0.20 * np.sin(2.0 * theta)
    elif family == "figure_eight":
        xy = np.stack([np.sin(theta), rng.uniform(0.64, 0.82) * np.sin(2.0 * theta)], axis=-1)
        # Separate the two crossing branches physically while retaining the
        # characteristic projected figure-eight topology.
        z = rng.uniform(1.40, 1.85) * np.cos(theta)
        roll = 0.22 * np.sin(2.0 * theta)
    elif family == "split_s":
        ratio = rng.uniform(0.62, 0.78)
        xy = np.stack([
            np.cos(theta) + rng.uniform(0.08, 0.16) * np.cos(2.0 * theta),
            ratio * np.sin(theta),
        ], axis=-1)
        z = rng.uniform(1.65, 2.20) * np.sin(theta - 0.30)
        # A smooth half-roll reaches an inverted high gate and unwinds over the
        # descending return.  This is an orientation demand, not an actor ID.
        roll = np.pi * 0.5 * (1.0 - np.cos(theta))
    elif family == "vertical_wave":
        ratio = rng.uniform(0.64, 0.82)
        xy = np.stack([np.cos(theta), ratio * np.sin(theta)], axis=-1)
        z = rng.uniform(1.45, 2.10) * np.sin(2.0 * theta + rng.uniform(-0.5, 0.5))
        roll = rng.uniform(0.25, 0.50) * np.sin(2.0 * theta)
    elif family == "mixed_3d":
        ratio = rng.uniform(0.60, 0.82)
        phase = rng.uniform(-np.pi, np.pi, size=3)
        xy = np.stack([
            np.cos(theta) + rng.uniform(0.08, 0.16) * np.cos(3.0 * theta + phase[0]),
            ratio * np.sin(theta) + rng.uniform(0.07, 0.14) * np.sin(2.0 * theta + phase[1]),
        ], axis=-1)
        z = (
            rng.uniform(0.70, 1.15) * np.sin(theta + phase[2])
            + rng.uniform(0.35, 0.70) * np.sin(3.0 * theta - phase[1])
        )
        roll = rng.uniform(0.25, 0.60) * np.sin(2.0 * theta + phase[0])
    else:
        raise ValueError(f"unknown procedural track family {family!r}")
    return xy.astype(np.float64), z.astype(np.float64), roll.astype(np.float64)


def _closed_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def _equal_arc_samples(
    xy: np.ndarray, z: np.ndarray, roll: np.ndarray, theta: np.ndarray, count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.column_stack([xy, z])
    closed = np.vstack([points, points[0]])
    distance = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distance)])
    query = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    theta_closed = np.concatenate([theta, [2.0 * np.pi]])
    roll_closed = np.concatenate([roll, [roll[0]]])
    sampled_theta = np.interp(query, cumulative, theta_closed)
    sampled_roll = np.interp(query, cumulative, roll_closed)
    sampled = np.column_stack([
        np.interp(sampled_theta, theta_closed, np.r_[xy[:, 0], xy[0, 0]]),
        np.interp(sampled_theta, theta_closed, np.r_[xy[:, 1], xy[0, 1]]),
        np.interp(sampled_theta, theta_closed, np.r_[z, z[0]]),
    ])
    return sampled, sampled_roll, sampled_theta


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * float(axis @ vector) * (1.0 - np.cos(angle))
    )


def generate_track(
    family: str,
    *,
    seed: int,
    name: str,
    split: str,
    config: ProceduralTrackConfig | None = None,
) -> Track:
    """Generate one deterministic course before static acceptance filtering."""

    cfg = config or ProceduralTrackConfig()
    rng = np.random.default_rng(int(seed))
    gate_count = int(rng.integers(cfg.minimum_gates, cfg.maximum_gates + 1))
    target_length = float(rng.uniform(cfg.minimum_length_m, cfg.maximum_length_m))
    dense_theta = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    xy_unit, z_dense, roll_dense = _curve(family, dense_theta, rng)

    # Solve a horizontal scale that gives the requested metric course length
    # while preserving the family-specific vertical excursion.
    low, high = 4.0, 20.0
    for _ in range(36):
        scale = 0.5 * (low + high)
        length = _closed_length(np.column_stack([scale * xy_unit, z_dense]))
        if length < target_length:
            low = scale
        else:
            high = scale
    horizontal_scale = 0.5 * (low + high)
    yaw = float(rng.uniform(-np.pi, np.pi))
    yaw_rotation = np.asarray([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)],
    ])
    xy_dense = horizontal_scale * xy_unit @ yaw_rotation.T
    center_z = float(rng.uniform(4.25, 4.75))
    sampled, sampled_roll, _ = _equal_arc_samples(
        xy_dense, center_z + z_dense, roll_dense, dense_theta, gate_count
    )
    translation = rng.uniform(-0.8, 0.8, size=2)
    sampled[:, :2] += translation

    gate_width = float(rng.uniform(*cfg.gate_width_range_m))
    gate_height = float(rng.uniform(*cfg.gate_height_range_m))
    gates: list[Gate] = []
    for index, center in enumerate(sampled):
        incoming = center - sampled[(index - 1) % gate_count]
        outgoing = sampled[(index + 1) % gate_count] - center
        incoming /= max(float(np.linalg.norm(incoming)), 1.0e-12)
        outgoing /= max(float(np.linalg.norm(outgoing)), 1.0e-12)
        tangent = incoming + outgoing
        tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
        up = np.asarray([0.0, 0.0, 1.0])
        up -= tangent * float(up @ tangent)
        if np.linalg.norm(up) < 1.0e-5:
            up = np.asarray([0.0, 1.0, 0.0])
            up -= tangent * float(up @ tangent)
        up /= np.linalg.norm(up)
        up = _rotate_about_axis(up, tangent, float(sampled_roll[index]))
        size_jitter = rng.uniform(0.96, 1.04, size=2)
        gates.append(Gate(
            position=center.astype(np.float32),
            quaternion_wxyz=forward_up_quaternion(tangent, up),
            size=(np.asarray([gate_width, gate_height]) * size_jitter).astype(np.float32),
            name=f"gate_{index:02d}",
        ))
    return Track(
        name=name,
        gates=tuple(gates),
        bounds=np.asarray(cfg.bounds, np.float32),
        loop=True,
        metadata={
            "source": "procedural-v1",
            "family": family,
            "split": str(split),
            "seed": int(seed),
            "target_length_m": target_length,
            "maneuvers": {
                "vertical": int(family in {"figure_eight", "split_s", "vertical_wave", "mixed_3d"}),
                "split_s": int(family == "split_s"),
                "crossing": int(family == "figure_eight"),
            },
        },
    )


def audit_procedural_track(
    track: Track,
    *,
    config: ProceduralTrackConfig | None = None,
    cache_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run static geometry, racing-line, and family-invariant checks."""

    from ..mpcc import RacingLinePlanner, RacingLinePlannerConfig

    cfg = config or ProceduralTrackConfig()
    family = str((track.metadata or {}).get("family", "unknown"))
    geometry = track.geometry_report(minimum_alignment=cfg.minimum_alignment)
    centers = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    segments = np.linalg.norm(np.roll(centers, -1, axis=0) - centers, axis=1)
    nonadjacent: list[float] = []
    for left in range(len(centers)):
        for right in range(left + 1, len(centers)):
            distance = (right - left) % len(centers)
            if distance in {1, len(centers) - 1}:
                continue
            nonadjacent.append(float(np.linalg.norm(centers[left] - centers[right])))
    line = RacingLinePlanner(RacingLinePlannerConfig(
        sample_count=cfg.planner_samples,
        offset_iterations=cfg.planner_offset_iterations,
        cache_directory=str(cache_directory) if cache_directory is not None else None,
    )).plan(track)
    query = np.linspace(0.0, line.length, 2400, endpoint=False)
    curvature = np.asarray(line.evaluate(query)["curvature"], np.float64)
    normals = np.stack([gate.normal for gate in track.gates]).astype(np.float64)
    turn_angles = np.degrees(np.arccos(np.clip(
        np.sum(normals * np.roll(normals, 1, axis=0), axis=1), -1.0, 1.0
    )))
    ups = np.stack([gate.up for gate in track.gates]).astype(np.float64)
    roll_span = float(np.degrees(np.max(np.arccos(np.clip(ups @ ups[0], -1.0, 1.0)))))
    vertical_excursion = float(np.ptp(centers[:, 2]))
    reasons: list[str] = []
    if not geometry["feasible"]:
        reasons.append("gate-geometry")
    if not cfg.minimum_length_m <= line.length <= cfg.maximum_length_m:
        reasons.append("course-length")
    if float(segments.min()) < cfg.minimum_gate_spacing_m:
        reasons.append("gate-spacing-minimum")
    if float(segments.max()) > cfg.maximum_gate_spacing_m:
        reasons.append("gate-spacing-maximum")
    if nonadjacent and min(nonadjacent) < cfg.minimum_nonadjacent_spacing_m:
        reasons.append("nonadjacent-gate-overlap")
    if float(curvature.max()) > cfg.maximum_curvature_m_inv:
        reasons.append("peak-curvature")
    if float(np.quantile(curvature, 0.99)) > cfg.maximum_p99_curvature_m_inv:
        reasons.append("p99-curvature")
    if family in {"figure_eight", "split_s", "vertical_wave"} and vertical_excursion < 2.4:
        reasons.append("missing-vertical-excursion")
    if family == "split_s" and roll_span < 135.0:
        reasons.append("missing-split-s-roll")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "track": track.name,
        "family": family,
        "track_fingerprint": track.fingerprint,
        "geometry_fingerprint": geometry_fingerprint(track),
        "gate_count": len(track.gates),
        "racing_line_length_m": float(line.length),
        "minimum_gate_spacing_m": float(segments.min()),
        "maximum_gate_spacing_m": float(segments.max()),
        "minimum_nonadjacent_spacing_m": min(nonadjacent) if nonadjacent else float("inf"),
        "vertical_excursion_m": vertical_excursion,
        "mean_turn_degrees": float(turn_angles.mean()),
        "maximum_turn_degrees": float(turn_angles.max()),
        "gate_up_span_degrees": roll_span,
        "maximum_curvature_m_inv": float(curvature.max()),
        "p99_curvature_m_inv": float(np.quantile(curvature, 0.99)),
        "geometry": geometry,
    }


def track_to_mapping(track: Track) -> dict[str, Any]:
    metadata = dict(track.metadata or {})
    return {
        "name": track.name,
        "loop": bool(track.loop),
        "bounds": np.asarray(track.bounds).tolist(),
        "metadata": metadata,
        "gates": [
            {
                "name": gate.name,
                "position": gate.position.tolist(),
                "quaternion_wxyz": gate.quaternion_wxyz.tolist(),
                "size": gate.size.tolist(),
                "enter_from_opposite_side": bool(gate.enter_from_opposite_side),
                "kind": gate.kind,
                "render": bool(gate.render),
            }
            for gate in track.gates
        ],
    }


def save_track_yaml(track: Track, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(track_to_mapping(track), stream, sort_keys=False)
    temporary.replace(destination)
    return destination


def _family_schedule(count: int) -> list[str]:
    raw = np.asarray([count * weight for _, weight in FAMILY_WEIGHTS], np.float64)
    quotas = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - quotas))[: count - int(quotas.sum())]:
        quotas[index] += 1
    return [
        family
        for (family, _), quota in zip(FAMILY_WEIGHTS, quotas)
        for _ in range(int(quota))
    ]


def generate_manifest(
    output: str | Path,
    *,
    train_count: int,
    validation_count: int,
    seed: int,
    config: ProceduralTrackConfig | None = None,
) -> Path:
    """Generate disjoint accepted train/validation track manifests."""

    cfg = config or ProceduralTrackConfig()
    if train_count < 1 or validation_count < 1:
        raise ValueError("manifest needs non-empty train and validation splits")
    root = Path(output)
    track_root = root / "tracks"
    audit_cache = root / "racing-lines"
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen_geometry: set[str] = set()
    for split, count, split_offset in (
        ("train", train_count, 0), ("validation", validation_count, 10_000_000),
    ):
        families = _family_schedule(count)
        np.random.default_rng(seed + split_offset).shuffle(families)
        for index, family in enumerate(families):
            accepted: tuple[Track, dict[str, Any], int] | None = None
            failures: list[str] = []
            for attempt in range(cfg.generation_attempts):
                track_seed = int(seed + split_offset + index * 1009 + attempt * 104729)
                name = f"proc_{split}_{index:03d}_{family}_{track_seed}"
                track = generate_track(
                    family, seed=track_seed, name=name, split=split, config=cfg
                )
                try:
                    audit = audit_procedural_track(
                        track, config=cfg, cache_directory=audit_cache
                    )
                except (ValueError, FloatingPointError) as error:
                    failures.append(type(error).__name__)
                    continue
                fingerprint = str(audit["geometry_fingerprint"])
                if audit["valid"] and fingerprint not in seen_geometry:
                    accepted = track, audit, track_seed
                    break
                failures.extend(str(item) for item in audit["reasons"])
            if accepted is None:
                raise RuntimeError(
                    f"could not generate accepted {split} {family} track {index}; "
                    f"recent failures={failures[-12:]}"
                )
            track, audit, track_seed = accepted
            seen_geometry.add(str(audit["geometry_fingerprint"]))
            track_path = save_track_yaml(track, track_root / split / f"{track.name}.yaml")
            records.append({
                "name": track.name,
                "path": str(track_path.relative_to(root)),
                "split": split,
                "family": family,
                "seed": track_seed,
                "track_fingerprint": track.fingerprint,
                "geometry_fingerprint": audit["geometry_fingerprint"],
                "static_audit": audit,
                "qualified_speed_mps": None,
                "qualification": None,
            })
    manifest = {
        "schema": "starscream-procedural-track-manifest-v1",
        "seed": int(seed),
        "config": asdict(cfg),
        "family_weights": dict(FAMILY_WEIGHTS),
        "records": records,
    }
    destination = root / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The balanced qualified manifest is a read-only projection of the
    # candidate manifest and deliberately preserves the record contract.  It
    # must pass through the same loader so audits and curricula consume the
    # exact tracks selected for training.
    supported_schemas = {
        "starscream-procedural-track-manifest-v1",
        "starscream-qualified-reference-training-manifest-v1",
        "starscream-dagger-qualified-reference-training-manifest-v1",
        "starscream-goal-conditioned-task-manifest-v1",
    }
    if payload.get("schema") not in supported_schemas:
        raise ValueError(f"unsupported procedural manifest {manifest_path}")
    records = payload.get("records", ())
    if not records:
        raise ValueError("procedural manifest contains no tracks")
    fingerprints = [str(item["geometry_fingerprint"]) for item in records]
    if payload.get("schema") == "starscream-goal-conditioned-task-manifest-v1":
        # Explicitly distinct termination tasks may share geometry; ordinary
        # course manifests retain the stricter geometry-deduplication rule.
        goals = [item.get("curriculum_completion_gates") for item in records]
        if any(not isinstance(goal, int) or isinstance(goal, bool) or goal < 1 for goal in goals):
            raise ValueError("goal-conditioned manifest requires positive integer goals")
        fingerprints = list(zip(fingerprints, goals))
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("procedural manifest contains geometry duplicates")
    return payload


def manifest_track_paths(
    path: str | Path,
    *,
    split: str | None = None,
    families: Sequence[str] | None = None,
    qualified_only: bool = False,
    dynamic_qualification_mode: str | None = None,
    minimum_qualified_speed: float | None = None,
    limit: int = 0,
) -> tuple[str, ...]:
    manifest_path = Path(path)
    payload = read_manifest(manifest_path)
    allowed = None if families is None else set(str(item) for item in families)

    mode = None if dynamic_qualification_mode is None else str(
        dynamic_qualification_mode
    ).strip().lower()
    if mode not in {None, "rl", "dagger"}:
        raise ValueError("dynamic qualification mode must be 'rl' or 'dagger'")

    def dynamically_admitted(item: Mapping[str, Any]) -> bool:
        if mode is None:
            return True
        admission = dict(item.get("mpcc_admission") or {})
        if admission.get("contract") == "starscream-mpcc-admission-v2":
            return bool(admission.get(
                "rl_eligible" if mode == "rl" else "dagger_eligible", False,
            ))
        if item.get("dynamic_qualification") is not None:
            # Local import avoids a module cycle while preserving one canonical
            # interpretation of legacy v6 all-start audit evidence.
            from .racing_manifold.expert_qualification import (
                evaluate_record_dynamic_qualification,
            )
            track = load_track(manifest_path.parent / str(item["path"]))
            expected = tuple(
                index for index, gate in enumerate(track.gates) if gate.render
            ) or tuple(range(len(track.gates)))
            decision = evaluate_record_dynamic_qualification(
                item, expected_start_indices=expected,
            )
            return bool(
                decision.rl_eligible if mode == "rl" else decision.dagger_eligible
            )
        # The racing-distribution qualification suite is already an all-start,
        # collision/clearance/solver checked MPCC contract.  It is stricter
        # than the minimum RL gate and acceptable for either mode.
        suite = dict(item.get("qualification_suite") or {})
        return bool(suite.get("passed"))

    selected = [
        item for item in payload["records"]
        if (split is None or str(item["split"]) == split)
        and (allowed is None or str(item["family"]) in allowed)
        and (not qualified_only or item.get("qualified_speed_mps") is not None)
        and dynamically_admitted(item)
        and (
            minimum_qualified_speed is None
            or (
                item.get("qualified_speed_mps") is not None
                and float(item["qualified_speed_mps"]) >= minimum_qualified_speed
            )
        )
    ]
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        suffix = "" if mode is None else f" after {mode} MPCC admission"
        raise ValueError(
            f"manifest selection under {manifest_path} is empty{suffix}"
        )
    return tuple(str((manifest_path.parent / item["path"]).resolve()) for item in selected)


def update_manifest_records(
    path: str | Path, updates: Mapping[str, Mapping[str, Any]],
) -> None:
    """Atomically merge qualification/collection results by track name."""

    manifest_path = Path(path)
    payload = read_manifest(manifest_path)
    known = {str(item["name"]) for item in payload["records"]}
    unknown = set(updates) - known
    if unknown:
        raise ValueError(f"manifest updates contain unknown tracks: {sorted(unknown)}")
    for record in payload["records"]:
        patch = updates.get(str(record["name"]))
        if patch:
            record.update(dict(patch))
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path)


def audit_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = read_manifest(manifest_path)
    records = payload["records"]
    split_fingerprints = {
        split: {str(item["geometry_fingerprint"]) for item in records if item["split"] == split}
        for split in ("train", "validation")
    }
    families = tuple(payload.get("family_weights", dict(FAMILY_WEIGHTS)))
    family_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        family_counts[split] = {
            family: sum(
                item["split"] == split and item["family"] == family for item in records
            )
            for family in families
        }
    static_valid = all(bool(item["static_audit"]["valid"]) for item in records)
    artifact_errors: list[str] = []
    for item in records:
        track_path = manifest_path.parent / str(item["path"])
        try:
            from .tracks import load_track
            track = load_track(track_path)
            if track.fingerprint != item["track_fingerprint"]:
                artifact_errors.append(f"{item['name']}: track fingerprint mismatch")
            if geometry_fingerprint(track) != item["geometry_fingerprint"]:
                artifact_errors.append(f"{item['name']}: geometry fingerprint mismatch")
            if track.name != item["name"]:
                artifact_errors.append(f"{item['name']}: YAML name mismatch")
        except (FileNotFoundError, KeyError, ValueError) as error:
            artifact_errors.append(f"{item['name']}: {type(error).__name__}: {error}")
    family_statistics: dict[str, dict[str, float]] = {}
    for family in families:
        selected = [item["static_audit"] for item in records if item["family"] == family]
        family_statistics[family] = {
            "tracks": len(selected),
            "mean_length_m": float(np.mean([item["racing_line_length_m"] for item in selected])),
            "mean_gates": float(np.mean([item["gate_count"] for item in selected])),
            "mean_vertical_excursion_m": float(np.mean([
                item["vertical_excursion_m"] for item in selected
            ])),
            "mean_turn_degrees": float(np.mean([item["mean_turn_degrees"] for item in selected])),
            "maximum_p99_curvature_m_inv": max(
                float(item["p99_curvature_m_inv"]) for item in selected
            ),
        }
    return {
        "passed": bool(
            static_valid
            and not artifact_errors
            and not (split_fingerprints["train"] & split_fingerprints["validation"])
            and all(all(count > 0 for count in counts.values()) for counts in family_counts.values())
        ),
        "static_valid": static_valid,
        "train_validation_geometry_overlap": len(
            split_fingerprints["train"] & split_fingerprints["validation"]
        ),
        "record_count": len(records),
        "artifact_errors": artifact_errors,
        "family_counts": family_counts,
        "family_statistics": family_statistics,
        "length_m": {
            "minimum": min(float(item["static_audit"]["racing_line_length_m"]) for item in records),
            "mean": float(np.mean([
                float(item["static_audit"]["racing_line_length_m"]) for item in records
            ])),
            "maximum": max(float(item["static_audit"]["racing_line_length_m"]) for item in records),
        },
    }


def write_manifest_gallery(path: str | Path, output: str | Path) -> Path:
    """Render a compact family/vertical-variation audit artifact."""

    import matplotlib.pyplot as plt

    from .tracks import load_track

    manifest_path = Path(path)
    payload = read_manifest(manifest_path)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    families = tuple(payload.get("family_weights", dict(FAMILY_WEIGHTS)))
    columns = 2
    rows = int(np.ceil(len(families) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 5 * rows), constrained_layout=True)
    axes_flat = np.asarray(axes, dtype=object).reshape(-1)
    for axis, family in zip(axes_flat, families):
        selected = [item for item in payload["records"] if item["family"] == family]
        for record in selected:
            track = load_track(manifest_path.parent / record["path"])
            centers = np.stack([gate.position for gate in track.gates])
            closed = np.vstack([centers, centers[0]])
            color = "#2563eb" if record["split"] == "train" else "#dc2626"
            axis.plot(closed[:, 0], closed[:, 1], color=color, alpha=0.32, linewidth=1.2)
            axis.scatter(
                centers[:, 0], centers[:, 1], c=centers[:, 2],
                cmap="viridis", vmin=0.0, vmax=9.0, s=13, alpha=0.75,
            )
        axis.set_title(f"{family} ({len(selected)} tracks)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-18, 18); axis.set_ylim(-18, 18)
        axis.grid(alpha=0.18)
        axis.set_xlabel("world x [m]"); axis.set_ylabel("world y [m]")
    figure.suptitle(
        "Procedural racing distribution — train blue, held-out red, gate color = height",
        fontsize=14,
    )
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
