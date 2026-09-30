"""Maneuver-composed procedural racing courses with semantic validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .procedural_tracks import (
    ProceduralTrackConfig, audit_procedural_track, geometry_fingerprint,
    save_track_yaml,
)
from .tracks import Gate, Track, forward_up_quaternion, load_track


PRIMITIVE_FAMILY_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("split_s_focus", 0.22),
    ("vertical_technical", 0.20),
    ("chicane_hairpin", 0.20),
    ("corkscrew_technical", 0.18),
    ("championship_mixed", 0.20),
)


@dataclass(frozen=True, slots=True)
class PrimitiveTrackConfig(ProceduralTrackConfig):
    minimum_length_m: float = 82.0
    maximum_length_m: float = 112.0
    minimum_gates: int = 16
    maximum_gates: int = 16
    minimum_gate_spacing_m: float = 2.45
    maximum_gate_spacing_m: float = 14.0
    minimum_nonadjacent_spacing_m: float = 2.35
    maximum_curvature_m_inv: float = 1.8
    maximum_p99_curvature_m_inv: float = 1.05
    minimum_alignment: float = 0.20
    gate_width_range_m: tuple[float, float] = (2.05, 2.40)
    gate_height_range_m: tuple[float, float] = (1.80, 2.20)
    bounds: tuple[tuple[float, float], ...] = (
        (-18.0, 18.0), (-18.0, 18.0), (0.0, 10.5),
    )
    generation_attempts: int = 180


def _base_course(family: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return a closed championship skeleton assembled from named primitives.

    Gates 6--9 are a geometric Split-S in the vertical XZ plane: the vehicle
    enters westbound at altitude, follows a descending half-loop, and exits
    eastbound below it. This is path topology, not merely rolled gate art.
    """

    points = np.asarray([
        [-12.0, -7.0, 3.1], [-5.2, -7.2, 3.2], [0.0, -3.4, 3.8],
        [5.0, -8.6, 4.2], [11.1, -5.3, 4.8], [14.0, 7.0, 7.4],
        [9.0, 7.0, 7.4], [5.3, 7.0, 6.3], [5.3, 7.0, 3.7],
        [9.0, 7.0, 2.6], [14.0, 7.0, 2.6], [13.0, 0.5, 3.0],
        [8.0, -1.2, 6.4], [2.5, 1.8, 2.5], [-4.2, 1.0, 5.6],
        [-10.5, -1.8, 3.0],
    ], np.float64)
    labels = [
        "sprint", "sprint", "chicane", "chicane", "hairpin", "split_s_setup",
        "split_s_entry", "split_s_apex", "split_s_descent", "split_s_exit",
        "hairpin", "dive", "vertical_ladder", "corkscrew", "corkscrew", "slalom",
    ]
    roll = np.zeros(len(points), np.float64)
    roll[6:10] = [0.0, np.pi * 0.5, np.pi, np.pi]
    roll[10] = np.pi * 0.5
    roll[12:15] = [0.35, -0.75, 0.65]

    if family == "split_s_focus":
        points[7:9, 0] -= rng.uniform(0.35, 0.85)
        points[5:7, 2] += rng.uniform(0.15, 0.45)
        points[9:11, 2] -= rng.uniform(0.10, 0.30)
    elif family == "vertical_technical":
        points[12, 2] += rng.uniform(0.45, 0.85)
        points[13, 2] -= rng.uniform(0.20, 0.45)
        points[14, 2] += rng.uniform(0.35, 0.70)
    elif family == "chicane_hairpin":
        points[2, 1] += rng.uniform(0.5, 1.0)
        points[3, 1] -= rng.uniform(0.4, 0.8)
        points[11, 0] += rng.uniform(0.4, 0.8)
        points[12, 1] -= rng.uniform(0.4, 0.8)
    elif family == "corkscrew_technical":
        points[12:15, 1] += np.asarray([-0.7, 0.9, -0.5]) * rng.uniform(0.7, 1.1)
        points[12:15, 2] += np.asarray([0.6, -0.4, 0.7]) * rng.uniform(0.6, 1.0)
        roll[12:15] *= 1.35
    elif family == "championship_mixed":
        points[2:5, 1] += rng.uniform(-0.55, 0.55, size=3)
        points[12:15, 2] += rng.uniform(-0.35, 0.45, size=3)
        points[7:9, 0] -= rng.uniform(0.15, 0.55)
    else:
        raise ValueError(f"unknown primitive course family {family!r}")
    return points, roll, labels


def generate_primitive_track(
    family: str, *, seed: int, name: str, split: str,
    config: PrimitiveTrackConfig | None = None,
) -> Track:
    cfg = config or PrimitiveTrackConfig()
    rng = np.random.default_rng(int(seed))
    points, rolls, labels = _base_course(family, rng)
    scale = float(rng.uniform(0.93, 1.04))
    points[:, :2] *= scale
    # Small gate-local perturbations preserve maneuver topology while making
    # train/validation geometry genuinely disjoint.
    jitter = rng.normal(0.0, [0.18, 0.18, 0.10], size=points.shape)
    jitter[6:10, 1] *= 0.18  # keep the Split-S in one clear vertical plane
    points += jitter
    yaw = float(rng.uniform(-np.pi, np.pi))
    rotation = np.asarray([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    points[:, :2] = points[:, :2] @ rotation.T
    points[:, :2] += rng.uniform(-0.55, 0.55, size=2)

    width = float(rng.uniform(*cfg.gate_width_range_m))
    height = float(rng.uniform(*cfg.gate_height_range_m))
    gates: list[Gate] = []
    for index, center in enumerate(points):
        incoming = center - points[(index - 1) % len(points)]
        outgoing = points[(index + 1) % len(points)] - center
        incoming /= max(float(np.linalg.norm(incoming)), 1e-12)
        outgoing /= max(float(np.linalg.norm(outgoing)), 1e-12)
        tangent = incoming + outgoing
        if np.linalg.norm(tangent) < 1e-5:
            tangent = outgoing
        tangent /= np.linalg.norm(tangent)
        up = np.asarray([0.0, 0.0, 1.0])
        up -= tangent * float(up @ tangent)
        if np.linalg.norm(up) < 1e-5:
            up = np.asarray([0.0, 1.0, 0.0])
            up -= tangent * float(up @ tangent)
        up /= np.linalg.norm(up)
        angle = float(rolls[index])
        up = (
            up * np.cos(angle) + np.cross(tangent, up) * np.sin(angle)
            + tangent * float(tangent @ up) * (1.0 - np.cos(angle))
        )
        size = np.asarray([width, height]) * rng.uniform(0.96, 1.04, size=2)
        gates.append(Gate(
            position=center.astype(np.float32),
            quaternion_wxyz=forward_up_quaternion(tangent, up),
            size=size.astype(np.float32), name=f"gate_{index:02d}_{labels[index]}",
        ))
    return Track(
        name=name, gates=tuple(gates), bounds=np.asarray(cfg.bounds, np.float32), loop=True,
        metadata={
            "source": "maneuver-primitives-v2", "family": family, "split": split,
            "seed": int(seed), "primitive_labels": labels,
            "primitive_spans": {
                "chicane": [1, 3], "hairpin": [3, 5], "split_s": [5, 10],
                "vertical_corkscrew": [11, 15], "slalom_return": [14, 1],
            },
        },
    )


def audit_primitive_track(
    track: Track, *, config: PrimitiveTrackConfig | None = None,
    cache_directory: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or PrimitiveTrackConfig()
    base = audit_procedural_track(track, config=cfg, cache_directory=cache_directory)
    centers = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    segments = np.roll(centers, -1, axis=0) - centers
    unit = segments / np.maximum(np.linalg.norm(segments, axis=1, keepdims=True), 1e-9)
    turn = np.degrees(np.arccos(np.clip(np.sum(unit * np.roll(unit, 1, axis=0), axis=1), -1, 1)))
    slope = np.abs(segments[:, 2]) / np.maximum(np.linalg.norm(segments, axis=1), 1e-9)
    split_entry = unit[5]
    split_exit = unit[9]
    split_heading_reversal = float(np.degrees(np.arccos(np.clip(split_entry @ split_exit, -1, 1))))
    split_descent = float(centers[6, 2] - centers[9, 2])
    split_axis = split_entry / max(np.linalg.norm(split_entry), 1e-9)
    split_chord = centers[9] - centers[6]
    apex_displacement = max(
        float(np.linalg.norm((centers[i] - centers[6]) - split_axis * ((centers[i] - centers[6]) @ split_axis)))
        for i in (7, 8)
    )
    hard_transitions = int(np.sum((turn >= 48.0) | (slope >= 0.28)))
    vertical_excursion = float(np.ptp(centers[:, 2]))
    reasons = list(base["reasons"])
    checks = {
        "split_s_heading_reversal_degrees": split_heading_reversal,
        "split_s_descent_m": split_descent,
        "split_s_apex_displacement_m": apex_displacement,
        "hard_transition_count": hard_transitions,
        "maximum_segment_turn_degrees": float(turn.max()),
        "vertical_excursion_m": vertical_excursion,
        "maximum_gate_width_m": float(max(g.size[0] for g in track.gates)),
        "maximum_gate_height_m": float(max(g.size[1] for g in track.gates)),
    }
    if split_heading_reversal < 145.0:
        reasons.append("weak-split-s-heading-reversal")
    if split_descent < 3.8:
        reasons.append("weak-split-s-descent")
    if apex_displacement < 3.0:
        reasons.append("weak-split-s-apex")
    if hard_transitions < 7:
        reasons.append("insufficient-hard-transitions")
    if vertical_excursion < 4.5:
        reasons.append("insufficient-vertical-range")
    if float(turn.max()) < 70.0:
        reasons.append("insufficient-peak-turn")
    return {**base, **checks, "valid": not reasons, "reasons": sorted(set(reasons))}


def _schedule(count: int) -> list[str]:
    raw = np.asarray([count * weight for _, weight in PRIMITIVE_FAMILY_WEIGHTS])
    quota = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - quota))[:count - int(quota.sum())]:
        quota[index] += 1
    return [family for (family, _), n in zip(PRIMITIVE_FAMILY_WEIGHTS, quota) for _ in range(n)]


def generate_primitive_manifest(
    output: str | Path, *, train_count: int, validation_count: int, seed: int,
    config: PrimitiveTrackConfig | None = None,
) -> Path:
    cfg = config or PrimitiveTrackConfig()
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split, count, offset in (("train", train_count, 0), ("validation", validation_count, 10_000_000)):
        families = _schedule(count)
        np.random.default_rng(seed + offset).shuffle(families)
        for index, family in enumerate(families):
            accepted = None
            failures: list[str] = []
            for attempt in range(cfg.generation_attempts):
                track_seed = int(seed + offset + index * 1009 + attempt * 104729)
                name = f"primitive_{split}_{index:03d}_{family}_{track_seed}"
                track = generate_primitive_track(family, seed=track_seed, name=name, split=split, config=cfg)
                fingerprint = geometry_fingerprint(track)
                audit = audit_primitive_track(track, config=cfg, cache_directory=root / "racing-lines")
                failures.extend(audit["reasons"])
                if audit["valid"] and fingerprint not in seen:
                    accepted = track, audit, track_seed, fingerprint
                    break
            if accepted is None:
                raise RuntimeError(f"could not generate {split} {family}: {sorted(set(failures))}")
            track, audit, track_seed, fingerprint = accepted
            relative = Path("tracks") / split / f"{track.name}.yaml"
            save_track_yaml(track, root / relative)
            artifact = load_track(root / relative)
            artifact_geometry = geometry_fingerprint(artifact)
            seen.add(artifact_geometry)
            records.append({
                "name": track.name, "family": family, "split": split, "seed": track_seed,
                "path": str(relative), "track_fingerprint": artifact.fingerprint,
                "geometry_fingerprint": artifact_geometry, "static_audit": audit,
                "primitive_labels": list((track.metadata or {})["primitive_labels"]),
                "qualified_speed_mps": None, "qualification": None,
            })
    payload = {
        "schema": "starscream-procedural-track-manifest-v1",
        "generator": "maneuver-primitives-v2", "seed": int(seed),
        "config": asdict(cfg),
        "family_weights": dict(PRIMITIVE_FAMILY_WEIGHTS),
        "records": records,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def repair_unqualified_primitive_manifest(
    path: str | Path, *, maximum_attempts: int = 180,
) -> dict[str, Any]:
    """Replace dynamically rejected slots without touching accepted courses.

    Rejected records remain in ``rejected_records`` and their YAML artifacts are
    retained. Replacement seeds are deterministic functions of the old seed,
    repair generation, and attempt, making every admission decision auditable.
    """

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("generator") != "maneuver-primitives-v2":
        raise ValueError("targeted repair requires a maneuver-primitives-v2 manifest")
    records = list(payload["records"])
    rejected_indices = [
        index for index, record in enumerate(records)
        if record.get("qualified_speed_mps") is None
        and record.get("qualification") is not None
    ]
    if not rejected_indices:
        return {"replaced": 0, "records": [], "manifest": str(manifest_path)}
    seen = {
        str(record["geometry_fingerprint"])
        for index, record in enumerate(records) if index not in rejected_indices
    }
    archive = list(payload.get("rejected_records", []))
    replacements: list[dict[str, Any]] = []
    cfg = PrimitiveTrackConfig()
    for record_index in rejected_indices:
        old = dict(records[record_index])
        history = list(old.get("repair_history", []))
        generation = len(history) + 1
        split = str(old["split"])
        family = str(old["family"])
        split_slot = sum(
            records[index]["split"] == split for index in range(record_index + 1)
        ) - 1
        accepted = None
        failures: list[str] = []
        for attempt in range(maximum_attempts):
            candidate_seed = int(old["seed"]) + generation * 1_000_003 + (attempt + 1) * 104_729
            name = f"primitive_{split}_{split_slot:03d}_{family}_{candidate_seed}"
            track = generate_primitive_track(
                family, seed=candidate_seed, name=name, split=split, config=cfg
            )
            audit = audit_primitive_track(
                track, config=cfg, cache_directory=manifest_path.parent / "racing-lines"
            )
            fingerprint = geometry_fingerprint(track)
            failures.extend(audit["reasons"])
            if audit["valid"] and fingerprint not in seen:
                accepted = track, audit, candidate_seed
                break
        if accepted is None:
            raise RuntimeError(
                f"could not statically repair {old['name']}; failures={sorted(set(failures))}"
            )
        track, audit, candidate_seed = accepted
        relative = Path("tracks") / split / f"{track.name}.yaml"
        save_track_yaml(track, manifest_path.parent / relative)
        artifact = load_track(manifest_path.parent / relative)
        artifact_geometry = geometry_fingerprint(artifact)
        seen.add(artifact_geometry)
        rejection = {
            **old,
            "rejected_utc": datetime.now(timezone.utc).isoformat(),
            "replaced_by": track.name,
        }
        archive.append(rejection)
        history.append({
            "name": old["name"], "seed": int(old["seed"]),
            "qualified_speed_mps": old.get("qualified_speed_mps"),
            "qualification": old.get("qualification"),
        })
        replacement = {
            "name": track.name, "family": family, "split": split,
            "seed": candidate_seed, "path": str(relative),
            "track_fingerprint": artifact.fingerprint,
            "geometry_fingerprint": artifact_geometry,
            "static_audit": audit,
            "primitive_labels": list((track.metadata or {})["primitive_labels"]),
            "qualified_speed_mps": None, "qualification": None,
            "repair_generation": generation, "repair_history": history,
        }
        records[record_index] = replacement
        replacements.append({
            "slot": f"{split}:{split_slot}", "family": family,
            "old": old["name"], "new": track.name,
            "new_seed": candidate_seed, "static_audit": audit,
        })
    payload["records"] = records
    payload["rejected_records"] = archive
    payload["last_repair_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path)
    return {
        "replaced": len(replacements), "records": replacements,
        "manifest": str(manifest_path),
    }
