"""Competition-calibrated procedural drone-racing courses.

Unlike the earlier primitive generator, this distribution samples complete
race formats.  Difficulty is concentrated into a few authentic local sectors
and separated by acceleration room.  Gate pose is independent of the chord
bisector, which is essential for stacked Split-S and hard-exit gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .procedural_tracks import geometry_fingerprint, save_track_yaml
from .tracks import Gate, Track, forward_up_quaternion, load_track


CHAMPIONSHIP_FORMAT_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("compact_split_s", 0.30),
    ("compact_hairpin", 0.20),
    ("ladder_sprint", 0.20),
    ("grand_prix", 0.30),
)

FORMAT_LINE_LENGTH_RANGES_M: dict[str, tuple[float, float]] = {
    "compact_split_s": (68.0, 86.0),
    "compact_hairpin": (78.0, 102.0),
    "ladder_sprint": (108.0, 142.0),
    "grand_prix": (150.0, 185.0),
}

FORMAT_HORIZONTAL_SCALE_RANGES: dict[str, tuple[float, float]] = {
    "compact_split_s": (0.78, 0.90),
    "compact_hairpin": (0.76, 0.88),
    "ladder_sprint": (0.88, 1.02),
    "grand_prix": (0.75, 0.87),
}


@dataclass(frozen=True, slots=True)
class ChampionshipTrackConfig:
    gate_aperture_range_m: tuple[float, float] = (1.45, 1.80)
    bounds: tuple[tuple[float, float], ...] = (
        (-32.0, 32.0), (-24.0, 24.0), (0.0, 9.0),
    )
    minimum_gate_spacing_m: float = 2.15
    minimum_nonadjacent_spacing_m: float = 1.80
    minimum_long_segment_fraction: float = 0.18
    long_segment_threshold_m: float = 9.0
    minimum_normal_chord_mismatch_fraction: float = 0.12
    maximum_normal_chord_mismatch_fraction: float = 0.78
    planner_samples: int = 1400
    planner_offset_iterations: int = 20
    generation_attempts: int = 160

    def __post_init__(self) -> None:
        low, high = self.gate_aperture_range_m
        if not (1.2 <= low <= high <= 2.5):
            raise ValueError("invalid championship gate aperture range")
        if self.minimum_gate_spacing_m <= 0 or self.generation_attempts < 1:
            raise ValueError("invalid championship generation contract")


def _templates(format_name: str) -> tuple[np.ndarray, list[str], dict[int, np.ndarray]]:
    """Return local gate centers, semantic labels, and directed normal overrides."""

    if format_name == "compact_split_s":
        points = np.asarray([
            [-11.0, -5.0, 3.1], [9.0, -5.0, 1.2], [10.0, 5.0, 1.3],
            [-3.5, 7.0, 4.2], [-3.5, 7.0, 1.45], [6.0, 1.5, 1.2],
            [-5.5, -1.0, 1.4],
        ], np.float64)
        labels = ["sprint", "sprint", "split_s_setup", "split_s_high",
                  "split_s_low", "sprint", "hard_exit"]
        overrides = {3: np.asarray([-1.0, 0.0, 0.0]), 4: np.asarray([1.0, 0.0, 0.0])}
    elif format_name == "compact_hairpin":
        points = np.asarray([
            [-12.0, -6.0, 1.4], [-2.0, -8.0, 1.2], [10.0, -6.0, 1.3],
            [12.0, 3.0, 1.5], [5.0, 8.0, 3.7], [-4.0, 8.0, 1.5],
            [-11.0, 4.0, 1.3], [-2.0, 1.0, 1.4], [5.0, -1.5, 1.3],
        ], np.float64)
        labels = ["sprint", "sprint", "hairpin_entry", "hairpin_exit", "dive",
                  "sprint", "hard_exit", "chicane", "chicane"]
        overrides = {}
    elif format_name == "ladder_sprint":
        points = np.asarray([
            [-17.0, -8.0, 1.3], [-6.0, -10.0, 1.2], [7.0, -9.0, 1.2],
            [17.0, -5.0, 1.5], [18.0, 4.0, 3.2], [14.0, 9.0, 5.4],
            [6.0, 10.0, 3.2], [-3.0, 10.0, 1.3], [-13.0, 7.0, 1.3],
            [-17.0, 1.0, 1.4], [-7.0, 0.0, 1.3], [4.0, 2.0, 1.2],
        ], np.float64)
        labels = ["sprint", "sprint", "sprint", "ladder_setup", "ladder_up",
                  "ladder_apex", "ladder_down", "sprint", "hairpin", "hard_exit",
                  "chicane", "chicane"]
        overrides = {}
    elif format_name == "grand_prix":
        points = np.asarray([
            [-25.0, -11.0, 1.4], [-14.0, -14.0, 1.2], [-2.0, -14.0, 1.2],
            [4.5, -13.7, 1.2], [11.0, -13.0, 1.3], [17.0, -11.0, 1.4],
            [23.0, -8.0, 1.5], [26.0, 1.0, 1.4], [22.0, 10.0, 1.5],
            [12.0, 14.0, 1.3], [1.0, 13.0, 1.4], [-10.0, 11.0, 1.4],
            [-20.0, 8.0, 4.5], [-20.0, 8.0, 1.5],
            [-9.0, 4.0, 1.3], [2.0, 7.0, 1.4], [13.0, 5.0, 1.3],
            [18.0, -1.0, 3.8], [10.0, -4.0, 1.4], [0.0, -2.0, 1.3],
            [-10.0, -3.0, 1.2], [-20.0, -5.0, 1.3],
        ], np.float64)
        labels = ["sprint", "sprint", "sprint", "sprint", "sprint", "sprint",
                  "hairpin", "hairpin", "sprint", "chicane", "split_s_setup", "split_s_setup",
                  "split_s_high", "split_s_low", "sprint", "chicane", "ladder_setup",
                  "ladder_apex", "ladder_down", "sprint", "sprint", "hard_exit"]
        overrides = {12: np.asarray([-1.0, 0.0, 0.0]), 13: np.asarray([1.0, 0.0, 0.0])}
    else:
        raise ValueError(f"unknown championship format {format_name!r}")
    return points, labels, overrides


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1.0e-12)


def generate_championship_track(
    format_name: str, *, seed: int, name: str, split: str,
    config: ChampionshipTrackConfig | None = None,
) -> Track:
    cfg = config or ChampionshipTrackConfig()
    rng = np.random.default_rng(int(seed))
    points, labels, overrides = _templates(format_name)
    scale = float(rng.uniform(*FORMAT_HORIZONTAL_SCALE_RANGES[format_name]))
    points[:, :2] *= scale
    points[:, 2] = 1.0 + (points[:, 2] - 1.0) * rng.uniform(0.90, 1.08)
    jitter = rng.normal(0.0, [0.32, 0.32, 0.10], size=points.shape)
    split_indices = {index for index, label in enumerate(labels) if label.startswith("split_s_")}
    for index in split_indices:
        jitter[index, :2] *= 0.12
    points += jitter
    # Preserve a genuinely stacked pair after perturbation.
    for index, label in enumerate(labels):
        if label == "split_s_low":
            points[index, :2] = points[index - 1, :2]

    yaw = float(rng.uniform(-np.pi, np.pi))
    rotation = np.asarray([
        [np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)],
    ])
    points[:, :2] = points[:, :2] @ rotation.T
    points[:, :2] += rng.uniform(-0.75, 0.75, size=2)
    aperture = float(rng.uniform(*cfg.gate_aperture_range_m))
    gates: list[Gate] = []
    normal_modes: list[str] = []
    for index, center in enumerate(points):
        incoming = _unit(center - points[(index - 1) % len(points)])
        outgoing = _unit(points[(index + 1) % len(points)] - center)
        if index in overrides:
            local = overrides[index].copy()
            local[:2] = local[:2] @ rotation.T
            normal = _unit(local)
            mode = "explicit_split_s"
        elif labels[index] in {"hard_exit", "hairpin", "hairpin_entry", "hairpin_exit"}:
            normal = incoming
            mode = "entry_biased"
        elif labels[index] in {"sprint", "ladder_setup"} and index % 3 == 0:
            normal = _unit(0.82 * incoming + 0.18 * outgoing)
            mode = "entry_biased"
        else:
            normal = _unit(incoming + outgoing)
            mode = "bisector"
        up = np.asarray([0.0, 0.0, 1.0])
        if labels[index] in {"ladder_up", "ladder_apex", "ladder_down"}:
            bank = float(rng.uniform(-0.20, 0.20))
            lateral = _unit(np.cross(up, normal))
            up = _unit(np.cos(bank) * up + np.sin(bank) * lateral)
        size = aperture * rng.uniform(0.96, 1.04, size=2)
        gates.append(Gate(
            position=center.astype(np.float32),
            quaternion_wxyz=forward_up_quaternion(normal, up),
            size=size.astype(np.float32),
            name=f"gate_{index:02d}_{labels[index]}",
        ))
        normal_modes.append(mode)
    return Track(
        name=name, gates=tuple(gates), bounds=np.asarray(cfg.bounds, np.float32), loop=True,
        metadata={
            "source": "competition-calibrated-v3", "format": format_name,
            "split": split, "seed": int(seed), "primitive_labels": labels,
            "normal_modes": normal_modes,
            "distribution_targets": {
                "swift_2022": "7 gates, 75 m published lap, 1.45 m aperture",
                "a2rl_2025_grand_challenge": "22 gates, 170 m, 17.225 s winning run",
                "a2rl_2025_drag": "3 gates, 4.43 s winning run",
            },
        },
    )


def _nonadjacent_distances(points: np.ndarray) -> list[float]:
    values: list[float] = []
    count = len(points)
    for left in range(count):
        for right in range(left + 1, count):
            if (right - left) in {1, count - 1}:
                continue
            values.append(float(np.linalg.norm(points[left] - points[right])))
    return values


def audit_championship_track(
    track: Track, *, config: ChampionshipTrackConfig | None = None,
    cache_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Validate race-format statistics without imposing chord-bisector gates."""

    from ..mpcc import RacingLinePlanner, RacingLinePlannerConfig

    cfg = config or ChampionshipTrackConfig()
    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    segments = np.roll(points, -1, axis=0) - points
    spacing = np.linalg.norm(segments, axis=1)
    units = segments / np.maximum(spacing[:, None], 1.0e-9)
    incoming = np.roll(units, 1, axis=0)
    normals = np.stack([gate.normal for gate in track.gates]).astype(np.float64)
    incoming_alignment = np.sum(normals * incoming, axis=1)
    outgoing_alignment = np.sum(normals * units, axis=1)
    mismatch = np.minimum(incoming_alignment, outgoing_alignment) < 0.30
    turns = np.degrees(np.arccos(np.clip(np.sum(incoming * units, axis=1), -1.0, 1.0)))
    line = RacingLinePlanner(RacingLinePlannerConfig(
        sample_count=cfg.planner_samples,
        offset_iterations=cfg.planner_offset_iterations,
        cache_directory=str(cache_directory) if cache_directory is not None else None,
    )).plan(track)
    gate_frames = line.evaluate(np.asarray(line.gate_progress))
    line_gate_alignment = np.sum(gate_frames["tangent"] * normals, axis=1)
    query = np.linspace(0.0, line.length, 2800, endpoint=False)
    sampled = line.evaluate(query)
    nonadjacent = _nonadjacent_distances(points)
    labels = list((track.metadata or {}).get("primitive_labels", ()))
    format_name = str((track.metadata or {}).get("format", "unknown"))
    reasons: list[str] = []
    aperture_min = float(min(np.min(gate.size) for gate in track.gates))
    mismatch_fraction = float(np.mean(mismatch))
    long_fraction = float(np.mean(spacing >= cfg.long_segment_threshold_m))
    line_range = FORMAT_LINE_LENGTH_RANGES_M.get(format_name)
    if line_range is None or not line_range[0] <= float(line.length) <= line_range[1]:
        reasons.append("race-format-length")
    if float(spacing.min()) < cfg.minimum_gate_spacing_m:
        reasons.append("gate-spacing-minimum")
    if nonadjacent and min(nonadjacent) < cfg.minimum_nonadjacent_spacing_m:
        # The deliberately stacked Split-S pair is adjacent and therefore excluded.
        reasons.append("nonadjacent-gate-overlap")
    # A 22-gate/170 m Grand Prix can have long collinear acceleration sectors
    # split by intermediate timing gates, so its per-segment threshold is lower.
    minimum_long_fraction = (
        0.12 if format_name == "grand_prix" else cfg.minimum_long_segment_fraction
    )
    if long_fraction < minimum_long_fraction:
        reasons.append("insufficient-acceleration-sectors")
    if not (
        cfg.minimum_normal_chord_mismatch_fraction <= mismatch_fraction
        <= cfg.maximum_normal_chord_mismatch_fraction
    ):
        reasons.append("gate-orientation-distribution")
    if float(line_gate_alignment.min()) < 0.94:
        reasons.append("racing-line-gate-direction")
    if aperture_min < cfg.gate_aperture_range_m[0] * 0.95:
        reasons.append("gate-aperture")
    if np.any(sampled["position"] < np.asarray(cfg.bounds)[:, 0] + 0.05) or np.any(
        sampled["position"] > np.asarray(cfg.bounds)[:, 1] - 0.05
    ):
        reasons.append("world-bounds")
    if "split_s_high" in labels:
        high = labels.index("split_s_high")
        low = labels.index("split_s_low")
        if np.linalg.norm(points[high, :2] - points[low, :2]) > 0.08:
            reasons.append("split-s-not-stacked")
        if points[high, 2] - points[low, 2] < 2.2:
            reasons.append("split-s-insufficient-drop")
        if float(normals[high] @ normals[low]) > -0.94:
            reasons.append("split-s-not-opposed")
    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "track": track.name,
        "format": format_name,
        "gate_count": len(track.gates),
        "center_chord_length_m": float(spacing.sum()),
        "racing_line_length_m": float(line.length),
        "mean_gate_spacing_m": float(spacing.mean()),
        "maximum_gate_spacing_m": float(spacing.max()),
        "long_segment_fraction": long_fraction,
        # Common manifest-audit aliases retained across generator versions.
        "mean_turn_degrees": float(turns.mean()),
        "maximum_turn_degrees": float(turns.max()),
        "mean_centerline_turn_degrees": float(turns.mean()),
        "maximum_centerline_turn_degrees": float(turns.max()),
        "normal_chord_mismatch_fraction": mismatch_fraction,
        "minimum_line_gate_alignment": float(line_gate_alignment.min()),
        "minimum_gate_aperture_m": aperture_min,
        "vertical_excursion_m": float(np.ptp(points[:, 2])),
        "maximum_curvature_m_inv": float(np.max(sampled["curvature"])),
        "p99_curvature_m_inv": float(np.quantile(sampled["curvature"], 0.99)),
        "geometry_fingerprint": geometry_fingerprint(track),
    }


def _schedule(count: int) -> list[str]:
    raw = np.asarray([count * weight for _, weight in CHAMPIONSHIP_FORMAT_WEIGHTS])
    quota = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - quota))[:count - int(quota.sum())]:
        quota[index] += 1
    return [name for (name, _), amount in zip(CHAMPIONSHIP_FORMAT_WEIGHTS, quota) for _ in range(amount)]


def generate_championship_manifest(
    output: str | Path, *, train_count: int, validation_count: int, seed: int,
    config: ChampionshipTrackConfig | None = None,
) -> Path:
    cfg = config or ChampionshipTrackConfig()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split, count, offset in (("train", train_count, 0), ("validation", validation_count, 10_000_000)):
        formats = _schedule(count)
        np.random.default_rng(seed + offset).shuffle(formats)
        for slot, format_name in enumerate(formats):
            accepted = None
            failures: list[str] = []
            for attempt in range(cfg.generation_attempts):
                track_seed = int(seed + offset + slot * 1009 + attempt * 104729)
                name = f"champ_{split}_{slot:03d}_{format_name}_{track_seed}"
                track = generate_championship_track(
                    format_name, seed=track_seed, name=name, split=split, config=cfg
                )
                report = audit_championship_track(
                    track, config=cfg, cache_directory=root / "racing-lines"
                )
                fingerprint = str(report["geometry_fingerprint"])
                failures.extend(report["reasons"])
                if report["valid"] and fingerprint not in seen:
                    accepted = track, report, track_seed, fingerprint
                    break
            if accepted is None:
                raise RuntimeError(
                    f"could not generate {split} {format_name}: {sorted(set(failures))}"
                )
            track, report, track_seed, fingerprint = accepted
            relative = Path("tracks") / split / f"{track.name}.yaml"
            save_track_yaml(track, root / relative)
            artifact = load_track(root / relative)
            artifact_fingerprint = geometry_fingerprint(artifact)
            seen.add(artifact_fingerprint)
            records.append({
                "name": artifact.name, "family": format_name, "split": split,
                "seed": track_seed, "path": str(relative),
                "track_fingerprint": artifact.fingerprint,
                "geometry_fingerprint": artifact_fingerprint,
                "static_audit": report, "qualified_speed_mps": None,
                "qualification": None,
            })
    payload = {
        "schema": "starscream-procedural-track-manifest-v1",
        "generator": "competition-calibrated-v3", "seed": int(seed),
        "config": asdict(cfg), "family_weights": dict(CHAMPIONSHIP_FORMAT_WEIGHTS),
        "records": records,
    }
    destination = root / "manifest.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def repair_unqualified_championship_manifest(
    path: str | Path, *, maximum_attempts: int = 180,
) -> dict[str, Any]:
    """Replace dynamically rejected championship slots deterministically.

    Static validity and family/split membership are preserved. Rejected records
    and their qualification telemetry remain archived in the manifest so the
    dynamic admission filter is fully auditable.
    """

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("generator") != "competition-calibrated-v3":
        raise ValueError("targeted repair requires a competition-calibrated-v3 manifest")
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
    cfg = ChampionshipTrackConfig()
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
            candidate_seed = (
                int(old["seed"]) + generation * 1_000_003 + (attempt + 1) * 104_729
            )
            name = f"champ_{split}_{split_slot:03d}_{family}_{candidate_seed}"
            track = generate_championship_track(
                family, seed=candidate_seed, name=name, split=split, config=cfg
            )
            audit = audit_championship_track(
                track, config=cfg,
                cache_directory=manifest_path.parent / "racing-lines",
            )
            fingerprint = geometry_fingerprint(track)
            failures.extend(audit["reasons"])
            if audit["valid"] and fingerprint not in seen:
                accepted = track, audit, candidate_seed
                break
        if accepted is None:
            raise RuntimeError(
                f"could not statically repair {old['name']}; "
                f"failures={sorted(set(failures))}"
            )
        track, audit, candidate_seed = accepted
        relative = Path("tracks") / split / f"{track.name}.yaml"
        save_track_yaml(track, manifest_path.parent / relative)
        artifact = load_track(manifest_path.parent / relative)
        artifact_geometry = geometry_fingerprint(artifact)
        seen.add(artifact_geometry)
        archive.append({
            **old,
            "rejected_utc": datetime.now(timezone.utc).isoformat(),
            "replaced_by": track.name,
        })
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    return {
        "replaced": len(replacements), "records": replacements,
        "manifest": str(manifest_path),
    }
