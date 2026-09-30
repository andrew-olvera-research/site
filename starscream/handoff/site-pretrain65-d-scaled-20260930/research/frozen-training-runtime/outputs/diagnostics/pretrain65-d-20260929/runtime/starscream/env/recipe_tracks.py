"""Recipe-balanced physical-gate courses for general racing policies.

The UTT import represented flags and obstacle maneuvers as mandatory virtual
checkpoints.  That made its route mathematically different from a physical
gate course and gave failures on those synthetic checkpoints disproportionate
authority during DAgger/PPO.  This module replaces that representation with a
bank of closed courses composed exclusively from traversable gate apertures.

The generator deliberately separates *topology* from metric realization.  A
family fixes an ordered transition recipe; scale, pose, aperture, and local
geometry are randomized while preserving the recipe.  Two recipes are
adaptations of independently sourced layouts (Swift and CDRA), and six span
missing transition combinations.  Canonical references remain immutable
held-out evaluations and are never copied into a generated manifest.
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


RECIPE_FAMILIES: tuple[str, ...] = (
    "swift_stack_sprint",
    "ladder_sprint_extended",
    "cdra_sparse_switchback",
    "sprint_brake_hairpin",
    "compound_chicane",
    "vertical_corkscrew",
    "physical_over_under",
    "grand_prix_mixed",
)

FAMILY_SPEED_FLOORS_MPS: dict[str, float] = {
    "swift_stack_sprint": 14.0,
    "ladder_sprint_extended": 10.0,
    "cdra_sparse_switchback": 14.0,
    "sprint_brake_hairpin": 8.0,
    "compound_chicane": 10.0,
    "vertical_corkscrew": 10.0,
    "physical_over_under": 8.0,
    "grand_prix_mixed": 10.0,
}

FAMILY_LENGTH_RANGES_M: dict[str, tuple[float, float]] = {
    "swift_stack_sprint": (64.0, 88.0),
    "ladder_sprint_extended": (98.0, 145.0),
    "cdra_sparse_switchback": (61.0, 92.0),
    "sprint_brake_hairpin": (88.0, 116.0),
    "compound_chicane": (96.0, 128.0),
    "vertical_corkscrew": (82.0, 122.0),
    "physical_over_under": (80.0, 120.0),
    "grand_prix_mixed": (145.0, 190.0),
}

FAMILY_BASE_SCALES: dict[str, float] = {
    "sprint_brake_hairpin": 0.88,
    "compound_chicane": 0.88,
    "grand_prix_mixed": 0.79,
}

# A reference-centered slot that repeatedly misses its frontier is not kept by
# relaxing the floor.  After three dynamic rejections, replace it with one of
# the physically equivalent high-speed vertical recipes.  At least one Swift
# train anchor is still required by the distribution audit/launcher.
FAMILY_REPAIR_FALLBACKS: dict[str, tuple[str, ...]] = {
    "swift_stack_sprint": ("physical_over_under", "ladder_sprint_extended"),
}


@dataclass(frozen=True, slots=True)
class RecipeTrackConfig:
    aperture_range_m: tuple[float, float] = (1.78, 2.18)
    horizontal_scale_range: tuple[float, float] = (0.94, 1.07)
    local_xy_jitter_m: float = 0.14
    local_z_jitter_m: float = 0.06
    orientation_jitter_degrees: float = 1.25
    minimum_gate_spacing_m: float = 2.0
    minimum_nonadjacent_spacing_m: float = 1.35
    minimum_line_gate_alignment: float = 0.91
    minimum_altitude_m: float = 0.88
    generation_attempts: int = 240
    bounds_margin_m: float = 4.0


def _asset_template(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root = Path(__file__).resolve().parents[1] / "assets" / "tracks"
    track = load_track(root / name)
    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    normals = np.stack([gate.normal for gate in track.gates]).astype(np.float64)
    labels = [gate.name.lower() for gate in track.gates]
    return points, normals, labels


def _synthetic_template(family: str) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    if family == "ladder_sprint_extended":
        points = np.asarray([
            [-17, -8, 1.3], [-6, -10, 1.2], [7, -9, 1.2], [17, -5, 1.5],
            [18, 4, 3.2], [14, 9, 5.4], [6, 10, 3.2], [-3, 10, 1.3],
            [-13, 7, 1.3], [-17, 1, 1.4], [-7, 0, 1.3], [4, 2, 1.2],
        ], np.float64)
        labels = ["sprint", "sprint", "sprint", "ladder_setup", "ladder_up",
                  "ladder_apex", "ladder_down", "sprint", "hairpin", "hard_exit",
                  "chicane", "chicane"]
    elif family == "sprint_brake_hairpin":
        points = np.asarray([
            [-18, -7, 1.3], [-5, -8, 1.2], [10, -7, 1.3], [17, -2, 1.4],
            [14, 7, 1.5], [4, 9, 1.3], [-5, 7, 3.4], [-13, 4, 1.3],
            [-8, -1, 1.2], [3, -2, 1.3],
        ], np.float64)
        labels = ["sprint", "sprint", "brake", "hairpin_entry", "hairpin_exit",
                  "sprint", "dive", "switchback", "chicane", "chicane"]
    elif family == "compound_chicane":
        points = np.asarray([
            [-18, -8, 1.3], [-7, -9, 1.2], [3, -5, 1.4], [12, -9, 1.3],
            [18, -2, 1.5], [14, 7, 1.4], [4, 9, 1.3], [-5, 5, 1.5],
            [-13, 9, 1.2], [-18, 2, 1.4], [-9, -1, 1.3], [1, 2, 1.4],
        ], np.float64)
        labels = ["sprint", "chicane_left", "chicane_right", "brake", "hairpin",
                  "sprint", "chicane_left", "chicane_right", "switchback", "hard_exit",
                  "chicane_left", "chicane_right"]
    elif family == "vertical_corkscrew":
        points = np.asarray([
            [-15, -7, 1.3], [-4, -9, 1.2], [8, -7, 1.4], [14, -1, 3.0],
            [10, 6, 5.4], [2, 8, 3.5], [-6, 6, 1.3], [-12, 1, 3.8],
            [-7, -3, 5.2], [1, -1, 2.9], [7, 2, 1.3],
        ], np.float64)
        labels = ["sprint", "sprint", "climb_setup", "corkscrew_up", "corkscrew_apex",
                  "corkscrew_down", "sprint", "vertical_switch", "vertical_switch",
                  "dive", "hard_exit"]
    elif family == "physical_over_under":
        points = np.asarray([
            [-15, -7, 1.3], [-3, -9, 1.2], [10, -7, 1.4], [15, 1, 4.8],
            [14.4, 1.2, 1.4], [7, 7, 1.3], [-3, 8, 1.4], [-12, 4, 4.6],
            [-8, 3.5, 1.4], [-8, -2, 1.3], [2, -1, 1.2],
        ], np.float64)
        labels = ["sprint", "sprint", "over_setup", "over_high", "under_low",
                  "sprint", "ladder_setup", "ladder_high", "ladder_low", "hard_exit", "chicane"]
    elif family == "grand_prix_mixed":
        points = np.asarray([
            [-26, -12, 1.3], [-14, -15, 1.2], [0, -15, 1.2], [13, -14, 1.3],
            [24, -9, 1.4], [28, 1, 1.4], [23, 12, 1.3], [11, 16, 1.2],
            [-2, 14, 1.4], [-14, 12, 4.8], [-14, 11.6, 1.4], [-24, 6, 1.3],
            [-19, -1, 1.4], [-8, 2, 1.2], [3, 6, 1.4], [15, 4, 3.8],
            [9, -3, 1.3], [-3, -1, 1.2], [-14, -5, 1.3],
        ], np.float64)
        labels = ["sprint", "sprint", "sprint", "sprint", "brake", "hairpin",
                  "sprint", "chicane", "split_setup", "split_high", "split_low",
                  "hard_exit", "chicane", "chicane", "ladder_setup", "ladder_high",
                  "ladder_low", "sprint", "hard_exit"]
    else:
        raise ValueError(f"unknown synthetic recipe family {family!r}")
    return points, None, labels


def _template(family: str) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    if family == "swift_stack_sprint":
        return _asset_template("swift_champion_2022_exact.yaml")
    if family == "cdra_sparse_switchback":
        return _asset_template("multigp_cdra_2026_reconstructed.yaml")
    return _synthetic_template(family)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1.0e-12)


def _rotate_xy(values: np.ndarray, yaw: float) -> np.ndarray:
    rotation = np.asarray([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    result = np.asarray(values, np.float64).copy()
    result[..., :2] = result[..., :2] @ rotation.T
    return result


def _stacked_pairs(points: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for index in range(len(points)):
        following = (index + 1) % len(points)
        if (np.linalg.norm(points[index, :2] - points[following, :2]) < 0.9
                and abs(points[index, 2] - points[following, 2]) > 1.5):
            result.append((index, following))
    return result


def generate_recipe_track(
    family: str, *, seed: int, name: str, split: str,
    config: RecipeTrackConfig | None = None,
) -> Track:
    if family not in RECIPE_FAMILIES:
        raise ValueError(f"unknown recipe family {family!r}")
    cfg = config or RecipeTrackConfig()
    rng = np.random.default_rng(int(seed))
    original, source_normals, labels = _template(family)
    source_sizes: np.ndarray | None = None
    if source_normals is not None:
        source_name = {
            "swift_stack_sprint": "swift_champion_2022_exact.yaml",
            "cdra_sparse_switchback": "multigp_cdra_2026_reconstructed.yaml",
        }[family]
        source_track = load_track(
            Path(__file__).resolve().parents[1] / "assets" / "tracks" / source_name
        )
        source_sizes = np.stack([gate.size for gate in source_track.gates]).astype(np.float64)
    points = original.copy()
    center = points[:, :2].mean(0)
    # Swift is the high-speed anchor.  MPCC's frontier on its stacked-gate
    # sequence is extremely sensitive to independent metric perturbations, so
    # augment it with rigid world transforms and observation/plant noise only.
    # The manifest labels this family in-distribution and never calls the exact
    # canonical evaluation zero-shot.
    scale = (
        1.0 if family == "swift_stack_sprint"
        else float(rng.uniform(*cfg.horizontal_scale_range))
        * FAMILY_BASE_SCALES.get(family, 1.0)
    )
    points[:, :2] = center + scale * (points[:, :2] - center)
    anchor = source_normals is not None
    anchor_factor = 0.30 if family == "swift_stack_sprint" else (0.55 if anchor else 1.0)
    jitter = (
        np.zeros_like(points) if family == "swift_stack_sprint"
        else rng.normal(0.0, [
            anchor_factor * cfg.local_xy_jitter_m,
            anchor_factor * cfg.local_xy_jitter_m,
            anchor_factor * cfg.local_z_jitter_m,
        ], size=points.shape)
    )
    pairs = _stacked_pairs(original)
    for left, right in pairs:
        joint = 0.5 * (jitter[left, :2] + jitter[right, :2])
        relative = rng.uniform(-0.035, 0.035, size=2)
        jitter[left, :2] = joint + relative
        jitter[right, :2] = joint - relative
    points += jitter
    z_offset = (
        0.0 if family == "swift_stack_sprint"
        else float(rng.uniform(0.05, 0.45))
    )
    points[:, 2] = np.maximum(points[:, 2] + z_offset, cfg.minimum_altitude_m)
    yaw = (
        0.0 if family == "swift_stack_sprint"
        else float(rng.uniform(-np.pi, np.pi))
    )
    centered = points.copy()
    centered[:, :2] -= center
    points = _rotate_xy(centered, yaw)
    translation = (
        rng.integers(4, 13, size=2).astype(np.float64)
        if family == "swift_stack_sprint"
        else rng.uniform(-1.0, 1.0, size=2)
    )
    points[:, :2] += translation

    if source_normals is not None:
        rotated_source_normals = _rotate_xy(source_normals, yaw)
    else:
        rotated_source_normals = None
    aperture = float(rng.uniform(*cfg.aperture_range_m))
    gates: list[Gate] = []
    normal_modes: list[str] = []
    for index, point in enumerate(points):
        incoming = _unit(point - points[(index - 1) % len(points)])
        outgoing = _unit(points[(index + 1) % len(points)] - point)
        tangent = incoming + outgoing
        if np.linalg.norm(tangent) < 0.20:
            tangent = incoming
        tangent = _unit(tangent)
        label = labels[index]
        if rotated_source_normals is not None:
            normal = _unit(rotated_source_normals[index])
            mode = "source_preserved"
        elif ("high" in label or "low" in label) and any(index in pair for pair in pairs):
            # A stacked pair must have opposed traversal directions.  The route
            # chord determines which side is the approach side.
            normal = tangent
            mode = "stacked_route"
        elif label in {"brake", "hard_exit", "hairpin", "hairpin_entry", "hairpin_exit"}:
            normal = incoming
            mode = "entry_biased"
        else:
            normal = tangent
            mode = "bisector"
        orientation_scale = 0.0 if anchor else 1.0
        local_yaw = float(np.radians(rng.uniform(
            -orientation_scale * cfg.orientation_jitter_degrees,
            orientation_scale * cfg.orientation_jitter_degrees,
        )))
        normal = _unit(_rotate_xy(normal[None], local_yaw)[0])
        up = np.asarray([0.0, 0.0, 1.0])
        if family == "swift_stack_sprint":
            assert source_sizes is not None
            size = source_sizes[index]
        elif anchor:
            assert source_sizes is not None
            size = source_sizes[index] * rng.uniform(0.99, 1.04)
        else:
            size = aperture * rng.uniform(0.97, 1.03, size=2)
        gates.append(Gate(
            position=point.astype(np.float32),
            quaternion_wxyz=forward_up_quaternion(normal, up),
            size=size.astype(np.float32), name=f"gate_{index:02d}_{label}",
        ))
        normal_modes.append(mode)
    lower = points.min(0) - cfg.bounds_margin_m
    upper = points.max(0) + cfg.bounds_margin_m
    lower[2] = 0.0
    upper[2] = max(upper[2], 8.0)
    return Track(
        name=name, gates=tuple(gates), bounds=np.stack([lower, upper], axis=1).astype(np.float32),
        loop=True,
        metadata={
            "source": "physical-gate-recipe-v1", "family": family,
            "split": split, "seed": int(seed), "primitive_labels": labels,
            "normal_modes": normal_modes, "horizontal_scale": scale,
            "global_yaw_radians": yaw, "canonical_reference_in_training": False,
            "all_checkpoints_are_physical_gate_apertures": True,
            "scientific_scope": (
                "reference-centered-adaptation" if family.startswith(("swift_", "cdra_"))
                else "topology-recipe-generation"
            ),
        },
    )


def transition_descriptors(track: Track) -> np.ndarray:
    """Per-gate ordered descriptors used for support/coverage auditing.

    Columns are segment length, turn angle, signed elevation change, incoming
    gate-plane mismatch, and normalized position in the lap.
    """
    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    normals = np.stack([gate.normal for gate in track.gates]).astype(np.float64)
    segments = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(segments, axis=1)
    unit = segments / np.maximum(lengths[:, None], 1.0e-9)
    incoming = np.roll(unit, 1, axis=0)
    turns = np.degrees(np.arccos(np.clip(np.sum(incoming * unit, axis=1), -1.0, 1.0)))
    mismatch = np.degrees(np.arccos(np.clip(np.abs(np.sum(normals * incoming, axis=1)), 0.0, 1.0)))
    phase = np.arange(len(points), dtype=np.float64) / len(points)
    return np.column_stack([lengths, turns, segments[:, 2], mismatch, phase])


def _nonadjacent_distances(points: np.ndarray) -> list[float]:
    values: list[float] = []
    count = len(points)
    for left in range(count):
        for right in range(left + 1, count):
            if right - left in {1, count - 1}:
                continue
            values.append(float(np.linalg.norm(points[left] - points[right])))
    return values


def audit_recipe_track(
    track: Track, *, config: RecipeTrackConfig | None = None,
    cache_directory: str | Path | None = None,
) -> dict[str, Any]:
    from ..mpcc import RacingLinePlanner, RacingLinePlannerConfig

    cfg = config or RecipeTrackConfig()
    family = str((track.metadata or {}).get("family", ""))
    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    descriptors = transition_descriptors(track)
    spacing = descriptors[:, 0]
    nonadjacent = _nonadjacent_distances(points)
    line = RacingLinePlanner(RacingLinePlannerConfig(
        sample_count=1400, offset_iterations=30,
        cache_directory=str(cache_directory) if cache_directory is not None else None,
    )).plan(track)
    gate_frames = line.evaluate(np.asarray(line.gate_progress))
    normals = np.stack([gate.normal for gate in track.gates]).astype(np.float64)
    alignment = np.sum(gate_frames["tangent"] * normals, axis=1)
    query = np.linspace(0.0, line.length, 2800, endpoint=False)
    sampled = line.evaluate(query)
    reasons: list[str] = []
    expected_range = FAMILY_LENGTH_RANGES_M.get(family)
    if expected_range is None or not expected_range[0] <= float(line.length) <= expected_range[1]:
        reasons.append("recipe-length")
    # The released Swift layout has a 1.982 m adjacent stacked transition.
    # Preserve that source geometry; the generic 2 m procedural threshold is
    # not a reason to deform an authoritative reference anchor.
    minimum_gate_spacing = (
        1.90 if family == "swift_stack_sprint" else cfg.minimum_gate_spacing_m
    )
    if float(spacing.min()) < minimum_gate_spacing:
        reasons.append("gate-spacing")
    if nonadjacent and min(nonadjacent) < cfg.minimum_nonadjacent_spacing_m:
        reasons.append("nonadjacent-overlap")
    if float(alignment.min()) < cfg.minimum_line_gate_alignment:
        reasons.append("racing-line-gate-direction")
    if any(gate.kind != "gate" or not gate.render for gate in track.gates):
        reasons.append("nonphysical-route-checkpoint")
    if float(points[:, 2].min()) < cfg.minimum_altitude_m - 1.0e-5:
        reasons.append("altitude-floor")
    if np.any(points < track.bounds[:, 0]) or np.any(points > track.bounds[:, 1]):
        reasons.append("world-bounds")
    if len(track.gates) >= 8 and float(np.max(descriptors[:, 1])) < 50.0:
        reasons.append("missing-hard-transition")
    return {
        "valid": not reasons, "reasons": sorted(set(reasons)),
        "track": track.name, "family": family, "gate_count": len(track.gates),
        "racing_line_length_m": float(line.length),
        "minimum_gate_spacing_m": float(spacing.min()),
        "maximum_gate_spacing_m": float(spacing.max()),
        "mean_turn_degrees": float(descriptors[:, 1].mean()),
        "maximum_turn_degrees": float(descriptors[:, 1].max()),
        "vertical_excursion_m": float(np.ptp(points[:, 2])),
        "minimum_line_gate_alignment": float(alignment.min()),
        "p99_curvature_m_inv": float(np.quantile(sampled["curvature"], 0.99)),
        "all_checkpoints_physical": all(g.kind == "gate" and g.render for g in track.gates),
        "geometry_fingerprint": geometry_fingerprint(track),
    }


def _family_schedule(per_family: int) -> list[str]:
    return [family for family in RECIPE_FAMILIES for _ in range(per_family)]


def generate_recipe_manifest(
    output: str | Path, *, train_per_family: int, validation_per_family: int,
    seed: int, config: RecipeTrackConfig | None = None,
) -> Path:
    cfg = config or RecipeTrackConfig()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split, per_family, offset in (
        ("train", train_per_family, 0),
        ("validation", validation_per_family, 10_000_000),
    ):
        schedule = _family_schedule(per_family)
        for slot, family in enumerate(schedule):
            accepted = None
            failures: list[str] = []
            for attempt in range(cfg.generation_attempts):
                track_seed = int(seed + offset + slot * 1009 + attempt * 104729)
                name = f"recipe_{split}_{slot:03d}_{family}_{track_seed}"
                track = generate_recipe_track(
                    family, seed=track_seed, name=name, split=split, config=cfg,
                )
                try:
                    audit = audit_recipe_track(
                        track, config=cfg, cache_directory=root / "racing-lines"
                    )
                except (ValueError, FloatingPointError) as error:
                    failures.append(type(error).__name__)
                    continue
                fingerprint = str(audit["geometry_fingerprint"])
                failures.extend(audit["reasons"])
                if audit["valid"] and fingerprint not in seen:
                    accepted = track, audit, track_seed, fingerprint
                    break
            if accepted is None:
                raise RuntimeError(
                    f"could not generate {split}/{family}; failures={sorted(set(failures))}"
                )
            track, audit, track_seed, fingerprint = accepted
            relative = Path("tracks") / split / f"{track.name}.yaml"
            save_track_yaml(track, root / relative)
            artifact = load_track(root / relative)
            artifact_geometry = geometry_fingerprint(artifact)
            if artifact_geometry in seen:
                raise RuntimeError(f"serialized geometry collision for {track.name}")
            seen.add(artifact_geometry)
            records.append({
                "name": artifact.name, "family": family, "split": split,
                "seed": track_seed, "path": str(relative),
                "track_fingerprint": artifact.fingerprint,
                "geometry_fingerprint": artifact_geometry, "static_audit": {
                    **audit, "geometry_fingerprint": artifact_geometry,
                },
                "speed_floor_mps": FAMILY_SPEED_FLOORS_MPS[family],
                "qualified_speed_mps": None, "qualification": None,
                "dagger_eligible_5inch": True,
                "leaderboard_comparison_eligible": False,
            })
    payload = {
        "schema": "starscream-procedural-track-manifest-v1",
        "generator": "physical-gate-recipe-v1", "seed": int(seed),
        "config": asdict(cfg),
        "family_weights": {family: 1.0 / len(RECIPE_FAMILIES) for family in RECIPE_FAMILIES},
        "excluded_sources": ["multigp-utt"],
        "scientific_scope": {
            "all_training_checkpoints_are_physical_gates": True,
            "canonical_references_in_training": False,
            "reference_centered_adaptation": True,
            "zero_shot_claim_for_anchor_tracks": False,
        },
        "records": records,
    }
    destination = root / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def repair_unqualified_recipe_manifest(
    path: str | Path, *, maximum_attempts: int = 240,
) -> dict[str, Any]:
    """Replace physical recipes rejected by dynamic speed admission.

    A slot is rejected when MPCC found no safe complete trajectory or when its
    highest accepted speed falls below the recipe-specific floor.  The old
    artifact and complete qualification trace remain in ``rejected_records``;
    replacement seeds are deterministic and preserve family/split balance.
    """

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("generator") != "physical-gate-recipe-v1":
        raise ValueError("targeted repair requires a physical-gate-recipe-v1 manifest")
    records = list(payload["records"])
    rejected_indices = [
        index for index, record in enumerate(records)
        if record.get("qualification") is not None
        and (
            record.get("qualified_speed_mps") is None
            or float(record["qualified_speed_mps"])
            < float(record.get(
                "speed_floor_mps", FAMILY_SPEED_FLOORS_MPS[str(record["family"])]
            ))
        )
    ]
    if not rejected_indices:
        return {"replaced": 0, "records": [], "manifest": str(manifest_path)}

    seen = {
        str(record["geometry_fingerprint"])
        for index, record in enumerate(records) if index not in rejected_indices
    }
    archive = list(payload.get("rejected_records", []))
    replacements: list[dict[str, Any]] = []
    cfg = RecipeTrackConfig(**payload.get("config", {}))
    for record_index in rejected_indices:
        old = dict(records[record_index])
        history = list(old.get("repair_history", []))
        generation = len(history) + 1
        split = str(old["split"])
        old_family = str(old["family"])
        split_slot = sum(
            records[index]["split"] == split for index in range(record_index + 1)
        ) - 1
        fallbacks = FAMILY_REPAIR_FALLBACKS.get(old_family, ())
        family = (
            fallbacks[split_slot % len(fallbacks)]
            if fallbacks and generation >= 4 else old_family
        )
        accepted = None
        failures: list[str] = []
        for attempt in range(maximum_attempts):
            candidate_seed = (
                int(old["seed"]) + generation * 1_000_003 + (attempt + 1) * 104_729
            )
            name = f"recipe_{split}_{split_slot:03d}_{family}_{candidate_seed}"
            track = generate_recipe_track(
                family, seed=candidate_seed, name=name, split=split, config=cfg,
            )
            try:
                audit = audit_recipe_track(
                    track, config=cfg,
                    cache_directory=manifest_path.parent / "racing-lines",
                )
            except (ValueError, FloatingPointError) as error:
                failures.append(type(error).__name__)
                continue
            failures.extend(audit["reasons"])
            fingerprint = geometry_fingerprint(track)
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
        if artifact_geometry in seen:
            raise RuntimeError(f"serialized repair collision for {track.name}")
        seen.add(artifact_geometry)
        archive.append({
            **old,
            "rejected_utc": datetime.now(timezone.utc).isoformat(),
            "rejection_reason": "missing-or-below-family-speed-floor",
            "replaced_family": family,
            "replaced_by": artifact.name,
        })
        history.append({
            "name": old["name"], "seed": int(old["seed"]),
            "family": old_family,
            "qualified_speed_mps": old.get("qualified_speed_mps"),
            "speed_floor_mps": old.get("speed_floor_mps"),
            "qualification": old.get("qualification"),
        })
        replacement = {
            "name": artifact.name, "family": family, "split": split,
            "seed": candidate_seed, "path": str(relative),
            "track_fingerprint": artifact.fingerprint,
            "geometry_fingerprint": artifact_geometry,
            "static_audit": {**audit, "geometry_fingerprint": artifact_geometry},
            "speed_floor_mps": FAMILY_SPEED_FLOORS_MPS[family],
            "qualified_speed_mps": None, "qualification": None,
            "dagger_eligible_5inch": True,
            "leaderboard_comparison_eligible": False,
            "repair_generation": generation, "repair_history": history,
        }
        records[record_index] = replacement
        replacements.append({
            "slot": f"{split}:{split_slot}", "family": family,
            "old_family": old_family,
            "old": old["name"], "old_speed_mps": old.get("qualified_speed_mps"),
            "floor_mps": FAMILY_SPEED_FLOORS_MPS[family],
            "new": artifact.name, "new_seed": candidate_seed,
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
