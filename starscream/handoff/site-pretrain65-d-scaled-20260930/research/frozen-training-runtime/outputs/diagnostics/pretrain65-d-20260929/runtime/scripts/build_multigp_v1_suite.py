#!/usr/bin/env python3
"""Materialize, audit, and augment the source-backed MultiGP v1 suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from starscream.env.multigp_tracks import (
    augment_multigp_track,
    canonical_multigp_tracks,
    exact_timing_suite,
    five_inch_dagger_sources,
)
from starscream.env.procedural_tracks import geometry_fingerprint, save_track_yaml
from starscream.mpcc import RacingLinePlanner, RacingLinePlannerConfig


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/workspace/outputs/procedural-tracks/multigp-v1"))
    parser.add_argument("--train-per-source", type=int, default=20)
    parser.add_argument("--validation-per-source", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026100101)
    return parser.parse_args()


def _route_audit(track, cache: Path) -> dict[str, Any]:
    planner = RacingLinePlanner(RacingLinePlannerConfig(
        sample_count=max(1200, 100 * len(track.gates)), offset_iterations=30,
        cache_directory=str(cache),
    ))
    line = planner.plan(track)
    query = np.linspace(0.0, line.length, max(2400, 160 * len(track.gates)), endpoint=False)
    sampled = line.evaluate(query)
    frames = line.evaluate(np.asarray(line.gate_progress))
    normals = np.stack([gate.normal for gate in track.gates])
    alignment = np.sum(frames["tangent"] * normals, axis=1)
    spacing = np.linalg.norm(
        np.roll(np.stack([gate.position for gate in track.gates]), -1, axis=0)
        - np.stack([gate.position for gate in track.gates]), axis=1,
    )
    reasons: list[str] = []
    flag_clearances: list[float] = []
    flag_clearance_by_name: dict[str, float] = {}
    for obstacle in (track.metadata or {}).get("physical_obstacles", ()):
        if str(obstacle.get("kind")) != "flag" or not bool(
            obstacle.get("collision_audit", True)
        ):
            continue
        base = np.asarray(obstacle["position_m"], np.float64)
        pole_height = float(obstacle.get("pole_height_m", 2.44))
        closest_z = np.clip(sampled["position"][:, 2], base[2], base[2] + pole_height)
        delta = sampled["position"] - np.column_stack([
            np.full(len(closest_z), base[0]),
            np.full(len(closest_z), base[1]),
            closest_z,
        ])
        clearance = float(np.linalg.norm(delta, axis=1).min() - 0.30)
        flag_clearances.append(clearance)
        flag_clearance_by_name[str(obstacle.get("name", "unnamed_flag"))] = clearance
    if not np.all(np.isfinite(sampled["position"])):
        reasons.append("nonfinite-racing-line")
    minimum_altitude_margin = float(
        np.min(sampled["position"][:, 2] - float(track.bounds[2, 0]))
    )
    maximum_altitude_margin = float(
        np.min(float(track.bounds[2, 1]) - sampled["position"][:, 2])
    )
    if minimum_altitude_margin < 0.20 or maximum_altitude_margin < 0.20:
        reasons.append("racing-line-world-bound-clearance")
    # A positive margin is the route invariant.  Requiring near-normal
    # traversal is invalid for split-S/over-gate segments: a feasible racing
    # line intentionally crosses the virtual plane while carrying substantial
    # vertical velocity.  Dynamic MPCC qualification still verifies the
    # aperture and one-way crossing itself.
    if float(np.min(alignment)) < 0.35:
        reasons.append("route-direction")
    if float(np.min(spacing)) < 0.20:
        reasons.append("route-checkpoint-spacing")
    if flag_clearances and min(flag_clearances) < 0.10:
        reasons.append("flag-pole-body-clearance")
    # Tiny WhUTT deliberately places whoop gates edge-to-edge and is audited
    # under its own plant.  For five-inch routes, reject only pathological
    # sub-5 cm planner radii here; dynamic feasibility is the MPCC gate.
    if bool((track.metadata or {}).get("five_inch", True)) and float(
        np.quantile(sampled["curvature"], 0.999)
    ) > 20.0:
        reasons.append("singular-curvature")
    source_length = (track.metadata or {}).get("source_route_length_m")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "route_checkpoint_count": len(track.gates),
        "rendered_gate_count": sum(gate.render for gate in track.gates),
        "racing_line_length_m": float(line.length),
        "source_route_length_m": source_length,
        "source_length_ratio": None if source_length is None else float(line.length / source_length),
        "minimum_checkpoint_spacing_m": float(np.min(spacing)),
        "minimum_line_direction_alignment": float(np.min(alignment)),
        "minimum_altitude_bound_clearance_m": minimum_altitude_margin,
        "maximum_altitude_bound_clearance_m": maximum_altitude_margin,
        "minimum_flag_pole_body_clearance_m": (
            min(flag_clearances) if flag_clearances else None
        ),
        "flag_pole_body_clearance_m": flag_clearance_by_name,
        "p999_curvature_m_inv": float(np.quantile(sampled["curvature"], 0.999)),
        "maximum_curvature_m_inv": float(np.max(sampled["curvature"])),
        "geometry_fingerprint": geometry_fingerprint(track),
        "track_fingerprint": track.fingerprint,
    }


def _record(track, path: Path, root: Path, split: str, audit: dict[str, Any]) -> dict[str, Any]:
    metadata = track.metadata or {}
    return {
        "name": track.name,
        "path": str(path.relative_to(root)),
        "split": split,
        "family": str(metadata.get("reference_name", track.name)),
        "geometry_fingerprint": geometry_fingerprint(track),
        "track_fingerprint": track.fingerprint,
        "source_geometry_fingerprint": metadata.get("source_geometry_fingerprint"),
        "reference_track_fingerprint": metadata.get("reference_track_fingerprint"),
        "geometry_fidelity": metadata.get("geometry_fidelity"),
        "leaderboard_comparison_eligible": bool(metadata.get("leaderboard_comparison_eligible", False)),
        "dagger_eligible_5inch": bool(metadata.get("dagger_eligible_5inch", False)),
        # Qualification and collection use this to produce deterministic,
        # episode-specific initial conditions. Canonical source tracks do not
        # have an augmentation seed, so zero is their stable sentinel.
        "seed": int(metadata.get("augmentation_seed", 0)),
        # One timing lap is the atomic imitation unit.  Without this explicit
        # contract, short UTT layouts inherit a global gate target and silently
        # turn into multi-lap episodes with a different state distribution.
        "dagger_laps": 1,
        "qualified_speed_mps": None,
        "static_audit": audit,
    }


def main() -> None:
    args = parse_arguments()
    if args.train_per_source < 1 or args.validation_per_source < 1:
        raise ValueError("each source needs train and validation augmentations")
    root = args.output.resolve()
    manifest = root / "manifest.json"
    previous_by_name: dict[str, dict[str, Any]] = {}
    if manifest.exists():
        previous_payload = json.loads(manifest.read_text(encoding="utf-8"))
        previous_by_name = {
            str(item["name"]): item
            for item in previous_payload.get("records", ())
        }
    cache = root / "racing-lines"
    records: list[dict[str, Any]] = []
    canonical_audits: dict[str, Any] = {}
    canonical = canonical_multigp_tracks()
    for track in canonical:
        path = root / "tracks" / "canonical" / f"{track.name}.yaml"
        save_track_yaml(track, path)
        audit = _route_audit(track, cache)
        canonical_audits[track.name] = audit
        records.append(_record(track, path, root, "canonical", audit))

    for source_index, source in enumerate(five_inch_dagger_sources()):
        for split, count, offset in (
            ("train", args.train_per_source, 0),
            ("validation", args.validation_per_source, 10_000_000),
        ):
            for slot in range(count):
                seed = int(args.seed + offset + source_index * 100_003 + slot * 1009)
                track = augment_multigp_track(
                    source, seed=seed,
                    name=f"{source.name}_{split}_aug_{slot:03d}_{seed}",
                )
                path = root / "tracks" / split / f"{track.name}.yaml"
                save_track_yaml(track, path)
                audit = _route_audit(track, cache)
                records.append(_record(track, path, root, split, audit))

    exact_names = {track.name for track in exact_timing_suite()}
    exact_hashes = {
        item["geometry_fingerprint"] for item in records
        if item["split"] == "canonical" and item["name"] in exact_names
    }
    train_hashes = {item["geometry_fingerprint"] for item in records if item["split"] == "train"}
    validation_hashes = {item["geometry_fingerprint"] for item in records if item["split"] == "validation"}
    duplicate_hashes = (exact_hashes & train_hashes) | (train_hashes & validation_hashes)
    invalid = [item["name"] for item in records if not item["static_audit"]["valid"]]
    if duplicate_hashes:
        raise RuntimeError(f"held-out/train geometry leakage: {sorted(duplicate_hashes)}")
    if invalid:
        raise RuntimeError(f"static route audit failed: {invalid}")

    # Rebuilding after a localized route fix must not discard hours of dynamic
    # qualification for byte-identical sibling records.  Qualification is
    # carried forward only when both geometry and complete Track fingerprints
    # still match; changed routes are deliberately reset to unqualified.
    preserved_qualifications = 0
    for item in records:
        previous = previous_by_name.get(str(item["name"]))
        if not previous or (
            previous.get("geometry_fingerprint") != item["geometry_fingerprint"]
            or previous.get("track_fingerprint") != item["track_fingerprint"]
        ):
            continue
        if "qualification" in previous:
            item["qualification"] = previous["qualification"]
            item["qualified_speed_mps"] = previous.get("qualified_speed_mps")
            preserved_qualifications += 1

    generalization_folds = [
        {
            "name": "fold_1",
            "held_out_families": ["multigp_utt01", "multigp_utt04_high_voltage"],
        },
        {
            "name": "fold_2",
            "held_out_families": ["multigp_utt02_tsunami", "multigp_utt06_fury"],
        },
        {
            "name": "fold_3",
            "held_out_families": ["multigp_utt03_bessel_run", "multigp_utt08_revenge"],
        },
        {
            "name": "fold_4",
            "held_out_families": ["multigp_utt05_nautilus", "multigp_utt10_prairie_rage"],
        },
    ]
    held_out_families = [
        family for fold in generalization_folds
        for family in fold["held_out_families"]
    ]
    if len(held_out_families) != len(set(held_out_families)):
        raise RuntimeError("generalization folds contain duplicate held-out families")
    if set(held_out_families) != exact_names:
        raise RuntimeError(
            "generalization folds must partition the exact timing suite: "
            f"folds={sorted(held_out_families)} exact={sorted(exact_names)}"
        )

    payload = {
        "schema": "starscream-procedural-track-manifest-v1",
        "generator": "source-backed-multigp-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "records": records,
        "scientific_scope": {
            "exact_timing_suite": sorted(exact_names),
            "reference_centered_training": True,
            "zero_shot_claim": False,
            "excluded_from_five_inch": ["multigp_utt07_tiny_whutt"],
            "excluded_from_exact_timing": ["multigp_utt09_mega_reconstruction"],
            "immutable_canonical_geometry": True,
        },
        # These folds support the genuinely unseen-family result.  The main
        # all-family augmented run is reference adaptation and is deliberately
        # not described as zero-shot.
        "generalization_folds": generalization_folds,
        "split_disjointness": {
            "exact_train_overlap": 0,
            "train_validation_overlap": 0,
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest)
    report = {
        "manifest": str(manifest),
        "canonical_tracks": len(canonical),
        "exact_timing_tracks": len(exact_names),
        "dagger_sources": len(five_inch_dagger_sources()),
        "train_tracks": sum(item["split"] == "train" for item in records),
        "validation_tracks": sum(item["split"] == "validation" for item in records),
        "preserved_qualifications": preserved_qualifications,
        "canonical_audits": canonical_audits,
    }
    report_path = root / "build-audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
