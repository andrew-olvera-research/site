#!/usr/bin/env python3
"""Build the leakage-safe v6.23 RL seed corpus.

The corpus combines retained v6.21 anchors with qualified, independently
generated v6.22 technical programs. Frozen real60 and real100-hard-v2 tracks
are used only as clone-protection inputs and are never copied into training.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex
from starscream.env.tracks import load_track
from scripts.audits.review_v622_geometry import resampled


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/course-pools/v623-rl-technical-v1"
LEGACY = ROOT / "outputs/course-pools/v621-additions/manifest.json"
TECHNICAL = ROOT / "outputs/v622-pretraining60/mpcc-frontier-v3/manifest.json"
PROTECTED = (
    ROOT / "configs/eval/v6_22_real60.manifest.json",
    ROOT / "configs/eval/v6_22_real100_hard_v2.manifest.json",
)

ROLE_BY_LABEL = {
    "long_low_braking": "low_altitude_braking",
    "radius_switch": "speed_regime_switch",
    "technical_vertical": "utt_vertical_sequence",
    "technical_compound": "a2rl_compound_sequence",
    "technical_wrong_side": "directed_approach",
    "ordered_3d_bridge": "ordered_3d_sequence",
    "stacked_reversal_bridge": "stacked_reversal",
    "flow_bridge": "technical_flow",
    "slalom_bridge": "technical_flow",
    "hairpin_bridge": "hairpin_sequence",
    "diving_hairpin_bridge": "vertical_hairpin",
    "go_around_bridge": "go_around",
    "long_braking_bridge": "long_braking",
}

# These are minimum role-level sampling probabilities. The remaining mass is
# allocated adaptively while preserving per-course priorities within each role.
SAMPLING_FLOORS = {
    "retention_anchor": 0.24,
    "low_altitude_braking": 0.08,
    "utt_vertical_sequence": 0.065,
    "a2rl_compound_sequence": 0.065,
    "directed_approach": 0.055,
    "speed_regime_switch": 0.055,
    "technical_flow": 0.07,
    "vertical_hairpin": 0.04,
    "hairpin_sequence": 0.04,
    "stacked_reversal": 0.03,
    "ordered_3d_sequence": 0.03,
    "long_braking": 0.015,
    "go_around": 0.015,
}
assert abs(sum(SAMPLING_FLOORS.values()) - 0.8) < 1e-9


def resolve(path: str | Path) -> Path:
    value = Path(path)
    if str(value).startswith("/workspace/"):
        return ROOT / value.relative_to("/workspace")
    return value if value.is_absolute() else ROOT / value


def records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("records", payload.get("tracks", [])))


def main() -> None:
    protected_rows = [row for path in PROTECTED for row in records(path)]
    protected_tracks = [load_track(resolve(row["path"])) for row in protected_rows]
    exact = CloneIndex(protected_tracks)
    continuous = CloneIndex([resampled(track) for track in protected_tracks])
    protected_fingerprints = {geometry_fingerprint(track) for track in protected_tracks}

    selected: list[dict] = []
    seen_names: set[str] = set()
    sources = (
        ("retention_anchor", records(LEGACY)),
        ("technical", records(TECHNICAL)),
    )
    rejected = Counter()
    for source_role, rows in sources:
        for source in rows:
            if source_role == "retention_anchor" and source.get("split") != "train":
                continue
            path = resolve(source["path"])
            track = load_track(path)
            fingerprint = geometry_fingerprint(track)
            exact_distance = exact.distance(track)
            continuous_distance = continuous.distance(resampled(track))
            if fingerprint in protected_fingerprints:
                rejected["exact_fingerprint"] += 1
                continue
            if exact_distance < 0.12:
                rejected["exact_shape"] += 1
                continue
            if continuous_distance < 0.06:
                rejected["continuous_shape"] += 1
                continue
            if track.name in seen_names:
                rejected["duplicate_name"] += 1
                continue
            seen_names.add(track.name)
            label = source.get("label") or source.get("family") or "legacy"
            role = "retention_anchor" if source_role == "retention_anchor" else ROLE_BY_LABEL[label]
            selected.append({
                **source,
                "name": track.name,
                "path": "/workspace/" + str(path.relative_to(ROOT)),
                "split": "train",
                "rl_role": role,
                "rl_source": source_role,
                "geometry_fingerprint": fingerprint,
                "minimum_protected_exact_distance": float(exact_distance),
                "minimum_protected_continuous_distance": float(continuous_distance),
                "zero_shot_protected": True,
            })

    role_counts = Counter(row["rl_role"] for row in selected)
    missing = sorted(set(SAMPLING_FLOORS) - set(role_counts))
    if missing:
        raise RuntimeError(f"RL corpus lacks required roles: {missing}")
    technical_count = sum(row["rl_source"] == "technical" for row in selected)
    if technical_count != 60:
        raise RuntimeError(f"expected all 60 qualified technical courses, got {technical_count}")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "starscream-procedural-track-manifest-v1",
        "rl_schema": "starscream-v623-rl-corpus-v1",
        "description": "Leakage-safe RL anchors plus independently generated UTT/A2RL-like and technical bridge courses.",
        "records": selected,
        "metadata": {
            "course_count": len(selected),
            "role_counts": dict(sorted(role_counts.items())),
            "sampling_floors": SAMPLING_FLOORS,
            "protected_manifests": [str(path.relative_to(ROOT)) for path in PROTECTED],
            "protected_course_count": len(protected_tracks),
            "clone_thresholds": {"exact": 0.12, "continuous": 0.06},
            "rejected": dict(rejected),
            "evaluation_courses_included": 0,
        },
    }
    atomic_json(OUT / "manifest.json", manifest)
    atomic_json(OUT / "sampling-contract.json", {
        "schema": "starscream-v623-rl-sampling-contract-v1",
        "role_floors": SAMPLING_FLOORS,
        "trainer_setting": "ppo_sampling_role_floors",
        "adaptive_remainder": 0.20,
        "adaptive_remainder_policy": "failure-priority across all roles, preserving within-role priorities",
        "online_generation": {
            "enabled": True,
            "continual_replacement": True,
            "max_bank_size": 200,
            "max_total_bank_size": 480,
            "retired_courses_consume_active_capacity": False,
            "protected_anchor_role": "retention_anchor",
            "generated_courses_share_adaptive_remainder": True,
        },
        "ppo_update": {
            "actor_early_stop_scope": "epoch",
            "actor_epochs": 3,
            "require_complete_actor_epoch": True,
        },
    })
    print(json.dumps(manifest["metadata"], indent=2))


if __name__ == "__main__":
    main()
