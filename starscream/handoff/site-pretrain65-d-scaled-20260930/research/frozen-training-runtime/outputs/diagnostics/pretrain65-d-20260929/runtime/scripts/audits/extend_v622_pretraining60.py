"""Extend the qualified v6.22 pretrain50 corpus with ten real60 bridge cells.

The original 50 qualified courses are reused byte-for-byte.  Only the missing
hairpin, diving-hairpin, go-around, and high-speed-braking bridges are new.
The frozen real60 and real100 manifests are protected by exact and continuous
clone-distance checks and are never copied into this training corpus.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint, save_track_yaml
from starscream.env.racing_manifold.benchmark_v22 import (
    CloneIndex, generate, requirement_cells, validate_geometry,
)
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track
from scripts.audits.review_v622_geometry import resampled
from scripts.audits.build_v622_pretraining50 import _cell_axes

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/v622-pretraining50/mpcc-frontier-v3/manifest.json"
BASE_PROFILES = ROOT / "outputs/v622-pretraining50/mpcc-frontier-v3/teacher-profiles.json"
REAL60 = ROOT / "configs/eval/v6_22_real60.manifest.json"
REAL100 = ROOT / "configs/eval/v6_22_real100_hard_v2.manifest.json"
OUT = ROOT / "outputs/v622-pretraining60"

NEW_QUOTAS = {
    "hairpin_bridge": ("hairpin_chain", 3),
    "diving_hairpin_bridge": ("diving_hairpin", 3),
    "go_around_bridge": ("go_around", 2),
    "long_braking_bridge": ("long_braking", 2),
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if str(p).startswith("/workspace/"):
        return ROOT / p.relative_to("/workspace")
    return p if p.is_absolute() else ROOT / p


def protected_tracks():
    paths = []
    for manifest_path in (REAL60, REAL100):
        for row in json.loads(manifest_path.read_text())["records"]:
            paths.append(resolve(row["path"]))
    return [load_track(p) for p in paths]


def candidate(label: str, family: str, index: int, trial: int):
    seed = 86200000 + index * 100003 + trial
    rng = np.random.default_rng(seed)
    name = f"v622_pretrain60_{label}_{index:02d}_{trial:02d}"
    count = int(rng.integers(8, 13))
    # These are intermediate transfer cells.  real100 retains the hard
    # variants; train60 should bridge the behavior manifold first.
    track = generate(family, count, seed, name, hard=False)
    return track, seed


def main():
    if not BASE.exists() or not BASE_PROFILES.exists():
        raise FileNotFoundError("qualified pretrain50 frontier is required")
    base = json.loads(BASE.read_text())
    if len(base["records"]) != 50:
        raise ValueError("expected exactly 50 qualified pretrain50 records")
    if OUT.exists() and (OUT / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    tracks_dir = OUT / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    protected = protected_tracks() + [load_track(resolve(r["path"])) for r in base["records"]]
    exact = CloneIndex(protected)
    continuous = CloneIndex([resampled(t) for t in protected])
    selected = []
    ledger = []
    for label, (family, quota) in NEW_QUOTAS.items():
        for index in range(quota):
            pool = []
            for trial in range(120):
                track, seed = candidate(label, family, index, trial)
                reasons = validate_geometry(track)
                if reasons:
                    continue
                ex = exact.distance(track)
                cont = continuous.distance(resampled(track))
                if ex < .12 or cont < .06:
                    continue
                cells = set(_cell_axes(track))
                # Favor new coarse witnesses, then shape novelty.  No policy
                # score or held-out outcome enters this selection.
                novelty = min(ex, cont)
                score = .025 * len(cells) + 1.5 * novelty
                pool.append((score, track, seed, ex, cont, cells))
            if not pool:
                raise RuntimeError(f"no independent candidates for {label}/{index}")
            pool.sort(key=lambda x: (-x[0], x[1].name))
            score, track, seed, ex, cont, cells = pool[0]
            # Each selected row is checked against all earlier additions.
            if exact.distance(track) < .12 or continuous.distance(resampled(track)) < .06:
                raise RuntimeError(f"candidate became a clone: {track.name}")
            path = save_track_yaml(track, tracks_dir / f"{track.name}.yaml")
            row = {
                "name": track.name,
                "path": "/workspace/" + str(path.relative_to(ROOT)),
                "split": "train", "suite": "pretraining60",
                "family": family, "label": label,
                "source": "v622-pretraining-behavior-program-v2",
                "seed": int(seed), "rank": 50 + len(selected),
                "geometry_fingerprint": geometry_fingerprint(track),
                "track_fingerprint": track.fingerprint,
                "fingerprint": geometry_fingerprint(track),
                "cells": sorted(requirement_cells(track)),
                "coarse_cells": sorted(cells),
                "geometry": geometry_record(track),
                "behavior_signature": dict((track.metadata or {}).get("behavior_signature", {})),
                "minimum_protected_clone_distance": float(cont),
                "qualified_speed_mps": None, "qualification": None,
                "mpcc_admission": {"contract": "starscream-mpcc-admission-v2",
                                   "rl_eligible": False, "dagger_eligible": False},
            }
            selected.append(row)
            ledger.append({"label": label, "family": family, "name": row["name"],
                           "score": score, "exact_distance": ex,
                           "continuous_distance": cont, "coarse_cells": sorted(cells)})
            exact.add(track); continuous.add(resampled(track))

    records = []
    for row in base["records"]:
        row = dict(row)
        row["suite"] = "pretraining60"
        records.append(row)
    records.extend(selected)
    records.sort(key=lambda r: int(r.get("rank", 0)))
    if len(records) != 60:
        raise AssertionError(len(records))

    manifest = {
        "schema": "starscream-procedural-track-manifest-v1",
        "seed": 8222060,
        "records": records,
        "family_weights": {x: 1.0 for x in sorted(set(r["family"] for r in records))},
        "source": "v6.22 pretraining60: qualified pretrain50 plus ten synthetic real60 bridge cells",
        "cell_plan": "outputs/v622-pretraining60/cell-plan.json",
        "admission": {"status": "pending_new_bridge_qualification",
                      "reused_pretrain50": 50, "new_bridge_courses": 10},
        "protected_suites": ["configs/eval/v6_22_real60.manifest.json",
                             "configs/eval/v6_22_real100_hard_v2.manifest.json"],
        "base_manifest_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
    }
    atomic_json(OUT / "manifest.json", manifest)
    atomic_json(OUT / "cell-plan.json", {
        "schema": "starscream-v622-pretraining60-cell-plan-v1",
        "course_count": 60, "reused_pretrain50": 50,
        "new_bridge_quotas": {k: v[1] for k, v in NEW_QUOTAS.items()},
        "selection_rule": "reuse qualified pretrain50; add independent nominal bridge cells; exact/continuous clone protection against real60, real100-hard-v2, and reused courses",
        "protected_suites": ["configs/eval/v6_22_real60.manifest.json",
                             "configs/eval/v6_22_real100_hard_v2.manifest.json"],
        "coverage_intent": "fill hairpin-chain, diving-hairpin, go-around, and high-speed long-braking gaps while preserving technical, radius-switch, long-low-braking, and bridge diversity",
    })
    atomic_json(OUT / "selection.json", {"reused": 50, "added": ledger,
                                         "new_course_count": len(selected)})
    counts = Counter(c for r in records for c in r.get("coarse_cells", []))
    atomic_json(OUT / "coverage.json", {
        "course_count": 60, "family_counts": dict(Counter(r["family"] for r in records)),
        "coarse_cell_course_witnesses": dict(sorted(counts.items())),
        "exact_clone_threshold": .12, "continuous_clone_threshold": .06,
        "real60_manifest_sha256": hashlib.sha256(REAL60.read_bytes()).hexdigest(),
        "real100_manifest_sha256": hashlib.sha256(REAL100.read_bytes()).hexdigest(),
    })
    suite = {"schema": "starscream-real-course-suite-v1",
             "description": "v6.22 60-course behavior-cell pretraining panel; not an evaluation benchmark",
             "active": [{"name": r["name"], "track": r["path"],
                        "geometry_fingerprint": r["geometry_fingerprint"],
                        "exposure": "fresh training geometry", "family": r["family"],
                        "label": r.get("label")} for r in records]}
    (ROOT / "configs/eval/v6_22_pretraining60.yaml").write_text(
        __import__("yaml").safe_dump(suite, sort_keys=False))
    print(json.dumps({"courses": 60, "reused": 50, "added": len(selected),
                      "families": dict(Counter(r["family"] for r in records)),
                      "manifest": str(OUT / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
