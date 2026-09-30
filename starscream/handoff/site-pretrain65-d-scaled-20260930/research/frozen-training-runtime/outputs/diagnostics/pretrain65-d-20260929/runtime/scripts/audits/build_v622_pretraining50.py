"""Build the v6.22 behavior-cell pretraining panel.

This is deliberately a training corpus builder, not another evaluation-suite
builder.  It samples behavior programs before geometry, protects both frozen
v6.22 evaluations from shape clones, and records the cell plan used to select
the final exactly-50-course panel.  MPCC admission is performed by
``qualify_v622_pretraining50.py`` after this materialization step.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint, save_track_yaml, track_to_mapping
from starscream.env.racing_manifold.benchmark_hard_v2 import STRATA, generate_hard_v2
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex, requirement_cells, validate_geometry, generate
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track
from scripts.audits.review_v622_geometry import resampled

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/v622-pretraining50"
REAL60 = ROOT / "configs/eval/v6_22_real60.manifest.json"
HARD = ROOT / "configs/eval/v6_22_real100_hard_v2.manifest.json"

# The count is intentional: this panel is the complete first v6.22 train
# corpus.  The technical quota is large enough that a2rl-like high gate-count
# behavior is a first-class mode rather than a single anchor.
QUOTAS = {
    "technical_vertical": 6,
    "technical_compound": 5,
    "technical_wrong_side": 5,
    "long_low_braking": 10,
    "radius_switch": 8,
    "ordered_3d_bridge": 4,
    "stacked_reversal_bridge": 4,
    "flow_bridge": 4,
    "slalom_bridge": 4,
}
assert sum(QUOTAS.values()) == 50


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if str(p).startswith("/workspace/"):
        return ROOT / p.relative_to("/workspace")
    return p if p.is_absolute() else ROOT / p


def _protected_tracks():
    tracks = []
    for manifest_path in (REAL60, HARD):
        data = json.loads(manifest_path.read_text())
        for row in data["records"]:
            tracks.append(load_track(_resolve(row["path"])))
    return tracks


def _cell_axes(track):
    """Coarse, interpretable behavior cells used for the selection ledger."""
    rows = geometry_record(track)["transitions"]
    out = set()
    for t in rows:
        approach = "short" if t["incoming_m"] < 12 else "medium" if t["incoming_m"] < 20 else "long" if t["incoming_m"] < 30 else "very_long"
        turn = "low" if t["turn_deg"] < 60 else "medium" if t["turn_deg"] < 120 else "high"
        dz = abs(t["height_change_m"])
        vertical = "flat" if dz < .75 else "mild" if dz < 2 else "strong" if dz < 3 else "extreme"
        aperture = "narrow" if min(t["width_m"], t["height_m"]) <= 1.65 else "normal" if min(t["width_m"], t["height_m"]) <= 2.2 else "wide"
        out.update((f"approach:{approach}", f"turn:{turn}", f"vertical:{vertical}", f"aperture:{aperture}"))
        if t["incoming_m"] >= 25 and t["gate_center_height_m"] <= 1.2:
            out.add("mode:long_low_braking")
        if t["incoming_m"] >= 25 and t["turn_deg"] >= 120:
            out.add("mode:long_high_turn")
        if dz >= 2 and t["turn_deg"] >= 90:
            out.add("mode:vertical_high_turn")
        if t["preceding_gate_on_exit_side"]:
            out.add("grammar:wrong_side_incidence")
        if t["reverse_entry"]:
            out.add("grammar:reverse_entry")
    return out


def _spec_for(label, index, trial):
    seed = 82200000 + index * 100003 + trial
    name = f"v622_pretrain50_{label}_{index:02d}_{trial:02d}"
    rng = np.random.default_rng(seed)
    if label.startswith("technical_"):
        stratum = {
            "technical_vertical": "vertical_chain",
            "technical_compound": "compound_reversal",
            "technical_wrong_side": "wrong_side_incidence",
        }[label]
        count = int(rng.integers(15, 19))
        track = generate_hard_v2(stratum, count, seed, name)
        family = f"a2rl_technical:{stratum}"
    elif label == "long_low_braking":
        track = generate_hard_v2("long_low_braking", int(rng.integers(13, 19)), seed, name)
        family = "behavior:long_low_braking"
    elif label == "radius_switch":
        track = generate_hard_v2("radius_switch", int(rng.integers(13, 19)), seed, name)
        family = "behavior:radius_switch"
    else:
        family_map = {
            "ordered_3d_bridge": "ordered_3d",
            "stacked_reversal_bridge": "stacked_reversal",
            "flow_bridge": "flow",
            "slalom_bridge": "slalom",
        }
        family = family_map[label]
        count = int(rng.integers(8, 13))
        track = generate(family, count, seed, name, hard=True)
    return track, family, seed


def _target_plan():
    # These are coverage goals, not mutually exclusive weights.  They make
    # the intended behavior distribution reviewable before any course is
    # accepted.  All are course-witness minima except total gate/mode goals.
    return {
        "schema": "starscream-v622-pretraining-cell-plan-v1",
        "course_count": 50,
        "quota_by_mode": QUOTAS,
        "minimum_course_witnesses": {
            "approach:short": 18, "approach:medium": 30,
            "approach:long": 30, "approach:very_long": 18,
            "turn:low": 35, "turn:medium": 38, "turn:high": 35,
            "vertical:flat": 40, "vertical:mild": 35,
            "vertical:strong": 25, "vertical:extreme": 8,
            "aperture:narrow": 25, "aperture:normal": 35, "aperture:wide": 18,
            "mode:long_low_braking": 10, "mode:long_high_turn": 18,
            "mode:vertical_high_turn": 16,
            "grammar:wrong_side_incidence": 10,
            "grammar:reverse_entry": 8,
        },
        "selection_rule": "within each quota, maximize unmet coarse-cell deficits, then normalized novelty against real60+real100-hard-v2 and selected courses; no policy scores",
        "protected_suites": [str(REAL60.relative_to(ROOT)), str(HARD.relative_to(ROOT))],
        "technical_definition": "15-18 gates with repeated high-turn transitions; vertical/compound/wrong-side programs; labeled a2rl_technical without copying a2rl geometry",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = _target_plan()
    atomic_json(OUT / "cell-plan.json", plan)
    protected = _protected_tracks()
    exact = CloneIndex(protected)
    continuous = CloneIndex([resampled(t) for t in protected])
    candidates = []
    # Generate an intentionally overcomplete pool.  Cell selection is done
    # after geometry and clone checks, so the generator cannot silently steer
    # the final panel toward a family-shaped artifact.
    for label, quota in QUOTAS.items():
        for index in range(quota):
            for trial in range(28):
                track, family, seed = _spec_for(label, index, trial)
                reasons = validate_geometry(track)
                if reasons:
                    continue
                ex = exact.distance(track)
                cont = continuous.distance(resampled(track))
                if ex < .12 or cont < .06:
                    continue
                candidates.append({
                    "label": label, "family": family, "seed": int(seed),
                    "name": track.name, "track": track,
                    "fingerprint": geometry_fingerprint(track),
                    "exact_distance": float(ex), "continuous_distance": float(cont),
                    "cells": sorted(requirement_cells(track)),
                    "coarse_cells": sorted(_cell_axes(track)),
                    "geometry": geometry_record(track),
                    "behavior_signature": dict((track.metadata or {}).get("behavior_signature", {})),
                })
    if len(candidates) < sum(QUOTAS.values()):
        raise RuntimeError(f"only {len(candidates)} independent candidates for 50-course panel")

    selected = []
    selection_ledger = []
    final_selection_exact = CloneIndex(protected)
    final_selection_cont = CloneIndex([resampled(t) for t in protected])
    by_label = defaultdict(list)
    for c in candidates:
        by_label[c["label"]].append(c)
    for label, quota in QUOTAS.items():
        pool = by_label[label]
        if len(pool) < quota:
            raise RuntimeError(f"{label}: {len(pool)} candidates, need {quota}")
        # Each quota is solved independently, but novelty is checked against
        # the already-selected panel too.  This retains the explicit count
        # contract while preventing same-stratum clones.
        local = []
        covered = Counter()
        target = plan["minimum_course_witnesses"]
        for rank in range(quota):
            choices = []
            for c in pool:
                if c in local:
                    continue
                # Candidate screening used only the protected evaluation bank.
                # Apply the same continuous/exact clone contract against the
                # already selected training courses before accepting a row.
                if selected:
                    candidate_track = c["track"]
                    if final_selection_exact.distance(candidate_track) < .12:
                        continue
                    if final_selection_cont.distance(resampled(candidate_track)) < .06:
                        continue
                cells = set(c["coarse_cells"])
                deficit_gain = sum(max(0, target.get(x, 0) - covered[x]) for x in cells)
                novelty = min(c["continuous_distance"], c["exact_distance"])
                # Slightly favor high technical/long courses without turning
                # policy outcomes into a selection criterion.
                objective = 2.0 * deficit_gain + 1.5 * novelty + .02 * len(cells)
                choices.append((objective, c))
            choices.sort(key=lambda x: (-x[0], x[1]["name"]))
            if not choices:
                raise RuntimeError(f"selection exhausted {label}")
            c = choices[0][1]; local.append(c); covered.update(c["coarse_cells"])
            selection_ledger.append(dict(rank=len(selected), label=label, name=c["name"], objective=float(choices[0][0]), coarse_cells=c["coarse_cells"], exact_distance=c["exact_distance"], continuous_distance=c["continuous_distance"]))
            selected.append(c)
            final_selection_exact.add(c["track"])
            final_selection_cont.add(resampled(c["track"]))
    if len(selected) != 50:
        raise AssertionError(len(selected))

    # Final all-panel novelty check, independent of candidate screening.
    final_exact = CloneIndex(protected)
    final_cont = CloneIndex([resampled(t) for t in protected])
    records = []
    track_dir = OUT / "tracks"
    for rank, c in enumerate(selected):
        t = c["track"]
        ex = final_exact.distance(t); cont = final_cont.distance(resampled(t))
        if ex < .12 or cont < .06:
            raise RuntimeError(f"final clone threshold failed {t.name}: {ex:.4f}/{cont:.4f}")
        path = save_track_yaml(t, track_dir / f"{t.name}.yaml")
        # Absolute container paths avoid the procedural-manifest loader
        # interpreting a repository-root-relative path relative to the
        # manifest directory (which would duplicate ``outputs/v622...``).
        rel = "/workspace/" + str(path.relative_to(ROOT))
        records.append({
            "name": t.name, "path": rel, "split": "train", "family": c["family"],
            "suite": "pretraining50",
            "source": "v622-pretraining-behavior-program", "seed": c["seed"],
            "rank": rank, "label": c["label"], "geometry_fingerprint": geometry_fingerprint(t),
            "track_fingerprint": t.fingerprint, "fingerprint": geometry_fingerprint(t),
            "cells": c["cells"], "coarse_cells": c["coarse_cells"],
            "geometry": c["geometry"], "behavior_signature": c["behavior_signature"],
            "minimum_protected_clone_distance": float(cont),
            "qualified_speed_mps": None, "qualification": None,
        })
        final_exact.add(t); final_cont.add(resampled(t))

    # Supported by the normal procedural manifest loader used by DAgger.
    manifest = {
        "schema": "starscream-procedural-track-manifest-v1",
        "seed": 8222022, "records": records,
        "family_weights": {x: 1.0 for x in sorted(set(r["family"] for r in records))},
        "source": "v6.22 behavior-cell pretraining panel; MPCC admission pending",
        "cell_plan": "outputs/v622-pretraining50/cell-plan.json",
        "admission": "run scripts/audits/qualify_v622_pretraining50.py before training",
    }
    atomic_json(OUT / "manifest.json", manifest)
    suite = {
        "schema": "starscream-real-course-suite-v1",
        "description": "v6.22 50-course behavior-cell pretraining panel; not an evaluation benchmark",
        "active": [{"name": r["name"], "track": r["path"], "geometry_fingerprint": r["geometry_fingerprint"], "exposure": "fresh training geometry", "family": r["family"], "label": r["label"]} for r in records],
    }
    suite_path = ROOT / "configs/eval/v6_22_pretraining50.yaml"
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
    # Course-level coverage and a stable manifest hash make the cell decision
    # auditable without re-running the generator.
    counts = Counter(cell for r in records for cell in r["coarse_cells"])
    req_counts = Counter(cell for r in records for cell in r["cells"])
    atomic_json(OUT / "selection.json", {"records": selection_ledger, "candidate_count": len(candidates), "selected_count": len(records), "quota_by_mode": QUOTAS})
    atomic_json(OUT / "coverage.json", {"course_count": len(records), "coarse_cell_course_witnesses": dict(sorted(counts.items())), "requirement_cell_course_witnesses": dict(sorted(req_counts.items())), "technical_course_count": sum(r["label"].startswith("technical_") for r in records), "protected_clone_thresholds": {"exact": .12, "continuous": .06}, "manifest_sha256": hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest()})
    print(json.dumps({"courses": len(records), "candidates": len(candidates), "technical": sum(r["label"].startswith("technical_") for r in records), "suite": str(suite_path), "manifest": str(OUT / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
