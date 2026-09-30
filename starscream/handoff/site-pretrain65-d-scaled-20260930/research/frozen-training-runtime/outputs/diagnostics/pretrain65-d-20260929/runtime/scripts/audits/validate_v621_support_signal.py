"""Retrospectively test whether v6.21 support additions predict real50 gains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from starscream.env.racing_manifold.atlas import ManifoldAtlas
from starscream.env.racing_manifold.descriptors import analyze_track_geometry
from starscream.env.racing_manifold.transition_cells import transition_cells
from starscream.env.real_course_suite import load_active_real_course_suite
from starscream.env.tracks import load_track


def mean(rows, key):
    return float(np.mean([row[key] for row in rows])) if rows else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-eval", type=Path, required=True)
    ap.add_argument("--new-eval", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    old_manifest = json.loads(Path("outputs/diagnostics/v6201-teacher-speed/frozen/manifest.json").read_text())
    new_manifest = json.loads(Path("outputs/course-pools/v621-additions/manifest.json").read_text())
    old_paths = [r["path"] for r in old_manifest["records"] if r["split"] == "train"]
    new_paths = [r["path"] for r in new_manifest["records"] if r["split"] == "train"]
    old_names = {Path(p).stem for p in old_paths}
    additions = [p for p in new_paths if Path(p).stem not in old_names]
    if len(old_paths) != 35 or len(new_paths) != 45 or len(additions) != 10:
        raise ValueError("unexpected v6.20.1/v6.21 roster sizes")
    old_tracks = [load_track(p) for p in old_paths]
    new_tracks = [load_track(p) for p in new_paths]
    old_profiles = [analyze_track_geometry(t) for t in old_tracks]
    new_profiles = [analyze_track_geometry(t) for t in new_tracks]
    suite_paths, _ = load_active_real_course_suite("configs/eval/v6_21_holdout_eval_50.yaml")
    suite_tracks = [load_track(p) for p in suite_paths]
    suite_profiles = [analyze_track_geometry(t) for t in suite_tracks]
    atlas = ManifoldAtlas(old_profiles + suite_profiles)
    old_metrics = json.loads(a.old_eval.read_text())["metrics"]
    new_metrics = json.loads(a.new_eval.read_text())["metrics"]
    old_cells = set().union(*(transition_cells(t) for t in old_tracks))
    new_cells = set().union(*(transition_cells(t) for t in new_tracks))
    added_cells = new_cells - old_cells

    courses = []
    transitions = []
    for track, profile in zip(suite_tracks, suite_profiles):
        prefix = f"track/{track.name}/"
        old_chain = atlas.transition_chain_support_profiles(profile, old_profiles, horizon=3)
        new_chain = atlas.transition_chain_support_profiles(profile, new_profiles, horizon=3)
        course = dict(
            name=track.name,
            old_success=old_metrics[prefix + "full_course_success"],
            new_success=new_metrics[prefix + "full_course_success"],
            success_delta=new_metrics[prefix + "full_course_success"] - old_metrics[prefix + "full_course_success"],
            old_chain3_p90=old_chain["p90_distance"], new_chain3_p90=new_chain["p90_distance"],
            chain3_reduction=old_chain["p90_distance"] - new_chain["p90_distance"],
            added_cell_count=len(transition_cells(track) & added_cells),
        )
        courses.append(course)
        count = len(track.gates)
        old_p = [old_metrics[prefix + f"p{i}"] for i in range(1, count + 1)]
        new_p = [new_metrics[prefix + f"p{i}"] for i in range(1, count + 1)]
        for index in range(1, count):
            old_survival = old_p[index] / old_p[index - 1] if old_p[index - 1] else None
            new_survival = new_p[index] / new_p[index - 1] if new_p[index - 1] else None
            if old_survival is None or new_survival is None:
                continue
            transitions.append(dict(
                course=track.name, gate=index,
                old_survival=old_survival, new_survival=new_survival,
                survival_delta=new_survival - old_survival,
                old_chain3=old_chain["per_start_gate_distance"][index],
                new_chain3=new_chain["per_start_gate_distance"][index],
                chain3_reduction=old_chain["per_start_gate_distance"][index] - new_chain["per_start_gate_distance"][index],
            ))
    course_reduction = np.asarray([r["chain3_reduction"] for r in courses])
    course_gain = np.asarray([r["success_delta"] for r in courses])
    transition_reduction = np.asarray([r["chain3_reduction"] for r in transitions])
    transition_gain = np.asarray([r["survival_delta"] for r in transitions])
    threshold = float(np.quantile(transition_reduction, .75))
    high = [r for r in transitions if r["chain3_reduction"] >= threshold and r["chain3_reduction"] > 0]
    none = [r for r in transitions if r["chain3_reduction"] <= 1e-12]
    with_cells = [r for r in courses if r["added_cell_count"] > 0]
    without_cells = [r for r in courses if r["added_cell_count"] == 0]
    summary = dict(
        old_courses=len(old_tracks), new_courses=len(new_tracks), additions=len(additions),
        real50_success_delta=mean(courses, "success_delta"),
        course_chain_reduction_success_correlation=float(np.corrcoef(course_reduction, course_gain)[0, 1]),
        transition_chain_reduction_survival_correlation=float(np.corrcoef(transition_reduction, transition_gain)[0, 1]),
        high_reduction_transition_count=len(high), high_reduction_mean_survival_delta=mean(high, "survival_delta"),
        zero_reduction_transition_count=len(none), zero_reduction_mean_survival_delta=mean(none, "survival_delta"),
        courses_with_new_cells=len(with_cells), courses_with_new_cells_success_delta=mean(with_cells, "success_delta"),
        courses_without_new_cells=len(without_cells), courses_without_new_cells_success_delta=mean(without_cells, "success_delta"),
        caveat="Retrospective association on one model/data change; validates routing signal, not causal sufficiency.",
    )
    out = dict(summary=summary, additions=[Path(p).stem for p in additions], courses=courses,
               transitions=transitions)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
