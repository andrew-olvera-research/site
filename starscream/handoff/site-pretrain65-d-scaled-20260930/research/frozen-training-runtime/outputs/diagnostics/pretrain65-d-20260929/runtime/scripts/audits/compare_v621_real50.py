"""Compare matched real50 reports by source and real venue, with exact counts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def wilson(successes, count):
    p, z = successes / count, 1.959963984540054
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - half, center + half]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=Path("outputs/course-pools/v621-holdout-eval/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {"baseline": json.loads(args.baseline.read_text()),
               "candidate": json.loads(args.candidate.read_text())}
    if args.reference:
        reports["rl5_reference"] = json.loads(args.reference.read_text())
    baseline = reports["baseline"]
    for report in reports.values():
        for key in ("seed", "episodes", "matched_track_seeds", "max_steps", "nominal_environment"):
            if report[key] != baseline[key]:
                raise ValueError(f"incompatible evaluation {key}")
        if set(report["tracks"]) != set(baseline["tracks"]):
            raise ValueError("incompatible course roster")
    if not baseline["matched_track_seeds"]:
        raise ValueError("requires matched per-track evaluation")
    records = json.loads(args.manifest.read_text())["records"]
    if {r["name"] for r in records} != set(baseline["tracks"]):
        raise ValueError("manifest differs from evaluated roster")
    count = int(baseline["episodes"])
    courses = []
    for record in records:
        row = {k: record[k] for k in ("name", "source", "gate_count")}
        prefix = f"track/{row['name']}/"
        for label, report in reports.items():
            m = report["metrics"]
            assert m[prefix + "episodes"] == count
            successes = round(m[prefix + "full_course_success"] * count)
            row[label] = dict(success=successes / count, successes=successes,
                              wilson95=wilson(successes, count),
                              gate_fraction=m[prefix + "mean_gates"] / row["gate_count"],
                              crash=m[prefix + "crash_rate"])
        row["delta"] = row["candidate"]["success"] - row["baseline"]["success"]
        courses.append(row)
    groups = {}
    rng = np.random.default_rng(2044091462)
    for group in ["all", *sorted({r["source"] for r in records})]:
        rows = [r for r in courses if group == "all" or r["source"] == group]
        summary = {"courses": len(rows), "episodes": len(rows) * count}
        for label in reports:
            summary[label] = {k: float(np.mean([r[label][k] for r in rows]))
                              for k in ("success", "gate_fraction", "crash")}
            summary[label]["successes"] = sum(r[label]["successes"] for r in rows)
        deltas = np.array([r["delta"] for r in rows])
        summary["delta"] = float(deltas.mean())
        draws = rng.choice(deltas, (20000, len(rows)), replace=True).mean(axis=1)
        summary["paired_course_bootstrap95"] = np.quantile(draws, [0.025, 0.975]).tolist()
        groups[group] = summary
        print(group, json.dumps(summary), flush=True)
    result = dict(
        baseline=str(args.baseline), candidate=str(args.candidate),
        candidate_steps=reports["candidate"]["checkpoint_environment_steps"],
        seed=baseline["seed"], episodes_per_course=count, groups=groups, courses=courses,
        uncertainty_note="Bootstrap resamples paired course deltas; descriptive geometry heterogeneity, not an episode-paired significance test.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in courses:
        if row["source"] == "real":
            print(row["name"], json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
