#!/usr/bin/env python3
"""Publish the completed v6.22 MPCC frontier without rerunning evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/v622-pretraining50/manifest.json"
DIRECTORY = ROOT / "outputs/v622-pretraining50/mpcc-frontier-v3"


def main() -> None:
    summary = json.loads((DIRECTORY / "summary.json").read_text())
    if not summary.get("complete") or len(summary.get("records", ())) != 50:
        raise ValueError("MPCC frontier is incomplete")
    manifest = json.loads(SOURCE.read_text())
    by_name = {row["name"]: row for row in summary["records"]}
    stronger_dart_path = ROOT / "outputs/v622-pretraining50/mpcc-frontier-v2/summary.json"
    stronger_dart = {
        row["name"]: bool(row["dart_eligible"])
        for row in json.loads(stronger_dart_path.read_text())["records"]
    }
    controllers, planners = {}, {}
    for row in manifest["records"]:
        result = by_name[row["name"]]
        selected = result["selected"]
        profile = "v622-frontier-" + row["fingerprint"][:12]
        row["qualified_speed_mps"] = selected["speed"]
        row["dart_eligible"] = bool(
            result["dart_eligible"]
            if selected["speed"] > result["admitted_speed_mps"]
            else stronger_dart[row["name"]]
        )
        row["dart_exemption_reason"] = None if result["dart_eligible"] else (
            "MPCC pace passes nominal/randomized but not the stronger perturbed cohort"
        )
        row["v622_pace_qualification"] = result
        row["qualification"] = deepcopy(row["qualification"])
        row["qualification"]["selected"] = {"teacher_profile": profile}
        controllers[profile] = selected["controller"]
        planners[profile] = selected["planner"]
    manifest["pace_frontier"] = {
        "schema": summary["schema"], "contract": summary["contract"],
        "search_ceiling_mps": 36., "grid_resolution_mps": 2.,
    }
    atomic_json(DIRECTORY / "manifest.json", manifest)
    atomic_json(DIRECTORY / "teacher-profiles.json", {
        "schema": summary["schema"], "contract": summary["contract"],
        "controller_profiles": controllers, "planner_profiles": planners,
    })
    atomic_json(DIRECTORY / "frontier-report.json", {
        "schema": "starscream-v622-pretraining50-frontier-report-v1",
        "contract": summary["contract"],
        "courses": [{
            "name": row["name"],
            "maximum_acceptable_speed_mps": row["qualified_speed_mps"],
            "dart_eligible": row["dart_eligible"],
            "teacher_profile": row["qualification"]["selected"]["teacher_profile"],
        } for row in manifest["records"]],
    })
    print(f"published 50 course frontiers; DART eligible={sum(r['dart_eligible'] for r in manifest['records'])}")


if __name__ == "__main__":
    main()
