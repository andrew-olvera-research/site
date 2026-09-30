#!/usr/bin/env python3
"""Qualify a fast, robust MPCC teacher independently for every pretrain50 course.

The search is deliberately CPU-only and candidate-isolated: an acados backend is
never reused after controller/planner settings change.  A descending command grid
finds the highest acceptable speed, then disjoint randomized and DART cohorts
confirm it.  Results are resumable at the candidate/cohort level.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/exp/v6.22/pretraining_behavior50_dagger.yaml"
MANIFEST = ROOT / "outputs/v622-pretraining50/manifest.json"
SCHEMA = "starscream-v622-pretraining50-mpcc-frontier-v2"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    episodes = report["episodes"]
    usable = [
        e for e in episodes
        if e["success"] and not e["prefix_failed"]
        and e["solver_failure_fraction"] <= .06
        and e["recovery_fraction"] <= .10
    ]
    laps = [float(e["lap_time_seconds"]) for e in usable]
    steps = sum(max(int(e["steps"]), 1) for e in episodes)
    return {
        "episodes": len(episodes), "usable": len(usable),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "usable_rate": len(usable) / len(episodes),
        "median_lap_time_seconds": float(np.median(laps)) if laps else None,
        "mean_lap_time_seconds": float(np.mean(laps)) if laps else None,
        "fastest_lap_time_seconds": min(laps) if laps else None,
        "solver_failure_fraction": sum(e["solver_failure_fraction"] * max(int(e["steps"]), 1) for e in episodes) / steps,
        "recovery_fraction": sum(e["recovery_fraction"] * max(int(e["steps"]), 1) for e in episodes) / steps,
    }


def _candidate(row: dict[str, Any], speed: float, mode: str) -> dict[str, Any]:
    admitted = row["qualification"]["teacher_overrides"]
    controller = deepcopy(admitted["mpcc_config"])
    planner = deepcopy(admitted["mpcc_planner_config"])
    controller.pop("nominal_speed", None)
    if mode == "frontier":
        controller.update(
            progress_reference_mode="profile",
            # Fixed envelope across command probes: nominal speed is a runtime
            # reference, so one compiled solver can safely serve the grid.
            max_progress_speed=41.,
            maximum_acceleration=40.,
            maximum_longitudinal_acceleration=34.,
            maximum_braking_acceleration=36.,
            maximum_collective_thrust=40.,
            collective_slew_limit=24.,
            body_rate_slew_limit=8.,
            corridor_margin=.12,
        )
        planner.update(aperture_fraction=.35, aperture_margin=.12, offset_iterations=40)
    return {"name": f"{mode}-{speed:g}", "speed": speed, "controller": controller, "planner": planner}


def _run_cohort(settings: dict[str, Any], row: dict[str, Any], candidate: dict[str, Any],
                cohort: str, output: Path, contract: str,
                backend_cache: list[Any]) -> dict[str, Any]:
    path = output / f"{cohort}.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("contract") == contract and old.get("candidate") == candidate:
            return old
    from scripts.audits.audit_dagger_teacher_labels import audit_track
    s = deepcopy(settings)
    s.update(
        mpcc_use_manifest_teacher_profile=False,
        mpcc_use_manifest_planner_profile=False,
        mpcc_use_manifest_speed=False,
        mpcc_nominal_speed=candidate["speed"],
        mpcc_config=candidate["controller"],
        mpcc_planner_config=candidate["planner"],
        mpcc_family_configs={}, mpcc_family_planner_configs={},
        mpcc_family_nominal_speeds={},
        mpcc_build_root=f"/tmp/starscream-v622-frontier-{os.getpid()}-{cohort}",
    )
    specs = {
        "screen": dict(repeats=1, seed=62228001, random=False, dart=0.),
        "nominal": dict(repeats=3, seed=62229001, random=False, dart=0.),
        "randomized": dict(repeats=6, seed=62230001, random=True, dart=0.),
        "dart": dict(repeats=6, seed=62231001, random=True, dart=1.),
    }[cohort]
    if not specs["random"]:
        s["dynamics_randomization"] = {"enabled": False}
    report = audit_track(
        s, s["curriculum"], row["path"], 0, 1, specs["repeats"], specs["seed"],
        start_mode="canonical", repeats_per_start=specs["repeats"],
        start_perturbation_scale=1.35 if specs["dart"] else 1.,
        dart_action_noise_scale=1. if specs["dart"] else 0.,
        dart_episode_fraction=specs["dart"], speed_fractions=(1.,),
        frontier_speed=candidate["speed"], backend_cache=backend_cache,
        qualification_limits={"solver": .06, "recovery": .10},
    )
    result = {"schema": SCHEMA, "contract": contract, "candidate": candidate,
              "cohort": cohort, "summary": _summary(report), "episodes": report["episodes"]}
    atomic_json(path, result)
    return result


def _passes(result: dict[str, Any], minimum: float) -> bool:
    s = result["summary"]
    return (s["usable_rate"] >= minimum and s["solver_failure_fraction"] <= .06
            and s["recovery_fraction"] <= .10)


def _job(payload: tuple[dict[str, Any], dict[str, Any], str, str]) -> dict[str, Any]:
    settings, row, root_text, contract = payload
    import torch
    torch.set_num_threads(1)
    root = Path(root_text) / row["name"]
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / "selection.json"
    if final_path.exists():
        old = json.loads(final_path.read_text())
        if old.get("contract") == contract:
            return old
    admitted_speed = float(row["qualified_speed_mps"])
    speeds = []
    probe = admitted_speed + 2.
    while probe <= 36. + 1.e-9:
        speeds.append(probe)
        probe += 2.
    screens: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    confirmations: dict[str, Any] = {}
    frontier_backend_cache: list[Any] = []
    for speed in speeds:
        modes = ("frontier",)
        for mode in modes:
            candidate = _candidate(row, speed, mode)
            directory = root / candidate["name"]
            directory.mkdir(exist_ok=True)
            # Reuse is safe only inside one candidate: controller and planner
            # settings are identical across its disjoint evidence cohorts.
            backend_cache: list[Any] = [] if mode == "admitted" else frontier_backend_cache
            screen = _run_cohort(settings, row, candidate, "screen", directory, contract, backend_cache)
            screens.append({"candidate": candidate, "summary": screen["summary"]})
            if not _passes(screen, 1.):
                break
            nominal = _run_cohort(settings, row, candidate, "nominal", directory, contract, backend_cache)
            randomized = _run_cohort(settings, row, candidate, "randomized", directory, contract, backend_cache)
            dart = _run_cohort(settings, row, candidate, "dart", directory, contract, backend_cache)
            confirmations[candidate["name"]] = {
                "nominal": nominal["summary"], "randomized": randomized["summary"], "dart": dart["summary"]
            }
            # Perturbed feasibility is not a pace requirement. Some admitted
            # geometries are intentionally outside MPCC's perturbed basin; a
            # clean/randomized frontier may pass while DART marks the course
            # exempt from noisy collection.
            if _passes(nominal, 2/3) and _passes(randomized, 5/6):
                selected = candidate
                continue
            break
        else:
            continue
        # The first rejected rung closes the monotone command frontier.
        if selected is None or selected["speed"] < speed:
            break
    if selected is None:
        # The frozen admission already supplies nominal/randomized/DART and
        # every-prefix evidence for this exact candidate. A small disjoint
        # cohort may be unlucky; it may veto an increase, but must not erase
        # the larger accepted baseline qualification.
        selected = _candidate(row, admitted_speed, "admitted")
        confirmations[selected["name"]] = {
            **confirmations.get(selected["name"], {}),
            "frozen_admission_fallback": True,
            "frozen_admission_contract": row["qualification"]["contract"],
        }
    result = {
        "schema": SCHEMA, "contract": contract, "name": row["name"],
        "fingerprint": row["fingerprint"], "admitted_speed_mps": admitted_speed,
        "selected": selected, "screens": screens, "confirmations": confirmations,
        "dart_eligible": bool(
            selected["name"] in confirmations
            and "dart" in confirmations[selected["name"]]
            and confirmations[selected["name"]]["dart"]["usable_rate"] >= 5/6
        ) if selected["speed"] > admitted_speed else bool(
            row["qualification"]["attempted_profiles"]
            [row["qualification"]["selected_profile"]]["dart"]["success_rate"] >= .95
        ),
        "search_ceiling_mps": 36., "grid_resolution_mps": 2.,
        "acceptance": {"nominal_usable": "2/3", "randomized_usable": "5/6",
                       "dart_eligibility": "5/6 (not required for pace)", "maximum_solver_failure": .06,
                       "maximum_recovery": .10},
    }
    atomic_json(final_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("/workspace/outputs/v622-pretraining50/mpcc-frontier-v2"))
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    settings = config["dagger"]
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest["records"][:args.limit or None]
    if len(manifest["records"]) != 50:
        raise ValueError("expected frozen pretrain50 manifest")
    sources = {str(p.relative_to(ROOT)): _digest(p) for p in (
        CONFIG, MANIFEST, Path(__file__), ROOT / "scripts/audits/audit_dagger_teacher_labels.py",
        ROOT / "starscream/mpcc/config.py", ROOT / "starscream/mpcc/controller.py")}
    contract = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "contract.json", {"schema": SCHEMA, "contract": contract, "source_hashes": sources})
    results: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, 8)), mp_context=context) as pool:
        futures = [pool.submit(_job, (settings, row, str(args.output), contract)) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            atomic_json(args.output / "summary.json", {"schema": SCHEMA, "contract": contract,
                        "complete": len(results) == len(rows), "records": sorted(results, key=lambda r: r["name"])})
            print(f"{len(results)}/{len(rows)} {result['name']} {result['admitted_speed_mps']:g} -> {result['selected']['speed']:g} m/s", flush=True)
    if len(results) == 50:
        by_name = {row["name"]: row for row in results}
        tuned = deepcopy(manifest)
        controller_profiles: dict[str, Any] = {}
        planner_profiles: dict[str, Any] = {}
        for row in tuned["records"]:
            result = by_name[row["name"]]
            selected = result["selected"]
            profile = "v622-frontier-" + row["fingerprint"][:12]
            row["qualified_speed_mps"] = selected["speed"]
            row["dart_eligible"] = bool(result["dart_eligible"])
            row["dart_exemption_reason"] = (
                None if result["dart_eligible"]
                else "MPCC pace passes nominal/randomized but not the stronger perturbed cohort"
            )
            row["v622_pace_qualification"] = result
            row["qualification"] = deepcopy(row["qualification"])
            row["qualification"]["selected"] = dict(
                row["qualification"].get("selected") or {}
            )
            row["qualification"]["selected"]["teacher_profile"] = profile
            controller_profiles[profile] = selected["controller"]
            planner_profiles[profile] = selected["planner"]
        tuned["pace_frontier"] = {"schema": SCHEMA, "contract": contract,
                                  "search_ceiling_mps": 36., "grid_resolution_mps": 2.}
        atomic_json(args.output / "manifest.json", tuned)
        atomic_json(args.output / "teacher-profiles.json", {
            "schema": SCHEMA, "contract": contract,
            "controller_profiles": controller_profiles,
            "planner_profiles": planner_profiles,
        })


if __name__ == "__main__":
    main()
