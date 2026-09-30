#!/usr/bin/env python3
"""Paired command probe on qualified real100-hard-v2 courses.

The manifest's speed_command is the accepted MPCC profile command, not a
certificate of time optimality. Courses without a qualified command are
excluded before random sampling and listed in provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "configs/eval/v6_22_real100_hard_v2.yaml"
MANIFEST = ROOT / "configs/eval/v6_22_real100_hard_v2.manifest.json"
CONFIG = ROOT / "configs/exp/v6.21.1/update_fix_dagger.yaml"
CHECKPOINT = ROOT / "outputs/checkpoints/starscream-v6.21.1.update-fix-pretrain65-dagger/best-step-121343545-full_course_success-0.63181818.pt"


def prepare(out: Path, *, seed: int, size: int, extra_size: int,
            checkpoint: Path) -> dict:
    records = json.loads(MANIFEST.read_text())["records"]
    suite = yaml.safe_load(SUITE.read_text())
    active = {row["name"]: row for row in suite["active"]}
    eligible = []
    excluded = []
    for row in records:
        qualification = row["qualification"]
        command = qualification.get("speed_command")
        if (qualification.get("qualified") is True and command is not None
                and float(command) > 0 and row["name"] in active
                and qualification.get("fingerprint") == active[row["name"]]["geometry_fingerprint"]):
            eligible.append(row)
        else:
            excluded.append(row["name"])
    if len(eligible) < size:
        raise ValueError(f"only {len(eligible)} courses have qualified commands")
    rng = random.Random(seed)
    selected = rng.sample(eligible, size)
    # Fix the extra-command cohort before seeing policy results. The raised
    # values intentionally test command extrapolation beyond some DAgger speeds.
    extra_candidates = [row for row in selected if float(row["qualification"]["speed_command"]) <= 20]
    extra = rng.sample(extra_candidates, min(extra_size, len(extra_candidates)))
    out.mkdir(parents=True, exist_ok=True)
    def write_arm(name: str, rows: list[dict], commands: dict[str, float]) -> dict:
        suite_data = {**suite, "active": [active[row["name"]] for row in rows]}
        suite_path = out / f"{name}-suite.yaml"
        suite_path.write_text(yaml.safe_dump(suite_data, sort_keys=False))
        config = yaml.safe_load(CONFIG.read_text())
        settings = config["dagger"]
        settings["evaluation_condition_on_manifest_speed"] = False
        settings["evaluation_track_speed_commands"] = {
            str(ROOT / row["path"]): float(commands[row["name"]]) for row in rows
        }
        config_path = out / f"{name}-config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        return {"name": name, "suite": str(suite_path), "config": str(config_path),
                "commands": commands, "courses": [row["name"] for row in rows]}
    arms = [
        write_arm("fixed16", selected, {row["name"]: 16.0 for row in selected}),
        write_arm("qualified", selected, {row["name"]: float(row["qualification"]["speed_command"]) for row in selected}),
        write_arm("raised", extra, {row["name"]: min(24.0, 1.2 * float(row["qualification"]["speed_command"])) for row in extra}),
    ]
    provenance = {
        "seed": seed, "selection": "uniform without replacement among frozen qualified commands",
        "benchmark_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint), "source_config": str(CONFIG),
        "qualification_meaning": "accepted MPCC profile command; not a time-optimal speed",
        "eligible_count": len(eligible), "excluded_names": excluded,
        "episodes_per_course": 8, "episode_seed": 2034091462,
        "randomized_environment": True, "max_steps": 6000, "arms": arms,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def run(provenance: dict, out: Path, workers: int, selected_arms: set[str]) -> None:
    for arm in provenance["arms"]:
        if arm["name"] not in selected_arms:
            continue
        output = out / f"{arm['name']}.json"
        if output.exists():
            continue
        command = [sys.executable, "-u", "scripts/eval_privileged_dagger.py",
                   "--config", arm["config"], "--checkpoint", provenance["checkpoint"],
                   "--section", "dagger", "--curriculum", "validation",
                   "--track-suite", arm["suite"], "--episodes", "8",
                   "--matched-track-seeds", "--randomized-environment",
                   "--max-steps", "6000", "--seed", "2034091462",
                   "--workers", str(workers), "--device", "cuda", "--output", str(output)]
        with (out / f"{arm['name']}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log,
                           stderr=subprocess.STDOUT, env={**os.environ,
                           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                           "OPENBLAS_NUM_THREADS": "1"})
        print(f"finished {arm['name']}: {output}", flush=True)


def summarize(provenance: dict, out: Path) -> None:
    rows = {}
    for arm in provenance["arms"]:
        path = out / f"{arm['name']}.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())["metrics"]
        for name in arm["courses"]:
            observed = metrics.get(f"track/{name}/target_speed_mps")
            if observed is None or abs(float(observed) - arm["commands"][name]) > 1e-5:
                raise ValueError(f"{arm['name']}/{name}: requested {arm['commands'][name]} but observed {observed}")
        rows[arm["name"]] = {
            "courses": len(arm["courses"]),
            "completion": metrics["full_course_success"],
            "crash": metrics["crash_rate"],
            "mean_gate_speed_mps": metrics["mean_gate_speed_mps"],
            "per_course": {name: {
                "command": arm["commands"][name],
                "completion": metrics.get(f"track/{name}/full_course_success"),
                "crash": metrics.get(f"track/{name}/crash_rate"),
                "gate_speed_mps": metrics.get(f"track/{name}/mean_gate_speed_mps"),
                "successful_median_lap_seconds": (
                    None if metrics.get(f"track/{name}/successful_median_steps") is None
                    else metrics[f"track/{name}/successful_median_steps"] / 130.0),
            } for name in arm["courses"]},
        }
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in row.items() if k != "per_course"}
                      for name, row in rows.items()}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/evals/heldout-speed-command-probe")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--courses", type=int, default=40)
    parser.add_argument("--extra-courses", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--arms", nargs="+", choices=("fixed16", "qualified", "raised"),
                        default=("fixed16", "qualified", "raised"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (out / "provenance.json").exists():
        provenance = json.loads((out / "provenance.json").read_text())
        if (provenance["seed"] != args.seed or len(provenance["arms"][0]["courses"]) != args.courses):
            raise ValueError("existing cohort differs from requested seed or size")
        if Path(provenance["checkpoint"]).resolve() != checkpoint:
            raise ValueError("existing cohort uses a different checkpoint")
    else:
        provenance = prepare(out, seed=args.seed, size=args.courses,
                             extra_size=args.extra_courses, checkpoint=checkpoint)
    if not args.prepare_only:
        run(provenance, out, args.workers, set(args.arms))
        summarize(provenance, out)


if __name__ == "__main__":
    main()
