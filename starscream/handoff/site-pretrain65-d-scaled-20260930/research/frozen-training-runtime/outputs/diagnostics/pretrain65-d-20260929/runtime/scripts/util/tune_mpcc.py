#!/usr/bin/env python3
"""Reproducible Bayesian tuning of MPCC weights and speed/safety settings."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from skopt import Optimizer
from skopt.space import Real

from starscream.env import FlightmareEnv
from starscream.env.tracks import load_track
from starscream.mpcc import MPCCConfig, MPCCController, MPCCWeights, RacingLinePlanner, RacingLinePlannerConfig
from starscream.mpcc.benchmark import BenchmarkRunner, build_scenarios


SPACE = [
    Real(10.0, 100.0, prior="log-uniform", name="contour"),
    Real(0.5, 15.0, prior="log-uniform", name="lag"),
    Real(0.5, 15.0, prior="log-uniform", name="progress"),
    Real(0.2, 5.0, prior="log-uniform", name="velocity"),
    Real(0.01, 0.5, prior="log-uniform", name="action_smoothness"),
    Real(2.0e3, 5.0e4, prior="log-uniform", name="slack"),
    Real(7.0, 18.0, name="nominal_speed"),
    Real(0.10, 0.35, name="corridor_margin"),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", default="figure8,split_s")
    parser.add_argument("--suite", choices=("smoke", "nominal", "recovery"), default="smoke")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("/data/mpcc-tuning"))
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def candidate_config(values: list[float]) -> MPCCConfig:
    contour, lag, progress, velocity, smoothness, slack, speed, margin = values
    weights = replace(
        MPCCWeights(),
        contour=contour,
        lag=lag,
        progress=progress,
        velocity=velocity,
        action_smoothness=smoothness,
        slack=slack,
    )
    return MPCCConfig(
        backend="acados",
        weights=weights,
        nominal_speed=speed,
        track_speed_overrides=(),
        corridor_margin=margin,
    )


def score_summary(summary: dict[str, float | int | str], max_steps: int) -> float:
    failure = 1.0 - float(summary["success_rate"])
    solver_failure = float(summary["solver_failure_rate"])
    margin = float(summary["minimum_constraint_margin"])
    time_penalty = max(0.0, float(summary["solve_time_p99_ms_worst"]) - 8.0)
    # Crashes/timeouts dominate speed; an unsafe faster candidate never wins.
    return (
        10000.0 * failure
        + 2500.0 * solver_failure
        + 500.0 * max(0.0, -margin)
        + 100.0 * time_penalty
        + max_steps * failure
    )


def main() -> None:
    arguments = parse_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    trials_path = arguments.output / "trials.jsonl"
    optimizer = Optimizer(SPACE, base_estimator="GP", acq_func="gp_hedge", random_state=arguments.random_state)
    tracks = [load_track(name.strip()) for name in arguments.tracks.split(",") if name.strip()]
    seeds = [int(seed) for seed in arguments.seeds.split(",") if seed]
    best: dict | None = None
    for trial_index in range(arguments.trials):
        values = optimizer.ask()
        config = candidate_config(values)
        summaries = []
        for track in tracks:
            line = RacingLinePlanner(
                RacingLinePlannerConfig(
                    offset_iterations=30,
                    cache_directory=str(arguments.output / "racing-lines"),
                )
            ).plan(track)
            env = FlightmareEnv(
                track=track.name,
                control_dt=1.0 / 90.0,
                action_delay=0.0,
                render_observations=False,
            )
            try:
                controller = MPCCController(
                    track,
                    line,
                    config=config,
                    build_directory=arguments.output / "solvers" / config.fingerprint[:16],
                )
                scenarios = build_scenarios(
                    track,
                    line,
                    arguments.suite,
                    seeds=seeds,
                    maximum_steps=arguments.max_steps,
                )
                report = BenchmarkRunner(
                    env,
                    controller,
                    output_directory=arguments.output / "benchmarks" / f"trial-{trial_index:04d}",
                ).run(scenarios)
                summaries.append(report.summary)
            finally:
                env.close()
        objective = float(np.mean([score_summary(summary, arguments.max_steps) for summary in summaries]))
        optimizer.tell(values, objective)
        record = {
            "trial": trial_index,
            "objective": objective,
            "parameters": dict(zip([dimension.name for dimension in SPACE], values, strict=True)),
            "config": asdict(config),
            "summaries": summaries,
        }
        with trials_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if best is None or objective < best["objective"]:
            best = record
            (arguments.output / "best.json").write_text(
                json.dumps(best, indent=2, sort_keys=True), encoding="utf-8"
            )
        print(json.dumps({"trial": trial_index, "objective": objective, "best": best["objective"]}))


if __name__ == "__main__":
    main()
