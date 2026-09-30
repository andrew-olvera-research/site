#!/usr/bin/env python3
"""Qualify MPCC under the exact randomized privileged DAgger reset contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from train_privileged_racing import (
    RoutedDaggerTeacher,
    configured_dagger_speed_fractions,
    dagger_start_perturbation_stage,
    load_config,
    make_env,
    manifest_track_values,
    parse_stage,
    ppo_ctbr_to_normalized,
    ppo_normalized_to_ctbr,
    reset_env,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--curriculum", choices=("train", "validation"), default="validation")
    parser.add_argument(
        "--tracks", type=Path, nargs="+", default=(),
        help="Optional diagnostic track override; bypasses the curriculum manifest.",
    )
    parser.add_argument("--episodes-per-track", type=int, default=8)
    parser.add_argument(
        "--repeats-per-start", type=int, default=0,
        help=(
            "When positive, run this many episodes from every selected start gate; "
            "otherwise --episodes-per-track is distributed across starts."
        ),
    )
    parser.add_argument(
        "--start-mode", choices=("stage", "canonical", "all"), default="stage",
        help="Use configured starts, force gate zero, or cycle every physical gate.",
    )
    parser.add_argument(
        "--start-perturbation-scale", type=float, default=1.0,
        help="Apply the DART reset-basin scale to DART episodes.",
    )
    parser.add_argument(
        "--dart-action-noise-scale", type=float, default=0.0,
        help="Execute noisy teacher actions using the configured normalized DART std.",
    )
    parser.add_argument(
        "--dart-episode-fraction", type=float, default=0.0,
        help="Deterministic fraction of repeats at each start that execute DART noise.",
    )
    parser.add_argument(
        "--speed-fractions", default="config",
        help="Comma-separated frontier fractions or 'config' for the DAgger setting.",
    )
    parser.add_argument(
        "--frontier-speed", type=float, default=None,
        help="Override every manifest frontier speed without changing the source manifest.",
    )
    parser.add_argument("--minimum-track-success", type=float, default=0.95)
    parser.add_argument("--maximum-track-solver-failure", type=float, default=0.01)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Optional qualification timeout override for every curriculum episode.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026082801)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_track(
    settings: dict[str, Any], raw_stage: dict[str, Any], track: str,
    track_index: int, track_count: int, episodes: int, seed_base: int,
    *, start_mode: str, repeats_per_start: int,
    start_perturbation_scale: float, dart_action_noise_scale: float,
    dart_episode_fraction: float, speed_fractions: tuple[float, ...],
    frontier_speed: float,
    backend_cache: list[Any] | None = None,
) -> dict[str, Any]:
    stage = replace(parse_stage(raw_stage), tracks=(track,))
    env = make_env(settings, track=track)
    cached = backend_cache or [None, None]
    expert = RoutedDaggerTeacher(env, settings, worker_index=track_index,
                                fast_backend=cached[0], recovery_backend=cached[1])
    if backend_cache is not None:
        backend_cache[:] = expert.backend_instances
    rows: list[dict[str, Any]] = []
    fallback_by_gate = np.zeros(stage.target_gates, np.int64)
    query_by_gate = np.zeros(stage.target_gates, np.int64)
    dart_noise_std = np.asarray(
        settings.get(
            "dagger_dart_action_noise_std_normalized", (0.0, 0.0, 0.0, 0.0)
        ),
        np.float32,
    )
    if dart_noise_std.shape != (4,) or np.any(dart_noise_std < 0.0):
        raise ValueError("configured DART action noise must contain four non-negative values")
    if not 0.0 <= dart_episode_fraction <= 1.0:
        raise ValueError("DART episode fraction must be in [0, 1]")
    if start_perturbation_scale < 1.0:
        raise ValueError("start perturbation scale must be >= 1")
    physical_starts = tuple(
        index for index, gate in enumerate(env.track.gates) if gate.render
    ) or tuple(range(len(env.track.gates)))
    selected_starts: tuple[int | None, ...]
    if start_mode == "stage":
        selected_starts = (None,)
    elif start_mode == "canonical":
        selected_starts = (0,)
    else:
        selected_starts = tuple(int(index) for index in physical_starts)
    if repeats_per_start > 0:
        episode_plan = [
            (start, repeat)
            for start in selected_starts for repeat in range(repeats_per_start)
        ]
    else:
        episode_plan = [
            (selected_starts[repeat % len(selected_starts)], repeat // len(selected_starts))
            for repeat in range(episodes)
        ]
    try:
        for repeat, (start_gate, start_repeat) in enumerate(episode_plan):
            episode_index = track_index + repeat * track_count
            seed = seed_base + 1009 * episode_index
            dart_repeats = max(
                int(round(
                    (repeats_per_start if repeats_per_start > 0 else episodes)
                    * dart_episode_fraction
                )),
                int(dart_episode_fraction > 0.0),
            )
            dart_active = bool(
                dart_action_noise_scale > 0.0 and start_repeat < dart_repeats
            )
            reset_stage = (
                dagger_start_perturbation_stage(stage, start_perturbation_scale)
                if dart_active and start_perturbation_scale > 1.0 else stage
            )
            if start_gate is not None:
                reset_stage = replace(
                    reset_stage, random_gate=False,
                    fixed_start_gate_index=int(start_gate),
                )
            observation, start_passed = reset_env(
                env, reset_stage, seed=seed, episode_index=episode_index
            )
            speed_fraction = speed_fractions[repeat % len(speed_fractions)]
            speed_command = float(frontier_speed * speed_fraction)
            expert.set_nominal_speed(speed_command)
            expert.reset()
            target_gates = min(stage.target_gates, len(env.track.gates))
            steps = failures = recovery = routed_recovery = fast_failures = 0
            solve_times: list[float] = []
            minimum_margin = float("inf")
            maximum_speed = collective_sum = maximum_collective = 0.0
            crashed = False
            while steps < stage.max_steps:
                gate = min(env.tracker.passed_count - start_passed, target_gates - 1)
                command = expert(observation)
                action = np.asarray(command.action.as_array(), np.float32)
                solver_failed = int(command.solver_status) != 0
                mode = int(np.asarray(command.diagnostics.get("mode", 0)).item())
                query_by_gate[gate] += 1
                fallback_by_gate[gate] += int(solver_failed)
                failures += int(solver_failed)
                recovery += int(mode != 0)
                routed_recovery += int(expert.last_routed)
                fast_failures += int(expert.last_fast_solver_failed)
                solve_times.append(float(command.solve_time))
                minimum_margin = min(minimum_margin, float(command.constraint_margin))
                collective_sum += float(action[0])
                maximum_collective = max(maximum_collective, float(action[0]))
                executed_action = action
                if dart_active and np.any(dart_noise_std > 0.0):
                    rng = np.random.default_rng(seed ^ 0xDA47A11 ^ steps)
                    normalized = ppo_ctbr_to_normalized(action, settings)
                    noisy = np.clip(
                        normalized + rng.normal(
                            0.0, dart_noise_std * dart_action_noise_scale
                        ),
                        -1.0, 1.0,
                    ).astype(np.float32)
                    executed_action = ppo_normalized_to_ctbr(noisy, settings)
                expert.observe_executed_action(executed_action)
                observation, _, terminated, _, info = env.step(executed_action)
                steps += 1
                maximum_speed = max(
                    maximum_speed,
                    float(np.linalg.norm(np.asarray(observation["state"])[7:10])),
                )
                gates = env.tracker.passed_count - start_passed
                crashed = bool(info.get("ground_contact") or info.get("unity_collision"))
                if terminated or crashed or gates >= target_gates:
                    break
            gates = env.tracker.passed_count - start_passed
            success = bool(gates >= target_gates and not crashed)
            rows.append({
                "episode_index": episode_index,
                "seed": seed,
                "start_gate_index": (
                    int(start_gate) if start_gate is not None
                    else int(reset_stage.fixed_start_gate_index or 0)
                ),
                "dart": dart_active,
                "speed_fraction": float(speed_fraction),
                "speed_command_mps": speed_command,
                "success": success,
                "crashed": crashed,
                "gates": int(gates),
                "target_gates": int(target_gates),
                "steps": steps,
                "lap_time_seconds": (
                    steps / float(settings.get("control_hz", 90)) if success else None
                ),
                "solver_failure_fraction": failures / max(steps, 1),
                "recovery_fraction": recovery / max(steps, 1),
                "routed_recovery_fraction": routed_recovery / max(steps, 1),
                "fast_solver_failure_fraction": fast_failures / max(steps, 1),
                "minimum_constraint_margin": minimum_margin,
                "solve_time_p95_ms": 1000.0 * float(np.quantile(solve_times, 0.95)),
                "mean_collective_mps2": collective_sum / max(steps, 1),
                "maximum_collective_mps2": maximum_collective,
                "maximum_speed_mps": maximum_speed,
            })
    finally:
        env.close()
    successful_laps = [
        float(item["lap_time_seconds"]) for item in rows
        if item["lap_time_seconds"] is not None
    ]
    total_steps = sum(int(item["steps"]) for item in rows)
    weighted = lambda key: sum(
        float(item[key]) * int(item["steps"]) for item in rows
    ) / max(total_steps, 1)
    start_summaries = {}
    for start in sorted({int(item["start_gate_index"]) for item in rows}):
        members = [item for item in rows if int(item["start_gate_index"]) == start]
        start_summaries[str(start)] = {
            "episodes": len(members),
            "success_rate": float(np.mean([item["success"] for item in members])),
            "dart_success_rate": (
                float(np.mean([item["success"] for item in members if item["dart"]]))
                if any(item["dart"] for item in members) else None
            ),
            "clean_success_rate": (
                float(np.mean([item["success"] for item in members if not item["dart"]]))
                if any(not item["dart"] for item in members) else None
            ),
        }
    return {
        "track": Path(track).stem,
        "track_path": track,
        "episodes": rows,
        "summary": {
            "success_rate": float(np.mean([item["success"] for item in rows])),
            "crash_rate": float(np.mean([item["crashed"] for item in rows])),
            "mean_gates": float(np.mean([item["gates"] for item in rows])),
            "successful_lap_time_seconds": (
                float(np.mean(successful_laps)) if successful_laps else None
            ),
            "solver_failure_fraction": weighted("solver_failure_fraction"),
            "recovery_fraction": weighted("recovery_fraction"),
            "routed_recovery_fraction": weighted("routed_recovery_fraction"),
            "fast_solver_failure_fraction": weighted(
                "fast_solver_failure_fraction"
            ),
            "minimum_constraint_margin": min(
                float(item["minimum_constraint_margin"]) for item in rows
            ),
            "fallback_fraction_by_gate": (
                fallback_by_gate / np.maximum(query_by_gate, 1)
            ).tolist(),
            "queries_by_gate": query_by_gate.tolist(),
            "by_start_gate": start_summaries,
        },
    }


def main() -> None:
    args = arguments()
    settings = dict(load_config(args.config)["dagger"])
    raw_stage = dict(
        settings["curriculum"] if args.curriculum == "train"
        else settings["evaluation_curriculum"]
    )
    if args.tracks:
        raw_stage.pop("track_manifest", None)
        raw_stage.pop("real_course_suite", None)
        raw_stage["tracks"] = tuple(str(path.resolve()) for path in args.tracks)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("max steps must be positive")
        raw_stage["max_steps"] = int(args.max_steps)
    stage = parse_stage(raw_stage)
    if args.speed_fractions == "config":
        speed_fractions = configured_dagger_speed_fractions(settings)
    else:
        speed_fractions = tuple(
            float(item) for item in args.speed_fractions.split(",") if item.strip()
        )
    if not speed_fractions or any(
        not np.isfinite(item) or item <= 0.0 for item in speed_fractions
    ):
        raise ValueError("speed fractions must be finite and positive")
    default_speed = float(settings.get(
        "dagger_speed_command", settings.get("mpcc_nominal_speed", stage.target_speed)
    ))
    track_speeds: dict[str, float] = {}
    if (
        bool(settings.get("dagger_condition_on_teacher_speed", False))
        and bool(settings.get("mpcc_use_manifest_speed", False))
        and settings.get("track_manifest") is not None
    ):
        track_speeds = manifest_track_values(
            str(settings["track_manifest"]), "qualified_speed_mps", cast=float,
        )
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(stage.tracks))) as executor:
        futures = [
            executor.submit(
                audit_track, settings, raw_stage, track, index, len(stage.tracks),
                args.episodes_per_track, args.seed,
                start_mode=args.start_mode,
                repeats_per_start=args.repeats_per_start,
                start_perturbation_scale=args.start_perturbation_scale,
                dart_action_noise_scale=args.dart_action_noise_scale,
                dart_episode_fraction=args.dart_episode_fraction,
                speed_fractions=speed_fractions,
                frontier_speed=(
                    float(args.frontier_speed) if args.frontier_speed is not None
                    else track_speeds.get(str(Path(track).resolve()), default_speed)
                ),
            )
            for index, track in enumerate(stage.tracks)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            summary = result["summary"]
            print(
                f"track={result['track']} success={summary['success_rate']:.3f} "
                f"lap={summary['successful_lap_time_seconds']} "
                f"fallback={summary['solver_failure_fraction']:.4f} "
                f"recovery={summary['recovery_fraction']:.4f} "
                f"routed={summary['routed_recovery_fraction']:.4f}", flush=True,
            )
    results.sort(key=lambda item: item["track"])
    summaries = [item["summary"] for item in results]
    aggregate = {
        "minimum_track_success_rate": min(item["success_rate"] for item in summaries),
        "mean_track_success_rate": float(np.mean([item["success_rate"] for item in summaries])),
        "maximum_track_solver_failure_fraction": max(
            item["solver_failure_fraction"] for item in summaries
        ),
        "mean_track_solver_failure_fraction": float(np.mean([
            item["solver_failure_fraction"] for item in summaries
        ])),
    }
    passed = bool(
        aggregate["minimum_track_success_rate"] >= args.minimum_track_success
        and aggregate["maximum_track_solver_failure_fraction"]
        <= args.maximum_track_solver_failure
    )
    payload = {
        "schema": "starscream-dagger-teacher-label-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "curriculum": args.curriculum,
        "episodes_per_track": args.episodes_per_track,
        "repeats_per_start": args.repeats_per_start,
        "start_mode": args.start_mode,
        "start_perturbation_scale": args.start_perturbation_scale,
        "dart_action_noise_scale": args.dart_action_noise_scale,
        "dart_episode_fraction": args.dart_episode_fraction,
        "speed_fractions": list(speed_fractions),
        "frontier_speed_override": args.frontier_speed,
        "max_steps_override": args.max_steps,
        "track_override": [str(path.resolve()) for path in args.tracks],
        "seed": args.seed,
        "tracks": results,
        "aggregate": aggregate,
        "promotion": {
            "minimum_track_success_required": args.minimum_track_success,
            "maximum_solver_failure_fraction_required": (
                args.maximum_track_solver_failure
            ),
            "passed": passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(aggregate, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
