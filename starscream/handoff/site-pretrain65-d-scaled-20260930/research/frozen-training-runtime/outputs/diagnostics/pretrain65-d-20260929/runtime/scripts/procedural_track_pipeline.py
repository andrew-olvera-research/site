#!/usr/bin/env python3
"""Generate, qualify, collect, and audit procedural racing courses.

This script is intentionally stage-oriented so the shell launcher can stop on
the first failed acceptance gate.  Qualification and collection use isolated
processes because Flightmare and ACADOS are CPU-heavy and not thread-safe.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from starscream.dataset import validate_episode
from starscream.env import FlightmareEnv
from starscream.env.dynamics_randomization import DynamicsRandomizationConfig
from starscream.env.collect import (
    EpisodeCollector, episode_outcome_metadata, evaluate_champion_episode,
    collection_episode_seed, save_episode_hdf5,
)
from starscream.env.championship_tracks import (
    generate_championship_manifest, repair_unqualified_championship_manifest,
)
from starscream.env.procedural_tracks import (
    ProceduralTrackConfig, audit_manifest, generate_manifest, read_manifest,
    update_manifest_records, write_manifest_gallery,
)
from starscream.env.primitive_tracks import (
    generate_primitive_manifest, repair_unqualified_primitive_manifest,
)
from starscream.env.reference_tracks import generate_reference_manifest
from starscream.env.recipe_tracks import (
    generate_recipe_manifest, repair_unqualified_recipe_manifest,
)
from starscream.env.tracks import load_track, matrix_quaternion
from starscream.mpcc import (
    MPCCConfig, MPCCController, RacingLinePlanner, RacingLinePlannerConfig,
)


DEFAULT_MANIFEST_ROOT = Path("/workspace/outputs/procedural-tracks/v1")
DEFAULT_DATASET_ROOT = Path("/data/flightmare/procedural-privileged-v1")
DEFAULT_SPEEDS = (9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_ROOT)
    generate.add_argument("--train-tracks", type=int, default=36)
    generate.add_argument("--validation-tracks", type=int, default=12)
    generate.add_argument("--seed", type=int, default=2026081801)
    generate.add_argument(
        "--style", choices=(
            "curves-v1", "primitives-v2", "championship-v3", "reference-v1",
            "recipe-v1",
        ),
        default="curves-v1"
    )
    generate.add_argument("--references", type=Path, nargs="+", default=())
    generate.add_argument(
        "--train-per-family", type=int, default=3,
        help="recipe-v1 only: accepted training geometries per transition family",
    )
    generate.add_argument(
        "--validation-per-family", type=int, default=1,
        help="recipe-v1 only: held-out geometries per transition family",
    )

    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_ROOT / "manifest.json")
    qualify.add_argument("--workers", type=int, default=6)
    qualify.add_argument(
        "--scenarios", type=int, default=3,
        help=(
            "Minimum scenarios per speed. all/explicit start modes always run "
            "at least one scenario for every selected physical gate."
        ),
    )
    qualify.add_argument(
        "--start-gates", default="",
        help="Optional comma-separated gate indices cycled across scenarios.",
    )
    qualify.add_argument(
        "--start-mode", choices=("all", "canonical", "explicit"), default="all",
        help=(
            "Qualification reset contract: canonical gate 0, every physical gate, "
            "or the explicit --start-gates list."
        ),
    )
    qualify.add_argument(
        "--perturb-starts", action="store_true",
        help="Qualify repeated perturbations of each selected start state.",
    )
    qualify.add_argument("--speeds", default=",".join(str(item) for item in DEFAULT_SPEEDS))
    qualify.add_argument("--backend", choices=("acados", "predictive"), default="acados")
    qualify.add_argument("--max-steps", type=int, default=1800)
    qualify.add_argument("--minimum-body-clearance", type=float, default=0.12)
    qualify.add_argument("--maximum-recovery-fraction", type=float, default=0.05)
    qualify.add_argument(
        "--maximum-solver-failure-fraction", type=float, default=None,
        help=(
            "Override the teacher-profile solver-failure allowance. A value of "
            "zero requires every qualification query to solve cleanly."
        ),
    )
    qualify.add_argument(
        "--maximum-gate-transition-progress-jump", type=float, default=0.5,
    )
    qualify.add_argument(
        "--teacher-profile", choices=(
            "default", "swift-fast-v1", "swift-frontier-v1", "a2rl-fast-v1",
            "swift-centered-frontier-v2", "swift-centered-frontier-v3",
            "a2rl-center-fast-v2",
            "a2rl-centered-stable-v3", "real-course-fast-v1",
        ), default="default",
    )
    qualify.add_argument(
        "--nominal-dynamics", action="store_true",
        help="Qualify geometry on the nominal plant; DAgger may still randomize dynamics.",
    )
    qualify.add_argument(
        "--qualification-regime", default="",
        help=(
            "Persist this pass under qualification_regimes and conservatively "
            "merge every completed regime into the legacy qualification fields."
        ),
    )
    qualify.add_argument("--resume", action="store_true")
    qualify.add_argument("--allow-insufficient", action="store_true")
    qualify.add_argument("--limit", type=int, default=0)
    qualify.add_argument(
        "--splits", default="",
        help="Optional comma-separated manifest splits to qualify.",
    )
    qualify.add_argument(
        "--families", default="",
        help="Optional comma-separated manifest families to qualify.",
    )
    qualify.add_argument(
        "--dagger-eligible-only", action="store_true",
        help="Exclude records not explicitly admitted to five-inch DAgger.",
    )
    qualify.add_argument(
        "--leaderboard-eligible-only", action="store_true",
        help="Exclude geometry that is not eligible for exact timing comparisons.",
    )

    repair = subparsers.add_parser("repair-unqualified")
    repair.add_argument(
        "--manifest", type=Path,
        default=Path("/workspace/outputs/procedural-tracks/primitives-v2/manifest.json"),
    )
    repair.add_argument("--maximum-attempts", type=int, default=180)
    repair.add_argument("--output", type=Path, default=None)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_ROOT / "manifest.json")
    collect.add_argument("--output", type=Path, default=DEFAULT_DATASET_ROOT)
    collect.add_argument("--workers", type=int, default=8)
    collect.add_argument("--train-episodes", type=int, default=3)
    collect.add_argument("--validation-episodes", type=int, default=1)
    collect.add_argument(
        "--train-nominal-episodes", type=int, default=0,
        help="Additional nominal-plant episodes per training track.",
    )
    collect.add_argument(
        "--validation-nominal-episodes", type=int, default=0,
        help="Additional nominal-plant episodes per validation track.",
    )
    collect.add_argument(
        "--splits", default="",
        help="Optional comma-separated manifest splits to collect.",
    )
    collect.add_argument("--route-gates", type=int, default=3)
    collect.add_argument(
        "--no-manifest-update", action="store_true",
        help="Keep a frozen manifest byte-identical after collection.",
    )
    collect.add_argument("--backend", choices=("acados", "predictive"), default="acados")
    collect.add_argument("--max-steps", type=int, default=1800)
    collect.add_argument(
        "--maximum-attempts-per-domain", type=int, default=0,
        help=(
            "Maximum rollout attempts for each dynamics domain. Zero keeps the "
            "legacy adaptive budget max(12, 8 * requested episodes)."
        ),
    )
    collect.add_argument("--minimum-body-clearance", type=float, default=0.12)
    collect.add_argument("--maximum-solver-failure-fraction", type=float, default=0.0)
    collect.add_argument("--mask-source", choices=("geometry", "none"), default="geometry")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_ROOT / "manifest.json")
    audit.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_ROOT)
    audit.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_ROOT / "dataset-audit.json")
    audit.add_argument("--train-episodes", type=int, default=3)
    audit.add_argument("--validation-episodes", type=int, default=1)
    audit.add_argument("--train-nominal-episodes", type=int, default=0)
    audit.add_argument("--validation-nominal-episodes", type=int, default=0)
    audit.add_argument(
        "--splits", default="",
        help="Optional comma-separated manifest splits expected in the dataset.",
    )
    audit.add_argument("--route-gates", type=int, default=3)
    audit.add_argument("--maximum-solver-failure-fraction", type=float, default=0.0)
    audit.add_argument("--require-masks", action="store_true")
    return parser.parse_args()


def _track_path(manifest_path: Path, record: Mapping[str, Any]) -> Path:
    return (manifest_path.parent / str(record["path"])).resolve()


def generate(args: argparse.Namespace) -> None:
    if args.style == "recipe-v1":
        path = generate_recipe_manifest(
            args.output,
            train_per_family=args.train_per_family,
            validation_per_family=args.validation_per_family,
            seed=args.seed,
        )
    elif args.style == "reference-v1":
        if not args.references:
            raise ValueError("reference-v1 generation requires --references")
        path = generate_reference_manifest(
            args.output, references=[str(path) for path in args.references],
            train_count=args.train_tracks, validation_count=args.validation_tracks,
            seed=args.seed,
        )
    elif args.style == "championship-v3":
        path = generate_championship_manifest(
            args.output, train_count=args.train_tracks,
            validation_count=args.validation_tracks, seed=args.seed,
        )
    elif args.style == "primitives-v2":
        path = generate_primitive_manifest(
            args.output, train_count=args.train_tracks,
            validation_count=args.validation_tracks, seed=args.seed,
        )
    else:
        path = generate_manifest(
            args.output, train_count=args.train_tracks,
            validation_count=args.validation_tracks, seed=args.seed,
        )
    print(f"generated manifest={path} style={args.style}", flush=True)


def _planner(track: Any, cache: Path, teacher_profile: str = "default") -> Any:
    values: dict[str, Any] = {"offset_iterations": 30, "cache_directory": str(cache)}
    if teacher_profile in {
        "swift-fast-v1", "swift-frontier-v1", "a2rl-fast-v1",
        "swift-centered-frontier-v2", "swift-centered-frontier-v3",
        "a2rl-center-fast-v2",
        "a2rl-centered-stable-v3", "real-course-fast-v1",
    }:
        values.update(sample_count=1400, aperture_fraction=0.20, aperture_margin=0.18)
    if teacher_profile in {"real-course-fast-v1", "swift-centered-frontier-v2"}:
        # The A2RL stacked/Split-S sequence exposed that a 20% gate-aperture
        # racing-line offset can be center-feasible yet clip the frame once the
        # vehicle body is included. Qualification therefore targets gate
        # centers; PPO may later discover safe aperture cuts from this manifold.
        values.update(aperture_fraction=0.10, aperture_margin=0.22)
    if teacher_profile == "swift-centered-frontier-v3":
        values.update(aperture_fraction=0.0, aperture_margin=0.30)
    if teacher_profile in {"a2rl-center-fast-v2", "a2rl-centered-stable-v3"}:
        # A2RL's stacked/Split-S sequence is body-clearance limited. The
        # generic aperture-cut racing line repeatedly completes the gate order
        # while clipping a frame. Center the line here; speed optimization is
        # still handled by MPCC and later PPO, not unsafe geometric shortcuts.
        values.update(aperture_fraction=0.0, aperture_margin=0.30)
    return RacingLinePlanner(RacingLinePlannerConfig(**values)).plan(track)


def _qualification_mpcc_config(
    profile: str, speed: float, backend: str,
) -> MPCCConfig:
    values: dict[str, Any] = {
        "backend": backend, "nominal_speed": float(speed), "actuation_delay": 0.011,
    }
    if profile in {
        "swift-fast-v1", "swift-frontier-v1", "a2rl-fast-v1",
        "swift-centered-frontier-v2", "swift-centered-frontier-v3",
        "a2rl-center-fast-v2",
        "a2rl-centered-stable-v3", "real-course-fast-v1",
    }:
        values.update(
            progress_reference_mode="profile", max_progress_speed=24.0,
            maximum_acceleration=24.0,
            maximum_longitudinal_acceleration=16.0,
            maximum_braking_acceleration=18.0,
            maximum_collective_thrust=40.0, collective_slew_limit=10.0,
            body_rate_slew_limit=4.0, corridor_margin=0.20,
        )
    if profile in {
        "swift-frontier-v1", "swift-centered-frontier-v2",
        "swift-centered-frontier-v3",
    } and speed >= 16.0:
        if speed >= 18.0:
            frontier = (27.0, 30.0, 22.0, 24.0, 14.0, 5.0, 0.17)
        elif speed >= 17.0:
            frontier = (26.0, 28.0, 20.0, 22.0, 13.0, 4.8, 0.18)
        else:
            frontier = (25.0, 26.0, 18.0, 20.0, 12.0, 4.5, 0.19)
        (max_progress, acceleration, longitudinal, braking,
         collective_slew, body_rate_slew, corridor) = frontier
        values.update(
            max_progress_speed=max_progress,
            maximum_acceleration=acceleration,
            maximum_longitudinal_acceleration=longitudinal,
            maximum_braking_acceleration=braking,
            collective_slew_limit=collective_slew,
            body_rate_slew_limit=body_rate_slew,
            corridor_margin=corridor,
        )
    if profile == "a2rl-centered-stable-v3":
        # Keep the 15 m/s A2RL command aggressive while reducing the QP
        # discontinuities caused by asking the stacked sequence to track the
        # Swift controller's 24 m/s progress frontier.
        values.update(
            max_progress_speed=21.0,
            maximum_acceleration=22.0,
            maximum_longitudinal_acceleration=15.0,
            maximum_braking_acceleration=18.0,
            collective_slew_limit=10.0,
            body_rate_slew_limit=4.0,
            corridor_margin=0.22,
        )
    return MPCCConfig(**values)


def _initial_state(
    line: Any,
    gate_index: int,
    *,
    seed: int,
    perturb: bool,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    approach = float(rng.uniform(2.8, 3.4)) if perturb else 3.0
    progress = float(line.gate_progress[int(gate_index)] - approach)
    frame = line.evaluate(progress)
    # The high-speed Swift frontier has a narrow cold-start basin.  Use the
    # repeatable race-entry state for geometry qualification; online DAgger
    # is responsible for broad state perturbations after lap establishment.
    lateral = (
        float(rng.uniform(-0.10, 0.10)) if perturb
        else float(np.random.default_rng(20260823).uniform(-0.10, 0.10))
    )
    vertical = float(rng.uniform(-0.06, 0.06)) if perturb else 0.0
    state = np.zeros(25, np.float32)
    state[0:3] = (
        frame["position"] + lateral * frame["lateral"] + vertical * frame["up"]
    )
    state[3:7] = matrix_quaternion(np.stack(
        [frame["tangent"], frame["lateral"], frame["up"]], axis=1
    ))
    initial_speed = float(rng.uniform(2.8, 3.8)) if perturb else 3.0
    state[7:10] = initial_speed * frame["tangent"]
    if perturb:
        state[10:13] = rng.uniform(-0.08, 0.08, size=3)
    return state


def _qualification_start_plan(
    starts: Sequence[int], *, scenarios: int, start_mode: str,
) -> tuple[int, ...]:
    """Build a deterministic plan without silently omitting start gates.

    Historically ``--start-mode all --scenarios 3`` sampled only three gates
    on a seven-gate course even though the CLI promised every physical gate.
    A passing all/explicit qualification now proves complete start coverage.
    Additional scenarios are distributed round-robin across those starts.
    """

    selected = tuple(int(item) for item in starts)
    if not selected or scenarios < 1:
        raise ValueError("qualification needs starts and a positive scenario count")
    count = scenarios if start_mode == "canonical" else max(scenarios, len(selected))
    return tuple(selected[index % len(selected)] for index in range(count))


def _quality(
    episode: dict[str, np.ndarray],
    *,
    track: Any,
    line: Any,
    maximum_lap_time: float | None,
    minimum_body_clearance: float,
    maximum_solver_failure_fraction: float = 0.0,
    maximum_recovery_fraction: float = 0.05,
    maximum_gate_transition_progress_jump: float = 0.5,
) -> tuple[bool, dict[str, Any]]:
    quality = evaluate_champion_episode(
        episode,
        expected_gates=len(track.gates),
        maximum_lap_time=maximum_lap_time,
        maximum_recovery_fraction=maximum_recovery_fraction,
        track=track,
        racing_line_length=line.length,
        maximum_gate_transition_progress_jump=maximum_gate_transition_progress_jump,
    )
    body_clearance = (
        float(quality.geometry_audit.minimum_body_clearance)
        if quality.geometry_audit is not None else float("-inf")
    )
    reasons = list(quality.reasons)
    if float(quality.solver_failure_fraction) <= maximum_solver_failure_fraction:
        reasons = [reason for reason in reasons if reason != "solver-failure"]
    if body_clearance < minimum_body_clearance:
        reasons.append("insufficient-body-clearance")
    accepted = bool(not reasons and body_clearance >= minimum_body_clearance)
    return accepted, {
        "accepted": accepted,
        "reasons": sorted(set(reasons)),
        "transitions": quality.transitions,
        "gates_passed": quality.gates_passed,
        "elapsed_time": quality.elapsed_time,
        "solver_failure_fraction": quality.solver_failure_fraction,
        "recovery_fraction": quality.recovery_fraction,
        "minimum_body_clearance": body_clearance,
        "minimum_center_clearance": (
            float(quality.geometry_audit.minimum_center_clearance)
            if quality.geometry_audit is not None else float("-inf")
        ),
        "maximum_gate_transition_progress_jump": (
            quality.maximum_gate_transition_progress_jump
        ),
    }


def _qualify_job(job: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    manifest_path = Path(str(job["manifest"]))
    record = dict(job["record"])
    track_path = _track_path(manifest_path, record)
    track = load_track(track_path)
    cache_root = manifest_path.parent / "qualification-racing-lines"
    teacher_profile = str(job.get("teacher_profile", "default"))
    line = _planner(track, cache_root, teacher_profile)
    env = FlightmareEnv(
        track=track,
        next_gates=3,
        image_size=(160, 128),
        control_dt=1.0 / 90.0,
        render_observations=False,
        mask_source="none",
        action_delay=0.011,
        terminate_on_collision=True,
        dynamics_randomization=DynamicsRandomizationConfig(
            enabled=not bool(job.get("nominal_dynamics", False))
        ),
    )
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    controller = MPCCController(
        track,
        line,
        config=_qualification_mpcc_config(
            teacher_profile, float(job["speeds"][0]), str(job["backend"]),
        ),
        build_directory=(
            manifest_path.parent / "acados-build" / track.fingerprint[:12]
        ),
    )
    try:
        for speed in job["speeds"]:
            controller.set_nominal_speed(float(speed))
            scenarios: list[dict[str, Any]] = []
            explicit_starts = tuple(int(item) for item in job.get("start_gates", ()))
            physical_starts = tuple(
                index for index, gate in enumerate(track.gates) if gate.render
            )
            start_mode = str(job.get("start_mode", "all"))
            if start_mode == "canonical":
                starts = (0,)
            elif start_mode == "explicit":
                if not explicit_starts:
                    raise ValueError("explicit qualification requires --start-gates")
                starts = explicit_starts
            else:
                starts = physical_starts or tuple(range(len(track.gates)))
            if any(index < 0 or index >= len(track.gates) for index in starts):
                raise ValueError(
                    f"invalid qualification start gates {starts} for {track.name}"
                )
            start_plan = _qualification_start_plan(
                starts, scenarios=int(job["scenarios"]), start_mode=start_mode,
            )
            for scenario, gate_index in enumerate(start_plan):
                controller.reset()
                episode = EpisodeCollector(
                    env,
                    controller,
                    metadata={
                        "procedural_qualification": 1,
                        "procedural_family": record["family"],
                        "mpcc_nominal_speed_mps": float(speed),
                    },
                ).collect(
                    int(job["max_steps"]),
                    reset_options={
                        "gate_index": gate_index,
                        "state": _initial_state(
                            line, gate_index,
                            seed=int(record["seed"]) + 1009 * scenario,
                            perturb=bool(job.get("perturb_starts", False)),
                        ),
                    },
                    seed=int(record["seed"]) + 1009 * scenario,
                    stop_after_gates=len(track.gates),
                )
                # The ceiling prevents a high nominal speed that internally
                # stalls from being called near-time-optimal.
                ideal_time = float(line.length) / float(speed)
                time_ceiling = max(1.75 * ideal_time, ideal_time + 2.5)
                accepted, report = _quality(
                    episode,
                    track=track,
                    line=line,
                    maximum_lap_time=time_ceiling,
                    minimum_body_clearance=float(job["minimum_body_clearance"]),
                    maximum_solver_failure_fraction=float(
                        job.get("maximum_solver_failure_fraction", 0.0)
                    ),
                    maximum_recovery_fraction=float(
                        job.get("maximum_recovery_fraction", 0.05)
                    ),
                    maximum_gate_transition_progress_jump=float(
                        job.get("maximum_gate_transition_progress_jump", 0.5)
                    ),
                )
                report.update(gate_index=gate_index, time_ceiling=time_ceiling)
                scenarios.append(report)
                # A speed is qualified only if every requested start passes.
                # Once one start fails, further starts at this speed cannot
                # change admission and merely waste controller time.
                if not accepted:
                    break
            passed = all(item["accepted"] for item in scenarios)
            attempt = {
                "speed_mps": float(speed),
                "passed": passed,
                "teacher_profile": teacher_profile,
                "start_mode": str(job.get("start_mode", "all")),
                "requested_scenarios": int(job["scenarios"]),
                "planned_scenarios": len(start_plan),
                "perturb_starts": bool(job.get("perturb_starts", False)),
                "qualified_start_gates": sorted({
                    int(item["gate_index"])
                    for item in scenarios if item["accepted"]
                }),
                "scenarios": scenarios,
                "mean_elapsed_time": float(np.mean([
                    item["elapsed_time"] for item in scenarios
                ])),
                "maximum_elapsed_time": max(item["elapsed_time"] for item in scenarios),
                "minimum_body_clearance": min(
                    item["minimum_body_clearance"] for item in scenarios
                ),
            }
            attempts.append(attempt)
            if passed:
                selected = attempt
                break
    finally:
        env.close()
    if selected is None:
        return track.name, {
            "qualified_speed_mps": None,
            "qualification": {
                "passed": False,
                "racing_line_length_m": float(line.length),
                "attempts": attempts,
            },
        }
    return track.name, {
        "qualified_speed_mps": float(selected["speed_mps"]),
        "qualification": {
            "passed": True,
            "method": "highest-all-scenario-safe-speed-grid-v1",
            "racing_line_length_m": float(line.length),
            "selected": selected,
            "attempts": attempts,
        },
    }


def _qualification_regime_patch(
    record: Mapping[str, Any], update: Mapping[str, Any], *,
    regime: str, nominal_dynamics: bool, backend: str,
) -> dict[str, Any]:
    """Merge one plant regime without losing earlier qualification evidence."""

    name = str(regime).strip()
    if not name or any(not (character.isalnum() or character in "-_") for character in name):
        raise ValueError("qualification regime must be a nonempty filename-safe name")
    regimes = dict(record.get("qualification_regimes") or {})
    qualification = dict(update["qualification"])
    speed = update.get("qualified_speed_mps")
    regimes[name] = {
        "backend": str(backend),
        "nominal_dynamics": bool(nominal_dynamics),
        "passed": bool(qualification.get("passed", False) and speed is not None),
        "qualified_speed_mps": None if speed is None else float(speed),
        "qualification": qualification,
    }
    passing = [
        (regime_name, item) for regime_name, item in regimes.items()
        if bool(item.get("passed")) and item.get("qualified_speed_mps") is not None
    ]
    suite_passed = bool(passing and len(passing) == len(regimes))
    selected_name = None
    selected_qualification: dict[str, Any]
    conservative_speed = None
    if suite_passed:
        # A randomized regime wins ties so dataset collection inherits starts
        # shown feasible under the wider plant distribution.
        selected_name, selected_regime = min(
            passing,
            key=lambda pair: (
                float(pair[1]["qualified_speed_mps"]),
                bool(pair[1].get("nominal_dynamics", False)),
                pair[0],
            ),
        )
        conservative_speed = float(selected_regime["qualified_speed_mps"])
        selected_qualification = dict(selected_regime["qualification"])
        selected_qualification["method"] = "qualification-regime-conservative-min-v1"
        selected_qualification["suite_regimes"] = sorted(regimes)
        selected_qualification["suite_selected_regime"] = selected_name
    else:
        selected_qualification = {
            "passed": False,
            "method": "qualification-regime-conservative-min-v1",
            "suite_regimes": sorted(regimes),
            "suite_selected_regime": None,
            "attempts": [],
        }
    return {
        "qualified_speed_mps": conservative_speed,
        "qualification": selected_qualification,
        "qualification_regimes": regimes,
        "qualification_suite": {
            "contract": "nominal-randomized-multistart-mpcc-v1",
            "passed": suite_passed,
            "regimes": sorted(regimes),
            "selected_regime": selected_name,
            "conservative_speed_mps": conservative_speed,
        },
    }


def qualify(args: argparse.Namespace) -> None:
    payload = read_manifest(args.manifest)
    speeds = tuple(float(item) for item in str(args.speeds).split(",") if item.strip())
    if not speeds or any(speed <= 0 for speed in speeds):
        raise ValueError("qualification speeds must be positive")
    # Highest-first is part of the near-time-optimal qualification contract.
    speeds = tuple(sorted(set(speeds), reverse=True))
    requested_splits = {
        item.strip() for item in str(args.splits).split(",") if item.strip()
    }
    requested_families = {
        item.strip() for item in str(args.families).split(",") if item.strip()
    }
    scoped_records = [
        record for record in payload["records"]
        if (not requested_splits or str(record.get("split")) in requested_splits)
        and (
            not requested_families
            or str(record.get("family")) in requested_families
        )
        and (not args.dagger_eligible_only or bool(record.get("dagger_eligible_5inch")))
        and (
            not args.leaderboard_eligible_only
            or bool(record.get("leaderboard_comparison_eligible"))
        )
    ]
    regime_name = str(args.qualification_regime).strip()
    selected_records = [
        record for record in scoped_records
        if not args.resume
        or (
            not bool(dict(record.get("qualification_regimes") or {}).get(
                regime_name, {}
            ).get("passed"))
            if regime_name else record.get("qualified_speed_mps") is None
        )
    ]
    if int(args.limit) > 0:
        selected_records = selected_records[:int(args.limit)]
    start_gates = tuple(
        int(item) for item in str(args.start_gates).split(",") if item.strip()
    )
    jobs = [{
        "manifest": str(args.manifest.resolve()),
        "record": record,
        "speeds": speeds,
        "backend": args.backend,
        "scenarios": args.scenarios,
        "max_steps": args.max_steps,
        "minimum_body_clearance": args.minimum_body_clearance,
        "teacher_profile": args.teacher_profile,
        "start_gates": start_gates,
        "start_mode": str(args.start_mode),
        "perturb_starts": bool(args.perturb_starts),
        "nominal_dynamics": bool(args.nominal_dynamics),
        "maximum_solver_failure_fraction": (
            float(args.maximum_solver_failure_fraction)
            if args.maximum_solver_failure_fraction is not None
            else 0.15 if args.teacher_profile == "real-course-fast-v1"
            else 0.15 if args.teacher_profile == "a2rl-center-fast-v2"
            else 0.10 if args.teacher_profile == "a2rl-fast-v1"
            else 0.05 if args.teacher_profile == "swift-frontier-v1" else 0.0
        ),
        "maximum_recovery_fraction": float(args.maximum_recovery_fraction),
        "maximum_gate_transition_progress_jump": float(
            args.maximum_gate_transition_progress_jump
        ),
    } for record in selected_records]
    updates: dict[str, dict[str, Any]] = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {executor.submit(_qualify_job, job): job for job in jobs}
        for future in as_completed(futures):
            name, update = future.result()
            if regime_name:
                current_payload = read_manifest(args.manifest)
                current = next(
                    record for record in current_payload["records"]
                    if str(record["name"]) == name
                )
                update = _qualification_regime_patch(
                    current, update, regime=regime_name,
                    nominal_dynamics=bool(args.nominal_dynamics),
                    backend=str(args.backend),
                )
            updates[name] = update
            # Qualification can be much slower for rejected high-curvature
            # courses because they exhaust the full speed grid.  Persist each
            # completed result so an interruption can genuinely resume instead
            # of discarding tens of minutes of controller validation.
            update_manifest_records(args.manifest, {name: update})
            selected = update.get("qualified_speed_mps")
            print(f"qualified track={name} speed={selected}", flush=True)
    update_manifest_records(args.manifest, updates)
    final_payload = read_manifest(args.manifest)
    scoped_names = {str(record["name"]) for record in scoped_records}
    records = [
        record for record in final_payload["records"]
        if str(record["name"]) in scoped_names
    ]
    failed = [str(record["name"]) for record in records if record.get("qualified_speed_mps") is None]
    qualified = [record for record in records if record.get("qualified_speed_mps") is not None]
    evaluated_splits = sorted({str(record["split"]) for record in records})
    split_counts = {
        split: {
            "total": sum(record["split"] == split for record in records),
            "qualified": sum(record["split"] == split for record in qualified),
        }
        for split in evaluated_splits
    }
    families_by_split = {
        split: {str(record["family"]) for record in records if record["split"] == split}
        for split in evaluated_splits
    }
    family_coverage = {
        split: sorted({str(record["family"]) for record in qualified if record["split"] == split})
        for split in evaluated_splits
    }
    passed = bool(
        records
        and not failed
        and all(
            counts["qualified"] >= int(np.ceil(0.75 * counts["total"]))
            for counts in split_counts.values()
        )
        and all(
            set(family_coverage[split]) == families_by_split[split]
            for split in evaluated_splits
        )
    )
    summary = {
        "passed": passed,
        "track_count": len(records),
        "qualified_track_count": len(qualified),
        "failed_tracks": failed,
        "split_counts": split_counts,
        "family_coverage": family_coverage,
        "speed_distribution": {
            str(speed): sum(record.get("qualified_speed_mps") == speed for record in records)
            for speed in speeds
        },
    }
    summary["qualification_regime"] = regime_name or None
    destination = args.manifest.parent / (
        f"qualification-summary-{regime_name}.json"
        if regime_name else "qualification-summary.json"
    )
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not passed and not args.allow_insufficient:
        raise RuntimeError(
            f"too few dynamically qualified procedural tracks; rejected={failed}"
        )


def repair_unqualified(args: argparse.Namespace) -> None:
    payload = read_manifest(args.manifest)
    generator = payload.get("generator")
    if generator == "competition-calibrated-v3":
        report = repair_unqualified_championship_manifest(
            args.manifest, maximum_attempts=args.maximum_attempts
        )
    elif generator == "maneuver-primitives-v2":
        report = repair_unqualified_primitive_manifest(
            args.manifest, maximum_attempts=args.maximum_attempts
        )
    elif generator == "physical-gate-recipe-v1":
        report = repair_unqualified_recipe_manifest(
            args.manifest, maximum_attempts=args.maximum_attempts
        )
    else:
        raise ValueError(f"repair is unsupported for generator {generator!r}")
    destination = args.output or args.manifest.parent / "last-repair.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def _existing_episode(
    path: Path, track_name: str, *, randomized_dynamics: bool | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r", swmr=True) as archive:
            validate_episode(archive)
            raw = np.asarray(archive["metadata/track"])[()]
            name = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            domain_matches = (
                randomized_dynamics is None
                or bool(np.asarray(archive["metadata/dynamics_randomization"]))
                == bool(randomized_dynamics)
            )
            return bool(
                name == track_name and domain_matches
                and np.asarray(archive["metadata/quality_accepted"])
            )
    except (OSError, KeyError, ValueError):
        return False


def _collect_job(job: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    manifest_path = Path(str(job["manifest"]))
    output_root = Path(str(job["output"]))
    record = dict(job["record"])
    track_path = _track_path(manifest_path, record)
    track = load_track(track_path)
    speed = float(record["qualified_speed_mps"])
    split = str(record["split"])
    nominal_episodes = int(job.get("nominal_episodes", 0))
    randomized_episodes = int(job.get("randomized_episodes", job.get("episodes", 0)))
    target_episodes = nominal_episodes + randomized_episodes
    track_output = output_root / split / track.name
    track_output.mkdir(parents=True, exist_ok=True)
    expected_files = [
        (track_output / f"expert-{domain}-{index:03d}.h5", randomized)
        for domain, requested, randomized in (
            ("nominal", nominal_episodes, False),
            ("randomized", randomized_episodes, True),
        )
        for index in range(requested)
    ]
    if expected_files and all(
        _existing_episode(path, track.name, randomized_dynamics=randomized)
        for path, randomized in expected_files
    ):
        return track.name, {
            "collection": {
                "passed": True,
                "episodes": target_episodes,
                "episodes_by_dynamics_domain": {
                    "nominal": nominal_episodes,
                    "randomized": randomized_episodes,
                },
                "rejected_attempts": 0,
                "output": str(track_output),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "resumed_without_rollout": True,
            }
        }
    cache_root = manifest_path.parent / "collection-racing-lines"
    selected_qualification = record["qualification"]["selected"]
    teacher_profile = str(selected_qualification.get("teacher_profile", "default"))
    line = _planner(track, cache_root, teacher_profile)
    controller = MPCCController(
        track,
        line,
        config=_qualification_mpcc_config(
            teacher_profile, speed, str(job["backend"]),
        ),
        build_directory=(
            manifest_path.parent / "acados-build" / track.fingerprint[:12]
        ),
    )
    maximum_lap_time = 1.08 * float(selected_qualification["maximum_elapsed_time"])
    qualified_starts = tuple(
        int(item) for item in selected_qualification.get("qualified_start_gates", ())
    )
    if not qualified_starts:
        qualified_starts = tuple(sorted({
            int(item["gate_index"])
            for item in selected_qualification.get("scenarios", ())
            if bool(item.get("accepted", False))
        }))
    if not qualified_starts:
        raise ValueError(f"{track.name} has no dynamically qualified collection start")
    accepted_total = 0
    rejected_total = 0
    accepted_by_domain: dict[str, int] = {}
    for domain, requested, randomized in (
        ("nominal", nominal_episodes, False),
        ("randomized", randomized_episodes, True),
    ):
        if requested <= 0:
            accepted_by_domain[domain] = 0
            continue
        env = FlightmareEnv(
            track=track,
            next_gates=int(job.get("route_gates", 3)),
            image_size=(160, 128),
            control_dt=1.0 / 90.0,
            render_observations=False,
            mask_source=str(job["mask_source"]),
            mask_size=(160, 128),
            retain_render_images=False,
            image_delay=0.033,
            action_delay=0.011,
            terminate_on_collision=True,
            dynamics_randomization=DynamicsRandomizationConfig(enabled=randomized),
        )
        existing_indices = {
            index for index in range(requested)
            if _existing_episode(
                track_output / f"expert-{domain}-{index:03d}.h5", track.name,
                randomized_dynamics=randomized,
            )
        }
        accepted = len(existing_indices)
        rejected = 0
        attempt = 0
        try:
            attempt_budget = int(job.get("maximum_attempts_per_domain", 0))
            if attempt_budget <= 0:
                attempt_budget = max(12, 8 * requested)
            while accepted < requested and attempt < attempt_budget:
                episode_index = next(
                    index for index in range(requested)
                    if index not in existing_indices
                )
                output_path = track_output / f"expert-{domain}-{episode_index:03d}.h5"
                gate_index = qualified_starts[
                    (episode_index + attempt) % len(qualified_starts)
                ]
                episode_seed = collection_episode_seed(
                    track.name, domain, episode_index, attempt + 1,
                )
                controller.reset()
                episode = EpisodeCollector(
                    env,
                    controller,
                    metadata={
                        "collector_policy": f"mpcc-acados-runtime-matched-{domain}",
                        "gatenet_checkpoint_sha256": "not-used-procedural-geometry-mask",
                        "procedural_dataset": "privileged-production-paired-v2",
                        "procedural_family": record["family"],
                        "procedural_split": split,
                        "procedural_geometry_fingerprint": record["geometry_fingerprint"],
                        "collection_dynamics_domain": domain,
                        "route_gate_count": int(job.get("route_gates", 3)),
                        "mpcc_nominal_speed_mps": speed,
                        "mpcc_speed_qualification": "nominal-randomized-multistart-mpcc-v1",
                        "perfect_expert_only": 1,
                        "dynamics_target": "next-task-state-delta",
                        "dynamics_validity": "not-gate-transition",
                    },
                ).collect(
                    int(job["max_steps"]),
                    reset_options={
                        "gate_index": gate_index,
                        "state": _initial_state(
                            line, gate_index, seed=episode_seed,
                            perturb=bool(selected_qualification.get("perturb_starts", True)),
                        ),
                        "spawn": {
                            "source": "procedural-mpcc-perfect",
                            "gate_index": gate_index,
                            "qualified_speed_mps": speed,
                            "dynamics_domain": domain,
                        },
                    },
                    seed=episode_seed,
                    stop_after_gates=len(track.gates),
                )
                episode.update(episode_outcome_metadata(episode))
                passed, report = _quality(
                    episode,
                    track=track,
                    line=line,
                    maximum_lap_time=maximum_lap_time,
                    minimum_body_clearance=float(job["minimum_body_clearance"]),
                    maximum_solver_failure_fraction=float(
                        job.get("maximum_solver_failure_fraction", 0.0)
                    ),
                )
                episode["metadata/quality_accepted"] = np.asarray(passed)
                episode["metadata/quality_elapsed_time"] = np.asarray(
                    report["elapsed_time"], np.float32
                )
                episode["metadata/quality_minimum_body_clearance"] = np.asarray(
                    report["minimum_body_clearance"], np.float32
                )
                episode["metadata/quality_minimum_center_clearance"] = np.asarray(
                    report["minimum_center_clearance"], np.float32
                )
                episode["metadata/quality_maximum_gate_transition_progress_jump"] = np.asarray(
                    report["maximum_gate_transition_progress_jump"], np.float32
                )
                attempt += 1
                if not passed:
                    rejected += 1
                    print(
                        f"rejected track={track.name} domain={domain} attempt={attempt} "
                        f"reasons={','.join(report['reasons'])}",
                        flush=True,
                    )
                    continue
                save_episode_hdf5(output_path, episode)
                existing_indices.add(episode_index)
                accepted += 1
                print(
                    f"saved track={track.name} domain={domain} "
                    f"episode={accepted}/{requested} "
                    f"steps={len(episode['action/ctbr'])} time={report['elapsed_time']:.3f}s",
                    flush=True,
                )
        finally:
            env.close()
        if accepted != requested:
            raise RuntimeError(
                f"could not collect {requested} {domain} perfect episodes for "
                f"{track.name}; accepted={accepted} rejected={rejected}"
            )
        accepted_by_domain[domain] = accepted
        accepted_total += accepted
        rejected_total += rejected
    if accepted_total != target_episodes:
        raise RuntimeError(f"collection accounting mismatch for {track.name}")
    return track.name, {
        "collection": {
            "passed": True,
            "episodes": accepted_total,
            "episodes_by_dynamics_domain": accepted_by_domain,
            "rejected_attempts": rejected_total,
            "output": str(track_output),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
    }


def collect(args: argparse.Namespace) -> None:
    payload = read_manifest(args.manifest)
    requested_splits = {
        item.strip() for item in str(args.splits).split(",") if item.strip()
    }
    qualified = [
        item for item in payload["records"]
        if item.get("qualified_speed_mps") is not None
        and (not requested_splits or str(item.get("split")) in requested_splits)
    ]
    if not qualified:
        raise RuntimeError("collection requires at least one dynamically qualified track")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [{
        "manifest": str(args.manifest.resolve()),
        "output": str(args.output.resolve()),
        "record": record,
        "randomized_episodes": (
            args.train_episodes if record["split"] == "train" else args.validation_episodes
        ),
        "nominal_episodes": (
            args.train_nominal_episodes
            if record["split"] == "train" else args.validation_nominal_episodes
        ),
        "backend": args.backend,
        "max_steps": args.max_steps,
        "minimum_body_clearance": args.minimum_body_clearance,
        "mask_source": args.mask_source,
        "route_gates": args.route_gates,
        "maximum_solver_failure_fraction": args.maximum_solver_failure_fraction,
        "maximum_attempts_per_domain": args.maximum_attempts_per_domain,
    } for record in qualified]
    updates: dict[str, dict[str, Any]] = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {executor.submit(_collect_job, job): job for job in jobs}
        for future in as_completed(futures):
            name, update = future.result()
            updates[name] = update
            print(f"collection-complete track={name}", flush=True)
    if not args.no_manifest_update:
        update_manifest_records(args.manifest, updates)


def _decode(value: Any) -> str:
    raw = np.asarray(value)[()]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def audit_dataset(args: argparse.Namespace) -> None:
    payload = read_manifest(args.manifest)
    if payload.get("engine") == "zero-shot-racing-distribution-v1":
        from starscream.env.racing_distribution import (
            RacingDistributionConfig, RacingDistributionValidator,
        )
        manifest_report = RacingDistributionValidator(
            RacingDistributionConfig(**dict(payload.get("config") or {})),
            cache_directory=args.manifest.parent / "racing-lines",
        ).validate_manifest(args.manifest, require_dynamic=True).to_mapping(
            include_track_reports=False
        )
    else:
        manifest_report = audit_manifest(args.manifest)
    requested_splits = {
        item.strip() for item in str(args.splits).split(",") if item.strip()
    }
    scoped_records = [
        item for item in payload["records"]
        if item.get("qualified_speed_mps") is not None
        and (not requested_splits or str(item.get("split")) in requested_splits)
    ]
    expected_domains = {
        str(item["name"]): {
            "randomized": (
                args.train_episodes
                if item["split"] == "train" else args.validation_episodes
            ),
            "nominal": (
                args.train_nominal_episodes
                if item["split"] == "train" else args.validation_nominal_episodes
            ),
        }
        for item in scoped_records
    }
    expected = {
        name: int(domains["randomized"] + domains["nominal"])
        for name, domains in expected_domains.items()
    }
    required_production = {
        "observation/state_estimate",
        "observation/state_estimate_std",
        "observation/state_estimate_valid",
        "observation/measured/body_rates",
        "observation/measured/motor_omega",
        "observation/flight_plan/records",
        "observation/previous_action",
        "observation/timestamp/sim",
        "observation/age/previous_action",
        "observation/valid/previous_action",
    }
    if args.require_masks:
        required_production.add("observation/gate_mask")
    required_privileged = {
        "observation/task_state",
        "observation/privileged/state",
        "observation/privileged/gate_state",
        "observation/privileged/dynamics",
        "observation/privileged/aerodynamics",
        "observation/privileged/progress",
    }
    required_targets = {"target/task_state_delta", "target/dynamics_valid"}
    errors: list[str] = []
    counts = {name: 0 for name in expected}
    domain_counts = {
        name: {"nominal": 0, "randomized": 0} for name in expected
    }
    transitions = 0
    sizes: list[int] = []
    mask_chunk_rows: list[int] = []
    dynamics_domains: list[np.ndarray] = []
    aerodynamic_domains: list[np.ndarray] = []
    randomized_domain_seeds: set[int] = set()
    solver_failure_fractions: list[float] = []
    for path in sorted(args.dataset.rglob("*.h5")):
        try:
            with h5py.File(path, "r", swmr=True) as archive:
                length = validate_episode(archive)
                track = _decode(archive["metadata/track"])
                if track not in counts:
                    errors.append(f"{path}: unknown track {track}")
                    continue
                counts[track] += 1
                transitions += length
                missing = sorted(
                    (required_production | required_privileged | required_targets)
                    - set(_dataset_paths(archive))
                )
                if missing:
                    errors.append(f"{path}: missing {missing}")
                if not bool(np.asarray(archive["metadata/quality_accepted"])):
                    errors.append(f"{path}: quality rejected")
                randomized = bool(np.asarray(
                    archive["metadata/dynamics_randomization"]
                ))
                domain = "randomized" if randomized else "nominal"
                domain_counts[track][domain] += 1
                route_records = np.asarray(
                    archive["observation/flight_plan/records"]
                )
                if route_records.ndim != 3 or route_records.shape[1] < int(args.route_gates):
                    errors.append(
                        f"{path}: route records do not materialize {args.route_gates} gates"
                    )
                if randomized:
                    dynamics_domains.append(np.asarray(
                        archive["observation/privileged/dynamics"][0], np.float64
                    ))
                    aerodynamic_domains.append(np.asarray(
                        archive["observation/privileged/aerodynamics"][0], np.float64
                    ))
                    randomized_domain_seeds.add(int(np.asarray(
                        archive["metadata/spawn/dynamics_domain_seed"]
                    )))
                controller_valid = np.asarray(archive["controller/valid"], np.bool_)
                solver_status = np.asarray(archive["controller/solver_status"], np.int32)
                solver_failure_fraction = float(np.mean(
                    (~controller_valid) | (solver_status != 0)
                ))
                solver_failure_fractions.append(solver_failure_fraction)
                if solver_failure_fraction > float(args.maximum_solver_failure_fraction):
                    errors.append(
                        f"{path}: solver failure fraction {solver_failure_fraction:.4f} "
                        f"exceeds {args.maximum_solver_failure_fraction:.4f}"
                    )
                expected_mask = ~np.asarray(archive["transition/gate_passed"], np.bool_)
                observed_mask = np.asarray(archive["target/dynamics_valid"], np.bool_)
                if not np.array_equal(expected_mask, observed_mask):
                    errors.append(f"{path}: dynamics mask mismatch")
                task = np.asarray(archive["observation/task_state"], np.float32)
                delta = np.asarray(archive["target/task_state_delta"], np.float32)
                if not np.allclose(delta, task[1:] - task[:-1], atol=1.0e-6):
                    errors.append(f"{path}: dynamics target mismatch")
                if "observation/gate_mask" in archive:
                    chunks = archive["observation/gate_mask"].chunks
                    if chunks is not None:
                        mask_chunk_rows.append(int(chunks[0]))
            sizes.append(path.stat().st_size)
        except (OSError, KeyError, ValueError) as error:
            errors.append(f"{path}: {type(error).__name__}: {error}")
    for name, target in expected.items():
        if counts[name] != target:
            errors.append(f"{name}: expected {target} episodes, found {counts[name]}")
        for domain, domain_target in expected_domains[name].items():
            if domain_counts[name][domain] != domain_target:
                errors.append(
                    f"{name}: expected {domain_target} {domain} episodes, "
                    f"found {domain_counts[name][domain]}"
                )
    episode_count = sum(counts.values())
    randomized_episode_count = sum(
        counts["randomized"] for counts in domain_counts.values()
    )
    if len(randomized_domain_seeds) != randomized_episode_count:
        errors.append(
            "randomized dynamics domains are not episode-unique: "
            f"{len(randomized_domain_seeds)}/{randomized_episode_count}"
        )
    dynamics_array = np.asarray(dynamics_domains, np.float64)
    aerodynamics_array = np.asarray(aerodynamic_domains, np.float64)
    if len(dynamics_array) and (
        np.any(np.ptp(dynamics_array[:, [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], axis=0) <= 0)
        or np.any(np.ptp(aerodynamics_array, axis=0) <= 0)
    ):
        errors.append("full dynamics/aerodynamics suite did not vary across episodes")
    report = {
        "passed": bool(manifest_report["passed"] and not errors),
        "manifest": manifest_report,
        "dataset_root": str(args.dataset),
        "episode_count": episode_count,
        "transition_count": transitions,
        "track_episode_counts": counts,
        "total_size_bytes": sum(sizes),
        "mean_episode_size_bytes": float(np.mean(sizes)) if sizes else 0.0,
        "mask_chunk_rows": {
            "minimum": min(mask_chunk_rows) if mask_chunk_rows else 0,
            "maximum": max(mask_chunk_rows) if mask_chunk_rows else 0,
        },
        "episode_counts_by_dynamics_domain": domain_counts,
        "dynamics_domain_unique_seeds": len(randomized_domain_seeds),
        "solver_failure_fraction": {
            "mean": float(np.mean(solver_failure_fractions))
            if solver_failure_fractions else 0.0,
            "maximum": max(solver_failure_fractions) if solver_failure_fractions else 0.0,
        },
        "dynamics_parameter_range": (
            {
                "minimum": dynamics_array.min(0).tolist(),
                "maximum": dynamics_array.max(0).tolist(),
            } if len(dynamics_array) else {}
        ),
        "aerodynamics_parameter_range": (
            {
                "minimum": aerodynamics_array.min(0).tolist(),
                "maximum": aerodynamics_array.max(0).tolist(),
            } if len(aerodynamics_array) else {}
        ),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["passed"]:
        raise RuntimeError(f"procedural dataset audit failed with {len(errors)} errors")


def _dataset_paths(archive: h5py.File) -> list[str]:
    paths: list[str] = []
    archive.visititems(
        lambda name, item: paths.append(name) if isinstance(item, h5py.Dataset) else None
    )
    return paths


def main() -> None:
    args = arguments()
    if args.command == "generate":
        generate(args)
        manifest = args.output / "manifest.json"
        report = audit_manifest(manifest)
        (args.output / "manifest-audit.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        write_manifest_gallery(manifest, args.output / "track-family-gallery.png")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        if not report["passed"]:
            raise RuntimeError("generated procedural manifest failed audit")
    elif args.command == "qualify":
        qualify(args)
    elif args.command == "repair-unqualified":
        repair_unqualified(args)
    elif args.command == "collect":
        collect(args)
    elif args.command == "audit":
        audit_dataset(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
