#!/usr/bin/env python3
"""Run a reproducible held-out probe of a privileged DAgger checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from train_privileged_racing import (
    evaluate_policy, load_config, parse_stage, stage_config,
)
from starscream.privileged_racing import load_policy_checkpoint
from starscream.env.real_course_suite import load_active_real_course_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--section", choices=("auto", "dagger", "ppo"), default="auto",
        help="configuration section; auto follows the checkpoint stage",
    )
    parser.add_argument(
        "--stage-index", type=int, default=0,
        help="index when the selected evaluation curriculum is a list",
    )
    parser.add_argument(
        "--curriculum", choices=("training", "validation", "reporting"),
        default="validation",
        help=(
            "evaluate the training split, held-out validation split, or "
            "reporting-only real-course suite"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--nominal-environment", action="store_true",
        help="disable dynamics/estimator/plan randomization for diagnosis",
    )
    parser.add_argument(
        "--randomized-environment", action="store_true",
        help=(
            "retain the configured dynamics/estimator/plan randomization even "
            "when reporting evaluation defaults to nominal"
        ),
    )
    parser.add_argument(
        "--nominal-dynamics", action="store_true",
        help="disable only dynamics randomization while preserving actor inputs",
    )
    parser.add_argument(
        "--workers", type=int,
        help="override evaluation worker count (useful beside a live trainer)",
    )
    parser.add_argument(
        "--matched-track-seeds", action="store_true",
        help=(
            "evaluate each track independently with the same seed sequence; "
            "--episodes is interpreted as episodes per track"
        ),
    )
    parser.add_argument(
        "--episode-index-offset", type=int, default=0,
        help="diagnostic offset applied to evaluation episode indices",
    )
    parser.add_argument(
        "--episode-index-stride", type=int, default=1,
        help="diagnostic stride applied to evaluation episode indices",
    )
    parser.add_argument("--historical-start-index", action="store_true",
                        help="Use the episode index for reset jitter, as in older real60 benchmarks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-steps", type=int,
        help="override the selected stage's episode step limit (long real courses)",
    )
    parser.add_argument(
        "--track-suite", type=Path,
        help=(
            "replace the selected curriculum's tracks with the active tracks "
            "from a frozen real-course suite"
        ),
    )
    parser.add_argument(
        "--speed-command-file", type=Path,
        help="JSON map of per-track evaluation speed commands",
    )
    args = parser.parse_args()
    if args.nominal_environment and args.randomized_environment:
        parser.error(
            "--nominal-environment and --randomized-environment are mutually exclusive"
        )
    config = load_config(args.config)
    policy, normalizer, payload, resolved = load_policy_checkpoint(
        args.checkpoint, args.device
    )
    section = str(payload.get("stage", "dagger")) if args.section == "auto" else args.section
    config = stage_config(config, section)
    if section not in config:
        raise ValueError(f"config has no {section!r} section")
    settings = dict(config[section])
    if args.historical_start_index:
        settings.pop("evaluation_fixed_start_gate_index", None)
    if args.speed_command_file is not None:
        if not args.speed_command_file.is_file():
            parser.error(f"speed command file missing: {args.speed_command_file}")
        settings["evaluation_track_speed_command_file"] = str(args.speed_command_file.resolve())
    if args.episode_index_offset < 0 or args.episode_index_stride < 1:
        parser.error("episode index offset must be nonnegative and stride positive")
    settings["evaluation_episode_index_offset"] = args.episode_index_offset
    settings["evaluation_episode_index_stride"] = args.episode_index_stride
    if args.workers is not None:
        if args.workers < 1:
            raise ValueError("workers must be positive")
        settings["evaluation_workers"] = args.workers
    stage_key = {
        "training": "curriculum",
        "validation": "evaluation_curriculum",
        "reporting": "reporting_evaluation_curriculum",
    }[args.curriculum]
    if stage_key not in settings and args.curriculum == "reporting":
        raise ValueError("config has no reporting_evaluation_curriculum")
    raw_stage = settings.get(stage_key, settings["curriculum"])
    if isinstance(raw_stage, list):
        if not 0 <= args.stage_index < len(raw_stage):
            raise IndexError(
                f"stage-index {args.stage_index} outside [0, {len(raw_stage)})"
            )
        raw_stage = raw_stage[args.stage_index]
    elif args.stage_index != 0:
        raise IndexError("stage-index must be zero for a scalar curriculum")
    stage = parse_stage(raw_stage)
    suite_metadata = None
    if args.track_suite is not None:
        suite_tracks, suite_metadata = load_active_real_course_suite(
            args.track_suite
        )
        stage = replace(
            stage,
            name=f"{stage.name}_{args.track_suite.stem}",
            tracks=tuple(str(track) for track in suite_tracks),
        )
    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error("--max-steps must be positive")
        stage = replace(stage, max_steps=int(args.max_steps))
    nominal_environment = bool(
        args.nominal_environment
        or (
            not args.randomized_environment
            and
            args.curriculum == "reporting"
            and settings.get("reporting_nominal_environment", False)
        )
    )
    if nominal_environment or args.nominal_dynamics:
        settings["dynamics_randomization"] = {"enabled": False}
    if nominal_environment:
        settings.pop("state_estimator_randomization", None)
        settings.pop("flight_plan_randomization", None)
        settings["policy_state_source"] = "truth"
        settings.pop("action_delay_range", None)
        settings["action_delay"] = float(
            settings.get("reporting_action_delay", 0.011)
        )
    seed = int(args.seed if args.seed is not None else settings.get("evaluation_seed", 20260900))
    if args.matched_track_seeds:
        if args.episodes < 1:
            parser.error("--episodes must be positive")
        per_track = []
        metrics = {}
        for track in stage.tracks:
            track_stage = replace(stage, tracks=(track,))
            track_metrics = evaluate_policy(
                policy, normalizer, settings, track_stage,
                count=args.episodes, seed_base=seed, device=args.device,
            )
            per_track.append(track_metrics)
            metrics.update({
                key: value for key, value in track_metrics.items()
                if key.startswith("track/")
            })
        # The matched mode is primarily a controlled per-track diagnostic.
        # Retain compact aggregate fields for CLI compatibility without
        # pretending that separately aggregated rollouts are one raw sample.
        for key in (
            "full_course_success", "mean_gates", "crash_rate", "mean_speed_mps",
            "mean_gate_speed_mps", "maximum_speed_mps",
            "mean_collective_command_mps2", "maximum_collective_command_mps2",
            "performance_weighted_success_hz90",
        ):
            values = [float(row[key]) for row in per_track if key in row]
            if values:
                metrics[key] = sum(values) / len(values)
        scores = [float(row["selection_score"]) for row in per_track]
        successes = [float(row["full_course_success"]) for row in per_track]
        metrics["episodes"] = float(args.episodes * len(per_track))
        metrics["episodes_per_track"] = float(args.episodes)
        metrics["selection_score"] = min(scores)
        metrics["minimum_track_selection_score"] = min(scores)
        metrics["minimum_track_full_course_success"] = min(successes)
    else:
        metrics = evaluate_policy(
            policy, normalizer, settings, stage,
            count=args.episodes, seed_base=seed, device=args.device,
        )
    report = {
        "checkpoint": str(resolved),
        "checkpoint_round": int(payload.get("round", -1)),
        "checkpoint_environment_steps": int(payload.get("environment_steps", -1)),
        "episodes": args.episodes,
        "matched_track_seeds": bool(args.matched_track_seeds),
        "episode_index_offset": int(args.episode_index_offset),
        "episode_index_stride": int(args.episode_index_stride),
        "seed": seed,
        "curriculum": args.curriculum,
        "nominal_environment": nominal_environment,
        "nominal_dynamics": bool(args.nominal_dynamics),
        "config_section": section,
        "stage_index": args.stage_index,
        "stage": stage.name,
        "max_steps": int(stage.max_steps),
        "tracks": [Path(track).stem for track in stage.tracks],
        "track_suite": (
            None if args.track_suite is None else str(args.track_suite)
        ),
        "track_suite_metadata": suite_metadata,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        f"eval checkpoint={resolved.name} full={metrics['full_course_success']:.3f} "
        f"mean_gates={metrics['mean_gates']:.2f} crash={metrics['crash_rate']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
