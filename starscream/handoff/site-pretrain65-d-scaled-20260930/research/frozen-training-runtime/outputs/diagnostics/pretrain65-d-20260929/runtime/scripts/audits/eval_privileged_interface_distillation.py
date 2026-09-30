#!/usr/bin/env python3
"""Deep paired-distribution evaluation of a distilled interface and its teacher."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train_privileged_interface_distillation import evaluate_closed_loop
from starscream.privileged_distillation import load_interface_policy
from starscream.privileged_racing import resolve_ranked_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rollout-laps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026084100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    settings = dict(config["distillation"])
    settings.update(
        evaluation_episodes=args.episodes, evaluation_workers=args.workers,
    )
    settings["evaluation"] = dict(settings["evaluation"])
    settings["evaluation"]["seed"] = args.seed
    curriculum = dict(settings["evaluation"]["curriculum"])
    curriculum["rollout_laps"] = args.rollout_laps
    if args.max_steps is not None:
        curriculum["max_steps"] = args.max_steps
    settings["evaluation"]["curriculum"] = curriculum
    checkpoint = resolve_ranked_checkpoint(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = load_interface_policy(state, args.device).eval()
    student = evaluate_closed_loop(policy, settings, args.device)
    teacher = evaluate_closed_loop(
        policy, settings, args.device, exact_teacher=True
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "episodes": args.episodes,
        "workers": args.workers, "seed": args.seed,
        "rollout_laps": args.rollout_laps,
        "student": student, "exact_teacher": teacher,
        "completion_gap": student["success"] - teacher["success"],
        "crash_gap": student["crash"] - teacher["crash"],
        "lap_time_gap_seconds": (
            student["successful_lap_time"] - teacher["successful_lap_time"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
