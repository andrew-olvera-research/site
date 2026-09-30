#!/usr/bin/env python3
"""Held-out matched and stress evaluation for a privileged multi-track policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train_privileged_racing import (
    evaluate_policy, load_config, parse_stage,
)
from starscream.privileged_racing import load_policy_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=384)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026084400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = dict(config["dagger"])
    settings.update(
        evaluation_backend="process", evaluation_workers=args.workers,
    )
    policy, normalizer, _, checkpoint = load_policy_checkpoint(
        args.checkpoint, args.device
    )
    matched = parse_stage(settings["curriculum"])
    stress = replace(
        matched, name=f"{matched.name}_stress",
        approach_distance=(2.0, 5.0), lateral_offset=(-0.50, 0.50),
        vertical_offset=(-0.30, 0.30), forward_speed=(2.0, 6.5),
        lateral_speed=(-0.70, 0.70), vertical_speed=(-0.42, 0.42),
        attitude_error_degrees=10.0, body_rate=0.75,
    )
    profiles = {}
    for index, stage in enumerate((matched, stress)):
        metrics = evaluate_policy(
            policy, normalizer, settings, stage, count=args.episodes,
            seed_base=args.seed + 100000 * index, device=args.device,
        )
        metrics["successful_lap_time_seconds"] = (
            metrics["successful_mean_steps"] / 90.0
        )
        profiles["matched" if index == 0 else "stress"] = {
            "stage": asdict(stage), "metrics": metrics,
        }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "episodes_per_profile": args.episodes,
        "workers": args.workers, "seed": args.seed, "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
