#!/usr/bin/env python3
"""Prepare the first 100M v6.21.1 post-training run."""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/exp/v6.23.rl.1/critic_v2_200m.yaml"
OUT = ROOT / "configs/exp/v6.21.1.rl.1/reward_frontier_100m.yaml"
CORPUS = ROOT / "outputs/course-pools/v623-rl-technical-v1/manifest.json"


def main() -> None:
    config = yaml.safe_load(BASE.read_text())
    corpus = json.loads(CORPUS.read_text())
    ppo = config["ppo"]
    target_steps = 100_000_000
    window_steps = int(ppo["ppo_rollout_window_steps"])
    lanes = int(ppo["rollout_envs"])
    role_floors = {
        role: float(value) * 0.625
        for role, value in ppo["ppo_sampling_role_floors"].items()
    }
    if abs(sum(role_floors.values()) - 0.5) > 1.0e-9:
        raise RuntimeError("v6.21.1.rl.1 role floors must reserve exactly half the batch")
    ppo.update({
        "run_name": "starscream-v6.21.1.rl.1-reward-frontier-100m",
        "track": "v6211-update-fix-technical105-medium-success",
        "seed": 2026092302,
        "target_environment_steps": target_steps,
        "cycles": math.ceil(target_steps / (window_steps * lanes)),
        "ppo_sampling_role_floors": role_floors,
        # This is the exact sampler that held RL7's last-100-window competence
        # near 0.62: prioritize failures, but stop hammering hopeless courses.
        "ppo_adaptive_level_replay": {
            "enabled": True,
            "competence_metric": "success",
            "priority": "failure",
            "failure_power": 1.0,
            "ema": 0.25,
            "minimum_multiplier": 0.4,
            "maximum_multiplier": 2.5,
            "stall_competence": 0.1,
            "stall_patience": 15,
            "stall_factor": 0.5,
            "stall_progress_epsilon": 0.05,
        },
        "ppo_sampling_success_band": [0.40, 0.75],
        "ppo_sampling_success_target": 0.60,
        "ppo_sampling_success_max_tilt": 10.0,
        # Early stopping evaluates the final policy after a complete epoch.
        # Drive the LR controller from that same measurement; minibatch-mean
        # KL systematically understates the displacement of the final policy.
        "kl_adaptation_statistic": "epoch",
        "evaluation_interval": 47,
        "reporting_evaluation_interval": 94,
        "checkpoint_rank_curriculum_stages": [0],
    })
    ppo["curriculum"][0]["minimum_environment_steps"] = target_steps
    config["checkpoint"]["run_name"] = ppo["run_name"]
    config["checkpoint"]["monitor"] = "full_course_success"
    config["checkpoint"]["mode"] = "max"
    config["wandb"]["name"] = ppo["run_name"]
    config["experiment_notes"] = (
        "v6.21.1.rl.1: 100M confirmation run from update-fix step 121343545. "
        "Uses the best-observed RL7 dense completion/time reward unchanged, a single "
        "16-22 m/s curriculum from step zero, critic-v2 with ten warmup windows, "
        "complete actor epochs with epoch-consistent KL adaptation, 50% role floors, "
        "and RL7 failure-priority sampling "
        "with its hopeless-course stall guard. Real60 and hard-v2 remain train-excluded."
    )
    if corpus["metadata"]["evaluation_courses_included"] != 0:
        raise RuntimeError("training corpus includes protected evaluation geometry")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({
        "output": str(OUT),
        "cycles": ppo["cycles"],
        "planned_steps": ppo["cycles"] * window_steps * lanes,
        "role_floor_mass": sum(role_floors.values()),
        "reward": ppo["reward"],
    }, indent=2))


if __name__ == "__main__":
    main()
