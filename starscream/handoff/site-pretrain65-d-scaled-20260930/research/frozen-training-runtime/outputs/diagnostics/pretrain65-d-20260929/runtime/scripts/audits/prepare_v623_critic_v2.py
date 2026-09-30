#!/usr/bin/env python3
"""Materialize the v6.23 200M high-speed PPO recipe with critic v2."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/exp/v6.21.rl.7/unified_real50_100m.yaml"
CORPUS = ROOT / "outputs/course-pools/v623-rl-technical-v1/manifest.json"
CONTRACT = ROOT / "outputs/course-pools/v623-rl-technical-v1/sampling-contract.json"
OUT = ROOT / "configs/exp/v6.23.rl.1/critic_v2_200m.yaml"
SOURCE = (
    "/workspace/outputs/checkpoints/"
    "starscream-v6.21.1.update-fix-pretrain65-dagger/"
    "best-step-121343545-full_course_success-0.63181818.pt"
)


def main() -> None:
    config = yaml.safe_load(BASE.read_text())
    corpus = json.loads(CORPUS.read_text())
    contract = json.loads(CONTRACT.read_text())
    if corpus["metadata"]["evaluation_courses_included"] != 0:
        raise RuntimeError("v6.23 corpus is not zero-shot safe")
    ppo = config["ppo"]
    target_steps = 200_000_000
    window_steps = int(ppo["ppo_rollout_window_steps"])
    lanes = int(ppo["rollout_envs"])
    ppo.update({
        "initial_checkpoint": SOURCE,
        "run_name": "starscream-v6.23.rl.1-critic-v2-200m",
        "track": "v623-technical105-role-floor-online-bank",
        "seed": 2026092301,
        "initial_environment_steps": 0,
        "target_environment_steps": target_steps,
        "cycles": math.ceil(target_steps / (window_steps * lanes)),
        "track_manifest": "/workspace/outputs/course-pools/v623-rl-technical-v1/manifest.json",
        "ppo_sampling_role_floors": contract["role_floors"],
        "actor_early_stop_scope": "epoch",
        "require_complete_actor_epoch": True,
        "actor_epochs": 3,
        "critic_architecture": "privileged_transformer_v2",
        "critic_include_history": True,
        "critic_include_episode_context": True,
        "critic_route_tokens": 24,
        "critic_course_vocab_size": 4096,
        "critic_model_dim": 320,
        "critic_transformer_depth": 4,
        "critic_attention_heads": 8,
        "critic_feedforward_dim": 640,
        "critic_dropout": 0.0,
        "critic_epochs": 8,
        "critic_minibatch_size": 4096,
        "critic_learning_rate": 2.0e-4,
        "critic_weight_decay": 1.0e-5,
        "critic_gradient_clip": 5.0,
        "critic_loss": "huber",
        "critic_normalize_loss_by_return_std": True,
        "ppo_critic_warmup_cycles": 10,
        "restore_actor_optimizer": False,
        "restore_critic_optimizer": False,
        "resume_online_course_bank": False,
        "initialize_online_course_bank_from_checkpoint": False,
    })
    ppo.pop("critic_initial_checkpoint", None)
    ppo.pop("actor_optimizer_initial_checkpoint", None)

    # One reward and one speed-forward curriculum from the first transition.
    stage = copy.deepcopy(ppo["curriculum"][-1])
    stage.update({
        "name": "high_speed_technical_retention",
        "target_speed": 20.0,
        "target_speed_range": [16.0, 22.0],
        "minimum_environment_steps": target_steps,
        "track_manifest": ppo["track_manifest"],
        "track_split": "train",
        "qualified_tracks_only": False,
    })
    ppo["curriculum"] = [stage]

    bank = dict(ppo.get("ppo_online_course_bank", {}))
    bank.update(contract["online_generation"])
    bank.update({
        "enabled": True,
        "protected_track_suite": "/workspace/configs/eval/v6_22_real100_hard_v2.yaml",
        "raceability": "nominal_mpcc",
    })
    bank.pop("retired_courses_consume_active_capacity", None)
    bank.pop("generated_courses_share_adaptive_remainder", None)
    bank.pop("protected_anchor_role", None)
    ppo["ppo_online_course_bank"] = bank

    config["experiment_notes"] = (
        "v6.23 critic-v2 foundation: update-fix actor, one high-speed reward/curriculum, "
        "105 leakage-safe anchors, role floors plus continual generation, complete PPO "
        "actor epochs, and a course-privileged Transformer value function. Real60 and "
        "real100-hard-v2 remain excluded from training."
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({
        "output": str(OUT),
        "courses": corpus["metadata"]["course_count"],
        "cycles": ppo["cycles"],
        "planned_steps": ppo["cycles"] * window_steps * lanes,
        "critic": ppo["critic_architecture"],
    }, indent=2))


if __name__ == "__main__":
    main()
