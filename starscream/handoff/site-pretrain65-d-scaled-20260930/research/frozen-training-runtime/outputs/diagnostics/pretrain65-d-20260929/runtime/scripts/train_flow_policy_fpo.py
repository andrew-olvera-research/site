#!/usr/bin/env python3
"""Curriculum online RL for the tokenizer-native shortcut-flow actor.

One policy decision is one three-tick normalized CTBR chunk.  The actor uses
FPO++ per-CFM-sample ratios and ASPO; the value function is an asymmetric MLP
over simulator state and is never part of deployment.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import random
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

try:  # Support both direct script execution and pytest/module imports.
    from scripts.eval_flight import ObservationHistory, normalized_to_ctbr
    from scripts.train_actor_bc import load_frozen_actor_stack, move_batch
    from scripts.train_flow_policy_bc import history_valid_steps, stratified_episode_split
except ModuleNotFoundError:
    from eval_flight import ObservationHistory, normalized_to_ctbr
    from train_actor_bc import load_frozen_actor_stack, move_batch
    from train_flow_policy_bc import history_valid_steps, stratified_episode_split
from starscream.DiT import TokenizerFlowPolicy
from starscream.control_policy import (
    BoundedResidualFlowPolicy,
    BoundedResidualGaussianPolicy,
    BoundedResidualKnotPolicy,
)
from starscream.actor_critic import diagonal_gaussian_kl
from starscream.checkpoint_manager import (
    CheckpointManager, capture_rng_state, restore_rng_state,
)
from starscream.dataloader import DreamerSequenceDataset
from starscream.env import FlightmareEnv
from starscream.env.tracks import quaternion_matrix
from starscream.flow_policy import tokenizer_policy_tokens
from starscream.fpo import (
    FlowValueCritic,
    conditional_flow_loss,
    configure_flow_actor_scope,
    fpo_plus_plus_loss,
    sample_cfm_conditions,
    variable_discount_gae,
)
from starscream.mpcc import (
    MPCCConfig, MPCCController, RacingLinePlanner, RacingLinePlannerConfig,
)
from starscream.racing_curriculum import (
    CurriculumRaceReward,
    CurriculumRewardConfig,
    GateChainCurriculum,
    RaceResetArchive,
    RacingCurriculum,
    RacingCurriculumStage,
    sample_curriculum_spawn,
)
from starscream.wandb import init_wandb


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    seen = set() if seen is None else seen
    if resolved in seen:
        raise ValueError(f"cyclic experiment config inheritance at {resolved}")
    seen.add(resolved)
    with resolved.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    parent = config.get("inherits")
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    return _merge_config(_load_config(parent_path, seen), config)


def parse_args() -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-update", action="store_true")
    parser.add_argument("--archive-probe", action="store_true")
    parser.add_argument("--disable-reset-archive", action="store_true")
    parser.add_argument("--frozen-actor-probe", action="store_true")
    parser.add_argument("--rollout-envs", type=int)
    preliminary, _ = parser.parse_known_args()
    config = _load_config(preliminary.config)
    settings = config["flow_fpo"]
    parser.set_defaults(
        device=str(settings.get("device", "cuda")),
        resume=bool(settings.get("resume", False)),
    )
    args = parser.parse_args()
    if args.smoke:
        config = copy.deepcopy(config)
        settings = config["flow_fpo"]
        settings.update(
            cycles=1, episodes_per_cycle=1, evaluation_episodes=1,
            actor_epochs=1, critic_epochs=1, minibatch_size=4,
            offline_workers=0, resume=False, initial_curriculum_stage=0,
        )
        first = dict(settings["curriculum"][0])
        first.update(max_steps=24, mpcc_prefix_steps=1, minimum_episodes=99)
        settings["curriculum"] = [first]
        config["output_root"] = "/tmp/starscream-flow-fpo-smoke"
        config["checkpoint"] = {
            "run_name": "actor-flow-fpo-curriculum-smoke",
            "monitor": "curriculum_score", "mode": "max", "top_k": 1,
        }
        config["wandb"] = {"enabled": False, "run_name": "actor-flow-fpo-curriculum-smoke"}
        args.resume = False
    elif args.archive_probe:
        config = copy.deepcopy(config)
        settings = config["flow_fpo"]
        selected = int(settings.get("initial_curriculum_stage", 0))
        stage = dict(settings["curriculum"][selected])
        stage.update(minimum_episodes=999999, advancement_window=999999)
        settings.update(
            cycles=6, episodes_per_cycle=64, rollout_envs=64,
            env_step_workers=64, evaluation_episodes=32, evaluation_interval=1,
            actor_epochs=2, critic_epochs=2, offline_updates_per_cycle=1,
            offline_workers=0, initial_curriculum_stage=0, resume=False,
            curriculum=[stage],
        )
        if settings.get("reset_curriculum"):
            settings["reset_curriculum"]["minimum_source_outcomes"] = 12
        config["output_root"] = "/tmp/starscream-flow-fpo-archive-probe"
        config["checkpoint"] = {
            "run_name": "actor-flow-fpo-archive-probe",
            "monitor": "curriculum_score", "mode": "max", "top_k": 1,
        }
        config["wandb"] = {
            "enabled": False, "run_name": "actor-flow-fpo-archive-probe"
        }
        args.resume = False
    elif args.benchmark:
        config = copy.deepcopy(config)
        settings = config["flow_fpo"]
        settings.update(
            cycles=1, episodes_per_cycle=128, evaluation_episodes=16,
            evaluation_interval=1, offline_workers=0, resume=False,
            initial_curriculum_stage=0,
        )
        if not args.full_update:
            settings.update(
                actor_epochs=1, critic_epochs=1, offline_updates_per_cycle=1
            )
        first = dict(settings["curriculum"][0])
        first.update(max_steps=90, minimum_episodes=999)
        settings["curriculum"] = [first]
        config["output_root"] = "/tmp/starscream-flow-fpo-benchmark"
        config["checkpoint"] = {
            "run_name": "actor-flow-fpo-curriculum-benchmark",
            "monitor": "curriculum_score", "mode": "max", "top_k": 1,
        }
        config["wandb"] = {
            "enabled": False, "run_name": "actor-flow-fpo-curriculum-benchmark"
        }
        args.resume = False
    if args.rollout_envs is not None:
        if args.rollout_envs < 1:
            parser.error("--rollout-envs must be positive")
        config["flow_fpo"]["rollout_envs"] = args.rollout_envs
        config["flow_fpo"]["env_step_workers"] = args.rollout_envs
    if args.disable_reset_archive:
        config["flow_fpo"].setdefault("reset_curriculum", {})["enabled"] = False
    if args.frozen_actor_probe:
        config["flow_fpo"]["actor_learning_rate"] = 0.0
        config["flow_fpo"]["offline_updates_per_cycle"] = 0
    return args, config


def critic_features(
    observation: dict[str, Any], env: FlightmareEnv, *,
    start_passed: int, target_gates: int, policy_steps: int, max_steps: int,
) -> np.ndarray:
    state = np.asarray(observation["state"], np.float32)
    task = np.asarray(observation["task_state"], np.float32)
    previous = np.asarray(observation["previous_action"], np.float32)
    achieved = env.tracker.passed_count - start_passed
    progress = env._ordered_course_progress(state[:3], env.tracker.index, env.tracker.lap)
    context = np.asarray([
        achieved / max(target_gates, 1),
        policy_steps / max(max_steps, 1),
        env.tracker.index / max(len(env.track.gates) - 1, 1),
        progress / max(env._course_length, 1e-6),
    ], np.float32)
    raw = np.concatenate([state, task, previous, context]).astype(np.float32)
    return (np.sign(raw) * np.log1p(np.abs(raw))).astype(np.float32)


CRITIC_INPUT_DIM = 25 + 19 + 4 + 4


@dataclass
class EpisodeResult:
    transitions: list[dict[str, torch.Tensor]]
    success: bool
    gates_passed: int
    target_gates: int
    crashed: bool
    reward: float
    raw_steps: int
    track: str
    spawn_source: str = "procedural"
    timed_out: bool = False
    teacher_steps: int = 0
    reward_components: dict[str, float] = field(default_factory=dict)
    first_gate_step: int = -1
    post_gate_steps: int = 0


@dataclass
class EpisodeSlot:
    env: FlightmareEnv
    history: ObservationHistory
    observation: dict[str, Any]
    track: str
    start_passed: int
    target_gates: int
    max_steps: int
    transitions: list[dict[str, torch.Tensor]]
    reward: float = 0.0
    raw_steps: int = 0
    crashed: bool = False
    success: bool = False
    done: bool = False
    timed_out: bool = False
    teacher_steps: int = 0
    spawn_source: str = "procedural"
    failure_frontier: deque[tuple] = field(
        default_factory=deque
    )
    reward_components: dict[str, float] = field(default_factory=dict)
    first_gate_step: int = -1


def _stack_histories(
    histories: list[ObservationHistory], device: str,
) -> dict[str, torch.Tensor]:
    batches = [history.batch("cpu", repeat_first_padding=True) for history in histories]
    return {
        key: torch.cat([batch[key] for batch in batches], dim=0).to(
            device, non_blocking=True
        )
        for key in batches[0]
    }


def _finish_advantages(
    transitions: list[dict[str, torch.Tensor]], settings: dict[str, Any],
) -> None:
    if not transitions:
        return
    rewards = torch.stack([item["reward"] for item in transitions])
    values = torch.stack([item["value"] for item in transitions])
    next_values = torch.stack([item["next_value"] for item in transitions])
    discounts = torch.stack([item["discount"] for item in transitions])
    advantages, returns = variable_discount_gae(
        rewards, values, next_values, discounts,
        gae_lambda=float(settings.get("gae_lambda", 0.95)),
    )
    for item, advantage, return_ in zip(transitions, advantages, returns):
        item["advantage"] = advantage
        item["return"] = return_


class OfflineExpertSource:
    """Small held-out-from-RL expert stream used only as a BC/shortcut anchor."""

    def __init__(self, settings: dict[str, Any], encoder_config: dict[str, Any], patch: int):
        train_paths, _ = stratified_episode_split(
            settings["offline_data"],
            float(settings.get("offline_validation_fraction", 0.05)),
            int(settings.get("seed", 0)),
        )
        history_raw = int(settings.get("history_steps", 3)) * patch
        horizon = int(settings.get("action_horizon", patch))
        dataset = DreamerSequenceDataset(
            settings["offline_data"], paths=train_paths,
            sequence_length=history_raw + horizon - 1,
            stride=int(settings.get("offline_stride", patch)), mode="dynamics",
            mask_size=tuple(encoder_config.get("image_size", (128, 160))),
            max_open_files=int(settings.get("max_open_files", 12)),
            validate_contents=False, include_privileged=False,
            cache_in_memory=False, require_controller_valid=False,
        )
        indices, _ = dataset.behavior_cloning_windows(
            history=history_raw, action_horizon=horizon
        )
        if not indices:
            raise ValueError("offline expert source has no legal BC windows")
        workers = int(settings.get("offline_workers", 2))
        self.loader = DataLoader(
            Subset(dataset, indices),
            batch_size=int(settings.get("offline_batch_size", 24)),
            shuffle=True, drop_last=True, num_workers=workers,
            pin_memory=True, persistent_workers=workers > 0,
            **({"prefetch_factor": 2, "multiprocessing_context": "spawn"} if workers else {}),
        )
        self.iterator: Iterator = iter(self.loader)
        self.history_raw = history_raw
        self.horizon = horizon

    def next(self) -> dict[str, torch.Tensor]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def make_reward(settings: dict[str, Any], stage: RacingCurriculumStage) -> CurriculumRaceReward:
    raw = dict(settings.get("reward", {}))
    raw["target_speed"] = stage.target_speed
    return CurriculumRaceReward(CurriculumRewardConfig(**raw))


def transition_target_gates(
    settings: dict[str, Any], spawn_source: str, default: int,
) -> int:
    """Select the atomic horizon for each gate-transition reset source."""

    raw = dict(settings.get("transition_curriculum", {}))
    if not bool(raw.get("enabled", False)):
        return int(default)
    keys = {
        "transition_pre": "pre_target_gates",
        "transition_post": "post_target_gates",
        "failure": "failure_target_gates",
        "gate": "post_target_gates",
        "intergate": "post_target_gates",
        "procedural": "procedural_target_gates",
    }
    key = keys.get(spawn_source, "procedural_target_gates")
    return max(1, int(raw.get(key, default)))


def transition_crossing_adjustment(
    slot: EpisodeSlot, info: dict[str, Any], gates_passed: int,
    settings: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Make the final gate the outcome while rewarding a controllable handoff."""

    raw = dict(settings.get("transition_curriculum", {}))
    if not bool(raw.get("enabled", False)) or not bool(info.get("gate_passed")):
        return 0.0, {}
    adjustment = 0.0
    components: dict[str, float] = {}
    if gates_passed < slot.target_gates:
        gate_reward = float(info.get("reward_components", {}).get("gate_pass", 0.0))
        scale = float(raw.get("intermediate_gate_reward_scale", 0.0))
        removed = (scale - 1.0) * gate_reward
        adjustment += removed
        components["intermediate_gate_adjustment"] = removed

        state = np.asarray(slot.observation["state"], np.float32)
        next_gate = slot.env.track.gates[slot.env.tracker.index]
        direction = np.asarray(next_gate.position - state[:3], np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        body_forward = quaternion_matrix(state[3:7])[:, 0]
        alignment = max(0.0, float(body_forward @ direction))
        velocity = np.asarray(state[7:10], np.float32)
        speed = float(np.linalg.norm(velocity))
        velocity_alignment = (
            max(0.0, float((velocity / speed) @ direction)) if speed > 1e-6 else 0.0
        )
        velocity_weight = float(np.clip(
            raw.get("handoff_velocity_weight", 0.0), 0.0, 1.0
        ))
        handoff_alignment = (
            (1.0 - velocity_weight) * alignment
            + velocity_weight * velocity_alignment
        )
        rate_scale = max(float(raw.get("handoff_body_rate_scale", 4.0)), 1e-6)
        rate_quality = float(np.exp(-0.5 * np.mean((state[10:13] / rate_scale) ** 2)))
        handoff = (
            float(raw.get("handoff_bonus", 4.0))
            * handoff_alignment * rate_quality
        )
        adjustment += handoff
        components.update({
            "next_gate_handoff": handoff,
            "next_gate_alignment": alignment,
            "next_gate_velocity_alignment": velocity_alignment,
            "next_gate_handoff_alignment": handoff_alignment,
            "handoff_rate_quality": rate_quality,
        })
    return adjustment, components


def chain_selection_score(
    metrics: dict[str, float], settings: dict[str, Any],
) -> float:
    """Lexicographic checkpoint score with first-gate competence as a gate."""

    raw = dict(settings.get("transition_curriculum", {}))
    floor = max(float(raw.get("first_gate_success_floor", 0.50)), 1e-6)
    first = float(metrics.get("probability_at_least_1_gate", 0.0))
    if first < floor:
        return first / floor
    second = float(metrics.get("probability_at_least_2_gates", 0.0))
    third = float(metrics.get("probability_at_least_3_gates", 0.0))
    survival = float(metrics.get("post_gate_survival_steps", 0.0))
    survival_scale = max(float(raw.get("survival_score_steps", 180.0)), 1.0)
    return 1.0 + second + 0.10 * third + 0.05 * min(survival / survival_scale, 1.0)


def actor_scope(stage: RacingCurriculumStage, settings: dict[str, Any]) -> str:
    minimum = settings.get("minimum_trainable_scope")
    if minimum is None:
        return stage.trainable_scope
    order = {"flow_tail": 0, "flow": 1, "flow_context_tail": 2, "all": 3}
    minimum = str(minimum)
    if minimum not in order:
        raise ValueError("minimum_trainable_scope is invalid")
    return max((stage.trainable_scope, minimum), key=order.__getitem__)


def configure_actor_scope(
    actor: TokenizerFlowPolicy | BoundedResidualFlowPolicy | BoundedResidualGaussianPolicy,
    scope: str,
) -> tuple[int, int]:
    if isinstance(actor, BoundedResidualGaussianPolicy):
        del scope
        actor.base.eval().requires_grad_(False)
        actor.residual_head.requires_grad_(True)
        actor.log_std.requires_grad_(True)
        return (
            sum(p.numel() for p in actor.parameters() if p.requires_grad),
            sum(p.numel() for p in actor.parameters()),
        )
    if isinstance(actor, BoundedResidualFlowPolicy):
        actor.base.eval().requires_grad_(False)
        return configure_flow_actor_scope(actor.flow, scope)
    return configure_flow_actor_scope(actor, scope)


def make_mpcc(env: FlightmareEnv, settings: dict[str, Any], action_delay: float) -> MPCCController:
    planner_arguments: dict[str, Any] = {
        "offset_iterations": int(settings.get("racing_line_iterations", 30)),
        "cache_directory": str(
            settings.get("racing_line_cache", "/workspace/outputs/evals/racing-lines")
        ),
    }
    for setting, argument in (
        ("racing_line_aperture_fraction", "aperture_fraction"),
        ("racing_line_aperture_margin", "aperture_margin"),
    ):
        if setting in settings:
            planner_arguments[argument] = float(settings[setting])
    line = RacingLinePlanner(RacingLinePlannerConfig(**planner_arguments)).plan(env.track)
    controller_arguments: dict[str, Any] = {
        "backend": str(settings.get("mpcc_backend", "predictive")),
        "actuation_delay": action_delay,
    }
    for setting, argument in (
        ("mpcc_nominal_speed", "nominal_speed"),
        ("mpcc_maximum_acceleration", "maximum_acceleration"),
        ("mpcc_maximum_longitudinal_acceleration", "maximum_longitudinal_acceleration"),
        ("mpcc_maximum_braking_acceleration", "maximum_braking_acceleration"),
    ):
        if setting in settings:
            controller_arguments[argument] = float(settings[setting])
    if "mpcc_track_speed_overrides" in settings:
        controller_arguments["track_speed_overrides"] = tuple(
            (str(name), float(speed))
            for name, speed in dict(settings["mpcc_track_speed_overrides"]).items()
        )
    controller = MPCCController(
        env.track, line,
        config=MPCCConfig(**controller_arguments),
    )
    controller.reset()
    return controller


@torch.no_grad()
def run_episode(
    actor: TokenizerFlowPolicy,
    encoder,
    action_tokenizer,
    critic: FlowValueCritic,
    stage: RacingCurriculumStage,
    settings: dict[str, Any],
    *, episode_index: int, seed: int, device: str, training: bool,
) -> EpisodeResult:
    track_name = stage.tracks[episode_index % len(stage.tracks)]
    env = FlightmareEnv(
        track=track_name, next_gates=3, image_size=(160, 128),
        control_dt=1.0 / 90.0, render_observations=False,
        mask_source="geometry", mask_size=(160, 128),
        image_delay=float(settings.get("image_delay", 0.033)),
        action_delay=stage.action_delay,
        reward_function=make_reward(settings, stage),
        terminate_on_collision=True,
    )
    transitions: list[dict[str, torch.Tensor]] = []
    reward_sum = 0.0
    raw_steps = 0
    crashed = False
    try:
        spawn = sample_curriculum_spawn(
            env.track, stage, seed=seed, episode_index=episode_index
        )
        observation, _ = env.reset(
            seed=seed,
            options={
                "gate_index": spawn.gate_index, "state": spawn.state,
                "spawn": dict(spawn.metadata),
            },
        )
        history = ObservationHistory(
            actor.history_steps * encoder.temporal_patch_size,
            tuple(encoder.image_size),
        )
        history.append(observation)
        if stage.mpcc_prefix_steps:
            expert = make_mpcc(env, settings, stage.action_delay)
            for _ in range(stage.mpcc_prefix_steps):
                command = expert(observation).action.as_array()
                observation, _, terminated, _, info = env.step(command)
                history.append(observation)
                if terminated:
                    return EpisodeResult([], False, 0, stage.target_gates, True, 0.0, 0, track_name)

        start_passed = env.tracker.passed_count
        target_gates = min(stage.target_gates, len(env.track.gates))
        amp = bool(settings.get("amp", True))
        gamma_raw = float(settings.get("discount_raw", 0.997))
        samples = int(settings.get("cfm_samples", 8))
        sampling_steps = int(settings.get("sampling_steps", 4))
        sampling_method = str(settings.get("sampling_method", "heun"))
        success = False
        while raw_steps < stage.max_steps and not crashed and not success:
            batch = history.batch(device, repeat_first_padding=True)
            observation_tokens, applied_tokens = tokenizer_policy_tokens(
                encoder, action_tokenizer, batch,
                history_raw_steps=history.length, amp=amp,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda") and amp):
                action_chunk = actor.sample(
                    observation_tokens, applied_tokens,
                    steps=sampling_steps, method=sampling_method,
                    previous_action=batch["previous_action"][:, history.length - 1],
                    deterministic=not training,
                ).float()
            critic_input = torch.from_numpy(critic_features(
                observation, env, start_passed=start_passed,
                target_gates=target_gates, policy_steps=raw_steps,
                max_steps=stage.max_steps,
            )).to(device)[None]
            value = critic(critic_input).float()
            if training:
                epsilon, flow_time, shortcut_step = sample_cfm_conditions(
                    1, samples, actor.action_horizon, actor.action_dim,
                    device=device, dtype=action_chunk.dtype,
                    time_beta=float(settings.get("cfm_time_beta", 1.0)),
                    step_size=1.0 / sampling_steps,
                )
            executed_rewards: list[float] = []
            valid = torch.zeros(1, actor.action_horizon, dtype=torch.bool, device=device)
            terminated = False
            for offset, normalized in enumerate(action_chunk[0]):
                observation, reward, terminated, _, info = env.step(
                    normalized_to_ctbr(normalized.cpu().numpy())
                )
                history.append(observation)
                raw_steps += 1
                valid[0, offset] = True
                gates = env.tracker.passed_count - start_passed
                success = gates >= target_gates
                crashed = bool(info.get("ground_contact") or info.get("unity_collision"))
                if success:
                    reward += float(settings.get("segment_completion_bonus", 25.0))
                executed_rewards.append(float(reward))
                reward_sum += float(reward)
                if terminated or success or raw_steps >= stage.max_steps:
                    break
            discounted_reward = sum(
                (gamma_raw ** index) * reward for index, reward in enumerate(executed_rewards)
            )
            next_input = torch.from_numpy(critic_features(
                observation, env, start_passed=start_passed,
                target_gates=target_gates, policy_steps=raw_steps,
                max_steps=stage.max_steps,
            )).to(device)[None]
            artificial_terminal = terminated or success
            next_value = torch.zeros_like(value) if artificial_terminal else critic(next_input).float()
            discount = 0.0 if artificial_terminal else gamma_raw ** len(executed_rewards)
            if training:
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=device.startswith("cuda") and amp,
                ):
                    old_cfm, _ = conditional_flow_loss(
                        actor, observation_tokens, applied_tokens, action_chunk,
                        epsilon, flow_time, shortcut_step, valid_steps=valid,
                        huber_delta=settings.get("cfm_huber_delta"),
                    )
                transitions.append({
                    "observation_tokens": observation_tokens[0].float().cpu(),
                    "action_tokens": applied_tokens[0].float().cpu(),
                    "critic_input": critic_input[0].cpu(),
                    "action": action_chunk[0].cpu(),
                    "valid": valid[0].cpu(),
                    "epsilon": epsilon[0].cpu(),
                    "flow_time": flow_time[0].cpu(),
                    "shortcut_step": shortcut_step[0].cpu(),
                    "old_cfm": old_cfm[0].float().cpu(),
                    "reward": torch.tensor(discounted_reward),
                    "discount": torch.tensor(discount),
                    "value": value[0].cpu(),
                    "next_value": next_value[0].cpu(),
                })
        if transitions:
            rewards = torch.stack([item["reward"] for item in transitions])
            values = torch.stack([item["value"] for item in transitions])
            next_values = torch.stack([item["next_value"] for item in transitions])
            discounts = torch.stack([item["discount"] for item in transitions])
            advantages, returns = variable_discount_gae(
                rewards, values, next_values, discounts,
                gae_lambda=float(settings.get("gae_lambda", 0.95)),
            )
            for item, advantage, return_ in zip(transitions, advantages, returns):
                item["advantage"] = advantage
                item["return"] = return_
        return EpisodeResult(
            transitions, success, env.tracker.passed_count - start_passed,
            target_gates, crashed, reward_sum, raw_steps, track_name,
        )
    finally:
        env.close()


@torch.no_grad()
def run_episode_batch(
    actor: TokenizerFlowPolicy,
    reference: TokenizerFlowPolicy,
    encoder,
    action_tokenizer,
    critic: FlowValueCritic,
    stage: RacingCurriculumStage,
    settings: dict[str, Any],
    *, episode_indices: list[int], seeds: list[int], device: str, training: bool,
    reset_archive: RaceResetArchive | None = None,
    gate_chain: GateChainCurriculum | None = None,
) -> list[EpisodeResult]:
    """Collect isolated simulators with one central batched GPU policy.

    Flightmare instances remain independent. Only their CPU ``step`` calls are
    threaded; tokenizer, flow-policy, critic and CFM/reference inference are
    executed once for all currently active environments.
    """

    if len(episode_indices) != len(seeds) or not episode_indices:
        raise ValueError("episode_indices and seeds must be non-empty and aligned")
    slots: list[EpisodeSlot] = []
    amp = bool(settings.get("amp", True))
    gamma_raw = float(settings.get("discount_raw", 0.997))
    samples = int(settings.get("cfm_samples", 8))
    sampling_steps = int(settings.get("sampling_steps", 4))
    sampling_method = str(settings.get("sampling_method", "heun"))
    worker_count = max(1, min(
        int(settings.get("env_step_workers", len(episode_indices))), len(episode_indices)
    ))
    try:
        for episode_index, seed in zip(episode_indices, seeds):
            track_name = stage.tracks[episode_index % len(stage.tracks)]
            env = FlightmareEnv(
                track=track_name, next_gates=3, image_size=(160, 128),
                control_dt=1.0 / 90.0, render_observations=False,
                mask_source="geometry", mask_size=(160, 128),
                geometry_renderer=str(settings.get("geometry_renderer", "exact")),
                image_delay=float(settings.get("image_delay", 0.033)),
                action_delay=stage.action_delay,
                reward_function=make_reward(settings, stage),
                terminate_on_collision=True,
            )
            spawn = (
                reset_archive.sample_spawn(
                    env.track, stage, seed=seed, episode_index=episode_index
                )
                if training and reset_archive is not None
                else sample_curriculum_spawn(
                    env.track, stage, seed=seed, episode_index=episode_index
                )
            )
            reset_options = {
                "gate_index": spawn.gate_index, "state": spawn.state,
                "spawn": dict(spawn.metadata),
            }
            if spawn.previous_action is not None:
                reset_options["previous_action"] = spawn.previous_action
            observation, _ = env.reset(
                seed=seed, options=reset_options,
            )
            bounded_actor = isinstance(
                actor, (BoundedResidualFlowPolicy, BoundedResidualGaussianPolicy)
            )
            history = ObservationHistory(
                (
                    actor.history_raw_steps if bounded_actor
                    else actor.history_steps * encoder.temporal_patch_size
                ),
                (
                    tuple(settings.get("image_size", [128, 160])) if bounded_actor
                    else tuple(encoder.image_size)
                ),
            )
            archived_history = dict(spawn.observation_history or {})
            if (
                archived_history
                and (
                    not bounded_actor
                    or {"deployable_task_state", "gate_index"}.issubset(archived_history)
                )
            ):
                history.restore(archived_history)
            else:
                history.append(observation)
            prefix_crash = False
            spawn_source = str(spawn.metadata.get("archive_kind", "procedural"))
            teacher_steps = 0
            gate_handoff = None
            prefix_limit = stage.mpcc_prefix_steps
            if spawn_source == "gate":
                handoff = settings.get("archive_gate_handoff_distance", [2.0, 3.0])
                handoff_rng = np.random.default_rng(seed + 49979687 * episode_index)
                gate_handoff = float(handoff_rng.uniform(float(handoff[0]), float(handoff[1])))
                prefix_limit = max(
                    prefix_limit, int(settings.get("archive_gate_handoff_max_steps", 540))
                )
            if prefix_limit:
                expert = make_mpcc(env, settings, stage.action_delay)
                for _ in range(prefix_limit):
                    if gate_handoff is not None and teacher_steps >= stage.mpcc_prefix_steps:
                        distance = float(np.linalg.norm(
                            np.asarray(observation["state"][:3], np.float32)
                            - env.track.gates[env.tracker.index].position
                        ))
                        if distance <= gate_handoff:
                            break
                    command = expert(observation).action.as_array()
                    observation, _, terminated, _, _ = env.step(command)
                    history.append(observation)
                    teacher_steps += 1
                    if terminated:
                        prefix_crash = True
                        break
            sampled_target = (
                gate_chain.sample(
                    len(env.track.gates), seed=seed, episode_index=episode_index
                )
                if training and gate_chain is not None and gate_chain.enabled
                else min(stage.target_gates, len(env.track.gates))
            )
            slot = EpisodeSlot(
                env=env, history=history, observation=observation, track=track_name,
                start_passed=env.tracker.passed_count,
                target_gates=min(
                    (
                        transition_target_gates(
                            settings, spawn_source, sampled_target,
                        )
                        if training else sampled_target
                    ),
                    len(env.track.gates),
                ),
                max_steps=stage.max_steps,
                transitions=[], crashed=prefix_crash, done=prefix_crash,
                spawn_source=spawn_source, teacher_steps=teacher_steps,
                failure_frontier=deque(
                    maxlen=(
                        reset_archive.history_raw_steps
                        if training and reset_archive is not None else 1
                    )
                ),
            )
            if training and gate_chain is not None and gate_chain.enabled:
                slot.max_steps = gate_chain.max_steps(
                    stage.max_steps, slot.target_gates
                )
            if slot.spawn_source != "procedural":
                gate_distance = float(np.linalg.norm(
                    np.asarray(observation["state"][:3], np.float32)
                    - env.track.gates[env.tracker.index].position
                ))
                conservative_speed = max(stage.target_speed * 0.65, 1.0)
                required = int(np.ceil(90.0 * (
                    gate_distance / conservative_speed
                    + float(settings.get("archive_horizon_margin_seconds", 1.0))
                )))
                slot.max_steps = min(
                    max(stage.max_steps, required),
                    int(settings.get("archive_max_steps", 540)),
                )
            slot.failure_frontier.append((
                env.tracker.index, np.asarray(observation["state"], np.float32).copy(),
                np.asarray(observation["previous_action"], np.float32).copy(),
                history.snapshot(),
            ))
            slots.append(slot)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            while any(not slot.done for slot in slots):
                active = [slot for slot in slots if not slot.done]
                batch = _stack_histories([slot.history for slot in active], device)
                if isinstance(actor, (BoundedResidualFlowPolicy, BoundedResidualGaussianPolicy)):
                    observation_tokens, applied_tokens = actor.policy_tokens(batch)
                else:
                    observation_tokens, applied_tokens = tokenizer_policy_tokens(
                        encoder, action_tokenizer, batch,
                        history_raw_steps=actor.history_steps * encoder.temporal_patch_size,
                        amp=amp,
                    )
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=device.startswith("cuda") and amp,
                ):
                    if isinstance(actor, BoundedResidualGaussianPolicy):
                        policy = actor.distribution(observation_tokens)
                        policy_action = (
                            policy.mean
                            if not training
                            else policy.mean + policy.std * torch.randn_like(policy.mean)
                        )
                        action_chunk = actor.latent_to_knot(
                            policy_action, observation_tokens,
                            batch["previous_action"][:, -1],
                        ).float()
                        policy_action = policy_action.float()
                        old_log_probability = policy.log_prob(policy_action).float()
                    elif isinstance(actor, BoundedResidualFlowPolicy):
                        action_chunk, policy_action = actor.sample_policy(
                            observation_tokens, applied_tokens,
                            steps=sampling_steps, method=sampling_method,
                            previous_action=batch["previous_action"][:, -1],
                            deterministic=not training,
                        )
                        action_chunk = action_chunk.float()
                        policy_action = policy_action.float()
                    else:
                        action_chunk = actor.sample(
                            observation_tokens, applied_tokens,
                            steps=sampling_steps, method=sampling_method,
                            previous_action=batch["previous_action"][:, -1],
                            deterministic=not training,
                        ).float()
                        policy_action = action_chunk

                if training:
                    critic_inputs = torch.from_numpy(np.stack([
                        critic_features(
                            slot.observation, slot.env,
                            start_passed=slot.start_passed,
                            target_gates=slot.target_gates,
                            policy_steps=slot.raw_steps,
                            max_steps=slot.max_steps,
                        )
                        for slot in active
                    ])).to(device, non_blocking=True)
                    values = critic(critic_inputs).float()
                    epsilon, flow_time, shortcut_step = sample_cfm_conditions(
                        len(active), samples, actor.action_horizon, actor.action_dim,
                        device=device, dtype=action_chunk.dtype,
                        time_beta=float(settings.get("cfm_time_beta", 1.0)),
                        step_size=1.0 / sampling_steps,
                    )
                    if isinstance(actor, BoundedResidualFlowPolicy):
                        epsilon.mul_(float(actor.flow.source_noise))
                    elif isinstance(actor, BoundedResidualGaussianPolicy):
                        epsilon = flow_time = shortcut_step = None
                else:
                    critic_inputs = values = epsilon = flow_time = shortcut_step = None
                    old_log_probability = None

                valid = torch.zeros(
                    len(active), actor.action_horizon, dtype=torch.bool, device=device
                )
                executed_rewards: list[list[float]] = [[] for _ in active]
                action_numpy = action_chunk.cpu().numpy()
                for offset in range(actor.action_horizon):
                    stepped = [index for index, slot in enumerate(active) if not slot.done]
                    futures = {
                        index: executor.submit(
                            active[index].env.step,
                            normalized_to_ctbr(action_numpy[index, offset]),
                        )
                        for index in stepped
                    }
                    for index in stepped:
                        slot = active[index]
                        observation, reward, terminated, _, info = futures[index].result()
                        slot.observation = observation
                        slot.history.append(observation)
                        slot.raw_steps += 1
                        valid[index, offset] = True
                        gates = slot.env.tracker.passed_count - slot.start_passed
                        slot.success = gates >= slot.target_gates
                        slot.crashed = bool(
                            info.get("ground_contact") or info.get("unity_collision")
                        )
                        if not slot.crashed:
                            slot.failure_frontier.append((
                                slot.env.tracker.index,
                                np.asarray(observation["state"], np.float32).copy(),
                                np.asarray(observation["previous_action"], np.float32).copy(),
                                slot.history.snapshot(),
                            ))
                        if bool(info.get("gate_passed")) and slot.first_gate_step < 0:
                            slot.first_gate_step = slot.raw_steps
                        if (
                            training and reset_archive is not None
                            and bool(info.get("gate_passed"))
                        ):
                            if bool(settings.get("transition_curriculum", {}).get("enabled", False)):
                                reset_archive.record_transition_crossing(
                                    slot.env.track, slot.env.tracker.index,
                                    list(slot.failure_frontier), observation["state"],
                                    observation["previous_action"],
                                    slot.history.snapshot(),
                                )
                            else:
                                reset_archive.record_gate_crossing(
                                    slot.env.track, slot.env.tracker.index,
                                    observation["state"], observation["previous_action"],
                                )
                        adjustment, adjustment_components = transition_crossing_adjustment(
                            slot, info, gates, settings
                        )
                        reward += adjustment
                        for name, value in adjustment_components.items():
                            slot.reward_components[name] = (
                                slot.reward_components.get(name, 0.0) + float(value)
                            )
                        if slot.success:
                            reward += float(settings.get("segment_completion_bonus", 25.0))
                            slot.reward_components["segment_completion"] = (
                                slot.reward_components.get("segment_completion", 0.0)
                                + float(settings.get("segment_completion_bonus", 25.0))
                            )
                        for name, value in info.get("reward_components", {}).items():
                            slot.reward_components[name] = (
                                slot.reward_components.get(name, 0.0) + float(value)
                            )
                        reward = float(reward)
                        executed_rewards[index].append(reward)
                        slot.reward += reward
                        slot.timed_out = bool(
                            not terminated and not slot.success
                            and slot.raw_steps >= slot.max_steps
                        )
                        slot.done = bool(terminated or slot.success or slot.timed_out)
                        if (
                            slot.done and not slot.success and training
                            and reset_archive is not None
                        ):
                            reset_archive.record_failure_frontier(
                                slot.env.track, list(slot.failure_frontier)
                            )

                if not training:
                    continue
                next_inputs = torch.from_numpy(np.stack([
                    critic_features(
                        slot.observation, slot.env,
                        start_passed=slot.start_passed,
                        target_gates=slot.target_gates,
                        policy_steps=slot.raw_steps,
                        max_steps=slot.max_steps,
                    )
                    for slot in active
                ])).to(device, non_blocking=True)
                next_values = critic(next_inputs).float()
                terminal = torch.tensor(
                    [slot.done for slot in active], device=device, dtype=torch.bool
                )
                next_values = torch.where(terminal, torch.zeros_like(next_values), next_values)
                if not isinstance(actor, BoundedResidualGaussianPolicy):
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16,
                        enabled=device.startswith("cuda") and amp,
                    ):
                        old_cfm, _ = conditional_flow_loss(
                            actor, observation_tokens, applied_tokens, policy_action,
                            epsilon, flow_time, shortcut_step, valid_steps=valid,
                            huber_delta=settings.get("cfm_huber_delta"),
                        )
                        _, reference_velocity = conditional_flow_loss(
                            reference, observation_tokens, applied_tokens, policy_action,
                            epsilon, flow_time, shortcut_step, valid_steps=valid,
                            huber_delta=settings.get("cfm_huber_delta"),
                        )
                for index, slot in enumerate(active):
                    rewards = executed_rewards[index]
                    discounted_reward = sum(
                        (gamma_raw ** tick) * reward
                        for tick, reward in enumerate(rewards)
                    )
                    discount = 0.0 if slot.done else gamma_raw ** len(rewards)
                    transition = {
                        "observation_tokens": observation_tokens[index].cpu(),
                        "action_tokens": applied_tokens[index].cpu(),
                        "critic_input": critic_inputs[index].cpu(),
                        "action": policy_action[index].cpu(),
                        "executed_action": action_chunk[index].cpu(),
                        "valid": valid[index].cpu(),
                        "reward": torch.tensor(discounted_reward),
                        "discount": torch.tensor(discount),
                        "value": values[index].cpu(),
                        "next_value": next_values[index].cpu(),
                    }
                    if isinstance(actor, BoundedResidualGaussianPolicy):
                        transition["old_log_prob"] = (
                            old_log_probability[index] * valid[index].float()
                        ).sum().cpu()
                    else:
                        transition.update({
                            "epsilon": epsilon[index].cpu(),
                            "flow_time": flow_time[index].cpu(),
                            "shortcut_step": shortcut_step[index].cpu(),
                            "old_cfm": old_cfm[index].float().cpu(),
                            "reference_velocity": reference_velocity[index].cpu(),
                        })
                    slot.transitions.append(transition)
        results: list[EpisodeResult] = []
        for slot in slots:
            _finish_advantages(slot.transitions, settings)
            results.append(EpisodeResult(
                slot.transitions, slot.success,
                slot.env.tracker.passed_count - slot.start_passed,
                slot.target_gates, slot.crashed, slot.reward, slot.raw_steps, slot.track,
                slot.spawn_source, slot.timed_out, slot.teacher_steps,
                slot.reward_components, slot.first_gate_step,
                (
                    max(0, slot.raw_steps - slot.first_gate_step)
                    if slot.first_gate_step >= 0 else 0
                ),
            ))
        return results
    finally:
        for slot in slots:
            slot.env.close()


def stack_transitions(episodes: list[EpisodeResult]) -> dict[str, torch.Tensor]:
    transitions = [item for episode in episodes for item in episode.transitions]
    if not transitions:
        raise RuntimeError("on-policy collection produced no transitions")
    return {key: torch.stack([item[key] for item in transitions]) for key in transitions[0]}


def next_offline_loss(
    source: OfflineExpertSource,
    actor: TokenizerFlowPolicy,
    encoder,
    action_tokenizer,
    settings: dict[str, Any],
    device: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = move_batch(source.next(), device)
    valid_history = history_valid_steps(
        settings, len(batch["mask"]), batch["mask"].device
    )
    observation_tokens, action_tokens = tokenizer_policy_tokens(
        encoder, action_tokenizer, batch,
        history_raw_steps=source.history_raw,
        valid_history_steps=valid_history,
        amp=bool(settings.get("amp", True)),
    )
    start = source.history_raw - 1
    target = batch["commanded_action"][:, start : start + source.horizon]
    previous = batch["previous_action"][:, source.history_raw - 1]
    return actor.shortcut_training_loss(
        target, observation_tokens, action_tokens, previous_action=previous,
        direct_weight=float(settings.get("shortcut_direct_weight", 0.5)),
        bootstrap_weight=float(settings.get("shortcut_bootstrap_weight", 1.0)),
    )


def update_policy(
    rollout: dict[str, torch.Tensor],
    actor: TokenizerFlowPolicy,
    critic: FlowValueCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    offline: OfflineExpertSource,
    encoder,
    action_tokenizer,
    stage: RacingCurriculumStage,
    settings: dict[str, Any],
    device: str,
) -> dict[str, float]:
    advantages = rollout["advantage"].float()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    advantages = advantages.clamp(
        -float(settings.get("advantage_clip", 10.0)),
        float(settings.get("advantage_clip", 10.0)),
    )
    count = len(advantages)
    batch_size = min(int(settings.get("minibatch_size", 24)), count)
    amp = bool(settings.get("amp", True))
    metrics: dict[str, list[float]] = {}
    target_kl = float(settings.get("target_kl", 0.0))
    anchor_weight = stage.anchor_weight * float(settings.get("anchor_scale", 1.0))
    bc_weight = stage.bc_weight * float(settings.get("bc_scale", 1.0))
    stop_actor = False

    for _ in range(int(settings.get("actor_epochs", 4))):
        for indices in torch.randperm(count).split(batch_size):
            def take(name: str) -> torch.Tensor:
                return rollout[name][indices].to(device, non_blocking=True)
            observation = take("observation_tokens")
            applied = take("action_tokens")
            actions = take("action")
            valid = take("valid")
            epsilon = take("epsilon")
            flow_time = take("flow_time")
            shortcut_step = take("shortcut_step")
            old_cfm = take("old_cfm")
            reference_velocity = take("reference_velocity")
            actor_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda") and amp):
                current_cfm, current_velocity = conditional_flow_loss(
                    actor, observation, applied, actions, epsilon,
                    flow_time, shortcut_step, valid_steps=valid,
                    huber_delta=settings.get("cfm_huber_delta"),
                )
                fpo = fpo_plus_plus_loss(
                    old_cfm, current_cfm, advantages[indices].to(device),
                    clip_epsilon=float(settings.get("clip_epsilon", 0.02)),
                    trust_region=str(settings.get("trust_region", "aspo")),
                    cfm_loss_clamp=float(settings.get("cfm_loss_clamp", 20.0)),
                    negative_cfm_clamp=float(settings.get("negative_cfm_clamp", 20.0)),
                    log_ratio_clamp=float(settings.get("log_ratio_clamp", 10.0)),
                    log_ratio_gain=float(settings.get("log_ratio_gain", 1.0)),
                )
                mask = valid[:, None, :, None].float()
                anchor = (
                    (current_velocity.float() - reference_velocity.float()).square() * mask
                ).sum() / mask.sum().clamp_min(1.0) / actor.action_dim
                loss = fpo.loss + anchor_weight * anchor
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), float(settings.get("actor_grad_clip", 1.0))
            )
            actor_optimizer.step()
            values = {
                "actor_loss": loss.detach(), "fpo_loss": fpo.loss,
                "anchor_loss": anchor.detach(),
                "ratio": fpo.ratio_mean, "ratio_std": fpo.ratio_std,
                "approximate_kl": fpo.approximate_kl,
                "clip_fraction": fpo.clip_fraction,
                "old_cfm_loss": fpo.old_cfm_loss,
                "current_cfm_loss": fpo.current_cfm_loss,
                "actor_gradient_norm": gradient.detach(),
                "log_ratio_gain": torch.as_tensor(
                    float(settings.get("log_ratio_gain", 1.0)), device=device
                ),
            }
            for name, value in values.items():
                metrics.setdefault(name, []).append(float(value.detach()))
            if target_kl > 0 and float(fpo.approximate_kl) > target_kl:
                stop_actor = True
                break
        if stop_actor:
            break

    # Tokenization is frozen and expensive. Keep BC as a small, explicit
    # manifold-preservation phase instead of rebuilding an expert batch inside
    # every FPO minibatch.
    for _ in range(int(settings.get("offline_updates_per_cycle", 4))):
        actor_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=device.startswith("cuda") and amp,
        ):
            if isinstance(actor, BoundedResidualFlowPolicy):
                anchor_count = min(
                    int(settings.get("offline_batch_size", batch_size)), count
                )
                anchor_indices = torch.randperm(count)[:anchor_count]
                bc_loss, bc_metrics = actor.zero_residual_loss(
                    rollout["observation_tokens"][anchor_indices].to(device),
                    rollout["action_tokens"][anchor_indices].to(device),
                )
            else:
                bc_loss, bc_metrics = next_offline_loss(
                    offline, actor, encoder, action_tokenizer, settings, device
                )
            weighted_bc = bc_weight * bc_loss
        weighted_bc.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), float(settings.get("actor_grad_clip", 1.0))
        )
        actor_optimizer.step()
        metrics.setdefault("bc_loss", []).append(float(bc_loss.detach()))
        metrics.setdefault("bc_gradient_norm", []).append(float(gradient.detach()))
        metrics.setdefault("shortcut_bootstrap_loss", []).append(float(
            bc_metrics["shortcut_bootstrap_loss"].detach()
        ))

    for _ in range(int(settings.get("critic_epochs", 8))):
        for indices in torch.randperm(count).split(batch_size):
            critic_optimizer.zero_grad(set_to_none=True)
            inputs = rollout["critic_input"][indices].to(device)
            targets = rollout["return"][indices].to(device)
            predicted = critic(inputs)
            value_loss = torch.nn.functional.smooth_l1_loss(predicted, targets)
            value_loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), float(settings.get("critic_grad_clip", 5.0))
            )
            critic_optimizer.step()
            metrics.setdefault("critic_loss", []).append(float(value_loss.detach()))
            metrics.setdefault("critic_gradient_norm", []).append(float(gradient.detach()))
    metrics["advantage_mean"] = [float(advantages.mean())]
    metrics["advantage_std"] = [float(advantages.std(unbiased=False))]
    metrics["return_mean"] = [float(rollout["return"].mean())]
    metrics["actor_early_stop"] = [float(stop_actor)]
    metrics["on_policy_transitions"] = [float(count)]
    metrics["effective_anchor_weight"] = [float(anchor_weight)]
    metrics["effective_bc_weight"] = [float(bc_weight)]
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def update_gaussian_residual_policy(
    rollout: dict[str, torch.Tensor],
    actor: BoundedResidualGaussianPolicy,
    reference: BoundedResidualGaussianPolicy,
    critic: FlowValueCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    stage: RacingCurriculumStage,
    settings: dict[str, Any],
    device: str,
) -> dict[str, float]:
    """Exact PPO over the raw pre-integrator Gaussian residual latent."""

    advantages = rollout["advantage"].float()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    advantages = advantages.clamp(
        -float(settings.get("advantage_clip", 8.0)),
        float(settings.get("advantage_clip", 8.0)),
    )
    count = len(advantages)
    batch_size = min(int(settings.get("minibatch_size", 256)), count)
    amp = bool(settings.get("amp", True))
    clip = float(settings.get("clip_epsilon", 0.10))
    target_kl = float(settings.get("target_kl", 0.012))
    entropy_weight = float(settings.get("entropy_weight", 0.001))
    anchor_weight = stage.anchor_weight * float(settings.get("anchor_scale", 1.0))
    zero_weight = stage.bc_weight * float(settings.get("bc_scale", 1.0))
    metrics: dict[str, list[float]] = {}
    stop_actor = False
    for _ in range(int(settings.get("actor_epochs", 4))):
        for indices in torch.randperm(count).split(batch_size):
            observation = rollout["observation_tokens"][indices].to(device, non_blocking=True)
            latent = rollout["action"][indices].to(device, non_blocking=True)
            valid = rollout["valid"][indices].to(device, non_blocking=True).float()
            old_log_probability = rollout["old_log_prob"][indices].to(device)
            advantage = advantages[indices].to(device)
            actor_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16,
                enabled=device.startswith("cuda") and amp,
            ):
                policy = actor.distribution(observation)
                log_probability = (policy.log_prob(latent).float() * valid).sum(dim=-1)
                log_ratio = (log_probability - old_log_probability).clamp(-10.0, 10.0)
                ratio = log_ratio.exp()
                surrogate = torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1.0 - clip, 1.0 + clip) * advantage,
                )
                policy_loss = -surrogate.mean()
                entropy = (policy.entropy().float() * valid).sum(dim=-1).mean()
                with torch.no_grad():
                    prior = reference.distribution(observation)
                anchor = diagonal_gaussian_kl(policy, prior).mean()
                zero_residual = policy.mean.float().square().mean()
                loss = (
                    policy_loss - entropy_weight * entropy
                    + anchor_weight * anchor + zero_weight * zero_residual
                )
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), float(settings.get("actor_grad_clip", 1.0))
            )
            actor_optimizer.step()
            approximate_kl = ((ratio - 1.0) - log_ratio).mean()
            values = {
                "actor_loss": loss.detach(), "policy_loss": policy_loss.detach(),
                "entropy": entropy.detach(), "anchor_loss": anchor.detach(),
                "zero_residual_loss": zero_residual.detach(),
                "ratio": ratio.mean().detach(),
                "ratio_std": ratio.std(unbiased=False).detach(),
                "approximate_kl": approximate_kl.detach(),
                "clip_fraction": ((ratio - 1.0).abs() > clip).float().mean(),
                "policy_std": policy.std.mean().detach(),
                "policy_mean_abs": policy.mean.abs().mean().detach(),
                "actor_gradient_norm": gradient.detach(),
            }
            for name, value in values.items():
                metrics.setdefault(name, []).append(float(value))
            if target_kl > 0 and float(approximate_kl.detach()) > target_kl:
                stop_actor = True
                break
        if stop_actor:
            break

    for _ in range(int(settings.get("critic_epochs", 4))):
        for indices in torch.randperm(count).split(batch_size):
            critic_optimizer.zero_grad(set_to_none=True)
            predicted = critic(rollout["critic_input"][indices].to(device))
            target = rollout["return"][indices].to(device)
            value_loss = torch.nn.functional.smooth_l1_loss(predicted, target)
            value_loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), float(settings.get("critic_grad_clip", 5.0))
            )
            critic_optimizer.step()
            metrics.setdefault("critic_loss", []).append(float(value_loss.detach()))
            metrics.setdefault("critic_gradient_norm", []).append(float(gradient.detach()))
    metrics.update({
        "advantage_mean": [float(advantages.mean())],
        "advantage_std": [float(advantages.std(unbiased=False))],
        "return_mean": [float(rollout["return"].mean())],
        "actor_early_stop": [float(stop_actor)],
        "on_policy_transitions": [float(count)],
        "effective_anchor_weight": [float(anchor_weight)],
        "effective_bc_weight": [float(zero_weight)],
    })
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def outcome_metrics(results: list[EpisodeResult]) -> dict[str, float]:
    if not results:
        return {"success_rate": 0.0, "gate_completion": 0.0, "crash_rate": 1.0}
    metrics = {
        "success_rate": float(np.mean([item.success for item in results])),
        "gate_completion": float(np.mean([
            min(1.0, item.gates_passed / max(item.target_gates, 1)) for item in results
        ])),
        "crash_rate": float(np.mean([item.crashed for item in results])),
        "episode_return": float(np.mean([item.reward for item in results])),
        "gates_passed": float(np.mean([item.gates_passed for item in results])),
        "environment_steps": float(sum(item.raw_steps for item in results)),
        "teacher_steps": float(sum(item.teacher_steps for item in results)),
        "probability_at_least_1_gate": float(np.mean([
            item.gates_passed >= 1 for item in results
        ])),
        "probability_at_least_2_gates": float(np.mean([
            item.gates_passed >= 2 for item in results
        ])),
        "probability_at_least_3_gates": float(np.mean([
            item.gates_passed >= 3 for item in results
        ])),
        "post_gate_survival_steps": float(np.mean([
            item.post_gate_steps for item in results if item.first_gate_step >= 0
        ])) if any(item.first_gate_step >= 0 for item in results) else 0.0,
    }
    action_chunks = [
        transition.get("executed_action", transition["action"]).float().numpy()
        for item in results for transition in item.transitions
        if "action" in transition
    ]
    if action_chunks:
        actions = np.stack(action_chunks).reshape(len(action_chunks), -1)
        metrics["exploration_action_std"] = float(actions.std(axis=0).mean())
        metrics["exploration_action_saturation"] = float(np.mean(np.abs(actions) >= 0.98))
        if len(actions) > 1:
            covariance = np.cov(actions, rowvar=False)
            eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
            probability = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
            metrics["exploration_action_effective_rank"] = float(
                np.exp(-np.sum(probability * np.log(probability + 1e-12)))
            )
        else:
            metrics["exploration_action_effective_rank"] = 0.0
        chunks = np.stack(action_chunks)
        metrics["exploration_within_chunk_slew"] = float(
            np.abs(np.diff(chunks, axis=1)).mean()
        )
    successful_returns = np.asarray(
        [item.reward for item in results if item.success], np.float64
    )
    failed_returns = np.asarray(
        [item.reward for item in results if not item.success], np.float64
    )
    metrics["successful_episode_return"] = (
        float(successful_returns.mean()) if successful_returns.size else 0.0
    )
    metrics["failed_episode_return"] = (
        float(failed_returns.mean()) if failed_returns.size else 0.0
    )
    metrics["outcome_return_gap"] = (
        metrics["successful_episode_return"] - metrics["failed_episode_return"]
    )
    metrics["outcome_return_auc"] = (
        float(np.mean(
            (successful_returns[:, None] > failed_returns[None, :])
            + 0.5 * (successful_returns[:, None] == failed_returns[None, :])
        ))
        if successful_returns.size and failed_returns.size else 0.5
    )
    component_names = sorted({
        name for item in results for name in item.reward_components
    })
    for name in component_names:
        metrics[f"reward_component_{name}"] = float(np.mean([
            item.reward_components.get(name, 0.0) for item in results
        ]))
    for source in (
        "procedural", "intergate", "gate", "failure",
        "transition_pre", "transition_post",
    ):
        metrics[f"spawn_{source}_fraction"] = float(np.mean([
            item.spawn_source == source for item in results
        ]))
        source_results = [item for item in results if item.spawn_source == source]
        metrics[f"spawn_{source}_success"] = (
            float(np.mean([item.success for item in source_results]))
            if source_results else 0.0
        )
        metrics[f"spawn_{source}_crash"] = (
            float(np.mean([item.crashed for item in source_results]))
            if source_results else 0.0
        )
        metrics[f"spawn_{source}_timeout"] = (
            float(np.mean([item.timed_out for item in source_results]))
            if source_results else 0.0
        )
        metrics[f"spawn_{source}_teacher_steps"] = (
            float(np.mean([item.teacher_steps for item in source_results]))
            if source_results else 0.0
        )
    for target_gates in sorted({item.target_gates for item in results}):
        target_results = [
            item for item in results if item.target_gates == target_gates
        ]
        metrics[f"target_{target_gates}_fraction"] = len(target_results) / len(results)
        metrics[f"target_{target_gates}_success"] = float(np.mean([
            item.success for item in target_results
        ]))
        metrics[f"target_{target_gates}_gates_passed"] = float(np.mean([
            item.gates_passed for item in target_results
        ]))
    return metrics


def collect_batched_episodes(
    actor: TokenizerFlowPolicy,
    reference: TokenizerFlowPolicy,
    encoder,
    action_tokenizer,
    critic: FlowValueCritic,
    stage: RacingCurriculumStage,
    settings: dict[str, Any],
    *, count: int, episode_start: int, seed_base: int, device: str, training: bool,
    reset_archive: RaceResetArchive | None = None,
    gate_chain: GateChainCurriculum | None = None,
) -> tuple[list[EpisodeResult], int]:
    results: list[EpisodeResult] = []
    attempted = 0
    parallel = max(1, int(settings.get("rollout_envs", 8)))
    maximum_attempts = count * 3 if training else count
    while len(results) < count and attempted < maximum_attempts:
        batch_count = min(
            parallel, maximum_attempts - attempted, count - len(results)
        )
        indices = [episode_start + attempted + index for index in range(batch_count)]
        seeds = [seed_base + 1009 * index for index in indices]
        batch = run_episode_batch(
            actor, reference, encoder, action_tokenizer, critic, stage, settings,
            episode_indices=indices, seeds=seeds, device=device, training=training,
            reset_archive=reset_archive,
            gate_chain=gate_chain,
        )
        attempted += batch_count
        if training:
            results.extend(item for item in batch if item.transitions)
        else:
            results.extend(batch)
    return results[:count], attempted


def main() -> None:
    args, config = parse_args()
    settings = config["flow_fpo"]
    # Keep checkpoint and W&B resume semantics identical when the CLI owns the
    # override (for example after revising a curriculum boundary).
    settings["resume"] = bool(args.resume)
    device = str(args.device)
    seed = int(settings.get("seed", 20260815))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(settings.get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", True))

    initial_path = Path(settings["actor"])
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    actor_family = str(settings.get("actor_family", "tokenizer_flow"))
    bounded_family = actor_family in {
        "bounded_residual_flow", "bounded_residual_gaussian"
    }
    if bounded_family:
        base = BoundedResidualKnotPolicy(**initial["model_config"])
        base.load_state_dict(initial["model"])
        if actor_family == "bounded_residual_gaussian":
            actor = BoundedResidualGaussianPolicy(
                base,
                history_raw_steps=int(settings.get("history_raw_steps", 9)),
                correction_logit_scale=float(settings.get("correction_logit_scale", 0.35)),
                **dict(settings.get("gaussian_residual", {})),
            ).to(device)
        else:
            actor = BoundedResidualFlowPolicy(
                base,
                history_raw_steps=int(settings.get("history_raw_steps", 9)),
                correction_logit_scale=float(settings.get("correction_logit_scale", 0.35)),
                flow=settings.get("residual_flow"),
            ).to(device)
        encoder = action_tokenizer = None
        encoder_config = None
        observation_path = action_path = None
        patch = actor.action_horizon
    else:
        actor = TokenizerFlowPolicy(**initial["model_config"])
        actor.load_state_dict(initial["model"])
        actor.to(device)
        loaded = load_frozen_actor_stack(
            None, Path(initial["observation_tokenizer"]),
            Path(initial["action_tokenizer"]), device,
        )
        _, encoder, action_tokenizer, encoder_config, observation_path, action_path, _ = loaded
        encoder.eval().requires_grad_(False).to(device)
        action_tokenizer.eval().requires_grad_(False).to(device)
        patch = int(encoder.temporal_patch_size)
        if actor.history_steps != 3 or actor.action_horizon != patch:
            raise ValueError("FPO curriculum requires the three-history, one-macro flow actor")
    reference = copy.deepcopy(actor).eval().requires_grad_(False).to(device)
    critic = FlowValueCritic(
        CRITIC_INPUT_DIM, int(settings.get("critic_hidden_dim", 512))
    ).to(device)
    stages = [RacingCurriculumStage.from_mapping(item) for item in settings["curriculum"]]
    curriculum = RacingCurriculum(
        stages, stage_index=int(settings.get("initial_curriculum_stage", 0))
    )
    reset_archive = RaceResetArchive(settings.get("reset_curriculum"))
    gate_chain = GateChainCurriculum(settings.get("gate_chain_curriculum"))
    configure_actor_scope(actor, actor_scope(curriculum.current, settings))
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=float(settings.get("actor_learning_rate", 5e-6)),
        betas=tuple(settings.get("adam_betas", [0.9, 0.95])),
        weight_decay=float(settings.get("actor_weight_decay", 1e-4)),
        fused=device.startswith("cuda") and bool(settings.get("fused_optimizer", True)),
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=float(settings.get("critic_learning_rate", 1e-4)),
        weight_decay=float(settings.get("critic_weight_decay", 1e-4)),
        fused=device.startswith("cuda") and bool(settings.get("fused_optimizer", True)),
    )
    offline = None if bounded_family else OfflineExpertSource(settings, encoder_config, patch)
    manager = CheckpointManager.from_config(config)
    logger = init_wandb(config)
    start_cycle = 0
    environment_steps = 0
    resumed = manager.resume(bool(args.resume), map_location=device)
    if resumed:
        actor.load_state_dict(resumed["model"])
        critic.load_state_dict(resumed["critic"])
        actor_optimizer.load_state_dict(resumed["actor_optimizer"])
        critic_optimizer.load_state_dict(resumed["critic_optimizer"])
        curriculum.load_state_dict(resumed["curriculum"])
        reset_archive.load_state_dict(resumed.get("reset_archive"))
        gate_chain.load_state_dict(resumed.get("gate_chain_curriculum"))
        restore_rng_state(resumed.get("rng_state"))
        start_cycle = int(resumed.get("cycle", 0))
        environment_steps = int(resumed.get("environment_steps", 0))
        episode_counter = int(resumed.get("episode_counter", 0))
    else:
        episode_counter = 0
        archive_path = settings.get("reset_archive_checkpoint")
        if archive_path:
            archive_checkpoint = torch.load(
                Path(archive_path), map_location="cpu", weights_only=False
            )
            archive_state = copy.deepcopy(archive_checkpoint.get("reset_archive") or {})
            entries = list(archive_state.get("entries", []))
            archive_state["entries"] = [
                entry for entry in entries
                if entry.get("observation_history") is not None
                and {"deployable_task_state", "gate_index"}.issubset(
                    entry["observation_history"]
                )
            ]
            reset_archive.load_state_dict(archive_state)
            print(
                f"reset_archive current_contract={len(archive_state['entries'])}/"
                f"{len(entries)} source={archive_path}", flush=True,
            )
    trainable, total = configure_actor_scope(
        actor, actor_scope(curriculum.current, settings)
    )
    print(
        f"flow_fpo contract={'exact_gaussian_residual_ppo' if isinstance(actor, BoundedResidualGaussianPolicy) else 'chunk_level_fpo_plus_plus_' + settings.get('trust_region', 'aspo')} "
        f"actor={initial_path} "
        f"stage={curriculum.current.name} trainable={trainable:,}/{total:,} "
        f"cfm_samples={0 if isinstance(actor, BoundedResidualGaussianPolicy) else settings.get('cfm_samples', 8)} "
        f"shortcut_anchor={'zero_residual_v1' if bounded_family else 'offline_expert'}",
        flush=True,
    )
    last_eval_metrics: dict[str, float] = {}
    for cycle in range(start_cycle, int(settings.get("cycles", 200))):
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        cycle_started = time.perf_counter()
        stage = curriculum.current
        trainable, total = configure_actor_scope(actor, actor_scope(stage, settings))
        requested = int(settings.get("episodes_per_cycle", 6))
        collection_started = time.perf_counter()
        collected, attempts = collect_batched_episodes(
            actor, reference, encoder, action_tokenizer, critic, stage, settings,
            count=requested, episode_start=episode_counter, seed_base=seed,
            device=device, training=True, reset_archive=reset_archive,
            gate_chain=gate_chain,
        )
        episode_counter += attempts
        collection_seconds = time.perf_counter() - collection_started
        if len(collected) < requested:
            raise RuntimeError(
                f"only {len(collected)}/{requested} rollout episodes produced transitions"
            )
        reset_archive.observe_outcomes([
            (item.spawn_source, item.success) for item in collected
        ])
        gate_chain.observe([
            (
                item.target_gates,
                min(1.0, item.gates_passed / max(item.target_gates, 1)),
            )
            for item in collected
        ])
        rollout = stack_transitions(collected)
        collected_steps = int(sum(item.raw_steps for item in collected))
        environment_steps += collected_steps
        update_started = time.perf_counter()
        if isinstance(actor, BoundedResidualGaussianPolicy):
            update = update_gaussian_residual_policy(
                rollout, actor, reference, critic, actor_optimizer,
                critic_optimizer, stage, settings, device,
            )
        else:
            update = update_policy(
                rollout, actor, critic, actor_optimizer, critic_optimizer,
                offline, encoder, action_tokenizer, stage, settings, device,
            )
        update_seconds = time.perf_counter() - update_started
        train_metrics = outcome_metrics(collected)
        eval_interval = max(1, int(settings.get("evaluation_interval", 1)))
        evaluate = (cycle + 1) % eval_interval == 0 or cycle == start_cycle
        evaluation_seconds = 0.0
        previous_stage = curriculum.stage_index
        advanced = False
        eval_metrics: dict[str, float] = {}
        if evaluate:
            evaluation_started = time.perf_counter()
            evaluation, _ = collect_batched_episodes(
                actor, reference, encoder, action_tokenizer, critic, stage, settings,
                count=int(settings.get("evaluation_episodes", 8)), episode_start=0,
                seed_base=int(settings.get("evaluation_seed", 20260816)),
                device=device, training=False,
            )
            evaluation_seconds = time.perf_counter() - evaluation_started
            eval_metrics = outcome_metrics(evaluation)
            advanced = curriculum.observe([item.success for item in evaluation])
            if advanced:
                trainable, total = configure_actor_scope(
                    actor, actor_scope(curriculum.current, settings)
                )
            eval_metrics["curriculum_score"] = previous_stage + eval_metrics["success_rate"]
            eval_metrics["chain_selection_score"] = chain_selection_score(
                eval_metrics, settings
            )
            last_eval_metrics = dict(eval_metrics)
        cycle_seconds = time.perf_counter() - cycle_started
        logged_train = {
            **update,
            **{f"collection_{key}": value for key, value in train_metrics.items()},
            "environment_steps": environment_steps,
            "curriculum_stage": previous_stage,
            "curriculum_stage_episodes": curriculum.stage_episodes,
            "curriculum_window_success": curriculum.success_rate,
            "trainable_fraction": trainable / total,
            "collection_seconds": collection_seconds,
            "update_seconds": update_seconds,
            "evaluation_seconds": evaluation_seconds,
            "cycle_seconds": cycle_seconds,
            "collection_env_steps_per_second": collected_steps / max(collection_seconds, 1e-6),
            "effective_env_steps_per_second": collected_steps / max(cycle_seconds, 1e-6),
            "rollout_envs": float(settings.get("rollout_envs", 8)),
            **reset_archive.metrics(),
            **gate_chain.metrics(),
        }
        if device.startswith("cuda"):
            logged_train.update({
                "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            })
        logged_eval = ({
            **eval_metrics, "environment_steps": environment_steps,
            "curriculum_stage": previous_stage,
            "curriculum_advanced": float(advanced),
        } if evaluate else {})
        print(
            f"cycle={cycle + 1} env_steps={environment_steps} stage={stage.name} "
            f"train_success={train_metrics['success_rate']:.3f} "
            f"eval_success={eval_metrics.get('success_rate', float('nan')):.3f} "
            f"policy_loss={update.get('fpo_loss', update.get('policy_loss', 0.0)):.5f} "
            f"ratio={update['ratio']:.5f} "
            f"kl={update['approximate_kl']:.6f} "
            f"clip={update['clip_fraction']:.4f} "
            f"spawn=p{train_metrics['spawn_procedural_fraction']:.2f}/"
            f"i{train_metrics['spawn_intergate_fraction']:.2f}/"
            f"g{train_metrics['spawn_gate_fraction']:.2f}/"
            f"f{train_metrics['spawn_failure_fraction']:.2f}/"
            f"pre{train_metrics['spawn_transition_pre_fraction']:.2f}/"
            f"post{train_metrics['spawn_transition_post_fraction']:.2f} "
            f"spawn_success=p{train_metrics['spawn_procedural_success']:.2f}/"
            f"i{train_metrics['spawn_intergate_success']:.2f}/"
            f"g{train_metrics['spawn_gate_success']:.2f}/"
            f"f{train_metrics['spawn_failure_success']:.2f}/"
            f"pre{train_metrics['spawn_transition_pre_success']:.2f}/"
            f"post{train_metrics['spawn_transition_post_success']:.2f} "
            f"spawn_timeout=p{train_metrics['spawn_procedural_timeout']:.2f}/"
            f"i{train_metrics['spawn_intergate_timeout']:.2f}/"
            f"g{train_metrics['spawn_gate_timeout']:.2f}/"
            f"f{train_metrics['spawn_failure_timeout']:.2f} "
            f"targets=1:{train_metrics.get('target_1_fraction', 0.0):.2f}"
            f"@{train_metrics.get('target_1_success', 0.0):.2f}/"
            f"2:{train_metrics.get('target_2_fraction', 0.0):.2f}"
            f"@{train_metrics.get('target_2_success', 0.0):.2f} "
            f"chain=p1:{train_metrics['probability_at_least_1_gate']:.2f}/"
            f"p2:{train_metrics['probability_at_least_2_gates']:.2f} "
            f"explore_std={train_metrics.get('exploration_action_std', 0.0):.3f} "
            f"collect_s={collection_seconds:.1f} update_s={update_seconds:.1f} "
            f"steps_s={collected_steps / max(cycle_seconds, 1e-6):.1f} "
            f"vram_gb={logged_train.get('cuda_peak_reserved_gb', 0.0):.2f} "
            f"advanced={int(advanced)}",
            flush=True,
        )
        logger.log_train(logged_train, environment_steps)
        if evaluate:
            logger.log_eval(logged_eval, environment_steps)
            manager.save_eval(
                logged_eval, step=environment_steps,
                summary=f"Deterministic RL curriculum evaluation at stage {stage.name}.",
                probes={"stage": stage.name, "curriculum": curriculum.state_dict()},
                kind="curriculum-flight",
            )
        checkpoint_payload = {
            "model": actor.state_dict(), "critic": critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "rng_state": capture_rng_state(),
            "model_config": (
                actor.model_config()
                if isinstance(actor, (BoundedResidualFlowPolicy, BoundedResidualGaussianPolicy))
                else initial["model_config"]
            ),
            "base_model_config": (
                initial["model_config"]
                if isinstance(actor, (BoundedResidualFlowPolicy, BoundedResidualGaussianPolicy))
                else None
            ),
            "base_checkpoint": (
                str(initial_path)
                if isinstance(actor, (BoundedResidualFlowPolicy, BoundedResidualGaussianPolicy))
                else None
            ),
            "training_config": config,
            "observation_tokenizer": (
                None if observation_path is None else str(observation_path)
            ),
            "action_tokenizer": None if action_path is None else str(action_path),
            "input_contract": (
                ({
                    **initial["input_contract"],
                    "rl_adapter": "bounded_residual_flow_correction",
                    "rl_latent": "pre_integrator_bounded_flow_residual",
                    "offline_anchor": "dagger_v1_zero_residual",
                } if isinstance(actor, BoundedResidualFlowPolicy) else {
                    **initial["input_contract"],
                    "rl_adapter": "bounded_residual_exact_gaussian_correction",
                    "rl_latent": "exact_gaussian_pre_integrator_residual",
                    "offline_anchor": "dagger_v1_zero_residual",
                })
                if isinstance(actor, (BoundedResidualFlowPolicy, BoundedResidualGaussianPolicy))
                else initial["input_contract"]
            ),
            "curriculum": curriculum.state_dict(),
            "reset_archive": reset_archive.state_dict(),
            "gate_chain_curriculum": gate_chain.state_dict(),
            "cycle": cycle + 1,
            "environment_steps": environment_steps,
            "episode_counter": episode_counter,
            "initial_actor": str(initial_path),
            "algorithm": (
                "Exact Gaussian PPO over pre-integrator bounded residual"
                if isinstance(actor, BoundedResidualGaussianPolicy)
                else "FPO++ per-sample CFM ratio with "
                f"{settings.get('trust_region', 'aspo').upper()}"
            ),
        }
        checkpoint_interval = max(1, int(settings.get("checkpoint_interval", eval_interval)))
        if evaluate or (cycle + 1) % checkpoint_interval == 0:
            manager.save(
                checkpoint_payload, step=environment_steps,
                metrics=logged_eval if evaluate else last_eval_metrics,
                rank=evaluate,
            )
    logger.finish()


if __name__ == "__main__":
    main()
