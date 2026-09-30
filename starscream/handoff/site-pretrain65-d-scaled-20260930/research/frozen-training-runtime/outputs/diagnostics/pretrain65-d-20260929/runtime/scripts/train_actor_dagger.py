#!/usr/bin/env python3
"""DAgger distillation for the frozen-tokenizer causal Gaussian actor."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import random
import time
import traceback
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from starscream.actor_critic import GaussianActor, build_actor_state
from starscream.checkpoint_manager import CheckpointManager, capture_rng_state
from starscream.env import FlightmareEnv
from starscream.racing_curriculum import RacingCurriculumStage, sample_curriculum_spawn
from starscream.wandb import init_wandb

try:
    from scripts.eval_flight import ObservationHistory, ctbr_to_normalized, normalized_to_ctbr
    from scripts.train_actor_bc import actor_bc_objective, load_frozen_actor_stack
    from scripts.train_flow_policy_dagger import expert_label, schedule
    from scripts.train_flow_policy_fpo import make_mpcc
except ModuleNotFoundError:
    from eval_flight import ObservationHistory, ctbr_to_normalized, normalized_to_ctbr
    from train_actor_bc import actor_bc_objective, load_frozen_actor_stack
    from train_flow_policy_dagger import expert_label, schedule
    from train_flow_policy_fpo import make_mpcc


_TEACHER_CACHE: dict[tuple[str, float], Any] = {}


@dataclass(slots=True)
class DAggerSample:
    state: torch.Tensor
    target: torch.Tensor
    action_timing: torch.Tensor
    priority: float
    transition: bool
    intervention: bool


@dataclass(slots=True)
class RawDAggerSample:
    history: dict[str, np.ndarray]
    target: torch.Tensor
    action_timing: torch.Tensor
    priority: float


class PrioritizedReplay:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.count = 0
        self.cursor = 0
        self.rng = np.random.default_rng(seed)
        self.states: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None
        self.action_timings: torch.Tensor | None = None
        self.priorities = np.zeros(self.capacity, np.float32)
        self.transitions = np.zeros(self.capacity, np.bool_)
        self.interventions = np.zeros(self.capacity, np.bool_)

    def __len__(self) -> int:
        return self.count

    def _allocate(self, sample: DAggerSample) -> None:
        self.states = torch.empty(
            (self.capacity, *sample.state.shape), dtype=torch.float16
        )
        self.targets = torch.empty(
            (self.capacity, *sample.target.shape), dtype=torch.float16
        )
        self.action_timings = torch.empty(
            (self.capacity, *sample.action_timing.shape), dtype=torch.float16
        )

    def add(self, sample: DAggerSample) -> None:
        if self.states is None:
            self._allocate(sample)
        assert self.states is not None and self.targets is not None
        assert self.action_timings is not None
        index = self.cursor
        self.states[index].copy_(sample.state)
        self.targets[index].copy_(sample.target)
        self.action_timings[index].copy_(sample.action_timing)
        self.priorities[index] = float(sample.priority)
        self.transitions[index] = bool(sample.transition)
        self.interventions[index] = bool(sample.intervention)
        self.cursor = (self.cursor + 1) % self.capacity
        self.count = min(self.capacity, self.count + 1)

    def sample(self, count: int, device: str) -> dict[str, torch.Tensor]:
        if self.count <= 0 or self.states is None:
            raise RuntimeError("cannot sample an empty DAgger replay")
        priorities = self.priorities[:self.count].astype(np.float64)
        indices = self.rng.choice(
            self.count, size=int(count), replace=self.count < count,
            p=priorities / priorities.sum(),
        )
        tensor_indices = torch.from_numpy(indices.astype(np.int64, copy=False))
        assert self.targets is not None and self.action_timings is not None
        return {
            "state": self.states.index_select(0, tensor_indices).to(device).float(),
            "target": self.targets.index_select(0, tensor_indices).to(device).float(),
            "action_timing": self.action_timings.index_select(0, tensor_indices).to(
                device
            ).float(),
        }

    def metrics(self) -> dict[str, float]:
        count = max(1, self.count)
        allocated = (
            sum(tensor.numel() * tensor.element_size() for tensor in (
                self.states, self.targets, self.action_timings
            ) if tensor is not None)
        )
        return {
            "replay_size": float(self.count),
            "replay_allocated_gib": allocated / (1024.0 ** 3),
            "replay_transition_fraction": float(self.transitions[:self.count].mean()),
            "replay_intervention_fraction": float(self.interventions[:self.count].mean()),
            "replay_priority_mean": float(self.priorities[:self.count].mean()),
        }


class RawHistoryReplay:
    """Small bit-packed replay for conservative observation-encoder adaptation."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.samples: list[RawDAggerSample] = []
        self.cursor = 0
        self.rng = np.random.default_rng(seed)

    def add(self, sample: RawDAggerSample) -> None:
        if self.capacity <= 0:
            return
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
        else:
            self.samples[self.cursor] = sample
            self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, count: int, device: str) -> dict[str, torch.Tensor]:
        priorities = np.asarray([sample.priority for sample in self.samples], np.float64)
        indices = self.rng.choice(
            len(self.samples), size=int(count), replace=len(self.samples) < count,
            p=priorities / priorities.sum(),
        )
        selected = [self.samples[int(index)] for index in indices]
        histories: list[dict[str, np.ndarray]] = []
        for sample in selected:
            snapshot = sample.history
            width = int(np.asarray(snapshot["mask_width"]).item())
            histories.append({
                "mask": np.unpackbits(
                    np.asarray(snapshot["mask_bits"], np.uint8),
                    axis=-1, count=width,
                ).astype(np.float32),
                **{
                    key: np.asarray(snapshot[key], np.float32)
                    for key in ("proprio", "route", "timing", "previous_action", "estimate")
                    if key in snapshot
                },
            })
        keys = histories[0].keys()
        result = {
            key: torch.from_numpy(np.stack([history[key] for history in histories])).to(
                device, non_blocking=True
            )
            for key in keys
        }
        result.update(
            target=torch.stack([sample.target for sample in selected]).to(device).float(),
            action_timing=torch.stack(
                [sample.action_timing for sample in selected]
            ).to(device).float(),
        )
        return result

    def metrics(self) -> dict[str, float]:
        return {"raw_replay_size": float(len(self.samples))}


def arguments() -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    preliminary, _ = parser.parse_known_args()
    with preliminary.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    parser.set_defaults(device=str(config["actor_dagger"].get("device", "cuda")))
    args = parser.parse_args()
    if args.smoke:
        config = copy.deepcopy(config)
        config["actor_dagger"].update(
            rounds=1, episodes_per_round=2, evaluation_episodes=2,
            updates_per_round=2, batch_size=2, max_steps=18,
            replay_capacity=64, raw_replay_capacity=8,
            rollout_envs=2, progress_log_interval=1,
            encoder_adaptation_start_round=1,
            encoder_update_fraction=1.0, encoder_batch_size=2,
        )
        config["actor_dagger"]["stage"]["max_steps"] = 18
        config["output_root"] = "/tmp/starscream-actor-dagger-smoke"
        config["checkpoint"] = {
            "run_name": "deployment-actor-dagger-smoke",
            "monitor": "selection_score", "mode": "max", "top_k": 1,
        }
        config["wandb"] = {
            "enabled": False, "run_name": "deployment-actor-dagger-smoke"
        }
    return args, config


def cached_teacher(env: FlightmareEnv, settings: dict[str, Any], delay: float):
    key = (env.track.fingerprint, float(delay))
    teacher = _TEACHER_CACHE.get(key)
    if teacher is None:
        teacher = make_mpcc(env, settings, delay)
        _TEACHER_CACHE[key] = teacher
    teacher.reset()
    return teacher


@torch.no_grad()
def actor_output(actor, encoder, action_tokenizer, history, device: str, amp: bool):
    batch = history.batch(device, repeat_first_padding=True)
    state = build_actor_state(encoder, batch, history.length, device, amp)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16,
        enabled=device.startswith("cuda") and amp,
    ):
        output = actor(state, action_tokenizer)
    return state[0].half().cpu(), output.mean[0, 0].float().cpu().numpy(), float(output.std.mean())


@torch.no_grad()
def collect_episode(
    actor: GaussianActor, encoder, action_tokenizer, replay: PrioritizedReplay,
    raw_replay: RawHistoryReplay,
    stage: RacingCurriculumStage, settings: dict[str, Any], *,
    round_index: int, episode_index: int, device: str, evaluation: bool,
) -> dict[str, float]:
    rounds = int(settings.get("rounds", 8))
    round_seed = 0 if evaluation else 1000003 * round_index
    seed = int(settings.get("seed", 0)) + round_seed + 7919 * episode_index
    rng = np.random.default_rng(seed)
    track_name = stage.tracks[episode_index % len(stage.tracks)]
    env = FlightmareEnv(
        track=track_name, next_gates=3, image_size=(160, 128),
        control_dt=1.0 / 90.0, render_observations=False,
        mask_source="geometry", mask_size=(160, 128),
        geometry_renderer=str(settings.get("geometry_renderer", "exact")),
        image_delay=float(settings.get("image_delay", 0.033)),
        action_delay=stage.action_delay, terminate_on_collision=True,
    )
    teacher = None
    try:
        spawn = sample_curriculum_spawn(
            env.track, stage, seed=seed, episode_index=episode_index
        )
        options: dict[str, Any] = {
            "gate_index": spawn.gate_index, "state": spawn.state,
            "spawn": dict(spawn.metadata),
        }
        if spawn.previous_action is not None:
            options["previous_action"] = spawn.previous_action
        observation, _ = env.reset(seed=seed, options=options)
        history = ObservationHistory(
            actor.temporal_context,
            tuple(encoder.image_size),
            estimate_dim=int(getattr(encoder, "estimate_dim", 0)),
            estimate_seed=seed,
        )
        history.append(observation)
        warmup_teacher = None
        if not evaluation or str(settings.get("history_warmup_policy", "mpcc")) == "mpcc":
            warmup_teacher = cached_teacher(env, settings, stage.action_delay)
        warmup_steps = int(settings.get("history_warmup_steps", history.length - 1))
        warmup_terminated = False
        for _ in range(max(0, warmup_steps)):
            warmup_action = (
                warmup_teacher(observation).action.as_array()
                if warmup_teacher is not None
                else np.asarray([9.81, 0.0, 0.0, 0.0], np.float32)
            )
            observation, _, warmup_terminated, _, _ = env.step(warmup_action)
            if warmup_teacher is not None:
                warmup_teacher.observe_executed_action(warmup_action)
            history.append(observation)
            if warmup_terminated:
                break
        if not evaluation:
            teacher = warmup_teacher or cached_teacher(env, settings, stage.action_delay)
            if bool(settings.get("require_acados_teacher", True)) and "acados" not in teacher.source:
                raise RuntimeError(f"DAgger requires ACADOS, got {teacher.source!r}")
        start_passed = env.tracker.passed_count
        beta_schedule = settings.get("teacher_beta_schedule")
        beta = (
            float(beta_schedule[min(round_index, len(beta_schedule) - 1)])
            if beta_schedule else schedule(
                float(settings.get("teacher_beta_start", 1.0)),
                float(settings.get("teacher_beta_end", 0.2)), round_index, rounds,
            )
        )
        threshold = schedule(
            float(settings.get("disagreement_threshold_start", 0.25)),
            float(settings.get("disagreement_threshold_end", 0.12)),
            round_index, rounds,
        )
        raw_steps = interventions = queries = invalid = solver_failures = 0
        gates_passed = 0
        crashed = success = False
        disagreement_sum = uncertainty_sum = 0.0
        after_gate_steps = -1
        max_steps = int(settings.get("max_steps", stage.max_steps))
        amp = bool(settings.get("amp", True))
        timing = torch.tensor(
            [[1.0, stage.action_delay * 90.0, 1.0]], dtype=torch.float32
        )
        crashed = bool(warmup_terminated)
        while raw_steps < max_steps and not crashed and not success:
            state, student, uncertainty = actor_output(
                actor, encoder, action_tokenizer, history, device, amp
            )
            intervention = False
            teacher_action = None
            valid = False
            disagreement = 0.0
            if teacher is not None:
                command = teacher(observation)
                queries += 1
                teacher_action, valid, solver_failed = expert_label(
                    command,
                    strict_solver=bool(settings.get("require_solver_clean_labels", True)),
                )
                invalid += int(not valid)
                solver_failures += int(solver_failed)
                if valid and teacher_action is not None:
                    disagreement = float(np.sqrt(np.mean((student - teacher_action) ** 2)))
                    intervention = bool(rng.random() < beta or disagreement > threshold)
                    transition = bool(
                        0 <= after_gate_steps <= int(
                            settings.get("transition_priority_steps", 90)
                        )
                    )
                    priority = 1.0 + min(2.0, disagreement)
                    if transition:
                        priority *= float(settings.get("transition_priority", 4.0))
                    if intervention:
                        priority *= float(settings.get("intervention_priority", 2.0))
                    replay.add(DAggerSample(
                        state=state, target=torch.from_numpy(teacher_action[None]).float(),
                        action_timing=timing.clone(), priority=priority,
                        transition=transition, intervention=intervention,
                    ))
                    capture_fraction = float(settings.get(
                        "raw_transition_capture_fraction" if transition
                        else "raw_capture_fraction",
                        0.05 if transition else 0.002,
                    ))
                    if rng.random() < capture_fraction:
                        raw_replay.add(RawDAggerSample(
                            history=history.snapshot(),
                            target=torch.from_numpy(teacher_action[None]).float(),
                            action_timing=timing.clone(),
                            priority=priority,
                        ))
            executed = (
                teacher_action.copy()
                if intervention and valid and teacher_action is not None else student.copy()
            )
            if intervention:
                noise = float(settings.get("teacher_execution_noise", 0.0))
                if noise:
                    executed = np.clip(executed + rng.normal(0.0, noise, 4), -1.0, 1.0)
            command_ctbr = normalized_to_ctbr(executed)
            observation, _, terminated, _, info = env.step(command_ctbr)
            if teacher is not None:
                teacher.observe_executed_action(command_ctbr)
            history.append(observation)
            raw_steps += 1
            interventions += int(intervention)
            disagreement_sum += disagreement
            uncertainty_sum += uncertainty
            gates_passed = env.tracker.passed_count - start_passed
            if bool(info.get("gate_passed")):
                after_gate_steps = 0
            elif after_gate_steps >= 0:
                after_gate_steps += 1
            success = gates_passed >= stage.target_gates
            crashed = bool(
                info.get("ground_contact") or info.get("unity_collision")
                or (terminated and not success)
            )
        return {
            "success": float(success), "p1": float(gates_passed >= 1),
            "p2": float(gates_passed >= 2), "p3": float(gates_passed >= 3),
            "gates_passed": float(gates_passed), "crash": float(crashed),
            "raw_steps": float(raw_steps),
            "successful_lap_time": raw_steps / 90.0 if success else float("nan"),
            "teacher_fraction": interventions / max(raw_steps, 1),
            "teacher_valid_fraction": 1.0 - invalid / max(queries, 1),
            "teacher_solver_failure_fraction": solver_failures / max(queries, 1),
            "disagreement": disagreement_sum / max(raw_steps, 1),
            "uncertainty": uncertainty_sum / max(raw_steps, 1),
            "teacher_beta": beta,
        }
    finally:
        env.close()


def mean_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in results[0]:
        values = np.asarray([row[key] for row in results], np.float64)
        metrics[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return metrics


def _deployment_dagger_worker(
    connection: Any,
    settings: dict[str, Any],
    stage: RacingCurriculumStage,
    worker_index: int,
    history_length: int,
    mask_size: tuple[int, int],
    estimate_dim: int,
) -> None:
    """Own one persistent simulator and ACADOS solver outside the learner GIL."""

    env: FlightmareEnv | None = None
    try:
        track_name = stage.tracks[worker_index % len(stage.tracks)]
        env = FlightmareEnv(
            track=track_name, next_gates=3, image_size=(160, 128),
            control_dt=1.0 / 90.0, render_observations=False,
            mask_source="geometry", mask_size=(160, 128),
            geometry_renderer=str(settings.get("geometry_renderer", "exact")),
            image_delay=float(settings.get("image_delay", 0.033)),
            action_delay=stage.action_delay, terminate_on_collision=True,
        )
        teacher = make_mpcc(env, settings, stage.action_delay)
        if bool(settings.get("require_acados_teacher", True)) and "acados" not in teacher.source:
            raise RuntimeError(f"DAgger requires ACADOS, got {teacher.source!r}")
        observation: dict[str, Any] | None = None
        history: ObservationHistory | None = None
        evaluation = False
        start_passed = after_gate_steps = steps = 0
        interventions = queries = invalid = solver_failures = 0
        disagreement_sum = uncertainty_sum = 0.0
        target_gates = min(stage.target_gates, len(env.track.gates))
        connection.send(("ready", worker_index))
        while True:
            request = connection.recv()
            operation = request[0]
            if operation == "close":
                connection.send(("closed", worker_index))
                return
            if operation == "reset":
                episode_index, seed, evaluation = (
                    int(request[1]), int(request[2]), bool(request[3])
                )
                spawn = sample_curriculum_spawn(
                    env.track, stage, seed=seed, episode_index=episode_index
                )
                options: dict[str, Any] = {
                    "gate_index": spawn.gate_index,
                    "state": spawn.state,
                    "spawn": dict(spawn.metadata),
                }
                if spawn.previous_action is not None:
                    options["previous_action"] = spawn.previous_action
                observation, _ = env.reset(seed=seed, options=options)
                teacher.reset()
                history = ObservationHistory(
                    history_length, mask_size,
                    estimate_dim=estimate_dim, estimate_seed=seed,
                )
                history.append(observation)
                warmup_terminated = False
                for _ in range(int(settings.get("history_warmup_steps", history_length - 1))):
                    warmup_action = teacher(observation).action.as_array()
                    observation, _, warmup_terminated, _, _ = env.step(warmup_action)
                    teacher.observe_executed_action(warmup_action)
                    history.append(observation)
                    if warmup_terminated:
                        break
                start_passed = env.tracker.passed_count
                after_gate_steps = -1
                steps = interventions = queries = invalid = solver_failures = 0
                disagreement_sum = uncertainty_sum = 0.0
                warmup_metrics = None
                if warmup_terminated:
                    warmup_metrics = {
                        "success": 0.0, "p1": 0.0, "p2": 0.0, "p3": 0.0,
                        "gates_passed": 0.0, "crash": 1.0, "raw_steps": 0.0,
                        "successful_lap_time": float("nan"),
                        "teacher_fraction": 1.0,
                        "teacher_valid_fraction": 1.0,
                        "teacher_solver_failure_fraction": 0.0,
                        "disagreement": 0.0, "uncertainty": 0.0,
                    }
                connection.send((
                    "reset", history.snapshot(), bool(warmup_terminated),
                    warmup_metrics,
                ))
                continue
            if operation != "advance" or observation is None or history is None:
                raise RuntimeError(f"invalid deployment DAgger operation {operation!r}")
            student = np.asarray(request[1], np.float32)
            uncertainty = float(request[2])
            teacher_draw = bool(request[3])
            threshold = float(request[4])
            valid = intervention = solver_failed = transition = False
            teacher_action = None
            disagreement = 0.0
            if not evaluation:
                command = teacher(observation)
                queries += 1
                teacher_action, valid, solver_failed = expert_label(
                    command,
                    strict_solver=bool(settings.get("require_solver_clean_labels", False)),
                )
                invalid += int(not valid)
                solver_failures += int(solver_failed)
                if valid and teacher_action is not None:
                    disagreement = float(np.sqrt(np.mean((student - teacher_action) ** 2)))
                    intervention = bool(teacher_draw or disagreement > threshold)
                    transition = bool(
                        0 <= after_gate_steps
                        <= int(settings.get("transition_priority_steps", 90))
                    )
            executed = (
                teacher_action.copy()
                if intervention and valid and teacher_action is not None
                else student.copy()
            )
            physical = normalized_to_ctbr(executed)
            observation, _, terminated, _, info = env.step(physical)
            if not evaluation:
                teacher.observe_executed_action(physical)
            history.append(observation)
            steps += 1
            interventions += int(intervention)
            disagreement_sum += disagreement
            uncertainty_sum += uncertainty
            gates = env.tracker.passed_count - start_passed
            if bool(info.get("gate_passed")):
                after_gate_steps = 0
            elif after_gate_steps >= 0:
                after_gate_steps += 1
            success = gates >= target_gates
            crashed = bool(
                info.get("ground_contact") or info.get("unity_collision")
                or (terminated and not success)
            )
            done = bool(success or crashed or steps >= int(settings.get("max_steps", stage.max_steps)))
            episode_metrics = None
            if done:
                episode_metrics = {
                    "success": float(success), "p1": float(gates >= 1),
                    "p2": float(gates >= 2), "p3": float(gates >= 3),
                    "gates_passed": float(gates), "crash": float(crashed),
                    "raw_steps": float(steps),
                    "successful_lap_time": steps / 90.0 if success else float("nan"),
                    "teacher_fraction": interventions / max(steps, 1),
                    "teacher_valid_fraction": 1.0 - invalid / max(queries, 1),
                    "teacher_solver_failure_fraction": solver_failures / max(queries, 1),
                    "disagreement": disagreement_sum / max(steps, 1),
                    "uncertainty": uncertainty_sum / max(steps, 1),
                }
            connection.send((
                "step", valid, teacher_action, solver_failed, intervention,
                transition, disagreement, history.packed_latest(), done,
                episode_metrics,
            ))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except BaseException:
            pass
        raise
    finally:
        if env is not None:
            env.close()
        connection.close()


@dataclass
class DeploymentDaggerSlot:
    connection: Any
    process: Any
    history: ObservationHistory
    done: bool = True


class ProcessDeploymentDaggerCollector:
    """Persistent rollout workers with batched V3 encoder/actor inference."""

    def __init__(
        self, actor, encoder, action_tokenizer, settings: Mapping[str, Any],
        stage: RacingCurriculumStage, device: str,
    ) -> None:
        self.actor = actor
        self.encoder = encoder
        self.action_tokenizer = action_tokenizer
        self.settings = dict(settings)
        self.stage = stage
        self.device = device
        self.amp = bool(settings.get("amp", True))
        self.parallel = int(settings.get("rollout_envs", 8))
        self.rng = np.random.default_rng(int(settings.get("seed", 0)) + 991)
        context = mp.get_context("spawn")
        self.slots: list[DeploymentDaggerSlot] = []
        # Start sequentially so the first worker builds the shared ACADOS cache
        # before the remaining workers load it.
        for index in range(self.parallel):
            parent, child = context.Pipe()
            process = context.Process(
                target=_deployment_dagger_worker,
                args=(
                    child, self.settings, stage, index, actor.temporal_context,
                    tuple(encoder.image_size), int(getattr(encoder, "estimate_dim", 0)),
                ),
                name=f"deployment-dagger-{index:02d}",
                daemon=True,
            )
            process.start(); child.close()
            slot = DeploymentDaggerSlot(
                parent, process,
                ObservationHistory(
                    actor.temporal_context, tuple(encoder.image_size),
                    estimate_dim=int(getattr(encoder, "estimate_dim", 0)),
                    estimate_seed=index,
                ),
            )
            response = parent.recv()
            if response[0] != "ready":
                raise RuntimeError(f"deployment DAgger worker failed: {response}")
            self.slots.append(slot)

    @staticmethod
    def _receive(slot: DeploymentDaggerSlot, expected: str) -> tuple[Any, ...]:
        response = slot.connection.recv()
        if response[0] == "error":
            raise RuntimeError(f"deployment DAgger worker failed:\n{response[1]}")
        if response[0] != expected:
            raise RuntimeError(f"expected {expected}, received {response[0]}")
        return response

    def _reset(
        self, slot: DeploymentDaggerSlot, episode_index: int,
        seed_base: int, evaluation: bool,
    ) -> dict[str, float] | None:
        seed = seed_base + 7919 * episode_index
        slot.connection.send(("reset", episode_index, seed, evaluation))
        _, snapshot, terminated, metrics = self._receive(slot, "reset")
        slot.history.restore(snapshot)
        slot.done = bool(terminated)
        return metrics

    def _reset_many(
        self, assignments: list[tuple[DeploymentDaggerSlot, int]],
        seed_base: int, evaluation: bool,
    ) -> list[dict[str, float]]:
        """Reset and MPCC-warm all newly assigned environments concurrently."""

        for slot, episode_index in assignments:
            seed = seed_base + 7919 * episode_index
            slot.connection.send(("reset", episode_index, seed, evaluation))
        ended: list[dict[str, float]] = []
        for slot, _ in assignments:
            _, snapshot, terminated, metrics = self._receive(slot, "reset")
            slot.history.restore(snapshot)
            slot.done = bool(terminated)
            if metrics is not None:
                ended.append(metrics)
        return ended

    def close(self) -> None:
        for slot in self.slots:
            if slot.process.is_alive():
                try: slot.connection.send(("close",))
                except (BrokenPipeError, EOFError): pass
        for slot in self.slots:
            if slot.process.is_alive():
                try: self._receive(slot, "closed")
                except (BrokenPipeError, EOFError, RuntimeError): pass
            slot.process.join(timeout=3.0)
            if slot.process.is_alive():
                slot.process.terminate(); slot.process.join(timeout=2.0)
            slot.connection.close()

    @torch.no_grad()
    def collect(
        self, *, episodes: int, beta: float, threshold: float, seed_base: int,
        evaluation: bool, replay: PrioritizedReplay, raw_replay: RawHistoryReplay,
    ) -> list[dict[str, float]]:
        self.actor.eval(); self.encoder.eval()
        results: list[dict[str, float]] = []
        scheduled = completed = 0
        initial_assignments: list[tuple[DeploymentDaggerSlot, int]] = []
        for slot in self.slots:
            if scheduled < episodes:
                initial_assignments.append((slot, scheduled))
                scheduled += 1
            else:
                slot.done = True
        initial_ended = self._reset_many(
            initial_assignments, seed_base, evaluation
        )
        for metrics in initial_ended:
            metrics["teacher_beta"] = float(beta)
            results.append(metrics)
        completed += len(initial_ended)
        started = time.perf_counter()
        environment_steps = 0
        while completed < episodes:
            active = [slot for slot in self.slots if not slot.done]
            batches = [slot.history.batch(self.device) for slot in active]
            batch = {
                key: torch.cat([item[key] for item in batches], dim=0)
                for key in batches[0]
            }
            state = build_actor_state(
                self.encoder, batch, self.actor.temporal_context,
                self.device, self.amp,
            )
            with torch.autocast(
                "cuda", dtype=torch.bfloat16,
                enabled=self.device.startswith("cuda") and self.amp,
            ):
                output = self.actor(state, self.action_tokenizer)
            students = output.mean[:, 0].float().cpu().numpy()
            uncertainties = output.std.float().mean((1, 2)).cpu().numpy()
            states = state.half().cpu()
            for slot, student, uncertainty in zip(active, students, uncertainties):
                slot.connection.send((
                    "advance", student, float(uncertainty),
                    self.rng.random() < beta, threshold,
                ))
            responses = [self._receive(slot, "step") for slot in active]
            finished: list[DeploymentDaggerSlot] = []
            for index, (slot, response) in enumerate(zip(active, responses)):
                (
                    _, valid, target, solver_failed, intervention, transition,
                    disagreement, packed, done, metrics,
                ) = response
                if valid and target is not None and not evaluation:
                    priority = 1.0 + min(2.0, float(disagreement))
                    if transition:
                        priority *= float(self.settings.get("transition_priority", 4.0))
                    if intervention:
                        priority *= float(self.settings.get("intervention_priority", 1.5))
                    timing = torch.tensor(
                        [[1.0, self.stage.action_delay * 90.0, 1.0]],
                        dtype=torch.float32,
                    )
                    replay.add(DAggerSample(
                        state=states[index], target=torch.from_numpy(target[None]).float(),
                        action_timing=timing, priority=priority,
                        transition=bool(transition), intervention=bool(intervention),
                    ))
                    capture = float(self.settings.get(
                        "raw_transition_capture_fraction" if transition
                        else "raw_capture_fraction", 0.0,
                    ))
                    if self.rng.random() < capture:
                        raw_replay.add(RawDAggerSample(
                            history=slot.history.snapshot(),
                            target=torch.from_numpy(target[None]).float(),
                            action_timing=timing, priority=priority,
                        ))
                slot.history.append_packed(packed)
                environment_steps += 1
                slot.done = bool(done)
                if slot.done:
                    if metrics is not None:
                        metrics["teacher_beta"] = float(beta)
                        results.append(metrics)
                    finished.append(slot)
            assignments: list[tuple[DeploymentDaggerSlot, int]] = []
            for slot in finished:
                completed += 1
                if scheduled < episodes:
                    assignments.append((slot, scheduled))
                    scheduled += 1
            ended_on_reset = self._reset_many(assignments, seed_base, evaluation)
            for metrics in ended_on_reset:
                metrics["teacher_beta"] = float(beta)
                results.append(metrics)
            completed += len(ended_on_reset)
            interval = int(self.settings.get("progress_log_interval", 5000))
            if interval > 0 and environment_steps and environment_steps % interval < len(active):
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(
                    f"collector evaluation={int(evaluation)} episodes={completed}/{episodes} "
                    f"steps={environment_steps} steps_per_second={environment_steps / elapsed:.1f} "
                    f"replay={len(replay)}",
                    flush=True,
                )
        return results


def train_updates(
    actor, encoder, action_tokenizer, optimizer, replay, raw_replay,
    encoder_anchor, settings, device: str, round_index: int,
    allow_encoder_adaptation: bool,
) -> dict[str, float]:
    actor.train()
    encoder.eval()
    totals: dict[str, list[float]] = {}
    updates = int(settings.get("updates_per_round", 1000))
    rng = np.random.default_rng(int(settings.get("seed", 0)) + 3571 * round_index)
    adaptation_enabled = (
        allow_encoder_adaptation
        and round_index + 1 >= int(settings.get("encoder_adaptation_start_round", 2))
        and bool(raw_replay.samples)
        and bool(encoder_anchor)
    )
    for update in range(updates):
        use_raw = adaptation_enabled and rng.random() < float(
            settings.get("encoder_update_fraction", 0.10)
        )
        batch = (
            raw_replay.sample(int(settings.get("encoder_batch_size", 4)), device)
            if use_raw else replay.sample(int(settings.get("batch_size", 128)), device)
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=device.startswith("cuda") and bool(settings.get("amp", True)),
        ):
            state = (
                build_actor_state(
                    encoder, batch, actor.temporal_context, device,
                    bool(settings.get("amp", True)), encoder_grad=True,
                )
                if use_raw else batch["state"]
            )
            output = actor(state, action_tokenizer)
            with torch.no_grad():
                target_tokens, _ = action_tokenizer.encode(
                    batch["target"], batch["action_timing"]
                )
            loss, metrics = actor_bc_objective(
                output, batch["target"], settings, update, target_tokens
            )
            anchor_loss = loss.new_zeros(())
            if use_raw:
                anchor_loss = torch.stack([
                    (parameter.float() - encoder_anchor[name]).square().mean()
                    for name, parameter in encoder.named_parameters()
                    if name in encoder_anchor
                ]).mean()
                loss = loss + float(settings.get("encoder_anchor_weight", 0.01)) * anchor_loss
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), float(settings.get("grad_clip", 2.0))
        )
        encoder_gradient = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in encoder.parameters() if parameter.requires_grad],
            float(settings.get("encoder_grad_clip", 0.25)),
        ) if encoder_anchor else torch.tensor(0.0)
        optimizer.step()
        values = {
            **metrics,
            "gradient_norm": gradient.detach(),
            "encoder_gradient_norm": encoder_gradient.detach(),
            "encoder_anchor_loss": anchor_loss.detach(),
            "encoder_update": float(use_raw),
        }
        for key, value in values.items():
            totals.setdefault(key, []).append(float(value))
    actor.eval()
    encoder.eval()
    return {key: float(np.mean(values)) for key, values in totals.items()}


def main() -> None:
    args, config = arguments()
    settings = config["actor_dagger"]
    device = str(args.device)
    seed = int(settings.get("seed", 2026081704))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    initial_path = Path(settings["actor"])
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    actor_config = dict(initial["model_config"])
    actor = GaussianActor(**actor_config).to(device)
    actor.load_state_dict(initial["model"])
    _, encoder, action_tokenizer, _, observation_path, action_path, _ = (
        load_frozen_actor_stack(
            None, Path(initial["observation_tokenizer"]),
            Path(initial["action_tokenizer"]), device,
        )
    )
    if "observation_encoder" in initial:
        encoder.load_state_dict(initial["observation_encoder"])
    encoder.eval().requires_grad_(False)
    encoder_prefixes = tuple(settings.get(
        "encoder_trainable_prefixes",
        ["estimate_patcher.", "modality_resamplers.fusion.",
         "modality_bottlenecks.fusion.", "macro_belief."],
    ))
    encoder_parameters = []
    for name, parameter in encoder.named_parameters():
        if any(name.startswith(prefix) for prefix in encoder_prefixes):
            parameter.requires_grad_(True)
            encoder_parameters.append(parameter)
    encoder_anchor = {
        name: parameter.detach().float().clone()
        for name, parameter in encoder.named_parameters()
        if parameter.requires_grad
    }
    action_tokenizer.eval().requires_grad_(False)
    actor.train()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": actor.parameters(),
                "lr": float(settings.get("learning_rate", 5e-5)),
            },
            {
                "params": encoder_parameters,
                "lr": float(settings.get("encoder_learning_rate", 3e-6)),
            },
        ],
        weight_decay=float(settings.get("weight_decay", 1e-5)),
        fused=device.startswith("cuda"),
    )
    replay = PrioritizedReplay(int(settings.get("replay_capacity", 60000)), seed)
    raw_replay = RawHistoryReplay(int(settings.get("raw_replay_capacity", 512)), seed + 1)
    stage = RacingCurriculumStage.from_mapping(settings["stage"])
    manager = CheckpointManager.from_config(config)
    logger = init_wandb(config)
    rounds = int(settings.get("rounds", 8))
    episodes = int(settings.get("episodes_per_round", 16))
    eval_episodes = int(settings.get("evaluation_episodes", 16))
    global_step = 0
    best_selection = float(settings.get("initial_selection_score", -float("inf")))
    best_p2 = float(settings.get("initial_p2", 0.0))
    print(
        f"actor_dagger actor={initial_path} actor_parameters="
        f"{sum(p.numel() for p in actor.parameters()):,} rounds={rounds} "
        f"episodes_per_round={episodes} target_gates={stage.target_gates}", flush=True,
    )
    process_collector = (
        ProcessDeploymentDaggerCollector(
            actor, encoder, action_tokenizer, settings, stage, device
        )
        if str(settings.get("collector_backend", "process")) == "process"
        else None
    )
    for round_index in range(rounds):
        started = time.perf_counter()
        actor.eval()
        beta_schedule = settings.get("teacher_beta_schedule")
        beta = (
            float(beta_schedule[min(round_index, len(beta_schedule) - 1)])
            if beta_schedule else schedule(
                float(settings.get("teacher_beta_start", 1.0)),
                float(settings.get("teacher_beta_end", 0.2)), round_index, rounds,
            )
        )
        threshold = schedule(
            float(settings.get("disagreement_threshold_start", 0.25)),
            float(settings.get("disagreement_threshold_end", 0.12)),
            round_index, rounds,
        )
        collection = (
            process_collector.collect(
                episodes=episodes, beta=beta, threshold=threshold,
                seed_base=seed + 1000003 * round_index,
                evaluation=False, replay=replay, raw_replay=raw_replay,
            )
            if process_collector is not None else [
                collect_episode(
                    actor, encoder, action_tokenizer, replay, raw_replay, stage, settings,
                    round_index=round_index, episode_index=index, device=device,
                    evaluation=False,
                )
                for index in range(episodes)
            ]
        )
        collect_metrics = mean_metrics(collection)
        pre_update_actor = {
            name: value.detach().cpu().clone()
            for name, value in actor.state_dict().items()
        }
        pre_update_encoder = {
            name: parameter.detach().cpu().clone()
            for name, parameter in encoder.named_parameters()
            if parameter.requires_grad
        }
        update_metrics = train_updates(
            actor, encoder, action_tokenizer, optimizer, replay, raw_replay,
            encoder_anchor, settings, device, round_index,
            best_p2 >= float(settings.get("encoder_adaptation_min_best_p2", 0.20)),
        )
        evaluation = (
            process_collector.collect(
                episodes=eval_episodes, beta=0.0, threshold=threshold,
                seed_base=int(settings.get("evaluation_seed", seed + 900000000)),
                evaluation=True, replay=replay, raw_replay=raw_replay,
            )
            if process_collector is not None else [
                collect_episode(
                    actor, encoder, action_tokenizer, replay, raw_replay, stage, settings,
                    round_index=round_index,
                    episode_index=100000 + index, device=device, evaluation=True,
                )
                for index in range(eval_episodes)
            ]
        )
        eval_metrics = mean_metrics(evaluation)
        global_step += int(sum(row["raw_steps"] for row in collection))
        selection = (
            float(settings.get("selection_success_weight", 4.0)) * eval_metrics["success"]
            + float(settings.get("selection_p3_weight", 0.5)) * eval_metrics["p3"]
            + float(settings.get("selection_p2_weight", 0.25)) * eval_metrics["p2"]
            + float(settings.get("selection_p1_weight", 0.1)) * eval_metrics["p1"]
            - float(settings.get("selection_crash_weight", 0.5)) * eval_metrics["crash"]
        )
        candidate_selection = selection
        rolled_back = bool(
            selection < best_selection - float(settings.get("rollback_tolerance", 0.0))
        )
        if rolled_back:
            actor.load_state_dict(pre_update_actor)
            encoder_state = encoder.state_dict()
            encoder_state.update(pre_update_encoder)
            encoder.load_state_dict(encoder_state)
            optimizer.state.clear()
        else:
            best_selection = selection
            best_p2 = max(best_p2, float(eval_metrics["p2"]))
        train_logged = {
            **{f"collect_{key}": value for key, value in collect_metrics.items()},
            **update_metrics, **replay.metrics(), **raw_replay.metrics(),
            "round": float(round_index + 1),
            "environment_steps": float(global_step),
            "rollback": float(rolled_back),
            "round_seconds": time.perf_counter() - started,
        }
        eval_logged = {
            **eval_metrics, "selection_score": candidate_selection,
            "best_selection_score": best_selection,
            "rollback": float(rolled_back),
            "environment_steps": float(global_step),
            "round": float(round_index + 1),
        }
        logger.log_train(train_logged, round_index + 1)
        logger.log_eval(eval_logged, round_index + 1)
        print(
            f"round={round_index + 1}/{rounds} env_steps={global_step} "
            f"replay={len(replay)} teacher={collect_metrics['teacher_fraction']:.3f} "
            f"valid={collect_metrics['teacher_valid_fraction']:.4f} "
            f"disagreement={collect_metrics['disagreement']:.3f} "
            f"loss={update_metrics['loss']:.4f} p1={eval_metrics['p1']:.3f} "
            f"p2={eval_metrics['p2']:.3f} p3={eval_metrics['p3']:.3f} "
            f"full={eval_metrics['success']:.3f} crash={eval_metrics['crash']:.3f} "
            f"rollback={int(rolled_back)} seconds={train_logged['round_seconds']:.1f} "
            f"replay_gib={train_logged['replay_allocated_gib']:.2f}",
            flush=True,
        )
        manager.save_eval(
            eval_logged, step=global_step,
            summary="Strict ACADOS DAgger distillation for the frozen-tokenizer causal actor.",
        )
        manager.save({
            "model": actor.state_dict(), "optimizer": optimizer.state_dict(),
            "observation_encoder": encoder.state_dict(),
            "model_config": actor_config, "training_config": config,
            "observation_tokenizer": str(observation_path),
            "action_tokenizer": str(action_path), "rng_state": capture_rng_state(),
            "round": round_index + 1, "initial_checkpoint": str(initial_path),
            "input_contract": {
                **dict(initial.get("input_contract", {})),
                "imitation_algorithm": "strict_acados_robot_gated_dagger",
            },
            "teacher_contract": {
                "backend": str(settings.get("mpcc_backend", "acados")),
                "require_acados": bool(settings.get("require_acados_teacher", True)),
                "require_solver_clean_labels": bool(
                    settings.get("require_solver_clean_labels", True)
                ),
            },
        }, step=global_step, metrics=eval_logged, rank=not rolled_back)
    if process_collector is not None:
        process_collector.close()
    logger.finish()


if __name__ == "__main__":
    main()
