#!/usr/bin/env python3
"""Process-isolated DAgger for the compact raw deployment-estimate policy."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_flight import ObservationHistory
from scripts.train_actor_dagger import expert_label
from scripts.train_compact_observation_ablation import (
    compact_features, load_config, make_dataset,
)
from scripts.train_flow_policy_fpo import make_mpcc
from starscream.checkpoint_manager import CheckpointManager, capture_rng_state
from starscream.dataloader import stratified_episode_split
from starscream.env import FlightmareEnv
from starscream.privileged_racing import (
    FeatureNormalizer, PrivilegedMLPPolicy, normalized_to_ctbr,
    resolve_ranked_checkpoint,
)
from starscream.racing_curriculum import RacingCurriculumStage, sample_curriculum_spawn
from starscream.racing_curriculum import SpawnSample
from starscream.env.tracks import forward_up_quaternion
from starscream.wandb import init_wandb


RAW_HISTORY = 36
POLICY_HISTORY = 18
FEATURE_DIM = 77
TASK_DIM = 19


def arguments() -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config = copy.deepcopy(config)
        config["dagger"].update(
            rounds=1, episodes_per_round=2, evaluation_episodes=2,
            rollout_envs=2, updates_per_round=2, batch_size=4,
            replay_capacity=64, max_steps=18, offline_max_windows=32,
            progress_log_interval=1,
        )
        config["dagger"]["curriculum"]["max_steps"] = 18
        config["output_root"] = "/tmp/starscream-compact-raw-dagger-smoke"
        config["checkpoint"] = {
            "run_name": "compact-raw-dagger-smoke", "monitor": "selection_score",
            "mode": "max", "top_k": 1,
        }
        config["wandb"] = {"enabled": False, "run_name": "compact-raw-dagger-smoke"}
    if args.disable_wandb:
        config["wandb"] = {"enabled": False}
    return args, config


def normalized_task(observation: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(observation["task_state"], np.float32).copy()
    value[0:3] /= 20.0; value[3:6] /= 30.0
    value[12:15] /= 6.0; value[15:19] /= 4000.0
    return value


def raw_record_feature(record: Mapping[str, np.ndarray]) -> np.ndarray:
    feature = np.concatenate([
        np.asarray(record["estimate"], np.float32),
        np.asarray(record["route"], np.float32).reshape(-1),
        np.asarray(record["previous_action"], np.float32),
        np.asarray(record["timing"], np.float32)[[3, 7]],
    ]).astype(np.float32)
    if feature.shape != (FEATURE_DIM,) or not np.all(np.isfinite(feature)):
        raise ValueError("raw compact feature must be finite and 77-dimensional")
    return feature


def raw_feature_sequence(history: ObservationHistory) -> np.ndarray:
    records = list(history.records)
    if not records:
        raise RuntimeError("raw observation history is empty")
    records = records[-POLICY_HISTORY:]
    if len(records) < POLICY_HISTORY:
        records = [records[0]] * (POLICY_HISTORY - len(records)) + records
    return np.stack([raw_record_feature(record) for record in records])


def sample_collection_spawn(
    track: Any, stage: RacingCurriculumStage, settings: Mapping[str, Any],
    *, seed: int, episode_index: int,
) -> SpawnSample:
    """Sample gate-local, incoming-segment, or realized post-gate geometry.

    DAgger v1 reset only on the active gate's directed normal.  That is a poor
    match for the difficult figure-eight transitions, where the vehicle reaches
    the next gate from the previous gate's exit direction.  This sampler keeps
    the reset contract auditable while exposing those transition geometries.
    """

    distributions = dict(settings.get("collection_spawn_distribution", {"gate_local": 1.0}))
    names = list(distributions)
    weights = np.asarray([float(distributions[name]) for name in names], np.float64)
    if not names or np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("collection_spawn_distribution must contain positive finite weights")
    weights /= weights.sum()
    rng = np.random.default_rng(int(seed) + 104729 * int(episode_index) + 17041)
    gate_weights = np.asarray(
        settings.get("collection_gate_weights", [1.0] * len(track.gates)), np.float64
    )
    if gate_weights.shape != (len(track.gates),) or np.any(gate_weights < 0) or gate_weights.sum() <= 0:
        raise ValueError("collection_gate_weights must match the track gate count")
    gate_index = int(rng.choice(len(track.gates), p=gate_weights / gate_weights.sum()))
    mode = str(rng.choice(names, p=weights))
    if mode == "gate_local":
        local_values = asdict(stage)
        local_values.update(tracks=list(stage.tracks), random_gate=False)
        local_stage = RacingCurriculumStage.from_mapping(local_values)
        spawn = sample_curriculum_spawn(
            track, local_stage, seed=seed, episode_index=gate_index,
        )
        return SpawnSample(
            spawn.state, spawn.gate_index,
            {**dict(spawn.metadata), "collection_spawn_mode": mode},
            spawn.previous_action, spawn.observation_history,
        )
    if mode not in {"incoming_segment", "post_gate"}:
        raise ValueError(f"unknown collection spawn mode {mode!r}")

    gate = track.gates[gate_index]
    previous = track.gates[(gate_index - 1) % len(track.gates)]
    if mode == "incoming_segment":
        direction = gate.position - previous.position
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        distance = float(rng.uniform(*settings.get("incoming_distance", [1.5, 5.0])))
        position = gate.position - distance * direction
        frame = gate.directed_rotation
    else:
        exit_distance = float(rng.uniform(*settings.get("post_gate_distance", [0.20, 1.50])))
        position = previous.position + exit_distance * previous.normal
        direction = gate.position - position
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        frame = previous.directed_rotation
    position += float(rng.uniform(*stage.lateral_offset)) * frame[:, 1]
    position += float(rng.uniform(*stage.vertical_offset)) * frame[:, 2]
    margin = np.asarray([0.20, 0.20, 0.70], np.float32)
    position = np.clip(position, track.bounds[:, 0] + margin, track.bounds[:, 1] - margin)
    signed = float((position - gate.position) @ gate.normal)
    if signed >= -0.10:
        position -= (signed + 0.10) * gate.normal
    speed = float(rng.uniform(*stage.forward_speed))
    velocity = speed * direction
    velocity += float(rng.uniform(*stage.lateral_speed)) * frame[:, 1]
    velocity += float(rng.uniform(*stage.vertical_speed)) * frame[:, 2]
    state = np.zeros(25, np.float32)
    state[:3] = position
    state[3:7] = forward_up_quaternion(direction)
    state[7:10] = velocity
    state[10:13] = rng.uniform(-stage.body_rate, stage.body_rate, size=3)
    metadata: dict[str, float | int | str] = {
        "curriculum_stage": stage.name, "gate_index": gate_index,
        "collection_spawn_mode": mode, "sampler": "transition-mixture-v1",
        "forward_speed": speed,
        "distance_to_active_gate": float(np.linalg.norm(gate.position - position)),
        "previous_gate_index": (gate_index - 1) % len(track.gates),
    }
    return SpawnSample(state, gate_index, metadata)


def _worker(
    connection: Any, settings: dict[str, Any], stage: RacingCurriculumStage,
    worker_index: int,
) -> None:
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
        start_passed = after_gate = steps = 0
        interventions = queries = invalid = solver_failures = 0
        disagreement_sum = 0.0
        target_gates = min(stage.target_gates, len(env.track.gates))
        connection.send(("ready", worker_index))
        while True:
            request = connection.recv()
            operation = request[0]
            if operation == "close":
                connection.send(("closed", worker_index)); return
            if operation == "reset":
                episode_index, seed, evaluation = int(request[1]), int(request[2]), bool(request[3])
                spawn = (
                    sample_curriculum_spawn(
                        env.track, stage, seed=seed, episode_index=episode_index
                    )
                    if evaluation else sample_collection_spawn(
                        env.track, stage, settings, seed=seed, episode_index=episode_index
                    )
                )
                options: dict[str, Any] = {
                    "gate_index": spawn.gate_index, "state": spawn.state,
                    "spawn": dict(spawn.metadata),
                }
                if spawn.previous_action is not None:
                    options["previous_action"] = spawn.previous_action
                observation, _ = env.reset(seed=seed, options=options)
                teacher.reset()
                history = ObservationHistory(
                    RAW_HISTORY, (128, 160), estimate_dim=32, estimate_seed=seed
                )
                history.append(observation)
                warmup_terminated = False
                for _ in range(int(settings.get("history_warmup_steps", 0))):
                    command = teacher(observation).action.as_array()
                    observation, _, warmup_terminated, _, _ = env.step(command)
                    teacher.observe_executed_action(command)
                    history.append(observation)
                    if warmup_terminated:
                        break
                start_passed = env.tracker.passed_count
                after_gate = -1; steps = interventions = queries = invalid = solver_failures = 0
                disagreement_sum = 0.0
                metrics = None
                if warmup_terminated:
                    metrics = {
                        "success": 0.0, "p1": 0.0, "p2": 0.0, "p3": 0.0,
                        "gates_passed": 0.0, "crash": 1.0, "raw_steps": 0.0,
                        "successful_lap_time": float("nan"), "teacher_fraction": 1.0,
                        "teacher_valid_fraction": 1.0,
                        "teacher_solver_failure_fraction": 0.0, "disagreement": 0.0,
                    }
                connection.send(("reset", history.snapshot(), warmup_terminated, metrics))
                continue
            if operation != "advance" or observation is None or history is None:
                raise RuntimeError(f"invalid compact DAgger operation {operation!r}")
            student = np.asarray(request[1], np.float32)
            teacher_draw, threshold = bool(request[2]), float(request[3])
            prior_task = normalized_task(observation)
            prior_gate = int(np.asarray(observation["flight_plan"]["index"])[0])
            valid = intervention = solver_failed = transition = False
            target = None; disagreement = 0.0
            if not evaluation:
                command = teacher(observation); queries += 1
                target, valid, solver_failed = expert_label(
                    command,
                    strict_solver=bool(settings.get("require_solver_clean_labels", False)),
                )
                invalid += int(not valid); solver_failures += int(solver_failed)
                if valid and target is not None:
                    disagreement = float(np.sqrt(np.mean((student - target) ** 2)))
                    intervention = bool(teacher_draw or disagreement > threshold)
                    transition = bool(
                        0 <= after_gate <= int(settings.get("transition_priority_steps", 120))
                    )
            executed = target.copy() if intervention and target is not None else student.copy()
            physical = normalized_to_ctbr(executed)
            observation, _, terminated, _, info = env.step(physical)
            if not evaluation:
                teacher.observe_executed_action(physical)
            history.append(observation)
            following_task = normalized_task(observation)
            following_gate = int(np.asarray(observation["flight_plan"]["index"])[0])
            steps += 1; interventions += int(intervention); disagreement_sum += disagreement
            gates = env.tracker.passed_count - start_passed
            if bool(info.get("gate_passed")):
                after_gate = 0
            elif after_gate >= 0:
                after_gate += 1
            success = gates >= target_gates
            crashed = bool(
                info.get("ground_contact") or info.get("unity_collision")
                or (terminated and not success)
            )
            done = bool(success or crashed or steps >= int(settings.get("max_steps", stage.max_steps)))
            metrics = None
            if done:
                metrics = {
                    "success": float(success), "p1": float(gates >= 1),
                    "p2": float(gates >= 2), "p3": float(gates >= 3),
                    "gates_passed": float(gates), "crash": float(crashed),
                    "raw_steps": float(steps),
                    "successful_lap_time": steps / 90.0 if success else float("nan"),
                    "teacher_fraction": interventions / max(steps, 1),
                    "teacher_valid_fraction": 1.0 - invalid / max(queries, 1),
                    "teacher_solver_failure_fraction": solver_failures / max(queries, 1),
                    "disagreement": disagreement_sum / max(steps, 1),
                }
            connection.send((
                "step", valid, target, solver_failed, intervention, transition,
                disagreement, following_task - prior_task,
                bool(following_gate == prior_gate), history.packed_latest(), done, metrics,
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
class ReplaySample:
    state: np.ndarray
    target: np.ndarray
    previous: np.ndarray
    dynamics: np.ndarray
    dynamics_valid: bool
    priority: float
    transition: bool
    intervention: bool


class CompactReplay:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity); self.count = 0; self.cursor = 0
        self.rng = np.random.default_rng(seed)
        self.states = torch.empty((capacity, POLICY_HISTORY, FEATURE_DIM), dtype=torch.float16)
        self.targets = torch.empty((capacity, 4), dtype=torch.float16)
        self.previous = torch.empty((capacity, 4), dtype=torch.float16)
        self.dynamics = torch.empty((capacity, TASK_DIM), dtype=torch.float16)
        self.dynamics_valid = np.zeros(capacity, np.bool_)
        self.priorities = np.zeros(capacity, np.float32)
        self.transitions = np.zeros(capacity, np.bool_)
        self.interventions = np.zeros(capacity, np.bool_)

    def __len__(self) -> int:
        return self.count

    def add(self, sample: ReplaySample) -> None:
        index = self.cursor
        self.states[index].copy_(torch.from_numpy(sample.state))
        self.targets[index].copy_(torch.from_numpy(sample.target))
        self.previous[index].copy_(torch.from_numpy(sample.previous))
        self.dynamics[index].copy_(torch.from_numpy(sample.dynamics))
        self.dynamics_valid[index] = sample.dynamics_valid
        self.priorities[index] = max(float(sample.priority), 1e-4)
        self.transitions[index] = sample.transition
        self.interventions[index] = sample.intervention
        self.cursor = (self.cursor + 1) % self.capacity
        self.count = min(self.capacity, self.count + 1)

    def sample(self, count: int, device: str) -> dict[str, torch.Tensor]:
        probabilities = self.priorities[:self.count].astype(np.float64)
        selected = self.rng.choice(
            self.count, size=count, replace=self.count < count,
            p=probabilities / probabilities.sum(),
        )
        index = torch.from_numpy(selected.astype(np.int64))
        return {
            "state": self.states.index_select(0, index).to(device).float(),
            "target": self.targets.index_select(0, index).to(device).float(),
            "previous": self.previous.index_select(0, index).to(device).float(),
            "dynamics": self.dynamics.index_select(0, index).to(device).float(),
            "dynamics_valid": torch.from_numpy(self.dynamics_valid[selected]).to(device),
        }

    def metrics(self) -> dict[str, float]:
        allocated = sum(
            value.numel() * value.element_size()
            for value in (self.states, self.targets, self.previous, self.dynamics)
        )
        count = max(self.count, 1)
        return {
            "replay_size": float(self.count),
            "replay_allocated_gib": allocated / (1024 ** 3),
            "replay_transition_fraction": float(self.transitions[:self.count].mean()),
            "replay_intervention_fraction": float(self.interventions[:self.count].mean()),
            "replay_dynamics_valid_fraction": float(self.dynamics_valid[:self.count].mean()),
            "replay_priority_mean": float(self.priorities[:self.count].sum() / count),
        }


@dataclass
class Slot:
    connection: Any
    process: Any
    history: ObservationHistory
    done: bool = True


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], np.float64)
        output[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return output


class ProcessCollector:
    def __init__(
        self, policy: PrivilegedMLPPolicy, normalizer: FeatureNormalizer,
        settings: Mapping[str, Any], stage: RacingCurriculumStage, device: str,
    ) -> None:
        self.policy = policy; self.normalizer = normalizer
        self.settings = dict(settings); self.stage = stage; self.device = device
        self.parallel = int(settings.get("rollout_envs", 12))
        self.rng = np.random.default_rng(int(settings.get("seed", 0)) + 991)
        context = mp.get_context("spawn")
        self.slots: list[Slot] = []
        # Sequential startup lets the first ACADOS worker populate shared code.
        for index in range(self.parallel):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker, args=(child, self.settings, stage, index),
                name=f"compact-raw-dagger-{index:02d}", daemon=True,
            )
            process.start(); child.close()
            slot = Slot(
                parent, process,
                ObservationHistory(RAW_HISTORY, (128, 160), estimate_dim=32, estimate_seed=index),
            )
            response = parent.recv()
            if response[0] != "ready":
                raise RuntimeError(f"compact DAgger worker failed: {response}")
            self.slots.append(slot)

    @staticmethod
    def receive(slot: Slot, expected: str) -> tuple[Any, ...]:
        response = slot.connection.recv()
        if response[0] == "error":
            raise RuntimeError(f"compact DAgger worker failed:\n{response[1]}")
        if response[0] != expected:
            raise RuntimeError(f"expected {expected}, received {response[0]}")
        return response

    def reset(self, slot: Slot, episode: int, seed_base: int, evaluation: bool):
        seed = seed_base + 7919 * episode
        slot.connection.send(("reset", episode, seed, evaluation))
        _, snapshot, terminated, metrics = self.receive(slot, "reset")
        slot.history.restore(snapshot); slot.done = bool(terminated)
        return metrics

    def collect(
        self, *, episodes: int, beta: float, threshold: float, seed_base: int,
        evaluation: bool, replay: CompactReplay | None,
    ) -> tuple[list[dict[str, float]], int]:
        self.policy.eval(); scheduled = completed = steps = 0
        results: list[dict[str, float]] = []
        for slot in self.slots:
            if scheduled < episodes:
                metrics = self.reset(slot, scheduled, seed_base, evaluation)
                scheduled += 1
                if metrics is not None:
                    results.append(metrics); completed += 1
            else:
                slot.done = True
        started = time.perf_counter()
        while completed < episodes:
            active = [slot for slot in self.slots if not slot.done]
            sequences = np.stack([raw_feature_sequence(slot.history) for slot in active])
            normalized = self.normalizer.numpy(sequences)
            tensor = torch.from_numpy(normalized).to(self.device)
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
            ):
                actions = self.policy(tensor).float().cpu().numpy()
            for slot, action in zip(active, actions):
                slot.connection.send((
                    "advance", action, bool(self.rng.random() < beta), threshold,
                ))
            responses = [self.receive(slot, "step") for slot in active]
            for slot, sequence, normalized_sequence, response in zip(
                active, sequences, normalized, responses
            ):
                (
                    _, valid, target, _, intervention, transition, disagreement,
                    dynamics, dynamics_valid, packed, done, metrics,
                ) = response
                if not evaluation and valid and target is not None and replay is not None:
                    priority = 1.0 + min(2.0, float(disagreement))
                    if transition:
                        priority *= float(self.settings.get("transition_priority", 4.0))
                    if intervention:
                        priority *= float(self.settings.get("intervention_priority", 1.5))
                    replay.add(ReplaySample(
                        normalized_sequence.astype(np.float32), np.asarray(target, np.float32),
                        sequence[-1, 71:75].astype(np.float32),
                        np.asarray(dynamics, np.float32), bool(dynamics_valid), priority,
                        bool(transition), bool(intervention),
                    ))
                slot.history.append_packed(packed); slot.done = bool(done); steps += 1
                if metrics is not None:
                    results.append(metrics); completed += 1
                    if scheduled < episodes:
                        reset_metrics = self.reset(slot, scheduled, seed_base, evaluation)
                        scheduled += 1
                        if reset_metrics is not None:
                            results.append(reset_metrics); completed += 1
            interval = int(self.settings.get("progress_log_interval", 5000))
            if interval > 0 and steps and steps % interval < len(active):
                print(
                    f"collector evaluation={int(evaluation)} episodes={completed}/{episodes} "
                    f"steps={steps} steps_s={steps / max(time.perf_counter()-started,1e-6):.1f} "
                    f"replay={len(replay) if replay is not None else 0}", flush=True,
                )
        return results[:episodes], steps

    def close(self) -> None:
        for slot in self.slots:
            if slot.process.is_alive():
                try: slot.connection.send(("close",))
                except (BrokenPipeError, EOFError): pass
        for slot in self.slots:
            if slot.process.is_alive():
                try: self.receive(slot, "closed")
                except (BrokenPipeError, EOFError, RuntimeError): pass
            slot.process.join(timeout=5)
            if slot.process.is_alive():
                slot.process.terminate(); slot.process.join(timeout=2)
            slot.connection.close()


@dataclass
class OfflineData:
    states: np.ndarray
    targets: np.ndarray
    previous: np.ndarray
    dynamics: np.ndarray
    dynamics_valid: np.ndarray


@torch.no_grad()
def load_offline(
    settings: Mapping[str, Any], normalizer: FeatureNormalizer, device: str,
) -> OfflineData:
    train_paths, _ = stratified_episode_split(
        settings["data"], float(settings.get("validation_fraction", 0.12)),
        int(settings.get("validation_seed", 2026081501)), tracks=settings["track"],
    )
    dataset_settings = dict(settings); dataset_settings["observation_source"] = "raw_estimate"
    dataset = make_dataset(dataset_settings, train_paths, None)
    maximum = min(int(settings.get("offline_max_windows", 0)) or len(dataset), len(dataset))
    subset = Subset(dataset, list(range(maximum)))
    loader = DataLoader(
        subset, batch_size=int(settings.get("offline_batch_size", 128)), shuffle=False,
        num_workers=int(settings.get("offline_workers", 2)),
    )
    states: list[np.ndarray] = []; targets: list[np.ndarray] = []
    previous: list[np.ndarray] = []; dynamics: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    for raw in loader:
        batch = {key: value.to(device) for key, value in raw.items()}
        features, _, _ = compact_features(
            batch, source="raw_estimate", policy_history=POLICY_HISTORY,
            raw_history=RAW_HISTORY, encoder=None, amp=True, encoder_grad=False,
        )
        index = RAW_HISTORY - 1
        states.append(normalizer.numpy(features.cpu().numpy()).astype(np.float16))
        targets.append(batch["commanded_action"][:, index].cpu().numpy().astype(np.float16))
        previous.append(batch["previous_action"][:, index].cpu().numpy().astype(np.float16))
        dynamics.append(
            (batch["task_state"][:, RAW_HISTORY] - batch["task_state"][:, index])
            .cpu().numpy().astype(np.float16)
        )
        valid.append(
            (batch["gate_index"][:, RAW_HISTORY, 0] == batch["gate_index"][:, index, 0])
            .cpu().numpy()
        )
    dataset.close()
    return OfflineData(
        np.concatenate(states), np.concatenate(targets), np.concatenate(previous),
        np.concatenate(dynamics), np.concatenate(valid),
    )


def loss_function(
    policy: PrivilegedMLPPolicy, state: torch.Tensor, target: torch.Tensor,
    previous: torch.Tensor, dynamics: torch.Tensor, dynamics_valid: torch.Tensor,
    settings: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted = policy(state)
    action = F.smooth_l1_loss(
        predicted, target, beta=float(settings.get("huber_beta", 0.05))
    )
    delta = F.smooth_l1_loss(
        predicted - previous, target - previous,
        beta=float(settings.get("delta_huber_beta", 0.04)),
    )
    predicted_dynamics = policy.predict_dynamics(state)
    dynamics_loss = (
        F.smooth_l1_loss(predicted_dynamics[dynamics_valid], dynamics[dynamics_valid], beta=0.02)
        if dynamics_valid.any() else predicted_dynamics.sum() * 0.0
    )
    total = action + float(settings.get("delta_weight", 0.35)) * delta + float(
        settings.get("dynamics_weight", 0.15)
    ) * dynamics_loss
    return total, {
        "loss": total, "action_loss": action, "delta_loss": delta,
        "dynamics_loss": dynamics_loss,
        "action_rmse": (predicted - target).square().mean().sqrt(),
    }


def train_round(
    policy: PrivilegedMLPPolicy, optimizer: torch.optim.Optimizer,
    offline: OfflineData, replay: CompactReplay, settings: Mapping[str, Any],
    device: str, rng: np.random.Generator,
) -> dict[str, float]:
    policy.train(); totals: dict[str, list[float]] = {}
    batch_size = int(settings.get("batch_size", 1536))
    online_fraction = float(settings.get("online_fraction", 0.55))
    for _ in range(int(settings.get("updates_per_round", 700))):
        online_count = min(int(round(batch_size * online_fraction)), len(replay))
        offline_count = batch_size - online_count
        selected = rng.integers(len(offline.states), size=offline_count)
        batches = [{
            "state": torch.from_numpy(offline.states[selected]).to(device).float(),
            "target": torch.from_numpy(offline.targets[selected]).to(device).float(),
            "previous": torch.from_numpy(offline.previous[selected]).to(device).float(),
            "dynamics": torch.from_numpy(offline.dynamics[selected]).to(device).float(),
            "dynamics_valid": torch.from_numpy(offline.dynamics_valid[selected]).to(device),
        }]
        if online_count:
            batches.append(replay.sample(online_count, device))
        batch = {
            key: torch.cat([value[key] for value in batches], 0)
            for key in batches[0]
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss, metrics = loss_function(policy, settings=settings, **batch)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float(settings.get("gradient_clip", 2.0))
        )
        optimizer.step()
        metrics["gradient_norm"] = gradient
        for key, value in metrics.items():
            totals.setdefault(key, []).append(float(value.detach()))
    policy.eval()
    return {key: float(np.mean(values)) for key, values in totals.items()}


def selection_score(metrics: Mapping[str, float], settings: Mapping[str, Any]) -> float:
    return (
        float(settings.get("selection_success_weight", 8.0)) * metrics["success"]
        + float(settings.get("selection_p3_weight", 2.0)) * metrics["p3"]
        + float(settings.get("selection_p2_weight", 0.75)) * metrics["p2"]
        + float(settings.get("selection_p1_weight", 0.10)) * metrics["p1"]
        - float(settings.get("selection_crash_weight", 0.50)) * metrics["crash"]
    )


def checkpoint_payload(
    policy: PrivilegedMLPPolicy, normalizer: FeatureNormalizer,
    optimizer: torch.optim.Optimizer, config: Mapping[str, Any],
    initial: Path, round_index: int, environment_steps: int,
) -> dict[str, Any]:
    return {
        "contract": "compact-raw-estimate-causal-direct-ctbr-dagger-v1",
        "observation_source": "raw_estimate", "model": policy.state_dict(),
        "model_config": policy.model_config(), "normalizer": normalizer.state_dict(),
        "optimizer": optimizer.state_dict(), "initial_checkpoint": str(initial),
        "round": round_index, "environment_steps": environment_steps,
        "training_config": dict(config), "rng_state": capture_rng_state(),
        "action_horizon": 1, "control_hz": 90,
    }


def run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    settings = config["dagger"]
    seed = int(settings.get("seed", 2026081804))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    initial_path = resolve_ranked_checkpoint(settings["initial_checkpoint"])
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    if initial.get("observation_source") != "raw_estimate":
        raise ValueError("compact raw DAgger requires a raw_estimate BC checkpoint")
    policy = PrivilegedMLPPolicy(**initial["model_config"]).to(args.device)
    policy.load_state_dict(initial["model"]); policy.eval()
    normalizer = FeatureNormalizer.from_state_dict(initial["normalizer"])
    if policy.input_dim != FEATURE_DIM or policy.context_steps != POLICY_HISTORY:
        raise ValueError("raw DAgger policy contract mismatch")
    print("loading matched compact offline replay", flush=True)
    offline = load_offline(settings, normalizer, args.device)
    print(f"offline_windows={len(offline.states)}", flush=True)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(settings.get("learning_rate", 1e-4)),
        weight_decay=float(settings.get("weight_decay", 1e-5)),
        fused=args.device.startswith("cuda"),
    )
    replay = CompactReplay(int(settings.get("replay_capacity", 120000)), seed)
    stage = RacingCurriculumStage.from_mapping(settings["curriculum"])
    manager = CheckpointManager.from_config(config); logger = init_wandb(config)
    collector = ProcessCollector(policy, normalizer, settings, stage, args.device)
    environment_steps = 0; rng = np.random.default_rng(seed)
    try:
        baseline_rows, _ = collector.collect(
            episodes=int(settings.get("evaluation_episodes", 96)), beta=0.0,
            threshold=1e9, seed_base=int(settings.get("evaluation_seed", 2026081850)),
            evaluation=True, replay=None,
        )
        baseline = mean_metrics(baseline_rows)
        baseline["selection_score"] = selection_score(baseline, settings)
        logger.log_eval(baseline, 0)
        print(
            f"baseline p1={baseline['p1']:.3f} p2={baseline['p2']:.3f} "
            f"p3={baseline['p3']:.3f} full={baseline['success']:.3f} "
            f"crash={baseline['crash']:.3f}", flush=True,
        )
        rounds = int(settings.get("rounds", 6))
        for round_index in range(1, rounds + 1):
            started = time.perf_counter()
            fraction = (round_index - 1) / max(rounds - 1, 1)
            beta = float(settings.get("teacher_beta_start", 0.45)) * (1-fraction) + float(
                settings.get("teacher_beta_end", 0.05)
            ) * fraction
            threshold = float(settings.get("disagreement_threshold_start", 0.14)) * (1-fraction) + float(
                settings.get("disagreement_threshold_end", 0.07)
            ) * fraction
            collection_started = time.perf_counter()
            collected, new_steps = collector.collect(
                episodes=int(settings.get("episodes_per_round", 96)), beta=beta,
                threshold=threshold, seed_base=seed + round_index * 1000003,
                evaluation=False, replay=replay,
            )
            collection_seconds = time.perf_counter() - collection_started
            environment_steps += new_steps
            collect_metrics = mean_metrics(collected)
            update_metrics = train_round(
                policy, optimizer, offline, replay, settings, args.device, rng
            )
            evaluation_rows, _ = collector.collect(
                episodes=int(settings.get("evaluation_episodes", 96)), beta=0.0,
                threshold=1e9, seed_base=int(settings.get("evaluation_seed", 2026081850)),
                evaluation=True, replay=None,
            )
            evaluation = mean_metrics(evaluation_rows)
            evaluation["selection_score"] = selection_score(evaluation, settings)
            train_metrics = {
                **update_metrics,
                **{f"collect_{key}": value for key, value in collect_metrics.items()},
                **replay.metrics(), "round": float(round_index),
                "environment_steps": float(environment_steps),
                "collection_steps_per_second": new_steps / max(collection_seconds, 1e-6),
                "round_seconds": time.perf_counter() - started,
            }
            eval_metrics = {
                **evaluation, "round": float(round_index),
                "environment_steps": float(environment_steps),
            }
            logger.log_train(train_metrics, round_index); logger.log_eval(eval_metrics, round_index)
            manager.save_eval(
                eval_metrics, step=environment_steps,
                summary="Compact raw-estimate direct-CTBR process DAgger evaluation.",
                kind="closed-loop",
            )
            manager.save(
                checkpoint_payload(
                    policy, normalizer, optimizer, config, initial_path,
                    round_index, environment_steps,
                ),
                step=environment_steps, metrics=eval_metrics,
            )
            print(
                f"round={round_index}/{rounds} env_steps={environment_steps} "
                f"replay={len(replay)} teacher={collect_metrics['teacher_fraction']:.3f} "
                f"valid={collect_metrics['teacher_valid_fraction']:.4f} "
                f"disagreement={collect_metrics['disagreement']:.3f} "
                f"loss={update_metrics['loss']:.4f} p1={evaluation['p1']:.3f} "
                f"p2={evaluation['p2']:.3f} p3={evaluation['p3']:.3f} "
                f"full={evaluation['success']:.3f} crash={evaluation['crash']:.3f} "
                f"steps_s={train_metrics['collection_steps_per_second']:.1f} "
                f"replay_gib={train_metrics['replay_allocated_gib']:.2f}", flush=True,
            )
    finally:
        collector.close(); logger.finish()


if __name__ == "__main__":
    parsed, configuration = arguments()
    run(parsed, configuration)
