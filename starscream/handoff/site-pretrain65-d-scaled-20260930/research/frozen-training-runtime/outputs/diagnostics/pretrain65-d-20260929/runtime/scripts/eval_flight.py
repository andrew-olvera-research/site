#!/usr/bin/env python3
"""Closed-loop Flightmare evaluation for actor and MPPI flight policies."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml

from starscream.actor_critic import (
    GaussianActor,
    WAMGaussianActor,
    actor_state_dimension,
    build_actor_state,
)
from starscream.DiT import TokenizerFlowPolicy
from starscream.control_policy import (
    BoundedResidualFlowPolicy,
    BoundedResidualGaussianPolicy,
    BoundedResidualKnotPolicy,
    MultiRateFlowPolicy,
)
from starscream.flow_policy import tokenizer_policy_tokens
from starscream.env import FlightmareEnv
from starscream.env.real_course_suite import load_active_real_course_suite
from starscream.env.tracks import Track, matrix_quaternion
from starscream.mpcc import (
    MPCCConfig, MPCCController, RacingLinePlanner, RacingLinePlannerConfig,
)
from starscream.mppi import build_mppi_flight_policy
from starscream.tokenizer import ActionTokenizer, MultiModalTokenizer
from starscream.dynamics import RaceDynamics
from starscream.wam import frozen_world_backbone_tokens


DEFAULT_TRACKS = (
    "figure8", "split_s", "big_s", "kidney",
    "swift_eval_inspired", "vertical_3d",
)
EXPECTED_ACTOR_CONTRACT = (
    "rolling_posterior_latents_with_deltas_decoded_proprio_exact_applied_"
    "actions_timing_and_tokenizer_state_estimate"
)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


class ObservationHistory:
    """Online equivalent of DreamerSequenceDataset observation preprocessing."""

    def __init__(
        self,
        length: int,
        mask_size: tuple[int, int],
        *,
        estimate_dim: int = 0,
        estimate_seed: int = 0,
        control_hz: float = 90.0,
    ) -> None:
        self.length = int(length)
        self.control_hz = float(control_hz)
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        self.mask_size = tuple(mask_size)
        self.estimate_dim = int(estimate_dim)
        if self.estimate_dim not in {0, 32}:
            raise ValueError(f"unsupported online deployment estimate width {self.estimate_dim}")
        self.estimate_rng = np.random.default_rng(int(estimate_seed))
        self.records: deque[dict[str, np.ndarray]] = deque(maxlen=self.length)
        self.gate_truth: deque[tuple[int, np.ndarray]] = deque(maxlen=max(64, self.length + 8))
        self.previous_camera_time: float | None = None
        self.previous_sim_time: float | None = None
        self.previous_gate_index: int | None = None
        self.previous_vio: np.ndarray | None = None
        self.previous_velocity: np.ndarray | None = None
        self.vio_drift = np.zeros(12, np.float32)
        self.vio_age = 0.0
        self.gate_age = 0.0
        self.held_gate = np.zeros(9, np.float32)

    def _deployment_estimate(
        self,
        observation: dict[str, Any],
        mask: np.ndarray,
        timing: np.ndarray,
    ) -> np.ndarray:
        """Online causal analogue of DreamerSequenceDataset proxy_v1."""

        task = np.asarray(observation["task_state"], np.float32).copy()
        task[0:3] /= 20.0
        task[3:6] /= 30.0
        task[12:15] /= 6.0
        task[15:19] /= 4000.0
        gate_index = int(np.asarray(observation["flight_plan"]["index"])[0])
        gate_changed = self.previous_gate_index is not None and gate_index != self.previous_gate_index

        state_noise = np.asarray(
            [0.15 / 20.0] * 3 + [0.25 / 30.0] * 3 + [0.025] * 6,
            np.float32,
        )
        drift_noise = np.asarray(
            [0.004 / 20.0] * 3 + [0.008 / 30.0] * 3 + [0.0015] * 6,
            np.float32,
        )
        if gate_changed:
            self.vio_drift.fill(0.0)
        else:
            self.vio_drift += self.estimate_rng.normal(0.0, drift_noise).astype(np.float32)
        vio = task[:12] + self.estimate_rng.normal(0.0, state_noise).astype(np.float32)
        vio += self.vio_drift
        vio_valid = bool(self.estimate_rng.random() >= 0.03) or gate_changed
        if not vio_valid and self.previous_vio is not None and not gate_changed:
            vio = self.previous_vio.copy()
            self.vio_age += 1.0
        else:
            self.vio_age = 0.0

        gates = observation["gates"]
        current_gate = np.concatenate(
            [
                np.asarray(gates["position"][0], np.float32) / 20.0,
                np.asarray(gates["normal"][0], np.float32),
                np.asarray(gates["up"][0], np.float32),
            ]
        )
        self.gate_truth.append((gate_index, current_gate))
        camera_age = max(0, int(round(float(timing[0]))))
        source_offset = len(self.gate_truth) - 1 - camera_age
        visible = bool(mask.sum() > 0.0)
        gate_valid = visible and self.estimate_rng.random() >= 0.10
        if source_offset < 0 or self.gate_truth[source_offset][0] != gate_index:
            gate_valid = False
        if gate_valid:
            delayed_gate = self.gate_truth[source_offset][1].copy()
            noise = self.estimate_rng.normal(0.0, 1.0, 9).astype(np.float32)
            noise[:3] *= 0.18 / 20.0
            noise[3:9] *= 0.03
            delayed_gate += noise
            normal = delayed_gate[3:6]
            normal /= max(float(np.linalg.norm(normal)), 1e-6)
            up = delayed_gate[6:9] - float(np.dot(delayed_gate[6:9], normal)) * normal
            up /= max(float(np.linalg.norm(up)), 1e-6)
            delayed_gate[3:6] = normal
            delayed_gate[6:9] = up
            self.held_gate = delayed_gate
            self.gate_age = float(camera_age)
        elif not gate_changed:
            self.gate_age += 1.0
        else:
            self.held_gate.fill(0.0)
            self.gate_age = 0.0

        acceleration = np.zeros(3, np.float32)
        if self.previous_velocity is not None and not gate_changed:
            acceleration = (task[3:6] - self.previous_velocity) * self.control_hz
        acceleration += self.estimate_rng.normal(0.0, 0.03, 3).astype(np.float32)
        acceleration = np.clip(acceleration, -2.0, 2.0)
        uncertainty = np.asarray(
            [
                0.15 + 0.03 * self.vio_age,
                0.25 + 0.05 * self.vio_age,
                0.025 + 0.005 * self.vio_age,
                0.18 + 0.04 * self.gate_age,
            ],
            np.float32,
        )
        estimate = np.concatenate(
            [
                vio,
                self.held_gate,
                acceleration,
                uncertainty,
                np.asarray([vio_valid, gate_valid], np.float32),
                np.clip(np.asarray([self.vio_age, self.gate_age]) / 36.0, 0.0, 2.0),
            ]
        ).astype(np.float32)
        self.previous_gate_index = gate_index
        self.previous_vio = vio.copy()
        self.previous_velocity = task[3:6].copy()
        return estimate

    def append(self, observation: dict[str, Any]) -> None:
        mask = np.asarray(observation["gate_mask"], np.float32)
        if mask.shape != self.mask_size:
            raise ValueError(
                f"online gate mask has shape {mask.shape}, expected {self.mask_size}"
            )
        mask = mask[None] / 255.0
        body_rates = np.asarray(observation["measured"]["body_rates"], np.float32)
        motor_omega = np.asarray(observation["measured"]["motor_omega"], np.float32)
        previous_action = np.asarray(observation["previous_action"], np.float32)
        # Preserve collective commands above the legacy 30 m/s^2 ceiling.
        # The raw interface remaps this losslessly into the active teacher's
        # action coordinates; clipping here made every 30--40 m/s^2 command
        # indistinguishable in the next observation.
        normalized_previous = np.concatenate([
            previous_action[:1] / 15.0 - 1.0,
            np.clip(previous_action[1:] / 6.0, -1.0, 1.0),
        ]).astype(np.float32)
        proprio = np.concatenate(
            [
                np.clip(body_rates / 6.0, -1.0, 1.0),
                np.clip(motor_omega / 4000.0, 0.0, 1.5),
                normalized_previous,
            ]
        ).astype(np.float32)
        route = np.asarray(observation["flight_plan"]["records"], np.float32).copy()
        route[..., 0:3] /= 20.0
        route[..., 9:11] /= 5.0

        timestamp = observation["timestamp"]
        age = observation["age"]
        valid = observation["valid"]
        camera_time = float(timestamp["camera"])
        sim_time = float(timestamp["sim"])
        camera_new = self.previous_camera_time is None or camera_time != self.previous_camera_time
        dt_steps = (
            1.0 if self.previous_sim_time is None
            else max(0.0, (sim_time - self.previous_sim_time) * self.control_hz)
        )
        timing = np.asarray(
            [
                float(age["camera"]) * self.control_hz,
                float(age["body_rates"]) * self.control_hz,
                float(age["motor_omega"]) * self.control_hz,
                float(age["previous_action"]) * self.control_hz,
                float(valid["camera"]),
                float(valid["body_rates"]),
                float(valid["motor_omega"]),
                float(valid["previous_action"]),
                float(camera_new),
                dt_steps,
            ],
            dtype=np.float32,
        )
        record = {
            "mask": mask,
            "proprio": proprio,
            "route": route,
            "timing": timing,
            "previous_action": normalized_previous,
            "deployable_task_state": np.asarray(
                observation["task_state"], np.float32
            ).copy(),
            "gate_index": np.asarray(
                observation["flight_plan"]["index"], np.int64
            ).copy(),
        }
        if self.estimate_dim:
            record["estimate"] = self._deployment_estimate(observation, mask, timing)
        record["deployable_task_state"][0:3] /= 20.0
        record["deployable_task_state"][3:6] /= 30.0
        record["deployable_task_state"][12:15] /= 6.0
        record["deployable_task_state"][15:19] /= 4000.0
        if not all(np.all(np.isfinite(value)) for value in record.values()):
            raise ValueError("non-finite online actor observation")
        self.records.append(record)
        self.previous_camera_time = camera_time
        self.previous_sim_time = sim_time

    def snapshot(self) -> dict[str, np.ndarray]:
        """Return a compact, checkpoint-safe copy of the causal policy history.

        Semantic masks dominate the storage cost and are binary by contract, so
        they are bit-packed.  All other online-observable channels remain exact
        float32 arrays.  Simulator timestamps are intentionally not persisted:
        after a reset the next sample belongs to a fresh clock domain, while the
        archived timing channels already preserve the history seen by the actor.
        """

        if not self.records:
            raise RuntimeError("cannot snapshot an empty actor history")
        records = list(self.records)
        masks = np.stack([record["mask"] for record in records]) >= 0.5
        return {
            "mask_bits": np.packbits(masks, axis=-1),
            "mask_width": np.asarray(masks.shape[-1], np.int32),
            **{
                key: np.stack([record[key] for record in records]).astype(
                    np.float32, copy=True
                )
                for key in (
                    "proprio", "route", "timing", "previous_action",
                    "deployable_task_state", "estimate",
                )
                if key in records[0]
            },
            **(
                {"gate_index": np.stack(
                    [record["gate_index"] for record in records]
                ).astype(np.int64, copy=True)}
                if "gate_index" in records[0] else {}
            ),
        }

    def packed_latest(self) -> dict[str, np.ndarray]:
        """Serialize one compact record for process-isolated rollout transport."""

        if not self.records:
            raise RuntimeError("cannot pack an empty actor history")
        record = self.records[-1]
        mask = np.asarray(record["mask"] >= 0.5, np.bool_)
        return {
            "mask_bits": np.packbits(mask, axis=-1),
            "mask_width": np.asarray(mask.shape[-1], np.int32),
            **{
                key: np.asarray(value).copy()
                for key, value in record.items()
                if key != "mask"
            },
        }

    def append_packed(self, payload: dict[str, np.ndarray]) -> None:
        """Append a record emitted by a persistent rollout worker."""

        width = int(np.asarray(payload["mask_width"]).item())
        mask = np.unpackbits(
            np.asarray(payload["mask_bits"], np.uint8), axis=-1, count=width
        ).astype(np.float32)
        if mask.shape != (1, *self.mask_size):
            raise ValueError(f"packed mask has incompatible shape {mask.shape}")
        record = {
            "mask": mask,
            **{
                key: np.asarray(value).copy()
                for key, value in payload.items()
                if key not in {"mask_bits", "mask_width"}
            },
        }
        if not all(np.all(np.isfinite(value)) for value in record.values()):
            raise ValueError("packed online actor observation is non-finite")
        self.records.append(record)

    def restore(self, snapshot: dict[str, np.ndarray]) -> None:
        """Restore a history captured at the exact archived physical state."""

        legacy = {
            "mask_bits", "mask_width", "proprio", "route", "timing",
            "previous_action",
        }
        current = legacy | {"deployable_task_state", "gate_index"}
        estimator = current | {"estimate"}
        if frozenset(snapshot) not in {
            frozenset(legacy), frozenset(current), frozenset(estimator)
        }:
            raise ValueError("archived actor history has an incompatible schema")
        width = int(np.asarray(snapshot["mask_width"]).item())
        mask_bits = np.asarray(snapshot["mask_bits"], np.uint8)
        masks = np.unpackbits(mask_bits, axis=-1, count=width).astype(np.float32)
        arrays = {
            "mask": masks,
            **{
                key: np.asarray(snapshot[key], np.float32)
                for key in (
                    "proprio", "route", "timing", "previous_action",
                    "deployable_task_state", "estimate",
                )
                if key in snapshot
            },
            **(
                {"gate_index": np.asarray(snapshot["gate_index"], np.int64)}
                if "gate_index" in snapshot else {}
            ),
        }
        lengths = {value.shape[0] for value in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("archived actor history channels are not aligned")
        count = lengths.pop()
        if not 1 <= count <= self.length or masks.shape[1:] != (1, *self.mask_size):
            raise ValueError("archived actor history has an invalid length or mask shape")
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("archived actor history contains non-finite values")
        self.records.clear()
        for index in range(count):
            self.records.append({
                key: value[index].copy() for key, value in arrays.items()
            })
        self.previous_camera_time = None
        self.previous_sim_time = None

    def batch(
        self, device: str, *, repeat_first_padding: bool = False
    ) -> dict[str, torch.Tensor]:
        if not self.records:
            raise RuntimeError("actor observation history is empty")
        if len(self.records) != self.length and not repeat_first_padding:
            raise RuntimeError(
                f"actor requires {self.length} observations, history has {len(self.records)}"
            )
        records = list(self.records)
        if repeat_first_padding and len(records) < self.length:
            records = [records[0]] * (self.length - len(records)) + records
        return {
            key: torch.from_numpy(np.stack([record[key] for record in records]))
            .unsqueeze(0).to(device, non_blocking=True)
            for key in records[0]
        }


@dataclass
class PolicyBundle:
    path: Path
    name: str
    step: int
    actor: GaussianActor
    encoder: MultiModalTokenizer
    action_tokenizer: ActionTokenizer
    history: int
    patch: int
    device: str
    amp: bool
    contract: dict[str, Any]

    @torch.no_grad()
    def action_chunk(
        self, history: ObservationHistory, *, sample: bool
    ) -> tuple[np.ndarray, np.ndarray, float]:
        batch = history.batch(self.device)
        started = time.perf_counter()
        state = build_actor_state(
            self.encoder, batch, self.history, self.device, self.amp
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda") and self.amp,
        ):
            output = self.actor(state, self.action_tokenizer)
        normalized = output.rsample() if sample else output.mode()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return (
            normalized[0].float().cpu().numpy(),
            output.std[0].float().cpu().numpy(),
            latency_ms,
        )


@dataclass
class WAMPolicyBundle:
    path: Path
    name: str
    step: int
    actor: WAMGaussianActor
    encoder: MultiModalTokenizer
    action_tokenizer: ActionTokenizer
    dynamics: RaceDynamics
    history: int
    patch: int
    device: str
    amp: bool
    contract: dict[str, Any]
    backbone_signal_index: int | None
    backbone_step_index: int

    @torch.no_grad()
    def action_chunk(
        self, history: ObservationHistory, *, sample: bool
    ) -> tuple[np.ndarray, np.ndarray, float]:
        batch = history.batch(self.device)
        started = time.perf_counter()
        tokens = frozen_world_backbone_tokens(
            self.encoder, self.action_tokenizer, self.dynamics, batch,
            history=self.history, device=self.device, amp=self.amp,
            signal_index=self.backbone_signal_index,
            step_index=self.backbone_step_index,
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda") and self.amp,
        ):
            output = self.actor(tokens)
        normalized = output.rsample() if sample else output.mode()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return (
            normalized[0].float().cpu().numpy(),
            output.std[0].float().cpu().numpy(), latency_ms,
        )


@dataclass
class FlowPolicyBundle:
    path: Path
    name: str
    step: int
    actor: TokenizerFlowPolicy
    encoder: MultiModalTokenizer
    action_tokenizer: ActionTokenizer
    history: int
    patch: int
    device: str
    amp: bool
    contract: dict[str, Any]
    sampling_steps: int
    sampling_method: str
    warmup_steps: int = 0

    @torch.no_grad()
    def action_chunk(
        self, history: ObservationHistory, *, sample: bool
    ) -> tuple[np.ndarray, np.ndarray, float]:
        batch = history.batch(self.device, repeat_first_padding=True)
        started = time.perf_counter()
        observation_tokens, action_tokens = tokenizer_policy_tokens(
            self.encoder,
            self.action_tokenizer,
            batch,
            history_raw_steps=self.history,
            amp=self.amp,
        )
        previous = batch["previous_action"][:, self.history - 1]
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda") and self.amp,
        ):
            normalized = self.actor.sample(
                observation_tokens,
                action_tokens,
                steps=self.sampling_steps,
                method=self.sampling_method,
                previous_action=previous,
                deterministic=not sample,
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return (
            normalized[0].float().cpu().numpy(),
            np.zeros((self.actor.action_horizon, self.actor.action_dim), np.float32),
            latency_ms,
        )


@dataclass
class MultiRateFlowPolicyBundle:
    path: Path
    name: str
    step: int
    actor: MultiRateFlowPolicy
    encoder: Any
    history: int
    patch: int
    device: str
    amp: bool
    contract: dict[str, Any]
    sampling_steps: int
    sampling_method: str
    warmup_steps: int = 0
    action_blend: float = 1.0
    max_action_delta: tuple[float, float, float, float] | None = None

    @torch.no_grad()
    def action_chunk(
        self, history: ObservationHistory, *, sample: bool
    ) -> tuple[np.ndarray, np.ndarray, float]:
        batch = history.batch(self.device, repeat_first_padding=True)
        started = time.perf_counter()
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda") and self.amp,
        ):
            normalized = self.actor.sample(
                batch, history=self.history,
                steps=self.sampling_steps, method=self.sampling_method,
                deterministic=not sample,
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        proposed = normalized[0].float().cpu().numpy()
        previous = batch["previous_action"][0, self.history - 1].float().cpu().numpy()
        proposed = previous + float(self.action_blend) * (proposed - previous)
        if self.max_action_delta is not None:
            limit = np.asarray(self.max_action_delta, np.float32)
            proposed = previous + np.clip(proposed - previous, -limit, limit)
        return (
            np.clip(proposed, -1.0, 1.0)[None],
            np.zeros((1, 4), np.float32),
            latency_ms,
        )


@dataclass
class BoundedKnotPolicyBundle:
    path: Path
    name: str
    step: int
    actor: BoundedResidualKnotPolicy
    encoder: Any
    history: int
    patch: int
    device: str
    amp: bool
    contract: dict[str, Any]
    warmup_steps: int = 0
    last_commanded: np.ndarray | None = None

    def reset(self) -> None:
        self.last_commanded = None

    @torch.no_grad()
    def action_chunk(
        self, history: ObservationHistory, *, sample: bool
    ) -> tuple[np.ndarray, np.ndarray, float]:
        del sample  # This first ablation is deliberately deterministic.
        batch = history.batch(self.device, repeat_first_padding=True)
        started = time.perf_counter()
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda") and self.amp,
        ):
            anchor = (
                torch.as_tensor(
                    self.last_commanded, device=self.device, dtype=torch.float32
                ).unsqueeze(0)
                if self.last_commanded is not None else None
            )
            knot = self.actor.sample_knot(
                batch, history=self.history, anchor_action=anchor
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        commands = knot[0].float().cpu().numpy()
        self.last_commanded = commands[-1].copy()
        return commands, np.zeros_like(commands), latency_ms


@dataclass
class BoundedFlowPolicyBundle:
    path: Path
    name: str
    step: int
    actor: BoundedResidualFlowPolicy
    encoder: Any
    history: int
    patch: int
    device: str
    amp: bool
    contract: dict[str, Any]
    sampling_steps: int = 4
    sampling_method: str = "heun"
    warmup_steps: int = 0
    last_commanded: np.ndarray | None = None

    def reset(self) -> None:
        self.last_commanded = None

    @torch.no_grad()
    def action_chunk(
        self, history: ObservationHistory, *, sample: bool
    ) -> tuple[np.ndarray, np.ndarray, float]:
        batch = history.batch(self.device, repeat_first_padding=True)
        started = time.perf_counter()
        anchor = (
            torch.as_tensor(
                self.last_commanded, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            if self.last_commanded is not None
            else batch["previous_action"][:, self.history - 1]
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda") and self.amp,
        ):
            observation, applied = self.actor.policy_tokens(batch)
            knot, _ = self.actor.sample_policy(
                observation, applied, previous_action=anchor,
                steps=self.sampling_steps, method=self.sampling_method,
                deterministic=not sample,
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        commands = knot[0].float().cpu().numpy()
        self.last_commanded = commands[-1].copy()
        return commands, np.zeros_like(commands), latency_ms


@dataclass
class BoundedGaussianPolicyBundle:
    path: Path
    name: str
    step: int
    actor: BoundedResidualGaussianPolicy
    encoder: Any
    history: int
    patch: int
    device: str
    amp: bool
    contract: dict[str, Any]
    warmup_steps: int = 0
    last_commanded: np.ndarray | None = None

    def reset(self) -> None:
        self.last_commanded = None

    @torch.no_grad()
    def action_chunk(
        self, history: ObservationHistory, *, sample: bool
    ) -> tuple[np.ndarray, np.ndarray, float]:
        batch = history.batch(self.device, repeat_first_padding=True)
        started = time.perf_counter()
        anchor = (
            torch.as_tensor(
                self.last_commanded, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
            if self.last_commanded is not None
            else batch["previous_action"][:, self.history - 1]
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda") and self.amp,
        ):
            observation, applied = self.actor.policy_tokens(batch)
            knot, latent = self.actor.sample_policy(
                observation, applied, previous_action=anchor,
                deterministic=not sample,
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        commands = knot[0].float().cpu().numpy()
        self.last_commanded = commands[-1].copy()
        return commands, latent[0].float().cpu().numpy(), latency_ms


def load_wam_policy(path: Path, device: str, amp: bool) -> WAMPolicyBundle:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = dict(checkpoint.get("input_contract", {}))
    if contract.get("actor_state") != (
        "frozen_causal_world_backbone_tokens_from_posterior_beliefs_and_applied_actions"
    ):
        raise ValueError(f"{path} is not a supported WAM-backbone policy")
    observation_path = Path(checkpoint["observation_tokenizer"])
    action_path = Path(checkpoint["action_tokenizer"])
    embedded_world = "world_model" in checkpoint and "world_model_config" in checkpoint
    dynamics_path = Path(checkpoint["world_model_checkpoint"])
    references = (observation_path, action_path) if embedded_world else (
        observation_path, action_path, dynamics_path
    )
    for reference in references:
        if not reference.is_file():
            raise FileNotFoundError(f"WAM checkpoint reference is absent: {reference}")

    observation_checkpoint = torch.load(observation_path, map_location="cpu", weights_only=False)
    raw_observation_config = observation_checkpoint.get("model_config", {})
    observation_config = dict(raw_observation_config.get("encoder", raw_observation_config))
    encoder = MultiModalTokenizer(**observation_config)
    encoder.load_state_dict(observation_checkpoint.get("encoder", observation_checkpoint["model"]))
    encoder.eval().requires_grad_(False).to(device)

    action_checkpoint = torch.load(action_path, map_location="cpu", weights_only=False)
    raw_action_config = action_checkpoint.get("model_config", {})
    action_config = dict(raw_action_config.get("tokenizer", raw_action_config))
    action_tokenizer = ActionTokenizer(**action_config)
    action_tokenizer.load_state_dict(action_checkpoint.get("tokenizer", action_checkpoint["model"]))
    action_tokenizer.eval().requires_grad_(False).to(device)

    if embedded_world:
        dynamics = RaceDynamics(**checkpoint["world_model_config"])
        dynamics.load_state_dict(checkpoint["world_model"])
    else:
        dynamics_checkpoint = torch.load(dynamics_path, map_location="cpu", weights_only=False)
        dynamics = RaceDynamics(**dynamics_checkpoint["model_config"])
        dynamics.load_state_dict(dynamics_checkpoint["model"])
    dynamics.eval().requires_grad_(False).to(device)

    actor = WAMGaussianActor(**checkpoint["model_config"])
    actor.load_state_dict(checkpoint["model"])
    actor.eval().requires_grad_(False).to(device)
    training = checkpoint.get("training_config", {}).get("wam_bc", {})
    history = int(training.get("history", 24))
    return WAMPolicyBundle(
        path=path, name=path.parent.name + "/" + path.name,
        step=int(checkpoint.get("step", 0)), actor=actor, encoder=encoder,
        action_tokenizer=action_tokenizer, dynamics=dynamics,
        history=history, patch=encoder.temporal_patch_size,
        device=device, amp=amp, contract=contract,
        backbone_signal_index=training.get("backbone_signal_index"),
        backbone_step_index=int(training.get("backbone_step_index", 0)),
    )


def load_flow_policy(path: Path, device: str, amp: bool) -> FlowPolicyBundle:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = dict(checkpoint.get("input_contract", {}))
    if contract.get("actor_state") != (
        "three_macro_frozen_observation_latents_plus_causal_applied_action_tokens"
    ):
        raise ValueError(f"{path} is not a supported tokenizer flow policy")
    observation_path = Path(checkpoint["observation_tokenizer"])
    action_path = Path(checkpoint["action_tokenizer"])
    for reference in (observation_path, action_path):
        if not reference.is_file():
            raise FileNotFoundError(f"flow-policy checkpoint reference is absent: {reference}")

    observation_checkpoint = torch.load(observation_path, map_location="cpu", weights_only=False)
    raw_observation_config = observation_checkpoint.get("model_config", {})
    observation_config = dict(raw_observation_config.get("encoder", raw_observation_config))
    encoder = MultiModalTokenizer(**observation_config)
    encoder.load_state_dict(observation_checkpoint.get("encoder", observation_checkpoint["model"]))
    encoder.eval().requires_grad_(False).to(device)

    action_checkpoint = torch.load(action_path, map_location="cpu", weights_only=False)
    raw_action_config = action_checkpoint.get("model_config", {})
    action_config = dict(raw_action_config.get("tokenizer", raw_action_config))
    action_tokenizer = ActionTokenizer(**action_config)
    action_tokenizer.load_state_dict(action_checkpoint.get("tokenizer", action_checkpoint["model"]))
    action_tokenizer.eval().requires_grad_(False).to(device)

    actor = TokenizerFlowPolicy(**checkpoint["model_config"])
    actor.load_state_dict(checkpoint["model"])
    actor.eval().requires_grad_(False).to(device)
    settings = checkpoint.get("training_config", {}).get("flow_policy_bc", {})
    history = actor.history_steps * encoder.temporal_patch_size
    if actor.action_horizon != action_tokenizer.temporal_patch_size:
        raise ValueError("flow policy and action tokenizer chunk lengths differ")
    return FlowPolicyBundle(
        path=path,
        name=path.parent.name + "/" + path.name,
        step=int(checkpoint.get("step", 0)),
        actor=actor,
        encoder=encoder,
        action_tokenizer=action_tokenizer,
        history=history,
        patch=encoder.temporal_patch_size,
        device=device,
        amp=amp,
        contract=contract,
        sampling_steps=int(settings.get("sampling_steps", 4)),
        sampling_method=str(settings.get("sampling_method", "heun")),
    )


def load_multirate_flow_policy(
    path: Path, device: str, amp: bool
) -> MultiRateFlowPolicyBundle:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = dict(checkpoint.get("input_contract", {}))
    if contract.get("observation") != "nine_raw_90hz_ticks_late_fused_fast_visual_route":
        raise ValueError(f"{path} is not a supported multi-rate flow policy")
    actor = MultiRateFlowPolicy(**checkpoint["model_config"])
    actor.load_state_dict(checkpoint["model"])
    actor.eval().requires_grad_(False).to(device)
    settings = checkpoint.get("training_config", {}).get("multirate_flow_bc", {})
    history = int(settings.get("history_raw_steps", 9))
    image_size = tuple(int(value) for value in settings.get("image_size", (128, 160)))
    return MultiRateFlowPolicyBundle(
        path=path, name=path.parent.name + "/" + path.name,
        step=int(checkpoint.get("step", 0)), actor=actor,
        encoder=SimpleNamespace(image_size=image_size), history=history, patch=1,
        device=device, amp=amp, contract=contract,
        sampling_steps=int(settings.get("sampling_steps", 1)),
        sampling_method=str(settings.get("sampling_method", "euler")),
    )


def load_bounded_knot_policy(
    path: Path, device: str, amp: bool
) -> BoundedKnotPolicyBundle:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = dict(checkpoint.get("input_contract", {}))
    if contract.get("action") != "bounded_residual_ctbr_control_knot":
        raise ValueError(f"{path} is not a supported bounded-knot policy")
    actor = BoundedResidualKnotPolicy(**checkpoint["model_config"])
    actor.load_state_dict(checkpoint["model"])
    actor.eval().requires_grad_(False).to(device)
    settings = checkpoint.get("training_config", {}).get("bounded_knot_bc", {})
    history = int(settings.get("history_raw_steps", 9))
    image_size = tuple(int(value) for value in settings.get("image_size", (128, 160)))
    if int(contract.get("action_chunk_steps", -1)) != actor.knot_steps:
        raise ValueError("bounded-knot checkpoint chunk contract is inconsistent")
    return BoundedKnotPolicyBundle(
        path=path, name=path.parent.name + "/" + path.name,
        step=int(checkpoint.get("step", 0)), actor=actor,
        encoder=SimpleNamespace(image_size=image_size), history=history, patch=1,
        device=device, amp=amp, contract=contract,
    )


def load_bounded_flow_policy(
    path: Path, device: str, amp: bool
) -> BoundedFlowPolicyBundle:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = dict(checkpoint.get("input_contract", {}))
    if contract.get("rl_adapter") != "bounded_residual_flow_correction":
        raise ValueError(f"{path} is not a supported bounded-flow RL policy")
    base = BoundedResidualKnotPolicy(**checkpoint["base_model_config"])
    actor = BoundedResidualFlowPolicy(base, **checkpoint["model_config"])
    actor.load_state_dict(checkpoint["model"])
    actor.eval().requires_grad_(False).to(device)
    settings = checkpoint.get("training_config", {}).get("flow_fpo", {})
    history = int(settings.get("history_raw_steps", actor.history_raw_steps))
    image_size = tuple(int(value) for value in settings.get("image_size", (128, 160)))
    return BoundedFlowPolicyBundle(
        path=path, name=path.parent.name + "/" + path.name,
        step=int(checkpoint.get("step", 0)), actor=actor,
        encoder=SimpleNamespace(image_size=image_size), history=history, patch=1,
        device=device, amp=amp, contract=contract,
        sampling_steps=int(settings.get("sampling_steps", 4)),
        sampling_method=str(settings.get("sampling_method", "heun")),
    )


def load_bounded_gaussian_policy(
    path: Path, device: str, amp: bool
) -> BoundedGaussianPolicyBundle:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = dict(checkpoint.get("input_contract", {}))
    if contract.get("rl_adapter") != "bounded_residual_exact_gaussian_correction":
        raise ValueError(f"{path} is not a supported bounded Gaussian RL policy")
    base = BoundedResidualKnotPolicy(**checkpoint["base_model_config"])
    actor = BoundedResidualGaussianPolicy(base, **checkpoint["model_config"])
    actor.load_state_dict(checkpoint["model"])
    actor.eval().requires_grad_(False).to(device)
    settings = checkpoint.get("training_config", {}).get("flow_fpo", {})
    history = int(settings.get("history_raw_steps", actor.history_raw_steps))
    image_size = tuple(int(value) for value in settings.get("image_size", (128, 160)))
    return BoundedGaussianPolicyBundle(
        path=path, name=path.parent.name + "/" + path.name,
        step=int(checkpoint.get("step", 0)), actor=actor,
        encoder=SimpleNamespace(image_size=image_size), history=history, patch=1,
        device=device, amp=amp, contract=contract,
    )


def load_any_policy(path: Path, device: str, amp: bool):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = dict(checkpoint.get("input_contract", {}))
    if contract.get("rl_adapter") == "bounded_residual_exact_gaussian_correction":
        return load_bounded_gaussian_policy(path, device, amp)
    if contract.get("rl_adapter") == "bounded_residual_flow_correction":
        return load_bounded_flow_policy(path, device, amp)
    if contract.get("action") == "bounded_residual_ctbr_control_knot":
        return load_bounded_knot_policy(path, device, amp)
    if contract.get("observation") == "nine_raw_90hz_ticks_late_fused_fast_visual_route":
        return load_multirate_flow_policy(path, device, amp)
    if contract.get("actor_state") == (
        "three_macro_frozen_observation_latents_plus_causal_applied_action_tokens"
    ):
        return load_flow_policy(path, device, amp)
    if str(contract.get("actor_state", "")).startswith("frozen_causal_world_backbone_tokens"):
        return load_wam_policy(path, device, amp)
    return load_policy(path, device, amp)


def load_policy(path: Path, device: str, amp: bool) -> PolicyBundle:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    contract = dict(checkpoint.get("input_contract", {}))
    if contract.get("actor_state") != EXPECTED_ACTOR_CONTRACT:
        raise ValueError(
            f"{path} actor-state contract is missing or incompatible: "
            f"{contract.get('actor_state')!r}"
        )
    if contract.get("action") != "normalized_commanded_ctbr":
        raise ValueError(f"{path} does not emit normalized commanded CTBR")
    observation_path = Path(checkpoint["observation_tokenizer"])
    action_path = Path(checkpoint["action_tokenizer"])
    references = [observation_path, action_path]
    if checkpoint.get("world_model_checkpoint"):
        references.append(Path(checkpoint["world_model_checkpoint"]))
    for reference in references:
        if not reference.is_file():
            raise FileNotFoundError(f"checkpoint contract reference is absent: {reference}")

    observation_checkpoint = torch.load(
        observation_path, map_location="cpu", weights_only=False
    )
    raw_observation_config = observation_checkpoint.get("model_config", {})
    observation_config = dict(
        raw_observation_config.get("encoder", raw_observation_config)
    )
    encoder = MultiModalTokenizer(**observation_config)
    encoder.load_state_dict(
        observation_checkpoint.get("encoder", observation_checkpoint["model"])
    )
    if "observation_encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["observation_encoder"])
    encoder.eval().requires_grad_(False).to(device)

    action_checkpoint = torch.load(action_path, map_location="cpu", weights_only=False)
    raw_action_config = action_checkpoint.get("model_config", {})
    action_config = dict(raw_action_config.get("tokenizer", raw_action_config))
    action_tokenizer = ActionTokenizer(**action_config)
    action_tokenizer.load_state_dict(
        action_checkpoint.get("tokenizer", action_checkpoint["model"])
    )
    action_tokenizer.eval().requires_grad_(False).to(device)

    actor_config = dict(checkpoint["model_config"])
    actor = GaussianActor(**actor_config)
    actor.load_state_dict(checkpoint["model"])
    actor.eval().requires_grad_(False).to(device)
    history = int(actor_config["temporal_context"]) * encoder.temporal_patch_size
    expected_width = actor_state_dimension(encoder)
    if int(actor_config["input_dim"]) != expected_width:
        raise ValueError(
            f"actor input width {actor_config['input_dim']} != shared contract {expected_width}"
        )
    if actor.action_horizon != action_tokenizer.temporal_patch_size:
        raise ValueError("actor chunk and action tokenizer macro patch do not match")
    if int(contract.get("action_chunk_steps", -1)) != actor.action_horizon:
        raise ValueError("checkpoint action_chunk_steps does not match actor")
    if encoder.temporal_patch_size != action_tokenizer.temporal_patch_size:
        raise ValueError("observation/action tokenizer patches do not match")
    return PolicyBundle(
        path=path,
        name=path.parent.name + "/" + path.name,
        step=int(checkpoint.get("step", 0)),
        actor=actor,
        encoder=encoder,
        action_tokenizer=action_tokenizer,
        history=history,
        patch=encoder.temporal_patch_size,
        device=device,
        amp=amp,
        contract=contract,
    )


def normalized_to_ctbr(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, np.float32)
    result = np.empty(4, np.float32)
    result[0] = (np.clip(action[0], -1.0, 1.0) + 1.0) * 15.0
    result[1:] = np.clip(action[1:], -1.0, 1.0) * 6.0
    return result


def ctbr_to_normalized(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, np.float32)
    return np.concatenate(
        [action[:1] / 15.0 - 1.0, action[1:] / 6.0]
    ).clip(-1.0, 1.0).astype(np.float32)


def spawn_state(track: Track, gate_index: int, profile: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gate = track.gates[gate_index]
    state = np.zeros(25, np.float32)
    if profile == "clean":
        lateral = rng.uniform(-0.10, 0.10)
        vertical = rng.uniform(-0.05, 0.05)
        speed = rng.uniform(2.5, 3.5)
        side_speed = rng.uniform(-0.15, 0.15)
        body_rate = rng.uniform(-0.10, 0.10, size=3)
    elif profile == "recovery":
        lateral = rng.uniform(-0.9, 0.9)
        vertical = rng.uniform(-0.4, 0.4)
        speed = rng.uniform(1.0, 4.0)
        side_speed = rng.uniform(-1.5, 1.5)
        body_rate = rng.uniform(-1.2, 1.2, size=3)
    else:
        raise ValueError(f"unknown spawn profile {profile!r}")
    state[0:3] = (
        gate.position - 3.0 * gate.normal
        + lateral * gate.lateral + vertical * gate.up
    )
    state[2] = max(float(state[2]), 0.75)
    state[3:7] = matrix_quaternion(gate.directed_rotation)
    state[7:10] = speed * gate.normal + side_speed * gate.lateral
    state[10:13] = body_rate
    return state


def evaluate_episode(
    policy: Any | None,
    *,
    track: str,
    profile: str,
    episode_index: int,
    seed: int,
    max_steps: int,
    sample_actions: bool,
    image_delay: float,
    action_delay: float,
    warmup_policy: str,
    expert_only: bool = False,
) -> dict[str, Any]:
    if policy is not None and hasattr(policy, "reset"):
        policy.reset()
    env = FlightmareEnv(
        track=track,
        next_gates=3,
        image_size=(160, 128),
        control_dt=1.0 / 90.0,
        render_observations=False,
        mask_source="geometry",
        mask_size=(160, 128),
        image_delay=image_delay,
        action_delay=action_delay,
        terminate_on_collision=True,
    )
    try:
        gate_index = episode_index % len(env.track.gates)
        episode_seed = seed + 1009 * episode_index
        state = spawn_state(env.track, gate_index, profile, episode_seed)
        observation, reset_info = env.reset(
            seed=episode_seed,
            options={"gate_index": gate_index, "state": state},
        )
        history_length = policy.history if policy is not None else 24
        mask_size = tuple(policy.encoder.image_size) if policy is not None else (128, 160)
        history = ObservationHistory(
            history_length,
            mask_size,
            estimate_dim=int(getattr(policy.encoder, "estimate_dim", 0)) if policy is not None else 0,
            estimate_seed=episode_seed,
        )
        history.append(observation)
        mpcc = None
        if warmup_policy == "mpcc" or expert_only:
            line = RacingLinePlanner(
                RacingLinePlannerConfig(
                    offset_iterations=30,
                    cache_directory="/workspace/outputs/evals/racing-lines",
                )
            ).plan(env.track)
            mpcc = MPCCController(
                env.track,
                line,
                config=MPCCConfig(backend="predictive", actuation_delay=action_delay),
            )
            mpcc.reset()
        terminated = False
        warmup_steps = int(getattr(policy, "warmup_steps", history_length - 1)) if policy is not None else history_length - 1
        last_info: dict[str, Any] = {
            "unity_collision": False, "ground_contact": False,
            "course_progress": 0.0, "time": 0.0,
        }
        for _ in range(warmup_steps):
            warmup_action = (
                mpcc(observation).action.as_array()
                if mpcc is not None
                else np.asarray([9.81, 0.0, 0.0, 0.0], np.float32)
            )
            observation, _, terminated, _, last_info = env.step(
                warmup_action
            )
            history.append(observation)
            if terminated:
                break

        start_passed = env.tracker.passed_count
        start_progress = env._ordered_course_progress(
            observation["state"][0:3], env.tracker.index, env.tracker.lap
        )
        start_time = float(observation["timestamp"]["sim"])
        reward_sum = 0.0
        commanded_normalized: list[np.ndarray] = []
        commanded_ctbr: list[np.ndarray] = []
        predicted_std: list[np.ndarray] = []
        inference_ms: list[float] = []
        planner_diagnostics: dict[str, list[float]] = defaultdict(list)
        minimum_altitude = float(observation["state"][2])
        maximum_speed = float(np.linalg.norm(observation["state"][7:10]))
        maximum_body_rate = float(np.linalg.norm(observation["state"][10:13]))
        maximum_progress = start_progress
        policy_steps = 0

        while not terminated and policy_steps < max_steps:
            if expert_only:
                started = time.perf_counter()
                expert_action = mpcc(observation).action.as_array()
                latency_ms = (time.perf_counter() - started) * 1000.0
                chunk = ctbr_to_normalized(expert_action)[None]
                chunk_std = np.zeros_like(chunk)
            else:
                if policy is None:
                    raise RuntimeError("actor policy is required unless expert_only=true")
                chunk, chunk_std, latency_ms = policy.action_chunk(
                    history, sample=sample_actions
                )
                for key, value in (getattr(policy, "last_diagnostics", None) or {}).items():
                    planner_diagnostics[key].append(float(value))
            inference_ms.append(latency_ms)
            for normalized_action, standard_deviation in zip(chunk, chunk_std):
                physical_action = normalized_to_ctbr(normalized_action)
                observation, reward, terminated, _, last_info = env.step(physical_action)
                history.append(observation)
                commanded_normalized.append(normalized_action.copy())
                commanded_ctbr.append(physical_action.copy())
                predicted_std.append(standard_deviation.copy())
                reward_sum += float(reward)
                policy_steps += 1
                minimum_altitude = min(minimum_altitude, float(observation["state"][2]))
                maximum_speed = max(
                    maximum_speed, float(np.linalg.norm(observation["state"][7:10]))
                )
                maximum_body_rate = max(
                    maximum_body_rate, float(np.linalg.norm(observation["state"][10:13]))
                )
                maximum_progress = max(
                    maximum_progress, float(last_info["course_progress"])
                )
                if env.tracker.passed_count - start_passed >= len(env.track.gates):
                    terminated = True
                    break
                if terminated or policy_steps >= max_steps:
                    break

        actions = np.asarray(commanded_normalized, np.float32)
        physical = np.asarray(commanded_ctbr, np.float32)
        uncertainty = np.asarray(predicted_std, np.float32)
        gates_passed = env.tracker.passed_count - start_passed
        completed = gates_passed >= len(env.track.gates)
        elapsed = float(last_info.get("time", start_time) - start_time)
        progress = max(0.0, maximum_progress - start_progress)
        course_length = max(float(env._course_length), 1e-6)
        saturation = (
            np.abs(actions) >= 0.98 if actions.size else np.zeros((0, 4), bool)
        )
        slew = (
            np.abs(np.diff(actions, axis=0)).mean(axis=0)
            if len(actions) > 1 else np.zeros(4, np.float32)
        )
        return {
            "checkpoint": "mpcc-predictive-baseline" if expert_only else policy.name,
            "checkpoint_step": 0 if expert_only else policy.step,
            "track": env.track.name,
            "track_fingerprint": reset_info["track_fingerprint"],
            "profile": profile,
            "episode": episode_index,
            "seed": episode_seed,
            "completed": completed,
            "gates_passed": gates_passed,
            "gate_completion": min(1.0, gates_passed / len(env.track.gates)),
            "progress_fraction": min(1.0, progress / course_length),
            "elapsed_seconds": elapsed,
            "policy_steps": policy_steps,
            "warmup_steps": warmup_steps,
            "warmup_policy": warmup_policy,
            "reward_sum": reward_sum,
            "terminated": bool(terminated and not completed),
            "unity_collision": bool(last_info.get("unity_collision", False)),
            "ground_contact": bool(last_info.get("ground_contact", False)),
            "minimum_altitude": minimum_altitude,
            "maximum_speed": maximum_speed,
            "maximum_body_rate": maximum_body_rate,
            "action_saturation_fraction": saturation.mean(axis=0) if len(saturation) else np.zeros(4),
            "any_action_saturation_fraction": saturation.any(axis=1).mean() if len(saturation) else 0.0,
            "normalized_action_slew": slew,
            "mean_ctbr": physical.mean(axis=0) if len(physical) else np.zeros(4),
            "mean_predicted_std": uncertainty.mean(axis=(0, 1)) if uncertainty.size else 0.0,
            "inference_ms_mean": float(np.mean(inference_ms)) if inference_ms else 0.0,
            "inference_ms_p95": float(np.quantile(inference_ms, 0.95)) if inference_ms else 0.0,
            "inference_ms_p99": float(np.quantile(inference_ms, 0.99)) if inference_ms else 0.0,
            "planner_diagnostics": {
                key: {
                    "mean": float(np.mean(values)),
                    "p95": float(np.quantile(values, 0.95)),
                }
                for key, values in planner_diagnostics.items()
            },
        }
    finally:
        env.close()


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["checkpoint"], row["track"], row["profile"])].append(row)
        by_checkpoint[row["checkpoint"]].append(row)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        successes = [row for row in items if row["completed"]]
        return {
            "episodes": len(items),
            "completion_rate": mean(items, "completed"),
            "mean_gate_completion": mean(items, "gate_completion"),
            "mean_progress_fraction": mean(items, "progress_fraction"),
            "crash_rate": mean(items, "terminated"),
            "ground_contact_rate": mean(items, "ground_contact"),
            "mean_reward": mean(items, "reward_sum"),
            "successful_lap_time": mean(successes, "elapsed_seconds") if successes else None,
            "mean_action_saturation": mean(items, "any_action_saturation_fraction"),
            "mean_inference_ms": mean(items, "inference_ms_mean"),
            "p95_inference_ms": max(
                (float(row["inference_ms_p95"]) for row in items), default=0.0
            ),
        }

    return {
        "by_track_profile": {
            "|".join(key): summarize(items) for key, items in sorted(grouped.items())
        },
        "by_checkpoint": {
            key: summarize(items) for key, items in sorted(by_checkpoint.items())
        },
    }


def save_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate actor and MPPI policies across Flightmare tracks."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--actor", type=Path, nargs="+")
    parser.add_argument(
        "--mppi-config",
        type=Path,
        nargs="+",
        help="TD-MPC2-style MPPI configs to evaluate beside actor policies",
    )
    parser.add_argument(
        "--actor-manifest",
        type=Path,
        nargs="+",
        help="top-k manifests whose best ranked checkpoint should be evaluated",
    )
    parser.add_argument("--tracks", nargs="+", default=list(DEFAULT_TRACKS))
    parser.add_argument(
        "--course-suite", type=Path,
        help="immutable real-course suite; replaces --tracks with its active exact geometry",
    )
    parser.add_argument(
        "--profiles", nargs="+", choices=("clean", "recovery"),
        default=["clean", "recovery"],
    )
    parser.add_argument("--episodes-per-track", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-actions", action="store_true")
    parser.add_argument(
        "--multirate-action-blend", type=float, default=1.0,
        help="blend a multi-rate proposal toward the previously applied command",
    )
    parser.add_argument(
        "--multirate-max-action-delta", type=float, nargs=4,
        help="optional per-tick normalized CTBR slew limit for multi-rate policies",
    )
    parser.add_argument(
        "--warmup-policy", choices=("hover", "mpcc"), default="mpcc",
        help="controller used to populate the required 24-frame causal history",
    )
    parser.add_argument("--include-mpcc-baseline", action="store_true")
    parser.add_argument("--image-delay", type=float, default=0.033)
    parser.add_argument("--action-delay", type=float, default=0.011)
    parser.add_argument("--output", type=Path)
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        with preliminary.config.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        parser.set_defaults(**config.get("flight_eval", {}))
    args = parser.parse_args()
    course_suite_report = None
    if args.course_suite is not None:
        args.tracks, course_suite_report = load_active_real_course_suite(args.course_suite)
    actor_paths = [Path(path) for path in (args.actor or [])]
    for raw_manifest in args.actor_manifest or []:
        manifest = Path(raw_manifest)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        ranked = payload.get("checkpoints", [])
        if not ranked:
            parser.error(f"actor manifest contains no ranked checkpoints: {manifest}")
        actor_paths.append(manifest.parent / ranked[0]["path"])
    mppi_configs = [Path(path) for path in (args.mppi_config or [])]
    if not actor_paths and not mppi_configs:
        parser.error(
            "--actor, --actor-manifest, or --mppi-config is required "
            "(directly or via --config)"
        )
    args.actor = actor_paths
    args.actor_manifest = [Path(path) for path in (args.actor_manifest or [])]
    if args.output is not None:
        args.output = Path(args.output)
    if args.episodes_per_track < 1 or args.max_steps < 1:
        parser.error("episodes-per-track and max-steps must be positive")
    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    policies: list[Any] = [
        load_any_policy(path, args.device, args.amp) for path in args.actor
    ]
    if not 0.0 < args.multirate_action_blend <= 1.0:
        parser.error("--multirate-action-blend must be in (0,1]")
    for policy in policies:
        if isinstance(policy, MultiRateFlowPolicyBundle):
            policy.action_blend = float(args.multirate_action_blend)
            policy.max_action_delta = (
                tuple(float(value) for value in args.multirate_max_action_delta)
                if args.multirate_max_action_delta is not None else None
            )
    for config_path in mppi_configs:
        with config_path.open("r", encoding="utf-8") as stream:
            mppi_payload = yaml.safe_load(stream) or {}
        mppi = dict(mppi_payload.get("mppi", {}))
        if "actor" not in mppi or "world_model" not in mppi:
            parser.error(f"mppi config requires actor and world_model: {config_path}")
        base = load_policy(Path(mppi["actor"]), args.device, args.amp)
        policies.append(
            build_mppi_flight_policy(
                base,
                config_path=config_path,
                settings=dict(mppi.get("planner", {})),
                world_model_path=Path(mppi["world_model"]),
                action_value_path=(
                    Path(mppi["action_value"]) if mppi.get("action_value") else None
                ),
            )
        )
    for policy in policies:
        decoder = getattr(policy.actor, "decoder_type", "wam_cross_attention_mlp")
        if isinstance(policy.actor, BoundedResidualGaussianPolicy):
            decoder = "bounded_exact_gaussian_residual_three_tick_knot"
            input_width = policy.actor.base.representation.d_model
        elif isinstance(policy.actor, BoundedResidualFlowPolicy):
            decoder = "bounded_residual_flow_corrected_three_tick_knot"
            input_width = policy.actor.base.representation.d_model
        elif isinstance(policy.actor, BoundedResidualKnotPolicy):
            decoder = "bounded_residual_three_tick_control_knot"
            input_width = policy.actor.representation.d_model
        elif isinstance(policy.actor, MultiRateFlowPolicy):
            decoder = "multirate_one_step_shortcut_flow"
            input_width = policy.actor.representation.d_model
        elif isinstance(policy.actor, TokenizerFlowPolicy):
            decoder = "tokenizer_shortcut_flow_transformer"
            input_width = policy.actor.d_model
        else:
            input_width = (
                getattr(policy.actor, "backbone_dim", None)
                or (
                    policy.actor.network[0].normalized_shape[0]
                    if policy.actor.temporal is None
                    else policy.actor.input_projection[0].normalized_shape[0]
                )
            )
        print(
            f"contract_ok checkpoint={policy.name} step={policy.step} "
            f"history={policy.history} patch={policy.patch} "
            f"decoder={decoder} input_dim={input_width}",
            flush=True,
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("/workspace/outputs/evals") / f"flight-policy-eval-{stamp}.json"
    report: dict[str, Any] = {
        "created_utc": stamp,
        "settings": vars(args),
        "contracts": [policy.contract for policy in policies],
        "episodes": [],
        "aggregate": {},
    }
    if course_suite_report is not None:
        report["real_course_suite"] = course_suite_report
    for policy in policies:
        for track in args.tracks:
            for profile in args.profiles:
                for episode_index in range(args.episodes_per_track):
                    row = evaluate_episode(
                        policy,
                        track=track,
                        profile=profile,
                        episode_index=episode_index,
                        seed=args.seed,
                        max_steps=args.max_steps,
                        sample_actions=args.sample_actions,
                        image_delay=args.image_delay,
                        action_delay=args.action_delay,
                        warmup_policy=args.warmup_policy,
                    )
                    report["episodes"].append(row)
                    report["aggregate"] = aggregate_rows(report["episodes"])
                    save_report(output, report)
                    print(
                        f"checkpoint_step={policy.step} track={track} profile={profile} "
                        f"episode={episode_index} completed={int(row['completed'])} "
                        f"gates={row['gates_passed']} progress={row['progress_fraction']:.3f} "
                        f"crash={int(row['terminated'])} time={row['elapsed_seconds']:.2f}s",
                        flush=True,
                    )
    if args.include_mpcc_baseline:
        for track in args.tracks:
            for profile in args.profiles:
                for episode_index in range(args.episodes_per_track):
                    row = evaluate_episode(
                        None,
                        track=track,
                        profile=profile,
                        episode_index=episode_index,
                        seed=args.seed,
                        max_steps=args.max_steps,
                        sample_actions=False,
                        image_delay=args.image_delay,
                        action_delay=args.action_delay,
                        warmup_policy="mpcc",
                        expert_only=True,
                    )
                    report["episodes"].append(row)
                    report["aggregate"] = aggregate_rows(report["episodes"])
                    save_report(output, report)
                    print(
                        f"checkpoint_step=0 track={track} profile={profile} "
                        f"episode={episode_index} completed={int(row['completed'])} "
                        f"gates={row['gates_passed']} progress={row['progress_fraction']:.3f} "
                        f"crash={int(row['terminated'])} time={row['elapsed_seconds']:.2f}s",
                        flush=True,
                    )
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
