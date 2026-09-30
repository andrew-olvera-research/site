"""Closed-loop drone-racing curriculum, spawn, and reward contracts.

The curriculum deliberately changes one axis at a time: first gate crossing,
then spawn recovery, gate chaining, course coverage, and finally speed.  MPCC
prefix steps are part of the initial condition and are never added to the
on-policy rollout.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .env.tracks import (
    Track, forward_up_quaternion, matrix_quaternion, quaternion_matrix,
)
from .rewards import RewardResult


def _pair(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain [minimum, maximum]")
    result = (float(value[0]), float(value[1]))
    if not np.all(np.isfinite(result)) or result[0] > result[1]:
        raise ValueError(f"{name} must be finite and ordered")
    return result


@dataclass(frozen=True, slots=True)
class RacingCurriculumStage:
    name: str
    tracks: tuple[str, ...]
    target_gates: int
    approach_distance: tuple[float, float]
    lateral_offset: tuple[float, float]
    vertical_offset: tuple[float, float]
    forward_speed: tuple[float, float]
    lateral_speed: tuple[float, float] = (0.0, 0.0)
    vertical_speed: tuple[float, float] = (0.0, 0.0)
    attitude_error_degrees: float = 0.0
    body_rate: float = 0.0
    mpcc_prefix_steps: int = 0
    action_delay: float = 0.0
    max_steps: int = 360
    rollout_laps: int = 1
    target_speed: float = 5.0
    target_speed_range: tuple[float, float] | None = None
    # When present, each episode conditions on its concrete track's qualified
    # MPCC speed multiplied by a sampled scale. This makes one mixed PPO stage
    # meaningful across short 17 m/s reference courses and long 7--12 m/s
    # technical courses without exposing track identity to the actor.
    manifest_speed_scale_range: tuple[float, float] | None = None
    random_gate: bool = True
    # ``random_gate=False`` preserves the legacy ordered-gate sweep used by
    # recovery evaluations.  Canonical race starts must be requested
    # explicitly so timing/distillation cannot silently cycle every gate.
    fixed_start_gate_index: int | None = None
    allow_archived_task_resets: bool = False
    # Route-only checkpoints encode power loops, split-S arcs, and flag
    # orbits. They must be traversed but are not valid cold-start planes.
    physical_gate_starts_only: bool = False
    minimum_episodes: int = 24
    advancement_window: int = 24
    advancement_success: float = 0.75
    anchor_weight: float = 0.1
    bc_weight: float = 0.1
    trainable_scope: str = "flow_tail"

    def __post_init__(self) -> None:
        if not self.name or not self.tracks:
            raise ValueError("curriculum stages need a name and at least one track")
        if (
            self.target_gates < 1 or self.max_steps < 1 or self.rollout_laps < 1
            or self.mpcc_prefix_steps < 0
        ):
            raise ValueError(
                "target_gates/max_steps/rollout_laps must be positive and prefix non-negative"
            )
        if self.minimum_episodes < 1 or self.advancement_window < 1:
            raise ValueError("curriculum evidence windows must be positive")
        if not 0.0 <= self.advancement_success <= 1.0:
            raise ValueError("advancement_success must be in [0,1]")
        if min(self.target_speed, self.anchor_weight, self.bc_weight) < 0:
            raise ValueError("stage speed and regularization weights must be non-negative")
        if self.target_speed_range is not None:
            low, high = self.target_speed_range
            if not np.isfinite(low + high) or low <= 0 or low > high:
                raise ValueError("target_speed_range must be positive, finite, and ordered")
        if self.manifest_speed_scale_range is not None:
            low, high = self.manifest_speed_scale_range
            if not np.isfinite(low + high) or low <= 0 or low > high:
                raise ValueError(
                    "manifest_speed_scale_range must be positive, finite, and ordered"
                )
        if self.trainable_scope not in {"flow_tail", "flow", "flow_context_tail", "all"}:
            raise ValueError("unknown flow-policy trainable_scope")
        for name in (
            "approach_distance", "lateral_offset", "vertical_offset",
            "forward_speed", "lateral_speed", "vertical_speed",
        ):
            low, high = getattr(self, name)
            if not np.isfinite(low + high) or low > high:
                raise ValueError(f"invalid stage interval {name}")
        if (
            self.approach_distance[0] <= 0 or self.attitude_error_degrees < 0
            or self.body_rate < 0 or self.action_delay < 0
        ):
            raise ValueError("approach distance must be positive and perturbations non-negative")
        if self.fixed_start_gate_index is not None and self.fixed_start_gate_index < 0:
            raise ValueError("fixed_start_gate_index must be non-negative")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RacingCurriculumStage":
        values = dict(raw)
        values["tracks"] = tuple(str(item) for item in values["tracks"])
        for name in (
            "approach_distance", "lateral_offset", "vertical_offset",
            "forward_speed", "lateral_speed", "vertical_speed",
            "target_speed_range",
            "manifest_speed_scale_range",
        ):
            if name in values and values[name] is not None:
                values[name] = _pair(values[name], name)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class SpawnSample:
    state: np.ndarray
    gate_index: int
    metadata: Mapping[str, float | int | str]
    previous_action: np.ndarray | None = None
    observation_history: Mapping[str, np.ndarray] | None = None


def _error_rotation(degrees: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.radians(degrees)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], np.float32)


def sample_curriculum_spawn(
    track: Track,
    stage: RacingCurriculumStage,
    *,
    seed: int,
    episode_index: int,
) -> SpawnSample:
    """Sample a finite state behind a selected gate with auditable difficulty."""

    rng = np.random.default_rng(int(seed) + 104729 * int(episode_index))
    candidates = (
        [index for index, gate in enumerate(track.gates) if gate.render]
        if stage.physical_gate_starts_only else list(range(len(track.gates)))
    )
    if not candidates:
        raise ValueError("physical-gate-only spawn requested for a track with no rendered gates")
    if stage.fixed_start_gate_index is not None:
        gate_index = int(stage.fixed_start_gate_index)
        if gate_index not in candidates:
            raise ValueError(
                f"fixed start gate {gate_index} is not an eligible start for {track.name}"
            )
    else:
        gate_index = (
            int(candidates[int(rng.integers(len(candidates)))])
            if stage.random_gate else int(candidates[int(episode_index) % len(candidates)])
        )
    gate = track.gates[gate_index]
    approach = float(rng.uniform(*stage.approach_distance))
    lateral = float(rng.uniform(*stage.lateral_offset))
    vertical = float(rng.uniform(*stage.vertical_offset))
    forward = float(rng.uniform(*stage.forward_speed))
    side = float(rng.uniform(*stage.lateral_speed))
    climb = float(rng.uniform(*stage.vertical_speed))
    attitude = rng.uniform(
        -stage.attitude_error_degrees, stage.attitude_error_degrees, size=3
    )
    rates = rng.uniform(-stage.body_rate, stage.body_rate, size=3)

    state = np.zeros(25, np.float32)
    position = (
        gate.position - approach * gate.normal
        + lateral * gate.lateral + vertical * gate.up
    )
    margin = np.asarray([0.20, 0.20, 0.70], np.float32)
    state[0:3] = np.clip(position, track.bounds[:, 0] + margin, track.bounds[:, 1] - margin)
    # Clipping must never move a sample through its active gate plane.
    if float((state[0:3] - gate.position) @ gate.normal) >= -0.10:
        state[0:3] = gate.position - max(0.25, approach) * gate.normal
    rotation = gate.directed_rotation @ _error_rotation(attitude)
    state[3:7] = matrix_quaternion(rotation)
    state[7:10] = forward * gate.normal + side * gate.lateral + climb * gate.up
    state[10:13] = rates
    if not np.all(np.isfinite(state)):
        raise RuntimeError("curriculum spawn generated a non-finite state")
    metadata: dict[str, float | int | str] = {
        "curriculum_stage": stage.name,
        "gate_index": gate_index,
        "approach_distance": approach,
        "lateral_offset": lateral,
        "vertical_offset": vertical,
        "forward_speed": forward,
        "lateral_speed": side,
        "vertical_speed": climb,
        "attitude_error_degrees": float(np.linalg.norm(attitude)),
        "body_rate_norm": float(np.linalg.norm(rates)),
        "mpcc_prefix_steps": stage.mpcc_prefix_steps,
        "sampler": (
            "fixed-gate-curriculum-v1"
            if stage.fixed_start_gate_index is not None
            else "ordered-gate-curriculum-v1"
        ),
    }
    return SpawnSample(state, gate_index, metadata)


@dataclass(frozen=True, slots=True)
class ArchivedRaceState:
    """A dynamically realized reset state and its causal command context."""

    track: str
    gate_index: int
    state: np.ndarray
    previous_action: np.ndarray
    kind: str
    observation_history: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        state = np.asarray(self.state, np.float32)
        action = np.asarray(self.previous_action, np.float32)
        if (
            not self.track or self.gate_index < 0 or self.kind not in {
                "gate", "failure", "transition_pre", "transition_post",
            }
            or state.shape != (25,) or action.shape != (4,)
            or not np.all(np.isfinite(state)) or not np.all(np.isfinite(action))
        ):
            raise ValueError("invalid archived racing state")
        history = None
        if self.observation_history is not None:
            history = {
                str(key): np.asarray(value).copy()
                for key, value in self.observation_history.items()
            }
            legacy_contract = {
                "mask_bits", "mask_width", "proprio", "route", "timing",
                "previous_action",
            }
            current_contract = legacy_contract | {
                "deployable_task_state", "gate_index",
            }
            lengths = {
                value.shape[0] for key, value in history.items()
                if key not in {"mask_width"}
            }
            if (
                frozenset(history) not in {
                    frozenset(legacy_contract), frozenset(current_contract),
                } or len(lengths) != 1
                or not all(np.all(np.isfinite(value)) for value in history.values())
            ):
                raise ValueError("invalid archived observation history")
        object.__setattr__(self, "state", state.copy())
        object.__setattr__(self, "previous_action", action.copy())
        object.__setattr__(self, "observation_history", history)

    def state_dict(self) -> dict[str, Any]:
        return {
            "track": self.track, "gate_index": self.gate_index,
            "state": self.state.copy(), "previous_action": self.previous_action.copy(),
            "kind": self.kind,
            "observation_history": (
                None if self.observation_history is None else {
                    key: value.copy() for key, value in self.observation_history.items()
                }
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArchivedRaceState":
        return cls(
            track=str(value["track"]), gate_index=int(value["gate_index"]),
            state=np.asarray(value["state"], np.float32),
            previous_action=np.asarray(value["previous_action"], np.float32),
            kind=str(value["kind"]),
            observation_history=value.get("observation_history"),
        )


class RaceResetArchive:
    """Swift-style gate states plus recoverable pre-failure frontiers.

    The archive never resets directly into a collision. Failure entries are
    selected from a configurable lookback window and must remain safely behind
    the active gate plane. A fixed procedural fraction preserves support beyond
    states already visited by the current policy.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        raw = dict(config or {})
        self.enabled = bool(raw.get("enabled", False))
        self.capacity = int(raw.get("capacity", 4096))
        self.procedural_probability = float(raw.get("procedural_probability", 0.30))
        self.gate_probability = float(raw.get("gate_probability", 0.35))
        self.failure_probability = float(raw.get("failure_probability", 0.35))
        self.intergate_probability = float(raw.get("intergate_probability", 0.0))
        self.transition_pre_probability = float(
            raw.get("transition_pre_probability", 0.0)
        )
        self.transition_post_probability = float(
            raw.get("transition_post_probability", 0.0)
        )
        self.intergate_distance = _pair(
            raw.get("intergate_distance", [2.5, 4.5]), "intergate_distance"
        )
        self.outcome_window = int(raw.get("outcome_window", 256))
        self.minimum_source_outcomes = int(raw.get("minimum_source_outcomes", 32))
        self.target_replay_success = float(raw.get("target_replay_success", 0.50))
        self.minimum_replay_scale = float(raw.get("minimum_replay_scale", 0.20))
        self.transition_pre_unlock_success = float(
            raw.get("transition_pre_unlock_success", 0.0)
        )
        self.transition_pre_locked_scale = float(
            raw.get("transition_pre_locked_scale", 0.0)
        )
        self.failure_lookbacks = tuple(
            int(value) for value in raw.get("failure_lookback_raw_steps", [18, 36, 54, 72])
        )
        self.transition_pre_lookbacks = tuple(
            int(value) for value in raw.get("transition_pre_lookback_raw_steps", [3, 9, 18])
        )
        self.failure_distance = _pair(
            raw.get("failure_gate_distance", [0.75, 8.0]), "failure_gate_distance"
        )
        self.position_noise = np.asarray(
            raw.get("position_noise_gate_frame", [0.12, 0.10, 0.08]), np.float32
        )
        self.velocity_noise = np.asarray(
            raw.get("velocity_noise_gate_frame", [0.15, 0.12, 0.10]), np.float32
        )
        self.attitude_noise_degrees = float(raw.get("attitude_noise_degrees", 1.5))
        self.body_rate_noise = float(raw.get("body_rate_noise", 0.08))
        self.minimum_altitude = float(raw.get("minimum_altitude", 0.35))
        self.maximum_speed = float(raw.get("maximum_speed", 12.0))
        self.maximum_body_rate = float(raw.get("maximum_body_rate", 5.5))
        probabilities = np.asarray([
            self.procedural_probability, self.gate_probability,
            self.failure_probability, self.intergate_probability,
            self.transition_pre_probability, self.transition_post_probability,
        ])
        if (
            self.capacity < 1 or not self.failure_lookbacks or min(self.failure_lookbacks) < 1
            or not self.transition_pre_lookbacks or min(self.transition_pre_lookbacks) < 1
            or probabilities.min() < 0 or probabilities.sum() <= 0
            or self.outcome_window < 1 or self.minimum_source_outcomes < 1
            or not 0 < self.target_replay_success <= 1
            or not 0 <= self.minimum_replay_scale <= 1
            or not 0 <= self.transition_pre_unlock_success <= 1
            or not 0 <= self.transition_pre_locked_scale <= 1
            or self.position_noise.shape != (3,) or self.velocity_noise.shape != (3,)
            or min(
                self.attitude_noise_degrees, self.body_rate_noise, self.minimum_altitude,
                self.maximum_speed, self.maximum_body_rate,
            ) < 0
        ):
            raise ValueError("invalid reset archive configuration")
        self._entries: deque[ArchivedRaceState] = deque(maxlen=self.capacity)
        self._sample_counts: Counter[str] = Counter()
        self._outcomes: dict[str, deque[float]] = {
            "procedural": deque(maxlen=self.outcome_window),
            "gate": deque(maxlen=self.outcome_window),
            "failure": deque(maxlen=self.outcome_window),
            "intergate": deque(maxlen=self.outcome_window),
            "transition_pre": deque(maxlen=self.outcome_window),
            "transition_post": deque(maxlen=self.outcome_window),
        }

    @property
    def gate_states(self) -> int:
        return sum(entry.kind == "gate" for entry in self._entries)

    @property
    def failure_states(self) -> int:
        return sum(entry.kind == "failure" for entry in self._entries)

    @property
    def transition_pre_states(self) -> int:
        return sum(entry.kind == "transition_pre" for entry in self._entries)

    @property
    def transition_post_states(self) -> int:
        return sum(entry.kind == "transition_post" for entry in self._entries)

    @property
    def history_raw_steps(self) -> int:
        return max(max(self.failure_lookbacks), max(self.transition_pre_lookbacks)) + 1

    def _safe_state(self, track: Track, gate_index: int, state: np.ndarray) -> bool:
        if state.shape != (25,) or not np.all(np.isfinite(state)):
            return False
        gate = track.gates[int(gate_index) % len(track.gates)]
        signed = float((state[:3] - gate.position) @ gate.normal)
        distance = float(np.linalg.norm(state[:3] - gate.position))
        return bool(
            state[2] >= max(self.minimum_altitude, track.bounds[2, 0] + 0.15)
            and np.all(state[:3] >= track.bounds[:, 0] + np.asarray([0.15, 0.15, 0.15]))
            and np.all(state[:3] <= track.bounds[:, 1] - np.asarray([0.15, 0.15, 0.15]))
            and signed <= -0.10
            and self.failure_distance[0] <= distance <= self.failure_distance[1]
            and np.linalg.norm(state[7:10]) <= self.maximum_speed
            and np.linalg.norm(state[10:13]) <= self.maximum_body_rate
        )

    def record_gate_crossing(
        self, track: Track, gate_index: int, state: np.ndarray, previous_action: np.ndarray,
    ) -> bool:
        """Record the realized state just after a pass, targeting the next gate."""

        if not self.enabled:
            return False
        state = np.asarray(state, np.float32)
        # Inter-gate segments can be longer than the failure replay distance.
        gate = track.gates[int(gate_index) % len(track.gates)]
        signed = float((state[:3] - gate.position) @ gate.normal)
        safe = (
            state.shape == (25,) and np.all(np.isfinite(state))
            and state[2] >= max(self.minimum_altitude, track.bounds[2, 0] + 0.15)
            and signed <= -0.10
            and np.linalg.norm(state[7:10]) <= self.maximum_speed
            and np.linalg.norm(state[10:13]) <= self.maximum_body_rate
        )
        if not safe:
            return False
        self._entries.append(ArchivedRaceState(
            track.name, int(gate_index) % len(track.gates), state,
            np.asarray(previous_action, np.float32), "gate",
        ))
        return True

    def record_transition_crossing(
        self,
        track: Track,
        next_gate_index: int,
        trajectory: Sequence[tuple],
        state: np.ndarray,
        previous_action: np.ndarray,
        observation_history: Mapping[str, np.ndarray] | None = None,
    ) -> int:
        """Archive both sides of a realized gate-to-gate boundary.

        ``transition_post`` starts isolate departure and next-gate acquisition.
        ``transition_pre`` starts preserve the actual incoming velocity, attitude,
        body rate, and command history while giving the causal observation history
        time to fill before the crossing.  Expanding the lookback distribution is
        therefore a reverse curriculum over the complete two-gate stitch.
        """

        if not self.enabled:
            return 0
        added = 0
        next_gate_index = int(next_gate_index) % len(track.gates)
        post_state = np.asarray(state, np.float32)
        next_gate = track.gates[next_gate_index]
        post_signed = float((post_state[:3] - next_gate.position) @ next_gate.normal)
        post_safe = bool(
            post_state.shape == (25,)
            and np.all(np.isfinite(post_state))
            and post_state[2] >= max(self.minimum_altitude, track.bounds[2, 0] + 0.15)
            and post_signed <= -0.10
            and np.linalg.norm(post_state[7:10]) <= self.maximum_speed
            and np.linalg.norm(post_state[10:13]) <= self.maximum_body_rate
        )
        if post_safe:
            self._entries.append(ArchivedRaceState(
                track.name, next_gate_index, post_state,
                np.asarray(previous_action, np.float32), "transition_post",
                observation_history,
            ))
            added += 1

        crossed_gate_index = (next_gate_index - 1) % len(track.gates)
        used: set[int] = set()
        for lookback in self.transition_pre_lookbacks:
            index = max(0, len(trajectory) - 1 - lookback)
            if index in used:
                continue
            used.add(index)
            item = trajectory[index]
            gate_index, archived_state, archived_action = item[:3]
            archived_history = item[3] if len(item) > 3 else None
            gate_index = int(gate_index) % len(track.gates)
            archived_state = np.asarray(archived_state, np.float32)
            if gate_index != crossed_gate_index:
                continue
            if not self._safe_state(track, gate_index, archived_state):
                continue
            self._entries.append(ArchivedRaceState(
                track.name, gate_index, archived_state,
                np.asarray(archived_action, np.float32), "transition_pre",
                archived_history,
            ))
            added += 1
        return added

    def record_failure_frontier(
        self,
        track: Track,
        trajectory: Sequence[tuple[int, np.ndarray, np.ndarray]],
    ) -> int:
        """Archive several safe states 0.2--0.8 s before a terminal failure."""

        if not self.enabled or not trajectory:
            return 0
        added = 0
        used: set[int] = set()
        for lookback in self.failure_lookbacks:
            index = max(0, len(trajectory) - 1 - lookback)
            if index in used:
                continue
            used.add(index)
            item = trajectory[index]
            gate_index, state, previous_action = item[:3]
            observation_history = item[3] if len(item) > 3 else None
            state = np.asarray(state, np.float32)
            if not self._safe_state(track, gate_index, state):
                continue
            self._entries.append(ArchivedRaceState(
                track.name, int(gate_index) % len(track.gates), state,
                np.asarray(previous_action, np.float32), "failure",
                observation_history,
            ))
            added += 1
        return added

    def _perturb(
        self, entry: ArchivedRaceState, track: Track, rng: np.random.Generator,
    ) -> SpawnSample:
        # A restored causal history is only correct for its exact physical
        # state.  Perturbing pose/velocity while reusing that history recreates
        # the partial-observability bug this archive is designed to remove.
        if entry.observation_history is not None:
            metadata: dict[str, float | int | str] = {
                "curriculum_stage": "archive",
                "gate_index": entry.gate_index,
                "sampler": f"race-reset-archive-{entry.kind}-history-v2",
                "archive_kind": entry.kind,
                "causal_history_restored": 1,
            }
            return SpawnSample(
                entry.state.copy(), entry.gate_index, metadata,
                entry.previous_action.copy(), entry.observation_history,
            )
        state = entry.state.copy()
        gate = track.gates[entry.gate_index]
        frame = gate.directed_rotation
        state[:3] += frame @ rng.normal(0.0, self.position_noise, size=3)
        state[7:10] += frame @ rng.normal(0.0, self.velocity_noise, size=3)
        state[10:13] += rng.normal(0.0, self.body_rate_noise, size=3)
        attitude = rng.uniform(
            -self.attitude_noise_degrees, self.attitude_noise_degrees, size=3
        )
        state[3:7] = matrix_quaternion(
            quaternion_matrix(state[3:7]) @ _error_rotation(attitude)
        )
        margin = np.asarray([0.20, 0.20, max(0.20, self.minimum_altitude)], np.float32)
        state[:3] = np.clip(state[:3], track.bounds[:, 0] + margin, track.bounds[:, 1] - margin)
        signed = float((state[:3] - gate.position) @ gate.normal)
        if signed > -0.10:
            state[:3] -= (signed + 0.10) * gate.normal
        previous_action = np.clip(
            entry.previous_action + rng.normal(0.0, [0.10, 0.04, 0.04, 0.04]),
            [0.0, -6.0, -6.0, -6.0], [15.0, 6.0, 6.0, 6.0],
        ).astype(np.float32)
        metadata: dict[str, float | int | str] = {
            "curriculum_stage": "archive",
            "gate_index": entry.gate_index,
            "sampler": f"race-reset-archive-{entry.kind}-v1",
            "archive_kind": entry.kind,
            "attitude_perturbation_degrees": float(np.linalg.norm(attitude)),
        }
        return SpawnSample(state, entry.gate_index, metadata, previous_action)

    def _sample_intergate(
        self,
        track: Track,
        stage: RacingCurriculumStage,
        rng: np.random.Generator,
        episode_index: int,
    ) -> SpawnSample:
        candidates = (
            [index for index, gate in enumerate(track.gates) if gate.render]
            if stage.physical_gate_starts_only else list(range(len(track.gates)))
        )
        if not candidates:
            raise ValueError(
                "physical-gate-only spawn requested for a track with no rendered gates"
            )
        gate_index = (
            int(candidates[int(rng.integers(len(candidates)))])
            if stage.random_gate else int(candidates[int(episode_index) % len(candidates)])
        )
        gate = track.gates[gate_index]
        previous = track.gates[(gate_index - 1) % len(track.gates)]
        incoming = np.asarray(gate.position - previous.position, np.float32)
        length = float(np.linalg.norm(incoming))
        tangent = incoming / max(length, 1e-6)
        maximum = min(self.intergate_distance[1], max(0.75, length - 0.5))
        minimum = min(self.intergate_distance[0], maximum)
        remaining = float(rng.uniform(minimum, maximum))
        lateral = float(rng.uniform(*stage.lateral_offset))
        vertical = float(rng.uniform(*stage.vertical_offset))
        position = (
            gate.position - remaining * tangent
            + lateral * gate.lateral + vertical * gate.up
        )
        margin = np.asarray([0.20, 0.20, 0.70], np.float32)
        position = np.clip(position, track.bounds[:, 0] + margin, track.bounds[:, 1] - margin)
        signed = float((position - gate.position) @ gate.normal)
        if signed > -0.10:
            position -= (signed + 0.10) * gate.normal
        speed = float(rng.uniform(*stage.forward_speed))
        side = float(rng.uniform(*stage.lateral_speed))
        climb = float(rng.uniform(*stage.vertical_speed))
        attitude = rng.uniform(
            -0.5 * stage.attitude_error_degrees,
            0.5 * stage.attitude_error_degrees,
            size=3,
        )
        state = np.zeros(25, np.float32)
        state[:3] = position
        nominal = quaternion_matrix(forward_up_quaternion(tangent, gate.up))
        state[3:7] = matrix_quaternion(nominal @ _error_rotation(attitude))
        state[7:10] = speed * tangent + side * gate.lateral + climb * gate.up
        state[10:13] = rng.uniform(-0.5 * stage.body_rate, 0.5 * stage.body_rate, size=3)
        metadata: dict[str, float | int | str] = {
            "curriculum_stage": stage.name,
            "gate_index": gate_index,
            "sampler": "gate-segment-intergate-v1",
            "archive_kind": "intergate",
            "remaining_gate_distance": remaining,
        }
        return SpawnSample(
            state, gate_index, metadata,
            np.asarray([9.81, 0.0, 0.0, 0.0], np.float32),
        )

    def sample_spawn(
        self,
        track: Track,
        stage: RacingCurriculumStage,
        *, seed: int, episode_index: int,
    ) -> SpawnSample:
        if not self.enabled:
            return sample_curriculum_spawn(
                track, stage, seed=seed, episode_index=episode_index
            )
        rng = np.random.default_rng(int(seed) + 32452843 * int(episode_index))
        available = {
            "gate": [entry for entry in self._entries if entry.track == track.name and entry.kind == "gate"],
            "failure": [entry for entry in self._entries if entry.track == track.name and entry.kind == "failure"],
            "transition_pre": [
                entry for entry in self._entries
                if entry.track == track.name and entry.kind == "transition_pre"
            ],
            "transition_post": [
                entry for entry in self._entries
                if entry.track == track.name and entry.kind == "transition_post"
            ],
        }
        kinds = ["procedural"]
        weights = [self.procedural_probability]
        if self.intergate_probability > 0:
            kinds.append("intergate"); weights.append(
                self.intergate_probability * self._replay_scale("intergate")
            )
        if available["gate"]:
            kinds.append("gate"); weights.append(
                self.gate_probability * self._replay_scale("gate")
            )
        if available["failure"]:
            kinds.append("failure"); weights.append(
                self.failure_probability * self._replay_scale("failure")
            )
        if available["transition_pre"] and self.transition_pre_probability > 0:
            kinds.append("transition_pre"); weights.append(
                self.transition_pre_probability * self._replay_scale("transition_pre")
            )
        if available["transition_post"] and self.transition_post_probability > 0:
            kinds.append("transition_post"); weights.append(
                self.transition_post_probability * self._replay_scale("transition_post")
            )
        weights_array = np.asarray(weights, np.float64)
        weights_array /= weights_array.sum()
        kind = str(rng.choice(kinds, p=weights_array))
        self._sample_counts[kind] += 1
        if kind == "procedural":
            return sample_curriculum_spawn(
                track, stage, seed=seed, episode_index=episode_index
            )
        if kind == "intergate":
            return self._sample_intergate(track, stage, rng, episode_index)
        entries = available[kind]
        entry = entries[int(rng.integers(len(entries)))]
        return self._perturb(entry, track, rng)

    def _replay_scale(self, kind: str) -> float:
        if kind == "transition_pre" and self.transition_pre_unlock_success > 0:
            post = self._outcomes["transition_post"]
            if (
                len(post) < self.minimum_source_outcomes
                or float(np.mean(post)) < self.transition_pre_unlock_success
            ):
                return self.transition_pre_locked_scale
        outcomes = self._outcomes[kind]
        if len(outcomes) < self.minimum_source_outcomes:
            return 1.0
        success = float(np.mean(outcomes))
        return float(np.clip(
            success / self.target_replay_success,
            self.minimum_replay_scale, 1.0,
        ))

    def observe_outcomes(self, outcomes: Sequence[tuple[str, bool]]) -> None:
        for source, success in outcomes:
            if source not in self._outcomes:
                raise ValueError(f"unknown reset source {source!r}")
            self._outcomes[source].append(float(bool(success)))

    def metrics(self) -> dict[str, float]:
        total = sum(self._sample_counts.values())
        result = {
            "reset_archive_gate_states": float(self.gate_states),
            "reset_archive_failure_states": float(self.failure_states),
            "reset_archive_transition_pre_states": float(self.transition_pre_states),
            "reset_archive_transition_post_states": float(self.transition_post_states),
            **{
                f"reset_sample_{kind}_fraction": count / max(total, 1)
                for kind, count in self._sample_counts.items()
            },
        }
        for kind, outcomes in self._outcomes.items():
            result[f"reset_{kind}_recent_success"] = (
                float(np.mean(outcomes)) if outcomes else 0.0
            )
            result[f"reset_{kind}_replay_scale"] = (
                1.0 if kind == "procedural" else self._replay_scale(kind)
            )
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.state_dict() for entry in self._entries],
            "sample_counts": dict(self._sample_counts),
            "outcomes": {key: list(value) for key, value in self._outcomes.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        self._entries.clear()
        for raw in state.get("entries", ()):
            self._entries.append(ArchivedRaceState.from_mapping(raw))
        self._sample_counts = Counter({
            str(key): int(value) for key, value in state.get("sample_counts", {}).items()
        })
        for key, values in state.get("outcomes", {}).items():
            if key in self._outcomes:
                self._outcomes[key].extend(float(value) for value in values)


class GateChainCurriculum:
    """Competence-weighted mixture of one- and multi-gate objectives.

    Chaining evidence is present from the first update, but a currently
    impossible horizon cannot consume the whole batch. Success rates are kept
    separately per requested gate count and down-weight horizons whose recent
    gate-completion fraction is below the configured learning frontier. Partial
    progress on a chain therefore remains evidence, unlike a binary lap label.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        raw = dict(config or {})
        self.enabled = bool(raw.get("enabled", False))
        probability_raw = raw.get("probabilities", {1: 1.0})
        self.probabilities = {
            int(gates): float(probability)
            for gates, probability in probability_raw.items()
        }
        self.outcome_window = int(raw.get("outcome_window", 256))
        self.minimum_outcomes = int(raw.get("minimum_outcomes", 32))
        self.target_success = float(raw.get("target_success", 0.50))
        self.minimum_scale = float(raw.get("minimum_scale", 0.15))
        self.extra_gate_steps = int(raw.get("extra_gate_steps", 220))
        if (
            not self.probabilities
            or min(self.probabilities) < 1
            or min(self.probabilities.values()) < 0
            or sum(self.probabilities.values()) <= 0
            or self.outcome_window < 1
            or self.minimum_outcomes < 1
            or not 0 < self.target_success <= 1
            or not 0 <= self.minimum_scale <= 1
            or self.extra_gate_steps < 0
        ):
            raise ValueError("invalid gate-chain curriculum configuration")
        self._outcomes = {
            gates: deque(maxlen=self.outcome_window) for gates in self.probabilities
        }
        self._sample_counts: Counter[int] = Counter()

    def _scale(self, gates: int) -> float:
        outcomes = self._outcomes[gates]
        if gates == min(self.probabilities) or len(outcomes) < self.minimum_outcomes:
            return 1.0
        success = float(np.mean(outcomes))
        return float(np.clip(
            success / self.target_success, self.minimum_scale, 1.0
        ))

    def sample(self, maximum_gates: int, *, seed: int, episode_index: int) -> int:
        if not self.enabled:
            return 1
        candidates = [gates for gates in self.probabilities if gates <= maximum_gates]
        if not candidates:
            return 1
        weights = np.asarray([
            self.probabilities[gates] * self._scale(gates) for gates in candidates
        ], np.float64)
        weights /= weights.sum()
        rng = np.random.default_rng(seed + 32452843 * (episode_index + 1))
        selected = int(rng.choice(candidates, p=weights))
        self._sample_counts[selected] += 1
        return selected

    def max_steps(self, base_steps: int, target_gates: int) -> int:
        return int(base_steps + max(0, target_gates - 1) * self.extra_gate_steps)

    def observe(self, outcomes: Sequence[tuple[int, float | bool]]) -> None:
        for gates, completion in outcomes:
            if gates in self._outcomes:
                self._outcomes[gates].append(float(np.clip(completion, 0.0, 1.0)))

    def metrics(self) -> dict[str, float]:
        total = max(sum(self._sample_counts.values()), 1)
        metrics: dict[str, float] = {}
        for gates in sorted(self.probabilities):
            outcomes = self._outcomes[gates]
            metrics[f"chain_{gates}_sample_fraction"] = (
                self._sample_counts[gates] / total
            )
            metrics[f"chain_{gates}_competence"] = (
                float(np.mean(outcomes)) if outcomes else 0.0
            )
            metrics[f"chain_{gates}_scale"] = self._scale(gates)
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_counts": dict(self._sample_counts),
            "outcomes": {
                str(gates): list(values) for gates, values in self._outcomes.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        self._sample_counts = Counter({
            int(gates): int(count)
            for gates, count in state.get("sample_counts", {}).items()
        })
        for gates, values in state.get("outcomes", {}).items():
            gate_count = int(gates)
            if gate_count in self._outcomes:
                self._outcomes[gate_count].extend(float(value) for value in values)


@dataclass(frozen=True, slots=True)
class CurriculumRewardConfig:
    progress_weight: float = 3.0
    progress_clip_m: float = 0.35
    center_potential_weight: float = 0.20
    heading_potential_weight: float = 0.10
    potential_discount: float = 0.999
    speed_progress_weight: float = 0.25
    gate_bonus: float = 12.0
    gate_quality_bonus: float = 8.0
    gate_speed_bonus_weight: float = 0.0
    gate_speed_cap_ratio: float = 2.0
    speed_tracking_weight: float = 0.0
    speed_tracking_overspeed_weight: float = 0.0
    speed_tracking_quality_floor: float = 0.25
    control_dt: float = 1.0 / 90.0
    crash_penalty: float = 20.0
    time_penalty: float = 0.005
    body_rate_weight: float = 0.01
    action_smoothness_weight: float = 0.005
    body_rate_scale: float = 6.0
    target_speed: float = 5.0

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("curriculum reward values must be finite and non-negative")
        if (
            self.progress_clip_m <= 0 or self.body_rate_scale <= 0
            or self.target_speed <= 0 or self.gate_speed_cap_ratio <= 0
            or self.control_dt <= 0
        ):
            raise ValueError("reward normalization scales must be positive")
        if not 0.0 <= self.speed_tracking_quality_floor < 1.0:
            raise ValueError("speed_tracking_quality_floor must be in [0,1)")
        if not 0 < self.potential_discount <= 1:
            raise ValueError("potential_discount must be in (0,1]")


class CurriculumRaceReward:
    """Potential-shaped racing reward with bounded auxiliary terms.

    Course progress is the dominant dense signal.  Centering and heading are
    discounted potential differences, so hovering cannot accumulate positive
    shaping reward.  Speed is rewarded only when the vehicle makes forward
    course progress.
    """

    def __init__(self, config: CurriculumRewardConfig | None = None) -> None:
        self.config = config or CurriculumRewardConfig()
        self._progress: float | None = None
        self._center: float | None = None
        self._heading: float | None = None
        self._gate_key: tuple[float, ...] | None = None
        self._previous_action = np.asarray([9.81, 0.0, 0.0, 0.0], np.float32)

    @staticmethod
    def _potentials(state: np.ndarray, gate: Any) -> tuple[float, float, float, float]:
        delta = state[0:3] - np.asarray(gate.position, np.float32)
        half_width = max(float(gate.size[0]) * 0.5, 1e-3)
        half_height = max(float(gate.size[1]) * 0.5, 1e-3)
        lateral = float(delta @ gate.lateral) / half_width
        vertical = float(delta @ gate.up) / half_height
        center = float(np.exp(-0.5 * (lateral * lateral + vertical * vertical)))
        # Body +X is the camera/vehicle forward axis.
        from .env.tracks import quaternion_matrix
        body_forward = quaternion_matrix(state[3:7])[:, 0]
        heading = 0.5 * (1.0 + float(np.clip(body_forward @ gate.normal, -1.0, 1.0)))
        forward_speed = float(state[7:10] @ gate.normal)
        quality = center
        return center, heading, forward_speed, quality

    @staticmethod
    def _key(gate: Any) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(gate.position, np.float64), 5))

    def reset(
        self, position_world: np.ndarray, active_gate: Any, *, course_progress: float | None = None
    ) -> None:
        state = np.zeros(25, np.float32)
        state[0:3] = np.asarray(position_world, np.float32)
        state[3] = 1.0
        center, heading, _, _ = self._potentials(state, active_gate)
        self._progress = None if course_progress is None else float(course_progress)
        self._center, self._heading = center, None
        self._gate_key = self._key(active_gate)
        self._previous_action[:] = [9.81, 0.0, 0.0, 0.0]

    def __call__(
        self,
        *,
        state: np.ndarray,
        action: Any,
        active_gate: Any,
        gate_passed: bool,
        crashed: bool,
        course_progress: float | None = None,
    ) -> RewardResult:
        state = np.asarray(state, np.float32)
        if state.shape != (25,) or not np.all(np.isfinite(state)):
            raise ValueError("curriculum reward requires a finite Flightmare state")
        action_array = np.asarray(
            action.as_array() if hasattr(action, "as_array") else action, np.float32
        )
        if action_array.shape != (4,) or not np.all(np.isfinite(action_array)):
            raise ValueError("curriculum reward requires a finite CTBR action")
        current_progress = float(course_progress if course_progress is not None else 0.0)
        previous_progress = current_progress if self._progress is None else self._progress
        progress_delta = float(np.clip(
            current_progress - previous_progress,
            -self.config.progress_clip_m,
            self.config.progress_clip_m,
        ))
        center, heading, forward_speed, gate_quality = self._potentials(state, active_gate)
        same_gate = self._gate_key == self._key(active_gate)
        previous_center = center if self._center is None or not same_gate else self._center
        previous_heading = heading if self._heading is None or not same_gate else self._heading
        cfg = self.config
        center_shaping = cfg.center_potential_weight * (
            cfg.potential_discount * center - previous_center
        )
        heading_shaping = cfg.heading_potential_weight * (
            cfg.potential_discount * heading - previous_heading
        )
        speed_progress = (
            cfg.speed_progress_weight * max(progress_delta, 0.0)
            * float(np.clip(forward_speed / cfg.target_speed, 0.0, 2.0))
        )
        rate_penalty = -cfg.body_rate_weight * float(np.clip(
            np.mean((state[10:13] / cfg.body_rate_scale) ** 2), 0.0, 1.0
        ))
        scale = np.asarray([15.0, 6.0, 6.0, 6.0], np.float32)
        smoothness = -cfg.action_smoothness_weight * float(np.clip(
            np.mean(((action_array - self._previous_action) / scale) ** 2), 0.0, 1.0
        ))
        gate_speed_ratio = float(np.clip(
            forward_speed / cfg.target_speed, 0.0, cfg.gate_speed_cap_ratio
        ))
        gate_speed = (
            cfg.gate_speed_bonus_weight * gate_speed_ratio if gate_passed else 0.0
        )
        # Track a commanded pace only where the vehicle is aligned with the
        # active route. This provides a persistent speed gradient on clean
        # segments without rewarding tangential velocity or fighting recovery.
        tracking_quality = center * heading
        tracking_gate = float(np.clip(
            (tracking_quality - cfg.speed_tracking_quality_floor)
            / (1.0 - cfg.speed_tracking_quality_floor),
            0.0, 1.0,
        ))
        signed_speed_error = forward_speed / cfg.target_speed - 1.0
        underspeed_error = max(-signed_speed_error, 0.0)
        overspeed_error = max(signed_speed_error, 0.0)
        speed_tracking = -cfg.speed_tracking_weight * cfg.control_dt * tracking_gate * (
            underspeed_error * underspeed_error
            + cfg.speed_tracking_overspeed_weight * overspeed_error * overspeed_error
        )
        gate = (
            cfg.gate_bonus + cfg.gate_quality_bonus * gate_quality + gate_speed
            if gate_passed else 0.0
        )
        crash = -cfg.crash_penalty if crashed else 0.0
        components = {
            "progress": cfg.progress_weight * progress_delta,
            "progress_delta": progress_delta,
            "center_potential": center_shaping,
            "center_quality": center,
            "heading_potential": heading_shaping,
            "heading_quality": heading,
            "speed_progress": speed_progress,
            "forward_speed": forward_speed,
            "gate_pass": gate,
            "gate_speed": gate_speed,
            "gate_speed_ratio": gate_speed_ratio,
            "speed_tracking": speed_tracking,
            "speed_tracking_gate": tracking_gate,
            "speed_error_ratio": signed_speed_error,
            "gate_quality": gate_quality,
            "body_rate": rate_penalty,
            "smoothness": smoothness,
            "time": -cfg.time_penalty,
            "crash": crash,
            "course_progress": current_progress,
        }
        total = sum(components[name] for name in (
            "progress", "center_potential", "heading_potential", "speed_progress",
            "speed_tracking", "gate_pass", "body_rate", "smoothness", "time", "crash",
        ))
        self._progress = current_progress
        self._center, self._heading = center, heading
        self._gate_key = None if gate_passed else self._key(active_gate)
        self._previous_action = action_array.copy()
        return RewardResult(float(total), components)


@dataclass(frozen=True, slots=True)
class Green2026StateRewardConfig:
    """State-policy reward from Green et al. (2026), Appendix A.1."""

    progress_weight: float = 1.0
    gate_pass_weight: float = 1.0
    crash_weight: float = -4.0
    body_rate_weight: float = -0.001
    thrust_difference_weight: float = -0.0001
    xy_rate_difference_weight: float = -0.0002
    yaw_rate_difference_weight: float = -0.0002
    linear_difference_factor: float = 0.01
    lowpass_cutoff_hz: float = 6.0
    control_dt: float = 1.0 / 90.0
    maximum_collective_thrust: float = 30.0
    maximum_body_rate: float = 6.0
    # Legacy defaults preserve historical experiments. New PPO runs can use
    # exact same-target displacement without gate-switch potential impulses.
    progress_mode: str = "ordered_potential"
    gate_error_mode: str = "endpoint"
    time_penalty_per_second: float = 0.0
    # Optional pace shaping used by controlled PPO ablations. Both terms are
    # zero by default, preserving the paper-faithful Green reward. The progress
    # term rewards speed only while making ordered progress; the tracking term
    # penalizes underspeed only while the vehicle is aligned and centered.
    target_speed: float = 16.5
    speed_progress_weight: float = 0.0
    speed_tracking_weight: float = 0.0
    speed_tracking_overspeed_weight: float = 0.0
    speed_tracking_quality_floor: float = 0.25

    def __post_init__(self) -> None:
        values = np.asarray([v for v in asdict(self).values()
                             if not isinstance(v, str)], np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Green-2026 reward values must be finite")
        if (self.progress_mode not in {"ordered_potential", "same_gate_distance"}
                or self.gate_error_mode not in {"endpoint", "intersection"}
                or self.time_penalty_per_second < 0
                or self.target_speed <= 0
                or self.speed_progress_weight < 0
                or self.speed_tracking_weight < 0
                or self.speed_tracking_overspeed_weight < 0
                or not 0.0 <= self.speed_tracking_quality_floor < 1.0):
            raise ValueError("invalid Green reward progress/crossing/time contract")
        if (
            self.progress_weight < 0.0 or self.gate_pass_weight < 0.0
            or self.crash_weight > 0.0 or self.body_rate_weight > 0.0
            or self.thrust_difference_weight > 0.0
            or self.xy_rate_difference_weight > 0.0
            or self.yaw_rate_difference_weight > 0.0
            or min(
                self.linear_difference_factor, self.lowpass_cutoff_hz,
                self.control_dt, self.maximum_collective_thrust,
                self.maximum_body_rate,
            ) <= 0.0
        ):
            raise ValueError("invalid Green-2026 state reward signs or scales")


class Green2026StateRaceReward:
    """Paper-faithful state reward with a 6 Hz command low-pass target."""

    def __init__(self, config: Green2026StateRewardConfig | None = None) -> None:
        self.config = config or Green2026StateRewardConfig()
        self._distance: float | None = None
        self._course_progress: float | None = None
        self._filtered_action = np.asarray([9.81, 0.0, 0.0, 0.0], np.float32)
        self._previous_position: np.ndarray | None = None

    def reset(
        self, position_world: np.ndarray, active_gate: Any,
        *, course_progress: float | None = None,
    ) -> None:
        position = np.asarray(position_world, np.float32)
        self._previous_position = position.copy()
        self._distance = float(np.linalg.norm(position - active_gate.position))
        self._course_progress = (
            None if course_progress is None else float(course_progress)
        )
        self._filtered_action[:] = [9.81, 0.0, 0.0, 0.0]

    def __call__(
        self,
        *,
        state: np.ndarray,
        action: Any,
        active_gate: Any,
        gate_passed: bool,
        crashed: bool,
        course_progress: float | None = None,
    ) -> RewardResult:
        state = np.asarray(state, np.float32)
        command = np.asarray(
            action.as_array() if hasattr(action, "as_array") else action,
            np.float32,
        )
        if (
            state.shape != (25,) or command.shape != (4,)
            or not np.all(np.isfinite(state)) or not np.all(np.isfinite(command))
        ):
            raise ValueError("Green-2026 reward requires finite Flightmare state/CTBR")
        cfg = self.config
        distance = float(np.linalg.norm(state[:3] - active_gate.position))
        previous_distance = distance if self._distance is None else self._distance
        # Raw active-gate distance is discontinuous when the tracker advances:
        # the first step after a crossing used to compare near-zero distance to
        # the old gate with the full distance to the new gate.  On a racing
        # course this injected a large negative TD impulse at every successful
        # transition and cancelled almost all dense progress.  Flightmare
        # supplies an unwrapped, centreline-aware course potential specifically
        # to keep this transition continuous.  Preserve the distance fallback
        # for standalone users which do not provide that potential.
        if cfg.progress_mode == "same_gate_distance":
            previous_position = (state[:3] if self._previous_position is None
                                 else self._previous_position)
            progress_delta = float(np.linalg.norm(
                previous_position - active_gate.position)) - distance
            current_progress = None
        elif course_progress is not None:
            current_progress = float(course_progress)
            previous_progress = (
                current_progress
                if self._course_progress is None
                else self._course_progress
            )
            progress_delta = current_progress - previous_progress
        else:
            current_progress = None
            progress_delta = previous_distance - distance
        progress = cfg.progress_weight * progress_delta

        forward_speed = float(state[7:10] @ active_gate.normal)
        speed_ratio = float(np.clip(
            forward_speed / cfg.target_speed, 0.0, 2.0
        ))
        speed_progress = (
            cfg.speed_progress_weight * max(progress_delta, 0.0) * speed_ratio
        )

        crossing_position = state[:3]
        if (gate_passed and cfg.gate_error_mode == "intersection"
                and self._previous_position is not None):
            normal = active_gate.directed_rotation[:, 0]
            previous_plane = float((self._previous_position - active_gate.position) @ normal)
            current_plane = float((state[:3] - active_gate.position) @ normal)
            denominator = current_plane - previous_plane
            if denominator <= 0 or not previous_plane <= 0 <= current_plane:
                raise ValueError("gate_passed without a directed segment-plane crossing")
            fraction = -previous_plane / denominator
            crossing_position = self._previous_position + fraction * (
                state[:3] - self._previous_position)
        delta = crossing_position - active_gate.position
        traversal_error = float(np.hypot(
            delta @ active_gate.lateral, delta @ active_gate.up
        ))
        half_width = max(0.5 * float(active_gate.size[0]), 1.0e-6)
        half_height = max(0.5 * float(active_gate.size[1]), 1.0e-6)
        lateral_error = float(delta @ active_gate.lateral) / half_width
        vertical_error = float(delta @ active_gate.up) / half_height
        center_quality = float(np.exp(-0.5 * (
            lateral_error * lateral_error + vertical_error * vertical_error
        )))
        from .env.tracks import quaternion_matrix
        body_forward = quaternion_matrix(state[3:7])[:, 0]
        heading_quality = 0.5 * (
            1.0 + float(np.clip(body_forward @ active_gate.normal, -1.0, 1.0))
        )
        tracking_quality = center_quality * heading_quality
        tracking_gate = float(np.clip(
            (tracking_quality - cfg.speed_tracking_quality_floor)
            / (1.0 - cfg.speed_tracking_quality_floor),
            0.0, 1.0,
        ))
        signed_speed_error = forward_speed / cfg.target_speed - 1.0
        underspeed_error = max(-signed_speed_error, 0.0)
        overspeed_error = max(signed_speed_error, 0.0)
        speed_tracking = (
            -cfg.speed_tracking_weight * cfg.control_dt * tracking_gate * (
                underspeed_error * underspeed_error
                + cfg.speed_tracking_overspeed_weight
                * overspeed_error * overspeed_error
            )
        )
        gate = (
            cfg.gate_pass_weight * (1.0 - traversal_error / half_width)
            if gate_passed else 0.0
        )
        crash = cfg.crash_weight if crashed else 0.0
        rate = cfg.body_rate_weight * float(
            np.sum((state[10:12] / cfg.maximum_body_rate) ** 2)
        )

        alpha = 1.0 - np.exp(
            -2.0 * np.pi * cfg.lowpass_cutoff_hz * cfg.control_dt
        )
        self._filtered_action += np.float32(alpha) * (
            command - self._filtered_action
        )
        maximum = np.asarray([
            cfg.maximum_collective_thrust,
            cfg.maximum_body_rate,
            cfg.maximum_body_rate,
            cfg.maximum_body_rate,
        ], np.float32)
        difference = (command - self._filtered_action) / maximum
        thrust_magnitude = abs(float(difference[0]))
        xy_magnitude = float(np.linalg.norm(difference[1:3]))
        yaw_magnitude = abs(float(difference[3]))

        def penalize(weight: float, magnitude: float) -> float:
            return weight * (
                cfg.linear_difference_factor * magnitude + magnitude * magnitude
            )

        thrust_smoothness = penalize(
            cfg.thrust_difference_weight, thrust_magnitude
        )
        xy_smoothness = penalize(
            cfg.xy_rate_difference_weight, xy_magnitude
        )
        yaw_smoothness = penalize(
            cfg.yaw_rate_difference_weight, yaw_magnitude
        )
        components = {
            "progress": progress,
            "progress_delta": progress_delta,
            "speed_progress": speed_progress,
            "speed_tracking": speed_tracking,
            "speed_tracking_gate": tracking_gate,
            "speed_error_ratio": signed_speed_error,
            "gate_pass": gate,
            "gate_error": traversal_error,
            "crash": crash,
            "body_rate": rate,
            "thrust_smoothness": thrust_smoothness,
            "xy_rate_smoothness": xy_smoothness,
            "yaw_rate_smoothness": yaw_smoothness,
            "gate_distance": distance,
            "time": -cfg.time_penalty_per_second * cfg.control_dt,
            # Physical diagnostic, not another reward term. The PPO collector
            # uses this for gate-crossing speeds even with the Green reward.
            "forward_speed": forward_speed,
        }
        total = sum(components[name] for name in (
            "progress", "speed_progress", "speed_tracking", "gate_pass", "crash", "body_rate",
            "thrust_smoothness", "xy_rate_smoothness", "yaw_rate_smoothness", "time",
        ))
        self._distance = distance
        self._course_progress = current_progress
        self._previous_position = state[:3].copy()
        return RewardResult(float(total), components)


class RacingCurriculum:
    """Monotonic, evidence-gated stage scheduler."""

    def __init__(self, stages: Sequence[RacingCurriculumStage], stage_index: int = 0) -> None:
        if not stages or not 0 <= stage_index < len(stages):
            raise ValueError("curriculum needs stages and a valid stage index")
        self.stages = tuple(stages)
        self.stage_index = int(stage_index)
        self.stage_episodes = 0
        self._successes: deque[float] = deque(maxlen=self.current.advancement_window)

    @property
    def current(self) -> RacingCurriculumStage:
        return self.stages[self.stage_index]

    @property
    def success_rate(self) -> float:
        return float(np.mean(self._successes)) if self._successes else 0.0

    def observe(self, successes: Sequence[bool]) -> bool:
        for success in successes:
            self._successes.append(float(bool(success)))
            self.stage_episodes += 1
        stage = self.current
        ready = (
            self.stage_index + 1 < len(self.stages)
            and self.stage_episodes >= stage.minimum_episodes
            and len(self._successes) >= stage.advancement_window
            and self.success_rate >= stage.advancement_success
        )
        if ready:
            self.stage_index += 1
            self.stage_episodes = 0
            self._successes = deque(maxlen=self.current.advancement_window)
        return ready

    def state_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "stage_episodes": self.stage_episodes,
            "successes": list(self._successes),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        index = int(state["stage_index"])
        if not 0 <= index < len(self.stages):
            raise ValueError("saved curriculum stage is incompatible")
        self.stage_index = index
        self.stage_episodes = int(state.get("stage_episodes", 0))
        self._successes = deque(
            (float(value) for value in state.get("successes", ())),
            maxlen=self.current.advancement_window,
        )
