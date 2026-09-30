"""Rollout collection with an intentionally transparent NumPy schema."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np

from ..types import CTBRAction, ControllerCommand
from ..gate_audit import GateTrajectoryAudit, audit_gate_trajectory
from ..tracks import Track


@dataclass(frozen=True, slots=True)
class EpisodeQuality:
    accepted: bool
    reasons: tuple[str, ...]
    transitions: int
    gates_passed: int
    elapsed_time: float
    solver_failure_fraction: float
    recovery_fraction: float
    geometry_audit: GateTrajectoryAudit | None = None
    maximum_gate_transition_progress_jump: float = 0.0


def evaluate_champion_episode(
    episode: dict[str, np.ndarray],
    *,
    expected_gates: int,
    maximum_lap_time: float | None = None,
    maximum_recovery_fraction: float = 0.10,
    track: Track | None = None,
    racing_line_length: float | None = None,
    maximum_gate_transition_progress_jump: float = 0.5,
) -> EpisodeQuality:
    """Apply a strict BC/expert-data gate without discarding benign variation."""

    transitions = int(len(episode.get("action/ctbr", ())))
    gate_events = np.asarray(episode.get("transition/gate_passed", ()), dtype=np.bool_)
    gates_passed = int(np.sum(gate_events))
    solver_status = np.asarray(
        episode.get("controller/solver_status", np.zeros(transitions)), dtype=np.int32
    )
    valid = np.asarray(
        episode.get("controller/valid", np.ones(transitions)), dtype=np.bool_
    )
    mode = np.asarray(
        episode.get("controller/diagnostics/mode", np.zeros(transitions)), dtype=np.int8
    )
    collision = bool(
        np.any(episode.get("transition/unity_collision", False))
        or np.any(episode.get("transition/ground_contact", False))
    )
    times = np.asarray(episode.get("transition/time", ()), dtype=np.float64)
    observation_times = np.asarray(
        episode.get("observation/timestamp/sim", ()), dtype=np.float64
    )
    if len(times) and len(observation_times):
        elapsed_time = float(times[-1] - observation_times[0])
    else:
        elapsed_time = float(times[-1]) if len(times) else 0.0
    solver_failure_fraction = float(np.mean(solver_status != 0)) if transitions else 1.0
    recovery_fraction = float(np.mean(mode == 1)) if transitions else 1.0
    reasons: list[str] = []
    if transitions == 0:
        reasons.append("empty")
    if collision:
        reasons.append("collision")
    if len(valid) != transitions or not np.all(valid):
        reasons.append("invalid-controller-command")
    if solver_failure_fraction > 0.0:
        reasons.append("solver-failure")
    if np.any(mode == 2):
        reasons.append("infeasible-mode")
    if recovery_fraction > maximum_recovery_fraction:
        reasons.append("excess-recovery")
    if gates_passed < expected_gates:
        reasons.append("incomplete-lap")
    elif gates_passed > expected_gates:
        reasons.append("excess-gate-events")
    geometry_audit = None
    if track is not None:
        states = episode.get("observation/privileged/state")
        progress = episode.get("observation/privileged/progress")
        if states is None or progress is None or len(np.asarray(progress)) == 0:
            reasons.append("geometry-audit-unavailable")
        else:
            start_gate_index = int(round(float(np.asarray(progress)[0, 3])))
            geometry_audit = audit_gate_trajectory(
                track,
                states,
                gate_events,
                start_gate_index=start_gate_index,
                expected_passes=expected_gates,
            )
            if not geometry_audit.labels_consistent:
                reasons.append("gate-label-mismatch")
            if not geometry_audit.complete:
                reasons.append("incomplete-geometry-lap")
            if not geometry_audit.center_clear:
                reasons.append("physical-aperture-violation")
            if not geometry_audit.body_clear:
                reasons.append("vehicle-clearance-violation")
    observed_transition_jump = 0.0
    reference_progress = np.asarray(
        episode.get("controller/reference_progress", ()), dtype=np.float64
    )
    if racing_line_length is not None and len(reference_progress) == transitions:
        event_indices = np.flatnonzero(gate_events) + 1
        event_indices = event_indices[event_indices < transitions]
        if len(event_indices):
            deltas = reference_progress[event_indices] - reference_progress[event_indices - 1]
            if track is None or track.loop:
                length = float(racing_line_length)
                deltas = (deltas + 0.5 * length) % length - 0.5 * length
            observed_transition_jump = float(np.max(np.abs(deltas)))
            if observed_transition_jump > maximum_gate_transition_progress_jump:
                reasons.append("gate-transition-progress-discontinuity")
    if maximum_lap_time is not None and elapsed_time > maximum_lap_time:
        reasons.append("slow-lap")
    return EpisodeQuality(
        accepted=not reasons,
        reasons=tuple(reasons),
        transitions=transitions,
        gates_passed=gates_passed,
        elapsed_time=elapsed_time,
        solver_failure_fraction=solver_failure_fraction,
        recovery_fraction=recovery_fraction,
        geometry_audit=geometry_audit,
        maximum_gate_transition_progress_jump=observed_transition_jump,
    )


def _flatten(prefix: str, value: Any, output: dict[str, list[np.ndarray]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}/{key}" if prefix else key, child, output)
    else:
        output.setdefault(prefix, []).append(np.asarray(value))


def _storage_options(key: str, array: np.ndarray) -> dict[str, Any]:
    """Choose sequence-friendly HDF5 chunks without bloating tiny fields.

    The previous writer enabled chunking and LZF on every non-scalar array.
    Most schema-v4 metadata and short vectors are smaller than HDF5's chunk
    index/filter headers, so that policy increased file size and open latency.
    Time-major tensors still benefit from compression, but their chunks should
    match causal reads instead of relying on h5py's shape-agnostic auto chunks.
    """

    if array.ndim == 0 or array.nbytes < 16 * 1024:
        return {}
    is_mask = key.endswith("_mask") or key.endswith("/gate_mask")
    bytes_per_step = max(int(array[0].nbytes), int(array.dtype.itemsize))
    target_bytes = 512 * 1024 if is_mask else 1024 * 1024
    rows = max(1, min(int(array.shape[0]), target_bytes // bytes_per_step))
    # Eighteen policy records are the dominant access pattern.  Do not make a
    # mask chunk smaller than one causal window when the episode permits it.
    if is_mask and array.shape[0] >= 18:
        rows = max(18, rows)
    chunks = (rows, *array.shape[1:])
    options: dict[str, Any] = {
        "chunks": chunks,
        "compression": "gzip" if is_mask else "lzf",
        "shuffle": bool(array.dtype.kind in {"b", "i", "u", "f"}),
    }
    if is_mask:
        options["compression_opts"] = 4
    return options


def save_episode_hdf5(path: str | Path, episode: dict[str, np.ndarray]) -> Path:
    """Atomically save one hierarchical HDF5 episode."""

    destination = Path(path)
    if destination.suffix not in {".h5", ".hdf5"}:
        destination = destination.with_suffix(".h5")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with h5py.File(temporary, "w", libver="latest") as archive:
        for key, value in episode.items():
            array = np.asarray(value)
            parent_name, _, dataset_name = key.rpartition("/")
            parent = archive.require_group(parent_name) if parent_name else archive
            if array.dtype.kind in {"U", "O"}:
                parent.create_dataset(
                    dataset_name,
                    data=str(array.item()),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            elif array.ndim == 0:
                parent.create_dataset(dataset_name, data=array)
            else:
                parent.create_dataset(
                    dataset_name,
                    data=array,
                    **_storage_options(key, array),
                )
        archive.flush()
    temporary.replace(destination)
    return destination

class EpisodeCollector:
    """Collect aligned observations and the CTBR action that produced each transition."""

    def __init__(
        self,
        env: gym.Env,
        policy: Callable[[dict[str, Any]], Any],
        *,
        segmentation_model: Callable[[np.ndarray], np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.segmentation_model = segmentation_model
        self.metadata = dict(metadata or {})

    def _prepare_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(observation)
        if self.segmentation_model is not None:
            if "rgb" not in observation:
                raise ValueError("GateNet collection requires RGB observations")
            probability = np.asarray(self.segmentation_model(observation["rgb"]), dtype=np.float32)
            if probability.ndim != 2 or not np.all(np.isfinite(probability)):
                raise ValueError("segmentation model must return a finite HxW probability map")
            prepared["gatenet_mask"] = np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8)
        return prepared

    def collect(
        self,
        steps: int,
        *,
        reset_options: dict | None = None,
        discount: float = 0.997,
        seed: int | None = None,
        stop_after_gates: int | None = None,
    ) -> dict[str, np.ndarray]:
        if steps < 1:
            raise ValueError("steps must be positive")
        observation, reset_info = self.env.reset(seed=seed, options=reset_options)
        observation = self._prepare_observation(observation)
        values: dict[str, list[np.ndarray]] = {}
        _flatten("observation", observation, values)
        terminated = truncated = False
        gates_passed = 0
        for step in range(steps):
            raw_command = self.policy(observation)
            observation_time = float(
                observation.get("timestamp", {}).get("sim", observation.get("time", 0.0))
            )
            if isinstance(raw_command, ControllerCommand):
                controller = raw_command
            else:
                controller = ControllerCommand.from_action(
                    raw_command,
                    timestamp=observation_time,
                    reference_state=observation.get("state", np.zeros(25, np.float32)),
                    source=type(self.policy).__name__,
                )
            action = controller.action.as_array()
            values.setdefault("action/ctbr", []).append(action)
            values.setdefault("controller/source_timestamp", []).append(
                np.asarray(controller.source_timestamp, np.float64)
            )
            values.setdefault("controller/receive_timestamp", []).append(
                np.asarray(controller.receive_timestamp, np.float64)
            )
            values.setdefault("controller/reference_state", []).append(controller.reference_state)
            values.setdefault("controller/reference_action", []).append(controller.reference_action)
            values.setdefault("controller/reference_progress", []).append(
                np.asarray(controller.reference_progress, np.float32)
            )
            values.setdefault("controller/solver_status", []).append(
                np.asarray(controller.solver_status, np.int32)
            )
            values.setdefault("controller/solve_time", []).append(
                np.asarray(controller.solve_time, np.float32)
            )
            values.setdefault("controller/constraint_margin", []).append(
                np.asarray(controller.constraint_margin, np.float32)
            )
            values.setdefault("controller/valid", []).append(np.asarray(controller.valid))
            for name, diagnostic in controller.diagnostics.items():
                values.setdefault(f"controller/diagnostics/{name}", []).append(
                    np.asarray(diagnostic)
                )
            normalized_action = np.concatenate(
                ([action[0] / 15.0 - 1.0], action[1:4] / 6.0)
            ).astype(np.float32)
            normalized_action = np.clip(normalized_action, -1.0, 1.0)
            dreamer_action = np.zeros(16, dtype=np.float32)
            dreamer_action[:4] = normalized_action
            values.setdefault("action/normalized", []).append(normalized_action)
            values.setdefault("action/dreamer16", []).append(dreamer_action)
            observation, reward, terminated, truncated, info = self.env.step(action)
            observation = self._prepare_observation(observation)
            _flatten("observation", observation, values)
            values.setdefault("transition/reward", []).append(np.asarray(reward, np.float32))
            values.setdefault("transition/terminated", []).append(np.asarray(terminated))
            values.setdefault("transition/truncated", []).append(np.asarray(truncated))
            values.setdefault("transition/step", []).append(np.asarray(step, np.int64))
            values.setdefault("transition/time", []).append(
                np.asarray(info.get("time", np.nan), np.float32)
            )
            values.setdefault("action/applied_ctbr", []).append(
                np.asarray(info.get("applied_ctbr", info.get("ctbr_command", action)), np.float32)
            )
            values.setdefault("action/applied_motor_thrusts", []).append(
                np.asarray(info.get("applied_motor_thrusts", np.zeros(4)), np.float32)
            )
            values.setdefault("action/applied_motor_omega", []).append(
                np.asarray(info.get("applied_motor_omega", np.zeros(4)), np.float32)
            )
            values.setdefault("action/applied_motor_normalized", []).append(
                np.asarray(info.get("applied_motor_normalized", np.zeros(4)), np.float32)
            )
            values.setdefault("action/command_timestamp", []).append(
                np.asarray(info.get("action_command_timestamp", observation_time), np.float64)
            )
            values.setdefault("action/applied_timestamp", []).append(
                np.asarray(info.get("action_applied_timestamp", observation_time), np.float64)
            )
            values.setdefault("transition/gate_index", []).append(
                np.asarray(info.get("gate_index", -1), np.int64)
            )
            values.setdefault("transition/gate_passed", []).append(
                np.asarray(info.get("gate_passed", False))
            )
            gates_passed += int(bool(info.get("gate_passed", False)))
            values.setdefault("transition/unity_collision", []).append(
                np.asarray(info.get("unity_collision", False))
            )
            values.setdefault("transition/ground_contact", []).append(
                np.asarray(info.get("ground_contact", False))
            )
            for name, component in info.get("reward_components", {}).items():
                values.setdefault(f"reward/{name}", []).append(
                    np.asarray(component, np.float32)
                )
            if terminated or truncated or (
                stop_after_gates is not None and gates_passed >= stop_after_gates
            ):
                break
        episode = {key: np.stack(items) for key, items in values.items()}
        length = len(episode["action/ctbr"])
        is_first = np.zeros(length, dtype=np.bool_)
        is_last = np.zeros(length, dtype=np.bool_)
        is_terminal = np.zeros(length, dtype=np.bool_)
        is_first[0] = True
        is_last[-1] = True
        is_terminal[-1] = bool(terminated)
        continuation = (~is_terminal).astype(np.float32)
        episode.update(
            continue_=continuation,
            discount=continuation * np.float32(discount),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            schema_version=np.asarray(
                4 if "observation/privileged/gate_state" in episode
                else 3 if "observation/flight_plan/records" in episode else 2,
                dtype=np.int64,
            ),
            track=np.asarray(reset_info.get("track", "unknown")),
        )
        # `continue` is a Python keyword and cannot be passed as a keyword above.
        episode["continue"] = episode.pop("continue_")
        episode["reward/total"] = episode["transition/reward"]
        if "observation/task_state" in episode:
            task = np.asarray(episode["observation/task_state"], np.float32)
            episode["target/task_state_delta"] = task[1:] - task[:-1]
            episode["target/dynamics_valid"] = ~np.asarray(
                episode["transition/gate_passed"], np.bool_
            )
        episode["controller/source"] = np.asarray(
            controller.source if length else type(self.policy).__name__
        )
        environment_metadata = (
            self.env.collection_metadata() if hasattr(self.env, "collection_metadata") else {}
        )
        combined_metadata = {**environment_metadata, **self.metadata}
        combined_metadata.setdefault("seed", reset_info.get("seed", -1))
        combined_metadata.setdefault("track_fingerprint", reset_info.get("track_fingerprint", "unknown"))
        for key, value in combined_metadata.items():
            episode[f"metadata/{key}"] = np.asarray(value)
        for key, value in reset_info.get("spawn", {}).items():
            episode[f"metadata/spawn/{key}"] = np.asarray(value)
        return episode
