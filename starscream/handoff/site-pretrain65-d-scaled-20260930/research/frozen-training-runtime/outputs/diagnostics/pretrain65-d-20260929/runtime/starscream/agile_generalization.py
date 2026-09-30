"""Compatibility primitives for Green et al. (2026) agile-flight training.

The paper's state actor observes one Markov transition descriptor

    [R_WB[:, :2], v_W, omega_B, a_prev, delta_p_1, delta_p_2]

where both gate tensors are expressed in the world frame.  ``delta_p_1`` is
the four corners of the next gate relative to the vehicle and ``delta_p_2``
is the corresponding-corner displacement from the next gate to the gate after
it.  Starscream may place a causal history encoder over these descriptors, but
the descriptor itself intentionally stays paper-compatible and 40 dimensional.

This module also contains the paper's Spearman flatness controller.  Keeping
the statistical primitive independent from PPO/DAgger makes its math directly
testable and prevents either trainer from silently changing the definition.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .env.tracks import quaternion_matrix
from .world_privileged import WORLD_OBSERVATION_CONTRACT, WORLD_OBSERVATION_CONTRACTS
from .plant_privileged import PLANT_OBSERVATION_CONTRACT, PLANT_SETTINGS_DIM


LEGACY_OBSERVATION_CONTRACT = "starscream_route_v1"
GREEN2026_OBSERVATION_CONTRACT = "green2026_markov_world_corners_v1"
GREEN2026_ROUTE_CHAIN_OBSERVATION_CONTRACT = (
    "green2026_markov_world_corner_chain_v2"
)
GREEN2026_FEATURE_DIM = 40
GREEN2026_GATE_COUNT = 2
GREEN2026_DYNAMICS_DIM = 16

# Semantic slices in the exact 40-D descriptor.
GREEN2026_ROTATION = slice(0, 6)
GREEN2026_LINEAR_VELOCITY = slice(6, 9)
GREEN2026_BODY_RATES = slice(9, 12)
GREEN2026_PREVIOUS_ACTION = slice(12, 16)
GREEN2026_NEXT_GATE_CORNERS = slice(16, 28)
GREEN2026_GATE_TO_GATE_CORNERS = slice(28, 40)


def observation_contract_feature_dim(
    contract: str, *, legacy_feature_dim: int,
    route_gates: int = GREEN2026_GATE_COUNT,
) -> int:
    """Return the actor feature width without weakening legacy validation."""

    contract = str(contract)
    if contract == PLANT_OBSERVATION_CONTRACT:
        return int(legacy_feature_dim) + PLANT_SETTINGS_DIM
    if contract in {LEGACY_OBSERVATION_CONTRACT, *WORLD_OBSERVATION_CONTRACTS}:
        return int(legacy_feature_dim)
    if contract == GREEN2026_OBSERVATION_CONTRACT:
        return GREEN2026_FEATURE_DIM
    if contract == GREEN2026_ROUTE_CHAIN_OBSERVATION_CONTRACT:
        route_gates = int(route_gates)
        if route_gates < 1:
            raise ValueError("route-chain observation needs at least one gate")
        # Dynamics/previous action plus one 12-D corner tensor per route gate:
        # absolute relative corners for gate zero, then corresponding-corner
        # displacements for every later gate.
        return GREEN2026_DYNAMICS_DIM + 12 * route_gates
    raise ValueError(f"unknown privileged observation contract {contract!r}")


def observation_contract_action_slice(contract: str, *, legacy_start: int) -> slice:
    if str(contract) in {LEGACY_OBSERVATION_CONTRACT, PLANT_OBSERVATION_CONTRACT, *WORLD_OBSERVATION_CONTRACTS}:
        return slice(int(legacy_start), int(legacy_start) + 4)
    if str(contract) in {
        GREEN2026_OBSERVATION_CONTRACT,
        GREEN2026_ROUTE_CHAIN_OBSERVATION_CONTRACT,
    }:
        return GREEN2026_PREVIOUS_ACTION
    raise ValueError(f"unknown privileged observation contract {contract!r}")


def _gate_corners(
    position: np.ndarray,
    normal: np.ndarray,
    up: np.ndarray,
    size: np.ndarray,
) -> np.ndarray:
    """Construct four consistently ordered corners in the supplied frame."""

    position = np.asarray(position, np.float32)
    normal = np.asarray(normal, np.float32)
    up = np.asarray(up, np.float32)
    size = np.asarray(size, np.float32)
    lateral = np.cross(normal, up)
    lateral /= max(float(np.linalg.norm(lateral)), 1.0e-8)
    up = up / max(float(np.linalg.norm(up)), 1.0e-8)
    half_width, half_height = 0.5 * size
    return np.stack([
        position + sy * half_width * lateral + sz * half_height * up
        for sy, sz in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    ]).astype(np.float32)


def green2026_features_from_components(
    *,
    state: np.ndarray,
    gate_positions_body: np.ndarray,
    gate_normals_body: np.ndarray,
    gate_up_body: np.ndarray,
    gate_sizes: np.ndarray,
    previous_action_normalized: np.ndarray,
) -> np.ndarray:
    """Build the exact state actor descriptor from simulator components.

    Flightmare already provides gate geometry in the vehicle body frame.  The
    paper defines both corner deltas in the world frame, so the relative gate
    geometry is rotated by ``R_WB`` before it is flattened.
    """

    state = np.asarray(state, np.float32)
    positions = np.asarray(gate_positions_body, np.float32)
    normals = np.asarray(gate_normals_body, np.float32)
    up = np.asarray(gate_up_body, np.float32)
    sizes = np.asarray(gate_sizes, np.float32)
    previous = np.asarray(previous_action_normalized, np.float32)
    if (
        state.ndim != 1 or state.shape[0] < 13
        or positions.shape != (GREEN2026_GATE_COUNT, 3)
        or normals.shape != positions.shape or up.shape != positions.shape
        or sizes.shape != (GREEN2026_GATE_COUNT, 2)
        or previous.shape != (4,)
    ):
        raise ValueError("incompatible components for Green-2026 observation")
    world_from_body = quaternion_matrix(state[3:7]).astype(np.float32)
    rotation_6d = world_from_body[:, :2].T.reshape(-1)
    corners_body = np.stack([
        _gate_corners(positions[index], normals[index], up[index], sizes[index])
        for index in range(GREEN2026_GATE_COUNT)
    ])
    corners_world_relative = np.einsum(
        "ij,gkj->gki", world_from_body, corners_body
    ).astype(np.float32)
    delta_p1 = corners_world_relative[0]
    delta_p2 = corners_world_relative[1] - corners_world_relative[0]
    result = np.concatenate([
        rotation_6d,
        state[7:10],
        state[10:13],
        previous,
        delta_p1.reshape(-1),
        delta_p2.reshape(-1),
    ]).astype(np.float32)
    if result.shape != (GREEN2026_FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ValueError("Green-2026 features must be finite and 40-dimensional")
    return result


def green2026_route_chain_features_from_components(
    *,
    state: np.ndarray,
    gate_positions_body: np.ndarray,
    gate_normals_body: np.ndarray,
    gate_up_body: np.ndarray,
    gate_sizes: np.ndarray,
    previous_action_normalized: np.ndarray,
) -> np.ndarray:
    """Build a Green-style Markov descriptor for an arbitrary route horizon.

    The first 16 values retain Green et al.'s dynamics/action state. Gate zero
    is represented by its four corners relative to the vehicle in world axes;
    each subsequent gate is represented by its corresponding-corner delta from
    the preceding gate. The chain is translation invariant, preserves complete
    gate pose/aperture geometry, and avoids repeating the vehicle-relative
    offset six times. For two gates it is numerically identical to the original
    40-D paper contract.
    """

    state = np.asarray(state, np.float32)
    positions = np.asarray(gate_positions_body, np.float32)
    normals = np.asarray(gate_normals_body, np.float32)
    up = np.asarray(gate_up_body, np.float32)
    sizes = np.asarray(gate_sizes, np.float32)
    previous = np.asarray(previous_action_normalized, np.float32)
    if (
        state.ndim != 1 or state.shape[0] < 13
        or positions.ndim != 2 or positions.shape[1:] != (3,)
        or len(positions) < 1
        or normals.shape != positions.shape or up.shape != positions.shape
        or sizes.shape != (len(positions), 2)
        or previous.shape != (4,)
    ):
        raise ValueError("incompatible components for Green route-chain observation")
    world_from_body = quaternion_matrix(state[3:7]).astype(np.float32)
    rotation_6d = world_from_body[:, :2].T.reshape(-1)
    corners_body = np.stack([
        _gate_corners(positions[index], normals[index], up[index], sizes[index])
        for index in range(len(positions))
    ])
    corners_world_relative = np.einsum(
        "ij,gkj->gki", world_from_body, corners_body
    ).astype(np.float32)
    route = np.concatenate([
        corners_world_relative[0:1],
        np.diff(corners_world_relative, axis=0),
    ], axis=0)
    result = np.concatenate([
        rotation_6d,
        state[7:10],
        state[10:13],
        previous,
        route.reshape(-1),
    ]).astype(np.float32)
    expected = GREEN2026_DYNAMICS_DIM + 12 * len(positions)
    if result.shape != (expected,) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"Green route-chain features must be finite and {expected}-dimensional"
        )
    return result


def green2026_observation_features(
    observation: Mapping[str, Any],
    *,
    action_encoder: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    gates = observation["gates"]
    return green2026_features_from_components(
        state=np.asarray(observation["state"], np.float32),
        gate_positions_body=np.asarray(gates["position"], np.float32)[:2],
        gate_normals_body=np.asarray(gates["normal"], np.float32)[:2],
        gate_up_body=np.asarray(gates["up"], np.float32)[:2],
        gate_sizes=np.asarray(gates["size"], np.float32)[:2],
        previous_action_normalized=action_encoder(
            np.asarray(observation["previous_action"], np.float32)
        ),
    )


def green2026_route_chain_observation_features(
    observation: Mapping[str, Any],
    *,
    route_gates: int,
    action_encoder: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    gates = observation["gates"]
    route_gates = int(route_gates)
    return green2026_route_chain_features_from_components(
        state=np.asarray(observation["state"], np.float32),
        gate_positions_body=np.asarray(gates["position"], np.float32)[:route_gates],
        gate_normals_body=np.asarray(gates["normal"], np.float32)[:route_gates],
        gate_up_body=np.asarray(gates["up"], np.float32)[:route_gates],
        gate_sizes=np.asarray(gates["size"], np.float32)[:route_gates],
        previous_action_normalized=action_encoder(
            np.asarray(observation["previous_action"], np.float32)
        ),
    )


def green2026_batch_features(
    *,
    states: np.ndarray,
    gate_positions_body: np.ndarray,
    gate_normals_body: np.ndarray,
    gate_up_body: np.ndarray,
    gate_sizes: np.ndarray,
    previous_actions_normalized: np.ndarray,
) -> np.ndarray:
    """Vector-safe offline adapter used by BC and scratch DAgger."""

    states = np.asarray(states, np.float32)
    previous = np.asarray(previous_actions_normalized, np.float32)
    count = len(states)
    arrays = (
        np.asarray(gate_positions_body, np.float32),
        np.asarray(gate_normals_body, np.float32),
        np.asarray(gate_up_body, np.float32),
        np.asarray(gate_sizes, np.float32),
    )
    if states.ndim != 2 or states.shape[1] < 13 or previous.shape != (count, 4):
        raise ValueError("offline Green-2026 state/action arrays are misaligned")
    if any(len(value) != count or value.shape[1] < 2 for value in arrays):
        raise ValueError("offline Green-2026 gate arrays need two aligned gates")
    return np.stack([
        green2026_features_from_components(
            state=states[index],
            gate_positions_body=arrays[0][index, :2],
            gate_normals_body=arrays[1][index, :2],
            gate_up_body=arrays[2][index, :2],
            gate_sizes=arrays[3][index, :2],
            previous_action_normalized=previous[index],
        )
        for index in range(count)
    ]).astype(np.float32)


def green2026_route_chain_batch_features(
    *,
    states: np.ndarray,
    gate_positions_body: np.ndarray,
    gate_normals_body: np.ndarray,
    gate_up_body: np.ndarray,
    gate_sizes: np.ndarray,
    previous_actions_normalized: np.ndarray,
    route_gates: int,
) -> np.ndarray:
    """Offline adapter for the variable-horizon Green route-chain contract."""

    states = np.asarray(states, np.float32)
    previous = np.asarray(previous_actions_normalized, np.float32)
    count = len(states)
    route_gates = int(route_gates)
    arrays = (
        np.asarray(gate_positions_body, np.float32),
        np.asarray(gate_normals_body, np.float32),
        np.asarray(gate_up_body, np.float32),
        np.asarray(gate_sizes, np.float32),
    )
    if states.ndim != 2 or states.shape[1] < 13 or previous.shape != (count, 4):
        raise ValueError("offline Green route-chain state/action arrays are misaligned")
    if route_gates < 1 or any(
        len(value) != count or value.shape[1] < route_gates for value in arrays
    ):
        raise ValueError(
            "offline Green route-chain gate arrays do not cover the requested route"
        )
    return np.stack([
        green2026_route_chain_features_from_components(
            state=states[index],
            gate_positions_body=arrays[0][index, :route_gates],
            gate_normals_body=arrays[1][index, :route_gates],
            gate_up_body=arrays[2][index, :route_gates],
            gate_sizes=arrays[3][index, :route_gates],
            previous_action_normalized=previous[index],
        )
        for index in range(count)
    ]).astype(np.float32)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling (SciPy-free)."""

    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_rank_correlation(values: Sequence[float]) -> float:
    values = np.asarray(values, np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Spearman history must contain at least two finite values")
    x = np.arange(len(values), dtype=np.float64)
    x -= x.mean()
    y = _rankdata(values)
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    # A constant return history is maximally flat, hence rho=0.
    return 0.0 if denominator <= 1.0e-12 else float((x @ y) / denominator)


@dataclass(frozen=True, slots=True)
class SpearmanSwitchResult:
    probability: float
    correlation: float | None
    ready: bool


class SpearmanTaskSwitcher:
    """Per-task learning-progress switcher from Green et al. Algorithm 2."""

    def __init__(self, *, window: int = 600, alpha: float = 1000.0) -> None:
        if int(window) < 2 or not np.isfinite(alpha) or float(alpha) <= 0.0:
            raise ValueError("Spearman switching needs window>=2 and alpha>0")
        self.window = int(window)
        self.alpha = float(alpha)
        self._returns: dict[str, deque[float]] = {}

    def observe(self, task: str, cumulative_return: float) -> SpearmanSwitchResult:
        value = float(cumulative_return)
        if not task or not np.isfinite(value):
            raise ValueError("task return must be finite and task non-empty")
        history = self._returns.setdefault(task, deque(maxlen=self.window))
        history.append(value)
        if len(history) < self.window:
            return SpearmanSwitchResult(0.0, None, False)
        rho = spearman_rank_correlation(history)
        probability = 1.0 / (1.0 + self.alpha * rho * rho)
        return SpearmanSwitchResult(float(probability), rho, True)

    def replace(self, task: str) -> None:
        self._returns.pop(str(task), None)

    def state_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "alpha": self.alpha,
            "returns": {name: list(values) for name, values in self._returns.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if int(state["window"]) != self.window or float(state["alpha"]) != self.alpha:
            raise ValueError("adaptive task-switch state uses a different contract")
        self._returns = {
            str(name): deque((float(value) for value in values), maxlen=self.window)
            for name, values in state.get("returns", {}).items()
        }


class DaggerLearningProgressBank:
    """Finite-pool DAgger proxy for the paper's adaptive PPO task bank.

    DAgger has no per-task policy return at every optimizer iteration.  This
    proxy therefore feeds each active task's closed-loop validation score into
    the same Spearman switch rule once per DAgger round.  The distinction is
    scientific, not cosmetic: configs must use a much shorter window and
    results must not be reported as an exact reproduction of Algorithm 2.
    """

    def __init__(
        self,
        tasks: Sequence[str],
        *,
        active_tasks: int,
        group_by_task: Mapping[str, str] | None = None,
        window: int = 8,
        alpha: float = 1000.0,
        seed: int = 0,
    ) -> None:
        self.tasks = tuple(str(task) for task in tasks)
        if len(self.tasks) != len(set(self.tasks)) or not self.tasks:
            raise ValueError("DAgger task bank requires unique non-empty tasks")
        if not 1 <= int(active_tasks) <= len(self.tasks):
            raise ValueError("invalid active DAgger task-bank size")
        if group_by_task is None:
            self.group_by_task = {task: "__all__" for task in self.tasks}
        else:
            self.group_by_task = {
                task: str(group_by_task[task]) for task in self.tasks
                if task in group_by_task
            }
            if set(self.group_by_task) != set(self.tasks):
                missing = set(self.tasks) - set(self.group_by_task)
                raise ValueError(
                    f"DAgger task-bank groups missing tasks {missing}"
                )
            if any(not group for group in self.group_by_task.values()):
                raise ValueError("DAgger task-bank groups must be non-empty")
        groups = tuple(sorted(set(self.group_by_task.values())))
        if group_by_task is not None and int(active_tasks) < len(groups):
            raise ValueError(
                "group-preserving DAgger bank needs at least one active task "
                "per group"
            )
        self.switcher = SpearmanTaskSwitcher(window=window, alpha=alpha)
        self.rng = np.random.default_rng(int(seed))
        initial_tasks: list[str] = []
        if group_by_task is not None:
            for group in groups:
                members = [
                    task for task in self.tasks
                    if self.group_by_task[task] == group
                ]
                initial_tasks.append(str(self.rng.choice(members)))
        remaining = [task for task in self.tasks if task not in initial_tasks]
        fill = int(active_tasks) - len(initial_tasks)
        if fill:
            selected = self.rng.choice(len(remaining), size=fill, replace=False)
            initial_tasks.extend(remaining[int(index)] for index in selected)
        self.active = initial_tasks
        self.switches = 0

    def observe(self, scores: Mapping[str, float]) -> dict[str, float]:
        missing = set(self.active) - set(scores)
        if missing:
            raise ValueError(f"DAgger task-bank scores missing active tasks {missing}")
        decisions: list[tuple[int, SpearmanSwitchResult]] = []
        for slot, task in enumerate(self.active):
            decisions.append((
                slot,
                self.switcher.observe(f"slot-{slot:03d}", float(scores[task])),
            ))
        replacements = 0
        for slot, decision in decisions:
            if not decision.ready or self.rng.random() >= decision.probability:
                continue
            outgoing = self.active[slot]
            group = self.group_by_task[outgoing]
            inactive = [
                task for task in self.tasks
                if task not in self.active and self.group_by_task[task] == group
            ]
            if not inactive:
                continue
            incoming = str(self.rng.choice(inactive))
            self.active[slot] = incoming
            self.switcher.replace(f"slot-{slot:03d}")
            self.switches += 1
            replacements += 1
        ready = sum(int(decision.ready) for _, decision in decisions)
        probabilities = [decision.probability for _, decision in decisions]
        return {
            "dagger_task_bank/ready": float(ready),
            "dagger_task_bank/replacements": float(replacements),
            "dagger_task_bank/total_switches": float(self.switches),
            "dagger_task_bank/switch_probability_mean": float(
                np.mean(probabilities)
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "tasks": list(self.tasks),
            "group_by_task": dict(self.group_by_task),
            "active": list(self.active),
            "switcher": self.switcher.state_dict(),
            "rng_state": self.rng.bit_generator.state,
            "switches": self.switches,
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if tuple(state["tasks"]) != self.tasks:
            raise ValueError("DAgger task-bank pool changed across resume")
        stored_groups = {
            str(task): str(group) for task, group in dict(
                state.get("group_by_task", {
                    task: "__all__" for task in self.tasks
                })
            ).items()
        }
        if stored_groups != self.group_by_task:
            raise ValueError("DAgger task-bank groups changed across resume")
        active = [str(task) for task in state["active"]]
        if len(active) != len(self.active) or not set(active) <= set(self.tasks):
            raise ValueError("DAgger active task-bank state is incompatible")
        self.active = active
        self.switcher.load_state_dict(state.get("switcher"))
        self.rng.bit_generator.state = state["rng_state"]
        self.switches = int(state.get("switches", 0))
