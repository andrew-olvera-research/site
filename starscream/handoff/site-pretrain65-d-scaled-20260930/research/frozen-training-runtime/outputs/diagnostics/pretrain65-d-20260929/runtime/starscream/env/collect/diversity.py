"""Stratified expert, recovery, near-crash, and crash data collection."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from enum import Enum
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ..tracks import Track, matrix_quaternion
from ..types import CTBRAction, ControllerCommand


class CollectionRegime(str, Enum):
    CHAMPION = "champion"
    RECOVERY = "recovery"
    NEAR_CRASH = "near_crash"
    CRASH = "crash"


@dataclass(frozen=True, slots=True)
class CollectionMixtureConfig:
    """Target episode mixture for a generalization dataset."""

    champion: float = 0.40
    recovery: float = 0.35
    near_crash: float = 0.15
    crash: float = 0.10

    def __post_init__(self) -> None:
        values = np.asarray(list(self.as_mapping().values()), dtype=np.float64)
        if np.any(values < 0) or not np.all(np.isfinite(values)) or not np.isclose(values.sum(), 1.0):
            raise ValueError("collection mixture weights must be finite, non-negative, and sum to one")

    def as_mapping(self) -> dict[CollectionRegime, float]:
        return {
            CollectionRegime.CHAMPION: self.champion,
            CollectionRegime.RECOVERY: self.recovery,
            CollectionRegime.NEAR_CRASH: self.near_crash,
            CollectionRegime.CRASH: self.crash,
        }


def allocate_regimes(
    episodes: int,
    mixture: CollectionMixtureConfig | None = None,
    *,
    seed: int = 0,
) -> tuple[CollectionRegime, ...]:
    """Largest-remainder allocation followed by a deterministic shuffle.

    Unlike independent categorical draws, every collection job closely matches
    its declared mixture even when only tens of episodes are requested.
    """

    if episodes < 1:
        raise ValueError("episodes must be positive")
    mixture = mixture or CollectionMixtureConfig()
    mapping = mixture.as_mapping()
    regimes = tuple(mapping)
    exact = np.asarray([episodes * mapping[regime] for regime in regimes])
    counts = np.floor(exact).astype(int)
    remaining = episodes - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:remaining]] += 1
    # If the job is large enough to cover every requested regime, guarantee
    # support instead of allowing rounding to erase a rare crash category.
    positive = np.flatnonzero(np.asarray([mapping[regime] for regime in regimes]) > 0)
    if episodes >= len(positive):
        for missing in positive[counts[positive] == 0]:
            donors = positive[counts[positive] > 1]
            donor = donors[int(np.argmax(counts[donors] - exact[donors]))]
            counts[donor] -= 1
            counts[missing] += 1
    schedule = [regime for regime, count in zip(regimes, counts) for _ in range(int(count))]
    np.random.default_rng(seed).shuffle(schedule)
    return tuple(schedule)


def _rotation_with_error(frame: dict[str, np.ndarray], euler_degrees: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.radians(euler_degrees)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    error = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    nominal = np.stack([frame["tangent"], frame["lateral"], frame["up"]], axis=1)
    return nominal @ error


class DiverseScenarioSampler:
    """Generate deterministic states spanning nominal and off-course flight."""

    def __init__(self, track: Track, racing_line: Any, *, seed: int = 0) -> None:
        self.track = track
        self.line = racing_line
        self.seed = int(seed)

    def sample(
        self,
        regime: CollectionRegime | str,
        index: int,
        gate_index: int | None = None,
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        regime = CollectionRegime(regime)
        rng = np.random.default_rng(self.seed + 7919 * int(index) + 101 * list(CollectionRegime).index(regime))
        target_gate = int(rng.integers(len(self.track.gates)) if gate_index is None else gate_index)

        recovery_difficulty = "not_applicable"
        if regime == CollectionRegime.RECOVERY:
            recovery_difficulty = "moderate" if int(index) % 2 == 0 else "hard"
            if recovery_difficulty == "moderate":
                approach = float(rng.uniform(1.5, 5.0))
                lateral = float(rng.choice([-1.0, 1.0]) * rng.uniform(1.7, 2.3))
                vertical = float(rng.uniform(-0.8, 0.8))
                euler_error = rng.uniform(-30.0, 30.0, 3)
                forward_speed = float(rng.uniform(0.0, 8.0))
                lateral_speed = float(rng.uniform(-2.5, 2.5))
                vertical_speed = float(rng.uniform(-1.5, 2.5))
                rates = rng.uniform(-2.0, 2.0, 3)
            else:
                approach = float(rng.uniform(0.5, 7.5))
                lateral = float(rng.choice([-1.0, 1.0]) * rng.uniform(2.3, 3.2))
                vertical = float(rng.uniform(-1.2, 1.2))
                euler_error = rng.uniform(-55.0, 55.0, 3)
                forward_speed = float(rng.uniform(-1.0, 10.0))
                lateral_speed = float(rng.uniform(-4.0, 4.0))
                vertical_speed = float(rng.uniform(-2.5, 4.0))
                rates = rng.uniform(-3.5, 3.5, 3)
        else:
            approach = float(rng.uniform(2.5, 4.5))
            lateral = float(np.clip(rng.normal(0.0, 0.10), -0.25, 0.25))
            vertical = float(np.clip(rng.normal(0.0, 0.08), -0.20, 0.20))
            euler_error = rng.uniform(-3.0, 3.0, 3)
            forward_speed = float(rng.uniform(3.0, 7.0))
            lateral_speed, vertical_speed = rng.normal(0.0, 0.15, 2)
            rates = rng.normal(0.0, 0.12, 3)

        progress = float(self.line.gate_progress[target_gate] - approach)
        frame = self.line.evaluate(progress)
        state = np.zeros(25, dtype=np.float32)
        position = (
            frame["position"] + lateral * frame["lateral"] + vertical * frame["up"]
        )
        margin = np.asarray([0.25, 0.25, 0.75 if regime == CollectionRegime.RECOVERY else 0.05])
        state[0:3] = np.clip(position, self.track.bounds[:, 0] + margin, self.track.bounds[:, 1] - margin)
        state[3:7] = matrix_quaternion(_rotation_with_error(frame, euler_error))
        state[7:10] = (
            forward_speed * frame["tangent"]
            + lateral_speed * frame["lateral"]
            + vertical_speed * frame["up"]
        )
        state[10:13] = rates
        metadata: dict[str, float | int | str] = {
            "distribution_regime": regime.value,
            "gate_index": target_gate,
            "line_progress": progress,
            "approach_distance": approach,
            "lateral_offset": lateral,
            "vertical_offset": vertical,
            "forward_speed": forward_speed,
            "lateral_speed": float(lateral_speed),
            "vertical_speed": float(vertical_speed),
            "attitude_error_degrees": float(np.linalg.norm(euler_error)),
            "body_rate_norm": float(np.linalg.norm(rates)),
            "recovery_difficulty": recovery_difficulty,
            "sampler": "stratified-racing-distribution-v1",
        }
        return state, metadata


class PerturbedControllerPolicy:
    """Inject labelled command bursts, then return control to the expert.

    Intervention commands are retained for dynamics/reward/continuation
    learning but marked ``controller/valid=false`` so they cannot silently be
    treated as behavior-cloning expert targets.
    """

    def __init__(
        self,
        controller: Callable[[dict[str, Any]], ControllerCommand],
        regime: CollectionRegime | str,
        *,
        seed: int = 0,
    ) -> None:
        self.controller = controller
        self.regime = CollectionRegime(regime)
        self.rng = np.random.default_rng(seed)
        if self.regime == CollectionRegime.CHAMPION:
            self.start, self.duration, self.severity = 10**9, 0, 0.0
        elif self.regime == CollectionRegime.RECOVERY:
            # The initial state is already broadly randomized.  Leave control
            # fully to the expert so this regime contains genuine recoveries,
            # not a second injected near-crash burst.
            self.start, self.duration, self.severity = 10**9, 0, 0.0
        elif self.regime == CollectionRegime.NEAR_CRASH:
            self.start = int(self.rng.integers(25, 81))
            self.duration = int(self.rng.integers(12, 31))
            self.severity = float(self.rng.uniform(0.55, 0.80))
        else:
            self.start = int(self.rng.integers(25, 81))
            self.duration = 180
            self.severity = 1.0
        direction = self.rng.choice([-1.0, 1.0], size=3)
        self._rate_target = direction * self.rng.uniform(3.5, 6.0, size=3)
        self._thrust_delta = float(self.rng.uniform(-8.0, 4.0))
        self._step = 0

    def reset(self) -> None:
        self._step = 0
        if hasattr(self.controller, "reset"):
            self.controller.reset()

    def __call__(self, observation: dict[str, Any]) -> ControllerCommand:
        if self._step == self.start + self.duration and hasattr(self.controller, "reset"):
            self.controller.reset()
        command = self.controller(observation)
        nominal = command.action.as_array()
        intervening = self.start <= self._step < self.start + self.duration
        applied_severity = self.severity if intervening else 0.0
        action = nominal.copy()
        if intervening:
            if self.regime == CollectionRegime.CRASH:
                action[0] = 0.0
                action[1:4] = self._rate_target
            else:
                target = nominal.copy()
                target[0] = np.clip(nominal[0] + self._thrust_delta, 0.0, 30.0)
                target[1:4] = self._rate_target
                action = (1.0 - self.severity) * nominal + self.severity * target
        action[0] = np.clip(action[0], 0.0, 30.0)
        action[1:4] = np.clip(action[1:4], -6.0, 6.0)
        diagnostics = {
            **command.diagnostics,
            "intervention": np.asarray(intervening, np.int8),
            "intervention_severity": np.asarray(applied_severity, np.float32),
            "nominal_action": nominal.astype(np.float32),
        }
        self._step += 1
        return replace(
            command,
            action=CTBRAction.from_array(action),
            valid=bool(command.valid and not intervening),
            source=f"{command.source}-distribution-v1",
            diagnostics=diagnostics,
        )


def episode_outcome_metadata(episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    transitions = len(episode.get("action/ctbr", ()))
    mode = np.asarray(episode.get("controller/diagnostics/mode", np.zeros(transitions)), dtype=np.int8)
    intervention = np.asarray(
        episode.get("controller/diagnostics/intervention", np.zeros(transitions)), dtype=np.bool_
    )
    collisions = np.asarray(episode.get("transition/unity_collision", False), dtype=np.bool_) | np.asarray(
        episode.get("transition/ground_contact", False), dtype=np.bool_
    )
    gates = np.asarray(episode.get("transition/gate_passed", False), dtype=np.bool_)
    recovered = bool(np.any(mode == 1) and np.any(gates[np.flatnonzero(mode == 1)[0] :])) if transitions else False
    return {
        "metadata/outcome_collision": np.asarray(bool(np.any(collisions))),
        "metadata/outcome_terminal": np.asarray(bool(np.any(episode.get("is_terminal", False)))),
        "metadata/outcome_recovered": np.asarray(recovered),
        "metadata/outcome_recovery_fraction": np.asarray(float(np.mean(mode == 1)) if transitions else 0.0, np.float32),
        "metadata/outcome_intervention_fraction": np.asarray(float(np.mean(intervention)) if transitions else 0.0, np.float32),
    }


@dataclass(frozen=True, slots=True)
class DistributionReport:
    episodes: int
    transitions: int
    episode_counts: dict[str, int]
    episode_fractions: dict[str, float]
    transition_fractions: dict[str, float]
    terminal_episode_fraction: float
    collision_episode_fraction: float
    recovered_episode_fraction: float
    recovery_regime_recovered_fraction: float
    near_crash_regime_survival_fraction: float
    crash_regime_collision_fraction: float
    recovery_transition_fraction: float
    intervention_transition_fraction: float
    distribution_total_variation: float
    all_regimes_present: bool

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return destination


def evaluate_collection_distribution(
    episodes: Iterable[dict[str, np.ndarray]],
    mixture: CollectionMixtureConfig | None = None,
) -> DistributionReport:
    mixture = mixture or CollectionMixtureConfig()
    episodes = list(episodes)
    counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    transitions = terminals = collisions = recovered = recovery_steps = interventions = 0
    recovery_regime_recovered = near_crash_survived = crash_regime_collided = 0
    for episode in episodes:
        raw_regime = np.asarray(episode.get("metadata/distribution_regime", "unknown")).item()
        regime = raw_regime.decode() if isinstance(raw_regime, bytes) else str(raw_regime)
        length = len(episode.get("action/ctbr", ()))
        counts[regime] += 1
        transition_counts[regime] += length
        transitions += length
        mode = np.asarray(episode.get("controller/diagnostics/mode", np.zeros(length)), dtype=np.int8)
        intervention = np.asarray(
            episode.get("controller/diagnostics/intervention", np.zeros(length)), dtype=np.bool_
        )
        collision = bool(
            np.any(episode.get("transition/unity_collision", False))
            or np.any(episode.get("transition/ground_contact", False))
        )
        terminal = bool(np.any(episode.get("is_terminal", False)))
        gate_events = np.asarray(episode.get("transition/gate_passed", np.zeros(length)), dtype=np.bool_)
        first_recovery = np.flatnonzero(mode == 1)
        did_recover = bool(len(first_recovery) and np.any(gate_events[first_recovery[0] :]))
        terminals += terminal
        collisions += collision
        recovered += did_recover
        recovery_regime_recovered += int(regime == CollectionRegime.RECOVERY.value and did_recover)
        near_crash_survived += int(regime == CollectionRegime.NEAR_CRASH.value and not terminal)
        crash_regime_collided += int(regime == CollectionRegime.CRASH.value and collision)
        recovery_steps += int(np.sum(mode == 1))
        interventions += int(np.sum(intervention))
    episode_count = len(episodes)
    fractions = {regime.value: counts[regime.value] / max(episode_count, 1) for regime in CollectionRegime}
    transition_fractions = {
        regime.value: transition_counts[regime.value] / max(transitions, 1) for regime in CollectionRegime
    }
    target = mixture.as_mapping()
    total_variation = 0.5 * sum(abs(fractions[regime.value] - target[regime]) for regime in CollectionRegime)
    return DistributionReport(
        episodes=episode_count,
        transitions=transitions,
        episode_counts={regime.value: counts[regime.value] for regime in CollectionRegime},
        episode_fractions=fractions,
        transition_fractions=transition_fractions,
        terminal_episode_fraction=terminals / max(episode_count, 1),
        collision_episode_fraction=collisions / max(episode_count, 1),
        recovered_episode_fraction=recovered / max(episode_count, 1),
        recovery_regime_recovered_fraction=(
            recovery_regime_recovered / max(counts[CollectionRegime.RECOVERY.value], 1)
        ),
        near_crash_regime_survival_fraction=(
            near_crash_survived / max(counts[CollectionRegime.NEAR_CRASH.value], 1)
        ),
        crash_regime_collision_fraction=(
            crash_regime_collided / max(counts[CollectionRegime.CRASH.value], 1)
        ),
        recovery_transition_fraction=recovery_steps / max(transitions, 1),
        intervention_transition_fraction=interventions / max(transitions, 1),
        distribution_total_variation=float(total_variation),
        all_regimes_present=all(counts[regime.value] > 0 for regime in CollectionRegime),
    )
