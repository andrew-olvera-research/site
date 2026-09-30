"""Deterministic MPCC scenario suites, metrics, and failure replays."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from numpy.typing import NDArray

from ..env.gate_audit import audit_gate_trajectory
from ..env.tracks import Track, matrix_quaternion
from .controller import MPCCController, MPCCMode
from .racing_line import RacingLine


F64 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    suite: str
    state: F64
    seed: int
    expected_gates: int = 1
    maximum_steps: int = 450
    metadata: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    name: str
    suite: str
    seed: int
    success: bool
    steps: int
    gates_passed: int
    expected_gates: int
    elapsed_time: float
    contour_rms: float
    contour_p95: float
    lag_rms: float
    minimum_constraint_margin: float
    saturation_fraction: float
    solver_failure_fraction: float
    solve_time_p50_ms: float
    solve_time_p95_ms: float
    solve_time_p99_ms: float
    maximum_body_rate: float
    maximum_collective_thrust: float
    mean_collective_thrust: float
    mean_speed: float
    maximum_speed: float
    mask_visibility_mean: float
    camera_alignment_mean: float
    gate_labels_consistent: bool
    minimum_center_clearance: float | None
    minimum_body_clearance: float | None
    maximum_gate_transition_progress_jump: float
    failure_reason: str = ""


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    suite: str
    track: str
    results: tuple[ScenarioResult, ...]
    controller_source: str
    racing_line_hash: str
    controller_hash: str

    @property
    def summary(self) -> dict[str, float | int | str]:
        if not self.results:
            return {"suite": self.suite, "track": self.track, "scenarios": 0}
        solves = np.asarray([result.solve_time_p99_ms for result in self.results])
        body_clearances = [
            result.minimum_body_clearance
            for result in self.results
            if result.minimum_body_clearance is not None
        ]
        return {
            "suite": self.suite,
            "track": self.track,
            "scenarios": len(self.results),
            "success_rate": float(np.mean([result.success for result in self.results])),
            "gate_pass_rate": float(
                sum(min(result.gates_passed, result.expected_gates) for result in self.results)
                / max(sum(result.expected_gates for result in self.results), 1)
            ),
            "solve_time_p99_ms_worst": float(np.max(solves)),
            "solver_failure_rate": float(
                np.mean([result.solver_failure_fraction for result in self.results])
            ),
            "minimum_constraint_margin": float(
                np.min([result.minimum_constraint_margin for result in self.results])
            ),
            "minimum_body_clearance": (
                float(np.min(body_clearances)) if body_clearances else None
            ),
            "maximum_gate_transition_progress_jump": float(
                np.max(
                    [result.maximum_gate_transition_progress_jump for result in self.results]
                )
            ),
        }

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary,
            "controller_source": self.controller_source,
            "racing_line_hash": self.racing_line_hash,
            "controller_hash": self.controller_hash,
            "results": [asdict(result) for result in self.results],
        }
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return destination


def _state_on_line(
    line: RacingLine,
    progress: float,
    *,
    speed: float = 0.0,
    lateral: float = 0.0,
    vertical: float = 0.0,
    rate_error: F64 | None = None,
    attitude_error_degrees: F64 | None = None,
) -> F64:
    frame = line.evaluate(progress)
    state = np.zeros(25, dtype=np.float64)
    state[0:3] = (
        frame["position"] + lateral * frame["lateral"] + vertical * frame["up"]
    )
    rotation = np.stack([frame["tangent"], frame["lateral"], frame["up"]], axis=1)
    if attitude_error_degrees is not None:
        roll, pitch, yaw = np.radians(np.asarray(attitude_error_degrees, dtype=np.float64))
        cr, sr, cp, sp, cy, sy = (
            np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        )
        local_error = np.asarray(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )
        rotation = rotation @ local_error
    state[3:7] = matrix_quaternion(rotation)
    state[7:10] = speed * frame["tangent"]
    state[10:13] = np.zeros(3) if rate_error is None else rate_error
    return state


def build_scenarios(
    track: Track,
    line: RacingLine,
    suite: str,
    *,
    seeds: Iterable[int] = (0,),
    maximum_steps: int = 450,
) -> list[Scenario]:
    if suite not in {"smoke", "nominal", "recovery", "robustness", "regression"}:
        raise ValueError("unknown benchmark suite")
    scenarios: list[Scenario] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if suite == "smoke":
            definitions = [
                ("hover", -3.5, 0.0, 0.0, 0.0),
                ("straight-acceleration", -4.0, 2.0, 0.0, 0.0),
                ("braking", -4.0, 11.0, 0.0, 0.0),
                ("single-gate", -3.0, 4.0, 0.0, 0.0),
                ("turn-90", float(line.gate_progress[1] - line.gate_progress[0] - 3.0), 5.0, 0.0, 0.0),
            ]
            for name, offset, speed, lateral, vertical in definitions:
                base = float(line.gate_progress[0])
                scenarios.append(
                    Scenario(
                        name=f"{name}-seed-{seed}",
                        suite=suite,
                        state=_state_on_line(line, base + offset, speed=speed, lateral=lateral, vertical=vertical),
                        seed=seed,
                        maximum_steps=min(maximum_steps, 240),
                        metadata={"gate_index": 1 if name == "turn-90" else 0},
                    )
                )
        elif suite == "nominal":
            for start_index in range(len(track.gates)):
                jitter = float(rng.uniform(-0.1, 0.1))
                scenarios.append(
                    Scenario(
                        name=f"lap-start-{start_index}-seed-{seed}",
                        suite=suite,
                        state=_state_on_line(
                            line,
                            line.gate_progress[start_index] - 3.0,
                            speed=3.0,
                            lateral=jitter,
                        ),
                        seed=seed,
                        expected_gates=len(track.gates),
                        maximum_steps=maximum_steps,
                        metadata={"gate_index": start_index},
                    )
                )
        elif suite == "recovery":
            for lateral in (0.5, 1.0, 1.8):
                for speed in (0.0, 5.0):
                    for attitude in (0.0, 20.0):
                        for rate_scale in (0.0, 1.0):
                            signed = lateral if rng.random() > 0.5 else -lateral
                            attitude_sign = 1.0 if rng.random() > 0.5 else -1.0
                            rates = rate_scale * rng.uniform(-1.5, 1.5, 3)
                            error_degrees = attitude_sign * np.asarray(
                                [attitude, 0.6 * attitude, 0.8 * attitude]
                            )
                            difficulty = "moderate" if lateral <= 1.0 and attitude <= 20.0 else "hard"
                            scenarios.append(
                                Scenario(
                                    name=(
                                        f"recovery-{difficulty}-lat-{signed:+.1f}-speed-{speed:.0f}"
                                        f"-att-{attitude:.0f}-rate-{rate_scale:.0f}-seed-{seed}"
                                    ),
                                    suite=suite,
                                    state=_state_on_line(
                                        line,
                                        line.gate_progress[0] - 3.0,
                                        speed=speed,
                                        lateral=signed,
                                        vertical=float(rng.uniform(-0.5, 0.5)),
                                        rate_error=rates,
                                        attitude_error_degrees=error_degrees,
                                    ),
                                    seed=seed,
                                    expected_gates=2,
                                    maximum_steps=min(maximum_steps, 360),
                                    metadata={
                                        "difficulty": difficulty,
                                        "lateral_error": signed,
                                        "initial_speed": speed,
                                        "attitude_error_degrees": attitude,
                                        "body_rate_error": rate_scale,
                                    },
                                )
                            )
        elif suite == "robustness":
            # These cases deliberately introduce command-channel mismatch on
            # top of the pinned native dynamics: effective thrust/mass error,
            # rate bias (disturbance), and action latency.
            variants = (
                ("nominal", 1.00, 0.000, 0.0),
                ("heavy", 0.88, 0.011, 0.0),
                ("light", 1.12, 0.011, 0.0),
                ("delay", 1.00, 0.022, 0.0),
                ("drag-bias", 0.94, 0.011, 0.35),
                ("combined", 0.86, 0.022, 0.45),
            )
            for variant, authority_scale, delay, rate_bias_scale in variants:
                signed = 0.6 if rng.random() > 0.5 else -0.6
                rate_bias = rate_bias_scale * rng.uniform(-1.0, 1.0, 3)
                scenarios.append(
                    Scenario(
                        name=f"robustness-{variant}-seed-{seed}",
                        suite=suite,
                        state=_state_on_line(
                            line,
                            line.gate_progress[0] - 3.0,
                            speed=5.0,
                            lateral=signed,
                            vertical=float(rng.uniform(-0.35, 0.35)),
                            rate_error=rng.uniform(-0.5, 0.5, 3),
                        ),
                        seed=seed,
                        expected_gates=2,
                        maximum_steps=min(maximum_steps, 360),
                        metadata={
                            "variant": variant,
                            "authority_scale": authority_scale,
                            "action_delay": delay,
                            "rate_bias_x": float(rate_bias[0]),
                            "rate_bias_y": float(rate_bias[1]),
                            "rate_bias_z": float(rate_bias[2]),
                        },
                    )
                )
        else:
            # Exact regression states are discovered from HDF5 replay files by
            # the CLI. An empty built-in suite is deliberate.
            continue
    return scenarios


def save_failure_replay(
    path: str | Path,
    scenario: Scenario,
    arrays: dict[str, list[np.ndarray]],
    result: ScenarioResult,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as archive:
        archive.attrs["scenario"] = scenario.name
        archive.attrs["suite"] = scenario.suite
        archive.attrs["seed"] = scenario.seed
        archive.attrs["expected_gates"] = scenario.expected_gates
        archive.attrs["metadata_json"] = json.dumps(scenario.metadata, sort_keys=True)
        archive.attrs["result_json"] = json.dumps(asdict(result), sort_keys=True)
        archive.create_dataset("initial_state", data=scenario.state)
        for name, values in arrays.items():
            if values:
                archive.create_dataset(name, data=np.stack(values), compression="lzf")
    return destination


def load_replay_scenarios(
    paths: Iterable[str | Path],
    track: Track,
) -> list[Scenario]:
    """Load exact initial conditions from prior HDF5 failure bundles."""

    scenarios: list[Scenario] = []
    for value in sorted(Path(path) for path in paths):
        with h5py.File(value, "r") as archive:
            original_suite = str(archive.attrs.get("suite", "regression"))
            if "expected_gates" in archive.attrs:
                expected_gates = int(archive.attrs["expected_gates"])
            elif original_suite == "nominal":
                expected_gates = len(track.gates)
            elif original_suite in {"recovery", "robustness"}:
                expected_gates = 2
            else:
                expected_gates = 1
            metadata = json.loads(str(archive.attrs.get("metadata_json", "{}")))
            metadata["replay_path"] = str(value)
            scenarios.append(
                Scenario(
                    name=f"regression-{value.stem}",
                    suite="regression",
                    state=np.asarray(archive["initial_state"], dtype=np.float64),
                    seed=int(archive.attrs.get("seed", 0)),
                    expected_gates=expected_gates,
                    maximum_steps=1200,
                    metadata=metadata,
                )
            )
    return scenarios


class BenchmarkRunner:
    def __init__(
        self,
        env: Any,
        controller: MPCCController,
        *,
        output_directory: str | Path,
    ) -> None:
        self.env = env
        self.controller = controller
        self.output_directory = Path(output_directory)

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        if hasattr(self.env, "action_delay"):
            self.env.action_delay = float(
                scenario.metadata.get("action_delay", self.env.action_delay)
            )
        observation, _ = self.env.reset(
            seed=scenario.seed,
            options={
                "state": scenario.state.astype(np.float32),
                "gate_index": int(scenario.metadata.get("gate_index", 0)),
            },
        )
        self.controller.reset()
        arrays: dict[str, list[np.ndarray]] = {
            "state": [],
            "action": [],
            "progress": [],
            "contour_lag_error": [],
            "constraint_residuals": [],
            "solve_time": [],
            "solver_status": [],
            "mode": [],
            "trajectory_state": [np.asarray(observation["state"], np.float32)],
            "gate_passed": [],
        }
        mask_visibility: list[float] = []
        camera_alignment: list[float] = []
        gates_passed = 0
        failure_reason = ""
        terminated = truncated = False
        for _ in range(scenario.maximum_steps):
            gate_mask = observation.get("gate_mask")
            if gate_mask is not None:
                mask_visibility.append(float(np.mean(np.asarray(gate_mask) > 0)))
            relative_gates = observation.get("gates", {}).get("position")
            if relative_gates is not None and len(relative_gates):
                relative = np.asarray(relative_gates[0], dtype=np.float64)
                camera_alignment.append(
                    float(relative[0] / max(np.linalg.norm(relative), 1.0e-8))
                )
            command = self.controller(observation)
            action = command.action.as_array()
            if not np.all(np.isfinite(action)):
                failure_reason = "non-finite-command"
                break
            diagnostics = command.diagnostics
            arrays["state"].append(np.asarray(observation["state"], np.float32))
            arrays["action"].append(action)
            arrays["progress"].append(np.asarray(command.reference_progress, np.float32))
            arrays["contour_lag_error"].append(np.asarray(diagnostics["contour_lag_error"][0]))
            arrays["constraint_residuals"].append(np.asarray(diagnostics["constraint_residuals"]))
            arrays["solve_time"].append(np.asarray(command.solve_time, np.float64))
            arrays["solver_status"].append(np.asarray(command.solver_status, np.int32))
            arrays["mode"].append(np.asarray(diagnostics["mode"], np.int8))
            applied_action = action.astype(np.float64, copy=True)
            applied_action[0] *= float(scenario.metadata.get("authority_scale", 1.0))
            applied_action[1:4] += np.asarray(
                [
                    scenario.metadata.get("rate_bias_x", 0.0),
                    scenario.metadata.get("rate_bias_y", 0.0),
                    scenario.metadata.get("rate_bias_z", 0.0),
                ],
                dtype=np.float64,
            )
            applied_action[0] = np.clip(
                applied_action[0],
                self.controller.config.minimum_collective_thrust,
                self.controller.config.maximum_collective_thrust,
            )
            applied_action[1:4] = np.clip(
                applied_action[1:4],
                -np.asarray(self.controller.vehicle.body_rate_max),
                np.asarray(self.controller.vehicle.body_rate_max),
            )
            observation, _, terminated, truncated, info = self.env.step(applied_action)
            gate_passed = bool(info.get("gate_passed", False))
            arrays["trajectory_state"].append(np.asarray(observation["state"], np.float32))
            arrays["gate_passed"].append(np.asarray(gate_passed, np.bool_))
            gates_passed += int(gate_passed)
            if info.get("unity_collision", False) or info.get("ground_contact", False):
                failure_reason = "collision"
                break
            if gates_passed >= scenario.expected_gates:
                break
            if terminated or truncated:
                failure_reason = "environment-termination"
                break
        steps = len(arrays["action"])
        contour_lag = np.stack(arrays["contour_lag_error"]) if steps else np.zeros((1, 2))
        solve_times = 1e3 * np.asarray(arrays["solve_time"], dtype=np.float64)
        actions = np.stack(arrays["action"]) if steps else np.zeros((1, 4))
        residuals = np.stack(arrays["constraint_residuals"]) if steps else np.zeros((1, 8))
        statuses = np.asarray(arrays["solver_status"], dtype=np.int32)
        geometry = audit_gate_trajectory(
            self.controller.track,
            np.stack(arrays["trajectory_state"]),
            np.asarray(arrays["gate_passed"], np.bool_),
            start_gate_index=int(scenario.metadata.get("gate_index", 0)),
            expected_passes=scenario.expected_gates,
        )
        gate_events = np.flatnonzero(np.asarray(arrays["gate_passed"], np.bool_)) + 1
        gate_events = gate_events[gate_events < len(arrays["progress"])]
        transition_jumps: list[float] = []
        progress_values = np.asarray(arrays["progress"], np.float64)
        for index in gate_events:
            delta = float(progress_values[index] - progress_values[index - 1])
            if self.controller.track.loop:
                delta = float(
                    (delta + 0.5 * self.controller.line.length)
                    % self.controller.line.length
                    - 0.5 * self.controller.line.length
                )
            transition_jumps.append(abs(delta))
        maximum_transition_jump = max(transition_jumps, default=0.0)
        physical_success = bool(geometry.qualified and maximum_transition_jump <= 0.5)
        success = gates_passed >= scenario.expected_gates and not failure_reason and physical_success
        if not failure_reason and not geometry.labels_consistent:
            failure_reason = "gate-label-mismatch"
        elif not failure_reason and not geometry.body_clear:
            failure_reason = "vehicle-clearance-violation"
        elif not failure_reason and maximum_transition_jump > 0.5:
            failure_reason = "gate-transition-progress-discontinuity"
        if not success and not failure_reason:
            failure_reason = "gate-timeout"
        result = ScenarioResult(
            name=scenario.name,
            suite=scenario.suite,
            seed=scenario.seed,
            success=success,
            steps=steps,
            gates_passed=gates_passed,
            expected_gates=scenario.expected_gates,
            elapsed_time=float(steps * getattr(self.env, "control_dt", 0.02)),
            contour_rms=float(np.sqrt(np.mean(contour_lag[:, 0] ** 2))),
            contour_p95=float(np.quantile(np.abs(contour_lag[:, 0]), 0.95)),
            lag_rms=float(np.sqrt(np.mean(contour_lag[:, 1] ** 2))),
            minimum_constraint_margin=float(np.min(residuals[:, 6:8])),
            saturation_fraction=float(
                np.mean(
                    (actions[:, 0] >= self.controller.config.maximum_collective_thrust - 1e-3)
                    | (np.max(np.abs(actions[:, 1:4]), axis=1) >= max(self.controller.vehicle.body_rate_max) - 1e-3)
                )
            ),
            solver_failure_fraction=float(np.mean(statuses != 0)) if steps else 1.0,
            solve_time_p50_ms=float(np.quantile(solve_times, 0.50)) if steps else float("inf"),
            solve_time_p95_ms=float(np.quantile(solve_times, 0.95)) if steps else float("inf"),
            solve_time_p99_ms=float(np.quantile(solve_times, 0.99)) if steps else float("inf"),
            maximum_body_rate=float(np.max(np.abs(actions[:, 1:4]))),
            maximum_collective_thrust=float(np.max(actions[:, 0])),
            mean_collective_thrust=float(np.mean(actions[:, 0])),
            mean_speed=float(np.mean(np.linalg.norm(
                np.stack(arrays["state"])[:, 7:10], axis=1
            ))) if steps else 0.0,
            maximum_speed=float(np.max(np.linalg.norm(
                np.stack(arrays["state"])[:, 7:10], axis=1
            ))) if steps else 0.0,
            mask_visibility_mean=float(np.mean(mask_visibility)) if mask_visibility else 0.0,
            camera_alignment_mean=float(np.mean(camera_alignment)) if camera_alignment else -1.0,
            gate_labels_consistent=geometry.labels_consistent,
            minimum_center_clearance=(
                geometry.minimum_center_clearance if geometry.recorded_events else None
            ),
            minimum_body_clearance=(
                geometry.minimum_body_clearance if geometry.recorded_events else None
            ),
            maximum_gate_transition_progress_jump=maximum_transition_jump,
            failure_reason=failure_reason,
        )
        if not success:
            replay = (
                self.output_directory
                / "failures"
                / f"{self.controller.track.name}-{scenario.name}.h5"
            )
            save_failure_replay(replay, scenario, arrays, result)
        return result

    def run(self, scenarios: Iterable[Scenario]) -> BenchmarkReport:
        scenarios = list(scenarios)
        results = tuple(self.run_scenario(scenario) for scenario in scenarios)
        suite = scenarios[0].suite if scenarios else "unknown"
        report = BenchmarkReport(
            suite=suite,
            track=self.controller.track.name,
            results=results,
            controller_source=self.controller.source,
            racing_line_hash=self.controller.line.fingerprint,
            controller_hash=self.controller.config.fingerprint,
        )
        report.save_json(self.output_directory / f"{self.controller.track.name}-{suite}.json")
        return report
