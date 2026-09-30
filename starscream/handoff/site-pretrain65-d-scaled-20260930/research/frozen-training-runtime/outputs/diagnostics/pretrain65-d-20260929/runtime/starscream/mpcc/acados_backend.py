"""acados SQP-RTI backend for the Starscream MPCC.

Imports are intentionally local to construction so track tools and datasets can
be used without a compiled acados installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import MPCCConfig, VehicleModelConfig


F64 = NDArray[np.float64]
NX = 18
NU = 5
NP = 51
MOTOR_OMEGA_SCALE = 1000.0


@dataclass(frozen=True, slots=True)
class AcadosResult:
    status: int
    states: F64
    controls: F64
    solve_time: float
    iterations: int
    residuals: F64
    slacks: F64
    wall_breakdown: F64


def acados_available() -> bool:
    try:
        import acados_template  # noqa: F401
        import casadi  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


class AcadosRTIBackend:
    """Warm-started nonlinear MPCC solved with preparation/feedback RTI phases."""

    def __init__(
        self,
        config: MPCCConfig,
        vehicle: VehicleModelConfig,
        world_bounds: F64,
        *,
        build_directory: str | Path | None = None,
    ) -> None:
        if not acados_available():
            raise RuntimeError(
                "acados is unavailable; rebuild the Starscream image or use backend='predictive'"
            )
        self.config = config
        self.native_stage_updates = False
        self._native_corridor_update = None
        self.vehicle = vehicle
        self._runtime_parameter_values = self._vehicle_runtime_parameters(vehicle)
        self.world_bounds = np.asarray(world_bounds, dtype=np.float64)
        if self.world_bounds.shape != (3, 2):
            raise ValueError("world_bounds must have shape (3, 2)")
        # acados exports C symbols from ``model.name``. Two MPCC envelopes in
        # one DAgger worker previously both used ``starscream_mpcc``, allowing
        # the dynamic loader to resolve a recovery solver against the fast
        # solver's generated functions. Include the structural config in the
        # symbol namespace so heterogeneous experts can coexist safely.
        self.model_name = f"starscream_mpcc_runtime51_{config.fingerprint[:12]}"
        self._solver = self._build_solver(build_directory)
        self._warm_states: F64 | None = None
        self._warm_controls: F64 | None = None

    def _build_solver(self, build_directory: str | Path | None):
        import casadi as ca
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

        cfg = self.vehicle
        weights = self.config.weights
        x = ca.SX.sym("x", NX)
        xdot = ca.SX.sym("xdot", NX)
        u = ca.SX.sym("u", NU)
        parameter = ca.SX.sym("p", NP)
        position, quaternion = x[0:3], x[3:7]
        velocity, omega, normalized_motors, progress = x[7:10], x[10:13], x[13:17], x[17]
        motors = MOTOR_OMEGA_SCALE * normalized_motors
        collective, commanded_rates, progress_rate = u[0], u[1:4], u[4]
        reference_position = parameter[0:3]
        tangent, lateral, up = parameter[3:6], parameter[6:9], parameter[9:12]
        reference_velocity = parameter[12:15]
        previous_action = parameter[15:19]
        reference_progress = parameter[21]
        reference_progress_rate = parameter[22]

        runtime_mass = parameter[23]
        runtime_arm_length = parameter[24]
        runtime_motor_omega_min = parameter[25]
        runtime_motor_omega_max = parameter[26]
        runtime_motor_tau = parameter[27]
        runtime_thrust_map = parameter[28:31]
        runtime_kappa = parameter[31]
        linear_drag = parameter[32:35]
        quadratic_drag = parameter[35:38]
        rotor_drag = parameter[38:41]
        angular_drag = parameter[41:44]
        wind_world = parameter[44:47]
        center_of_mass = parameter[47:50]
        effective_thrust_omega_max = parameter[50]
        inertia = (
            runtime_mass / 12.0 * runtime_arm_length**2
            * ca.DM([4.5, 4.5, 7.0])
        )
        inertia_inverse = 1.0 / inertia
        arm = runtime_arm_length * np.sqrt(0.5)
        allocation = ca.vertcat(
            ca.DM.ones(1, 4),
            arm * ca.DM([[1.0, -1.0, -1.0, 1.0]]),
            arm * ca.DM([[-1.0, -1.0, 1.0, 1.0]]),
            runtime_kappa * ca.DM([[1.0, -1.0, 1.0, -1.0]]),
        )
        allocation_inverse = ca.inv(allocation)
        rate_gain = ca.DM(cfg.rate_gain)
        omega_cross_jomega = ca.cross(omega, inertia * omega)
        desired_torque = inertia * rate_gain * (commanded_rates - omega) + omega_cross_jomega
        desired_wrench = ca.vertcat(runtime_mass * collective, desired_torque)
        desired_thrust = allocation_inverse @ desired_wrench
        a, b, c = runtime_thrust_map[0], runtime_thrust_map[1], runtime_thrust_map[2]
        thrust_max = a * effective_thrust_omega_max**2 + b * effective_thrust_omega_max + c
        desired_thrust = ca.fmin(ca.fmax(desired_thrust, 0.0), thrust_max)
        discriminant = ca.fmax(b * b - 4.0 * a * (c - desired_thrust), 0.0)
        desired_motor_omega = (-b + ca.sqrt(discriminant)) / (2.0 * a)
        desired_motor_omega = ca.fmin(
            ca.fmax(desired_motor_omega, runtime_motor_omega_min), runtime_motor_omega_max
        )
        motor_derivative = (
            (desired_motor_omega - motors) / (runtime_motor_tau * MOTOR_OMEGA_SCALE)
        )
        motor_thrust = ca.fmin(ca.fmax(a * motors**2 + b * motors + c, 0.0), thrust_max)
        wrench = allocation @ motor_thrust

        qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        rotation = ca.vertcat(
            ca.horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)),
            ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)),
        )
        quaternion_derivative = 0.5 * ca.vertcat(
            -qx * omega[0] - qy * omega[1] - qz * omega[2],
            qw * omega[0] + qy * omega[2] - qz * omega[1],
            qw * omega[1] - qx * omega[2] + qz * omega[0],
            qw * omega[2] + qx * omega[1] - qy * omega[0],
        )
        air_velocity_body = rotation.T @ (velocity - wind_world)
        drag_force_body = (
            -linear_drag * air_velocity_body
            -quadratic_drag * ca.fabs(air_velocity_body) * air_velocity_body
            -ca.sum1(motors) * rotor_drag * air_velocity_body
        )
        thrust_body = ca.vertcat(0.0, 0.0, wrench[0])
        acceleration = (
            rotation @ (thrust_body + drag_force_body) / runtime_mass
            + ca.DM([0.0, 0.0, -cfg.gravity])
        )
        aerodynamic_torque = -angular_drag * omega + ca.cross(center_of_mass, thrust_body)
        omega_derivative = inertia_inverse * (
            wrench[1:4] + aerodynamic_torque - omega_cross_jomega
        )
        explicit = ca.vertcat(
            velocity,
            quaternion_derivative,
            acceleration,
            omega_derivative,
            motor_derivative,
            progress_rate,
        )

        if self.config.progress_coupled_contouring:
            # Local spatial-path model. The former objective compared against
            # a fixed reference position at each horizon timestamp, making it
            # trajectory tracking rather than MPCC. Coupling the local path
            # point to theta lets the optimizer trade progress against lag.
            path_position = reference_position + tangent * (
                progress - reference_progress
            )
        else:
            path_position = reference_position
        error = position - path_position
        lag_error = ca.dot(error, tangent)
        contour_error = error - tangent * lag_error
        quaternion_norm_error = ca.dot(quaternion, quaternion) - 1.0
        stage_residual = ca.vertcat(
            np.sqrt(weights.contour) * contour_error,
            np.sqrt(weights.lag) * lag_error,
            np.sqrt(weights.velocity) * (velocity - reference_velocity),
            np.sqrt(weights.body_rate) * omega,
            np.sqrt(weights.action) * (u[0:4] - ca.DM([cfg.gravity, 0.0, 0.0, 0.0])),
            np.sqrt(weights.action_smoothness) * (u[0:4] - previous_action),
            np.sqrt(weights.progress) * (progress_rate - reference_progress_rate),
            np.sqrt(weights.progress_tracking) * (progress - reference_progress),
            np.sqrt(weights.attitude) * quaternion_norm_error,
        )
        terminal_residual = ca.vertcat(
            np.sqrt(weights.terminal * weights.contour) * contour_error,
            np.sqrt(weights.terminal * weights.lag) * lag_error,
            np.sqrt(weights.terminal * weights.velocity) * (velocity - reference_velocity),
            np.sqrt(weights.terminal * weights.progress_tracking) * (progress - reference_progress),
            np.sqrt(weights.terminal * weights.attitude) * quaternion_norm_error,
        )

        model = AcadosModel()
        model.name = self.model_name
        model.x, model.xdot, model.u, model.p = x, xdot, u, parameter
        model.f_expl_expr = explicit
        model.f_impl_expr = xdot - explicit
        model.cost_y_expr = stage_residual
        model.cost_y_expr_e = terminal_residual
        model.con_h_expr = ca.vertcat(ca.dot(error, lateral), ca.dot(error, up))

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.config.horizon
        ocp.parameter_values = np.concatenate([
            np.zeros(23, np.float64), self._runtime_parameters()
        ])
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.eye(int(stage_residual.shape[0]))
        ocp.cost.W_e = np.eye(int(terminal_residual.shape[0]))
        ocp.cost.yref = np.zeros(int(stage_residual.shape[0]))
        ocp.cost.yref_e = np.zeros(int(terminal_residual.shape[0]))
        ocp.cost.Zl = weights.slack * np.ones(2)
        ocp.cost.Zu = weights.slack * np.ones(2)
        ocp.cost.zl = 0.1 * weights.slack * np.ones(2)
        ocp.cost.zu = 0.1 * weights.slack * np.ones(2)

        ocp.constraints.x0 = np.zeros(NX)
        ocp.constraints.idxbu = np.arange(NU)
        ocp.constraints.lbu = np.asarray(
            [self.config.minimum_collective_thrust, *(-np.asarray(cfg.body_rate_max)), -self.config.reverse_progress_speed]
        )
        ocp.constraints.ubu = np.asarray(
            [self.config.maximum_collective_thrust, *cfg.body_rate_max, self.config.max_progress_speed]
        )
        # The motor states are part of the prediction model, so they need
        # physical bounds just like position.  Without these, SQP can exploit
        # negative rotor speeds to create a mathematically cheap but impossible
        # trajectory and feed a zero-thrust action to Flightmare.
        ocp.constraints.idxbx = np.asarray([0, 1, 2, 13, 14, 15, 16])
        ocp.constraints.lbx = np.concatenate([self.world_bounds[:, 0], np.zeros(4)])
        ocp.constraints.ubx = np.concatenate(
            [
                self.world_bounds[:, 1],
                np.full(4, cfg.motor_omega_max / MOTOR_OMEGA_SCALE),
            ]
        )
        ocp.constraints.lh = np.asarray([-1.0, -1.0])
        ocp.constraints.uh = np.asarray([1.0, 1.0])
        ocp.constraints.idxsh = np.asarray([0, 1])
        ocp.constraints.lsh = np.zeros(2)
        ocp.constraints.ush = np.zeros(2)

        options = ocp.solver_options
        options.nlp_solver_type = "SQP_RTI"
        options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        options.hpipm_mode = "ROBUST"
        options.hessian_approx = "GAUSS_NEWTON"
        options.levenberg_marquardt = 1.0e-4
        options.integrator_type = "IRK"
        options.sim_method_num_stages = 2
        options.sim_method_num_steps = 1
        options.qp_solver_cond_N = min(8, self.config.horizon)
        options.qp_solver_iter_max = 100
        options.nlp_solver_max_iter = 1
        options.print_level = 0
        options.time_steps = np.asarray(self.config.time_steps)
        options.tf = self.config.horizon_seconds

        if build_directory is not None:
            destination = Path(build_directory).resolve()
            destination.mkdir(parents=True, exist_ok=True)
            ocp.code_export_directory = str(destination / "c_generated_code")
            json_file = str(destination / f"{self.model_name}.json")
        else:
            json_file = f"{self.model_name}.json"
        return AcadosOcpSolver(ocp, json_file=json_file)

    def reset(self) -> None:
        self._warm_states = None
        self._warm_controls = None
        self._solver.reset()

    def set_vehicle(self, vehicle: VehicleModelConfig) -> None:
        """Update runtime plant parameters without rebuilding generated code."""

        old = self.vehicle
        self.vehicle = vehicle
        self._runtime_parameter_values = self._vehicle_runtime_parameters(vehicle)
        # Gust parameters may change every tick; bounds only change at plant
        # resets. Rotor speed bounds are distinct from the thrust saturation
        # envelope (Flightmare's nominal YAML intentionally has different caps).
        if (old.motor_omega_max, old.body_rate_max) != (vehicle.motor_omega_max, vehicle.body_rate_max):
            self._set_runtime_bounds()
            self.reset()

    def _set_runtime_bounds(self) -> None:
        lower = np.concatenate([self.world_bounds[:, 0], np.zeros(4)])
        upper = np.concatenate([self.world_bounds[:, 1],
            np.full(4, self.vehicle.motor_omega_max / MOTOR_OMEGA_SCALE)])
        input_lower = np.asarray([self.config.minimum_collective_thrust,
            *(-np.asarray(self.vehicle.body_rate_max)), -self.config.reverse_progress_speed])
        input_upper = np.asarray([self.config.maximum_collective_thrust,
            *self.vehicle.body_rate_max, self.config.max_progress_speed])
        for index in range(1, self.config.horizon):
            self._solver.constraints_set(index, "lbx", lower)
            self._solver.constraints_set(index, "ubx", upper)
            self._solver.constraints_set(index, "lbu", input_lower)
            self._solver.constraints_set(index, "ubu", input_upper)
        # Stage zero state is an equality and its input gets the current slew
        # box in solve(); neither may be replaced with intermediate bounds.

    @staticmethod
    def _vehicle_runtime_parameters(cfg: VehicleModelConfig) -> F64:
        return np.asarray([
            cfg.mass, cfg.arm_length, cfg.motor_omega_min, cfg.motor_omega_max,
            cfg.motor_tau, *cfg.thrust_map, cfg.kappa,
            *cfg.linear_drag, *cfg.quadratic_drag, *cfg.rotor_drag,
            *cfg.angular_drag, *cfg.wind_world, *cfg.center_of_mass,
            cfg.effective_thrust_omega_max,
        ], np.float64)

    def _runtime_parameters(self) -> F64:
        return self._runtime_parameter_values

    @staticmethod
    def pack_state(state: F64, motor_omega: F64, progress: float) -> F64:
        return np.concatenate(
            [
                state[0:3],
                state[3:7],
                state[7:10],
                state[10:13],
                motor_omega / MOTOR_OMEGA_SCALE,
                [progress],
            ]
        ).astype(np.float64)

    def set_world_bounds(self, world_bounds: F64) -> None:
        """Retarget runtime position bounds when a worker changes tracks.

        The constrained state indices are structural, but their lower/upper
        values are ordinary ACADOS runtime data. Updating them avoids compiling
        one identical solver per procedural track while preserving each track's
        actual world-bound contract.
        """

        bounds = np.asarray(world_bounds, dtype=np.float64)
        if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)):
            raise ValueError("world_bounds must be a finite (3, 2) array")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("world_bounds lower limits must precede upper limits")
        if np.allclose(self.world_bounds, bounds):
            return
        self.world_bounds = bounds.copy()
        self._set_runtime_bounds()
        self.reset()

    def solve(
        self,
        initial_state: F64,
        reference: dict[str, F64],
        previous_action: F64,
        *,
        warm_start: bool,
        initial_guess_states: F64 | None = None,
        initial_guess_controls: F64 | None = None,
    ) -> AcadosResult:
        wall_started = time.perf_counter()
        solver = self._solver
        horizon = self.config.horizon
        # Clear dual/QP memory before installing the shifted primal warm start.
        # Reusing HPIPM duals after the spatial reference is relinearized can
        # otherwise terminate at ACADOS_MINSTEP even for a feasible trajectory.
        solver.reset()
        required = {"position", "tangent", "lateral", "up", "velocity", "progress", "corridor_lateral", "corridor_vertical"}
        if required - reference.keys():
            raise ValueError(f"missing reference arrays: {sorted(required - reference.keys())}")
        action_references = np.repeat(previous_action[None], horizon + 1, axis=0)
        if warm_start and self._warm_states is not None and self._warm_controls is not None:
            shifted_states = np.vstack([self._warm_states[1:], self._warm_states[-1]])
            shifted_controls = np.vstack([self._warm_controls[1:], self._warm_controls[-1]])
            solver.set_flat("x", shifted_states.reshape(-1))
            solver.set_flat("u", shifted_controls.reshape(-1))
            action_references[1:] = shifted_controls[:, 0:4]
        else:
            if initial_guess_states is not None and initial_guess_controls is not None:
                if initial_guess_states.shape != (horizon + 1, NX) or initial_guess_controls.shape != (horizon, NU):
                    raise ValueError("cold-start guesses have incorrect horizon shapes")
                solver.set_flat("x", np.asarray(initial_guess_states).reshape(-1))
                solver.set_flat("u", np.asarray(initial_guess_controls).reshape(-1))
                action_references[1:] = initial_guess_controls[:, 0:4]
            else:
                guess_states = np.repeat(initial_state[None], horizon + 1, axis=0)
                guess_states[:, 17] = reference["progress"]
                solver.set_flat("x", guess_states.reshape(-1))
                hover = np.asarray([self.vehicle.gravity, 0.0, 0.0, 0.0, 0.0])
                solver.set_flat("u", np.tile(hover, horizon))

        progress_speed = np.asarray(
            reference.get("progress_speed", np.zeros(horizon + 1)),
            dtype=np.float64,
        )
        runtime = np.broadcast_to(
            self._runtime_parameters(), (horizon + 1, len(self._runtime_parameters()))
        )
        parameters = np.column_stack([
            reference["position"],
            reference["tangent"],
            reference["lateral"],
            reference["up"],
            reference["velocity"],
            action_references,
            reference["corridor_lateral"],
            reference["corridor_vertical"],
            reference["progress"],
            progress_speed,
            runtime,
        ]).astype(np.float64, copy=False)
        solver.set_flat("p", parameters.reshape(-1))
        corridor_upper = np.column_stack([
            reference["corridor_lateral"], reference["corridor_vertical"]
        ]).astype(np.float64, copy=False)
        corridor_lower = -corridor_upper
        if self.native_stage_updates:
            if self._native_corridor_update is None:
                from .native_prediction import NativeCorridorUpdate
                self._native_corridor_update = NativeCorridorUpdate(solver, horizon)
            self._native_corridor_update(corridor_lower, corridor_upper)
        else:
            for index in range(1, horizon):
                solver.constraints_set(index, "lh", corridor_lower[index])
                solver.constraints_set(index, "uh", corridor_upper[index])
        # The plant receives a hard-slew-limited command.  Put that same
        # envelope on the first OCP input so the trajectory optimized by RTI
        # starts with the command that will actually be executed.  Previously
        # the controller clipped the solved command only after optimization;
        # in tight turns that model mismatch produced alternating saturated
        # body-rate requests, loss of thrust, and eventual departure from the
        # racing line despite an ACADOS_SUCCESS status.
        input_lower = np.asarray(
            [
                self.config.minimum_collective_thrust,
                *(-np.asarray(self.vehicle.body_rate_max)),
                -self.config.reverse_progress_speed,
            ],
            dtype=np.float64,
        )
        input_upper = np.asarray(
            [
                self.config.maximum_collective_thrust,
                *self.vehicle.body_rate_max,
                self.config.max_progress_speed,
            ],
            dtype=np.float64,
        )
        slew = np.asarray(
            [
                self.config.collective_slew_limit,
                *([self.config.body_rate_slew_limit] * 3),
            ],
            dtype=np.float64,
        )
        input_lower[:4] = np.maximum(input_lower[:4], previous_action - slew)
        input_upper[:4] = np.minimum(input_upper[:4], previous_action + slew)
        solver.constraints_set(0, "lbu", input_lower)
        solver.constraints_set(0, "ubu", input_upper)
        # A shifted horizon's former stage-one command can fall outside the
        # new stage-zero slew box.  Start the QP from a primal-feasible command
        # instead of asking HPIPM to recover from that artificial violation.
        solver.set(
            0,
            "u",
            np.clip(np.asarray(solver.get(0, "u"), dtype=np.float64), input_lower, input_upper),
        )
        solver.set(0, "lbx", initial_state)
        solver.set(0, "ubx", initial_state)
        setup_ready = time.perf_counter()

        solver.options_set("rti_phase", 1)
        preparation_status = int(solver.solve())
        solver.options_set("rti_phase", 2)
        status = int(solver.solve())
        solve_ready = time.perf_counter()
        # ACADOS_READY (5) is the expected return after preparation-only RTI.
        if preparation_status not in {0, 5} and status == 0:
            status = preparation_status
        states = solver.get_flat("x").reshape(horizon + 1, NX)
        controls = solver.get_flat("u").reshape(horizon, NU)
        extraction_ready = time.perf_counter()
        if status == 0 and np.all(np.isfinite(states)) and np.all(np.isfinite(controls)):
            self._warm_states = states.copy()
            self._warm_controls = controls.copy()
        else:
            self._warm_states = None
            self._warm_controls = None
        try:
            residuals = np.asarray(solver.get_stats("residuals"), dtype=np.float64).reshape(-1)
        except Exception:
            residuals = np.zeros(4, dtype=np.float64)
        try:
            iterations = int(np.max(np.asarray(solver.get_stats("sqp_iter"))))
        except Exception:
            iterations = 1
        solve_time = float(np.sum(np.asarray(solver.get_stats("time_tot"))))
        try:
            slacks = np.concatenate([
                solver.get_flat("sl"), solver.get_flat("su")
            ])
        except Exception:
            slacks = np.zeros(0)
        return AcadosResult(
            status, states, controls, solve_time, iterations, residuals, slacks,
            np.asarray([
                setup_ready - wall_started,
                solve_ready - setup_ready,
                extraction_ready - solve_ready,
                time.perf_counter() - extraction_ready,
            ], np.float64),
        )
