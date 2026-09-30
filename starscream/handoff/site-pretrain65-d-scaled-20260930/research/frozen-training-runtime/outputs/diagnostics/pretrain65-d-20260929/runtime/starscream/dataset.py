"""On-disk episode schema and validation for Starscream world-model data."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import h5py
import numpy as np


REQUIRED_TRANSITION_KEYS = (
    "action/ctbr",
    "action/normalized",
    "action/dreamer16",
    "reward/total",
    "continue",
    "discount",
    "is_first",
    "is_last",
    "is_terminal",
)
REQUIRED_OBSERVATION_KEYS = (
    "observation/state",
    "observation/state_estimate",
    "observation/state_estimate_std",
    "observation/gates/position",
    "observation/gates/normal",
    "observation/gates/up",
    "observation/gates/size",
    "observation/gates/enter_from_opposite_side",
)
V3_OBSERVATION_KEYS = (
    "observation/flight_plan/records",
    "observation/task_state",
)
V4_OBSERVATION_KEYS = (
    "observation/gate_mask",
    "observation/measured/body_rates",
    "observation/measured/motor_omega",
    "observation/timestamp/sim",
    "observation/timestamp/camera",
    "observation/timestamp/body_rates",
    "observation/timestamp/motor_omega",
    "observation/timestamp/state_estimate",
    "observation/timestamp/previous_action",
    "observation/valid/camera",
    "observation/valid/body_rates",
    "observation/valid/motor_omega",
    "observation/valid/state_estimate",
    "observation/valid/previous_action",
    "observation/age/camera",
    "observation/age/body_rates",
    "observation/age/motor_omega",
    "observation/age/state_estimate",
    "observation/age/previous_action",
    "observation/privileged/state",
    "observation/privileged/gate_state",
    "observation/privileged/dynamics",
    "observation/privileged/camera_extrinsics",
    "observation/privileged/camera_intrinsics",
    "observation/privileged/sky_state",
    "observation/privileged/actuator",
    "observation/privileged/disturbance",
    "observation/privileged/progress",
)
V4_TRANSITION_KEYS = (
    "action/applied_ctbr",
    "action/applied_motor_thrusts",
    "action/applied_motor_omega",
    "action/applied_motor_normalized",
    "action/command_timestamp",
    "action/applied_timestamp",
    "controller/source_timestamp",
    "controller/receive_timestamp",
    "controller/reference_state",
    "controller/reference_action",
    "controller/reference_progress",
    "controller/solver_status",
    "controller/solve_time",
    "controller/constraint_margin",
    "controller/valid",
)
V4_METADATA_KEYS = (
    "metadata/contract",
    "metadata/control_dt",
    "metadata/track",
    "metadata/track_fingerprint",
    "metadata/coordinate_convention",
    "metadata/action_convention",
    "metadata/dynamics_parameter_names",
    "metadata/actuator_parameter_names",
    "metadata/state_parameter_names",
    "metadata/sky_state_parameter_names",
    "metadata/mask_source",
    "metadata/image_delay_requested",
    "metadata/action_delay_requested",
    "metadata/retains_direct_images",
    "metadata/collector_policy",
    "metadata/gatenet_checkpoint_sha256",
    "metadata/seed",
    "controller/source",
)


def _dataset_keys(episode) -> list[str]:
    if isinstance(episode, h5py.File):
        keys: list[str] = []
        episode.visititems(lambda name, item: keys.append(name) if isinstance(item, h5py.Dataset) else None)
        return keys
    return list(episode.keys())


def validate_episode(episode: Mapping[str, np.ndarray] | h5py.File, *, require_images: bool = False) -> int:
    keys = _dataset_keys(episode)
    missing = [key for key in (*REQUIRED_TRANSITION_KEYS, *REQUIRED_OBSERVATION_KEYS) if key not in episode]
    if require_images:
        missing.extend(
            key for key in (
                "observation/rgb",
                "observation/depth",
                "observation/segmentation",
                "observation/optical_flow",
            ) if key not in episode
        )
    if missing:
        raise ValueError(f"episode is missing required fields: {', '.join(missing)}")
    schema_version = int(np.asarray(episode["schema_version"])) if "schema_version" in episode else 1
    if schema_version >= 3:
        missing_v3 = [key for key in V3_OBSERVATION_KEYS if key not in episode]
        if missing_v3:
            raise ValueError(f"schema v3 episode is missing: {', '.join(missing_v3)}")
    if schema_version >= 4:
        missing_v4 = [
            key for key in (*V4_OBSERVATION_KEYS, *V4_TRANSITION_KEYS, *V4_METADATA_KEYS)
            if key not in episode
        ]
        if missing_v4:
            raise ValueError(f"schema v4 episode is missing: {', '.join(missing_v4)}")
    length = int(episode["action/ctbr"].shape[0])
    if length < 1:
        raise ValueError("episode contains no transitions")
    for key in REQUIRED_TRANSITION_KEYS:
        if np.asarray(episode[key]).shape[0] != length:
            raise ValueError(f"transition field {key!r} is not length {length}")
    if schema_version >= 4:
        for key in V4_TRANSITION_KEYS:
            if np.asarray(episode[key]).shape[0] != length:
                raise ValueError(f"schema-v4 transition field {key!r} is not length {length}")
    for key in (item for item in keys if item.startswith("observation/")):
        if np.asarray(episode[key]).shape[0] != length + 1:
            raise ValueError(f"observation field {key!r} must contain T+1 frames")
    route_gate_count = (
        int(np.asarray(episode["metadata/route_gate_count"])[()])
        if "metadata/route_gate_count" in episode else 3
    )
    if route_gate_count < 1:
        raise ValueError("route-gate count must be positive")
    gate_shapes = {
        "observation/gates/position": (route_gate_count, 3),
        "observation/gates/normal": (route_gate_count, 3),
        "observation/gates/up": (route_gate_count, 3),
        "observation/gates/size": (route_gate_count, 2),
        "observation/gates/enter_from_opposite_side": (route_gate_count,),
    }
    for key, shape in gate_shapes.items():
        if key in episode and np.asarray(episode[key]).shape[1:] != shape:
            raise ValueError(
                f"relative-gate field {key!r} must have shape (T+1, {shape})"
            )
    if episode["action/ctbr"].shape[1:] != (4,):
        raise ValueError("CTBR action must have shape (T, 4)")
    if episode["action/dreamer16"].shape[1:] != (16,):
        raise ValueError("Dreamer action must have shape (T, 16)")
    if "observation/flight_plan/records" in episode:
        if episode["observation/flight_plan/records"].shape[1:] != (
            route_gate_count, 13,
        ):
            raise ValueError(
                "flight-plan records must agree with metadata/route_gate_count"
            )
    if "observation/task_state" in episode:
        if episode["observation/task_state"].shape[1:] != (19,):
            raise ValueError("task-state target must have shape (T+1, 19)")
    if schema_version >= 4:
        expected_shapes = {
            "observation/measured/body_rates": (3,),
            "observation/measured/motor_omega": (4,),
            "observation/privileged/state": (25,),
            "observation/privileged/gate_state": (19,),
            "observation/privileged/dynamics": (15,),
            "observation/privileged/camera_extrinsics": (7,),
            "observation/privileged/camera_intrinsics": (4,),
            "observation/privileged/sky_state": (23,),
            "observation/privileged/actuator": (30,),
            "observation/privileged/disturbance": (10,),
            "observation/privileged/progress": (6,),
            "controller/reference_state": (25,),
            "controller/reference_action": (4,),
            "action/applied_ctbr": (4,),
            "action/applied_motor_thrusts": (4,),
            "action/applied_motor_omega": (4,),
            "action/applied_motor_normalized": (4,),
        }
        for key, shape in expected_shapes.items():
            if episode[key].shape[1:] != shape:
                raise ValueError(f"schema-v4 field {key!r} must end in {shape}")
        sim_time = np.asarray(episode["observation/timestamp/sim"], dtype=np.float64)
        if not np.all(np.isfinite(sim_time)) or np.any(np.diff(sim_time) < 0):
            raise ValueError("schema-v4 simulator timestamps must be finite and monotonic")
        camera_time = np.asarray(episode["observation/timestamp/camera"], dtype=np.float64)
        camera_age = np.asarray(episode["observation/age/camera"], dtype=np.float64)
        if np.any(camera_time > sim_time + 1e-7) or np.any(camera_age < -1e-7):
            raise ValueError("schema-v4 camera data cannot originate in the future")
        if not np.allclose(camera_age, sim_time - camera_time, atol=2e-6, rtol=0):
            raise ValueError("schema-v4 camera age must equal simulator time minus capture time")
        command_time = np.asarray(episode["action/command_timestamp"], dtype=np.float64)
        applied_time = np.asarray(episode["action/applied_timestamp"], dtype=np.float64)
        if not np.all(np.isfinite(command_time)) or not np.all(np.isfinite(applied_time)):
            raise ValueError("schema-v4 action timestamps must be finite")
        if np.any(applied_time > command_time + 1e-7):
            raise ValueError("schema-v4 applied actions cannot be newer than their command step")
        motor_action = np.asarray(episode["action/applied_motor_normalized"])
        if np.any(motor_action < -1e-6) or np.any(motor_action > 1.0 + 1e-6):
            raise ValueError("schema-v4 normalized motor actuation must lie in [0, 1]")
        mask = np.asarray(episode["observation/gate_mask"])
        if mask.dtype != np.uint8 or not np.all((mask == 0) | (mask == 255)):
            raise ValueError("schema-v4 gate masks must be binary uint8 values 0/255")
        retains_images = int(np.asarray(episode["metadata/retains_direct_images"]))
        if not retains_images:
            leaked = [
                key for key in (
                    "observation/rgb", "observation/depth", "observation/segmentation",
                    "observation/optical_flow",
                ) if key in episode
            ]
            if leaked:
                raise ValueError(f"mask-only schema-v4 episode contains direct images: {leaked}")
        if not np.all(np.isfinite(np.asarray(episode["observation/privileged/state"]))):
            raise ValueError("schema-v4 privileged state must be finite")
    return length


def load_episode(path: str | Path, *, require_images: bool = False) -> dict[str, np.ndarray]:
    """Load a complete episode for inspection; training uses direct HDF5 slices."""

    episode: dict[str, np.ndarray] = {}
    with h5py.File(Path(path), "r", swmr=True) as archive:
        validate_episode(archive, require_images=require_images)
        def collect(name, item):
            if isinstance(item, h5py.Dataset):
                value = item[()]
                if isinstance(value, bytes):
                    value = np.asarray(value.decode("utf-8"))
                episode[name] = np.asarray(value)
        archive.visititems(collect)
    return episode
