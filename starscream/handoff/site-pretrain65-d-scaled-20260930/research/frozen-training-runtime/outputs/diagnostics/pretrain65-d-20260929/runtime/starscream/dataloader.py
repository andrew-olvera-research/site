"""PyTorch sequence dataset over collected Starscream HDF5 episodes."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
from math import ceil
from pathlib import Path
import random

import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from .dataset import validate_episode


REGIME_TO_INDEX = {
    "champion": 0,
    "recovery": 1,
    "near_crash": 2,
    "crash": 3,
    "unknown": 4,
}

TAIL_PHASE_NAMES = (
    "ordinary", "deviation_onset", "near_crash", "recovery", "collision_imminent"
)

GATE_PHASE_NAMES = (
    "ordinary", "approach", "crossing", "post_crossing", "next_gate_acquisition"
)


def derive_tail_phases(
    *,
    mode: np.ndarray,
    intervention: np.ndarray,
    projection_distance: np.ndarray,
    constraint_margin: np.ndarray,
    collision: np.ndarray,
    regime: str,
    onset_lookback: int = 45,
    rejoin_horizon: int = 45,
    collision_lookback: int = 45,
) -> np.ndarray:
    """Derive causal racing phases from aligned collection diagnostics.

    These labels are privileged training metadata, never encoder inputs. Phase
    precedence is collision > recovery > near-crash > onset > ordinary.
    """

    arrays = [
        np.asarray(value).reshape(-1)
        for value in (mode, intervention, projection_distance, constraint_margin, collision)
    ]
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise ValueError("tail phase diagnostic arrays must have equal lengths")
    mode, intervention, projection_distance, constraint_margin, collision = arrays
    length = len(mode)
    phase = np.zeros(length, dtype=np.int8)
    intervention = intervention.astype(np.bool_)
    collision = collision.astype(np.bool_)
    severe_deviation = (projection_distance > 2.0) | (constraint_margin < -1.0)
    deviation = (projection_distance > 1.0) | (constraint_margin < 0.0)
    recovery = mode == 1
    collision_imminent = collision | (mode == 2)
    if regime == "crash":
        collision_imminent |= intervention
    near_crash = severe_deviation | intervention

    critical = deviation | near_crash | recovery | collision_imminent
    starts = np.flatnonzero(critical & ~np.r_[False, critical[:-1]])
    for start in starts:
        phase[max(0, start - int(onset_lookback)) : start] = 1
    phase[near_crash] = np.maximum(phase[near_crash], 2)
    phase[recovery] = np.maximum(phase[recovery], 3)

    recovery_ends = np.flatnonzero(recovery & ~np.r_[recovery[1:], False])
    for end in recovery_ends:
        stop = min(length, end + 1 + int(rejoin_horizon))
        phase[end + 1 : stop] = np.maximum(phase[end + 1 : stop], 3)

    collision_indices = np.flatnonzero(collision_imminent)
    for index in collision_indices:
        phase[max(0, index - int(collision_lookback)) : index + 1] = 4
    return phase


def derive_gate_transition_phases(
    gate_passed: np.ndarray,
    *,
    approach_lookback: int = 18,
    post_crossing_horizon: int = 10,
    acquisition_horizon: int = 45,
) -> np.ndarray:
    """Label short gate-boundary phases using only trajectory outcomes.

    These labels are training metadata and auxiliary targets, never policy
    inputs.  The deployable representation receives causal gate-index changes
    and time since a realized crossing instead.
    """

    passed = np.asarray(gate_passed, dtype=np.bool_).reshape(-1)
    phase = np.zeros(len(passed), dtype=np.int8)
    crossings = np.flatnonzero(passed)
    # Paint broad acquisition regions first so the closer approach/crossing
    # labels below retain precedence when consecutive windows overlap.
    for index in crossings:
        start = min(len(phase), index + 1 + int(post_crossing_horizon))
        stop = min(len(phase), index + 1 + int(acquisition_horizon))
        phase[start:stop] = 4
    for index in crossings:
        phase[max(0, index - int(approach_lookback)) : index] = 1
        phase[index] = 2
        stop = min(len(phase), index + 1 + int(post_crossing_horizon))
        phase[index + 1 : stop] = 3
    return phase


class _MemoryGroup:
    def __init__(self, arrays: dict[str, np.ndarray], prefix: str = "") -> None:
        self.arrays = arrays
        self.prefix = prefix.strip("/")

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def __contains__(self, name: str) -> bool:
        key = self._key(name)
        return key in self.arrays or any(value.startswith(key + "/") for value in self.arrays)

    def __getitem__(self, name: str):
        key = self._key(name)
        if key in self.arrays:
            return self.arrays[key]
        if any(value.startswith(key + "/") for value in self.arrays):
            return _MemoryGroup(self.arrays, key)
        raise KeyError(key)

    def close(self) -> None:
        return None


def stratified_episode_split(
    root: str | Path, validation_fraction: float, seed: int,
    tracks: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[list[Path], list[Path]]:
    """Split by track and collection regime so rare failures reach validation."""

    root = Path(root)
    paths = sorted((*root.glob("*.h5"), *root.glob("*.hdf5")))
    if not paths:
        raise ValueError(f"no HDF5 episodes found under {root}")
    requested_tracks = (
        {str(tracks)} if isinstance(tracks, str)
        else {str(value) for value in tracks} if tracks is not None
        else None
    )
    if requested_tracks is not None:
        selected: list[Path] = []
        for path in paths:
            with h5py.File(path, "r", swmr=True) as episode:
                if "metadata/track" not in episode:
                    track = path.stem.split("_")[0]
                else:
                    value = np.asarray(episode["metadata/track"]).item()
                    track = value.decode() if isinstance(value, bytes) else str(value)
            if track in requested_tracks:
                selected.append(path)
        paths = selected
        if not paths:
            raise ValueError(
                f"no HDF5 episodes under {root} matched tracks={sorted(requested_tracks)}"
            )
    if len(paths) == 1 or validation_fraction <= 0:
        return paths, []
    groups: dict[tuple[str, str], list[Path]] = {}
    for path in paths:
        with h5py.File(path, "r", swmr=True) as episode:
            def metadata(name: str, fallback: str) -> str:
                key = f"metadata/{name}"
                if key not in episode:
                    return fallback
                value = np.asarray(episode[key]).item()
                return value.decode() if isinstance(value, bytes) else str(value)

            key = (metadata("track", path.stem.split("_")[0]), metadata("distribution_regime", "unknown"))
        groups.setdefault(key, []).append(path)
    # Tiny legacy datasets often contain one episode per stratum. Fall back to
    # the original global deterministic split rather than empty validation.
    if not any(len(group) > 1 for group in groups.values()):
        shuffled = list(paths)
        random.Random(seed).shuffle(shuffled)
        count = min(len(paths) - 1, max(1, round(len(paths) * validation_fraction)))
        return shuffled[count:], shuffled[:count]
    train: list[Path] = []
    validation: list[Path] = []
    for group_index, key in enumerate(sorted(groups)):
        group = groups[key]
        random.Random(seed + 1009 * group_index).shuffle(group)
        if len(group) == 1:
            train.extend(group)
            continue
        count = min(len(group) - 1, max(1, round(len(group) * validation_fraction)))
        validation.extend(group[:count])
        train.extend(group[count:])
    return sorted(train), sorted(validation)


class DreamerSequenceDataset(Dataset):
    """Returns aligned current/next observations and Dreamer transition targets."""

    def __init__(
        self,
        root: str | Path,
        sequence_length: int = 16,
        stride: int = 1,
        *,
        mode: str = "legacy",
        mask_size: tuple[int, int] = (128, 160),
        prefer_gatenet: bool = True,
        require_controller_valid: bool = False,
        max_open_files: int = 8,
        validate_contents: bool = True,
        include_privileged: bool = True,
        cache_in_memory: bool = False,
        include_actions: bool = False,
        route_source: str = "flight_plan",
        route_target_source: str = "input",
        deployment_estimate_source: str = "none",
        deployment_estimate_seed: int = 0,
        paths: list[str | Path] | None = None,
    ) -> None:
        root = Path(root)
        self.paths = (
            sorted(Path(path) for path in paths)
            if paths is not None
            else sorted((*root.glob("*.h5"), *root.glob("*.hdf5")))
        )
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        if mode not in {"legacy", "tokenizer", "action_tokenizer", "dynamics", "control"}:
            raise ValueError(
                "mode must be 'legacy', 'tokenizer', 'action_tokenizer', "
                "'dynamics', or 'control'"
            )
        self.mode = mode
        self.mask_size = tuple(int(item) for item in mask_size)
        self.prefer_gatenet = bool(prefer_gatenet)
        self.require_controller_valid = bool(require_controller_valid)
        self.validate_contents = bool(validate_contents)
        self.include_privileged = bool(include_privileged)
        self.cache_in_memory = bool(cache_in_memory)
        self.include_actions = bool(include_actions)
        self.route_source = str(route_source)
        if self.route_source not in {"flight_plan", "body_relative_gates"}:
            raise ValueError("route_source must be 'flight_plan' or 'body_relative_gates'")
        self.route_target_source = str(route_target_source)
        if self.route_target_source not in {"input", "body_relative_gates"}:
            raise ValueError(
                "route_target_source must be 'input' or 'body_relative_gates'"
            )
        self.deployment_estimate_source = str(deployment_estimate_source)
        if self.deployment_estimate_source not in {"none", "proxy_v1"}:
            raise ValueError(
                "deployment_estimate_source must be 'none' or 'proxy_v1'"
            )
        if (
            self.deployment_estimate_source == "proxy_v1"
            and self.route_target_source != "body_relative_gates"
        ):
            raise ValueError(
                "proxy_v1 requires route_target_source: body_relative_gates"
            )
        self.deployment_estimate_seed = int(deployment_estimate_seed)
        self.max_open_files = int(max_open_files)
        if self.max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        # Handles are opened lazily inside each DataLoader worker and retained in
        # a small LRU. This avoids both an h5py.File open per sample and hundreds
        # of descriptors per persistent worker on large collections.
        self._files: OrderedDict[Path, h5py.File | _MemoryGroup] = OrderedDict()
        if self.sequence_length < 1 or self.stride < 1:
            raise ValueError("sequence_length and stride must be positive")
        self.windows: list[tuple[Path, int]] = []
        self.terminal_windows: list[bool] = []
        self.episode_regimes: dict[Path, int] = {}
        self.episode_tail_phases: dict[Path, np.ndarray] = {}
        self.episode_gate_phases: dict[Path, np.ndarray] = {}
        self.window_tail_phases: list[int] = []
        self.window_gate_phases: list[int] = []
        for path in self.paths:
            with h5py.File(path, "r", swmr=True) as episode:
                length = (
                    validate_episode(episode)
                    if self.validate_contents
                    else int(episode["action/ctbr"].shape[0])
                )
                starts = list(range(0, length - self.sequence_length + 1, self.stride))
                # A strided index does not necessarily land on the final legal
                # window. Always include it so successful finishes and crashes
                # are represented in continuation training.
                final_start = length - self.sequence_length
                if final_start >= 0 and (not starts or starts[-1] != final_start):
                    starts.append(final_start)
                if self.require_controller_valid and "controller/valid" in episode:
                    controller_valid = np.asarray(episode["controller/valid"][:], dtype=np.bool_)
                    starts = [
                        start
                        for start in starts
                        if np.all(controller_valid[start : start + self.sequence_length])
                    ]
                regime = "unknown"
                if "metadata/distribution_regime" in episode:
                    value = np.asarray(episode["metadata/distribution_regime"]).item()
                    regime = value.decode() if isinstance(value, bytes) else str(value)
                self.episode_regimes[path] = REGIME_TO_INDEX.get(
                    regime, REGIME_TO_INDEX["unknown"]
                )
                def array(name: str, default) -> np.ndarray:
                    return np.asarray(episode[name][:]) if name in episode else np.asarray(default)

                collision = (
                    array("transition/unity_collision", np.zeros(length, np.bool_)).astype(bool)
                    | array("transition/ground_contact", np.zeros(length, np.bool_)).astype(bool)
                    | array("transition/terminated", np.zeros(length, np.bool_)).astype(bool)
                )
                phases = derive_tail_phases(
                    mode=array("controller/diagnostics/mode", np.zeros(length, np.int8)),
                    intervention=array(
                        "controller/diagnostics/intervention", np.zeros(length, np.int8)
                    ),
                    projection_distance=array(
                        "controller/diagnostics/projection_distance", np.zeros(length, np.float32)
                    ),
                    constraint_margin=array(
                        "controller/constraint_margin", np.ones(length, np.float32)
                    ),
                    collision=collision,
                    regime=regime,
                )
                self.episode_tail_phases[path] = phases
                gate_phases = derive_gate_transition_phases(
                    array("transition/gate_passed", np.zeros(length, np.bool_))
                )
                self.episode_gate_phases[path] = gate_phases
            self.windows.extend((path, start) for start in starts)
            self.terminal_windows.extend(start + self.sequence_length == length for start in starts)
            self.window_tail_phases.extend(
                int(phases[start : start + self.sequence_length].max(initial=0))
                for start in starts
            )
            self.window_gate_phases.extend(
                int(gate_phases[start : start + self.sequence_length].max(initial=0))
                for start in starts
            )

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array))

    @contextmanager
    def _episode(self, path: Path):
        episode = self._files.pop(path, None)
        if episode is None:
            episode = self._materialize_episode(path) if self.cache_in_memory else h5py.File(path, "r", swmr=True)
        self._files[path] = episode
        while len(self._files) > self.max_open_files:
            _, stale = self._files.popitem(last=False)
            stale.close()
        yield episode

    def _materialize_episode(self, path: Path) -> _MemoryGroup:
        """Load only arrays consumed by the selected training mode."""

        arrays: dict[str, np.ndarray] = {}
        with h5py.File(path, "r", swmr=True) as archive:
            def add(*keys: str) -> None:
                for key in keys:
                    if key in archive and key not in arrays:
                        arrays[key] = np.asarray(archive[key][()])

            if self.mode == "action_tokenizer":
                add(
                    "action/applied_ctbr", "action/ctbr", "action/normalized",
                    "action/dreamer16", "action/command_timestamp", "action/applied_timestamp",
                    "action/applied_motor_normalized", "observation/timestamp/sim",
                    "observation/privileged/gate_state", "observation/task_state",
                    "observation/flight_plan/index",
                    "is_first", "is_last",
                )
                return _MemoryGroup(arrays)

            mask_candidates = (
                ("observation/gatenet_mask",) if self.prefer_gatenet else ()
            ) + (
                "observation/sim_gate_mask", "observation/gate_mask", "observation/segmentation"
            )
            for key in mask_candidates:
                if key in archive:
                    add(key)
                    break
            add(
                "observation/flight_plan/records", "observation/flight_plan/index",
                "observation/measured/body_rates",
                "observation/state", "observation/measured/motor_omega",
                "observation/motor_omega", "observation/previous_action",
                "observation/privileged/gate_state", "observation/task_state",
                "observation/timestamp/camera", "observation/timestamp/sim",
                "is_first",
            )
            if (
                self.route_source == "body_relative_gates"
                or self.route_target_source == "body_relative_gates"
                or self.deployment_estimate_source == "proxy_v1"
            ):
                add(
                    "observation/gates/position", "observation/gates/normal",
                    "observation/gates/up", "observation/gates/size",
                    "observation/gates/enter_from_opposite_side",
                    "observation/gates/mask",
                )
            if self.mode == "control":
                add(
                    "observation/state_estimate", "observation/state_estimate_std",
                    "observation/state_estimate_valid", "observation/flight_plan/index",
                    "action/applied_ctbr", "action/ctbr", "action/normalized",
                    "action/dreamer16", "action/command_timestamp",
                    "action/applied_timestamp", "reward/total", "continue",
                    "discount", "is_last", "is_terminal", "controller/valid",
                    "controller/reference_action",
                    "controller/diagnostics/intervention",
                    "transition/gate_passed", "reward/progress_delta",
                )
            for name in ("camera", "body_rates", "motor_omega", "previous_action"):
                add(f"observation/valid/{name}", f"observation/age/{name}")
            if self.include_privileged:
                for source in (
                    "observation/privileged/state", "observation/privileged/sky_state",
                    "observation/privileged/dynamics", "observation/privileged/actuator",
                    "observation/privileged/camera_extrinsics",
                    "observation/privileged/camera_intrinsics",
                    "observation/privileged/disturbance", "observation/privileged/progress",
                    "observation/state_estimate", "observation/state_estimate_std",
                ):
                    add(source)
            if self.include_actions:
                add(
                    "action/applied_ctbr", "action/ctbr", "action/command_timestamp",
                    "action/applied_timestamp",
                )
            if self.mode == "dynamics":
                add(
                    "action/dreamer16", "action/ctbr", "action/applied_ctbr",
                    "action/command_timestamp", "action/applied_timestamp",
                    "reward/total", "continue", "discount", "is_last", "is_terminal",
                    "controller/valid", "controller/reference_action",
                    "controller/diagnostics/intervention",
                    "transition/gate_passed", "reward/progress_delta",
                    "reward/gate_quality",
                )
        return _MemoryGroup(arrays)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_files"] = OrderedDict()
        return state

    def close(self) -> None:
        for episode in self._files.values():
            episode.close()
        self._files.clear()

    def __del__(self):
        files = getattr(self, "_files", None)
        if files:
            self.close()

    def _deployment_proxy_estimate(
        self,
        *,
        path: Path,
        start: int,
        task_state: torch.Tensor,
        route_target: torch.Tensor,
        mask: torch.Tensor,
        gate_index: torch.Tensor,
        timing: torch.Tensor,
    ) -> torch.Tensor:
        """Build a causal, noisy proxy for deployable VIO/IMU/gate estimates.

        Exact simulator values are used only to simulate noisy sensor outputs.
        The contract intentionally exposes noise, dropout, camera age, and
        uncertainty so this cannot be confused with the ground-truth
        ``state_estimate`` feedthrough currently stored in replay.

        Layout (32): VIO gate-state position/velocity/attitude (12), delayed
        active-gate position/normal/up (9), normalized acceleration (3), four
        uncertainty values, two valid bits, and two normalized ages.
        """

        if self.deployment_estimate_source != "proxy_v1":
            raise RuntimeError("deployment proxy requested while disabled")
        payload = f"{path.resolve()}:{start}:{self.deployment_estimate_seed}".encode()
        seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
        generator = torch.Generator().manual_seed(seed)
        length = task_state.shape[0]
        active_index = gate_index[..., 0] if gate_index.ndim == 2 else gate_index
        gate_changed = torch.zeros(length, dtype=torch.bool)
        gate_changed[1:] = active_index[1:] != active_index[:-1]

        state_noise_scale = task_state.new_tensor(
            [0.15 / 20.0] * 3 + [0.25 / 30.0] * 3 + [0.025] * 6
        )
        drift_scale = task_state.new_tensor(
            [0.004 / 20.0] * 3 + [0.008 / 30.0] * 3 + [0.0015] * 6
        )
        vio = task_state[:, :12].clone()
        vio = vio + torch.randn(vio.shape, generator=generator) * state_noise_scale
        drift = torch.randn(vio.shape, generator=generator) * drift_scale
        drift[gate_changed] = 0.0
        for index in range(1, length):
            if not gate_changed[index]:
                drift[index] += drift[index - 1]
        vio = vio + drift
        vio_valid = torch.rand(length, generator=generator) >= 0.03
        vio_valid[gate_changed] = True
        vio_age = torch.zeros(length)
        for index in range(length):
            if not vio_valid[index] and index and not gate_changed[index]:
                vio[index] = vio[index - 1]
                vio_age[index] = vio_age[index - 1] + 1.0

        visible = mask.flatten(1).sum(-1) > 0
        gate_valid = visible & (torch.rand(length, generator=generator) >= 0.10)
        delayed_gate = torch.zeros(length, 9)
        gate_age = torch.zeros(length)
        for index in range(length):
            camera_age = max(0, int(round(float(timing[index, 0]))))
            source = index - camera_age
            if source < 0 or active_index[source] != active_index[index]:
                gate_valid[index] = False
            if gate_valid[index]:
                delayed_gate[index] = route_target[source, 0, :9]
                gate_age[index] = float(camera_age)
            elif index and not gate_changed[index]:
                delayed_gate[index] = delayed_gate[index - 1]
                gate_age[index] = gate_age[index - 1] + 1.0
        gate_noise = torch.randn(delayed_gate.shape, generator=generator)
        gate_noise[:, :3] *= 0.18 / 20.0
        gate_noise[:, 3:9] *= 0.03
        delayed_gate = delayed_gate + gate_noise * gate_valid[:, None]
        normal = F.normalize(delayed_gate[:, 3:6], dim=-1, eps=1e-6)
        up = delayed_gate[:, 6:9]
        up = F.normalize(up - (up * normal).sum(-1, keepdim=True) * normal, dim=-1, eps=1e-6)
        delayed_gate[:, 3:6] = normal
        delayed_gate[:, 6:9] = up

        acceleration = torch.zeros(length, 3)
        acceleration[1:] = (task_state[1:, 3:6] - task_state[:-1, 3:6]) * 90.0
        acceleration[gate_changed] = 0.0
        acceleration += torch.randn(acceleration.shape, generator=generator) * 0.03
        acceleration.clamp_(-2.0, 2.0)

        uncertainty = torch.stack(
            (
                0.15 + 0.03 * vio_age,
                0.25 + 0.05 * vio_age,
                0.025 + 0.005 * vio_age,
                0.18 + 0.04 * gate_age,
            ),
            dim=-1,
        )
        valid = torch.stack((vio_valid.float(), gate_valid.float()), dim=-1)
        age = torch.stack((vio_age, gate_age), dim=-1).div(36.0).clamp(0.0, 2.0)
        return torch.cat((vio, delayed_gate, acceleration, uncertainty, valid, age), dim=-1)

    def sampling_weights(self, terminal_weight: float = 1.0) -> torch.Tensor:
        return torch.tensor(
            [terminal_weight if terminal else 1.0 for terminal in self.terminal_windows],
            dtype=torch.double,
        )

    def behavior_cloning_windows(
        self, *, history: int, action_horizon: int,
        require_solver_clean: bool = False,
        phase_kind: str = "tail",
        require_previous_target_clean: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Return windows with expert-only targets and their target phases.

        Invalid or injected commands are allowed in the observation history so
        the actor can learn expert recoveries after a bad trajectory. Only the
        CTBR actions receiving behavior-cloning supervision must be valid expert
        commands. Phase labels describe the supervised target chunk, rather
        than the entire history window.
        """

        history = int(history)
        action_horizon = int(action_horizon)
        target_offset = history - 1
        if history < 1 or action_horizon < 1:
            raise ValueError("history and action_horizon must be positive")
        if target_offset + action_horizon > self.sequence_length:
            raise ValueError("BC target chunk extends beyond the dataset window")

        if phase_kind not in {"tail", "gate"}:
            raise ValueError("phase_kind must be 'tail' or 'gate'")
        windows_by_path: dict[Path, list[tuple[int, int]]] = {}
        for index, (path, start) in enumerate(self.windows):
            windows_by_path.setdefault(path, []).append((index, start))

        eligible: list[int] = []
        target_phases: list[int] = []
        for path, windows in windows_by_path.items():
            with h5py.File(path, "r", swmr=True) as episode:
                length = int(episode["action/ctbr"].shape[0])
                controller_valid = np.asarray(
                    episode["controller/valid"][:]
                    if "controller/valid" in episode
                    else np.ones(length, np.bool_),
                    dtype=np.bool_,
                )
                intervention = np.asarray(
                    episode["controller/diagnostics/intervention"][:]
                    if "controller/diagnostics/intervention" in episode
                    else np.zeros(length, np.int8),
                    dtype=np.bool_,
                )
                solver_clean = np.asarray(
                    episode["controller/solver_status"][:] == 0
                    if require_solver_clean and "controller/solver_status" in episode
                    else np.ones(length, np.bool_),
                    dtype=np.bool_,
                )
            phases = (
                self.episode_gate_phases[path]
                if phase_kind == "gate"
                else self.episode_tail_phases[path]
            )
            for index, start in windows:
                target_start = start + target_offset
                target_stop = target_start + action_horizon
                clean_start = target_start - 1 if require_previous_target_clean else target_start
                if (
                    clean_start >= 0
                    and controller_valid[clean_start:target_stop].all()
                    and not intervention[clean_start:target_stop].any()
                    and solver_clean[clean_start:target_stop].all()
                ):
                    eligible.append(index)
                    target_phases.append(
                        int(phases[target_start:target_stop].max(initial=0))
                    )
        return eligible, target_phases

    def one_step_behavior_cloning_windows(
        self, *, history: int
    ) -> tuple[list[int], list[int]]:
        """Return causally valid single-action expert targets and gate phases."""

        target_offset = int(history) - 1
        if history < 1 or target_offset >= self.sequence_length:
            raise ValueError("history must select a target inside the dataset window")
        eligible: list[int] = []
        phases: list[int] = []
        by_path: dict[Path, list[tuple[int, int]]] = {}
        for index, (path, start) in enumerate(self.windows):
            by_path.setdefault(path, []).append((index, start))
        for path, windows in by_path.items():
            with h5py.File(path, "r", swmr=True) as episode:
                length = int(episode["action/ctbr"].shape[0])
                valid = np.asarray(
                    episode["controller/valid"][:]
                    if "controller/valid" in episode else np.ones(length, np.bool_),
                    dtype=np.bool_,
                )
                intervention = np.asarray(
                    episode["controller/diagnostics/intervention"][:]
                    if "controller/diagnostics/intervention" in episode
                    else np.zeros(length, np.bool_),
                    dtype=np.bool_,
                )
                solver_clean = np.asarray(
                    episode["controller/solver_status"][:] == 0
                    if "controller/solver_status" in episode
                    else np.ones(length, np.bool_),
                    dtype=np.bool_,
                )
            gate_phases = self.episode_gate_phases[path]
            for index, start in windows:
                target = start + target_offset
                if valid[target] and not intervention[target] and solver_clean[target]:
                    eligible.append(index)
                    phases.append(int(gate_phases[target]))
        return eligible, phases

    def __getitem__(self, index: int) -> dict:
        path, start = self.windows[index]
        stop = start + self.sequence_length
        with self._episode(path) as episode:
            # Do not recursively materialize the full observation tree for the
            # compact tokenizer/dynamics paths. Large privileged arrays made the
            # input pipeline CPU-bound and starved the GPU.
            observation: dict[str, torch.Tensor] = {}
            next_observation: dict[str, torch.Tensor] = {}
            if self.mode == "legacy":
                def collect_observation(name, item):
                    if isinstance(item, h5py.Dataset):
                        observation[name] = self._tensor(item[start:stop])
                        next_observation[name] = self._tensor(item[start + 1:stop + 1])
                episode["observation"].visititems(collect_observation)
            # The action tokenizer is deliberately handled before constructing
            # the legacy transition bundle. Older code read reward, continuation,
            # Dreamer actions, and other unused arrays for every action sample.
            if self.mode == "action_tokenizer":
                return self._action_tokenizer_sample(episode, start, stop)

            legacy = None
            if self.mode in {"legacy", "dynamics", "control"}:
                legacy = {
                    "obs": observation,
                    "next_obs": next_observation,
                    "action": self._tensor(episode["action/dreamer16"][start:stop]),
                    "ctbr_action": self._tensor(episode["action/ctbr"][start:stop]),
                    "reward": self._tensor(episode["reward/total"][start:stop]),
                    "continue": self._tensor(episode["continue"][start:stop]),
                    "discount": self._tensor(episode["discount"][start:stop]),
                    "is_first": self._tensor(episode["is_first"][start:stop]),
                    "is_last": self._tensor(episode["is_last"][start:stop]),
                    "is_terminal": self._tensor(episode["is_terminal"][start:stop]),
                }
            if self.mode == "legacy":
                return legacy

            mask_key = None
            if self.prefer_gatenet and "gatenet_mask" in episode["observation"]:
                mask_key = "gatenet_mask"
            elif "sim_gate_mask" in episode["observation"]:
                mask_key = "sim_gate_mask"
            elif "gate_mask" in episode["observation"]:
                mask_key = "gate_mask"
            elif "segmentation" in episode["observation"]:
                mask_key = "segmentation"
            if mask_key is None:
                raise ValueError(f"{path} has no segmentation mask for {self.mode} mode")

            mask = self._tensor(episode["observation"][mask_key][start:stop + 1]).float()
            if mask.ndim == 4:
                mask = mask.amax(-1)
            mask = mask[:, None] / 255.0
            if tuple(mask.shape[-2:]) != self.mask_size:
                mask = F.interpolate(mask, size=self.mask_size, mode="bilinear", align_corners=False)

            body_relative_plan = None
            if (
                self.route_source == "body_relative_gates"
                or self.route_target_source == "body_relative_gates"
            ):
                required_gate_fields = (
                    "position", "normal", "up", "size",
                    "enter_from_opposite_side", "mask",
                )
                missing_gate_fields = [
                    name for name in required_gate_fields
                    if f"observation/gates/{name}" not in episode
                ]
                if missing_gate_fields:
                    raise ValueError(
                        f"{path} lacks body-relative gate fields {missing_gate_fields}"
                    )
                body_relative_plan = torch.cat(
                    [
                        self._tensor(episode[f"observation/gates/{name}"][start:stop + 1]).float()
                        for name in ("position", "normal", "up", "size")
                    ]
                    + [
                        self._tensor(episode[f"observation/gates/{name}"][start:stop + 1])
                        .float().unsqueeze(-1)
                        for name in ("enter_from_opposite_side", "mask")
                    ],
                    dim=-1,
                )
            if self.route_source == "body_relative_gates":
                plan = body_relative_plan
            elif "observation/flight_plan/records" in episode:
                plan = self._tensor(episode["observation/flight_plan/records"][start:stop + 1]).float()
            else:
                plan = torch.zeros(self.sequence_length + 1, 3, 13)
            plan = plan.clone()
            plan[..., 0:3] /= 20.0
            plan[..., 9:11] /= 5.0
            route_target = (
                body_relative_plan.clone()
                if self.route_target_source == "body_relative_gates"
                else plan.clone()
            )
            if self.route_target_source == "body_relative_gates":
                route_target[..., 0:3] /= 20.0
                route_target[..., 9:11] /= 5.0
            if "observation/measured/body_rates" in episode:
                body_rates = self._tensor(
                    episode["observation/measured/body_rates"][start:stop + 1]
                ).float()
            else:
                body_rates = self._tensor(
                    episode["observation/state"][start:stop + 1, 10:13]
                ).float()
            body_rates = (body_rates / 6.0).clamp(-1, 1)
            motor_omega = self._tensor(
                episode["observation/measured/motor_omega"][start:stop + 1]
                if "observation/measured/motor_omega" in episode
                else episode["observation/motor_omega"][start:stop + 1]
                if "observation/motor_omega" in episode
                else np.zeros((self.sequence_length + 1, 4), np.float32)
            ).float().div(4000.0).clamp(0, 1.5)
            previous_action = self._tensor(
                episode["observation/previous_action"][start:stop + 1]
                if "observation/previous_action" in episode
                else np.zeros((self.sequence_length + 1, 4), np.float32)
            ).float()
            # Same normalization used by collection for actions.
            previous_action = torch.cat(
                [previous_action[:, :1] / 15.0 - 1.0, previous_action[:, 1:] / 6.0], -1
            ).clamp(-1, 1)
            vector = torch.cat([body_rates, motor_omega, previous_action, plan.flatten(1)], -1)
            proprio = torch.cat([body_rates, motor_omega, previous_action], dim=-1)
            task_state = self._tensor(
                episode["observation/privileged/gate_state"][start:stop + 1]
                if "observation/privileged/gate_state" in episode
                else episode["observation/task_state"][start:stop + 1]
                if "observation/task_state" in episode
                else np.zeros((self.sequence_length + 1, 19), np.float32)
            ).float()
            task_state = task_state.clone()
            task_state[..., 0:3] /= 20.0
            task_state[..., 3:6] /= 30.0
            task_state[..., 12:15] /= 6.0
            task_state[..., 15:19] /= 4000.0
            deployable_task_state = self._tensor(
                episode["observation/task_state"][start:stop + 1]
                if "observation/task_state" in episode
                else np.zeros((self.sequence_length + 1, 19), np.float32)
            ).float()
            deployable_task_state = deployable_task_state.clone()
            deployable_task_state[..., 0:3] /= 20.0
            deployable_task_state[..., 3:6] /= 30.0
            deployable_task_state[..., 12:15] /= 6.0
            deployable_task_state[..., 15:19] /= 4000.0
            state_estimate = self._tensor(
                episode["observation/state_estimate"][start:stop + 1]
                if "observation/state_estimate" in episode
                else episode["observation/state"][start:stop + 1]
            ).float()
            state_estimate_std = self._tensor(
                episode["observation/state_estimate_std"][start:stop + 1]
                if "observation/state_estimate_std" in episode
                else np.zeros((self.sequence_length + 1, 25), np.float32)
            ).float()
            gate_index = self._tensor(
                episode["observation/flight_plan/index"][start:stop + 1]
                if "observation/flight_plan/index" in episode
                else np.zeros((self.sequence_length + 1, 3), np.int64)
            ).long()
            privileged = {}
            privileged_sources = {
                "state": "observation/privileged/state",
                "sky_state": "observation/privileged/sky_state",
                "dynamics": "observation/privileged/dynamics",
                "actuator": "observation/privileged/actuator",
                "camera_extrinsics": "observation/privileged/camera_extrinsics",
                "camera_intrinsics": "observation/privileged/camera_intrinsics",
                "disturbance": "observation/privileged/disturbance",
                "progress": "observation/privileged/progress",
                "state_estimate": "observation/state_estimate",
                "state_estimate_std": "observation/state_estimate_std",
            }
            if self.include_privileged:
                for name, source in privileged_sources.items():
                    if source in episode:
                        privileged[name] = self._tensor(episode[source][start:stop + 1]).float()
            reset = torch.zeros(self.sequence_length + 1, dtype=torch.bool)
            reset[0] = bool(episode["is_first"][start])
            if "observation/valid/camera" in episode:
                def observation_field(group: str, name: str, *, boolean: bool = False):
                    value = self._tensor(episode[f"observation/{group}/{name}"][start:stop + 1])
                    return value.bool() if boolean else value.float()

                camera_valid = observation_field("valid", "camera", boolean=True)
                body_valid = observation_field("valid", "body_rates", boolean=True)
                motor_valid = observation_field("valid", "motor_omega", boolean=True)
                previous_valid = observation_field("valid", "previous_action", boolean=True)
                camera_time = self._tensor(
                    episode["observation/timestamp/camera"][start:stop + 1]
                ).double()
                camera_new = torch.ones_like(camera_valid)
                camera_new[1:] = camera_time[1:] != camera_time[:-1]
                if start > 0:
                    previous_camera_time = float(episode["observation/timestamp/camera"][start - 1])
                    camera_new[0] = bool(float(camera_time[0]) != previous_camera_time)
                sim_time = self._tensor(
                    episode["observation/timestamp/sim"][start:stop + 1]
                ).double()
                dt = torch.zeros_like(sim_time, dtype=torch.float32)
                dt[1:] = (sim_time[1:] - sim_time[:-1]).float() * 90.0
                if start > 0:
                    previous_sim_time = float(episode["observation/timestamp/sim"][start - 1])
                    dt[0] = float((sim_time[0] - previous_sim_time) * 90.0)
                elif len(sim_time) > 1:
                    dt[0] = dt[1]
                timing = torch.stack(
                    [
                        observation_field("age", "camera") * 90.0,
                        observation_field("age", "body_rates") * 90.0,
                        observation_field("age", "motor_omega") * 90.0,
                        observation_field("age", "previous_action") * 90.0,
                        camera_valid.float(),
                        body_valid.float(),
                        motor_valid.float(),
                        previous_valid.float(),
                        camera_new.float(),
                        dt,
                    ],
                    dim=-1,
                )
            else:
                camera_valid = torch.ones(self.sequence_length + 1, dtype=torch.bool)
                timing = torch.zeros(self.sequence_length + 1, 10)
                timing[:, 4:9] = 1.0

            if self.mode == "tokenizer":
                estimate = None
                if self.deployment_estimate_source == "proxy_v1":
                    estimate = self._deployment_proxy_estimate(
                        path=path,
                        start=start,
                        task_state=task_state[:-1],
                        route_target=route_target[:-1],
                        mask=mask[:-1],
                        gate_index=gate_index[:-1],
                        timing=timing[:-1],
                    )
                result = {
                    "mask": mask[:-1],
                    "vector": vector[:-1],
                    "proprio": proprio[:-1],
                    "route": plan[:-1],
                    # Training-only perception label. For deployment tokenizers,
                    # the encoder still receives the known track-only flight plan;
                    # simulator-relative gates may supervise the fused belief but
                    # are never passed into encode().
                    "route_target": route_target[:-1],
                    "task_state": task_state[:-1],
                    "gate_index": gate_index[:-1],
                    "timing": timing[:-1],
                    "valid": camera_valid[:-1],
                    "reset": reset[:-1],
                    # Episode-level collection regime is a training-only label.
                    # It is never consumed by the tokenizer encoder at inference.
                    "distribution_regime": torch.tensor(
                        self.episode_regimes[path], dtype=torch.long
                    ),
                    "tail_phase": torch.from_numpy(
                        self.episode_tail_phases[path][start:stop].astype(
                            np.int64, copy=False
                        )
                    ),
                }
                if estimate is not None:
                    result["estimate"] = estimate
                if self.include_actions:
                    applied = self._tensor(
                        episode["action/applied_ctbr"][start:stop]
                        if "action/applied_ctbr" in episode
                        else episode["action/ctbr"][start:stop]
                    ).float()
                    normalized_action = torch.cat(
                        [applied[:, :1] / 15.0 - 1.0, applied[:, 1:] / 6.0], -1
                    ).clamp(-1, 1)
                    transition_time = self._tensor(
                        episode["observation/timestamp/sim"][start:stop + 1]
                    ).double()
                    transition_dt = (transition_time[1:] - transition_time[:-1]).float()
                    if "action/command_timestamp" in episode:
                        command_time = self._tensor(
                            episode["action/command_timestamp"][start:stop]
                        ).double()
                        applied_time = self._tensor(
                            episode["action/applied_timestamp"][start:stop]
                        ).double()
                        action_delay = (command_time - applied_time).clamp_min(0).float()
                    else:
                        action_delay = torch.zeros_like(transition_dt)
                    result.update(
                        normalized_action=normalized_action,
                        action_timing=torch.stack(
                            [transition_dt * 90.0, action_delay * 90.0,
                             torch.ones_like(transition_dt)], -1
                        ),
                    )
                result.update({f"privileged_{name}": value[:-1] for name, value in privileged.items()})
                return result
            if "action/applied_ctbr" in episode:
                applied_ctbr = self._tensor(episode["action/applied_ctbr"][start:stop]).float()
                applied_normalized = torch.cat(
                    [applied_ctbr[:, :1] / 15.0 - 1.0, applied_ctbr[:, 1:] / 6.0], -1
                ).clamp(-1, 1)
            else:
                applied_normalized = legacy["action"][:, :4].float()
            if "observation/timestamp/sim" in episode:
                transition_time = self._tensor(
                    episode["observation/timestamp/sim"][start:stop + 1]
                ).double()
                transition_dt = (transition_time[1:] - transition_time[:-1]).float()
            else:
                transition_dt = torch.full((self.sequence_length,), 1.0 / 90.0)
            if "action/command_timestamp" in episode:
                command_time = self._tensor(episode["action/command_timestamp"][start:stop]).double()
                applied_time = self._tensor(episode["action/applied_timestamp"][start:stop]).double()
                action_delay = (command_time - applied_time).clamp_min(0).float()
            else:
                action_delay = torch.zeros_like(transition_dt)
            deployment_estimate = None
            if self.deployment_estimate_source == "proxy_v1":
                deployment_estimate = self._deployment_proxy_estimate(
                    path=path,
                    start=start,
                    task_state=task_state,
                    route_target=route_target,
                    mask=mask,
                    gate_index=gate_index,
                    timing=timing,
                )
            result = {
                "mask": mask,
                "vector": vector,
                "proprio": proprio,
                "route": plan,
                "task_state": task_state,
                "deployable_task_state": deployable_task_state,
                "state_estimate": state_estimate,
                "state_estimate_std": state_estimate_std,
                "gate_index": gate_index,
                "timing": timing,
                # Explicit causal policy signals. In particular,
                # previous_action is the command actually applied by the env,
                # not the expert target selected for the current transition.
                "previous_action": previous_action,
                "body_rates": body_rates,
                "motor_omega": motor_omega,
                "action": legacy["ctbr_action"].float(),
                "commanded_action": legacy["action"][:, :4].float(),
                "normalized_action": applied_normalized,
                "action_timing": torch.stack(
                    [transition_dt * 90.0, action_delay * 90.0, torch.ones_like(transition_dt)], -1
                ),
                "reward": legacy["reward"].float(),
                # Reward primitives used by state-derived model-based control.
                # Older replay has progress_delta but not centered gate quality;
                # a passed gate defaults to quality one for that data only.
                "progress_delta": self._tensor(
                    episode["reward/progress_delta"][start:stop]
                    if "reward/progress_delta" in episode
                    else np.zeros(self.sequence_length, np.float32)
                ).float(),
                "gate_passed": self._tensor(
                    episode["transition/gate_passed"][start:stop]
                    if "transition/gate_passed" in episode
                    else np.zeros(self.sequence_length, np.bool_)
                ).bool(),
                "gate_quality": self._tensor(
                    episode["reward/gate_quality"][start:stop]
                    if "reward/gate_quality" in episode
                    else episode["transition/gate_passed"][start:stop].astype(np.float32)
                    if "transition/gate_passed" in episode
                    else np.zeros(self.sequence_length, np.float32)
                ).float(),
                "continue": legacy["continue"].float(),
                "is_first": legacy["is_first"],
                "is_last": legacy["is_last"],
                "is_terminal": legacy["is_terminal"],
                "controller_valid": self._tensor(
                    episode["controller/valid"][start:stop]
                    if "controller/valid" in episode
                    else np.ones(self.sequence_length, np.bool_)
                ).bool(),
                # The hybrid online collector evaluates MPCC on every step,
                # including actor-controlled steps. This reference is the
                # counterfactual teacher action used for onset/recovery
                # distillation; commanded_action remains the action actually
                # selected by the hybrid policy.
                "expert_action": (
                    lambda reference: torch.cat(
                        [reference[:, :1] / 15.0 - 1.0, reference[:, 1:] / 6.0], -1
                    ).clamp(-1, 1)
                )(
                    self._tensor(
                        episode["controller/reference_action"][start:stop]
                        if "controller/reference_action" in episode
                        else episode["action/ctbr"][start:stop]
                    ).float()
                ),
                "intervention": self._tensor(
                    episode["controller/diagnostics/intervention"][start:stop]
                    if "controller/diagnostics/intervention" in episode
                    else np.zeros(self.sequence_length, np.int8)
                ).bool(),
                "tail_phase": torch.from_numpy(
                    self.episode_tail_phases[path][start:stop].astype(
                        np.int64, copy=False
                    )
                ),
                "gate_phase": torch.from_numpy(
                    self.episode_gate_phases[path][start:stop].astype(
                        np.int64, copy=False
                    )
                ),
                "valid": camera_valid[:-1],
                "reset": reset,
            }
            if deployment_estimate is not None:
                result["estimate"] = deployment_estimate
            result.update({f"privileged_{name}": value for name, value in privileged.items()})
            return result

    def _action_tokenizer_sample(
        self, episode: h5py.File, start: int, stop: int
    ) -> dict[str, torch.Tensor]:
        if "action/applied_ctbr" in episode:
            applied = self._tensor(episode["action/applied_ctbr"][start:stop]).float()
        else:
            applied = self._tensor(episode["action/ctbr"][start:stop]).float()
        applied_normalized = torch.cat(
            [applied[:, :1] / 15.0 - 1.0, applied[:, 1:] / 6.0], dim=-1
        ).clamp(-1, 1)
        if "action/normalized" in episode:
            commanded = self._tensor(episode["action/normalized"][start:stop]).float()
        else:
            commanded = self._tensor(episode["action/dreamer16"][start:stop, :4]).float()
        if "observation/timestamp/sim" in episode:
            sim_time = self._tensor(episode["observation/timestamp/sim"][start:stop + 1]).double()
            dt = (sim_time[1:] - sim_time[:-1]).float()
        else:
            dt = torch.full((self.sequence_length,), 1.0 / 90.0)
        if "action/command_timestamp" in episode:
            command_time = self._tensor(episode["action/command_timestamp"][start:stop]).double()
            applied_time = self._tensor(episode["action/applied_timestamp"][start:stop]).double()
            # Applied timestamps identify the older command currently at the
            # actuator, so command - applied is the causal action age.
            delay = (command_time - applied_time).clamp_min(0).float()
        else:
            delay = torch.zeros_like(dt)
        motor = self._tensor(
            episode["action/applied_motor_normalized"][start:stop]
            if "action/applied_motor_normalized" in episode
            else np.zeros((self.sequence_length, 4), np.float32)
        ).float()
        result = {
            "applied_action": applied_normalized,
            "commanded_action": commanded,
            "applied_motor": motor,
            "action_timing": torch.stack(
                [dt * 90.0, delay * 90.0, torch.ones_like(dt)], dim=-1
            ),
            "is_first": self._tensor(episode["is_first"][start:stop]),
            "is_last": self._tensor(episode["is_last"][start:stop]),
        }
        state_key = (
            "observation/privileged/gate_state"
            if "observation/privileged/gate_state" in episode
            else "observation/task_state"
            if "observation/task_state" in episode
            else None
        )
        if state_key is not None:
            task_state = self._tensor(episode[state_key][start:stop + 1]).float().clone()
            task_state[..., 0:3] /= 20.0
            task_state[..., 3:6] /= 30.0
            task_state[..., 12:15] /= 6.0
            task_state[..., 15:19] /= 4000.0
            # T+1 privileged labels condition only the training-time effect
            # decoder; the frozen action encoder still receives CTBR and timing.
            result["task_state"] = task_state
        if "observation/flight_plan/index" in episode:
            result["gate_index"] = self._tensor(
                episode["observation/flight_plan/index"][start:stop + 1]
            ).long()
        return result


class LocalityAwareBatchSampler(Sampler[list[int]]):
    """Shuffle every window while keeping each cache block to a few episodes.

    Global ``DataLoader(shuffle=True)`` is pathological for HDF5 episode files:
    almost every sample targets a different file and defeats a bounded handle
    cache. This sampler shuffles episode order and windows within every episode,
    mixes several episodes per batch, and drains cache-sized episode blocks
    before moving on. It preserves full-epoch coverage without global random I/O.
    """

    def __init__(
        self,
        dataset: DreamerSequenceDataset,
        batch_size: int,
        *,
        episodes_per_batch: int = 4,
        cache_size: int | None = None,
        windows_per_locality_block: int = 64,
        drop_last: bool = True,
        seed: int = 0,
        phase_fractions: tuple[float, ...] | list[float] | None = None,
        terminal_fraction: float = 0.0,
        eligible_indices: list[int] | tuple[int, ...] | None = None,
        phase_labels: list[int] | tuple[int, ...] | None = None,
        phase_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.episodes_per_batch = int(episodes_per_batch)
        self.cache_size = int(cache_size or dataset.max_open_files)
        self.windows_per_locality_block = int(windows_per_locality_block)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.phase_fractions = (
            tuple(float(value) for value in phase_fractions)
            if phase_fractions is not None else None
        )
        self.terminal_fraction = float(terminal_fraction)
        self.eligible_indices = (
            list(range(len(dataset)))
            if eligible_indices is None else [int(index) for index in eligible_indices]
        )
        self.phase_labels = (
            dataset.window_tail_phases
            if phase_labels is None else [int(value) for value in phase_labels]
        )
        self.phase_names = tuple(phase_names or TAIL_PHASE_NAMES)
        if len(self.phase_labels) != len(dataset):
            raise ValueError("phase_labels must contain one value per dataset window")
        if not self.eligible_indices:
            raise ValueError("eligible_indices cannot be empty")
        if min(self.eligible_indices) < 0 or max(self.eligible_indices) >= len(dataset):
            raise ValueError("eligible_indices contains an out-of-range dataset index")
        if not 0.0 <= self.terminal_fraction < 1.0:
            raise ValueError("terminal_fraction must be in [0, 1)")
        self.epoch = 0
        if self.batch_size < 1 or self.episodes_per_batch < 1:
            raise ValueError("batch_size and episodes_per_batch must be positive")
        if self.episodes_per_batch > self.cache_size:
            raise ValueError("episodes_per_batch cannot exceed the worker file-cache size")
        if self.windows_per_locality_block < 1:
            raise ValueError("windows_per_locality_block must be positive")
        self.indices_by_path: dict[Path, list[int]] = {}
        for index in self.eligible_indices:
            path, _ = dataset.windows[index]
            self.indices_by_path.setdefault(path, []).append(index)
        self.indices_by_phase: dict[int, list[int]] = {}
        for index in self.eligible_indices:
            phase = self.phase_labels[index]
            self.indices_by_phase.setdefault(int(phase), []).append(index)
        if self.phase_fractions is not None:
            if len(self.phase_fractions) != len(self.phase_names):
                raise ValueError("phase_fractions must contain one value per tail phase")
            if any(value < 0 for value in self.phase_fractions):
                raise ValueError("phase_fractions must be nonnegative")
            if abs(sum(self.phase_fractions) - 1.0) > 1e-6:
                raise ValueError("phase_fractions must sum to one")
            missing = [
                self.phase_names[index] for index, fraction in enumerate(self.phase_fractions)
                if fraction > 0 and not self.indices_by_phase.get(index)
            ]
            if missing:
                raise ValueError(f"no windows available for requested phases: {missing}")
        self.terminal_indices = [
            index for index in self.eligible_indices
            if getattr(dataset, "terminal_windows", [False] * len(dataset))[index]
        ]
        if self.terminal_fraction and not self.terminal_indices:
            raise ValueError("terminal_fraction requested but the dataset has no terminal windows")

    def __len__(self) -> int:
        length = len(self.dataset)
        return length // self.batch_size if self.drop_last else ceil(length / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        epoch_indices_by_path = self.indices_by_path
        if self.phase_fractions is not None:
            sampled: list[int] = []
            terminal_count = round(len(self.dataset) * self.terminal_fraction)
            if terminal_count:
                sampled.extend(rng.choices(self.terminal_indices, k=terminal_count))
            phase_total = len(self.dataset) - terminal_count
            remaining = phase_total
            for phase, fraction in enumerate(self.phase_fractions):
                count = (
                    remaining if phase == len(self.phase_fractions) - 1
                    else round(phase_total * fraction)
                )
                remaining -= count
                pool = self.indices_by_phase[phase]
                sampled.extend(rng.choices(pool, k=count))
            epoch_indices_by_path = {}
            for index in sampled:
                path = self.dataset.windows[index][0]
                epoch_indices_by_path.setdefault(path, []).append(index)
        paths = list(epoch_indices_by_path)
        rng.shuffle(paths)
        chunk_size = ceil(self.batch_size / self.episodes_per_batch)
        carry: list[int] = []
        for group_start in range(0, len(paths), self.cache_size):
            group = paths[group_start : group_start + self.cache_size]
            chunks: dict[Path, list[list[int]]] = {}
            for path in group:
                indices = list(epoch_indices_by_path[path])
                locality_blocks = [
                    indices[start : start + self.windows_per_locality_block]
                    for start in range(0, len(indices), self.windows_per_locality_block)
                ]
                rng.shuffle(locality_blocks)
                # Keep adjacent overlapping windows together inside a shuffled
                # block so the HDF5 raw-chunk cache can reuse the same pages.
                chunks[path] = [
                    block[start : start + chunk_size]
                    for block in locality_blocks
                    for start in range(0, len(block), chunk_size)
                ]
            while chunks:
                active = list(chunks)
                rng.shuffle(active)
                for path in active[: self.episodes_per_batch]:
                    carry.extend(chunks[path].pop())
                    if not chunks[path]:
                        del chunks[path]
                while len(carry) >= self.batch_size:
                    yield carry[: self.batch_size]
                    del carry[: self.batch_size]
        if carry and not self.drop_last:
            yield carry
