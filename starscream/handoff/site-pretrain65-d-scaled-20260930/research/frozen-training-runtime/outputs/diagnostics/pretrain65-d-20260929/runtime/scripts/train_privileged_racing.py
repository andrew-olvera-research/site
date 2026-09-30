#!/usr/bin/env python3
"""Sequential-stage trainer for the privileged causal racing baseline.

Stages share one causal continuous-history policy contract:

* ``bc`` fits MPCC commands and a one-step task-state dynamics auxiliary;
* ``dagger`` gathers corrective labels with parallel isolated MPCC agents;
* ``ppo`` updates the complete warm-started recurrent actor with exact
  tanh-Gaussian likelihoods in a single-track dense-reward curriculum.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import ctypes
import gc
import hashlib
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import math
import multiprocessing as mp
from multiprocessing.connection import wait as wait_connections
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

# Window collectors import shared helpers through the scripts namespace.
# Direct CLI execution otherwise puts only /workspace/scripts on sys.path;
# imported smoke harnesses already had the root and masked this launch bug.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from starscream.agile_generalization import (
    DaggerLearningProgressBank,
    GREEN2026_OBSERVATION_CONTRACT,
    LEGACY_OBSERVATION_CONTRACT,
    SpearmanTaskSwitcher,
    observation_contract_action_slice,
)
from starscream.aggrevate import (
    AggreVaTeConfig,
    AggreVaTeImitationBatch,
    AggreVaTeImitationReplayBuffer,
    AggreVaTeQCritic,
    AggreVaTeQueryBatch,
    AggreVaTeReplayBuffer,
    CostNormalizer,
    aggrevate_actor_loss,
    aggrevate_critic_loss,
    discounted_cost_to_go,
    minimum_time_terminal_cost,
)
from starscream.checkpoint_manager import (
    CheckpointManager, capture_rng_state, restore_rng_state,
)
from starscream.dagger_collection import collection_config, track_collection_config, collection_beta, CollectionControl
from starscream.dagger_quality import (FIELDS as QUALITY_FIELDS, WIDTH as QUALITY_WIDTH,
    quality_config, track_quality_config, planned_plane_crossing, TrajectoryGuard, QualityReplayPlan, quality_subsample,
    mark_precursors, replay_quality_config, online_minibatch_quality_config, ReferenceGateEvents)
from starscream.dagger_replay import DaggerReplayStore
from starscream.behavior_feedback import sampler_feedback, update_behavior_sampler
from starscream.dagger_row_buffer import NumericRowBuffer
from starscream.dagger_throughput import HierarchicalReplayPlan, prefetched_batches, hardware_overlay
from starscream.update_transport import UpdateBatchTransfer, scalar_lists_to_host
from starscream.inference_graph import inference_callable
from starscream.env import FlightmareEnv
from starscream.env.real_course_suite import load_active_real_course_suite
from starscream.env.procedural_tracks import manifest_track_paths, read_manifest
from starscream.env.tracks import load_track, quaternion_matrix
from starscream.fpo import (
    fpo_plus_plus_loss,
    sample_cfm_conditions,
)
from starscream.mpcc import (
    MPCCConfig, MPCCController, RacingLinePlanner, RacingLinePlannerConfig,
)
from starscream.online_manifold_rl import OnlineManifestCurriculum
from starscream.env.racing_manifold.task_selector import completion_gate_count
from starscream.privileged_racing import (
    TASK_DIM, FeatureNormalizer, PrivilegedMLPPolicy, SquashedGaussian,
    PrivilegedCourseValue, PrivilegedTransformerValue, PrivilegedValue,
    checkpoint_payload,
    episode_split, flatten_metrics,
    initialize_scratch_policy, load_one_step_data, load_policy_checkpoint,
    normalized_to_ctbr,
    observation_features, privileged_feature_dim, resolve_ranked_checkpoint,
    route_gates_from_feature_dim, sequence_indices,
    ctbr_to_normalized,
)
from starscream.racing_curriculum import (
    CurriculumRaceReward, CurriculumRewardConfig,
    Green2026StateRaceReward, Green2026StateRewardConfig,
    RacingCurriculumStage, sample_curriculum_spawn,
)
from starscream.wandb import init_wandb


DEFAULT_CONTROL_HZ = 90.0


def release_process_heap() -> bool:
    """Collect unreachable objects and return free glibc arenas to the OS."""

    gc.collect()
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is None:
        return False
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return bool(trim(0))


def configured_control_hz(settings: Mapping[str, Any]) -> float:
    """Return the validated outer-policy/action update frequency."""

    frequency = float(settings.get("control_hz", DEFAULT_CONTROL_HZ))
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("control_hz must be finite and positive")
    return frequency


def configured_control_dt(settings: Mapping[str, Any]) -> float:
    """Return the simulator integration interval for one policy action."""

    return 1.0 / configured_control_hz(settings)


def _scale_mpcc_slew_limits_for_control_rate(
    values: Mapping[str, Any], settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep MPCC command slew per physical second invariant across rates.

    ``MPCCConfig`` expresses its final command envelope per controller query.
    Reusing those numerical limits at a higher query frequency silently gives
    the teacher more physical command authority.  Frequency experiments opt in
    to this conversion after every family/frontier override has been resolved.
    """

    result = dict(values)
    if not bool(settings.get("mpcc_preserve_physical_slew_rate", False)):
        return result
    reference_hz = float(settings.get(
        "control_reference_hz", DEFAULT_CONTROL_HZ
    ))
    if not np.isfinite(reference_hz) or reference_hz <= 0.0:
        raise ValueError("control_reference_hz must be finite and positive")
    ratio = reference_hz / configured_control_hz(settings)
    defaults = MPCCConfig()
    for name in ("collective_slew_limit", "body_rate_slew_limit"):
        result[name] = float(result.get(name, getattr(defaults, name))) * ratio
    return result


def _normalize_dynamics_delta_for_control_rate(
    value: np.ndarray, settings: Mapping[str, Any],
) -> np.ndarray:
    """Express a one-tick dynamics delta in the reference-rate units."""

    result = np.asarray(value, np.float32)
    if not bool(settings.get("dynamics_preserve_reference_rate", False)):
        return result
    reference_hz = float(settings.get(
        "control_reference_hz", DEFAULT_CONTROL_HZ
    ))
    if not np.isfinite(reference_hz) or reference_hz <= 0.0:
        raise ValueError("control_reference_hz must be finite and positive")
    return (result * (configured_control_hz(settings) / reference_hz)).astype(
        np.float32
    )


def _terminate_worker_with_parent() -> None:
    """Terminate a simulator worker if its trainer disappears on Linux.

    Multiprocessing cleanup cannot run when the trainer receives SIGKILL (for
    example from the OOM killer). Without this parent-death signal, spawned
    Flightmare/ACADOS workers are reparented to PID 1 and retain their solver
    memory indefinitely. The parent check closes the small installation race.
    """

    if os.name != "posix":
        return
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _configure_dagger_worker_threads(settings: Mapping[str, Any]) -> Any:
    """Prevent each MPCC process from creating a full CPU thread pool.

    Spawned workers import PyTorch and NumPy before entering their worker
    function.  On the current host that means eight OpenMP and sixteen
    OpenBLAS threads *per worker* unless the loaded libraries are explicitly
    limited.  The rollout hot path is one sequential simulator/solver per
    process, so nested pools add contention and memory without parallelizing a
    trajectory.
    """

    threads = int(settings.get("dagger_worker_cpu_threads", 1))
    if threads < 1:
        raise ValueError("dagger_worker_cpu_threads must be positive")
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        # A third-party import may have initialized the inter-op pool already;
        # threadpoolctl below still limits the native numerical libraries.
        pass
    try:
        from threadpoolctl import threadpool_limits

        # Keep the controller alive for the worker lifetime.  Restoring the
        # previous limits while MPCC is active would reintroduce oversubscription.
        return threadpool_limits(limits=threads)
    except ImportError:
        return None


def _memory_snapshot() -> dict[str, float]:
    from starscream.memory_pressure import memory_pressure_snapshot
    return memory_pressure_snapshot()


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and value.get("__replace__") is True:
            result[key] = copy.deepcopy({
                child_key: child_value
                for child_key, child_value in value.items()
                if child_key != "__replace__"
            })
        elif isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError(f"cyclic config inheritance at {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("inherits", None)
    if parent is None:
        return raw
    parent_path = (path.parent / str(parent)).resolve()
    return _merge(load_config(parent_path, seen), raw)


def stage_config(config: dict[str, Any], stage: str) -> dict[str, Any]:
    result = copy.deepcopy(config)
    settings = result[stage]
    parent_stage = settings.pop("inherits_stage", None)
    if parent_stage is not None:
        if str(parent_stage) not in result:
            raise ValueError(
                f"stage {stage!r} inherits missing stage {parent_stage!r}"
            )
        settings = _merge(result[str(parent_stage)], settings)
        result[stage] = settings
    run_name = str(settings["run_name"])
    result["checkpoint"] = {
        "run_name": run_name,
        "monitor": str(settings.get("monitor", "selection_score")),
        "mode": str(settings.get("monitor_mode", "max")),
        "top_k": int(settings.get("top_k", 3)),
    }
    result["wandb"] = _merge(
        dict(result.get("wandb", {})),
        {"run_name": run_name, "tags": list(settings.get("tags", []))},
    )
    return result


def parse_args() -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("bc", "dagger", "aggrevate", "ppo", "fpo"), required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rollout-envs", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--hardware-profile", type=Path, default=None,
                        help="DAgger/PPO YAML overlay restricted to scheduling/performance keys")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.hardware_profile is not None:
        if args.stage not in {"dagger", "ppo"}:
            raise ValueError("hardware profiles apply only to DAgger and PPO")
        with args.hardware_profile.open() as handle:
            overrides = yaml.safe_load(handle)
        if not isinstance(overrides, dict):
            raise ValueError("hardware profile must be a flat mapping")
        config[args.stage] = hardware_overlay(config[args.stage], overrides, stage=args.stage)
    if args.smoke:
        config = copy.deepcopy(config)
        config[args.stage]["run_name"] = f"privileged-racing-{args.stage}-smoke"
        config[args.stage]["evaluation_episodes"] = 2
        if args.stage == "bc":
            config[args.stage].update(steps=2, validation_interval=1, batch_size=16)
        elif args.stage == "dagger":
            configured_checkpoint = config[args.stage].get("initial_checkpoint", "")
            if configured_checkpoint is None:
                smoke_checkpoint = None
            else:
                configured_checkpoint = str(configured_checkpoint)
                smoke_checkpoint = (
                    configured_checkpoint
                    if configured_checkpoint and Path(configured_checkpoint).exists()
                    else "/tmp/starscream-privileged-bc-smoke/checkpoints/"
                    "privileged-racing-bc-smoke"
                )
            smoke_fractions = tuple(config[args.stage].get(
                "dagger_teacher_speed_fractions", (1.0,)
            ))
            config[args.stage].update(
                rounds=1, episodes_per_round=2 * len(smoke_fractions), rollout_envs=2,
                evaluation_workers=2, updates_per_round=2, batch_size=16,
                max_steps=18,
                track_limit=2,
                online_fraction=1.0,
                online_replay_capacity=1000,
                dagger_permanent_expert_rounds=0,
                dagger_permanent_expert_fraction=0.0,
                dagger_permanent_expert_required_families=[],
                teacher_beta_start=1.0,
                teacher_beta_end=1.0,
                teacher_beta_schedule_rounds=2,
                resume_checkpoint=None,
                dagger_resume_expert_refresh_rounds=0,
                dagger_require_successful_episodes=False,
                dagger_minimum_successful_episodes_per_track=0,
                dagger_coverage_retry_episodes_per_track=0,
                reporting_evaluation_curriculum=None,
                frontier_evaluation_speed_fractions=[],
                mpcc_build_root="/tmp/starscream-acados-dagger-smoke",
                initial_checkpoint=smoke_checkpoint,
            )
            smoke_manifest = config[args.stage].get("track_manifest")
            if smoke_manifest is not None and Path(str(smoke_manifest)).is_file():
                smoke_families = sorted({
                    str(record["family"])
                    for record in read_manifest(smoke_manifest).get("records", ())
                })
                config[args.stage].update(
                    dagger_transition_start_gate_indices={
                        family: [0] for family in smoke_families
                    },
                    dagger_transition_start_episode_fraction=1.0,
                    dagger_transition_start_mode_weights={"expert_prefix": 1.0},
                    dagger_hierarchical_replay_sampling=True,
                )
            config[args.stage]["curriculum"]["max_steps"] = 18
            config[args.stage]["curriculum"]["track_limit"] = 2
            evaluation_curriculum = config[args.stage].get("evaluation_curriculum")
            if isinstance(evaluation_curriculum, Mapping):
                evaluation_curriculum["max_steps"] = 18
                evaluation_curriculum["track_limit"] = 2
            # The smoke deliberately limits tracks; retain dynamic sampler
            # validation, but restrict its base weights to that same subset.
            if smoke_manifest is not None and Path(str(smoke_manifest)).is_file():
                smoke_family_by_track = manifest_track_values(
                    smoke_manifest, "family", cast=str,
                )
                smoke_active_families = {
                    smoke_family_by_track[str(Path(track).resolve())]
                    for track in configured_tracks(config[args.stage])
                }
                if config[args.stage].get("track_sampling_family_weights"):
                    config[args.stage]["track_sampling_family_weights"] = {
                        name: weight for name, weight in
                        config[args.stage]["track_sampling_family_weights"].items()
                        if name in smoke_active_families
                    }
        elif args.stage == "aggrevate":
            config[args.stage].update(
                rounds=1, queries_per_round=2, rollout_envs=1,
                evaluation_episodes=2, evaluation_workers=1,
                critic_updates_per_round=2, actor_updates_per_round=2,
                critic_batch_size=4, actor_batch_size=4,
            )
            algorithm = dict(config[args.stage].get("algorithm", {}))
            algorithm.update(query_horizon=8, query_min_step=0, query_max_step=4)
            config[args.stage]["algorithm"] = algorithm
            smoke_curriculum = copy.deepcopy(config["dagger"]["curriculum"])
            smoke_curriculum.update(max_steps=8)
            config[args.stage]["curriculum"] = smoke_curriculum
            smoke_evaluation = copy.deepcopy(
                config["dagger"].get(
                    "evaluation_curriculum", config["dagger"]["curriculum"]
                )
            )
            smoke_evaluation.update(max_steps=8)
            config[args.stage]["evaluation_curriculum"] = smoke_evaluation
        else:
            configured_checkpoint = str(config[args.stage].get("initial_checkpoint", ""))
            config[args.stage].update(
                cycles=1, episodes_per_cycle=2, rollout_envs=2,
                evaluation_workers=2, reporting_evaluation_episodes=2,
                actor_epochs=1, critic_epochs=1,
                minibatch_size=8,
                initial_checkpoint=(
                    configured_checkpoint
                    if configured_checkpoint and Path(configured_checkpoint).exists()
                    else "/tmp/starscream-privileged-dagger-smoke/checkpoints/"
                    "privileged-racing-dagger-smoke/latest.pt"
                ),
            )
            config[args.stage]["curriculum"] = [
                _merge(config[args.stage]["curriculum"][0], {
                    "max_steps": 18, "minimum_environment_steps": 999999,
                })
            ]
            if config[args.stage].get("checkpoint_rank_curriculum_stages") is not None:
                config[args.stage]["checkpoint_rank_curriculum_stages"] = [0]
            evaluation_curriculum = config[args.stage].get("evaluation_curriculum")
            if isinstance(evaluation_curriculum, list):
                config[args.stage]["evaluation_curriculum"] = [
                    _merge(evaluation_curriculum[0], {"max_steps": 18})
                ]
            reporting_curriculum = config[args.stage].get(
                "reporting_evaluation_curriculum"
            )
            if isinstance(reporting_curriculum, list):
                config[args.stage]["reporting_evaluation_curriculum"] = [
                    _merge(reporting_curriculum[0], {"max_steps": 18})
                ]
            elif isinstance(reporting_curriculum, Mapping):
                config[args.stage]["reporting_evaluation_curriculum"] = _merge(
                    reporting_curriculum, {"max_steps": 18}
                )
        config["wandb"] = {"enabled": False}
        config["output_root"] = f"/tmp/starscream-privileged-{args.stage}-smoke"
    settings = config[args.stage]
    if args.rollout_envs is not None:
        settings["rollout_envs"] = args.rollout_envs
    if args.episodes is not None:
        episode_key = (
            "queries_per_round" if args.stage == "aggrevate"
            else "episodes_per_round" if args.stage == "dagger"
            else "episodes_per_cycle"
        )
        if args.stage != "bc":
            settings[episode_key] = args.episodes
    if args.max_steps is not None and args.stage != "bc":
        curriculum = settings["curriculum"]
        if isinstance(curriculum, list):
            for stage in curriculum:
                stage["max_steps"] = args.max_steps
        else:
            curriculum["max_steps"] = args.max_steps
    if args.cycles is not None and args.stage in {"ppo", "fpo"}:
        settings["cycles"] = args.cycles
    if args.evaluation_episodes is not None and args.stage != "bc":
        settings["evaluation_episodes"] = args.evaluation_episodes
    if args.run_name is not None:
        settings["run_name"] = args.run_name
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.disable_wandb:
        # Disable the remote connection without discarding explicitly requested
        # local audit events (used by preflight and coordinator runs).
        config["wandb"] = {**config.get("wandb", {}), "enabled": False}
    return args, stage_config(config, args.stage)


class CausalHistory:
    def __init__(self, context_steps: int) -> None:
        self.context_steps = int(context_steps)
        if self.context_steps < 1:
            raise ValueError("context_steps must be positive")
        self._values: np.ndarray | None = None

    def reset(
        self, observation: Mapping[str, Any],
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        feature = (
            observation_features(observation)
            if settings is None
            else ppo_observation_features(observation, settings)
        )
        self.reset_feature(feature)

    def reset_feature(self, feature: np.ndarray) -> None:
        feature = np.asarray(feature, np.float32)
        if feature.ndim != 1:
            raise ValueError("causal history features must be one-dimensional")
        if (
            self._values is None
            or self._values.shape != (self.context_steps, len(feature))
        ):
            self._values = np.empty(
                (self.context_steps, len(feature)), dtype=np.float32
            )
        self._values[:] = feature

    def append(
        self, observation: Mapping[str, Any],
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        feature = (
            observation_features(observation)
            if settings is None
            else ppo_observation_features(observation, settings)
        )
        self.append_feature(feature)

    def append_feature(self, feature: np.ndarray) -> None:
        feature = np.asarray(feature, np.float32)
        if self._values is None:
            raise RuntimeError("causal history is not initialized")
        if feature.shape != self._values.shape[1:]:
            raise ValueError(
                "causal history feature shape changed: "
                f"expected={self._values.shape[1:]} actual={feature.shape}"
            )
        # Context is only 18x103 in current policies.  A contiguous shift is
        # cheaper than allocating 18 Python arrays plus np.stack every control
        # step, and makes array() a zero-allocation view.
        self._values[:-1] = self._values[1:]
        self._values[-1] = feature

    def array(self) -> np.ndarray:
        if self._values is None:
            raise RuntimeError("causal history is not initialized")
        return self._values


def ppo_normalized_to_ctbr(
    action: np.ndarray, settings: Mapping[str, Any],
) -> np.ndarray:
    """Decode the configured normalized CTBR action contract.

    Legacy PPO runs may append a cubic authority tail to the original linear
    0--30 m/s^2 map. New high-authority imitation runs use an invertible
    logit/sigmoid collective map: hover and the old warm-start manifold remain
    near their former coordinates, while 30--40 m/s^2 no longer occupies a
    vanishingly small saturated tanh tail.
    """

    normalized = np.asarray(action, np.float32).clip(-1.0, 1.0)
    mapping = str(settings.get("collective_action_mapping", "legacy_cubic_tail"))
    if mapping == "logit_sigmoid_v1":
        maximum = float(settings.get("collective_action_maximum", 40.0))
        reference = float(settings.get("collective_action_reference_thrust", 15.0))
        scale = float(settings.get("collective_action_logit_scale", 1.6))
        if not 0.0 < reference < maximum or scale <= 0.0:
            raise ValueError("invalid logit-sigmoid collective-action mapping")
        clipped = np.clip(normalized[0], -1.0 + 1.0e-6, 1.0 - 1.0e-6)
        latent = np.arctanh(clipped)
        bias = np.log(reference / (maximum - reference))
        collective = maximum / (1.0 + np.exp(-(scale * latent + bias)))
        command = normalized_to_ctbr(normalized)
        command[0] = collective
        return command.astype(np.float32)
    if mapping != "legacy_cubic_tail":
        raise ValueError(f"unsupported collective-action mapping {mapping!r}")
    command = normalized_to_ctbr(normalized)
    maximum = float(settings.get("ppo_maximum_collective_thrust", 30.0))
    if maximum <= 30.0:
        return command
    threshold = float(settings.get("ppo_collective_authority_threshold", 0.80))
    power = float(settings.get("ppo_collective_authority_power", 3.0))
    if not 0.0 <= threshold < 1.0 or power < 1.0 or not np.isfinite(maximum):
        raise ValueError("invalid PPO collective-authority warp")
    fraction = float(np.clip(
        (normalized[0] - threshold) / (1.0 - threshold), 0.0, 1.0
    ))
    command[0] += (maximum - 30.0) * fraction**power
    return command.astype(np.float32)


def ppo_ctbr_to_normalized(
    action: np.ndarray, settings: Mapping[str, Any],
) -> np.ndarray:
    """Invert :func:`ppo_normalized_to_ctbr` for causal previous-action input."""

    physical = np.asarray(action, np.float32)
    mapping = str(settings.get("collective_action_mapping", "legacy_cubic_tail"))
    if mapping == "logit_sigmoid_v1":
        maximum = float(settings.get("collective_action_maximum", 40.0))
        reference = float(settings.get("collective_action_reference_thrust", 15.0))
        scale = float(settings.get("collective_action_logit_scale", 1.6))
        if not 0.0 < reference < maximum or scale <= 0.0:
            raise ValueError("invalid logit-sigmoid collective-action mapping")
        ratio = np.clip(float(physical[0]) / maximum, 1.0e-6, 1.0 - 1.0e-6)
        bias = np.log(reference / (maximum - reference))
        latent = (np.log(ratio / (1.0 - ratio)) - bias) / scale
        normalized = ctbr_to_normalized(physical)
        normalized[0] = np.tanh(latent)
        return normalized.astype(np.float32)
    if mapping != "legacy_cubic_tail":
        raise ValueError(f"unsupported collective-action mapping {mapping!r}")
    maximum = float(settings.get("ppo_maximum_collective_thrust", 30.0))
    threshold = float(settings.get("ppo_collective_authority_threshold", 0.80))
    threshold_command = 15.0 * (threshold + 1.0)
    if maximum <= 30.0 or float(physical[0]) <= threshold_command:
        return ctbr_to_normalized(physical)
    low, high = threshold, 1.0
    for _ in range(24):
        middle = 0.5 * (low + high)
        candidate = ppo_normalized_to_ctbr(
            np.asarray([middle, 0.0, 0.0, 0.0], np.float32), settings
        )[0]
        if candidate < physical[0]:
            low = middle
        else:
            high = middle
    normalized = ctbr_to_normalized(physical)
    normalized[0] = 0.5 * (low + high)
    return normalized.astype(np.float32)


def action_feature_slice(settings: Mapping[str, Any]) -> slice:
    start = TASK_DIM + int(settings.get("route_gate_count", 3)) * 13
    return observation_contract_action_slice(
        str(settings.get(
            "observation_contract", LEGACY_OBSERVATION_CONTRACT
        )),
        legacy_start=start,
    )


def ppo_observation_features(
    observation: Mapping[str, Any], settings: Mapping[str, Any],
) -> np.ndarray:
    """Keep previous-action features in the checkpoint's normalized coordinates."""

    if settings.get('observation_contract') in {'starscream_world_route_v2', 'starscream_world_displacement_v3'}:
        if settings.get('policy_state_source', 'truth') != 'truth':
            raise ValueError('world teacher contract requires privileged truth')
        if (settings.get('flight_plan_randomization') or {}).get('enabled', False):
            raise ValueError('world teacher requires consistent unrandomized route records')

    previous_action_mapping = str(settings.get(
        "previous_action_feature_mapping", "legacy_linear"
    ))
    if previous_action_mapping == "legacy_linear":
        action_encoder = ctbr_to_normalized
    elif previous_action_mapping == "configured_action_contract":
        action_encoder = lambda action: ppo_ctbr_to_normalized(action, settings)
    else:
        raise ValueError(
            "previous_action_feature_mapping must be legacy_linear or "
            "configured_action_contract"
        )
    feature = observation_features(
        observation,
        route_gates=int(settings.get("route_gate_count", 3)),
        observation_contract=str(settings.get(
            "observation_contract", LEGACY_OBSERVATION_CONTRACT
        )),
        action_encoder=action_encoder,
    )
    return feature


def dynamics_observation_task(observation, settings):
    """Use the actor's fixed-world state only for the new teacher contract."""
    if settings.get('observation_contract') in {'starscream_world_route_v2', 'starscream_world_displacement_v3'}:
        return ppo_observation_features(observation, settings)[:TASK_DIM]
    return np.asarray(observation['task_state'], np.float32)


def remap_offline_action_contract(data: Any, settings: Mapping[str, Any]) -> None:
    """Move legacy offline labels/features into the configured action coordinates."""

    if str(settings.get("collective_action_mapping", "legacy_cubic_tail")) == "legacy_cubic_tail":
        return
    contract = str(getattr(
        data, "observation_contract", settings.get(
            "observation_contract", LEGACY_OBSERVATION_CONTRACT
        )
    ))
    if contract == LEGACY_OBSERVATION_CONTRACT:
        route_gates = route_gates_from_feature_dim(data.features.shape[1])
        action_offset = TASK_DIM + route_gates * 13
    else:
        action_offset = int(action_feature_slice(settings).start)
    for values in (data.actions, data.previous_actions):
        physical = normalized_to_ctbr(values)
        values[:] = np.stack([
            ppo_ctbr_to_normalized(item, settings) for item in physical
        ]).astype(np.float32)
    if getattr(data, "action_chunks", None) is not None and data.action_chunks.size:
        flat = data.action_chunks.reshape(-1, 4)
        physical = normalized_to_ctbr(flat)
        flat[:] = np.stack([
            ppo_ctbr_to_normalized(item, settings) for item in physical
        ]).astype(np.float32)
    if settings.get('previous_action_feature_mapping', 'legacy_linear') == 'configured_action_contract':
        data.features[:, action_offset:action_offset + 4] = data.previous_actions


def action_contract_metadata(settings: Mapping[str, Any]) -> dict[str, Any]:
    mapping = str(settings.get("collective_action_mapping", "legacy_cubic_tail"))
    metadata: dict[str, Any] = {"collective_mapping": mapping}
    if mapping == "logit_sigmoid_v1":
        metadata.update({
            "collective_maximum": float(settings.get("collective_action_maximum", 40.0)),
            "collective_reference_thrust": float(
                settings.get("collective_action_reference_thrust", 15.0)
            ),
            "collective_logit_scale": float(
                settings.get("collective_action_logit_scale", 1.6)
            ),
        })
    return metadata


def normalized_ctbr_tensor(
    action: torch.Tensor, settings: Mapping[str, Any],
) -> torch.Tensor:
    """Differentiably decode normalized actions for physically scaled losses."""

    mapping = str(settings.get("collective_action_mapping", "legacy_cubic_tail"))
    # This path runs inside bfloat16 autocast during DAgger. In bf16,
    # ``1 - 1e-6`` rounds back to exactly one and atanh(1) poisons the actor
    # with infinities. Decode in fp32 and use a representable safety margin.
    clipped = action.float().clamp(-1.0 + 1.0e-4, 1.0 - 1.0e-4)
    if mapping == "logit_sigmoid_v1":
        maximum = float(settings.get("collective_action_maximum", 40.0))
        reference = float(settings.get("collective_action_reference_thrust", 15.0))
        scale = float(settings.get("collective_action_logit_scale", 1.6))
        bias = math.log(reference / (maximum - reference))
        collective = maximum * torch.sigmoid(
            scale * torch.atanh(clipped[..., 0]) + bias
        )
    elif mapping == "legacy_cubic_tail":
        collective = 15.0 * (clipped[..., 0] + 1.0)
    else:
        raise ValueError(f"unsupported collective-action mapping {mapping!r}")
    return torch.cat([collective.unsqueeze(-1), 6.0 * clipped[..., 1:4]], dim=-1)


def mpcc_speed_frontier_profile(
    settings: Mapping[str, Any], family: str,
) -> str:
    """Resolve a family-qualified MPCC authority envelope."""

    configured = settings.get("mpcc_family_speed_frontier_profiles", {})
    default = str(settings.get("mpcc_speed_frontier_profile", ""))
    if family and isinstance(configured, Mapping):
        return str(configured.get(family, default))
    return default


def make_gate_reference(env, settings):
    """Build the exact MPCC reference geometry without constructing a solver."""
    manifest_path = settings.get("track_manifest") or settings.get(
        "mpcc_track_manifest"
    )
    match: Mapping[str, Any] | None = None
    if manifest_path is not None:
        payload = read_manifest(str(manifest_path))
        match = next(
            (item for item in payload["records"] if item["name"] == env.track.name),
            None,
        )
    family = str(match.get("family", "")) if match is not None else ""
    planner_values = dict(settings.get("mpcc_planner_config", {}))
    family_planner_configs = settings.get("mpcc_family_planner_configs", {})
    if family and isinstance(family_planner_configs, Mapping):
        family_planner_override = family_planner_configs.get(family)
        if isinstance(family_planner_override, Mapping):
            planner_values = _merge(planner_values, family_planner_override)
    if bool(settings.get("mpcc_use_manifest_teacher_profile", False)) and match:
        selected = dict((match.get("qualification") or {}).get("selected") or {})
        teacher_profile = str(selected.get("teacher_profile", ""))
        profile_configs = settings.get("mpcc_manifest_teacher_profile_planner_configs", {})
        profile_override = (
            profile_configs.get(teacher_profile)
            if isinstance(profile_configs, Mapping) else None
        )
        if not teacher_profile or not isinstance(profile_override, Mapping):
            raise ValueError(
                f"track {env.track.name!r} has no configured manifest teacher "
                f"profile planner for {teacher_profile!r}"
            )
        planner_values = _merge(planner_values, profile_override)
    planner_values.setdefault(
        "offset_iterations", int(settings.get("racing_line_iterations", 30))
    )
    planner_values.setdefault(
        "cache_directory",
        str(settings.get("racing_line_cache", "/workspace/outputs/evals/racing-lines")),
    )
    line = RacingLinePlanner(RacingLinePlannerConfig(**planner_values)).plan(env.track)
    return line, match, family, manifest_path


def make_mpcc(
    env: FlightmareEnv,
    settings: Mapping[str, Any],
    *,
    worker_index: int | None = None,
    backend_instance: Any | None = None,
) -> MPCCController:
    diagnostics_level = str(settings.get("mpcc_diagnostics_level", "full"))
    if (
        diagnostics_level == "minimal"
        and (
            float(settings.get("topology_weight", 0.0)) > 0.0
            or (
                float(settings.get("action_chunk_weight", 0.0)) > 0.0
                and str(settings.get(
                    "action_chunk_target_source", "mpcc_open_loop"
                )) == "mpcc_open_loop"
            )
        )
    ):
        raise ValueError(
            "minimal MPCC diagnostics are incompatible with MPCC horizon targets"
        )
    line, match, family, manifest_path = make_gate_reference(env, settings)
    nominal_speed = float(settings.get("mpcc_nominal_speed", 5.5))
    if bool(settings.get("mpcc_use_manifest_speed", False)):
        if manifest_path is not None:
            if match is None or match.get("qualified_speed_mps") is None:
                raise ValueError(
                    f"track {env.track.name!r} has no qualified speed in {manifest_path}"
                )
            speed_scale = float(settings.get("mpcc_manifest_speed_scale", 1.0))
            family_scales = settings.get("mpcc_family_manifest_speed_scales", {})
            if (
                family and isinstance(family_scales, Mapping)
                and family in family_scales
            ):
                speed_scale = float(family_scales[family])
            nominal_speed = float(match["qualified_speed_mps"]) * speed_scale
    controller_values = dict(settings.get("mpcc_config", {}))
    if bool(settings.get("mpcc_use_manifest_teacher_profile", False)) and match:
        selected = dict((match.get("qualification") or {}).get("selected") or {})
        teacher_profile = str(selected.get("teacher_profile", ""))
        profile_configs = settings.get(
            "mpcc_manifest_teacher_profile_controller_configs", {}
        )
        if profile_configs:
            profile_override = (
                profile_configs.get(teacher_profile)
                if isinstance(profile_configs, Mapping) else None
            )
            if not teacher_profile or not isinstance(profile_override, Mapping):
                raise ValueError(
                    f"track {env.track.name!r} has no configured manifest teacher "
                    f"profile controller for {teacher_profile!r}"
                )
            controller_values = _merge(controller_values, profile_override)
    family_configs = settings.get("mpcc_family_configs", {})
    if family and isinstance(family_configs, Mapping):
        family_override = family_configs.get(family)
        if isinstance(family_override, Mapping):
            controller_values = _merge(controller_values, family_override)
    family_speeds = settings.get("mpcc_family_nominal_speeds", {})
    if family and isinstance(family_speeds, Mapping) and family in family_speeds:
        nominal_speed = float(family_speeds[family])
    nominal_speed = float(controller_values.pop("nominal_speed", nominal_speed))
    frontier_profile = mpcc_speed_frontier_profile(settings, family)
    if frontier_profile == "swift-v1":
        if nominal_speed >= 18.0:
            frontier = (27.0, 30.0, 22.0, 24.0, 14.0, 5.0, 0.17)
        elif nominal_speed >= 17.0:
            frontier = (26.0, 28.0, 20.0, 22.0, 13.0, 4.8, 0.18)
        elif nominal_speed >= 16.0:
            frontier = (25.0, 26.0, 18.0, 20.0, 12.0, 4.5, 0.19)
        else:
            frontier = None
        if frontier is not None:
            (max_progress, acceleration, longitudinal, braking,
             collective_slew, body_rate_slew, corridor) = frontier
            controller_values.update(
                max_progress_speed=max_progress,
                maximum_acceleration=acceleration,
                maximum_longitudinal_acceleration=longitudinal,
                maximum_braking_acceleration=braking,
                maximum_collective_thrust=40.0,
                collective_slew_limit=collective_slew,
                body_rate_slew_limit=body_rate_slew,
                corridor_margin=corridor,
            )
    controller_values = _scale_mpcc_slew_limits_for_control_rate(
        controller_values, settings
    )
    controller_values.update(
        backend=str(settings.get("mpcc_backend", "predictive")),
        nominal_speed=nominal_speed,
        actuation_delay=float(settings.get("action_delay", 0.0)),
    )
    from starscream.mpcc.backend_pool import select_backend, register_backend
    controller_config = MPCCConfig(**controller_values)
    # Track switches may also switch teacher authority. Reusing the worker's
    # first compiled solver would silently keep its old weights/bounds/slew.
    backend_instance, backend_pool = select_backend(backend_instance, controller_config)
    if settings.get("mpcc_prediction_backend") == "native" and not settings.get("mpcc_allow_experimental_prediction", False):
        raise ValueError("full native prediction is experimental and not bit-exact; use native_exact")
    controller = MPCCController(
        env.track,
        line,
        config=controller_config,
        build_directory=(
            Path(str(settings.get("mpcc_build_root", "/tmp/starscream-acados-dagger")))
            / f"worker-{worker_index:02d}"
            / controller_config.fingerprint[:12]
            if worker_index is not None else None
        ),
        backend_instance=backend_instance,
        diagnostics_level=diagnostics_level,
        prediction_backend=str(settings.get("mpcc_prediction_backend", "numpy")),
        native_reference=bool(settings.get("mpcc_native_reference", False)),
        native_projection=bool(settings.get("mpcc_native_projection", False)),
        cache_runtime_vehicle=bool(settings.get("mpcc_cache_runtime_vehicle", False)),
        native_model_step=bool(settings.get("mpcc_native_model_step", False)),
    )
    register_backend(backend_pool, controller.backend_instance)
    if controller.backend_instance is not None:
        controller.backend_instance.native_stage_updates = bool(settings.get("mpcc_native_stage_updates", False))
    controller.reset()
    return controller


def _recovery_teacher_settings(
    settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    raw = settings.get("dagger_recovery_teacher")
    if not isinstance(raw, Mapping):
        return None
    result = copy.deepcopy(dict(settings))
    # The recovery controller is a separate, deliberately conservative MPCC
    # contract. Do not recursively merge the fast teacher's authority limits.
    result["mpcc_config"] = copy.deepcopy(dict(raw.get("mpcc_config", {})))
    result["mpcc_family_configs"] = copy.deepcopy(
        dict(raw.get("mpcc_family_configs", {}))
    )
    result["mpcc_family_nominal_speeds"] = copy.deepcopy(
        dict(raw.get("mpcc_family_nominal_speeds", {}))
    )
    for key in (
        "mpcc_nominal_speed", "mpcc_build_root", "mpcc_backend",
        "mpcc_use_manifest_speed", "mpcc_manifest_speed_scale",
        "mpcc_family_manifest_speed_scales", "mpcc_planner_config",
        "mpcc_family_planner_configs", "mpcc_use_manifest_teacher_profile",
        "mpcc_manifest_teacher_profile_planner_configs",
        "mpcc_manifest_teacher_profile_controller_configs",
    ):
        if key in raw:
            result[key] = copy.deepcopy(raw[key])
    result.setdefault("mpcc_use_manifest_speed", False)
    return result


class RoutedDaggerTeacher:
    """Fast nominal expert with a separately qualified recovery controller."""

    def __init__(
        self,
        env: FlightmareEnv,
        settings: Mapping[str, Any],
        *,
        worker_index: int | None = None,
        fast_backend: Any | None = None,
        recovery_backend: Any | None = None,
    ) -> None:
        self.settings = settings
        self.fast = make_mpcc(
            env, settings, worker_index=worker_index,
            backend_instance=fast_backend,
        )
        recovery_settings = _recovery_teacher_settings(settings)
        self.recovery = (
            make_mpcc(
                env, recovery_settings, worker_index=worker_index,
                backend_instance=recovery_backend,
            )
            if recovery_settings is not None else None
        )
        self._nominal_speed = float(self.fast.config.nominal_speed)
        adaptive_speed = settings.get("dagger_recovery_speed")
        self._adaptive_recovery_speed = (
            float(adaptive_speed) if adaptive_speed is not None else None
        )
        if (
            self._adaptive_recovery_speed is not None
            and self._adaptive_recovery_speed >= self._nominal_speed
        ):
            raise ValueError("dagger_recovery_speed must be below nominal MPCC speed")
        if self.recovery is not None and self._adaptive_recovery_speed is not None:
            raise ValueError(
                "configure either dagger_recovery_teacher or dagger_recovery_speed"
            )
        self._adaptive_recovery_active = False
        self._remaining_recovery_steps = 0
        self._fast_failure_streak = 0
        self.last_routed = False
        self.last_fast_solver_failed = False

    @property
    def backend_instances(self) -> tuple[Any | None, Any | None]:
        return (
            self.fast.backend_instance,
            self.recovery.backend_instance if self.recovery is not None else None,
        )

    def reset(self) -> None:
        if self._adaptive_recovery_active:
            self.fast.set_nominal_speed(self._nominal_speed)
        else:
            self.fast.reset()
        if self.recovery is not None:
            self.recovery.reset()
        self._remaining_recovery_steps = 0
        self._fast_failure_streak = 0
        self._adaptive_recovery_active = False
        self.last_routed = False
        self.last_fast_solver_failed = False

    def set_nominal_speed(self, speed: float) -> None:
        """Retarget one persistent MPCC instance for crossed-pace DAgger."""

        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("DAgger teacher speed must be finite and positive")
        if (
            self._adaptive_recovery_speed is not None
            and self._adaptive_recovery_speed >= speed
        ):
            raise ValueError(
                "adaptive recovery speed must remain below every crossed teacher speed"
            )
        self._nominal_speed = speed
        self.fast.set_nominal_speed(speed)
        if self.recovery is not None:
            self.recovery.reset()
        self._adaptive_recovery_active = False
        self._remaining_recovery_steps = 0
        self._fast_failure_streak = 0
        self.last_routed = False
        self.last_fast_solver_failed = False

    def __call__(self, observation: dict[str, Any]) -> Any:
        command = self.fast(observation)
        action = np.asarray(command.action.as_array(), np.float32)
        mode = int(np.asarray(command.diagnostics.get("mode", 0)).item())
        self.last_fast_solver_failed = int(command.solver_status) != 0
        minimum_margin = float(self.settings.get(
            "dagger_fast_teacher_minimum_constraint_margin", -float("inf")
        ))
        mode_is_clean = bool(
            mode == 0
            or not self.settings.get("dagger_route_fast_recovery_mode", False)
        )
        fast_clean = bool(
            command.valid
            and np.all(np.isfinite(action))
            and not self.last_fast_solver_failed
            and mode_is_clean
            and float(command.constraint_margin) >= minimum_margin
        )
        self._fast_failure_streak = (
            0 if fast_clean else self._fast_failure_streak + 1
        )
        if (
            self._adaptive_recovery_speed is not None
            and not self._adaptive_recovery_active
            and not fast_clean
        ):
            # Use one solver and one controller state. Retargeting the spatial
            # profile avoids generated-symbol conflicts between two ACADOS
            # instances and creates a coherent fast-to-recovery trajectory.
            self.fast.set_nominal_speed(self._adaptive_recovery_speed)
            self._adaptive_recovery_active = True
            command = self.fast(observation)
        if self._adaptive_recovery_active:
            self.last_routed = True
            return command
        if (
            self.recovery is not None
            and self._fast_failure_streak >= int(self.settings.get(
                "dagger_recovery_trigger_failures", 1
            ))
        ):
            self._remaining_recovery_steps = max(
                self._remaining_recovery_steps,
                int(self.settings.get("dagger_recovery_hold_steps", 18)),
            )
        self.last_routed = bool(
            self.recovery is not None and self._remaining_recovery_steps > 0
        )
        if self.last_routed:
            command = self.recovery(observation)
            self._remaining_recovery_steps -= 1
        return command

    def observe_executed_action(self, action: np.ndarray) -> None:
        self.fast.observe_executed_action(action)
        if self.recovery is not None:
            self.recovery.observe_executed_action(action)


def mpcc_topology_target_dim(settings: Mapping[str, Any]) -> int:
    """Return the configured expert-plan target width."""

    indices = tuple(int(item) for item in settings.get(
        "topology_target_horizon_indices", (1, 6, 12, 24, 34)
    ))
    if not indices or any(item < 0 for item in indices):
        raise ValueError("topology horizon indices must be non-negative")
    contract = str(settings.get("topology_target_contract", "gate_frame_v1"))
    if contract == "action_chunk_v1":
        width = 4
    elif contract == "gate_frame_v1":
        width = 7
    elif contract == "body_plan_action_chunk_v2":
        # Body-frame position displacement, body-frame velocity, normalized
        # CTBR command, and racing-line progress displacement.
        width = 3 + 3 + 4 + 1
    else:
        raise ValueError(f"unknown topology target contract: {contract}")
    return width * len(indices)


def mpcc_topology_target(
    command: Any,
    observation: Mapping[str, Any],
    env: FlightmareEnv,
    settings: Mapping[str, Any],
) -> np.ndarray:
    """Project the expert horizon into translation/rotation-invariant gate coordinates.

    Each selected horizon contributes gate-frame position displacement, gate-frame
    velocity, and racing-line progress displacement.  Fixed physical scales make
    the target contract portable to unseen tracks and future HDF5 collection.
    """

    indices = tuple(int(item) for item in settings.get(
        "topology_target_horizon_indices", (1, 6, 12, 24, 34)
    ))
    if not indices or any(item < 0 for item in indices):
        raise ValueError("topology horizon indices must be non-negative")
    contract = str(settings.get("topology_target_contract", "gate_frame_v1"))
    if contract == "action_chunk_v1":
        predicted_actions = np.asarray(
            command.diagnostics["predicted_action_horizon"], np.float32
        )
        if max(indices) >= len(predicted_actions):
            raise ValueError(
                f"action chunk {indices} is incompatible with MPCC horizon "
                f"{len(predicted_actions)}"
            )
        target = np.concatenate([
            ppo_ctbr_to_normalized(predicted_actions[horizon], settings)
            for horizon in indices
        ]).astype(np.float32)
        if (
            target.shape != (mpcc_topology_target_dim(settings),)
            or not np.all(np.isfinite(target))
        ):
            raise ValueError("MPCC action-chunk target is not finite and aligned")
        return target
    predicted = np.asarray(
        command.diagnostics["predicted_state_horizon"], np.float32
    )
    progress = np.asarray(
        command.diagnostics["predicted_progress_horizon"], np.float32
    )
    if max(indices) >= len(predicted) or len(progress) != len(predicted):
        raise ValueError(
            f"topology horizon {indices} is incompatible with MPCC horizon {len(predicted)}"
        )
    state = np.asarray(
        observation.get("privileged", {}).get("state", observation["state"]),
        np.float32,
    )
    if contract == "gate_frame_v1":
        gate = env.track.gates[env.tracker.index]
        local_from_world = np.asarray(gate.directed_rotation.T, np.float32)
    elif contract == "body_plan_action_chunk_v2":
        # The active-gate frame jumps at every crossing. The current body frame
        # evolves continuously and is still invariant to global translation and
        # rotation, making it a stable plan target through gate transitions.
        local_from_world = np.asarray(
            quaternion_matrix(state[3:7]).T, np.float32
        )
    else:
        raise ValueError(f"unknown topology target contract: {contract}")
    position_scale = float(settings.get("topology_position_scale_m", 8.0))
    velocity_scale = float(settings.get("topology_velocity_scale_mps", 20.0))
    progress_scale = float(settings.get("topology_progress_scale_m", 20.0))
    if min(position_scale, velocity_scale, progress_scale) <= 0.0:
        raise ValueError("topology target scales must be positive")
    values: list[np.ndarray] = []
    predicted_actions = None
    if contract == "body_plan_action_chunk_v2":
        predicted_actions = np.asarray(
            command.diagnostics["predicted_action_horizon"], np.float32
        )
        if max(indices) >= len(predicted_actions):
            raise ValueError(
                f"plan action horizon {indices} is incompatible with "
                f"MPCC horizon {len(predicted_actions)}"
            )
    for horizon in indices:
        position_delta = local_from_world @ (
            predicted[horizon, 0:3] - state[0:3]
        )
        velocity = local_from_world @ predicted[horizon, 7:10]
        progress_delta = np.asarray(
            [(progress[horizon] - float(command.reference_progress)) / progress_scale],
            np.float32,
        )
        values.extend([position_delta / position_scale, velocity / velocity_scale])
        if predicted_actions is not None:
            values.append(ppo_ctbr_to_normalized(
                predicted_actions[horizon], settings
            ))
        values.append(progress_delta)
    target = np.concatenate(values).astype(np.float32)
    if (
        target.shape != (mpcc_topology_target_dim(settings),)
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("MPCC topology target is not finite and aligned")
    return target


def mpcc_topology_target_is_valid(
    command: Any,
    target: np.ndarray,
    settings: Mapping[str, Any],
) -> bool:
    """Qualify a plan label independently from its usable CTBR command.

    ACADOS can return a usable bounded first command while a failed long-horizon
    state trajectory remains finite but numerically enormous. Such commands are
    valid DAgger labels; their plans are not. Keep the two validity contracts
    separate so rare horizon corruption cannot dominate replay optimization.
    """

    if bool(settings.get("topology_require_solver_success", False)) and int(
        command.solver_status
    ) != 0:
        return False
    indices = tuple(int(item) for item in settings.get(
        "topology_target_horizon_indices", (1, 6, 12, 24, 34)
    ))
    contract = str(settings.get("topology_target_contract", "gate_frame_v1"))
    target = np.asarray(target, np.float32)
    if not np.all(np.isfinite(target)):
        return False
    if contract != "body_plan_action_chunk_v2":
        maximum = float(settings.get("topology_target_maximum_absolute", 8.0))
        return bool(maximum > 0.0 and np.max(np.abs(target)) <= maximum)
    expected = len(indices) * 11
    if target.shape != (expected,):
        raise ValueError(
            f"body-plan target shape changed: expected {(expected,)}, got {target.shape}"
        )
    shaped = target.reshape(len(indices), 11)
    limits = np.asarray([
        float(settings.get("topology_max_normalized_position", 3.0)),
        float(settings.get("topology_max_normalized_velocity", 2.5)),
        float(settings.get("topology_max_normalized_action", 1.001)),
        float(settings.get("topology_max_normalized_progress", 2.0)),
    ], np.float32)
    if np.any(~np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError("topology qualification limits must be finite and positive")
    return bool(
        np.max(np.abs(shaped[:, 0:3])) <= limits[0]
        and np.max(np.abs(shaped[:, 3:6])) <= limits[1]
        and np.max(np.abs(shaped[:, 6:10])) <= limits[2]
        and np.max(np.abs(shaped[:, 10])) <= limits[3]
        # MPCC reference progress must move forward over the selected horizon.
        and np.min(shaped[:, 10]) >= -float(settings.get(
            "topology_progress_backward_tolerance", 0.02
        ))
    )


def mpcc_action_chunk_target(
    command: Any, settings: Mapping[str, Any],
) -> np.ndarray:
    """Normalize selected MPCC commands for receding-horizon supervision."""

    offsets = tuple(int(item) for item in settings.get(
        "action_chunk_offsets", ()
    ))
    if not offsets or offsets[0] != 0 or any(item < 0 for item in offsets):
        raise ValueError(
            "action_chunk_offsets must begin at zero and remain non-negative"
        )
    if tuple(sorted(set(offsets))) != offsets:
        raise ValueError("action_chunk_offsets must be unique and increasing")
    predicted = np.asarray(
        command.diagnostics["predicted_action_horizon"], np.float32
    )
    if offsets[-1] >= len(predicted):
        raise ValueError(
            f"action chunk {offsets} exceeds MPCC horizon {len(predicted)}"
        )
    target = np.stack([
        ppo_ctbr_to_normalized(predicted[offset], settings)
        for offset in offsets
    ]).astype(np.float32)
    if target.shape != (len(offsets), 4) or not np.all(np.isfinite(target)):
        raise ValueError("MPCC action chunk is not finite and aligned")
    return target


def replanned_teacher_action_chunks(
    actions: Sequence[np.ndarray], offsets: Sequence[int],
) -> list[np.ndarray]:
    """Build chunks from commands emitted by successive MPCC replans.

    The final available command is repeated at episode boundaries.  Unlike the
    legacy open-loop horizon, every non-terminal tail command was the first
    command of a solve at the corresponding observed state.
    """

    normalized_offsets = tuple(int(value) for value in offsets)
    if (
        not normalized_offsets
        or normalized_offsets[0] != 0
        or tuple(sorted(set(normalized_offsets))) != normalized_offsets
        or any(value < 0 for value in normalized_offsets)
    ):
        raise ValueError("replanned action chunk offsets must be unique and start at zero")
    commands = [np.asarray(value, np.float32) for value in actions]
    if any(value.shape != (4,) or not np.all(np.isfinite(value)) for value in commands):
        raise ValueError("replanned action chunk commands must be finite CTBR[4]")
    if not commands:
        return []
    return [
        np.stack([
            commands[min(index + offset, len(commands) - 1)]
            for offset in normalized_offsets
        ]).astype(np.float32)
        for index in range(len(commands))
    ]


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Numerically stable SO(3) logarithm for a near-identity rotation."""

    rotation = np.asarray(rotation, np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.asarray([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ], np.float64)
    if angle < 1.0e-6:
        return (0.5 * skew).astype(np.float32)
    return (0.5 * angle / np.sin(angle) * skew).astype(np.float32)


def mpcc_invariant_dynamics_target_dim(settings: Mapping[str, Any]) -> int:
    contract = str(settings.get("dynamics_target_contract", "task_delta_v1"))
    if contract == "task_delta_v1":
        return TASK_DIM
    if contract == "mpcc_body_delta_v2":
        # Local position, velocity, attitude rotation-vector, and body-rate deltas.
        return 12
    raise ValueError(f"unknown dynamics target contract: {contract}")


def mpcc_invariant_dynamics_target(
    command: Any,
    observation: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> np.ndarray:
    """Build a gate-invariant one-step target under the expert's own action.

    The legacy target used the environment's executed transition, which may
    have come from the learner or a DART perturbation, and became invalid when
    the active gate frame changed. This target instead uses MPCC's state after
    its first planned command and remains valid at gate crossings.
    """

    if str(settings.get("dynamics_target_contract", "task_delta_v1")) != (
        "mpcc_body_delta_v2"
    ):
        raise ValueError("invariant dynamics target requires mpcc_body_delta_v2")
    predicted = np.asarray(
        command.diagnostics["predicted_state_horizon"], np.float32
    )
    if len(predicted) < 2:
        raise ValueError("MPCC state horizon has no one-step dynamics target")
    state = np.asarray(
        observation.get("privileged", {}).get("state", observation["state"]),
        np.float32,
    )
    future = predicted[1]
    local_from_world = np.asarray(quaternion_matrix(state[3:7]).T, np.float32)
    current_rotation = np.asarray(quaternion_matrix(state[3:7]), np.float64)
    future_rotation = np.asarray(quaternion_matrix(future[3:7]), np.float64)
    scales = np.asarray(settings.get(
        "invariant_dynamics_scales", (0.50, 2.0, 0.20, 2.0)
    ), np.float32)
    if scales.shape != (4,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("invariant_dynamics_scales must contain four positive values")
    target = np.concatenate([
        local_from_world @ (future[0:3] - state[0:3]) / scales[0],
        local_from_world @ (future[7:10] - state[7:10]) / scales[1],
        _rotation_vector(current_rotation.T @ future_rotation) / scales[2],
        (future[10:13] - state[10:13]) / scales[3],
    ]).astype(np.float32)
    target = _normalize_dynamics_delta_for_control_rate(target, settings)
    if target.shape != (12,) or not np.all(np.isfinite(target)):
        raise ValueError("MPCC invariant dynamics target is not finite and aligned")
    return target


def observed_invariant_dynamics_targets(
    states: np.ndarray,
    next_states: np.ndarray,
    settings: Mapping[str, Any],
) -> np.ndarray:
    """Vectorized body-local transition targets for clean offline expert data."""

    states = np.asarray(states, np.float32)
    next_states = np.asarray(next_states, np.float32)
    if states.shape != next_states.shape or states.ndim != 2 or states.shape[1] < 13:
        raise ValueError("offline invariant dynamics requires aligned [N,D>=13] states")
    scales = np.asarray(settings.get(
        "invariant_dynamics_scales", (0.50, 2.0, 0.20, 2.0)
    ), np.float32)
    if scales.shape != (4,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("invariant_dynamics_scales must contain four positive values")

    quaternion = states[:, 3:7].astype(np.float64)
    future_quaternion = next_states[:, 3:7].astype(np.float64)
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12)
    future_quaternion /= np.maximum(
        np.linalg.norm(future_quaternion, axis=1, keepdims=True), 1e-12
    )
    w, x, y, z = quaternion.T
    world_from_body = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)
    position_delta = np.einsum(
        "nji,nj->ni", world_from_body,
        next_states[:, 0:3] - states[:, 0:3],
    )
    velocity_delta = np.einsum(
        "nji,nj->ni", world_from_body,
        next_states[:, 7:10] - states[:, 7:10],
    )

    w1 = quaternion[:, :1]
    v1 = quaternion[:, 1:]
    w2 = future_quaternion[:, :1]
    v2 = future_quaternion[:, 1:]
    relative_w = w1 * w2 + np.sum(v1 * v2, axis=1, keepdims=True)
    relative_v = w1 * v2 - w2 * v1 - np.cross(v1, v2)
    flip = relative_w[:, 0] < 0.0
    relative_w[flip] *= -1.0
    relative_v[flip] *= -1.0
    vector_norm = np.linalg.norm(relative_v, axis=1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative_w, 1.0e-12, None))
    rotation_vector = np.where(
        vector_norm > 1.0e-9,
        angle * relative_v / np.maximum(vector_norm, 1.0e-12),
        2.0 * relative_v,
    )
    targets = np.concatenate([
        position_delta / scales[0],
        velocity_delta / scales[1],
        rotation_vector / scales[2],
        (next_states[:, 10:13] - states[:, 10:13]) / scales[3],
    ], axis=1).astype(np.float32)
    if targets.shape != (len(states), 12) or not np.all(np.isfinite(targets)):
        raise ValueError("offline invariant dynamics targets are not finite")
    return targets


def make_reward(
    settings: Mapping[str, Any], target_speed: float,
) -> CurriculumRaceReward | Green2026StateRaceReward:
    raw = dict(settings.get("reward", {}))
    contract = str(settings.get("reward_contract", "starscream_curriculum_v1"))
    if contract == "green2026_state_v1":
        actual_dt = configured_control_dt(settings)
        if 'control_dt' in raw and not np.isclose(float(raw['control_dt']), actual_dt, rtol=1e-7, atol=1e-10):
            raise ValueError('Green reward control_dt must match the environment control timestep')
        raw.setdefault(
            "maximum_collective_thrust",
            float(settings.get("collective_action_maximum", 30.0)),
        )
        raw.setdefault(
            "maximum_body_rate",
            float(settings.get("maximum_body_rate", 6.0)),
        )
        raw.setdefault("control_dt", actual_dt)
        raw.setdefault("target_speed", float(target_speed))
        return Green2026StateRaceReward(Green2026StateRewardConfig(**raw))
    if contract != "starscream_curriculum_v1":
        raise ValueError(f"unknown racing reward contract {contract!r}")
    raw["target_speed"] = float(target_speed)
    raw.setdefault("control_dt", configured_control_dt(settings))
    return CurriculumRaceReward(CurriculumRewardConfig(**raw))


def make_env(
    settings: Mapping[str, Any], *,
    reward: CurriculumRaceReward | Green2026StateRaceReward | None = None,
    track: str | None = None,
) -> FlightmareEnv:
    configured = settings.get("tracks", (settings.get("track", "figure8"),))
    tracks = (configured,) if isinstance(configured, str) else tuple(configured)
    compact_observation = bool(settings.get('collection_compact_observation', False))
    if compact_observation and str(settings.get('observation_contract', LEGACY_OBSERVATION_CONTRACT)) not in {LEGACY_OBSERVATION_CONTRACT, 'starscream_route_plant_v1'}:
        raise ValueError('compact collection observations only support the legacy privileged contract')
    return FlightmareEnv(
        track=str(track or tracks[0]),
        next_gates=int(settings.get("route_gate_count", 3)),
        image_size=(160, 128),
        control_dt=configured_control_dt(settings),
        render_observations=False,
        mask_source="none",
        action_delay=float(settings.get("action_delay", 0.0)),
        action_delay_range=(
            None if settings.get("action_delay_range") is None
            else tuple(float(value) for value in settings["action_delay_range"])
        ),
        state_estimator_randomization=settings.get("state_estimator_randomization"),
        policy_state_source=str(settings.get("policy_state_source", "truth")),
        flight_plan_randomization=settings.get("flight_plan_randomization"),
        maximum_collective_thrust=float(
            settings.get("collective_action_maximum", 30.0)
        ),
        reward_function=reward,
        terminate_on_collision=True,
        dynamics_randomization=settings.get("dynamics_randomization"),
        collection_observation=compact_observation,
        plant_settings_observation=(settings.get('observation_contract') == 'starscream_route_plant_v1'),
    )


def reset_env(
    env: FlightmareEnv,
    stage: RacingCurriculumStage,
    *,
    seed: int,
    episode_index: int,
    rollout_laps: int | None = None,
) -> tuple[dict[str, Any], int]:
    spawn = sample_curriculum_spawn(
        env.track, stage, seed=seed, episode_index=episode_index
    )
    archived_options = None
    if stage.allow_archived_task_resets:
        from starscream.env.racing_manifold.classical_tasks import archived_task_reset
        archived_options = archived_task_reset(env.track, seed, episode_index)
        if archived_options is not None and int(stage.rollout_laps if rollout_laps is None else rollout_laps)!=1:
            raise ValueError('archived suffix tasks require one finite route')
    observation, _ = env.reset(
        seed=seed,
        options={
            "gate_index": spawn.gate_index,
            "state": spawn.state,
            "spawn": dict(spawn.metadata),
            "route_plan_total_gates": (
                min(stage.target_gates, len(env.track.gates))
                * int(stage.rollout_laps if rollout_laps is None else rollout_laps)
            ),
            **(archived_options or {}),
        },
    )
    return observation, env.tracker.passed_count


def dagger_start_perturbation_stage(
    stage: RacingCurriculumStage, scale: float,
) -> RacingCurriculumStage:
    """Widen a canonical DART reset basin without changing gate topology."""

    scale = float(scale)
    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("DART start perturbation scale must be finite and >= 1")

    def expand(interval: tuple[float, float], *, floor: float | None = None) -> tuple[float, float]:
        center = 0.5 * (interval[0] + interval[1])
        half = 0.5 * (interval[1] - interval[0]) * scale
        low, high = center - half, center + half
        if floor is not None:
            low = max(low, floor)
            high = max(high, low)
        return float(low), float(high)

    return replace(
        stage,
        approach_distance=expand(stage.approach_distance, floor=0.25),
        lateral_offset=expand(stage.lateral_offset),
        vertical_offset=expand(stage.vertical_offset),
        forward_speed=expand(stage.forward_speed, floor=0.25),
        lateral_speed=expand(stage.lateral_speed),
        vertical_speed=expand(stage.vertical_speed),
        attitude_error_degrees=stage.attitude_error_degrees * scale,
        body_rate=stage.body_rate * scale,
    )


def _manifest_tracks(raw: Mapping[str, Any]) -> tuple[str, ...] | None:
    manifest = raw.get("track_manifest")
    if manifest is None:
        return None
    return manifest_track_paths(
        str(manifest),
        split=str(raw.get("track_split", "train")),
        families=raw.get("track_families"),
        qualified_only=bool(raw.get("qualified_tracks_only", False)),
        dynamic_qualification_mode=raw.get("dynamic_qualification_mode"),
        minimum_qualified_speed=(
            float(raw["minimum_qualified_speed"])
            if raw.get("minimum_qualified_speed") is not None else None
        ),
        limit=int(raw.get("track_limit", 0)),
    )


def parse_stage(raw: Mapping[str, Any]) -> RacingCurriculumStage:
    values = dict(raw)
    manifest_tracks = _manifest_tracks(values)
    explicit = values.get("tracks", ())
    explicit_tracks = (
        (str(explicit),) if isinstance(explicit, str)
        else tuple(str(item) for item in explicit)
    )
    stage_tracks = list(manifest_tracks if manifest_tracks is not None else explicit_tracks)
    real_course_suite = values.get("real_course_suite")
    if real_course_suite is not None:
        suite_tracks, _ = load_active_real_course_suite(str(real_course_suite))
        stage_tracks.extend(suite_tracks)
    if stage_tracks:
        values["tracks"] = tuple(dict.fromkeys(stage_tracks))
    for key in (
        "track_manifest", "track_split", "track_families",
        "qualified_tracks_only", "dynamic_qualification_mode",
        "minimum_qualified_speed", "track_limit",
        "real_course_suite",
    ):
        values.pop(key, None)
    values.pop("minimum_environment_steps", None)
    values.pop("advancement_lap_time_seconds", None)
    values.pop("advancement_metric", None)
    values.pop("rollback_full_course_threshold", None)
    return RacingCurriculumStage.from_mapping(values)


def episode_target_speed(stage: RacingCurriculumStage, seed: int) -> float:
    """Draw a deterministic per-episode pace command from the stage interval."""

    interval = stage.target_speed_range
    if interval is None or interval[0] == interval[1]:
        return float(stage.target_speed if interval is None else interval[0])
    generator = np.random.default_rng(int(seed) ^ 0x5EED5EED)
    return float(generator.uniform(interval[0], interval[1]))


def manifest_track_values(
    manifest_path: str | Path, field: str, *, cast: Any = float,
) -> dict[str, Any]:
    """Resolve per-track manifest metadata against the concrete track paths."""

    path = Path(manifest_path)
    manifest = read_manifest(path)
    values: dict[str, Any] = {}
    for record in manifest["records"]:
        if record.get(field) is None:
            continue
        track = str((path.parent / str(record["path"])).resolve())
        value = cast(record[field])
        if track in values and values[track] != value:
            raise ValueError(
                f"manifest has conflicting {field!r} values for {track!r}"
            )
        values[track] = value
    return values


def evaluation_track_speed_commands(settings: Mapping[str, Any]) -> dict[str, float]:
    """Explicit held-out pace commands, separate from the training manifest."""
    commands = {}
    command_file = settings.get('evaluation_track_speed_command_file')
    file_commands = {}
    if command_file:
        payload = json.loads(Path(command_file).read_text())
        file_commands = payload.get('commands', payload)
        if not isinstance(file_commands, dict):
            raise ValueError('evaluation speed command file must contain a commands mapping')
    for path, value in {**file_commands,
                        **settings.get('evaluation_track_speed_commands', {})}.items():
        speed = float(value)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('evaluation track speed commands must be finite and positive')
        track = str(Path(path).resolve())
        if track in commands and commands[track] != speed:
            raise ValueError(f'conflicting evaluation speed commands for {track}')
        commands[track] = speed
    return commands


def configured_ppo_manifest_speed_field(settings: Mapping[str, Any]) -> str:
    """Return the manifest field used as PPO's per-track pace command.

    MPCC qualification and an RL curriculum target are different scientific
    claims.  Keeping the field configurable lets an admitted-but-unqualified
    course receive a deliberate pace command without pretending that the
    classical controller certified that speed.
    """

    field = str(
        settings.get("ppo_manifest_speed_field", "qualified_speed_mps")
    ).strip()
    if not field:
        raise ValueError("ppo_manifest_speed_field must be a non-empty field name")
    return field


def configured_track_rollout_laps(
    settings: Mapping[str, Any], tracks: Sequence[str],
) -> dict[str, int]:
    """Return validated DAgger lap targets, defaulting each track to one lap."""

    default = int(settings.get("dagger_rollout_laps", 1))
    if default < 1:
        raise ValueError("dagger_rollout_laps must be positive")
    result = {str(Path(track).resolve()): default for track in tracks}
    manifest_path = settings.get("track_manifest")
    if manifest_path is not None:
        manifest_laps = manifest_track_values(
            str(manifest_path), "dagger_laps", cast=int,
        )
        result.update({
            track: laps for track, laps in manifest_laps.items()
            if track in result
        })
    invalid = {track: laps for track, laps in result.items() if laps < 1}
    if invalid:
        raise ValueError(f"manifest dagger_laps must be positive: {invalid}")
    return result


def ppo_episode_rollout_laps(
    settings: Mapping[str, Any],
    stage: RacingCurriculumStage,
    episode_seed: int,
    *,
    training: bool,
) -> int:
    """Choose a deterministic lap target for PPO training episodes.

    Evaluation remains at the stage's fixed lap count. Training may mix lap
    counts so looping geometry exposes both terminal padding at a one-lap
    finish and wrapped future-route references before another lap.
    """

    if not training or settings.get("ppo_rollout_lap_mixture") is None:
        return int(stage.rollout_laps)
    raw = settings["ppo_rollout_lap_mixture"]
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("ppo_rollout_lap_mixture must map lap counts to weights")
    laps = np.asarray([int(item) for item in raw], np.int64)
    weights = np.asarray([float(raw[item]) for item in raw], np.float64)
    if (
        np.any(laps < 1) or len(set(laps.tolist())) != len(laps)
        or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0)
    ):
        raise ValueError("PPO lap mixture requires unique positive laps and weights")
    weights /= weights.sum()
    generator = np.random.default_rng(int(episode_seed) ^ 0x1A9C0DE)
    return int(generator.choice(laps, p=weights))


def ppo_reliability_reward(
    reward: float,
    progress_before: float,
    progress_after: float,
    done: bool,
    result: Mapping[str, Any] | None,
    settings: Mapping[str, Any],
) -> float:
    """Add a bounded gate-survival objective to the environment reward.

    Ordered gate progress is a monotonic, topology-invariant signal. Weighting
    later crossings more strongly approximates the conditional-survival
    factorization of course completion without exposing a track identifier to
    the policy. Completion and late crashes remain true terminal events.
    """

    before = min(max(float(progress_before), 0.0), 1.0)
    after = min(max(float(progress_after), 0.0), 1.0)
    if not math.isfinite(before) or not math.isfinite(after):
        raise ValueError("PPO route progress must be finite")
    power = float(settings.get("ppo_survival_progress_power", 2.0))
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("ppo_survival_progress_power must be finite and positive")
    progress_delta = max(after - before, 0.0)
    late_scale = 1.0 + float(
        settings.get("ppo_late_gate_survival_multiplier", 0.0)
    ) * after**power
    adjusted = float(reward) + float(
        settings.get("ppo_gate_survival_bonus", 0.0)
    ) * progress_delta * late_scale
    if done and result is not None:
        success = int(result.get("gates", 0)) >= int(result.get("target_gates", 1))
        if success:
            adjusted += float(settings.get("ppo_course_completion_bonus", 0.0))
        elif bool(result.get("crashed", False)):
            adjusted -= float(settings.get("ppo_failure_penalty", 0.0)) * (
                1.0
                + float(settings.get("ppo_late_failure_multiplier", 0.0))
                * before**power
            )
    return adjusted


def configured_dagger_speed_fractions(
    settings: Mapping[str, Any],
) -> tuple[float, ...]:
    """Return the crossed pace targets used on every DAgger topology."""

    raw = settings.get("dagger_teacher_speed_fractions", (1.0,))
    fractions = tuple(float(item) for item in raw)
    if (
        not fractions
        or any(not np.isfinite(item) or item <= 0.0 or item > 1.0 for item in fractions)
        or len(set(fractions)) != len(fractions)
    ):
        raise ValueError(
            "dagger_teacher_speed_fractions must contain unique values in (0,1]"
        )
    return tuple(sorted(fractions))


def dagger_topology_group_ids(
    settings: Mapping[str, Any],
    tracks: np.ndarray,
    speed_commands: np.ndarray,
    course_progress: np.ndarray,
    gate_phases: np.ndarray | None = None,
) -> np.ndarray:
    """Encode family x relative-pace x course-phase cells for robust ERM.

    These IDs are trainer metadata only.  Neither family nor track identity is
    supplied to the actor.
    """

    manifest_path = settings.get("track_manifest")
    if manifest_path is None:
        raise ValueError("topology grouping requires a track manifest")
    family_by_track = manifest_track_values(manifest_path, "family", cast=str)
    frontier_by_track = manifest_track_values(
        manifest_path, "qualified_speed_mps", cast=float,
    )
    families = tuple(sorted(set(family_by_track.values())))
    family_to_id = {name: index for index, name in enumerate(families)}
    fractions = np.asarray(
        configured_dagger_speed_fractions(settings), np.float32
    )
    edges = np.asarray(
        settings.get("dagger_topology_progress_edges", (0.0, 0.34, 0.67, 1.01)),
        np.float32,
    )
    if (
        edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0) or edges[0] > 0.0 or edges[-1] < 1.0
    ):
        raise ValueError(
            "dagger_topology_progress_edges must increase from <=0 to >=1"
        )
    result = np.empty(len(tracks), np.int16)
    pace_count = len(fractions)
    phase_count = len(edges) - 1
    include_gate_phase = bool(settings.get(
        "dagger_group_include_gate_phase", False
    ))
    configured_gate_phases = tuple(int(item) for item in settings.get(
        "dagger_group_gate_phases", (0, 2, 3, 4)
    ))
    if include_gate_phase:
        if gate_phases is None or len(gate_phases) != len(tracks):
            raise ValueError("gate-phase topology grouping requires aligned phases")
        if len(set(configured_gate_phases)) != len(configured_gate_phases):
            raise ValueError("dagger_group_gate_phases must be unique")
        gate_phase_to_id = {
            value: index for index, value in enumerate(configured_gate_phases)
        }
        gate_phase_count = len(configured_gate_phases)
        if not gate_phase_count:
            raise ValueError("dagger_group_gate_phases cannot be empty")
    else:
        gate_phase_to_id = {}
        gate_phase_count = 1
    for index, (raw_track, speed, progress) in enumerate(zip(
        tracks, speed_commands, course_progress
    )):
        track = str(Path(str(raw_track)).resolve())
        if track not in family_by_track or track not in frontier_by_track:
            raise ValueError(f"track {track!r} is missing topology metadata")
        ratio = float(speed) / max(float(frontier_by_track[track]), 1.0e-6)
        pace = int(np.argmin(np.abs(fractions - ratio)))
        phase = int(np.searchsorted(edges[1:-1], float(progress), side="right"))
        base_group = (
            family_to_id[family_by_track[track]] * pace_count * phase_count
            + pace * phase_count
            + phase
        )
        if include_gate_phase:
            raw_gate_phase = int(np.asarray(gate_phases).reshape(-1)[index])
            gate_phase = gate_phase_to_id.get(
                raw_gate_phase, gate_phase_to_id.get(0, 0)
            )
            base_group = base_group * gate_phase_count + gate_phase
        result[index] = base_group
    return result


def dagger_should_rollback(
    evaluation: Mapping[str, Any],
    safe_evaluation: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> bool:
    """Apply absolute legacy floors and/or degradation-relative safeguards."""

    if not bool(settings.get("dagger_enable_safety_rollback", False)):
        return False
    checks: list[bool] = []
    absolute_full = settings.get("dagger_rollback_full_course_threshold")
    if absolute_full is not None:
        checks.append(
            float(evaluation.get("full_course_success", 0.0))
            < float(absolute_full)
        )
    absolute_track = settings.get("dagger_rollback_minimum_track_success")
    if absolute_track is not None:
        checks.append(
            float(evaluation.get("minimum_track_full_course_success", 1.0))
            < float(absolute_track)
        )
    maximum_full_drop = settings.get("dagger_rollback_max_full_course_drop")
    if maximum_full_drop is not None:
        checks.append(
            float(evaluation.get("full_course_success", 0.0))
            < float(safe_evaluation.get("full_course_success", 0.0))
            - float(maximum_full_drop)
        )
    maximum_track_drop = settings.get("dagger_rollback_maximum_track_drop")
    if maximum_track_drop is not None:
        checks.append(
            float(evaluation.get("minimum_track_full_course_success", 1.0))
            < float(safe_evaluation.get(
                "minimum_track_full_course_success", 1.0
            )) - float(maximum_track_drop)
        )
    maximum_individual_drop = settings.get(
        "dagger_rollback_maximum_individual_track_drop"
    )
    if maximum_individual_drop is not None:
        suffix = "/full_course_success"
        track_keys = {
            key for key in safe_evaluation
            if key.startswith("track/") and key.endswith(suffix)
        }
        checks.extend(
            float(evaluation.get(key, 0.0))
            < float(safe_evaluation.get(key, 0.0)) - float(maximum_individual_drop)
            for key in track_keys
        )
    return any(checks)


@dataclass
class RolloutSlot:
    env: FlightmareEnv
    observation: dict[str, Any]
    history: CausalHistory
    start_passed: int
    episode_index: int
    target_gates: int
    max_steps: int
    target_speed: float = 0.0
    expert: RoutedDaggerTeacher | None = None
    steps: int = 0
    total_return: float = 0.0
    crashed: bool = False
    done: bool = False
    steps_since_crossing: int | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)


@torch.no_grad()
def evaluate_policy(
    policy: PrivilegedMLPPolicy,
    normalizer: FeatureNormalizer,
    settings: Mapping[str, Any],
    stage: RacingCurriculumStage,
    *,
    count: int,
    seed_base: int,
    device: str,
    collector: ProcessRaceCollector | None = None,
) -> dict[str, float]:
    if str(settings.get("evaluation_backend", "thread")) == "process":
        if collector is not None:
            if collector.policy is not policy or collector.stage != stage:
                raise ValueError("persistent evaluator policy/stage mismatch")
            # Match fresh-worker affinity rather than carrying last eval's order.
            for index, slot in enumerate(collector.slots):
                slot.track = stage.tracks[index % len(stage.tracks)]
            from starscream.evaluation_suite import selection_metrics
            return selection_metrics(collector.evaluate(episodes=count, seed_base=seed_base), settings, stage)
        collector = ProcessRaceCollector(
            policy, normalizer, settings, stage, device,
            workers=min(int(settings.get("evaluation_workers", 12)), int(count)),
        )
        try:
            from starscream.evaluation_suite import selection_metrics
            return selection_metrics(collector.evaluate(episodes=count, seed_base=seed_base), settings, stage)
        finally:
            collector.close()
    policy.eval()
    results: list[dict[str, Any]] = []
    explicit_speeds = evaluation_track_speed_commands(settings)
    parallel = min(int(settings.get("evaluation_workers", 16)), int(count))
    episode_index = 0
    while episode_index < count:
        batch_count = min(parallel, count - episode_index)
        slots: list[RolloutSlot] = []
        try:
            for offset in range(batch_count):
                schedule_index = episode_index + offset
                index_offset = int(settings.get("evaluation_episode_index_offset", 0))
                index_stride = int(settings.get("evaluation_episode_index_stride", 1))
                if index_offset < 0 or index_stride < 1:
                    raise ValueError("evaluation episode index offset/stride is invalid")
                index = index_offset + index_stride * schedule_index
                from starscream.evaluation_suite import selection_episode_seed
                episode_seed = selection_episode_seed(settings, stage.tracks[schedule_index % len(stage.tracks)], index, seed_base, len(stage.tracks))
                target_speed = episode_target_speed(stage, episode_seed)
                target_speed = explicit_speeds.get(
                    str(Path(stage.tracks[schedule_index % len(stage.tracks)]).resolve()),
                    target_speed,
                )
                env = make_env(
                    settings,
                    track=stage.tracks[schedule_index % len(stage.tracks)],
                    reward=make_reward(settings, target_speed),
                )
                observation, start = reset_env(
                    env, stage, seed=episode_seed, episode_index=int(settings.get('evaluation_fixed_start_gate_index', index))
                )
                history = CausalHistory(policy.context_steps)
                history.reset_feature(ppo_observation_features(observation, settings))
                slots.append(RolloutSlot(
                    env, observation, history, start, index,
                    completion_gate_count(env.track,stage.target_gates,enabled=settings.get("allow_curriculum_completion_override",False)) * stage.rollout_laps,
                    stage.max_steps,
                    target_speed,
                ))
            with ThreadPoolExecutor(max_workers=batch_count) as executor:
                while any(not slot.done for slot in slots):
                    active = [slot for slot in slots if not slot.done]
                    histories = np.stack([slot.history.array() for slot in active])
                    batch = torch.from_numpy(normalizer.numpy(histories)).to(
                        device, non_blocking=True
                    )
                    speed_commands = torch.as_tensor(
                        [slot.target_speed for slot in active],
                        device=device, dtype=batch.dtype,
                    )
                    actions = policy(batch, speed_commands).float().cpu().numpy()
                    futures = {
                        index: executor.submit(
                            slot.env.step,
                            ppo_normalized_to_ctbr(actions[index], settings),
                        )
                        for index, slot in enumerate(active)
                    }
                    for index, slot in enumerate(active):
                        observation, reward, terminated, _, info = futures[index].result()
                        slot.observation = observation
                        slot.history.append_feature(
                            ppo_observation_features(observation, settings)
                        )
                        slot.steps += 1
                        slot.total_return += float(reward)
                        gates = slot.env.tracker.passed_count - slot.start_passed
                        slot.crashed = bool(
                            info.get("ground_contact") or info.get("unity_collision")
                        )
                        slot.done = bool(
                            terminated or gates >= slot.target_gates
                            or slot.steps >= slot.max_steps
                        )
            results.extend({
                "gates": slot.env.tracker.passed_count - slot.start_passed,
                "crashed": slot.crashed,
                "return": slot.total_return,
                "steps": slot.steps,
                "track": slot.env.track.name,
                "target_gates": slot.target_gates,
                "target_speed_mps": slot.target_speed,
            } for slot in slots)
        finally:
            for slot in slots:
                slot.env.close()
        episode_index += batch_count
    from starscream.evaluation_suite import selection_metrics
    return selection_metrics(multitrack_metrics(results, stage.target_gates * stage.rollout_laps), settings, stage)


def evaluate_policy_matched_tracks(
    policy: PrivilegedMLPPolicy,
    normalizer: FeatureNormalizer,
    settings: Mapping[str, Any],
    stage: RacingCurriculumStage,
    *,
    count_per_track: int,
    seed_base: int,
    device: str,
) -> dict[str, float]:
    """Evaluate every track on the same episode/dynamics seed cohort.

    Interleaving tracks by global episode index assigns each geometry a
    different slice of domain randomization. That is harmless for a large
    aggregate benchmark, but it can reverse an online curriculum decision at
    the small per-track sample counts used between PPO updates.
    """

    if count_per_track < 1:
        raise ValueError("matched evaluation count_per_track must be positive")
    rows = [
        evaluate_policy(
            policy, normalizer, settings, replace(stage, tracks=(track,)),
            count=count_per_track, seed_base=seed_base, device=device,
        )
        for track in stage.tracks
    ]
    metrics: dict[str, float] = {}
    shared = set.intersection(*(
        {key for key in row if not key.startswith("track/")} for row in rows
    ))
    for key in shared:
        values = [float(row[key]) for row in rows]
        finite = [value for value in values if np.isfinite(value)]
        if not finite:
            metrics[key] = float("nan")
        elif key == "episodes":
            metrics[key] = float(sum(finite))
        elif key.startswith("maximum_"):
            metrics[key] = float(max(finite))
        else:
            metrics[key] = float(np.mean(finite))
    for row in rows:
        metrics.update({
            key: float(value) for key, value in row.items()
            if key.startswith("track/")
        })
    track_scores = [float(row["selection_score"]) for row in rows]
    track_success = [float(row["full_course_success"]) for row in rows]
    metrics["episodes_per_track"] = float(count_per_track)
    metrics["aggregate_selection_score"] = float(
        np.mean([float(row["aggregate_selection_score"]) for row in rows])
    )
    metrics["minimum_track_selection_score"] = min(track_scores)
    metrics["minimum_track_full_course_success"] = min(track_success)
    metrics["selection_score"] = min(track_scores)
    return metrics


def multitrack_metrics(
    results: list[Mapping[str, Any]], gate_count: int,
) -> dict[str, float]:
    metrics = flatten_metrics(results, gate_count)
    if results and all('reference_misses' in r for r in results):
        metrics.update(gate_behavior_metrics(results))
    tracks = sorted({str(item.get("track", "unknown")) for item in results})
    track_scores: list[float] = []
    track_full_course_success: list[float] = []
    track_performance_weighted_success: list[float] = []
    for track in tracks:
        track_metrics = flatten_metrics(
            [item for item in results if str(item.get("track", "unknown")) == track],
            gate_count,
        )
        if results and all('reference_misses' in r for r in results):
            track_metrics.update(gate_behavior_metrics([r for r in results if str(r.get('track', 'unknown')) == track]))
        track_scores.append(track_metrics["selection_score"])
        track_full_course_success.append(track_metrics["full_course_success"])
        track_performance_weighted_success.append(
            track_metrics["performance_weighted_success_hz90"]
        )
        metrics.update({f"track/{track}/{key}": value for key, value in track_metrics.items()})
    metrics["aggregate_selection_score"] = metrics["selection_score"]
    metrics["minimum_track_selection_score"] = min(track_scores)
    metrics["minimum_track_full_course_success"] = min(
        track_full_course_success
    )
    metrics["selection_score"] = min(track_scores)
    metrics["performance_weighted_success_score"] = float(
        np.mean(track_performance_weighted_success)
    )
    return metrics


def gate_behavior_metrics(rows):
    success = np.asarray([not r['crashed'] and r['gates'] >= r['target_gates'] for r in rows])
    clean = np.asarray([r['reference_misses'] == 0 for r in rows])
    timely = np.asarray([r.get('within_clean_deadline', True) for r in rows])
    return dict(timely_success=float(np.mean(success & timely)),
                clean_success=float(np.mean(success & clean)),
                clean_timely_success=float(np.mean(success & clean & timely)),
                recovered_success=float(np.mean(success & ~clean)),
                genuine_miss_episode_rate=float(np.mean(~clean)),
                raw_plane_crossings=float(np.mean([r['raw_plane_crossings'] for r in rows])),
                planned_plane_crossings=float(np.mean([r['planned_plane_crossings'] for r in rows])))


def configured_tracks(settings: Mapping[str, Any]) -> tuple[str, ...]:
    manifest_tracks = _manifest_tracks(settings)
    if manifest_tracks is not None:
        return manifest_tracks
    raw = settings.get("tracks", (settings.get("track", "figure8"),))
    return (str(raw),) if isinstance(raw, str) else tuple(str(item) for item in raw)


def balanced_choice(
    rng: np.random.Generator, indices: np.ndarray, track_ids: np.ndarray,
    size: int,
) -> np.ndarray:
    groups = [indices[track_ids[indices] == item] for item in np.unique(track_ids[indices])]
    if not groups or any(not len(group) for group in groups):
        raise ValueError("balanced sampling requires data from every configured track")
    counts = [size // len(groups) + int(index < size % len(groups)) for index in range(len(groups))]
    selected = np.concatenate([
        rng.choice(group, size=count, replace=True) for group, count in zip(groups, counts)
    ])
    rng.shuffle(selected)
    return selected


def balanced_progress_choice(
    rng: np.random.Generator,
    track_ids: np.ndarray,
    course_progress: np.ndarray,
    size: int,
    *,
    late_fraction: float,
    late_threshold: float,
) -> np.ndarray:
    """Balance tracks while reserving replay capacity for late-course states."""

    indices = np.arange(len(track_ids))
    if not 0.0 <= late_fraction <= 1.0 or not 0.0 <= late_threshold <= 1.0:
        raise ValueError("invalid late-course replay fractions")
    late = indices[course_progress >= late_threshold]
    ordinary = indices[course_progress < late_threshold]
    late_count = int(round(size * late_fraction)) if len(late) else 0
    late_count = min(late_count, size)
    ordinary_count = size - late_count
    selected: list[np.ndarray] = []
    if late_count:
        selected.append(balanced_choice(rng, late, track_ids, late_count))
    if ordinary_count:
        source = ordinary if len(ordinary) else indices
        selected.append(balanced_choice(rng, source, track_ids, ordinary_count))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


def dagger_event_mask(
    actions: np.ndarray,
    previous_actions: np.ndarray,
    dynamics_valid: np.ndarray,
    settings: Mapping[str, Any],
    gate_phases: np.ndarray | None = None,
) -> np.ndarray:
    """Identify sparse control events that uniform trajectory replay dilutes.

    During expert execution, ``previous_actions`` is the preceding teacher
    command, so action deltas mark braking and turn onset. During learner
    execution it is the issued learner command, making the same test an
    intervention/disagreement signal. Sustained body-rate commands and gate
    crossings cover the rest of the maneuver window without using track IDs.
    """

    actions = np.asarray(actions, np.float32)
    previous_actions = np.asarray(previous_actions, np.float32)
    dynamics_valid = np.asarray(dynamics_valid, np.bool_).reshape(-1)
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError("DAgger event actions must have shape [N, 4]")
    if previous_actions.shape != actions.shape or len(dynamics_valid) != len(actions):
        raise ValueError("DAgger event arrays must remain aligned")
    action_delta = np.max(np.abs(actions - previous_actions), axis=1)
    body_rate_magnitude = np.max(np.abs(actions[:, 1:]), axis=1)
    collective_drop = previous_actions[:, 0] - actions[:, 0]
    control_events = (
        (action_delta >= float(settings.get(
            "dagger_event_action_delta_threshold", 0.08
        )))
        | (body_rate_magnitude >= float(settings.get(
            "dagger_event_body_rate_threshold", 0.35
        )))
        | (collective_drop >= float(settings.get(
            "dagger_event_collective_drop_threshold", 0.04
        )))
        | ~dynamics_valid
    )
    configured_gate_phases = tuple(int(item) for item in settings.get(
        "dagger_event_gate_phases", ()
    ))
    if any(item < 0 or item > 4 for item in configured_gate_phases):
        raise ValueError("dagger_event_gate_phases must contain phase ids in [0,4]")
    if configured_gate_phases:
        if gate_phases is None:
            raise ValueError("gate-phase event replay requires aligned gate phases")
        gate_phases = np.asarray(gate_phases, np.int8).reshape(-1)
        if len(gate_phases) != len(actions):
            raise ValueError("DAgger gate phases must align with actions")
        phase_events = np.isin(gate_phases, configured_gate_phases)
    else:
        phase_events = np.zeros(len(actions), np.bool_)
    if bool(settings.get("dagger_event_include_control_events", True)):
        return control_events | phase_events
    return phase_events


def balanced_event_progress_choice(
    rng: np.random.Generator,
    track_ids: np.ndarray,
    course_progress: np.ndarray,
    event_mask: np.ndarray,
    size: int,
    *,
    event_fraction: float,
    late_fraction: float,
    late_threshold: float,
) -> np.ndarray:
    """Balance tracks while reserving disjoint event and late-course strata."""

    track_ids = np.asarray(track_ids)
    course_progress = np.asarray(course_progress, np.float32).reshape(-1)
    event_mask = np.asarray(event_mask, np.bool_).reshape(-1)
    if not (len(track_ids) == len(course_progress) == len(event_mask)):
        raise ValueError("event-balanced replay arrays must remain aligned")
    if not 0.0 <= event_fraction <= 1.0:
        raise ValueError("invalid DAgger event replay fraction")
    if not 0.0 <= late_fraction <= 1.0 or event_fraction + late_fraction > 1.0:
        raise ValueError("event and late replay fractions must sum to <= 1")
    if not 0.0 <= late_threshold <= 1.0:
        raise ValueError("invalid late-course replay threshold")
    indices = np.arange(len(track_ids))
    events = indices[event_mask]
    late = indices[(~event_mask) & (course_progress >= late_threshold)]
    ordinary = indices[(~event_mask) & (course_progress < late_threshold)]
    event_count = int(round(size * event_fraction)) if len(events) else 0
    late_count = int(round(size * late_fraction)) if len(late) else 0
    if event_count + late_count > size:
        late_count = size - event_count
    ordinary_count = size - event_count - late_count
    selected: list[np.ndarray] = []
    for source, count in (
        (events, event_count), (late, late_count), (ordinary, ordinary_count),
    ):
        if count:
            fallback = source if len(source) else indices
            selected.append(balanced_choice(rng, fallback, track_ids, count))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


def balanced_family_trajectory_choice(
    rng: np.random.Generator,
    family_ids: np.ndarray,
    track_ids: np.ndarray,
    gate_indices: np.ndarray,
    event_mask: np.ndarray,
    teacher_modes: np.ndarray,
    occupancy_modes: np.ndarray,
    size: int,
    *,
    family_weights: Mapping[int, float] | None = None,
    nominal_fraction: float = 0.40,
    critical_fraction: float = 0.35,
    recovery_fraction: float = 0.25,
    plan: HierarchicalReplayPlan | None = None,
) -> np.ndarray:
    """Sample family, concrete track, trajectory stratum, then active gate.

    This prevents long/failure-heavy families from owning a batch.  The three
    strata are disjoint by priority: critical control/gate events, recovery
    controller or cold-reset occupancy, and the remaining nominal-fast states.
    Empty strata fall back to the complete family pool rather than changing the
    requested family quota.
    """

    if plan is not None:
        if plan.size != size:
            raise ValueError("cached replay plan quota changed within an update phase")
        return plan.sample(rng)
    return HierarchicalReplayPlan(
        family_ids, track_ids, gate_indices, event_mask, teacher_modes, occupancy_modes,
        size, family_weights=family_weights, nominal_fraction=nominal_fraction,
        critical_fraction=critical_fraction, recovery_fraction=recovery_fraction,
    ).sample(rng)


def stratified_replay_subsample(
    rng: np.random.Generator,
    track_ids: np.ndarray,
    gate_indices: np.ndarray,
    event_mask: np.ndarray,
    teacher_modes: np.ndarray,
    occupancy_modes: np.ndarray,
    maximum_rows: int,
) -> np.ndarray:
    """Retain a unique, concrete-track-balanced subset of one DAgger round.

    Dense 90 Hz rollouts contain heavy temporal redundancy.  Keeping every row
    turns a bounded replay into an approximately eight-round FIFO.  This
    selector keeps equal concrete-track quotas and spreads each quota across
    nominal, critical, and recovery occupancy before filling any spare slots.
    It therefore extends replay history without increasing resident memory.
    """

    arrays = [
        np.asarray(track_ids).reshape(-1),
        np.asarray(gate_indices).reshape(-1),
        np.asarray(event_mask, np.bool_).reshape(-1),
        np.asarray(teacher_modes).reshape(-1),
        np.asarray(occupancy_modes).reshape(-1),
    ]
    if len({len(item) for item in arrays}) != 1:
        raise ValueError("stratified replay arrays must remain aligned")
    rows = len(arrays[0])
    if maximum_rows <= 0 or maximum_rows >= rows:
        return np.arange(rows, dtype=np.int64)
    selected: list[int] = []
    remaining = set(range(rows))
    tracks = np.unique(arrays[0])
    quotas = np.full(len(tracks), maximum_rows // len(tracks), np.int64)
    quotas[:maximum_rows % len(tracks)] += 1
    for track, quota in zip(tracks, quotas):
        track_pool = np.flatnonzero(arrays[0] == track)
        strata = (
            track_pool[
                (~arrays[2][track_pool])
                & (arrays[3][track_pool] == 0)
                & (arrays[4][track_pool] != 1)
            ],
            track_pool[arrays[2][track_pool]],
            track_pool[
                (~arrays[2][track_pool])
                & ((arrays[3][track_pool] != 0) | (arrays[4][track_pool] == 1))
            ],
        )
        stratum_quotas = np.asarray([0.40, 0.35, 0.25]) * int(quota)
        counts = np.floor(stratum_quotas).astype(np.int64)
        counts[:int(quota - counts.sum())] += 1
        for pool, count in zip(strata, counts):
            available = np.asarray([item for item in pool if item in remaining])
            take = min(int(count), len(available))
            if take:
                chosen = rng.choice(available, take, replace=False)
                selected.extend(int(item) for item in chosen)
                remaining.difference_update(int(item) for item in chosen)
        shortfall = int(quota) - sum(arrays[0][item] == track for item in selected)
        if shortfall > 0:
            available = np.asarray([
                item for item in track_pool if item in remaining
            ], np.int64)
            take = min(shortfall, len(available))
            if take:
                chosen = rng.choice(available, take, replace=False)
                selected.extend(int(item) for item in chosen)
                remaining.difference_update(int(item) for item in chosen)
    if len(selected) < maximum_rows:
        available = np.fromiter(remaining, np.int64)
        chosen = rng.choice(
            available, maximum_rows - len(selected), replace=False,
        )
        selected.extend(int(item) for item in chosen)
    result = np.asarray(selected, np.int64)
    rng.shuffle(result)
    if len(result) != maximum_rows or len(np.unique(result)) != len(result):
        raise RuntimeError("stratified replay subsample lost size or uniqueness")
    return result


def dagger_episode_tracks(
    tracks: tuple[str, ...], episodes: int, settings: Mapping[str, Any], seed: int,
) -> list[str]:
    """Build a deterministic, coverage-preserving DAgger track schedule.

    Family weighting is deliberately confined to rollout collection. Offline
    and online gradient batches remain balanced by concrete track ID, so extra
    failure-state collection cannot silently turn into task-frequency bias.
    """

    if episodes < len(tracks):
        raise ValueError("DAgger episodes_per_round must cover every track")
    if bool(settings.get("dagger_family_total_balanced_sampling", False)):
        manifest_path = settings.get("track_manifest")
        if manifest_path is None:
            raise ValueError("family-total DAgger sampling requires track_manifest")
        family_by_track = manifest_track_values(
            str(manifest_path), "family", cast=str,
        )
        resolved = tuple(str(Path(track).resolve()) for track in tracks)
        missing = [track for track in resolved if track not in family_by_track]
        if missing:
            raise ValueError(
                f"family-total DAgger sampling lacks metadata for {missing}"
            )
        members: dict[str, list[int]] = {}
        for index, track in enumerate(resolved):
            members.setdefault(family_by_track[track], []).append(index)
        configured = {
            str(key): float(value)
            for key, value in dict(
                settings.get("track_sampling_family_weights", {})
            ).items()
        }
        families = tuple(sorted(members))
        family_weights = np.asarray(
            [configured.get(family, 1.0) for family in families], np.float64
        )
        if (
            not np.all(np.isfinite(family_weights))
            or np.any(family_weights <= 0.0)
        ):
            raise ValueError("DAgger family weights must be finite and positive")
        minimum = np.asarray([len(members[family]) for family in families], np.int64)
        raw = episodes * family_weights / family_weights.sum()
        family_quotas = np.maximum(np.floor(raw).astype(np.int64), minimum)
        while int(family_quotas.sum()) > episodes:
            candidates = np.flatnonzero(family_quotas > minimum)
            if not len(candidates):
                raise ValueError("family-total DAgger coverage exceeds episode budget")
            index = int(candidates[np.argmax(family_quotas[candidates] - raw[candidates])])
            family_quotas[index] -= 1
        while int(family_quotas.sum()) < episodes:
            index = int(np.argmax(raw - family_quotas))
            family_quotas[index] += 1
        rng = np.random.default_rng(seed)
        schedule: list[str] = []
        for family, family_quota in zip(families, family_quotas.tolist()):
            indices = np.asarray(members[family], np.int64)
            base, residual = divmod(int(family_quota), len(indices))
            quotas = np.full(len(indices), base, np.int64)
            order = rng.permutation(len(indices))
            quotas[order[:residual]] += 1
            schedule.extend(
                tracks[index]
                for index, count in zip(indices.tolist(), quotas.tolist())
                for _ in range(count)
            )
        rng.shuffle(schedule)
        return schedule
    family_weights = {
        str(key): float(value)
        for key, value in dict(settings.get("track_sampling_family_weights", {})).items()
    }
    exact_weights = {
        str(key): float(value)
        for key, value in dict(settings.get("track_sampling_weights", {})).items()
    }
    weights = np.asarray([
        exact_weights.get(
            track,
            next(
                (weight for family, weight in family_weights.items() if family in Path(track).stem),
                1.0,
            ),
        )
        for track in tracks
    ], np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("DAgger track sampling weights must be finite and positive")
    remaining = episodes - len(tracks)
    quotas = np.ones(len(tracks), np.int64)
    if remaining:
        raw = remaining * weights / weights.sum()
        extra = np.floor(raw).astype(np.int64)
        quotas += extra
        residual = remaining - int(extra.sum())
        order = np.argsort(-(raw - extra), kind="stable")
        quotas[order[:residual]] += 1
    schedule = [
        track for track, count in zip(tracks, quotas.tolist()) for _ in range(count)
    ]
    np.random.default_rng(seed).shuffle(schedule)
    return schedule


def dagger_episode_start_gates(
    schedule: tuple[str, ...] | list[str], settings: Mapping[str, Any], seed: int,
) -> list[int | None]:
    """Choose deterministic canonical or targeted transition starts.

    Target gate indices are trainer-only curriculum metadata. They are never
    exposed to the actor. Family-specific fractions preserve canonical starts
    while injecting recovery occupancy immediately before known weak phases.
    """

    manifest_path = settings.get("track_manifest")
    if manifest_path is None:
        raise ValueError("targeted DAgger starts require track_manifest")
    configured = dict(settings.get("dagger_transition_start_gate_indices", {}))
    per_track_indices = manifest_track_values(
        str(manifest_path), "flight_mimic_start_gates",
        cast=lambda values: tuple(int(item) for item in values),
    )
    if not configured and not per_track_indices:
        return [None] * len(schedule)
    family_by_track = manifest_track_values(
        str(manifest_path), "family", cast=str,
    )
    fractions = {
        str(key): float(value)
        for key, value in dict(
            settings.get("dagger_transition_start_fraction_by_family", {})
        ).items()
    }
    default_fraction = float(settings.get(
        "dagger_transition_start_episode_fraction", 0.0
    ))
    gate_weights = {
        str(family): {
            int(gate): float(weight)
            for gate, weight in dict(weights).items()
        }
        for family, weights in dict(settings.get(
            "dagger_transition_start_gate_weights_by_family", {}
        )).items()
    }
    result: list[int | None] = []
    for episode_index, raw_track in enumerate(schedule):
        track = str(Path(str(raw_track)).resolve())
        if track not in family_by_track:
            raise ValueError(f"targeted DAgger start lacks family for {track}")
        family = family_by_track[track]
        raw_indices = per_track_indices.get(track, configured.get(family, ()))
        indices = tuple(int(item) for item in raw_indices)
        if any(item < 0 for item in indices) or len(indices) != len(set(indices)):
            raise ValueError(
                f"targeted DAgger gates for {family} must be unique non-negative indices"
            )
        fraction = fractions.get(family, default_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"targeted DAgger fraction for {family} must be in [0,1]"
            )
        rng = np.random.default_rng(
            int(seed) + 104729 * (episode_index + 1) + 1009
        )
        if indices and rng.random() < fraction:
            configured_weights = gate_weights.get(family, {})
            weights = np.asarray(
                [configured_weights.get(item, 1.0) for item in indices],
                np.float64,
            )
            if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
                raise ValueError(
                    f"targeted DAgger gate weights for {family} must be "
                    "finite and positive"
                )
            result.append(int(rng.choice(indices, p=weights / weights.sum())))
        else:
            result.append(None)
    return result


def dagger_transition_start_mode(settings: Mapping[str, Any]) -> str:
    """Resolve how a targeted DAgger episode reaches its handoff gate."""

    mode = str(settings.get("dagger_transition_start_mode", "cold_reset"))
    if mode not in {"cold_reset", "expert_prefix"}:
        raise ValueError(
            "dagger_transition_start_mode must be cold_reset or expert_prefix"
        )
    return mode


DAGGER_OCCUPANCY_MODE_IDS = {
    "canonical": 0,
    "cold_reset": 1,
    "expert_prefix": 2,
}


def dagger_episode_start_modes(
    schedule: tuple[str, ...] | list[str],
    start_gates: tuple[int | None, ...] | list[int | None],
    settings: Mapping[str, Any],
    seed: int,
) -> list[str]:
    """Choose a deterministic per-episode occupancy source.

    Canonical episodes keep the natural race start. Targeted episodes may mix
    cold gate-local recovery resets with expert-flown prefixes that preserve a
    physically realized incoming trajectory and causal history.
    """

    if len(schedule) != len(start_gates):
        raise ValueError("DAgger start gates and track schedule must align")
    configured = settings.get("dagger_transition_start_mode_weights")
    if configured is None:
        default = dagger_transition_start_mode(settings)
        weights_by_name = {default: 1.0}
    else:
        weights_by_name = {
            str(name): float(value) for name, value in dict(configured).items()
        }
    if (
        not weights_by_name
        or any(name not in {"cold_reset", "expert_prefix"} for name in weights_by_name)
        or any(not np.isfinite(value) or value <= 0.0 for value in weights_by_name.values())
    ):
        raise ValueError(
            "dagger_transition_start_mode_weights must contain positive "
            "cold_reset/expert_prefix weights"
        )
    names = tuple(weights_by_name)
    probabilities = np.asarray([weights_by_name[name] for name in names], np.float64)
    probabilities /= probabilities.sum()
    modes: list[str] = []
    for episode_index, gate_index in enumerate(start_gates):
        if gate_index is None:
            modes.append("canonical")
            continue
        rng = np.random.default_rng(
            int(seed) + 130363 * (episode_index + 1) + 9176
        )
        modes.append(str(rng.choice(names, p=probabilities)))
    return modes


def dagger_expert_prefix_forces_teacher(
    requested_gate_index: int | None,
    passed_count: int,
    start_passed: int,
) -> bool:
    """Return whether MPCC must still advance a natural causal prefix.

    A target index of two means that MPCC flies gates zero and one, then the
    learner first acts while approaching gate two. The actor history is thus
    populated by the exact states and executed actions that led there.
    """

    if requested_gate_index is None:
        return False
    if requested_gate_index < 0:
        raise ValueError("expert-prefix gate index must be non-negative")
    return int(passed_count) - int(start_passed) < int(requested_gate_index)


def dagger_teacher_beta(
    round_index: int,
    settings: Mapping[str, Any],
    *,
    refresh_permanent_expert: bool = False,
) -> float:
    """Resolve expert execution without stretching a proven anneal.

    ``teacher_beta_schedule_rounds`` defaults to the total run length, exactly
    preserving the historical linear schedule.  A smaller explicit value lets
    a longer run reach the old terminal occupancy at the same round and hold it
    there for a controlled scale experiment.
    """

    from starscream.dagger_schedule import recovery_anneal, dart_collection_round
    if dart_collection_round(round_index, settings):
        return 1.0
    anneal = recovery_anneal(round_index, settings)
    if anneal:
        if refresh_permanent_expert:
            raise ValueError("recovery anneal does not use permanent refresh")
        return anneal['teacher_beta']
    total_rounds = int(settings.get("rounds", 8))
    schedule_rounds = int(settings.get(
        "teacher_beta_schedule_rounds", total_rounds
    ))
    if round_index < 1 or total_rounds < 1:
        raise ValueError("DAgger round indices and total rounds must be positive")
    if schedule_rounds < 2:
        raise ValueError("teacher_beta_schedule_rounds must be at least two")
    permanent_rounds = int(settings.get("dagger_permanent_expert_rounds", 0))
    if refresh_permanent_expert or round_index <= permanent_rounds:
        return 1.0
    fraction = min(1.0, (round_index - 1) / (schedule_rounds - 1))
    start = float(settings.get("teacher_beta_start", 0.35))
    end = float(settings.get("teacher_beta_end", 0.05))
    beta = start * (1.0 - fraction) + end * fraction
    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("DAgger teacher beta must remain in [0,1]")
    return collection_beta(beta, collection_config(settings))


def dagger_track_allows_dart(track: str, settings: Mapping[str, Any]) -> bool:
    """Return false for courses whose MPCC qualification excludes perturbations."""
    resolved = str(Path(track).resolve())
    exempt = {str(Path(item).resolve()) for item in settings.get("dagger_dart_exempt_tracks", ())}
    return resolved not in exempt


def dagger_successful_episode_filter(
    round_index: int,
    settings: Mapping[str, Any],
    *,
    refresh_permanent_expert: bool = False,
) -> bool:
    """Limit full-lap-only replay to the explicitly qualified warmup.

    Historical configs used ``dagger_require_successful_episodes`` as an
    all-round switch. That is useful while constructing the permanent expert
    anchor, but after learner actions enter the rollout it discards exactly the
    failed states DAgger is meant to relabel. The optional round limit keeps
    historical behavior by default while allowing ordinary failure-state
    DAgger after a clean expert-only warmup.
    """

    from starscream.dagger_schedule import recovery_anneal
    if recovery_anneal(round_index, settings).get('pure_expert'):
        return True
    if not bool(settings.get("dagger_require_successful_episodes", False)):
        return False
    configured = settings.get("dagger_successful_coverage_rounds")
    if configured is None:
        return True
    rounds = int(configured)
    if rounds < 0:
        raise ValueError("dagger_successful_coverage_rounds must be non-negative")
    return bool(refresh_permanent_expert or round_index <= rounds)


def configure_dagger_action_head(
    policy: PrivilegedMLPPolicy,
    settings: Mapping[str, Any],
    action_chunk_offsets: tuple[int, ...],
) -> None:
    """Upgrade and validate optional privileged action objectives."""

    action_chunk_weight = float(settings.get("action_chunk_weight", 0.0))
    action_chunk_source = str(settings.get(
        "action_chunk_target_source", "mpcc_open_loop"
    ))
    if action_chunk_source not in {"mpcc_open_loop", "replanned_teacher", "coherent_teacher"}:
        raise ValueError(
            "action_chunk_target_source must be mpcc_open_loop or replanned_teacher"
        )
    if action_chunk_weight > 0.0 and action_chunk_source in {"replanned_teacher", "coherent_teacher"}:
        if str(settings.get("collector_backend", "process")) != "process":
            raise ValueError(
                "replanned teacher action chunks require the process collector"
            )
        if action_chunk_offsets != tuple(range(len(action_chunk_offsets))):
            raise ValueError(
                "replanned teacher chunks require contiguous offsets from zero"
            )

    flow_head_settings = settings.get("flow_action_head")
    mixture_head_settings = settings.get("mixture_action_head")
    flow_enabled = isinstance(flow_head_settings, Mapping) and bool(
        flow_head_settings.get("enabled", True)
    )
    mixture_enabled = isinstance(mixture_head_settings, Mapping) and bool(
        mixture_head_settings.get("enabled", True)
    )
    if flow_enabled and mixture_enabled:
        raise ValueError("flow and categorical-residual action heads are exclusive")
    if flow_enabled:
        policy.enable_shortcut_flow_action_head(
            action_horizon=int(flow_head_settings.get(
                "action_horizon", len(action_chunk_offsets)
            )),
            depth=int(flow_head_settings.get("depth", 3)),
            heads=int(flow_head_settings.get("heads", 8)),
            mlp_ratio=int(flow_head_settings.get("mlp_ratio", 4)),
            sampling_steps=int(flow_head_settings.get("sampling_steps", 3)),
            sampling_method=str(flow_head_settings.get(
                "sampling_method", "euler"
            )),
            source_noise=float(flow_head_settings.get("source_noise", 0.25)),
            shortcut_step_sizes=tuple(float(item) for item in flow_head_settings.get(
                "shortcut_step_sizes", (1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0)
            )),
        )
    if mixture_enabled:
        codebook_path = Path(str(mixture_head_settings.get("codebook", "")))
        if not codebook_path.is_file():
            raise FileNotFoundError(
                f"action mixture codebook not found: {codebook_path}"
            )
        with np.load(codebook_path, allow_pickle=False) as codebook:
            if "centers" not in codebook:
                raise ValueError("action mixture codebook has no centers")
            centers = np.asarray(codebook["centers"], np.float32)
            cluster_weights = (
                np.asarray(codebook["cluster_weights"], np.float32)
                if "cluster_weights" in codebook else None
            )
        policy.enable_action_mixture_head(
            centers,
            cluster_weights,
            residual_logit_scale=float(mixture_head_settings.get(
                "residual_logit_scale", 2.0
            )),
        )
        if float(mixture_head_settings.get("mode_weight", 0.0)) <= 0.0:
            raise ValueError("action mixture DAgger requires mode_weight > 0")
        if action_chunk_offsets not in {(), (0,)}:
            raise ValueError("action mixture DAgger is single-step only")
    if action_chunk_source == 'coherent_teacher' and policy.action_head_type != 'mlp':
        raise ValueError('coherent teacher chunks currently support only the direct MLP action head')
    if policy.action_head_type != "shortcut_flow":
        return
    if action_chunk_weight <= 0.0:
        raise ValueError("shortcut-flow DAgger requires action_chunk_weight > 0")
    if action_chunk_offsets != tuple(range(policy.action_chunk_steps)):
        raise ValueError(
            "shortcut-flow DAgger requires contiguous action chunk offsets"
        )
    if policy.flow_sampling_steps < 1:
        raise ValueError("shortcut-flow sampling steps must be positive")
    if policy.flow_context_mode == 'unified_route_dit':
        if float(settings.get('shortcut_bootstrap_weight',1.)) != 0. or float(settings.get('shortcut_direct_weight',.5)) != 1.:
            raise ValueError('unified route DiT uses continuous-time CFM, not shortcut forcing')
        if float(settings.get('dagger_anchor_distillation_weight',0.)) > 0.:
            raise ValueError('MLP anchor loss is not a flow likelihood')
        return
    deployment_step = 1.0 / policy.flow_sampling_steps
    shortcut_steps = tuple(policy.flow_shortcut_step_sizes)
    if not shortcut_steps or not any(
        abs(step - deployment_step) <= 1.0e-9 for step in shortcut_steps
    ):
        raise ValueError(
            "shortcut-flow training step sizes must include the deployed "
            f"integration step {deployment_step:g}"
        )
    if max(shortcut_steps) > deployment_step + 1.0e-9:
        raise ValueError(
            "shortcut-flow training cannot use a step larger than the "
            "deployed integration step"
        )
    if float(settings.get("dagger_anchor_distillation_weight", 0.0)) > 0.0:
        raise ValueError("MLP anchor distillation is incompatible with flow DAgger")


def configured_ppo_track_weights(
    tracks: Sequence[str], settings: Mapping[str, Any],
) -> np.ndarray:
    """Resolve explicit/family PPO weights against concrete manifest paths.

    These weights are sampler metadata only. They never enter the actor or
    critic observation. Family lookup uses manifest metadata rather than
    filename substring matching so a renamed materialized course cannot
    silently change the optimization distribution.
    """

    family_weights = {
        str(key): float(value)
        for key, value in dict(
            settings.get("ppo_track_sampling_family_weights", {})
        ).items()
    }
    exact_weights = {
        str(key): float(value)
        for key, value in dict(settings.get("ppo_track_sampling_weights", {})).items()
    }
    family_by_track: dict[str, str] = {}
    manifest_path = settings.get("track_manifest")
    if manifest_path is not None:
        family_by_track = manifest_track_values(
            str(manifest_path), "family", cast=str,
        )
    values: list[float] = []
    for raw_track in tracks:
        resolved = str(Path(str(raw_track)).resolve())
        name = Path(resolved).stem
        family = family_by_track.get(resolved, "")
        value = exact_weights.get(
            resolved,
            exact_weights.get(
                str(raw_track),
                exact_weights.get(name, family_weights.get(family, 1.0)),
            ),
        )
        values.append(float(value))
    weights = np.asarray(values, np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("PPO track weights must be finite and positive")
    return weights


def ppo_episode_tracks(
    tracks: Sequence[str], episodes: int, settings: Mapping[str, Any], seed: int,
    weights_override: np.ndarray | None = None,
) -> list[str]:
    """Return a deterministic weighted, shuffled, coverage-first schedule."""

    tracks = tuple(str(track) for track in tracks)
    if not tracks or episodes < 1:
        raise ValueError("PPO scheduling requires tracks and positive episodes")
    rng = np.random.default_rng(seed)
    weights = (
        configured_ppo_track_weights(tracks, settings)
        if weights_override is None
        else np.asarray(weights_override, np.float64)
    )
    if weights.shape != (len(tracks),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("PPO scheduling weights must be finite, positive, and match tracks")
    if episodes < len(tracks):
        selected = rng.choice(
            len(tracks), size=episodes, replace=False, p=weights / weights.sum()
        )
        schedule = [tracks[int(index)] for index in selected]
        rng.shuffle(schedule)
        return schedule
    quotas = np.ones(len(tracks), np.int64)
    remaining = episodes - len(tracks)
    if remaining:
        raw = remaining * weights / weights.sum()
        extra = np.floor(raw).astype(np.int64)
        quotas += extra
        residual = remaining - int(extra.sum())
        order = np.argsort(-(raw - extra), kind="stable")
        quotas[order[:residual]] += 1
    schedule = [
        track for track, count in zip(tracks, quotas.tolist())
        for _ in range(count)
    ]
    rng.shuffle(schedule)
    return schedule


def adaptive_frontier_weights(
    base_weights: np.ndarray,
    previous_competence: np.ndarray,
    observed_competence: np.ndarray,
    observed_advantage: np.ndarray,
    *,
    blend: float,
    ema: float,
    target: float,
    width: float,
    minimum_multiplier: float,
    maximum_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prioritize tracks near the learning frontier without chasing failures."""

    base = np.asarray(base_weights, np.float64)
    previous = np.asarray(previous_competence, np.float64)
    observed = np.asarray(observed_competence, np.float64)
    novelty = np.asarray(observed_advantage, np.float64)
    if not (base.shape == previous.shape == observed.shape == novelty.shape):
        raise ValueError("adaptive PPO arrays must have identical shapes")
    if (
        not 0.0 <= blend <= 1.0 or not 0.0 < ema <= 1.0
        or not 0.0 <= target <= 1.0 or width <= 0.0
        or minimum_multiplier <= 0.0 or maximum_multiplier < minimum_multiplier
    ):
        raise ValueError("invalid adaptive PPO frontier settings")
    updated = np.where(
        np.isfinite(observed),
        (1.0 - ema) * previous + ema * observed,
        previous,
    )
    progress = np.abs(updated - previous)
    frontier = np.exp(-0.5 * np.square((updated - target) / width))
    finite_novelty = np.where(np.isfinite(novelty), np.maximum(novelty, 0.0), 0.0)
    novelty_scale = np.percentile(finite_novelty, 75.0)
    novelty_normalized = finite_novelty / max(float(novelty_scale), 1.0e-6)
    score = (
        0.55 * frontier
        + 0.30 * np.minimum(progress / max(ema, 1.0e-6), 1.0)
        + 0.15 * np.minimum(novelty_normalized, 2.0)
    )
    score /= max(float(np.mean(score)), 1.0e-6)
    multiplier = np.clip(
        (1.0 - blend) + blend * score,
        minimum_multiplier,
        maximum_multiplier,
    )
    return base * multiplier, updated, multiplier


def dagger_family_survival_statistics(
    evaluation: Mapping[str, Any], manifest_path: str | Path, *,
    route_horizon: int = 6,
) -> tuple[dict[str, float], dict[str, int]]:
    """Reduce canonical closed-loop curves to source-family competence/frontiers.

    The competence target is deliberately local: mean survival through the
    actor's route horizon.  Full-lap success would make long tracks look hard
    merely because they contain more gates.  The transition frontier is the
    active gate at the steepest survival drop and is trainer-only metadata.
    """

    if route_horizon < 2:
        raise ValueError("dynamic DAgger route_horizon must be at least two")
    records = read_manifest(Path(manifest_path))["records"]
    competence_values: dict[str, list[float]] = {}
    frontier_values: dict[str, list[int]] = {}
    for record in records:
        if str(record.get("source_stratum", "")) != "canonical":
            continue
        source = str(record.get("source_family", ""))
        name = str(record.get("name", ""))
        if not source or not name:
            continue
        prefix = f"track/{name}/"
        survival = [
            float(evaluation[f"{prefix}p{gate}"])
            for gate in range(1, route_horizon + 1)
            if f"{prefix}p{gate}" in evaluation
        ]
        if not survival:
            continue
        survival_array = np.clip(np.asarray(survival, np.float64), 0.0, 1.0)
        competence_values.setdefault(source, []).append(float(survival_array.mean()))
        # p_k - p_{k+1} is failure while acquiring zero-indexed gate k.
        # Gate zero cannot be a transition start, so canonical starts retain it.
        if len(survival_array) > 1:
            drops = survival_array[:-1] - survival_array[1:]
            frontier = max(1, int(np.argmax(drops)) + 1)
        else:
            frontier = 1
        available = tuple(int(item) for item in record.get(
            "flight_mimic_start_gates", ()
        ))
        if available:
            frontier = min(available, key=lambda item: abs(item - frontier))
        frontier_values.setdefault(source, []).append(int(frontier))
    competence = {
        source: float(np.mean(values))
        for source, values in competence_values.items()
    }
    frontiers = {
        source: int(round(float(np.median(values))))
        for source, values in frontier_values.items()
    }
    return competence, frontiers


def update_dynamic_dagger_sampler(
    *,
    base_family_weights: Mapping[str, float],
    source_by_family: Mapping[str, str],
    observed_competence: Mapping[str, float],
    observed_frontiers: Mapping[str, int],
    observed_losses: Mapping[str, float],
    previous_state: Mapping[str, Any] | None,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], dict[str, dict[int, float]]]:
    """Build a stable competence-frontier sampler for the next DAgger round.

    Closed-loop behavior is the primary signal; replay objective is a bounded
    tie-breaker.  Source-family multipliers preserve the configured
    canonical/augmentation ratio and hard floors/caps prevent forgetting or
    a single unsolved family from consuming collection.
    """

    if config.get("behavior_map"):
        return update_behavior_sampler(
            update_dynamic_dagger_sampler, base_family_weights=base_family_weights,
            source_by_family=source_by_family, observed_losses=observed_losses,
            previous_state=previous_state, config=config)
    sources = tuple(sorted(set(source_by_family.values())))
    if not sources:
        raise ValueError("dynamic DAgger sampling requires source families")
    previous = dict(previous_state or {})
    old_competence = {
        source: float(dict(previous.get("competence", {})).get(
            source, observed_competence.get(source, 0.0)
        ))
        for source in sources
    }
    old_losses = {
        source: float(dict(previous.get("loss", {})).get(
            source, observed_losses.get(source, 1.0)
        ))
        for source in sources
    }
    old_multipliers = {
        source: float(dict(previous.get("multipliers", {})).get(source, 1.0))
        for source in sources
    }
    old_frontiers = {
        source: float(dict(previous.get("frontiers", {})).get(
            source, observed_frontiers.get(source, 1)
        ))
        for source in sources
    }
    competence_ema = float(config.get("competence_ema", 0.30))
    loss_ema = float(config.get("loss_ema", 0.25))
    multiplier_ema = float(config.get("multiplier_ema", 0.40))
    if not all(0.0 < item <= 1.0 for item in (
        competence_ema, loss_ema, multiplier_ema,
    )):
        raise ValueError("dynamic DAgger EMA values must be in (0,1]")
    competence: dict[str, float] = {}
    losses: dict[str, float] = {}
    progress: dict[str, float] = {}
    frontiers: dict[str, float] = {}
    for source in sources:
        observed = float(observed_competence.get(source, old_competence[source]))
        observed_loss = float(observed_losses.get(source, old_losses[source]))
        if not np.isfinite(observed) or not 0.0 <= observed <= 1.0:
            observed = old_competence[source]
        if not np.isfinite(observed_loss) or observed_loss <= 0.0:
            observed_loss = old_losses[source]
        competence[source] = (
            (1.0 - competence_ema) * old_competence[source]
            + competence_ema * observed
        )
        losses[source] = (
            (1.0 - loss_ema) * old_losses[source] + loss_ema * observed_loss
        )
        progress[source] = max(observed - old_competence[source], 0.0)
        frontier_observed = float(observed_frontiers.get(
            source, old_frontiers[source]
        ))
        frontiers[source] = (
            (1.0 - competence_ema) * old_frontiers[source]
            + competence_ema * frontier_observed
        )

    median_loss = max(float(np.median(list(losses.values()))), 1.0e-8)
    target = float(config.get("target_competence", 0.55))
    width = float(config.get("frontier_width", 0.28))
    progress_scale = float(config.get("progress_scale", 0.12))
    if not 0.0 <= target <= 1.0 or width <= 0.0 or progress_scale <= 0.0:
        raise ValueError("invalid dynamic DAgger frontier settings")
    scores: dict[str, float] = {}
    for source in sources:
        deficit = 1.0 - competence[source]
        frontier_score = math.exp(
            -0.5 * ((competence[source] - target) / width) ** 2
        )
        learning_progress = min(progress[source] / progress_scale, 1.0)
        loss_ratio = min(max(losses[source] / median_loss, 0.5), 2.0) / 2.0
        scores[source] = (
            0.55 * deficit
            + 0.20 * frontier_score
            + 0.15 * learning_progress
            + 0.10 * loss_ratio
        )
    score_mean = float(np.mean(list(scores.values())))
    priority_temperature = float(config.get("priority_temperature", 0.18))
    if priority_temperature <= 0.0:
        raise ValueError("dynamic DAgger priority_temperature must be positive")
    relative_priority = {
        source: math.exp(
            np.clip(
                (scores[source] - score_mean) / priority_temperature,
                -8.0, 8.0,
            )
        )
        for source in sources
    }
    relative_mean = max(
        float(np.mean(list(relative_priority.values()))), 1.0e-8
    )
    blend = float(config.get("blend", 0.75))
    minimum = float(config.get("minimum_multiplier", 0.55))
    maximum = float(config.get("maximum_multiplier", 1.80))
    if not 0.0 <= blend <= 1.0 or minimum <= 0.0 or maximum < minimum:
        raise ValueError("invalid dynamic DAgger multiplier constraints")
    multipliers: dict[str, float] = {}
    for source in sources:
        desired = np.clip(
            (1.0 - blend) + blend * relative_priority[source] / relative_mean,
            minimum, maximum,
        )
        multipliers[source] = float(np.clip(
            (1.0 - multiplier_ema) * old_multipliers[source]
            + multiplier_ema * desired,
            minimum, maximum,
        ))
    family_weights = {
        family: float(base_family_weights[family]) * multipliers[source_by_family[family]]
        for family in base_family_weights
    }
    gate_floor = float(config.get("gate_weight_floor", 0.20))
    gate_sigma = float(config.get("gate_weight_sigma", 0.85))
    if gate_floor <= 0.0 or gate_sigma <= 0.0:
        raise ValueError("dynamic DAgger gate weights must be positive")
    gate_weights = {
        family: {
            gate: gate_floor + math.exp(
                -0.5 * ((gate - frontiers[source_by_family[family]]) / gate_sigma) ** 2
            )
            for gate in range(1, int(config.get("maximum_gate_index", 64)) + 1)
        }
        for family in base_family_weights
    }
    probabilities = np.asarray([
        multipliers[source] for source in sources
    ], np.float64)
    probabilities /= probabilities.sum()
    state = {
        "competence": competence,
        "loss": losses,
        "multipliers": multipliers,
        "frontiers": frontiers,
        "family_weights": family_weights,
        "gate_weights": gate_weights,
        "priority_entropy": -float(np.sum(
            probabilities * np.log(probabilities + 1.0e-12)
        )),
    }
    return state, family_weights, gate_weights


def batch_sequences(
    features: np.ndarray,
    episode_ids: np.ndarray,
    indices: np.ndarray,
    context_steps: int,
) -> np.ndarray:
    return features[sequence_indices(episode_ids, indices, context_steps)]


def topology_robust_objective(
    per_sample: torch.Tensor,
    group_ids: torch.Tensor | None,
    *,
    robust_weight: float,
    temperature: float,
    fast_statistics: bool = False,
    group_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smoothly emphasize the worst represented topology/pace/phase cell.

    ``tau * log(mean(exp(L_g / tau)))`` is a differentiable upper envelope
    that equals the common loss when all groups are tied and approaches the
    worst group as ``tau`` decreases.  Mixing it with ordinary ERM prevents a
    tiny hard cell from consuming the entire update.
    """

    if per_sample.ndim != 1 or not len(per_sample):
        raise ValueError("robust objective requires a non-empty loss vector")
    if not 0.0 <= robust_weight <= 1.0 or temperature <= 0.0:
        raise ValueError("invalid robust objective weight or temperature")
    mean_loss = per_sample.mean()
    if group_ids is None:
        one = mean_loss.new_tensor(1.0)
        return mean_loss, mean_loss, mean_loss, one
    groups = torch.as_tensor(group_ids, device=per_sample.device).reshape(-1)
    if groups.shape != per_sample.shape:
        raise ValueError("group IDs must align with per-sample losses")
    if fast_statistics and robust_weight == 0.0 and group_count is not None:
        # A CPU-validated replay bound avoids dynamic unique/nonzero and host
        # synchronization in every update. These statistics do not affect loss.
        from starscream.dagger_throughput import fixed_group_statistics
        return (mean_loss, *fixed_group_statistics(per_sample, groups, group_count))
    unique = torch.unique(groups[groups >= 0], sorted=True)
    if not len(unique):
        one = mean_loss.new_tensor(1.0)
        return mean_loss, mean_loss, mean_loss, one
    if fast_statistics and robust_weight == 0.0:
        # Reporting must not launch a dynamically sized gather and synchronize
        # once per course, nor backpropagate through an objective of weight zero.
        # The optimized objective and its gradient are exactly the existing ERM.
        with torch.no_grad():
            membership = groups.unsqueeze(0) == unique.unsqueeze(1)
            group_losses = torch.where(membership, per_sample.detach().unsqueeze(0), 0.).sum(-1)
            group_losses = group_losses / membership.sum(-1).clamp_min(1)
        return mean_loss, group_losses.mean(), group_losses.max(), group_losses.std(unbiased=False)
    group_losses = torch.stack([
        per_sample[groups == group].mean() for group in unique
    ])
    tau = per_sample.new_tensor(float(temperature))
    smooth_max = tau * (
        torch.logsumexp(group_losses / tau, dim=0)
        - math.log(len(group_losses))
    )
    objective = (
        (1.0 - float(robust_weight)) * mean_loss
        + float(robust_weight) * smooth_max
    )
    return objective, group_losses.mean(), group_losses.max(), group_losses.std(
        unbiased=False
    )


def bounded_replay_append(
    existing: np.ndarray, incoming: np.ndarray, capacity: int,
) -> np.ndarray:
    """Append without first allocating an uncapped old+new replay array."""

    if capacity < 1:
        raise ValueError("replay capacity must be positive")
    if existing.ndim != incoming.ndim or existing.shape[1:] != incoming.shape[1:]:
        raise ValueError(
            "replay arrays must have compatible trailing dimensions: "
            f"existing={existing.shape} incoming={incoming.shape}"
        )
    if len(incoming) >= capacity:
        return incoming[-capacity:]
    if len(existing) == capacity and existing.flags.writeable:
        # Once replay is full, rotate it in place.  The previous concatenate
        # path temporarily duplicated the entire history replay every round
        # (over a GiB for current 18x103 float16 settings).
        count = len(incoming)
        if count:
            existing[:-count] = existing[count:]
            existing[-count:] = incoming
        return existing
    keep = min(len(existing), capacity - len(incoming))
    if keep == 0:
        return incoming
    return np.concatenate([existing[-keep:], incoming], axis=0)


def imitation_loss(
    policy: PrivilegedMLPPolicy,
    histories: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    dynamics_targets: torch.Tensor,
    settings: Mapping[str, Any],
    dynamics_valid: torch.Tensor | None = None,
    speed_commands: torch.Tensor | None = None,
    topology_targets: torch.Tensor | None = None,
    topology_valid: torch.Tensor | None = None,
    group_ids: torch.Tensor | None = None,
    anchor_actions: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
    action_chunk_targets: torch.Tensor | None = None,
    action_chunk_valid: torch.Tensor | None = None,
    executed_actions: torch.Tensor | None = None,
    reward_component_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from starscream.imitation_objective import action_dimension_weights, masked_objective
    if policy.action_head_type == "shortcut_flow":
        if action_chunk_targets is None or action_chunk_valid is None:
            raise ValueError("shortcut-flow DAgger requires aligned action chunks")
        if not bool(torch.all(action_chunk_valid.to(torch.bool))):
            raise ValueError("shortcut-flow DAgger received an invalid action chunk")
        flow_context, flow_base, _ = policy._flow_context(
            histories, speed_commands
        )
        stabilization_weight = float(settings.get(
            "flow_stabilization_weight", 0.15
        ))
        flow_per_sample, dynamics, flow_metrics, stabilization_per_sample = (
            policy.flow_training_objective(
                histories,
                action_chunk_targets,
                speed_commands,
                direct_weight=float(settings.get("shortcut_direct_weight", 0.5)),
                bootstrap_weight=float(settings.get(
                    "shortcut_bootstrap_weight", 1.0
                )),
                stabilization_noise=float(settings.get(
                    "flow_stabilization_noise", 0.06
                )),
                stabilization_remaining_time=float(settings.get(
                    "flow_stabilization_remaining_time", 1.0 / 3.0
                )),
                stabilization_enabled=stabilization_weight > 0.0,
                context=flow_context,
                base=flow_base,
            )
        )
        dynamics_per_sample = F.smooth_l1_loss(
            dynamics, dynamics_targets,
            beta=float(settings.get("dynamics_huber_beta", 0.10)),
            reduction="none",
        ).mean(-1)
        if dynamics_valid is None:
            dynamics_loss = dynamics_per_sample.mean()
            dynamics_objective_per_sample = dynamics_per_sample
        else:
            dynamics_loss, dynamics_objective_per_sample = masked_objective(
                dynamics_per_sample, dynamics_valid
            )
        total_per_sample = (
            float(settings.get("flow_objective_weight", 1.0)) * flow_per_sample
            + stabilization_weight * stabilization_per_sample
            + float(settings.get("dynamics_weight", 0.15))
            * dynamics_objective_per_sample
        )
        total, group_loss_mean, group_loss_max, group_loss_std = (
            topology_robust_objective(
                total_per_sample,
                group_ids,
                robust_weight=float(settings.get(
                    "dagger_group_robust_weight", 0.0
                )),
                temperature=float(settings.get(
                    "dagger_group_robust_temperature", 0.10
                )),
                fast_statistics=bool(settings.get('dagger_fast_group_statistics', False)),
            )
        )
        # Deployment is exactly three deterministic Euler steps followed by
        # receding-horizon execution of command zero. Velocity-field fitting
        # alone does not guarantee that this numerical endpoint matches MPCC,
        # so supervise the actual deployment contract on a memory-bounded
        # sub-batch.
        deployment_endpoint_loss = total.new_zeros(())
        deployment_physical_loss = total.new_zeros(())
        deployment_collective_mae = total.new_zeros(())
        deployment_body_rate_mae = total.new_zeros(())
        deployment_weight = float(settings.get(
            "flow_deployment_endpoint_weight", 0.0
        ))
        deployment_physical_weight = float(settings.get(
            "flow_deployment_physical_weight",
            settings.get("physical_action_weight", 0.0),
        ))
        if deployment_weight > 0.0 or deployment_physical_weight > 0.0:
            deployment_fraction = float(settings.get(
                "flow_deployment_endpoint_fraction", 0.25
            ))
            if not 0.0 < deployment_fraction <= 1.0:
                raise ValueError(
                    "flow_deployment_endpoint_fraction must be in (0,1]"
                )
            endpoint_count = max(1, min(
                len(histories),
                int(round(len(histories) * deployment_fraction)),
            ))
            endpoint_speed = (
                None if speed_commands is None
                else speed_commands[:endpoint_count]
            )
            deployed_chunk = policy.differentiable_action_chunk(
                histories[:endpoint_count], endpoint_speed,
                context=flow_context[:endpoint_count],
            )
            endpoint_targets = action_chunk_targets[:endpoint_count]
            endpoint_error = F.smooth_l1_loss(
                deployed_chunk,
                endpoint_targets,
                beta=float(settings.get(
                    "flow_deployment_endpoint_huber_beta", 0.05
                )),
                reduction="none",
            ).mean(-1)
            horizon_weights = endpoint_error.new_ones(
                policy.action_chunk_steps
            )
            horizon_weights[0] = float(settings.get(
                "flow_deployment_first_action_weight", 2.0
            ))
            deployment_endpoint_loss = (
                endpoint_error * horizon_weights[None]
            ).sum(-1).div(horizon_weights.sum()).mean()
            deployed_physical = normalized_ctbr_tensor(deployed_chunk, settings)
            target_physical = normalized_ctbr_tensor(endpoint_targets, settings)
            physical_scales = deployed_physical.new_tensor(
                settings.get("physical_action_scales", [10.0, 3.0, 3.0, 3.0])
            )
            physical_error = F.smooth_l1_loss(
                deployed_physical / physical_scales,
                target_physical / physical_scales,
                beta=float(settings.get("physical_huber_beta", 0.10)),
                reduction="none",
            ).mean(-1)
            deployment_physical_loss = (
                physical_error * horizon_weights[None]
            ).sum(-1).div(horizon_weights.sum()).mean()
            physical_absolute = (deployed_physical - target_physical).abs()
            deployment_collective_mae = physical_absolute[..., 0].mean()
            deployment_body_rate_mae = physical_absolute[..., 1:4].mean()
            total = (
                total
                + deployment_weight * deployment_endpoint_loss
                + deployment_physical_weight * deployment_physical_loss
            )
        zero = total.new_zeros(())
        return total, {
            "loss": total,
            "action_loss": flow_metrics["flow_matching_loss"],
            "delta_loss": zero,
            "physical_action_loss": deployment_physical_loss,
            "physical_delta_loss": zero,
            "dynamics_loss": dynamics_loss,
            "topology_loss": zero,
            "plan_state_loss": zero,
            "action_chunk_loss": flow_per_sample.mean(),
            "anchor_loss": zero,
            "group_loss_mean": group_loss_mean,
            "group_loss_max": group_loss_max,
            "group_loss_std": group_loss_std,
            "flow_matching_loss": flow_metrics["flow_matching_loss"],
            "shortcut_bootstrap_loss": flow_metrics[
                "shortcut_bootstrap_loss"
            ],
            "flow_endpoint_mse": flow_metrics["endpoint_mse"],
            "flow_stabilization_loss": stabilization_per_sample.mean(),
            "flow_deployment_endpoint_loss": deployment_endpoint_loss,
            "flow_deployment_physical_loss": deployment_physical_loss,
            "flow_deployment_collective_mae_mps2": deployment_collective_mae,
            "flow_deployment_body_rate_mae_rps": deployment_body_rate_mae,
            "flow_step_size": flow_metrics["step_size"],
        }
    mixture_mode_per_sample = actions.new_zeros(len(actions))
    mixture_mode_loss = actions.new_zeros(())
    mixture_mode_accuracy = actions.new_zeros(())
    mixture_mode_entropy = actions.new_zeros(())
    mixture_codebook_mse = actions.new_zeros(())
    mixture_deployed_action_loss = actions.new_zeros(())
    mixture_settings = settings.get("mixture_action_head", {})
    if not isinstance(mixture_settings, Mapping):
        mixture_settings = {}
    shared_encoding: torch.Tensor | None = None
    if policy.action_mixture_mode_head is None:
        predicted, dynamics, topology, predicted_action_chunk, shared_encoding = (
            policy.training_predictions_with_chunk_and_encoding(
                histories, speed_commands
            )
        )
        deployed_predicted = predicted
    else:
        (
            deployed_predicted,
            dynamics,
            topology,
            mode_logits,
            action_candidates,
        ) = policy.mixture_training_predictions(histories, speed_commands)
        assert policy.action_mixture_centers is not None
        centers = policy.action_mixture_centers.to(
            device=actions.device, dtype=actions.dtype
        )
        squared_distance = (
            actions[:, None, :] - centers[None, :, :]
        ).square().sum(-1)
        target_mode = squared_distance.argmin(dim=-1)
        batch = torch.arange(len(actions), device=actions.device)
        predicted = action_candidates[batch, target_mode]
        predicted_action_chunk = predicted[:, None]
        mode_probabilities = mode_logits.float().softmax(dim=-1)
        target_probability = mode_probabilities.gather(
            1, target_mode[:, None]
        )[:, 0]
        cluster_weights = (
            None if policy.action_mixture_cluster_weights is None
            else policy.action_mixture_cluster_weights.float()
        )
        cross_entropy = F.cross_entropy(
            mode_logits.float(), target_mode,
            weight=cluster_weights, reduction="none",
        )
        focal_gamma = float(mixture_settings.get("focal_gamma", 2.0))
        if not np.isfinite(focal_gamma) or focal_gamma < 0.0:
            raise ValueError("mixture_action_focal_gamma must be non-negative")
        mixture_mode_per_sample = (
            (1.0 - target_probability).pow(focal_gamma) * cross_entropy
        )
        mixture_mode_loss = mixture_mode_per_sample.mean()
        mixture_mode_accuracy = (
            mode_logits.argmax(dim=-1) == target_mode
        ).float().mean()
        mixture_mode_entropy = -(
            mode_probabilities
            * mode_probabilities.clamp_min(1.0e-8).log()
        ).sum(-1).mean()
        mixture_codebook_mse = squared_distance.gather(
            1, target_mode[:, None]
        ).mean().div(4.0)
        mixture_deployed_action_loss = F.smooth_l1_loss(
            deployed_predicted, actions,
            beta=float(settings.get("huber_beta", 0.05)),
        )
    dimension_weights = action_dimension_weights(settings, predicted)
    action_per_sample = (F.smooth_l1_loss(
        predicted, actions, beta=float(settings.get("huber_beta", 0.05)),
        reduction="none",
    ) * dimension_weights).mean(-1)
    action_loss = action_per_sample.mean()
    delta_per_sample = (F.smooth_l1_loss(
        predicted - previous_actions,
        actions - previous_actions,
        beta=float(settings.get("delta_huber_beta", 0.04)),
        reduction="none",
    ) * dimension_weights).mean(-1)
    delta_loss = delta_per_sample.mean()
    predicted_physical = normalized_ctbr_tensor(predicted, settings)
    target_physical = normalized_ctbr_tensor(actions, settings)
    previous_physical = normalized_ctbr_tensor(previous_actions, settings)
    physical_scale_values = settings.get("physical_action_scales", [10.0, 3.0, 3.0, 3.0])
    if len(physical_scale_values) != 4 or any(
        not math.isfinite(float(value)) or float(value) <= 0 for value in physical_scale_values
    ):
        raise ValueError("physical_action_scales must contain four positive finite values")
    physical_scales = torch.as_tensor(
        settings.get('_dagger_physical_scales', physical_scale_values),
        device=predicted.device, dtype=predicted.dtype,
    )
    physical_action_per_sample = (F.smooth_l1_loss(
        (predicted_physical - target_physical) / physical_scales,
        torch.zeros_like(predicted_physical),
        beta=float(settings.get("physical_huber_beta", 0.05)),
        reduction="none",
    ) * dimension_weights).mean(-1)
    physical_action_loss = physical_action_per_sample.mean()
    physical_delta_per_sample = (F.smooth_l1_loss(
        (
            (predicted_physical - previous_physical)
            - (target_physical - previous_physical)
        ) / physical_scales,
        torch.zeros_like(predicted_physical),
        beta=float(settings.get("physical_delta_huber_beta", 0.04)),
        reduction="none",
    ) * dimension_weights).mean(-1)
    physical_delta_loss = physical_delta_per_sample.mean()
    dynamics_per_sample = F.smooth_l1_loss(
        dynamics, dynamics_targets,
        beta=float(settings.get("dynamics_huber_beta", 0.10)),
        reduction="none",
    ).mean(-1)
    if dynamics_valid is None:
        dynamics_loss = dynamics_per_sample.mean()
        dynamics_objective_per_sample = dynamics_per_sample
    else:
        dynamics_loss, dynamics_objective_per_sample = masked_objective(
            dynamics_per_sample, dynamics_valid
        )
    topology_loss = predicted.new_zeros(())
    plan_state_loss = predicted.new_zeros(())
    action_chunk_loss = predicted.new_zeros(())
    action_chunk_per_sample = predicted.new_zeros(len(predicted))
    topology_per_sample = predicted.new_zeros(len(predicted))
    plan_metric_per_sample = predicted.new_zeros(len(predicted))
    action_chunk_metric_per_sample = predicted.new_zeros(len(predicted))
    topology_weight = float(settings.get("topology_weight", 0.0))
    if topology_weight > 0.0:
        if topology is None or topology_targets is None:
            raise ValueError("topology-weighted imitation requires aligned targets")
        if topology.shape != topology_targets.shape:
            raise ValueError(
                f"topology prediction {topology.shape} does not match target "
                f"{topology_targets.shape}"
            )
        topology_elementwise = F.smooth_l1_loss(
            topology, topology_targets,
            beta=float(settings.get("topology_huber_beta", 0.05)),
            reduction="none",
        )
        topology_contract = str(settings.get(
            "topology_target_contract", "gate_frame_v1"
        ))
        if topology_contract == "body_plan_action_chunk_v2":
            horizons = len(tuple(settings.get(
                "topology_target_horizon_indices", (1, 6, 12, 24, 34)
            )))
            shaped = topology_elementwise.reshape(-1, horizons, 11)
            plan_elements = torch.cat([shaped[..., :6], shaped[..., 10:]], dim=-1)
            action_elements = shaped[..., 6:10]
            plan_per_sample = plan_elements.mean(dim=(-1, -2))
            action_chunk_per_sample = action_elements.mean(dim=(-1, -2))
            topology_per_sample = (
                plan_per_sample
                + float(settings.get("topology_action_chunk_weight", 1.0))
                * action_chunk_per_sample
            )
            plan_metric_per_sample = plan_per_sample
            action_chunk_metric_per_sample = action_chunk_per_sample
        elif topology_contract == "action_chunk_v1":
            action_chunk_metric_per_sample = topology_elementwise.mean(-1)
            topology_per_sample = (
                float(settings.get("topology_action_chunk_weight", 1.0))
                * action_chunk_metric_per_sample
            )
        else:
            topology_per_sample = topology_elementwise.mean(-1)
            plan_metric_per_sample = topology_per_sample
        if topology_valid is None:
            topology_loss = topology_per_sample.mean()
            plan_state_loss = plan_metric_per_sample.mean()
            action_chunk_loss = action_chunk_metric_per_sample.mean()
        else:
            topology_mask = topology_valid.to(
                topology_per_sample.dtype
            ).reshape(-1)
            if topology_mask.shape != topology_per_sample.shape:
                raise ValueError("topology-valid mask must align with targets")
            topology_loss = (
                topology_per_sample * topology_mask
            ).sum() / topology_mask.sum().clamp_min(1.0)
            plan_state_loss = (
                plan_metric_per_sample * topology_mask
            ).sum() / topology_mask.sum().clamp_min(1.0)
            action_chunk_loss = (
                action_chunk_metric_per_sample * topology_mask
            ).sum() / topology_mask.sum().clamp_min(1.0)
            topology_loss, topology_per_sample = masked_objective(
                topology_per_sample, topology_valid
            )
    action_chunk_weight = float(settings.get("action_chunk_weight", 0.0))
    if action_chunk_weight > 0.0:
        if action_chunk_targets is None:
            raise ValueError("action-chunk imitation requires aligned targets")
        if predicted_action_chunk.shape != action_chunk_targets.shape:
            raise ValueError(
                f"action chunk prediction {predicted_action_chunk.shape} does not "
                f"match target {action_chunk_targets.shape}"
            )
        if settings.get('action_chunk_target_source') == 'coherent_teacher':
            from starscream.coherent_action_chunks import decode_coherent_chunk_targets
            action_chunk_targets, coherent_valid = decode_coherent_chunk_targets(action_chunk_targets)
            action_chunk_valid = coherent_valid if action_chunk_valid is None else (action_chunk_valid.bool() & coherent_valid)
        action_chunk_elementwise = (F.smooth_l1_loss(
            predicted_action_chunk,
            action_chunk_targets,
            beta=float(settings.get("action_chunk_huber_beta", 0.05)),
            reduction="none",
        ) * dimension_weights).mean(dim=-1)
        exclude_first = bool(settings.get(
            "action_chunk_exclude_first_from_auxiliary", False
        ))
        start = 1 if exclude_first else 0
        if start >= action_chunk_elementwise.shape[1]:
            raise ValueError(
                "action chunk auxiliary excludes every predicted command"
            )
        decay = float(settings.get("action_chunk_horizon_decay", 1.0))
        if not 0.0 < decay <= 1.0:
            raise ValueError("action_chunk_horizon_decay must be in (0,1]")
        horizon_weights = action_chunk_elementwise.new_tensor([
            decay ** offset
            for offset in range(action_chunk_elementwise.shape[1])
        ])[start:]
        horizon_weights = horizon_weights / horizon_weights.sum()
        action_chunk_per_sample = (
            action_chunk_elementwise[:, start:] * horizon_weights[None]
        ).sum(-1)
        if action_chunk_valid is None:
            action_chunk_loss = action_chunk_per_sample.mean()
        else:
            action_chunk_loss, action_chunk_per_sample = masked_objective(
                action_chunk_per_sample, action_chunk_valid
            )
    reward_aux_weight = float(settings.get("reward_aux_weight", 0.0))
    reward_aux_per_sample = predicted.new_zeros(len(predicted))
    reward_aux_loss = predicted.new_zeros(())
    if reward_aux_weight > 0.0:
        if policy.reward_aux_head is None:
            raise ValueError("reward auxiliary weight requires a policy reward head")
        if executed_actions is None or reward_component_targets is None:
            raise ValueError("reward auxiliary requires executed actions and targets")
        if reward_component_targets.shape != (
            len(predicted), policy.reward_aux_dim
        ):
            raise ValueError("reward auxiliary target shape does not match policy")
        predicted_reward = policy.predict_reward_components(
            histories, executed_actions
        )
        # Component-wise symlog keeps sparse gate/crash events and small dense
        # shaping terms on compatible scales without fitting mutable replay
        # statistics during an ablation.
        reward_targets = torch.sign(reward_component_targets) * torch.log1p(
            reward_component_targets.abs()
        )
        reward_aux_per_sample = F.smooth_l1_loss(
            predicted_reward,
            reward_targets,
            beta=float(settings.get("reward_aux_huber_beta", 0.10)),
            reduction="none",
        ).mean(-1)
        reward_aux_loss = reward_aux_per_sample.mean()
    total_per_sample = (
        float(settings.get("action_weight", 1.0)) * action_per_sample
        + float(settings.get("delta_weight", 0.35)) * delta_per_sample
        + float(settings.get("physical_action_weight", 0.0))
        * physical_action_per_sample
        + float(settings.get("physical_delta_weight", 0.0))
        * physical_delta_per_sample
        + float(settings.get("dynamics_weight", 0.15))
        * dynamics_objective_per_sample
        + topology_weight * topology_per_sample
        + action_chunk_weight * action_chunk_per_sample
        + reward_aux_weight * reward_aux_per_sample
        + float(mixture_settings.get("mode_weight", 0.0))
        * mixture_mode_per_sample
    )
    total, group_loss_mean, group_loss_max, group_loss_std = topology_robust_objective(
        total_per_sample,
        group_ids,
        robust_weight=float(settings.get("dagger_group_robust_weight", 0.0)),
        temperature=float(settings.get("dagger_group_robust_temperature", 0.10)),
        fast_statistics=bool(settings.get('dagger_fast_group_statistics', False)),
        group_count=settings.get('dagger_statistics_group_count'),
    )
    anchor_loss = predicted.new_zeros(())
    if float(settings.get("dagger_anchor_distillation_weight", 0.0)) > 0.0:
        if anchor_actions is None or anchor_mask is None:
            raise ValueError("anchor distillation requires actions and a mask")
        mask = anchor_mask.to(dtype=torch.bool, device=predicted.device).reshape(-1)
        if mask.shape != predicted.shape[:1]:
            raise ValueError("anchor mask must align with the imitation batch")
        if mask.any():
            anchor_loss = F.smooth_l1_loss(
                deployed_predicted[mask], anchor_actions[mask],
                beta=float(settings.get("huber_beta", 0.05)),
            )
            total = total + float(
                settings["dagger_anchor_distillation_weight"]
            ) * anchor_loss
    gate_contrastive_loss = predicted.new_zeros(())
    gate_contrastive_weight = float(settings.get("gate_contrastive_weight", 0.0))
    if gate_contrastive_weight > 0.0:
        if policy.gate_contrastive_dim <= 0:
            raise ValueError(
                "gate contrastive weight requires model.gate_contrastive_dim > 0"
            )
        if shared_encoding is None:
            raise ValueError(
                "gate contrastive objective does not support the mixture action head"
            )
        gate_contrastive_loss = policy.gate_contrastive_loss_from_encoding(
            shared_encoding, histories,
            temperature=float(settings.get(
                "gate_contrastive_temperature", 0.10
            )),
            maximum_samples=int(settings.get(
                "gate_contrastive_maximum_samples", 256
            )),
        )
        total = total + gate_contrastive_weight * gate_contrastive_loss
    return total, {
        "loss": total, "action_loss": action_loss,
        "delta_loss": delta_loss, "physical_action_loss": physical_action_loss,
        "physical_delta_loss": physical_delta_loss,
        "dynamics_loss": dynamics_loss,
        "topology_loss": topology_loss,
        "plan_state_loss": plan_state_loss,
        "action_chunk_loss": action_chunk_loss,
        "reward_aux_loss": reward_aux_loss,
        "gate_contrastive_loss": gate_contrastive_loss,
        "anchor_loss": anchor_loss,
        "group_loss_mean": group_loss_mean,
        "group_loss_max": group_loss_max,
        "group_loss_std": group_loss_std,
        "mixture_mode_loss": mixture_mode_loss,
        "mixture_mode_accuracy": mixture_mode_accuracy,
        "mixture_mode_entropy": mixture_mode_entropy,
        "mixture_codebook_mse": mixture_codebook_mse,
        "mixture_deployed_action_loss": mixture_deployed_action_loss,
    }


def run_bc(config: dict[str, Any], device: str) -> None:
    settings = config["bc"]
    seed = int(settings.get("seed", 20260815))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tracks = configured_tracks(settings)
    action_chunk_weight = float(settings.get("action_chunk_weight", 0.0))
    action_chunk_offsets = tuple(int(item) for item in settings.get(
        "action_chunk_offsets", ()
    )) if action_chunk_weight > 0.0 else ()
    if action_chunk_weight > 0.0 and (
        not action_chunk_offsets or action_chunk_offsets[0] != 0
    ):
        raise ValueError(
            "receding-horizon BC requires action_chunk_offsets beginning at zero"
        )
    data = load_one_step_data(
        settings["data"], track=tracks,
        require_controller_valid=True,
        maximum_steps=int(settings.get("maximum_dataset_steps", 0)),
        route_gates=int(settings.get("route_gate_count", 3)),
        action_chunk_offsets=action_chunk_offsets,
        observation_contract=str(settings.get(
            "observation_contract", LEGACY_OBSERVATION_CONTRACT
        )),
    )
    remap_offline_action_contract(data, settings)
    train_indices, validation_indices = episode_split(
        data, float(settings.get("validation_fraction", 0.12)), seed
    )
    normalizer = FeatureNormalizer.fit(data.features[train_indices])
    normalized = normalizer.numpy(data.features)
    dynamics_contract = str(settings.get(
        "dynamics_target_contract", "task_delta_v1"
    ))
    if dynamics_contract == "mpcc_body_delta_v2":
        dynamics_targets = observed_invariant_dynamics_targets(
            data.states, data.next_states, settings
        )
        dynamics_valid_array = np.ones(len(data.features), np.bool_)
        dynamics_mean = np.zeros(dynamics_targets.shape[1], np.float32)
        dynamics_std = np.ones(dynamics_targets.shape[1], np.float32)
    else:
        dynamics_train_indices = train_indices[data.dynamics_valid[train_indices]]
        if not len(dynamics_train_indices):
            raise ValueError("training split contains no frame-consistent dynamics targets")
        dynamics_mean = data.next_task_deltas[dynamics_train_indices].mean(0).astype(np.float32)
        dynamics_std = np.maximum(
            data.next_task_deltas[dynamics_train_indices].std(0), 1.0e-4
        ).astype(np.float32)
        dynamics_targets = (
            (data.next_task_deltas - dynamics_mean) / dynamics_std
        ).astype(np.float32)
        dynamics_valid_array = np.asarray(data.dynamics_valid, np.bool_)
    model_config = dict(settings.get("model", {}))
    observation_contract = str(settings.get(
        "observation_contract", LEGACY_OBSERVATION_CONTRACT
    ))
    expected_input_dim = privileged_feature_dim(
        int(settings.get("route_gate_count", 3)), observation_contract,
    )
    model_config.setdefault("observation_contract", observation_contract)
    model_config.setdefault("input_dim", expected_input_dim)
    if (
        str(model_config["observation_contract"]) != observation_contract
        or int(model_config["input_dim"]) != expected_input_dim
    ):
        raise ValueError(
            "BC model and dataset observation contracts do not match: "
            f"contract={observation_contract} expected_dim={expected_input_dim}"
        )
    if action_chunk_weight > 0.0:
        configured_steps = int(model_config.get(
            "action_chunk_steps", len(action_chunk_offsets)
        ))
        if configured_steps != len(action_chunk_offsets):
            raise ValueError(
                "model action_chunk_steps must match action_chunk_offsets"
            )
        model_config["action_chunk_steps"] = configured_steps
    policy = PrivilegedMLPPolicy(**model_config).to(device)
    topology_weight = float(settings.get("topology_weight", 0.0))
    topology_target_dim = mpcc_topology_target_dim(settings)
    if topology_weight > 0.0:
        policy.enable_topology_conditioning(topology_target_dim)
    if dynamics_contract == "mpcc_body_delta_v2":
        policy.enable_action_conditioned_dynamics(dynamics_targets.shape[1])
    topology_targets = np.zeros(
        (len(data.features), topology_target_dim if topology_weight > 0.0 else 0),
        np.float32,
    )
    topology_valid_array = np.full(
        len(data.features), topology_weight > 0.0, np.bool_
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(settings.get("learning_rate", 3.0e-4)),
        weight_decay=float(settings.get("weight_decay", 1.0e-5)),
        fused=device.startswith("cuda"),
    )
    # Combined BC->DAgger configs have distinct run namespaces.  Honor the BC
    # stage name instead of silently writing its checkpoint into the later
    # DAgger directory.
    stage_config = copy.deepcopy(config)
    stage_run_name = str(settings.get(
        "checkpoint_run_name", settings.get("run_name", "privileged-bc")
    ))
    stage_config.setdefault("checkpoint", {})["run_name"] = stage_run_name
    stage_config.setdefault("wandb", {})["run_name"] = str(
        settings.get("run_name", stage_run_name)
    )
    manager = CheckpointManager.from_config(stage_config)
    logger = init_wandb(stage_config)
    batch_size = int(settings.get("batch_size", 2048))
    amp = bool(settings.get("amp", True)) and device.startswith("cuda")
    started = time.perf_counter()
    print(
        f"privileged_bc tracks={','.join(tracks)} samples={len(data.features)} "
        f"train={len(train_indices)} validation={len(validation_indices)} "
        f"episodes={len(data.paths)} context={policy.context_steps} "
        f"parameters={sum(p.numel() for p in policy.parameters()):,}",
        flush=True,
    )
    for step in range(1, int(settings.get("steps", 5000)) + 1):
        selected = balanced_choice(rng, train_indices, data.track_ids, batch_size)
        histories = torch.from_numpy(
            batch_sequences(normalized, data.episode_ids, selected, policy.context_steps)
        ).to(device, non_blocking=True)
        actions = torch.from_numpy(data.actions[selected]).to(device, non_blocking=True)
        previous = torch.from_numpy(data.previous_actions[selected]).to(device, non_blocking=True)
        dynamics = torch.from_numpy(dynamics_targets[selected]).to(device, non_blocking=True)
        dynamics_valid = torch.from_numpy(dynamics_valid_array[selected]).to(
            device, non_blocking=True
        )
        topology = torch.from_numpy(topology_targets[selected]).to(
            device, non_blocking=True
        )
        topology_valid = torch.from_numpy(topology_valid_array[selected]).to(
            device, non_blocking=True
        )
        action_chunks = torch.from_numpy(data.action_chunks[selected]).to(
            device, non_blocking=True
        )
        action_chunk_valid = torch.ones(
            len(selected), device=device, dtype=torch.bool
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, pieces = imitation_loss(
                policy, histories, actions, previous, dynamics, settings,
                dynamics_valid, topology_targets=topology,
                topology_valid=topology_valid,
                action_chunk_targets=action_chunks,
                action_chunk_valid=action_chunk_valid,
            )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float(settings.get("gradient_clip", 2.0))
        )
        optimizer.step()
        if step == 1 or step % int(settings.get("log_interval", 25)) == 0:
            elapsed = max(time.perf_counter() - started, 1.0e-6)
            metrics = {
                **{name: float(value.detach()) for name, value in pieces.items()},
                "gradient_norm": float(gradient),
                "samples_per_second": step * batch_size / elapsed,
            }
            logger.log_train(metrics, step)
            print(
                f"step={step} loss={metrics['loss']:.5f} "
                f"action={metrics['action_loss']:.5f} delta={metrics['delta_loss']:.5f} "
                f"chunk={metrics['action_chunk_loss']:.5f} "
                f"dynamics={metrics['dynamics_loss']:.5f} "
                f"samples_s={metrics['samples_per_second']:.0f}", flush=True,
            )
        if step % int(settings.get("validation_interval", 250)) == 0 or step == int(settings.get("steps", 5000)):
            policy.eval()
            limit = min(int(settings.get("validation_samples", 8192)), len(validation_indices))
            selected = balanced_choice(rng, validation_indices, data.track_ids, limit)
            with torch.no_grad():
                histories = torch.from_numpy(batch_sequences(
                    normalized, data.episode_ids, selected, policy.context_steps
                )).to(device)
                actions = torch.from_numpy(data.actions[selected]).to(device)
                previous = torch.from_numpy(data.previous_actions[selected]).to(device)
                dynamics = torch.from_numpy(dynamics_targets[selected]).to(device)
                dynamics_valid = torch.from_numpy(dynamics_valid_array[selected]).to(device)
                topology = torch.from_numpy(topology_targets[selected]).to(device)
                topology_valid = torch.from_numpy(
                    topology_valid_array[selected]
                ).to(device)
                action_chunks = torch.from_numpy(
                    data.action_chunks[selected]
                ).to(device)
                action_chunk_valid = torch.ones(
                    len(selected), device=device, dtype=torch.bool
                )
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    validation_loss, pieces = imitation_loss(
                        policy, histories, actions, previous, dynamics, settings,
                        dynamics_valid, topology_targets=topology,
                        topology_valid=topology_valid,
                        action_chunk_targets=action_chunks,
                        action_chunk_valid=action_chunk_valid,
                    )
            metrics = {
                **{name: float(value) for name, value in pieces.items()},
                "selection_score": -float(validation_loss),
            }
            logger.log_eval(metrics, step)
            manager.save(
                checkpoint_payload(
                    policy, normalizer, stage="bc", track=",".join(tracks),
                    optimizer=optimizer,
                    extra={
                        "dynamics_target_mean": dynamics_mean,
                        "dynamics_target_std": dynamics_std,
                        "training_config": config,
                    },
                ),
                step=step, metrics=metrics,
            )
            print(
                f"validation step={step} loss={metrics['loss']:.5f} "
                f"selection={metrics['selection_score']:.5f}", flush=True,
            )
            policy.train()
    logger.finish()


@dataclass
class DaggerBatch:
    histories: list[np.ndarray] | NumericRowBuffer = field(default_factory=list)
    actions: list[np.ndarray] | NumericRowBuffer = field(default_factory=list)
    previous_actions: list[np.ndarray] | NumericRowBuffer = field(default_factory=list)
    dynamics: list[np.ndarray] | NumericRowBuffer = field(default_factory=list)
    dynamics_valid: list[bool] = field(default_factory=list)
    tracks: list[str] = field(default_factory=list)
    course_progress: list[float] = field(default_factory=list)
    # Privileged replay metadata only; never exposed to the actor.
    # ordinary=0, crossing=2, post-crossing=3, next-gate acquisition=4.
    gate_phases: list[int] = field(default_factory=list)
    active_gate_indices: list[int] = field(default_factory=list)
    teacher_modes: list[int] = field(default_factory=list)
    # 0 canonical race start, 1 cold gate-local reset, 2 expert-flown prefix.
    occupancy_modes: list[int] = field(default_factory=list)
    trajectory: list[np.ndarray] = field(default_factory=list)
    quality_counts: dict[str, int] = field(default_factory=dict)
    speed_commands: list[float] = field(default_factory=list)
    action_chunks: list[np.ndarray] = field(default_factory=list)
    executed_actions: list[np.ndarray] = field(default_factory=list)
    reward_components: list[np.ndarray] = field(default_factory=list)
    topology_targets: list[np.ndarray] = field(default_factory=list)
    topology_valid: list[bool] = field(default_factory=list)
    # True only for transitions belonging to an episode that reached its full
    # requested route without collision. This supports immutable full-lap
    # expert replay without discarding corrective labels from failed rollouts.
    successful_labels: list[bool] = field(default_factory=list)
    valid_queries: int = 0
    total_queries: int = 0
    solver_failures: int = 0
    recovery_queries: int = 0
    executed_teacher: int = 0
    environment_steps: int = 0
    accepted_episodes: int = 0
    accepted_multilap_episodes: int = 0
    rejected_episodes: int = 0
    pure_expert_aborted_episodes: int = 0
    # Expert actions were recorded before noise, while these steps executed a
    # bounded DART perturbation in the plant to elicit expert recoveries.
    dart_noisy_steps: int = 0
    scheduled_canonical_start_episodes: int = 0
    scheduled_targeted_start_episodes: int = 0
    scheduled_expert_prefix_episodes: int = 0
    completed_expert_prefix_handoffs: int = 0
    expert_prefix_forced_steps: int = 0
    expert_prefix_handoff_speed_sum: float = 0.0
    accepted_episodes_by_track: dict[str, int] = field(default_factory=dict)
    # Process/thread collectors already normalize the actor history for policy
    # inference.  Retaining that exact normalized half-precision history avoids
    # a second full-dataset float32 materialization after every collection.
    histories_normalized: bool = False
    teacher_solve_seconds: float = 0.0
    teacher_wall_seconds: float = 0.0
    environment_step_seconds: float = 0.0
    feature_seconds: float = 0.0
    controller_wall_breakdown: np.ndarray = field(
        default_factory=lambda: np.zeros(4, np.float64)
    )
    backend_wall_breakdown: np.ndarray = field(
        default_factory=lambda: np.zeros(4, np.float64)
    )


def merge_dagger_batches(target: DaggerBatch, source: DaggerBatch) -> DaggerBatch:
    """Append a supplemental collection without losing accounting."""

    if target.histories and target.histories_normalized != source.histories_normalized:
        raise ValueError("cannot merge raw and normalized DAgger histories")
    if not target.histories:
        target.histories_normalized = source.histories_normalized

    for name in (
        "histories", "actions", "previous_actions", "dynamics",
        "dynamics_valid", "tracks", "course_progress", "gate_phases",
        "active_gate_indices", "teacher_modes", "occupancy_modes", "trajectory",
        "speed_commands", "action_chunks",
        "executed_actions", "reward_components",
        "topology_targets", "topology_valid",
        "successful_labels",
    ):
        getattr(target, name).extend(getattr(source, name))
    for name in (
        "valid_queries", "total_queries", "solver_failures", "recovery_queries",
        "executed_teacher", "environment_steps", "accepted_episodes",
        "accepted_multilap_episodes", "rejected_episodes",
        "pure_expert_aborted_episodes",
        "dart_noisy_steps", "scheduled_canonical_start_episodes",
        "scheduled_targeted_start_episodes", "scheduled_expert_prefix_episodes",
        "completed_expert_prefix_handoffs", "expert_prefix_forced_steps",
    ):
        setattr(target, name, getattr(target, name) + getattr(source, name))
    for key, value in source.quality_counts.items():
        target.quality_counts[key] = target.quality_counts.get(key, 0) + value
    target.expert_prefix_handoff_speed_sum += source.expert_prefix_handoff_speed_sum
    target.teacher_solve_seconds += source.teacher_solve_seconds
    target.teacher_wall_seconds += source.teacher_wall_seconds
    target.environment_step_seconds += source.environment_step_seconds
    target.feature_seconds += source.feature_seconds
    target.controller_wall_breakdown += source.controller_wall_breakdown
    target.backend_wall_breakdown += source.backend_wall_breakdown
    for track, count in source.accepted_episodes_by_track.items():
        target.accepted_episodes_by_track[track] = (
            target.accepted_episodes_by_track.get(track, 0) + count
        )
    return target


def _dagger_process_worker(
    connection: Any,
    settings: dict[str, Any],
    stage: RacingCurriculumStage,
    worker_index: int,
) -> None:
    """Own one simulator and ACADOS solver outside the learner's Python GIL."""

    _terminate_worker_with_parent()
    # This guard is intentionally created before Flightmare and ACADOS.  It
    # changes scheduling only; controller math, tolerances, and labels remain
    # untouched.
    _worker_threadpool_guard = _configure_dagger_worker_threads(settings)
    worker_seed = (
        int(settings.get("seed", 20260816))
        + 1_000_003 * (int(worker_index) + 1)
    )
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)
    quality = quality_config(settings)
    collection = collection_config(settings)
    control = None
    guard = None
    env: FlightmareEnv | None = None
    try:
        worker_track = stage.tracks[worker_index % len(stage.tracks)]
        env = make_env(settings, track=worker_track)
        expert = RoutedDaggerTeacher(env, settings, worker_index=worker_index)
        shared_fast_backend, shared_recovery_backend = expert.backend_instances
        observation: dict[str, Any] | None = None
        current_feature: np.ndarray | None = None
        overlap_teacher = bool(settings.get('dagger_overlap_teacher_inference', False))
        pending_teacher = None
        start_passed = 0
        steps = 0
        target_gates = min(stage.target_gates, len(env.track.gates))
        rollout_laps = 1
        dart_noise_scale = 0.0
        transition_start_mode = "canonical"
        prefix_handoff_gate: int | None = None
        prefix_handoff_reported = False
        dart_rng = np.random.default_rng(worker_index)
        steps_since_crossing: int | None = None
        dart_noise_std = np.asarray(
            settings.get("dagger_dart_action_noise_std_normalized", (0.0, 0.0, 0.0, 0.0)),
            np.float32,
        )
        if (
            dart_noise_std.shape != (4,)
            or not np.all(np.isfinite(dart_noise_std))
            or np.any(dart_noise_std < 0.0)
        ):
            raise ValueError(
                "dagger_dart_action_noise_std_normalized must be four non-negative values"
            )
        connection.send(("ready", worker_index))
        def query_teacher():
            # Both actor and expert condition on the same fixed plant state.
            # Capture targets BEFORE the expert, exactly as in the serial path.
            task = dynamics_observation_task(observation, settings)
            previous = current_feature[action_feature_slice(settings)].copy()
            fraction = (env.tracker.passed_count - start_passed) / max(target_gates, 1)
            gate_index = int(env.tracker.index)
            started = time.perf_counter()
            answer = expert(observation)
            return task, previous, fraction, gate_index, answer, time.perf_counter() - started

        while True:
            request = connection.recv()
            operation = request[0]
            if operation == "close":
                connection.send(("closed", worker_index))
                return
            if operation == "reset":
                episode_index, seed = int(request[1]), int(request[2])
                requested_track = str(request[3]) if len(request) > 3 else worker_track
                rollout_laps = int(request[4]) if len(request) > 4 else 1
                target_speed = (
                    float(request[5]) if len(request) > 5
                    else float(expert.fast.config.nominal_speed)
                )
                dart_noise_scale = float(request[6]) if len(request) > 6 else 0.0
                requested_start_gate = (
                    None
                    if len(request) <= 7 or request[7] is None
                    else int(request[7])
                )
                transition_start_scale = (
                    float(request[8]) if len(request) > 8 else 1.0
                )
                transition_start_mode = (
                    str(request[9]) if len(request) > 9
                    else dagger_transition_start_mode(settings)
                )
                if transition_start_mode not in DAGGER_OCCUPANCY_MODE_IDS:
                    raise ValueError(
                        f"unknown DAgger episode start mode {transition_start_mode!r}"
                    )
                if not np.isfinite(dart_noise_scale) or dart_noise_scale < 0.0:
                    raise ValueError("DAgger DART noise scale must be finite and non-negative")
                if (
                    not np.isfinite(transition_start_scale)
                    or transition_start_scale < 1.0
                ):
                    raise ValueError(
                        "DAgger transition-start perturbation scale must be finite and >= 1"
                    )
                dart_rng = np.random.default_rng(seed ^ 0xDA47A11)
                prefix_handoff_gate = (
                    requested_start_gate
                    if transition_start_mode == "expert_prefix" else None
                )
                prefix_handoff_reported = False
                if rollout_laps < 1:
                    raise ValueError("DAgger rollout_laps must be positive")
                if requested_track != worker_track:
                    env.close()
                    worker_track = requested_track
                    env = make_env(settings, track=worker_track)
                    expert = RoutedDaggerTeacher(
                        env, settings, worker_index=worker_index,
                        fast_backend=shared_fast_backend,
                        recovery_backend=shared_recovery_backend,
                    )
                expert.set_nominal_speed(target_speed)
                if int(settings.get("reward_aux_dim", 0)) > 0:
                    env.reward_function = make_reward(settings, target_speed)
                target_gates = (
                    min(stage.target_gates, len(env.track.gates)) * rollout_laps
                )
                reset_scale = max(
                    transition_start_scale
                    if transition_start_mode == "cold_reset" else 1.0,
                    float(settings.get("dagger_dart_start_perturbation_scale", 1.0))
                    if dart_noise_scale > 0.0 else 1.0,
                )
                reset_stage = (
                    dagger_start_perturbation_stage(stage, reset_scale)
                    if reset_scale > 1.0 else stage
                )
                if (
                    requested_start_gate is not None
                    and transition_start_mode == "cold_reset"
                ):
                    reset_stage = replace(
                        reset_stage,
                        random_gate=False,
                        fixed_start_gate_index=requested_start_gate,
                    )
                observation, start_passed = reset_env(
                    env, reset_stage,
                    seed=seed, episode_index=episode_index,
                    rollout_laps=rollout_laps,
                )
                expert.reset()
                guard = TrajectoryGuard(track_quality_config(quality, worker_track), configured_control_dt(settings), seed) if quality else None
                control = (CollectionControl(track_collection_config(collection, worker_track),
                    configured_control_dt(settings), seed, enabled=bool(request[10]) if len(request) > 10 else False)
                    if collection else None)
                steps = 0
                steps_since_crossing = None
                current_feature = ppo_observation_features(observation, settings)
                connection.send(("reset", current_feature))
                pending_teacher = query_teacher() if overlap_teacher else None
                continue
            if operation == "advance_teacher":
                if observation is None or current_feature is None:
                    raise RuntimeError('teacher-only advance before reset')
                if pending_teacher is None:
                    pending_teacher = query_teacher()
                candidate = pending_teacher[4]
                if not (bool(candidate.valid) and np.all(np.isfinite(candidate.action.as_array()))):
                    # The original mixed policy falls back to the learner here.
                    # Ask for it lazily; retain the exact query, do NOT solve twice.
                    connection.send(('need_policy',))
                    continue
                operation = 'advance'
                request = ('advance', np.zeros(4, np.float32), True)
            if operation != "advance" or observation is None or current_feature is None:
                raise RuntimeError(f"invalid DAgger worker operation: {operation}")
            policy_action = np.asarray(request[1], np.float32)
            use_teacher_draw = bool(request[2])
            execution_mode = str(request[3]) if len(request) > 3 else "mixture"
            query_noise = (
                np.asarray(request[4], np.float32)
                if len(request) > 4 else np.zeros(4, np.float32)
            )
            if execution_mode not in {
                "mixture", "learner_query", "expert_query",
                "teacher_perturbation_query", "post_query_expert",
            }:
                raise ValueError(f"unknown interactive-imitation action mode {execution_mode!r}")
            if query_noise.shape != (4,) or not np.all(np.isfinite(query_noise)):
                raise ValueError("interactive-imitation query noise must be finite CTBR[4]")
            prefix_forced_teacher = dagger_expert_prefix_forces_teacher(
                prefix_handoff_gate, env.tracker.passed_count, start_passed
            )
            prefix_handoff = bool(
                prefix_handoff_gate is not None
                and not prefix_forced_teacher
                and not prefix_handoff_reported
            )
            if prefix_handoff:
                prefix_handoff_reported = True
            prefix_handoff_speed = (
                float(np.linalg.norm(
                    np.asarray(observation["state"], np.float32)[7:10]
                ))
                if prefix_handoff else 0.0
            )
            (prior_task, previous_action, prior_gate_fraction, prior_gate_index,
             command, teacher_wall_seconds) = (
                pending_teacher if pending_teacher is not None else query_teacher())
            pending_teacher = None
            physical_teacher = np.asarray(command.action.as_array(), np.float32)
            finite = bool(command.valid) and bool(np.all(np.isfinite(physical_teacher)))
            solver_failed = int(command.solver_status) != 0
            require_clean = bool(settings.get("dagger_require_solver_success", False))
            minimum_margin = float(settings.get(
                "dagger_minimum_teacher_constraint_margin", -float("inf")
            ))
            valid = bool(
                finite
                and (not require_clean or not solver_failed)
                and float(command.constraint_margin) >= minimum_margin
            )
            topology_target = (
                mpcc_topology_target(command, observation, env, settings)
                if valid and float(settings.get("topology_weight", 0.0)) > 0.0
                else None
            )
            topology_target_valid = bool(
                topology_target is not None
                and mpcc_topology_target_is_valid(command, topology_target, settings)
            )
            if topology_target is not None and not topology_target_valid:
                topology_target = np.zeros_like(topology_target)
            action_chunk_target = (
                mpcc_action_chunk_target(command, settings)
                if (
                    valid
                    and float(settings.get("action_chunk_weight", 0.0)) > 0.0
                    and str(settings.get(
                        "action_chunk_target_source", "mpcc_open_loop"
                    )) == "mpcc_open_loop"
                )
                else None
            )
            invariant_dynamics = (
                mpcc_invariant_dynamics_target(command, observation, settings)
                if valid and str(settings.get(
                    "dynamics_target_contract", "task_delta_v1"
                )) == "mpcc_body_delta_v2"
                else None
            )
            quality_force = False
            if guard is not None:
                quality_force = guard.before(
                    float(command.diagnostics.get('trajectory_reference_distance', float('nan'))),
                    bool(command.diagnostics.get('missed_gate_recovery', False)))
                quality_gate = env.track.gates[env.tracker.index]
                quality_side = float((np.asarray(observation['state'])[:3] - quality_gate.position) @ quality_gate.normal)
                quality_occurrence = env.tracker.passed_count
                if execution_mode != 'mixture':
                    raise ValueError('trajectory quality requires ordinary DAgger mixture execution')
            collection_teacher = collection_learner = False
            if control is not None:
                if execution_mode != 'mixture':
                    raise ValueError('collection controls require ordinary DAgger execution')
                collection_teacher, collection_learner = control.before(
                    float(command.diagnostics.get('trajectory_reference_distance', float('nan'))),
                    bool(command.diagnostics.get('missed_gate_recovery', False)), prefix=prefix_forced_teacher)
                quality_gate = env.track.gates[env.tracker.index]
                quality_side = float((np.asarray(observation['state'])[:3] - quality_gate.position) @ quality_gate.normal)
                quality_occurrence = env.tracker.passed_count
            force_expert = quality_force or collection_teacher or execution_mode in {"expert_query", "post_query_expert"}
            force_learner = collection_learner or execution_mode == "learner_query"
            executed_teacher = finite and (
                force_expert
                or (
                    not force_learner
                    and execution_mode == "mixture"
                    and (use_teacher_draw or prefix_forced_teacher)
                )
            )
            physical_action = (
                physical_teacher if executed_teacher
                else ppo_normalized_to_ctbr(policy_action, settings)
            )
            if execution_mode == "teacher_perturbation_query" and finite:
                perturbed = np.clip(
                    ppo_ctbr_to_normalized(physical_teacher, settings)
                    + query_noise,
                    -1.0,
                    1.0,
                ).astype(np.float32)
                physical_action = ppo_normalized_to_ctbr(perturbed, settings)
            dart_noisy = bool(
                execution_mode == "mixture"
                and not quality_force
                and not collection_teacher
                and executed_teacher
                and dart_noise_scale > 0.0
                and np.any(dart_noise_std > 0.0)
            )
            if dart_noisy:
                clean_normalized = ppo_ctbr_to_normalized(
                    physical_teacher, settings
                )
                noisy_normalized = np.clip(
                    clean_normalized
                    + dart_rng.normal(0.0, dart_noise_std * dart_noise_scale),
                    -1.0,
                    1.0,
                ).astype(np.float32)
                physical_action = ppo_normalized_to_ctbr(noisy_normalized, settings)
            # Delay compensation and slew bounds for the next query must use
            # the action that actually entered the plant, not an unexecuted
            # counterfactual expert recommendation.
            expert.observe_executed_action(physical_action)
            environment_started = time.perf_counter()
            observation, environment_reward, terminated, _, info = env.step(
                physical_action
            )
            environment_step_seconds = time.perf_counter() - environment_started
            feature_started = time.perf_counter()
            next_feature = ppo_observation_features(observation, settings)
            feature_seconds = time.perf_counter() - feature_started
            gate_passed = bool(info.get("gate_passed", False))
            if invariant_dynamics is None:
                dynamics_target = _normalize_dynamics_delta_for_control_rate(
                    dynamics_observation_task(observation, settings) - prior_task,
                    settings,
                )
                dynamics_target_valid = not gate_passed
            else:
                dynamics_target = invariant_dynamics
                dynamics_target_valid = True
            post_horizon = int(settings.get(
                "dagger_gate_phase_post_crossing_horizon", 12
            ))
            acquisition_horizon = int(settings.get(
                "dagger_gate_phase_acquisition_horizon", 48
            ))
            if post_horizon < 0 or acquisition_horizon < post_horizon:
                raise ValueError("invalid DAgger gate-phase horizons")
            if gate_passed:
                gate_phase = 2
            elif steps_since_crossing is not None and steps_since_crossing < post_horizon:
                gate_phase = 3
            elif (
                steps_since_crossing is not None
                and steps_since_crossing < acquisition_horizon
            ):
                gate_phase = 4
            else:
                gate_phase = 0
            if gate_passed:
                steps_since_crossing = 0
            elif steps_since_crossing is not None:
                steps_since_crossing += 1
            steps += 1
            gates = env.tracker.passed_count - start_passed
            crashed = bool(info.get("ground_contact") or info.get("unity_collision"))
            done = bool(terminated or gates >= target_gates or steps >= stage.max_steps)
            quality_row = None
            if guard is not None:
                quality_position = np.asarray(observation['state'])[:3]
                quality_post_side = float((quality_position - quality_gate.position) @ quality_gate.normal)
                planned_crossing = False
                if not gate_passed and quality_side < 0 <= quality_post_side:
                    planned_crossing = planned_plane_crossing(
                        expert.fast.line, quality_gate, prior_gate_index, len(env.track.gates),
                        quality_position, expert.fast._last_progress, guard.config['nominal_distance'])
                quality_row, quality_stop = guard.after(pre_side=quality_side,
                    post_side=quality_post_side, planned_crossing=planned_crossing,
                    passed=gate_passed, gate_occurrence=quality_occurrence,
                    teacher=executed_teacher, dart=dart_noisy, valid=valid,
                    crashed=crashed, done=done, finished=gates >= target_gates)
                done = done or quality_stop
            collection_events = None
            if control is not None:
                quality_position = np.asarray(observation['state'])[:3]
                quality_post_side = float((quality_position - quality_gate.position) @ quality_gate.normal)
                planned_crossing = False
                if not gate_passed and quality_side < 0 <= quality_post_side:
                    planned_crossing = planned_plane_crossing(
                        expert.fast.line, quality_gate, prior_gate_index, len(env.track.gates),
                        quality_position, expert.fast._last_progress, control.config['plane_tolerance'])
                quality_row, collection_stop, collection_events = control.after(
                    pre_side=quality_side, post_side=quality_post_side, passed=gate_passed,
                    gate_occurrence=quality_occurrence, teacher=executed_teacher, dart=dart_noisy,
                    valid=valid, crashed=crashed, done=done, finished=gates >= target_gates,
                    planned_crossing=planned_crossing)
                done = done or collection_stop
            response = (
                "step",
                valid,
                ppo_ctbr_to_normalized(physical_teacher, settings) if valid else None,
                previous_action,
                dynamics_target,
                dynamics_target_valid,
                next_feature,
                executed_teacher,
                done,
                worker_track,
                solver_failed,
                int(expert.last_routed),
                prior_gate_fraction,
                gate_phase,
                bool(gates >= target_gates and not crashed),
                rollout_laps,
                topology_target,
                topology_target_valid,
                action_chunk_target,
                float(command.solve_time),
                float(teacher_wall_seconds),
                dart_noisy,
                prefix_forced_teacher,
                prefix_handoff,
                prefix_handoff_speed,
                prior_gate_index,
                DAGGER_OCCUPANCY_MODE_IDS[transition_start_mode],
                float(environment_step_seconds),
                float(feature_seconds),
                np.asarray(
                    command.diagnostics.get(
                        "controller_wall_breakdown", np.zeros(4, np.float32)
                    ),
                    np.float32,
                ),
                float(environment_reward),
                bool(gate_passed),
                bool(crashed),
                float(gates / max(target_gates, 1)),
                ppo_ctbr_to_normalized(physical_action, settings),
                execution_mode,
                np.asarray([
                    float(info.get("reward_components", {}).get(name, 0.0))
                    for name in PPO_REWARD_COMPONENTS
                ], np.float32),
                np.asarray(command.diagnostics.get('backend_wall_breakdown', np.zeros(4)), np.float32),
            )
            if settings.get('dagger_log_teacher_controller_modes', False) or control is not None:
                response = response + (quality_row, (
                    int(np.asarray(command.diagnostics.get('mode', 0)).item()),
                    bool(command.diagnostics.get('missed_gate_recovery', False)),
                    solver_failed,
                ))
            elif guard is not None:
                response = response + (quality_row,)
            if control is not None:
                response = response + (collection_events,)
            connection.send(response)
            current_feature = next_feature
            # Report s[t+1] first so GPU actor inference can start immediately;
            # solve its MPCC query on this CPU before waiting for a[t+1]. No
            # plant advance or policy-version change occurs during this overlap.
            # Terminal episodes must not mutate solver state with an extra query.
            if overlap_teacher and not done:
                pending_teacher = query_teacher()
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
class ProcessDaggerSlot:
    connection: Any
    process: Any
    history: CausalHistory
    track: str
    done: bool = True
    episode_batch: DaggerBatch = field(default_factory=DaggerBatch)
    episode_output_indices: list[int] = field(default_factory=list)
    pending_recovery_indices: list[int] = field(default_factory=list)
    chunk_step_events: list[tuple[int | None, bool]] = field(default_factory=list)
    target_speed: float = 0.0
    waiting_for: str | None = None
    episode_index: int = -1
    episode_step: int = 0
    episode_start_gate: int = 0
    pending_normalized_history: np.ndarray | None = None
    force_policy_once: bool = False
    awaiting_lazy_response: bool = False


class ProcessDaggerCollector:
    """Process-isolated rollout agents with one persistent ACADOS solver each."""

    def _new_batch(self, *, compact=False):
        batch = DaggerBatch(histories_normalized=True)
        if (compact and self.settings.get('dagger_chunked_labels', False)
                and self.settings.get('dagger_require_successful_episodes', False)):
            for field in ('histories', 'actions', 'previous_actions', 'dynamics'):
                setattr(batch, field, NumericRowBuffer())
        return batch

    def __init__(
        self,
        policy: PrivilegedMLPPolicy,
        normalizer: FeatureNormalizer,
        settings: Mapping[str, Any],
        stage: RacingCurriculumStage,
        device: str,
    ) -> None:
        self.policy = policy
        self.normalizer = normalizer
        self.settings = dict(settings)
        quality_config(settings)
        collection_config(settings)
        self.stage = stage
        self.device = device
        self.parallel = int(settings.get("rollout_envs", 12))
        self.inference_policy = inference_callable(policy, settings, self.parallel)
        self.lazy_teacher_actions = bool(settings.get('dagger_lazy_teacher_actions', False)) and not bool(collection_config(settings))
        if self.lazy_teacher_actions and (policy.action_head_type != 'mlp'
                or getattr(policy, 'action_mixture_mode_head', None) is not None):
            raise ValueError('lazy teacher actions require a deterministic MLP actor')
        self.host_inference_policy = None
        if settings.get("dagger_host_inference_graph", False):
            if settings.get("compile_policy_inference", False):
                raise ValueError("host inference graph cannot wrap compiled inference")
            from starscream.inference_graph import HostPolicyInferenceGraphs
            self.host_inference_policy = HostPolicyInferenceGraphs(policy, self.parallel)
        collection_dtype = str(settings.get(
            "dagger_collection_history_dtype",
            settings.get("dagger_replay_history_dtype", "float32"),
        ))
        if collection_dtype not in {"float16", "float32"}:
            raise ValueError(
                "dagger_collection_history_dtype must be float16 or float32"
            )
        self.collection_history_dtype = (
            np.float16 if collection_dtype == "float16" else np.float32
        )
        self.closed = False
        self.track_target_speeds: dict[str, float] = {}
        self.midtrain_quality = dict(settings.get('midtrain_quality',{}))
        self.midtrain_reference = {}
        if self.midtrain_quality:
            if not settings.get('dagger_require_successful_episodes',False):
                raise ValueError('midtrain quality requires completed-episode collection')
            manifest=read_manifest(Path(str(settings['track_manifest'])))
            for record in manifest['records']:
                if record.get('split')!='train':continue
                qualification=record.get('qualification',{})
                cohort=qualification.get('pace_cohorts',{}).get('randomized',{})
                reference=float(cohort.get('median',0))
                if reference<=0 or int(cohort.get('successes',0))<1:
                    raise ValueError('midtrain course lacks qualified randomized lap time')
                gate_count=int(record.get('gate_count') or len(load_track(record['path']).gates))
                self.midtrain_reference[str(Path(record['path']).resolve())]=(reference,gate_count)
        # MPCC pace selection and actor pace conditioning are separate
        # contracts. Older experiments used the latter flag for both; retain
        # that behavior as a fallback while allowing an unconditioned Green
        # actor to receive qualified near-frontier teacher labels.
        use_manifest_teacher_speed = bool(settings.get(
            "dagger_use_manifest_teacher_speed",
            settings.get("dagger_condition_on_teacher_speed", False),
        ))
        if use_manifest_teacher_speed:
            manifest_path = Path(str(settings["track_manifest"]))
            manifest = read_manifest(manifest_path)
            self.track_target_speeds = manifest_track_values(
                manifest_path, "qualified_speed_mps", cast=float,
            )
        self.speed_fractions = configured_dagger_speed_fractions(settings)
        self.track_rollout_laps = configured_track_rollout_laps(
            self.settings, stage.tracks
        )
        start_method = str(settings.get(
            "dagger_process_start_method", "spawn"
        ))
        if start_method not in {"spawn", "forkserver", "fork"}:
            raise ValueError(
                "dagger_process_start_method must be spawn, forkserver, or fork"
            )
        if start_method == "fork" and os.name != "posix":
            raise ValueError("fork DAgger workers require a POSIX host")
        context = mp.get_context(start_method)
        self.slots: list[ProcessDaggerSlot] = []
        try:
            for index in range(self.parallel):
                parent, child = context.Pipe()
                if settings.get('dagger_packed_transport', False):
                    from starscream.dagger_transport import PackedDaggerPipe
                    parent, child = PackedDaggerPipe(parent), PackedDaggerPipe(child)
                process = context.Process(
                    target=_dagger_process_worker,
                    args=(child, self.settings, stage, index),
                    name=f"dagger-mpcc-{index:02d}",
                )
                process.start()
                child.close()
                self.slots.append(
                    ProcessDaggerSlot(
                        parent, process, CausalHistory(policy.context_steps),
                        stage.tracks[index % len(stage.tracks)],
                    )
                )
            for slot in self.slots:
                response = slot.connection.recv()
                if response[0] != "ready":
                    raise RuntimeError(f"DAgger worker startup failed: {response}")
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    @staticmethod
    def _receive(slot: ProcessDaggerSlot, expected: str) -> tuple[Any, ...]:
        response = slot.connection.recv()
        if response[0] == "error":
            raise RuntimeError(f"DAgger worker failed:\n{response[1]}")
        if expected == 'step' and response[0] == 'need_policy':
            return response
        if response[0] != expected:
            raise RuntimeError(f"expected DAgger worker {expected}, received {response[0]}")
        return response

    def _finalize_replanned_action_chunks(
        self, output: DaggerBatch, slot: ProcessDaggerSlot,
    ) -> None:
        if self.settings.get('action_chunk_target_source') == 'coherent_teacher':
            from starscream.coherent_action_chunks import build_coherent_chunks
            target = slot.episode_batch if self.settings.get('dagger_require_successful_episodes', False) else output
            horizon = len(self.settings['action_chunk_offsets'])
            for row, chunk in build_coherent_chunks(target.actions, slot.chunk_step_events, horizon).items():
                target.action_chunks[row] = chunk
            return
        if (
            float(self.settings.get("action_chunk_weight", 0.0)) <= 0.0
            or str(self.settings.get(
                "action_chunk_target_source", "mpcc_open_loop"
            )) != "replanned_teacher"
        ):
            return
        offsets = tuple(int(value) for value in self.settings.get(
            "action_chunk_offsets", ()
        ))
        require_success = bool(self.settings.get(
            "dagger_require_successful_episodes", False
        ))
        if require_success:
            episode = slot.episode_batch
            chunks = replanned_teacher_action_chunks(episode.actions, offsets)
            if len(chunks) != len(episode.action_chunks):
                raise RuntimeError("replanned episode chunks lost alignment")
            episode.action_chunks[:] = chunks
            return
        indices = slot.episode_output_indices
        actions = [output.actions[index] for index in indices]
        chunks = replanned_teacher_action_chunks(actions, offsets)
        if len(chunks) != len(indices):
            raise RuntimeError("replanned output chunks lost alignment")
        for index, chunk in zip(indices, chunks, strict=True):
            output.action_chunks[index] = chunk

    def _reset_slot(
        self, slot: ProcessDaggerSlot, index: int, seed_base: int, track: str,
        target_speed: float, dart_noise_scale: float = 0.0,
        start_gate_index: int | None = None,
        transition_start_scale: float = 1.0,
        transition_start_mode: str = "canonical",
        collection_active: bool = False,
    ) -> None:
        rollout_laps = self.track_rollout_laps[str(Path(track).resolve())]
        slot.connection.send((
            "reset", index, seed_base + 1009 * index, track, rollout_laps,
            float(target_speed), float(dart_noise_scale), start_gate_index,
            float(transition_start_scale), str(transition_start_mode), bool(collection_active),
        ))
        response = self._receive(slot, "reset")
        slot.history.reset_feature(response[1])
        slot.track = str(track)
        slot.target_speed = float(target_speed)
        slot.done = False
        slot.episode_batch = self._new_batch()
        slot.episode_output_indices = []
        slot.chunk_step_events = []

    def _record_step_response(
        self,
        output: DaggerBatch,
        slot: ProcessDaggerSlot,
        normalized_history: np.ndarray,
        response: tuple[Any, ...],
    ) -> bool:
        (
            _, valid, teacher_action, previous_action, dynamics,
            dynamics_valid, feature, used_teacher, done, track,
            solver_failed, teacher_mode, gate_fraction,
            gate_phase,
            successful_episode,
            rollout_laps,
            topology_target,
            topology_target_valid,
            action_chunk_target,
            teacher_solve_seconds,
            teacher_wall_seconds,
            dart_noisy,
            prefix_forced_teacher,
            prefix_handoff,
            prefix_handoff_speed,
            active_gate_index,
            occupancy_mode,
            environment_step_seconds,
            feature_seconds,
            controller_wall_breakdown,
        ) = response[:30]
        quality_row = None
        if (self.settings.get('dagger_trajectory_quality', {}).get('enabled', False)
                or self.settings.get('dagger_collection_control')):
            if len(response) < 39 or np.shape(response[38]) != (QUALITY_WIDTH,):
                raise RuntimeError('DAgger worker omitted trajectory quality telemetry')
            quality_row = np.asarray(response[38], np.float64).copy()
            for key, value in (('queries', 1), ('misses', quality_row[1]),
                    ('backward', quality_row[2]), ('planned_plane_crossings', quality_row[17]), ('stopped', quality_row[11]),
                    ('internal_teacher_recovery_steps', quality_row[16]),
                    ('unqualified_steps', quality_row[0] == -1),
                    ('qualified_segments', quality_row[10] == 1), ('failed_segments', quality_row[10] == -1)):
                output.quality_counts[key] = output.quality_counts.get(key, 0) + int(value)
        reward_aux_enabled = int(self.settings.get("reward_aux_dim", 0)) > 0
        if reward_aux_enabled:
            if len(response) < 37:
                raise RuntimeError("DAgger worker omitted reward auxiliary labels")
            executed_action = np.asarray(response[34], np.float32)
            reward_components = np.asarray(response[36], np.float32)
            if executed_action.shape != (4,) or reward_components.shape != (
                len(PPO_REWARD_COMPONENTS),
            ):
                raise RuntimeError("DAgger reward auxiliary labels lost shape")
        if len(response) > 39:
            controller_mode, internal_recovery, feedback_fallback = response[39]
            for key, value in ((f'teacher_mode_{controller_mode}_queries', 1),
                               ('teacher_internal_recovery_queries', internal_recovery),
                               ('teacher_feedback_fallback_queries', feedback_fallback),
                               ('teacher_feedback_fallback_labels', feedback_fallback and valid)):
                output.quality_counts[key] = output.quality_counts.get(key, 0) + int(value)
        if len(response) > 40:
            for key, value in response[40].items():
                name = 'collection_' + key
                output.quality_counts[name] = output.quality_counts.get(name, 0) + int(value)
        output.total_queries += 1
        output.environment_steps += 1
        output.executed_teacher += int(used_teacher)
        output.solver_failures += int(solver_failed)
        output.recovery_queries += int(teacher_mode != 0)
        output.teacher_solve_seconds += float(teacher_solve_seconds)
        output.teacher_wall_seconds += float(teacher_wall_seconds)
        output.environment_step_seconds += float(environment_step_seconds)
        output.feature_seconds += float(feature_seconds)
        output.controller_wall_breakdown += np.asarray(
            controller_wall_breakdown, np.float64
        )
        if len(response) > 37:
            output.backend_wall_breakdown += np.asarray(response[37], np.float64)
        output.dart_noisy_steps += int(dart_noisy)
        output.expert_prefix_forced_steps += int(prefix_forced_teacher)
        output.completed_expert_prefix_handoffs += int(prefix_handoff)
        output.expert_prefix_handoff_speed_sum += float(prefix_handoff_speed)
        coherent_row = None
        if valid:
            output.valid_queries += 1
            target = (
                slot.episode_batch
                if bool(self.settings.get(
                    "dagger_require_successful_episodes", False
                )) else output
            )
            target.histories.append(normalized_history)
            coherent_row = len(target.histories) - 1
            target.actions.append(np.asarray(teacher_action, np.float32))
            target.previous_actions.append(np.asarray(previous_action, np.float32))
            target.dynamics.append(np.asarray(dynamics, np.float32))
            target.dynamics_valid.append(bool(dynamics_valid))
            target.tracks.append(sys.intern(str(track)))
            target.course_progress.append(float(gate_fraction))
            target.gate_phases.append(int(gate_phase))
            target.active_gate_indices.append(int(active_gate_index))
            target.teacher_modes.append(int(teacher_mode))
            target.occupancy_modes.append(int(occupancy_mode))
            if quality_row is not None:
                target.trajectory.append(quality_row)
            if quality_row is not None and quality_row[0] == 3:
                slot.pending_recovery_indices.append(len(target.trajectory)-1)
            target.speed_commands.append(float(slot.target_speed))
            if topology_target is not None:
                target.topology_targets.append(
                    np.asarray(topology_target, np.float32)
                )
                target.topology_valid.append(bool(topology_target_valid))
            if action_chunk_target is not None:
                target.action_chunks.append(
                    np.asarray(action_chunk_target, np.float32)
                )
            elif (
                float(self.settings.get("action_chunk_weight", 0.0)) > 0.0
                and str(self.settings.get(
                    "action_chunk_target_source", "mpcc_open_loop"
                )) in {"replanned_teacher", "coherent_teacher"}
            ):
                offsets = tuple(int(value) for value in self.settings.get(
                    "action_chunk_offsets", ()
                ))
                target.action_chunks.append(np.zeros(
                    (len(offsets), 4), np.float32
                ))
            if reward_aux_enabled:
                target.executed_actions.append(executed_action)
                target.reward_components.append(reward_components)
            target.successful_labels.append(False)
            if target is output:
                slot.episode_output_indices.append(len(output.histories) - 1)
        if self.settings.get('action_chunk_target_source') == 'coherent_teacher':
            slot.chunk_step_events.append((coherent_row, bool(used_teacher and not dart_noisy and not solver_failed)))
        if quality_row is not None and (quality_row[10] != 0 or done):
            for index in slot.pending_recovery_indices:
                output.trajectory[index][0] = 2 if quality_row[10] == 1 else -1
            slot.pending_recovery_indices.clear()
        slot.history.append_feature(feature)
        slot.done = bool(done)
        if not slot.done:
            return False
        if quality_row is not None and self.settings.get('dagger_trajectory_quality', {}).get('enabled', False):
            marked = mark_precursors(output.trajectory, slot.episode_output_indices,
                self.settings['dagger_trajectory_quality'], configured_control_hz(self.settings), quality_row)
            output.quality_counts['precursor_rows'] = output.quality_counts.get('precursor_rows', 0) + marked
        self._finalize_replanned_action_chunks(output, slot)
        if bool(self.settings.get("dagger_require_successful_episodes", False)):
            qualified_episode=bool(successful_episode)
            if self.midtrain_quality and qualified_episode:
                from starscream.midtrain_quality import reference_steps,admit
                reference,gates=self.midtrain_reference[str(Path(track).resolve())]
                target=reference_steps(reference,gates,slot.episode_start_gate,int(rollout_laps))
                qualified_episode=admit(self.midtrain_quality['method'],True,
                    slot.episode_step,target,str(track),slot.episode_index,self.collection_seed_base)
            if qualified_episode:
                output.histories.extend(slot.episode_batch.histories)
                output.actions.extend(slot.episode_batch.actions)
                output.previous_actions.extend(slot.episode_batch.previous_actions)
                output.dynamics.extend(slot.episode_batch.dynamics)
                output.dynamics_valid.extend(slot.episode_batch.dynamics_valid)
                output.tracks.extend(slot.episode_batch.tracks)
                output.course_progress.extend(slot.episode_batch.course_progress)
                output.gate_phases.extend(slot.episode_batch.gate_phases)
                output.active_gate_indices.extend(
                    slot.episode_batch.active_gate_indices
                )
                output.teacher_modes.extend(slot.episode_batch.teacher_modes)
                output.occupancy_modes.extend(slot.episode_batch.occupancy_modes)
                output.trajectory.extend(slot.episode_batch.trajectory)
                output.speed_commands.extend(slot.episode_batch.speed_commands)
                output.action_chunks.extend(slot.episode_batch.action_chunks)
                output.executed_actions.extend(slot.episode_batch.executed_actions)
                output.reward_components.extend(slot.episode_batch.reward_components)
                output.topology_targets.extend(slot.episode_batch.topology_targets)
                output.topology_valid.extend(slot.episode_batch.topology_valid)
                output.successful_labels.extend(
                    [True] * len(slot.episode_batch.histories)
                )
                output.accepted_episodes += 1
                output.accepted_multilap_episodes += int(int(rollout_laps) > 1)
                output.accepted_episodes_by_track[str(track)] = (
                    output.accepted_episodes_by_track.get(str(track), 0) + 1
                )
            else:
                output.rejected_episodes += 1
        elif successful_episode:
            for index in slot.episode_output_indices:
                output.successful_labels[index] = True
            output.accepted_episodes += 1
            output.accepted_multilap_episodes += int(int(rollout_laps) > 1)
            output.accepted_episodes_by_track[str(track)] = (
                output.accepted_episodes_by_track.get(str(track), 0) + 1
            )
        else:
            output.rejected_episodes += 1
        return True

    @staticmethod
    def _deterministic_teacher_draw(
        seed: int, episode_index: int, episode_step: int, beta: float,
    ) -> bool:
        """Schedule-independent Bernoulli draw for asynchronous collection."""

        value = (
            int(seed)
            + 0x9E3779B97F4A7C15 * (int(episode_index) + 1)
            + 0xD1B54A32D192ED03 * (int(episode_step) + 1)
        ) & ((1 << 64) - 1)
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
        value ^= value >> 31
        return (value / float(1 << 64)) < float(beta)

    def _collect_async(
        self,
        *,
        output: DaggerBatch,
        pending: dict[str, deque[tuple[int, float, int | None, float, str]]],
        collection_tracks: tuple[str, ...],
        episodes: int,
        beta: float,
        seed_base: int,
        dart: bool,
    ) -> DaggerBatch:
        """Keep MPCC workers busy while batching whichever actors are ready.

        The original collector imposed a global barrier after every simulator
        step.  A single difficult ACADOS query therefore idled all other
        workers.  This event loop preserves one ordered trajectory per worker
        but permits bounded lag between workers and micro-batches actor calls.
        """

        self.last_profile = dict(wait_seconds=0.0, inference_seconds=0.0,
                                 inference_batches=0, inference_rows=0)
        traffic_before = [sum(getattr(slot.connection, key, 0) for slot in self.slots)
                          for key in ('sent_bytes', 'received_bytes')]
        early_teacher = bool(self.settings.get('dagger_dispatch_teacher_first', False))
        pure_expert = bool(self.settings.get('dagger_pure_expert_collection', False))
        if pure_expert and (beta != 1.0 or dart or not self.lazy_teacher_actions
                            or not self.settings.get('dagger_require_successful_episodes')):
            raise ValueError('pure expert collection requires beta=1, lazy queries, successful episodes and no DART')
        minimum_batch = int(self.settings.get(
            "dagger_async_min_inference_batch", 4
        ))
        batch_wait_seconds = 1.0e-3 * float(self.settings.get(
            "dagger_async_batch_wait_ms", 1.0
        ))
        if minimum_batch < 1 or batch_wait_seconds < 0.0:
            raise ValueError("invalid asynchronous DAgger batching settings")
        if beta >= 1.0 - 1.0e-12:
            # No actor call is needed in expert-only warmup rounds. Dispatch
            # each worker as soon as it becomes ready instead of holding it for
            # an inference micro-batch that will never be executed.
            minimum_batch = 1
            batch_wait_seconds = 0.0

        def episode_dart_scale(index: int, track: str) -> float:
            if not dart:
                return 0.0
            if not dagger_track_allows_dart(track, self.settings):
                return 0.0
            fraction = float(self.settings.get(
                "dagger_dart_expert_episode_fraction", 1.0
            ))
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("dagger_dart_expert_episode_fraction must be in [0,1]")
            draw = np.random.default_rng(seed_base + 7919 * (index + 1)).random()
            return 1.0 if draw < fraction else 0.0

        def schedule(slot: ProcessDaggerSlot) -> bool:
            track = slot.track if pending.get(slot.track) else ""
            if not track:
                track = next(
                    (item for item in collection_tracks if pending[item]), ""
                )
            if not track:
                slot.done = True
                slot.waiting_for = None
                return False
            index, target_speed, start_gate_index, transition_scale, start_mode = (
                pending[track].popleft()
            )
            rollout_laps = self.track_rollout_laps[str(Path(track).resolve())]
            slot.connection.send((
                "reset", index, seed_base + 1009 * index, track, rollout_laps,
                float(target_speed), episode_dart_scale(index, track),
                start_gate_index, float(transition_scale), str(start_mode), beta < 1.0,
            ))
            slot.track = str(track)
            slot.target_speed = float(target_speed)
            slot.done = False
            slot.waiting_for = "reset"
            slot.episode_index = int(index)
            slot.episode_step = 0
            slot.episode_start_gate = int(start_gate_index or 0)
            slot.pending_normalized_history = None
            slot.force_policy_once = False
            slot.awaiting_lazy_response = False
            slot.episode_batch = self._new_batch()
            slot.episode_output_indices = []
            slot.chunk_step_events = []
            return True

        for slot in self.slots:
            schedule(slot)
        self.collection_seed_base=int(seed_base)
        completed = 0
        ready_slots: list[ProcessDaggerSlot] = []
        event_inference = bool(self.settings.get('dagger_event_inference', False))
        if event_inference and self.host_inference_policy is None:
            raise ValueError('Event inference requires the host CUDA inference graph')
        pending_inference = None

        def send_actions(dispatch, normalized_histories, teacher_draws, lazy,
                         early_indices, policy_actions):
            for index, (slot, policy_action) in enumerate(zip(dispatch, policy_actions)):
                if index in early_indices:
                    continue
                if slot.waiting_for is not None:
                    raise RuntimeError('Cannot overwrite an in-flight DAgger action/history')
                use_teacher = teacher_draws[index]
                slot.pending_normalized_history = normalized_histories[index].astype(
                    self.collection_history_dtype, copy=True)
                request = (('advance_teacher',) if lazy and use_teacher and not slot.force_policy_once
                           else ('advance', policy_action, use_teacher))
                slot.connection.send(request)
                slot.awaiting_lazy_response = request[0] == 'advance_teacher'
                slot.force_policy_once = False
                slot.waiting_for = 'step'
                slot.episode_step += 1

        while completed < episodes:
            if pending_inference is not None and pending_inference[0].ready():
                ticket, saved, indices, started = pending_inference
                actions = np.zeros((len(saved[0]), 4), np.float32)
                actions[indices] = ticket.result()
                send_actions(*saved, actions)
                self.last_profile['inference_seconds'] += time.perf_counter() - started
                pending_inference = None
            waiting = [
                slot for slot in self.slots if slot.waiting_for is not None
            ]
            desired = min(minimum_batch, len(waiting) + len(ready_slots))
            deadline: float | None = None
            while (len(ready_slots) < desired or pending_inference is not None) and waiting:
                timeout = (
                    None if not ready_slots
                    else max(
                        0.0,
                        (deadline or time.perf_counter()) - time.perf_counter(),
                    )
                )
                wait_started = time.perf_counter()
                if pending_inference is not None:
                    timeout = .0002
                ready_connections = wait_connections(
                    [slot.connection for slot in waiting], timeout=timeout
                )
                self.last_profile["wait_seconds"] += time.perf_counter() - wait_started
                if not ready_connections:
                    break
                connection_ids = {id(item) for item in ready_connections}
                responding = [
                    slot for slot in waiting
                    if id(slot.connection) in connection_ids
                ]
                for slot in responding:
                    expected = slot.waiting_for
                    assert expected is not None
                    response = self._receive(slot, expected)
                    slot.waiting_for = None
                    if response[0] == 'need_policy':
                        if not self.lazy_teacher_actions or not slot.awaiting_lazy_response:
                            raise RuntimeError('unexpected/repeated lazy teacher fallback')
                        if pure_expert:
                            # MPCC declared this state infeasible. Reject the
                            # whole trajectory before executing a placeholder
                            # or learner action; coverage retries refill it.
                            output.rejected_episodes += 1
                            output.pure_expert_aborted_episodes += 1
                            slot.episode_batch = self._new_batch()
                            slot.pending_normalized_history = None
                            slot.awaiting_lazy_response = False
                            completed += 1
                            schedule(slot)
                            continue
                        slot.awaiting_lazy_response = False
                        slot.force_policy_once = True
                        slot.episode_step -= 1
                        slot.pending_normalized_history = None
                        ready_slots.append(slot)
                        if deadline is None:
                            deadline = time.perf_counter() + batch_wait_seconds
                        continue
                    slot.awaiting_lazy_response = False
                    if expected == "reset":
                        slot.history.reset_feature(response[1])
                        ready_slots.append(slot)
                        if deadline is None:
                            deadline = time.perf_counter() + batch_wait_seconds
                        continue
                    normalized_history = slot.pending_normalized_history
                    if normalized_history is None:
                        raise RuntimeError("asynchronous DAgger history was lost")
                    slot.pending_normalized_history = None
                    if self._record_step_response(
                        output, slot, normalized_history, response
                    ):
                        completed += 1
                        schedule(slot)
                    else:
                        ready_slots.append(slot)
                        if deadline is None:
                            deadline = time.perf_counter() + batch_wait_seconds
                waiting = [
                    slot for slot in self.slots if slot.waiting_for is not None
                ]
                if (
                    ready_slots and deadline is not None
                    and time.perf_counter() >= deadline
                ):
                    break

                if pending_inference is not None and pending_inference[0].ready():
                    break

            if pending_inference is not None:
                # With no other worker to service, block on the event instead
                # of spinning. Otherwise return to servicing ready pipes.
                if not waiting:
                    pending_inference[0].result()
                continue
            if not ready_slots:
                continue
            dispatch = ready_slots
            ready_slots = []
            histories = np.stack(
                [slot.history.array() for slot in dispatch], axis=0
            )
            # Normalize in place, then create exactly one retained history in
            # the configured replay dtype.  Previously this path allocated a
            # float32 stack, a second normalized stack, a per-slot copy, and a
            # later per-label dtype conversion.
            np.subtract(histories, self.normalizer.mean, out=histories)
            np.divide(histories, self.normalizer.std, out=histories)
            normalized_histories = histories
            teacher_draws = [self._deterministic_teacher_draw(
                seed_base, slot.episode_index, slot.episode_step, beta) for slot in dispatch]
            lazy = pure_expert or (self.lazy_teacher_actions and 0. < beta < 1.0 - 1.0e-12)
            actor_indices = [i for i, slot in enumerate(dispatch)
                             if not lazy or not teacher_draws[i] or slot.force_policy_once]
            early_indices = set()
            if (early_teacher or event_inference) and lazy:
                # These workers need no learner result. Advance them while the
                # host waits for actor inference on the other workers.
                for index, slot in enumerate(dispatch):
                    if teacher_draws[index] and not slot.force_policy_once:
                        slot.pending_normalized_history = normalized_histories[index].astype(
                            self.collection_history_dtype, copy=True)
                        slot.connection.send(('advance_teacher',))
                        slot.awaiting_lazy_response = True
                        slot.waiting_for = 'step'
                        slot.episode_step += 1
                        early_indices.add(index)
            if beta >= 1.0 - 1.0e-12:
                # The policy action is counterfactual in a pure expert round.
                policy_actions = np.zeros((len(dispatch), 4), np.float32)
            elif not actor_indices:
                policy_actions = np.zeros((len(dispatch), 4), np.float32)
            else:
                inference_started = time.perf_counter()
                actor_histories = normalized_histories[actor_indices] if lazy else normalized_histories
                actor_slots = [dispatch[i] for i in actor_indices] if lazy else dispatch
                if self.host_inference_policy is not None:
                    policy_actions = self.host_inference_policy.predict_numpy(
                        actor_histories,
                        np.asarray([slot.target_speed for slot in actor_slots], np.float32),
                        asynchronous=event_inference,
                    )
                else:
                    tensor = torch.from_numpy(actor_histories).to(self.device)
                    with torch.no_grad():
                        speed_tensor = torch.as_tensor(
                            [slot.target_speed for slot in actor_slots],
                            device=self.device, dtype=tensor.dtype,
                        )
                        policy_actions = self.inference_policy(
                            tensor, speed_tensor
                        ).float().cpu().numpy()
                self.last_profile["inference_batches"] += 1
                self.last_profile["inference_rows"] += len(actor_slots)
                if event_inference:
                    pending_inference = (policy_actions,
                        (dispatch, normalized_histories, teacher_draws, lazy, early_indices),
                        actor_indices, inference_started)
                    continue
                self.last_profile["inference_seconds"] += time.perf_counter() - inference_started
                if lazy:
                    expanded = np.zeros((len(dispatch), 4), np.float32)
                    expanded[actor_indices] = policy_actions
                    policy_actions = expanded
            send_actions(dispatch, normalized_histories, teacher_draws, lazy,
                         early_indices, policy_actions)

        for key, previous in zip(('sent_bytes', 'received_bytes'), traffic_before):
            self.last_profile['ipc_'+key] = sum(
                getattr(slot.connection, key, 0) for slot in self.slots) - previous
        if not (
            len(output.histories) == len(output.actions)
            == len(output.previous_actions) == len(output.dynamics)
            == len(output.dynamics_valid) == len(output.tracks)
            == len(output.course_progress) == len(output.gate_phases)
            == len(output.active_gate_indices) == len(output.teacher_modes)
            == len(output.occupancy_modes)
            == len(output.speed_commands)
            == len(output.successful_labels)
        ):
            raise RuntimeError("asynchronous DAgger labels lost alignment")
        if (
            float(self.settings.get("topology_weight", 0.0)) > 0.0
            and (
                len(output.topology_targets) != len(output.histories)
                or len(output.topology_valid) != len(output.histories)
            )
        ):
            raise RuntimeError("asynchronous DAgger topology targets lost alignment")
        if (
            float(self.settings.get("action_chunk_weight", 0.0)) > 0.0
            and len(output.action_chunks) != len(output.histories)
        ):
            raise RuntimeError("asynchronous DAgger action chunks lost alignment")
        if int(self.settings.get("reward_aux_dim", 0)) > 0 and (
            len(output.executed_actions) != len(output.histories)
            or len(output.reward_components) != len(output.histories)
        ):
            raise RuntimeError("asynchronous DAgger reward labels lost alignment")
        return output

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for slot in self.slots:
            if slot.process.is_alive():
                try:
                    slot.connection.send(("close",))
                except (BrokenPipeError, EOFError):
                    pass
        for slot in self.slots:
            if slot.process.is_alive():
                try:
                    self._receive(slot, "closed")
                except (BrokenPipeError, EOFError, RuntimeError):
                    pass
            slot.process.join(timeout=5.0)
            if slot.process.is_alive():
                slot.process.terminate()
                slot.process.join(timeout=2.0)
            if slot.process.is_alive():
                slot.process.kill()
                slot.process.join(timeout=2.0)
            slot.connection.close()
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def collect(
        self, *, episodes: int, beta: float, seed_base: int,
        tracks: tuple[str, ...] | None = None,
        dart: bool = False,
    ) -> DaggerBatch:
        self.policy.eval()
        output = self._new_batch(compact=True)
        completed = 0
        collection_tracks = tuple(tracks or self.stage.tracks)
        unknown = set(collection_tracks) - set(self.stage.tracks)
        if unknown:
            raise ValueError(f"DAgger supplemental tracks are outside the stage: {unknown}")
        schedule = dagger_episode_tracks(
            collection_tracks, episodes, self.settings, seed_base
        )
        start_gates = dagger_episode_start_gates(
            schedule, self.settings, seed_base
        )
        start_modes = dagger_episode_start_modes(
            schedule, start_gates, self.settings, seed_base
        )
        transition_scale = float(self.settings.get(
            "dagger_transition_start_perturbation_scale", 1.0
        ))
        if not np.isfinite(transition_scale) or transition_scale < 1.0:
            raise ValueError(
                "dagger_transition_start_perturbation_scale must be finite and >= 1"
            )
        output.scheduled_targeted_start_episodes = sum(
            item is not None for item in start_gates
        )
        output.scheduled_expert_prefix_episodes = (
            sum(mode == "expert_prefix" for mode in start_modes)
        )
        output.scheduled_canonical_start_episodes = (
            len(start_gates) - output.scheduled_targeted_start_episodes
        )
        pending: dict[
            str, deque[tuple[int, float, int | None, float, str]]
        ] = {}
        schedule_rng = np.random.default_rng(seed_base ^ 0x51EEDF10)
        for track in collection_tracks:
            indices = [
                index for index, item in enumerate(schedule) if item == track
            ]
            fractions = np.resize(
                np.asarray(self.speed_fractions, np.float64), len(indices)
            )
            if self.settings.get('dagger_teacher_speed_fraction_weights'):
                from starscream.dagger_schedule import weighted_speed_fractions
                fractions = np.asarray(weighted_speed_fractions(
                    self.speed_fractions,
                    self.settings['dagger_teacher_speed_fraction_weights'], len(indices),
                    rng=schedule_rng if self.settings.get('dagger_teacher_speed_stochastic_quotas', False) else None,
                ), np.float64)
            schedule_rng.shuffle(fractions)
            frontier = self.track_target_speeds.get(
                str(Path(track).resolve()), float(self.policy.default_speed_command)
            )
            pending[track] = deque(
                (
                    index,
                    float(frontier * fraction),
                    start_gates[index],
                    transition_scale if start_gates[index] is not None else 1.0,
                    start_modes[index],
                )
                for index, fraction in zip(indices, fractions)
            )

        if bool(self.settings.get("dagger_async_collection", False)):
            return self._collect_async(
                output=output,
                pending=pending,
                collection_tracks=collection_tracks,
                episodes=episodes,
                beta=beta,
                seed_base=seed_base,
                dart=dart,
            )

        def episode_dart_scale(index: int, track: str) -> float:
            if not dart:
                return 0.0
            if not dagger_track_allows_dart(track, self.settings):
                return 0.0
            fraction = float(self.settings.get(
                "dagger_dart_expert_episode_fraction", 1.0
            ))
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("dagger_dart_expert_episode_fraction must be in [0,1]")
            draw = np.random.default_rng(seed_base + 7919 * (index + 1)).random()
            return 1.0 if draw < fraction else 0.0

        def schedule(slot: ProcessDaggerSlot) -> bool:
            track = slot.track if pending.get(slot.track) else ""
            if not track:
                track = next((item for item in collection_tracks if pending[item]), "")
            if not track:
                slot.done = True
                return False
            index, target_speed, start_gate_index, start_scale, start_mode = (
                pending[track].popleft()
            )
            self._reset_slot(
                slot, index, seed_base, track, target_speed,
                episode_dart_scale(index, track), start_gate_index, start_scale,
                start_mode, collection_active=beta < 1.0,
            )
            return True

        for slot in self.slots:
            schedule(slot)
        rng = np.random.default_rng(seed_base)
        while completed < episodes:
            active = [slot for slot in self.slots if not slot.done]
            histories = np.stack([slot.history.array() for slot in active])
            normalized_histories = self.normalizer.numpy(histories)
            tensor = torch.from_numpy(normalized_histories).to(self.device)
            with torch.no_grad():
                speed_tensor = torch.as_tensor(
                    [slot.target_speed for slot in active],
                    device=self.device, dtype=tensor.dtype,
                )
                policy_actions = self.inference_policy(tensor, speed_tensor).float().cpu().numpy()
            for slot, policy_action in zip(active, policy_actions):
                slot.connection.send(("advance", policy_action, rng.random() < beta))
            responses = [self._receive(slot, "step") for slot in active]
            finished: list[ProcessDaggerSlot] = []
            for active_index, (slot, response) in enumerate(zip(active, responses)):
                if self._record_step_response(
                    output, slot, normalized_histories[active_index], response
                ):
                    finished.append(slot)
            for slot in finished:
                completed += 1
                schedule(slot)
        if not (
            len(output.histories) == len(output.actions)
            == len(output.previous_actions) == len(output.dynamics)
            == len(output.dynamics_valid) == len(output.tracks)
            == len(output.course_progress)
            == len(output.gate_phases)
            == len(output.active_gate_indices)
            == len(output.teacher_modes)
            == len(output.occupancy_modes)
            == len(output.speed_commands)
            == len(output.successful_labels)
        ):
            raise RuntimeError("process DAgger labels lost alignment")
        if (
            float(self.settings.get("topology_weight", 0.0)) > 0.0
            and (
                len(output.topology_targets) != len(output.histories)
                or len(output.topology_valid) != len(output.histories)
            )
        ):
            raise RuntimeError("process DAgger topology targets lost alignment")
        if (
            float(self.settings.get("action_chunk_weight", 0.0)) > 0.0
            and len(output.action_chunks) != len(output.histories)
        ):
            raise RuntimeError("process DAgger action chunks lost alignment")
        return output

    def collect_aggrevate(
        self,
        *,
        queries: int,
        rollin_beta: float,
        seed_base: int,
        config: AggreVaTeConfig,
        tracks: tuple[str, ...] | None = None,
        maximum_attempt_multiplier: int = 4,
    ) -> tuple[
        AggreVaTeQueryBatch, AggreVaTeImitationBatch, dict[str, float]
    ]:
        """Collect canonical roll-in/query/expert-rollout cost-to-go samples.

        Each accepted episode contributes exactly one query. Before the query
        the state distribution is induced by the scheduled expert/learner
        mixture. At the uniformly sampled query time one bounded action is
        executed; all later commands come from MPCC. This is the continuous
        control realization of the AggreVaTe data contract.
        """

        config.validate()
        if queries < 1 or not 0.0 <= float(rollin_beta) <= 1.0:
            raise ValueError("invalid AggreVaTe collection request")
        if maximum_attempt_multiplier < 1:
            raise ValueError("maximum_attempt_multiplier must be positive")
        self.policy.eval()
        collection_tracks = tuple(tracks or self.stage.tracks)
        unknown = set(collection_tracks) - set(self.stage.tracks)
        if unknown:
            raise ValueError(f"AggreVaTe tracks are outside the stage: {unknown}")
        maximum_attempts = int(queries) * int(maximum_attempt_multiplier)
        schedule = dagger_episode_tracks(
            collection_tracks, maximum_attempts, self.settings, seed_base
        )
        track_ids = {
            str(Path(track).resolve()): index
            for index, track in enumerate(self.stage.tracks)
        }
        rng = np.random.default_rng(seed_base ^ 0xA663E)
        resolved_collection_tracks = tuple(
            str(Path(track).resolve()) for track in collection_tracks
        )
        base_quota, quota_extra = divmod(queries, len(resolved_collection_tracks))
        query_quota = {
            track: base_quota + int(index < quota_extra)
            for index, track in enumerate(resolved_collection_tracks)
        }
        accepted_by_track = {track: 0 for track in resolved_collection_tracks}
        mode_names = (
            "expert_query", "learner_query", "teacher_perturbation_query"
        )
        mode_probabilities = np.asarray([
            config.expert_query_fraction,
            config.learner_query_fraction,
            config.teacher_perturbation_fraction,
        ], np.float64)
        mode_ids = {name: index for index, name in enumerate(mode_names)}
        noise_std = np.asarray(config.teacher_noise_std, np.float32)

        states: dict[int, dict[str, Any]] = {}
        next_attempt = 0
        accepted: list[dict[str, Any]] = []
        imitation_rows: dict[str, list[Any]] = {
            name: [] for name in AggreVaTeImitationBatch.__dataclass_fields__
        }
        rejected_before_query = 0
        rejected_invalid_query = 0
        rejected_over_quota = 0
        total_steps = 0
        teacher_steps = 0

        def schedule_slot(slot: ProcessDaggerSlot) -> bool:
            nonlocal next_attempt
            if len(accepted) >= queries:
                slot.done = True
                states.pop(id(slot), None)
                return False
            while next_attempt < maximum_attempts:
                attempt = next_attempt
                next_attempt += 1
                track = schedule[attempt]
                resolved_track = str(Path(track).resolve())
                if accepted_by_track[resolved_track] < query_quota[resolved_track]:
                    break
            else:
                slot.done = True
                states.pop(id(slot), None)
                return False
            frontier = self.track_target_speeds.get(
                str(Path(track).resolve()),
                float(self.policy.default_speed_command),
            )
            fractions = self.speed_fractions
            target_speed = float(frontier * fractions[attempt % len(fractions)])
            if not np.isfinite(target_speed) or target_speed <= 0.0:
                raise ValueError(
                    f"AggreVaTe has no positive MPCC pace for track {track}"
                )
            self._reset_slot(
                slot, attempt, seed_base, track, target_speed,
                0.0, None, 1.0, "canonical",
            )
            query_high = min(
                config.query_max_step,
                max(int(self.stage.max_steps) - 1, config.query_min_step),
            )
            if config.query_progress_strata:
                center = config.query_progress_strata[
                    attempt % len(config.query_progress_strata)
                ]
                query_progress = float(np.clip(
                    center + rng.uniform(
                        -config.query_progress_jitter,
                        config.query_progress_jitter,
                    ), 0.0, 1.0,
                ))
                query_step: int | None = None
            else:
                query_progress = -1.0
                query_step = int(rng.integers(
                    config.query_min_step, query_high + 1
                ))
            mode = str(rng.choice(mode_names, p=mode_probabilities))
            states[id(slot)] = {
                "attempt": attempt,
                "step": 0,
                "query_step": query_step,
                "query_max_step": query_high,
                "query_progress": query_progress,
                "gate_fraction": 0.0,
                "mode": mode,
                "queried": False,
                "query_valid": False,
                "history": None,
                "query_action": None,
                "expert_action": None,
                "costs": [],
                "remaining_fraction": 1.0,
                "query_steps_after": 0,
                "track": str(track),
                "speed": target_speed,
            }
            return True

        for slot in self.slots:
            schedule_slot(slot)

        while len(accepted) < queries:
            active = [slot for slot in self.slots if not slot.done]
            if not active:
                break
            for slot in active:
                state = states[id(slot)]
                if (
                    not bool(state["queried"])
                    and state["query_step"] is None
                    and int(state["step"]) >= config.query_min_step
                    and (
                        float(state["gate_fraction"])
                        >= float(state["query_progress"])
                        or int(state["step"]) >= int(state["query_max_step"])
                    )
                ):
                    state["query_step"] = int(state["step"])
            raw_histories = np.stack([slot.history.array() for slot in active])
            normalized_histories = self.normalizer.numpy(raw_histories)
            policy_indices = [
                index for index, slot in enumerate(active)
                if not bool(states[id(slot)]["queried"])
            ]
            policy_actions = np.zeros((len(active), 4), np.float32)
            if policy_indices:
                policy_tensor = torch.from_numpy(
                    normalized_histories[policy_indices]
                ).to(self.device)
                speed_tensor = torch.as_tensor(
                    [states[id(active[index])]["speed"] for index in policy_indices],
                    device=self.device, dtype=policy_tensor.dtype,
                )
                with torch.no_grad():
                    inferred = self.policy(
                        policy_tensor, speed_tensor
                    ).float().cpu().numpy()
                policy_actions[np.asarray(policy_indices)] = inferred

            for index, slot in enumerate(active):
                state = states[id(slot)]
                step = int(state["step"])
                query_step = state["query_step"]
                if query_step is None or step < int(query_step):
                    execution_mode = "mixture"
                    teacher_draw = self._deterministic_teacher_draw(
                        seed_base, int(state["attempt"]), step, rollin_beta
                    )
                    noise = np.zeros(4, np.float32)
                elif step == int(state["query_step"]):
                    execution_mode = str(state["mode"])
                    teacher_draw = execution_mode == "expert_query"
                    noise = (
                        rng.normal(0.0, noise_std).astype(np.float32)
                        if execution_mode == "teacher_perturbation_query"
                        else np.zeros(4, np.float32)
                    )
                    state["history"] = normalized_histories[index].astype(
                        self.collection_history_dtype, copy=True
                    )
                else:
                    execution_mode = "post_query_expert"
                    teacher_draw = True
                    noise = np.zeros(4, np.float32)
                slot.connection.send((
                    "advance", policy_actions[index], teacher_draw,
                    execution_mode, noise,
                ))

            responses = [self._receive(slot, "step") for slot in active]
            finished: list[ProcessDaggerSlot] = []
            for index, (slot, response) in enumerate(zip(active, responses)):
                state = states[id(slot)]
                step = int(state["step"])
                total_steps += 1
                teacher_steps += int(bool(response[7]))
                if bool(response[1]):
                    imitation_rows["histories"].append(
                        normalized_histories[index].astype(
                            self.collection_history_dtype, copy=True
                        )
                    )
                    imitation_rows["expert_actions"].append(
                        np.asarray(response[2], np.float32)
                    )
                    imitation_rows["previous_actions"].append(
                        np.asarray(response[3], np.float32)
                    )
                    imitation_rows["dynamics"].append(
                        np.asarray(response[4], np.float32)
                    )
                    imitation_rows["dynamics_valid"].append(bool(response[5]))
                    imitation_rows["speed_commands"].append(float(state["speed"]))
                    imitation_rows["track_ids"].append(
                        track_ids[str(Path(state["track"]).resolve())]
                    )
                if (
                    state["query_step"] is not None
                    and step == int(state["query_step"])
                ):
                    state["queried"] = True
                    state["query_valid"] = bool(response[1])
                    if state["query_valid"]:
                        state["expert_action"] = np.asarray(response[2], np.float32)
                        state["query_action"] = np.asarray(response[34], np.float32)
                        state["remaining_fraction"] = float(
                            1.0 - np.clip(float(response[12]), 0.0, 1.0)
                        )
                if bool(state["queried"]):
                    state["costs"].append(float(config.step_cost))
                    state["query_steps_after"] += 1
                state["step"] = step + 1
                state["gate_fraction"] = float(response[12])
                slot.history.append_feature(response[6])
                slot.done = bool(response[8])
                horizon_reached = bool(
                    state["queried"]
                    and int(state["query_steps_after"]) >= config.query_horizon
                )
                if slot.done or horizon_reached:
                    finished.append(slot)
                    if not bool(state["queried"]):
                        rejected_before_query += 1
                    elif not bool(state["query_valid"]):
                        rejected_invalid_query += 1
                    else:
                        completed = bool(response[14])
                        crashed = bool(response[32])
                        gate_fraction = float(response[33])
                        terminal = minimum_time_terminal_cost(
                            completed=completed,
                            crashed=crashed,
                            gate_fraction=gate_fraction,
                            config=config,
                        )
                        resolved_track = str(Path(state["track"]).resolve())
                        if accepted_by_track[resolved_track] >= query_quota[resolved_track]:
                            rejected_over_quota += 1
                            continue
                        accepted_by_track[resolved_track] += 1
                        accepted.append({
                            "histories": state["history"],
                            "query_actions": state["query_action"],
                            "expert_actions": state["expert_action"],
                            "costs_to_go": discounted_cost_to_go(
                                state["costs"], config.discount,
                                terminal_cost=terminal,
                            ),
                            "remaining_fractions": state["remaining_fraction"],
                            "speed_commands": state["speed"],
                            "track_ids": track_ids[str(
                                Path(state["track"]).resolve()
                            )],
                            "query_steps": state["query_step"],
                            "query_modes": mode_ids[str(state["mode"])],
                            "completed": completed,
                            "crashed": crashed,
                        })
            for slot in finished:
                schedule_slot(slot)

        if len(accepted) < queries:
            raise RuntimeError(
                "AggreVaTe query coverage failed: "
                f"accepted={len(accepted)} requested={queries} "
                f"attempted={next_attempt} before_query={rejected_before_query} "
                f"invalid_query={rejected_invalid_query}"
            )
        accepted = accepted[:queries]
        batch = AggreVaTeQueryBatch(**{
            name: np.asarray([item[name] for item in accepted])
            for name in AggreVaTeQueryBatch.__dataclass_fields__
        })
        dynamics_dim = mpcc_invariant_dynamics_target_dim(self.settings)
        imitation = AggreVaTeImitationBatch(
            histories=np.asarray(
                imitation_rows["histories"], dtype=self.collection_history_dtype
            ).reshape(-1, self.policy.context_steps, self.policy.input_dim),
            expert_actions=np.asarray(
                imitation_rows["expert_actions"], np.float32
            ).reshape(-1, 4),
            previous_actions=np.asarray(
                imitation_rows["previous_actions"], np.float32
            ).reshape(-1, 4),
            dynamics=np.asarray(
                imitation_rows["dynamics"], np.float32
            ).reshape(-1, dynamics_dim),
            dynamics_valid=np.asarray(
                imitation_rows["dynamics_valid"], np.bool_
            ),
            speed_commands=np.asarray(
                imitation_rows["speed_commands"], np.float32
            ),
            track_ids=np.asarray(imitation_rows["track_ids"], np.int16),
        )
        metrics = {
            "queries": float(len(batch.histories)),
            "hybrid_imitation_labels": float(len(imitation.histories)),
            "attempts": float(next_attempt),
            "query_acceptance_fraction": len(batch.histories) / max(next_attempt, 1),
            "rejected_before_query": float(rejected_before_query),
            "rejected_invalid_query": float(rejected_invalid_query),
            "rejected_over_quota": float(rejected_over_quota),
            "environment_steps": float(total_steps),
            "teacher_execution_fraction": teacher_steps / max(total_steps, 1),
            "cost_mean": float(batch.costs_to_go.mean()),
            "cost_std": float(batch.costs_to_go.std()),
            "completion_fraction": float(batch.completed.mean()),
            "crash_fraction": float(batch.crashed.mean()),
        }
        for name, mode_id in mode_ids.items():
            metrics[f"query_mode/{name}"] = float(
                np.mean(batch.query_modes == mode_id)
            )
        for track_id, track in enumerate(self.stage.tracks):
            metrics[f"query_track/{Path(track).stem}"] = float(
                np.mean(batch.track_ids == track_id)
            )
        return batch, imitation, metrics


class ParallelDaggerCollector:
    """Persistent isolated env/MPCC agents with batched GPU actor inference."""

    def __init__(
        self,
        policy: PrivilegedMLPPolicy,
        normalizer: FeatureNormalizer,
        settings: Mapping[str, Any],
        stage: RacingCurriculumStage,
        device: str,
    ) -> None:
        self.policy = policy
        self.normalizer = normalizer
        if settings.get("dagger_trajectory_quality", {}).get("enabled", False):
            raise ValueError("trajectory quality requires ProcessDaggerCollector")
        self.settings = settings
        self.stage = stage
        self.device = device
        self.parallel = int(settings.get("rollout_envs", 12))
        collection_dtype = str(settings.get(
            "dagger_collection_history_dtype",
            settings.get("dagger_replay_history_dtype", "float32"),
        ))
        if collection_dtype not in {"float16", "float32"}:
            raise ValueError(
                "dagger_collection_history_dtype must be float16 or float32"
            )
        self.collection_history_dtype = (
            np.float16 if collection_dtype == "float16" else np.float32
        )
        self.slots: list[RolloutSlot] = []
        # ACADOS construction is expensive. Build exactly one solver per agent
        # and reuse it across every episode and DAgger round.
        for index in range(self.parallel):
            env = make_env(
                settings, track=stage.tracks[index % len(stage.tracks)]
            )
            observation, start = reset_env(env, stage, seed=index, episode_index=index)
            history = CausalHistory(policy.context_steps)
            history.reset(observation, settings)
            self.slots.append(RolloutSlot(
                env, observation, history, start, index,
                min(stage.target_gates, len(env.track.gates)), stage.max_steps,
                expert=RoutedDaggerTeacher(env, settings), done=True,
            ))
        self.executor = ThreadPoolExecutor(max_workers=self.parallel)

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        for slot in self.slots:
            slot.env.close()

    def _reset_slot(self, slot: RolloutSlot, index: int, seed_base: int) -> None:
        observation, start = reset_env(
            slot.env, self.stage,
            seed=seed_base + 1009 * index, episode_index=index,
        )
        assert slot.expert is not None
        slot.expert.reset()
        slot.observation = observation
        slot.history.reset_feature(ppo_observation_features(observation, self.settings))
        slot.start_passed = start
        slot.episode_index = index
        slot.target_gates = min(self.stage.target_gates, len(slot.env.track.gates))
        slot.max_steps = self.stage.max_steps
        slot.steps = 0
        slot.total_return = 0.0
        slot.crashed = False
        slot.done = False
        slot.steps_since_crossing = None

    def collect(
        self, *, episodes: int, beta: float, seed_base: int,
        tracks: tuple[str, ...] | None = None, dart: bool = False,
    ) -> DaggerBatch:
        if tracks is not None or dart:
            raise ValueError(
                "supplemental-track and DART collection require process collector"
            )
        if bool(self.settings.get("dagger_require_successful_episodes", False)):
            raise ValueError(
                "successful-episode label qualification requires process collector"
            )
        self.policy.eval()
        output = DaggerBatch(histories_normalized=True)
        scheduled = 0
        completed = 0
        for slot in self.slots:
            if scheduled < episodes:
                self._reset_slot(slot, scheduled, seed_base)
                scheduled += 1
            else:
                slot.done = True
        rng = np.random.default_rng(seed_base)
        while completed < episodes:
            active = [slot for slot in self.slots if not slot.done]
            histories = np.stack([slot.history.array() for slot in active])
            normalized_histories = self.normalizer.numpy(histories)
            tensor = torch.from_numpy(normalized_histories).to(self.device)
            with torch.no_grad():
                policy_actions = self.policy(tensor).float().cpu().numpy()
            teacher_started = time.perf_counter()
            commands = list(self.executor.map(
                lambda slot: slot.expert(slot.observation), active
            ))
            output.teacher_wall_seconds += time.perf_counter() - teacher_started
            output.teacher_solve_seconds += sum(
                float(command.solve_time) for command in commands
            )
            output.total_queries += len(commands)
            executed: list[np.ndarray] = []
            recorded: list[bool] = []
            for slot, command, policy_action, history, normalized_history in zip(
                active, commands, policy_actions, histories, normalized_histories
            ):
                finite = bool(command.valid) and np.all(
                    np.isfinite(command.action.as_array())
                )
                valid = bool(
                    finite
                    and (
                        not bool(self.settings.get(
                            "dagger_require_solver_success", False
                        ))
                        or int(command.solver_status) == 0
                    )
                    and float(command.constraint_margin) >= float(
                        self.settings.get(
                            "dagger_minimum_teacher_constraint_margin",
                            -float("inf"),
                        )
                    )
                )
                recorded.append(valid)
                if valid:
                    output.valid_queries += 1
                    output.histories.append(
                        normalized_history.astype(self.collection_history_dtype)
                    )
                    output.actions.append(ppo_ctbr_to_normalized(
                        command.action.as_array(), self.settings
                    ))
                    output.previous_actions.append(
                        history[-1, action_feature_slice(self.settings)].copy()
                    )
                    output.tracks.append(slot.env.track.name)
                    output.course_progress.append(
                        (slot.env.tracker.passed_count - slot.start_passed)
                        / max(slot.target_gates, 1)
                    )
                use_teacher = finite and rng.random() < beta
                output.executed_teacher += int(use_teacher)
                physical_action = (
                    command.action.as_array() if use_teacher
                    else ppo_normalized_to_ctbr(policy_action, self.settings)
                )
                slot.expert.observe_executed_action(physical_action)
                executed.append(physical_action)
                output.solver_failures += int(command.solver_status != 0)
                output.recovery_queries += int(slot.expert.last_routed)
            futures = {
                index: self.executor.submit(slot.env.step, executed[index])
                for index, slot in enumerate(active)
            }
            finished: list[RolloutSlot] = []
            for index, slot in enumerate(active):
                prior_task = dynamics_observation_task(slot.observation, self.settings)
                observation, _, terminated, _, info = futures[index].result()
                gate_passed = bool(info.get("gate_passed", False))
                if recorded[index]:
                    output.dynamics.append(_normalize_dynamics_delta_for_control_rate(
                        dynamics_observation_task(observation, self.settings) - prior_task,
                        self.settings,
                    ))
                    output.dynamics_valid.append(not gate_passed)
                    post_horizon = int(self.settings.get(
                        "dagger_gate_phase_post_crossing_horizon", 12
                    ))
                    acquisition_horizon = int(self.settings.get(
                        "dagger_gate_phase_acquisition_horizon", 48
                    ))
                    if gate_passed:
                        gate_phase = 2
                    elif (
                        slot.steps_since_crossing is not None
                        and slot.steps_since_crossing < post_horizon
                    ):
                        gate_phase = 3
                    elif (
                        slot.steps_since_crossing is not None
                        and slot.steps_since_crossing < acquisition_horizon
                    ):
                        gate_phase = 4
                    else:
                        gate_phase = 0
                    output.gate_phases.append(gate_phase)
                if gate_passed:
                    slot.steps_since_crossing = 0
                elif slot.steps_since_crossing is not None:
                    slot.steps_since_crossing += 1
                slot.observation = observation
                slot.history.append_feature(
                    ppo_observation_features(observation, self.settings)
                )
                slot.steps += 1
                output.environment_steps += 1
                gates = slot.env.tracker.passed_count - slot.start_passed
                slot.crashed = bool(info.get("ground_contact") or info.get("unity_collision"))
                slot.done = bool(
                    terminated or gates >= slot.target_gates or slot.steps >= slot.max_steps
                )
                if slot.done:
                    finished.append(slot)
            for slot in finished:
                completed += 1
                if scheduled < episodes:
                    self._reset_slot(slot, scheduled, seed_base)
                    scheduled += 1
        if not (
            len(output.histories) == len(output.actions)
            == len(output.previous_actions) == len(output.dynamics)
            == len(output.dynamics_valid) == len(output.tracks)
            == len(output.course_progress) == len(output.gate_phases)
        ):
            raise RuntimeError("DAgger label and dynamics arrays lost alignment")
        return output


def run_dagger(config: dict[str, Any], device: str) -> None:
    settings = config["dagger"]
    from starscream.evaluation_suite import configure_selection_suite
    configure_selection_suite(settings)
    print('dagger_memory_environment '+json.dumps(_memory_snapshot(),sort_keys=True),flush=True)
    seed = int(settings.get("seed", 20260816))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tracks = configured_tracks(settings)
    route_gate_count = int(settings.get("route_gate_count", 3))
    if bool(settings.get("dagger_require_matched_evaluation_speed", False)):
        if not bool(settings.get("dagger_condition_on_teacher_speed", False)):
            raise ValueError(
                "matched DAgger evaluation requires actor speed conditioning"
            )
        if not bool(settings.get("evaluation_condition_on_manifest_speed", False)):
            raise ValueError(
                "matched DAgger evaluation requires manifest speed conditioning"
            )
        if configured_ppo_manifest_speed_field(settings) != "qualified_speed_mps":
            raise ValueError(
                "matched DAgger evaluation must use qualified_speed_mps"
            )
    action_chunk_weight = float(settings.get("action_chunk_weight", 0.0))
    action_chunk_offsets = tuple(int(item) for item in settings.get(
        "action_chunk_offsets", ()
    )) if action_chunk_weight > 0.0 else ()
    if action_chunk_weight > 0.0 and (
        not action_chunk_offsets or action_chunk_offsets[0] != 0
    ):
        raise ValueError(
            "receding-horizon DAgger requires action_chunk_offsets beginning at zero"
        )
    action_chunk_source = str(settings.get(
        "action_chunk_target_source", "mpcc_open_loop"
    ))
    if action_chunk_source not in {"mpcc_open_loop", "replanned_teacher", "coherent_teacher"}:
        raise ValueError(
            "action_chunk_target_source must be mpcc_open_loop or replanned_teacher"
        )
    if (
        action_chunk_weight > 0.0
        and action_chunk_source in {"replanned_teacher", "coherent_teacher"}
        and str(settings.get("collector_backend", "process")) != "process"
    ):
        raise ValueError(
            "replanned teacher action chunks require the process collector"
        )
    gate_contrastive_weight = float(settings.get("gate_contrastive_weight", 0.0))
    if gate_contrastive_weight < 0.0:
        raise ValueError("gate_contrastive_weight cannot be negative")
    reward_aux_weight = float(settings.get("reward_aux_weight", 0.0))
    reward_aux_dim = int(settings.get("reward_aux_dim", 0))
    if reward_aux_weight < 0.0 or reward_aux_dim < 0:
        raise ValueError("reward auxiliary weight and dimension cannot be negative")
    if (reward_aux_weight > 0.0) != (reward_aux_dim > 0):
        raise ValueError(
            "reward_aux_weight and reward_aux_dim must be enabled together"
        )
    if reward_aux_dim not in {0, len(PPO_REWARD_COMPONENTS)}:
        raise ValueError(
            "DAgger reward auxiliary must predict the complete RL reward vector"
        )
    if reward_aux_dim and str(settings.get("collector_backend", "process")) != "process":
        raise ValueError("reward auxiliary DAgger requires the process collector")
    if reward_aux_dim and str(settings.get("action_head_type", "direct_mlp")) == "shortcut_flow":
        raise ValueError(
            "reward auxiliary DAgger is only qualified for the direct action head"
        )
    resume_checkpoint = settings.get("resume_checkpoint")
    initial_checkpoint = resume_checkpoint or settings.get("initial_checkpoint")
    statistics_checkpoint = settings.get("dagger_initialization_stats_checkpoint")
    if (
        statistics_checkpoint is not None
        and initial_checkpoint is not None
        and resume_checkpoint is None
    ):
        raise ValueError(
            "dagger_initialization_stats_checkpoint is only valid for scratch DAgger"
        )
    offline_fraction = 1.0 - float(settings.get("online_fraction", 0.55)) - float(
        settings.get("dagger_permanent_expert_fraction", 0.0)
    )
    if reward_aux_dim and abs(offline_fraction) > 1.0e-6:
        raise ValueError(
            "reward auxiliary DAgger requires fully online/permanent replay"
        )
    if statistics_checkpoint is not None and abs(offline_fraction) > 1.0e-6:
        raise ValueError(
            "statistics-only scratch DAgger requires online_fraction plus "
            "dagger_permanent_expert_fraction to equal one"
        )
    data = None
    train_indices = np.empty(0, np.int64)
    # A warm-start checkpoint already carries the feature normalizer and
    # dynamics-target units.  When online + permanent-expert replay fills the
    # complete batch, loading a legacy offline corpus is both unnecessary and
    # scientifically harmful for task-isolation experiments: its rows can
    # neither be sampled nor should its track manifest be required to exist.
    # Scratch initialization still needs either statistics_checkpoint or
    # offline data to establish those units.
    requires_offline_data = bool(
        statistics_checkpoint is None
        and (initial_checkpoint is None or abs(offline_fraction) > 1.0e-6)
    )
    if requires_offline_data:
        offline_manifest = settings.get("offline_track_manifest")
        offline_tracks = (
            manifest_track_paths(
                str(offline_manifest),
                split=str(settings.get("offline_track_split", "train")),
                qualified_only=bool(settings.get("offline_qualified_tracks_only", True)),
                dynamic_qualification_mode=settings.get(
                    "offline_dynamic_qualification_mode"
                ),
            )
            if offline_manifest is not None else tracks
        )
        data = load_one_step_data(
            settings["data"], track=offline_tracks, route_gates=route_gate_count,
            action_chunk_offsets=action_chunk_offsets,
            observation_contract=str(settings.get(
                "observation_contract", LEGACY_OBSERVATION_CONTRACT
            )),
        )
        remap_offline_action_contract(data, settings)
        train_indices, _ = episode_split(
            data, float(settings.get("validation_fraction", 0.12)), seed
        )
    resume_payload: dict[str, Any] | None = None
    requested_context = settings.get("context_steps_override")
    if initial_checkpoint:
        policy, normalizer, initial, initial_path = load_policy_checkpoint(
            initial_checkpoint, device,
            context_steps=(None if requested_context is None else int(requested_context)),
        )
        from starscream.dagger_contracts import validate_previous_action_mapping
        validate_previous_action_mapping(initial, settings)
        if resume_checkpoint:
            resume_payload = initial
            if str(initial.get("stage")) != "dagger":
                raise ValueError("DAgger resume_checkpoint is not a DAgger checkpoint")
            if int(initial.get("round", 0)) <= 0:
                raise ValueError("DAgger resume_checkpoint has no completed round")
        dynamics_mean = np.asarray(initial["dynamics_target_mean"], np.float32)
        dynamics_std = np.asarray(initial["dynamics_target_std"], np.float32)
    else:
        # Scratch DAgger still uses immutable offline transitions to define the
        # feature and auxiliary-target units, but receives no BC-trained weights.
        # A statistics-only checkpoint preserves those units without retaining
        # or sampling a stale offline corpus when all update rows come from the
        # current run's online and permanent-expert replay pools.
        if statistics_checkpoint is not None:
            statistics_path = resolve_ranked_checkpoint(statistics_checkpoint)
            statistics_payload = torch.load(
                statistics_path, map_location="cpu", weights_only=False
            )
            from starscream.dagger_contracts import validate_previous_action_mapping
            validate_previous_action_mapping(statistics_payload, settings)
            statistics_contract = statistics_payload.get("observation_contract",
                statistics_payload.get('model_config', {}).get('observation_contract'))
            configured_observation_contract = str(settings.get(
                "observation_contract", LEGACY_OBSERVATION_CONTRACT
            ))
            if (
                statistics_contract is not None
                and str(statistics_contract) != configured_observation_contract
            ):
                raise ValueError(
                    "DAgger statistics observation contract mismatch: "
                    f"statistics={statistics_contract} "
                    f"config={configured_observation_contract}"
                )
            statistics_route_gates = statistics_payload.get("route_gate_count")
            if (
                statistics_route_gates is not None
                and int(statistics_route_gates) != route_gate_count
            ):
                raise ValueError(
                    "DAgger statistics route width mismatch: "
                    f"statistics={statistics_route_gates} "
                    f"config={route_gate_count}"
                )
            normalizer = FeatureNormalizer.from_state_dict(
                statistics_payload["normalizer"]
            )
            dynamics_mean = np.asarray(
                statistics_payload["dynamics_target_mean"], np.float32
            )
            dynamics_std = np.asarray(
                statistics_payload["dynamics_target_std"], np.float32
            )
        else:
            assert data is not None
            normalizer = FeatureNormalizer.fit(data.features[train_indices])
        model_config = dict(settings.get("model", config.get("bc", {}).get("model", {})))
        observation_contract = str(settings.get(
            "observation_contract", LEGACY_OBSERVATION_CONTRACT
        ))
        expected_input_dim = privileged_feature_dim(
            route_gate_count,
            observation_contract,
        )
        configured_input_dim = int(model_config.get("input_dim", expected_input_dim))
        if configured_input_dim != expected_input_dim:
            raise ValueError(
                f"route_gate_count={route_gate_count} requires model input_dim="
                f"{expected_input_dim}, received {configured_input_dim}"
            )
        model_config["input_dim"] = expected_input_dim
        model_config["reward_aux_dim"] = reward_aux_dim
        model_config.setdefault("observation_contract", observation_contract)
        if str(model_config["observation_contract"]) != observation_contract:
            raise ValueError("DAgger model and data observation contracts differ")
        policy = initialize_scratch_policy(
            model_config, device,
            context_steps=(
                None if requested_context is None else int(requested_context)
            ),
        )
        if statistics_checkpoint is None:
            assert data is not None
            valid_dynamics = data.next_task_deltas[
                train_indices[data.dynamics_valid[train_indices]]
            ]
            if not len(valid_dynamics):
                raise ValueError("scratch DAgger requires valid offline dynamics targets")
            dynamics_mean = valid_dynamics.mean(0).astype(np.float32)
            dynamics_std = np.maximum(valid_dynamics.std(0), 1.0e-4).astype(np.float32)
        if normalizer.mean.shape != (expected_input_dim,):
            raise ValueError(
                "DAgger initialization statistics feature width mismatch: "
                f"expected={expected_input_dim} received={normalizer.mean.shape}"
            )
        initial_path = Path("scratch-random-initialization")
    if policy.reward_aux_dim != reward_aux_dim:
        raise ValueError(
            "DAgger policy reward auxiliary contract differs from configuration"
        )
    if gate_contrastive_weight > 0.0 and policy.gate_contrastive_dim <= 0:
        raise ValueError(
            "gate contrastive objective requires model.gate_contrastive_dim > 0"
        )
    configured_contract = str(settings.get(
        "observation_contract", LEGACY_OBSERVATION_CONTRACT
    ))
    if policy.observation_contract != configured_contract:
        raise ValueError(
            "DAgger checkpoint observation contract mismatch: "
            f"checkpoint={policy.observation_contract} config={configured_contract}"
        )
    configure_dagger_action_head(policy, settings, action_chunk_offsets)
    policy.bind_feature_normalizer(normalizer)
    if action_chunk_weight > 0.0 and (
        policy.action_chunk_steps != len(action_chunk_offsets)
    ):
        raise ValueError(
            f"checkpoint action_chunk_steps={policy.action_chunk_steps} does not "
            f"match configured offsets={action_chunk_offsets}"
        )
    dagger_speed_command = settings.get("dagger_speed_command")
    if dagger_speed_command is not None:
        if not policy.speed_conditioning:
            raise ValueError(
                "dagger_speed_command requires a speed-conditioned checkpoint"
            )
        command = float(dagger_speed_command)
        if command <= 0:
            raise ValueError("dagger_speed_command must be positive")
        policy.default_speed_command = command
    topology_weight = float(settings.get("topology_weight", 0.0))
    topology_target_dim = mpcc_topology_target_dim(settings)
    topology_replay_dim = topology_target_dim if topology_weight > 0.0 else 0
    if topology_weight > 0.0:
        policy.enable_topology_conditioning(topology_target_dim)
    dynamics_target_contract = str(settings.get(
        "dynamics_target_contract", "task_delta_v1"
    ))
    dynamics_replay_dim = mpcc_invariant_dynamics_target_dim(settings)
    if dynamics_target_contract == "mpcc_body_delta_v2":
        policy.enable_action_conditioned_dynamics(dynamics_replay_dim)
    anchor_policy: PrivilegedMLPPolicy | None = None
    if float(settings.get("dagger_anchor_distillation_weight", 0.0)) > 0.0:
        anchor_checkpoint = settings.get(
            "dagger_anchor_checkpoint", settings.get("initial_checkpoint")
        )
        if not anchor_checkpoint:
            raise ValueError("anchor distillation requires a checkpoint")
        anchor_policy, anchor_normalizer, _, _ = load_policy_checkpoint(
            anchor_checkpoint, device
        )
        if not (
            np.allclose(anchor_normalizer.mean, normalizer.mean)
            and np.allclose(anchor_normalizer.std, normalizer.std)
        ):
            raise ValueError("anchor and learner feature normalizers differ")
        anchor_policy.eval().requires_grad_(False)
    normalized = None if data is None else normalizer.numpy(data.features)
    offline_dynamics = None
    offline_dynamics_valid = None
    if data is not None:
        if dynamics_target_contract == "task_delta_v1":
            offline_dynamics = (
                (data.next_task_deltas - dynamics_mean) / dynamics_std
            ).astype(np.float32)
            offline_dynamics_valid = np.asarray(data.dynamics_valid, np.bool_)
        else:
            # Clean offline expert transitions provide the same body-local forward
            # target as online MPCC labels without the old active-gate discontinuity.
            offline_dynamics = observed_invariant_dynamics_targets(
                data.states, data.next_states, settings
            )
            offline_dynamics_valid = np.ones(len(data.features), np.bool_)
    base_learning_rate = float(settings.get("learning_rate", 1.0e-4))
    optimizer_parameters: Any = policy.parameters()
    if policy.action_head_type == "shortcut_flow":
        assert policy.flow_action_head is not None
        flow_parameters = list(policy.flow_action_head.parameters())
        flow_parameter_ids = {id(item) for item in flow_parameters}
        backbone_parameters = [
            item for item in policy.parameters()
            if id(item) not in flow_parameter_ids
        ]
        optimizer_parameters = [
            {
                "params": backbone_parameters,
                "lr": base_learning_rate * float(settings.get(
                    "flow_backbone_learning_rate_scale",
                    1.0 if policy.flow_context_mode == 'unified_route_dit' else 0.25
                )),
            },
            {"params": flow_parameters, "lr": base_learning_rate},
        ]
    from starscream.training_acceleration import configure_policy_acceleration
    configure_policy_acceleration(policy, settings)
    dagger_loss = imitation_loss
    if settings.get('compile_dagger_loss', False):
        if not device.startswith('cuda') or policy.action_head_type != 'mlp':
            raise ValueError('compiled DAgger loss currently requires a CUDA MLP action head')
        if not settings.get('dagger_static_group_statistics', False) or not settings.get('dagger_fast_group_statistics', False):
            raise ValueError('compiled DAgger loss requires static fast group statistics')
        import torch._inductor.config as inductor_config
        inductor_config.compile_threads = int(settings.get('compile_threads', 1))
        dagger_loss = torch.compile(imitation_loss, fullgraph=True, dynamic=False,
                                    options={'triton.cudagraphs': False})
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=base_learning_rate,
        weight_decay=float(settings.get("weight_decay", 1.0e-5)),
        fused=device.startswith("cuda"),
    )
    if resume_payload is not None:
        restore_optimizer = bool(settings.get(
            "dagger_resume_optimizer_state", True
        ))
        if restore_optimizer:
            if not isinstance(resume_payload.get("optimizer"), Mapping):
                raise ValueError("DAgger resume checkpoint is missing optimizer state")
            optimizer.load_state_dict(resume_payload["optimizer"])
        else:
            print(
                "dagger_optimizer_reset reason=structural_objective_upgrade",
                flush=True,
            )
        resume_learning_rate = settings.get("resume_optimizer_learning_rate")
        if resume_learning_rate is not None:
            resume_learning_rate = float(resume_learning_rate)
            if not np.isfinite(resume_learning_rate) or resume_learning_rate <= 0.0:
                raise ValueError(
                    "resume_optimizer_learning_rate must be finite and positive"
                )
            for group in optimizer.param_groups:
                group["lr"] = resume_learning_rate
    manager = CheckpointManager.from_config(config)
    logger = init_wandb(config)
    manifest_path = (
        Path(str(settings["track_manifest"]))
        if settings.get("track_manifest") is not None else None
    )
    manifest_hash = (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if manifest_path is not None and manifest_path.is_file() else None
    )
    family_by_track = (
        manifest_track_values(manifest_path, "family", cast=str)
        if manifest_path is not None else {
            str(Path(track).resolve()): Path(track).stem for track in tracks
        }
    )
    resolved_tracks = tuple(str(Path(track).resolve()) for track in tracks)
    missing_track_families = set(resolved_tracks) - set(family_by_track)
    if missing_track_families:
        raise ValueError(
            f"DAgger replay contract lacks track families: {missing_track_families}"
        )
    family_names = tuple(sorted({family_by_track[track] for track in resolved_tracks}))
    family_to_id = {family: index for index, family in enumerate(family_names)}
    track_to_id = {track: index for index, track in enumerate(resolved_tracks)}
    track_family_ids = np.asarray([
        family_to_id[family_by_track[track]] for track in resolved_tracks
    ], np.int16)
    dynamic_sampler_config = settings.get("dagger_dynamic_sampling", {})
    dynamic_sampler_enabled = bool(
        isinstance(dynamic_sampler_config, Mapping)
        and dynamic_sampler_config.get("enabled", False)
    )
    base_family_weights = {
        str(name): float(weight)
        for name, weight in dict(settings.get(
            "track_sampling_family_weights", {}
        )).items()
    }
    source_by_family: dict[str, str] = {}
    if dynamic_sampler_enabled:
        if not bool(settings.get("dagger_family_total_balanced_sampling", False)):
            raise ValueError(
                "dynamic DAgger sampling requires family-total balanced collection"
            )
        if not bool(settings.get("dagger_hierarchical_replay_sampling", False)):
            raise ValueError(
                "dynamic DAgger sampling requires hierarchical replay"
            )
        if not base_family_weights or set(base_family_weights) != set(family_names):
            raise ValueError(
                "dynamic DAgger base weights must cover every replay family exactly"
            )
        assert manifest_path is not None
        source_by_track = manifest_track_values(
            manifest_path, "source_family", cast=str,
        )
        for track in resolved_tracks:
            family = family_by_track[track]
            source = source_by_track.get(track)
            if source is None:
                raise ValueError(f"dynamic DAgger source family missing for {track}")
            if family in source_by_family and source_by_family[family] != source:
                raise ValueError(
                    f"replay family {family} spans multiple source families"
                )
            source_by_family[family] = source
    from starscream.dagger_admission import admission_schedule, admitted_tracks
    fixed_admission = admission_schedule(settings, tracks)
    trajectory_quality = quality_config(settings)
    if settings.get('dagger_online_minibatch_quality') is not None:
        if not trajectory_quality:
            raise ValueError('online minibatch quality override requires qualified replay')
        online_minibatch_quality_config(trajectory_quality, settings['dagger_online_minibatch_quality'])
    collection_control = collection_config(settings)
    quality_width = QUALITY_WIDTH if trajectory_quality or collection_control else 0
    if trajectory_quality and trajectory_quality.get('audit_only', False):
        raise ValueError('audit-only trajectory telemetry must never be used for training')
    replay_contract = {
        "observation_contract": configured_contract,
        "route_gate_count": route_gate_count,
        "context_steps": int(policy.context_steps),
        "input_dim": int(policy.input_dim),
        "action_chunk_offsets": list(action_chunk_offsets),
        "topology_replay_dim": int(topology_replay_dim),
        "topology_validity_contract": {
            "require_solver_success": bool(settings.get(
                "topology_require_solver_success", False
            )),
            "max_position": float(settings.get(
                "topology_max_normalized_position", 3.0
            )),
            "max_velocity": float(settings.get(
                "topology_max_normalized_velocity", 2.5
            )),
            "max_action": float(settings.get(
                "topology_max_normalized_action", 1.001
            )),
            "max_progress": float(settings.get(
                "topology_max_normalized_progress", 2.0
            )),
        },
        "dynamics_replay_dim": int(dynamics_replay_dim),
        "dynamics_target_contract": dynamics_target_contract,
        "action_contract": action_contract_metadata(settings),
        "tracks": list(resolved_tracks),
        "track_families": track_family_ids.tolist(),
        "family_names": list(family_names),
        "track_manifest_sha256": manifest_hash,
    }
    if resume_payload is None or 'normalization' in resume_payload.get('dagger_replay', {}).get('contract', {}):
        replay_contract['previous_action_feature_mapping'] = settings.get('previous_action_feature_mapping', 'legacy_linear')
        replay_contract['normalization'] = {
            'feature_mean': normalizer.mean.tolist(), 'feature_std': normalizer.std.tolist(),
            'dynamics_mean': dynamics_mean.tolist(), 'dynamics_std': dynamics_std.tolist(),
        }
    if fixed_admission:
        replay_contract["course_admission"] = [
            {"start_round": start, "tracks": [str(Path(p).resolve()) for p in selected]}
            for start, selected in fixed_admission
        ]
    if collection_control:
        replay_contract['collection_control_v1'] = collection_control
        replay_contract['trajectory_fields'] = QUALITY_FIELDS
    if trajectory_quality:
        replay_contract['trajectory_quality_v1'] = trajectory_quality
        replay_contract['trajectory_fields'] = QUALITY_FIELDS
    if configured_control_hz(settings) != DEFAULT_CONTROL_HZ:
        replay_contract["control_hz"] = configured_control_hz(settings)
        replay_contract["control_dt"] = configured_control_dt(settings)
    if action_chunk_weight > 0.0:
        replay_contract["action_chunk_target_source"] = action_chunk_source
    # Preserve the fingerprint of older reward-free DAgger replay stores.  The
    # auxiliary contract is material only when its labels are actually present.
    if reward_aux_dim:
        replay_contract["reward_aux_dim"] = reward_aux_dim
        replay_contract["reward_component_names"] = list(PPO_REWARD_COMPONENTS)
    replay_store = DaggerReplayStore.create(
        manager.checkpoint_dir / "dagger-replay",
        replay_contract,
        resume_metadata=(
            None if resume_payload is None else resume_payload.get("dagger_replay")
        ),
        require_resume_state=bool(
            resume_payload is not None
            and settings.get("dagger_resume_require_replay_state", False)
        ),
    )
    if resume_payload is not None:
        restore_rng_state(resume_payload.get("rng_state"))
        if resume_payload.get("numpy_rng_state") is not None:
            rng.bit_generator.state = copy.deepcopy(
                resume_payload["numpy_rng_state"]
            )
    stage = parse_stage(settings["curriculum"])
    evaluation_stage = parse_stage(
        settings.get("evaluation_curriculum", settings["curriculum"])
    )
    reporting_stage = (
        parse_stage(settings["reporting_evaluation_curriculum"])
        if settings.get("reporting_evaluation_curriculum") is not None else None
    )

    def run_reporting_evaluation(
        step: int, *, round_index: int = 0,
    ) -> dict[str, float] | None:
        if reporting_stage is None:
            return None
        interval = int(settings.get("reporting_evaluation_interval", 1))
        final_round = int(settings.get("rounds", 8))
        if (
            round_index > 0 and round_index != final_round
            and (interval <= 0 or round_index % interval != 0)
        ):
            return None
        reporting_settings = dict(settings)
        if bool(settings.get("reporting_nominal_environment", False)):
            reporting_settings["dynamics_randomization"] = {"enabled": False}
            reporting_settings.pop("state_estimator_randomization", None)
            reporting_settings.pop("flight_plan_randomization", None)
            reporting_settings["policy_state_source"] = "truth"
            reporting_settings.pop("action_delay_range", None)
            reporting_settings["action_delay"] = float(
                settings.get("reporting_action_delay", 0.011)
            )
        metrics = evaluate_policy(
            policy, normalizer, reporting_settings, reporting_stage,
            count=int(settings.get("reporting_evaluation_episodes", 24)),
            seed_base=int(settings.get("reporting_evaluation_seed", 20261000)),
            device=device,
        )
        logger.log_eval({
            f"reporting/{reporting_stage.name}/{key}": value
            for key, value in metrics.items()
        }, step)
        print(
            f"reporting_eval stage={reporting_stage.name} step={step} "
            f"full={metrics['full_course_success']:.3f} "
            f"mean_gates={metrics['mean_gates']:.2f} "
            f"crash={metrics['crash_rate']:.3f}",
            flush=True,
        )
        return metrics

    def run_frontier_evaluation(
        step: int, *, round_index: int = 0,
    ) -> dict[str, float] | None:
        fractions = tuple(float(item) for item in settings.get(
            "frontier_evaluation_speed_fractions", ()
        ))
        if not fractions:
            return None
        interval = int(settings.get("frontier_evaluation_interval", 2))
        final_round = int(settings.get("rounds", 8))
        if (
            round_index > 0 and round_index != final_round
            and (interval <= 0 or round_index % interval != 0)
        ):
            return None
        summaries: list[dict[str, float]] = []
        for fraction in fractions:
            if not 0.0 < fraction <= 1.0:
                raise ValueError("frontier evaluation fractions must be in (0,1]")
            evaluation_settings = dict(settings)
            evaluation_settings["evaluation_manifest_speed_scale"] = fraction
            metrics = evaluate_policy(
                policy, normalizer, evaluation_settings, evaluation_stage,
                count=int(settings.get("frontier_evaluation_episodes", 60)),
                seed_base=(
                    int(settings.get("frontier_evaluation_seed", 20261100))
                    + int(round(1000.0 * fraction))
                ),
                device=device,
            )
            summaries.append(metrics)
            logger.log_eval({
                f"frontier/{fraction:.3f}/{key}": value
                for key, value in metrics.items()
            }, step)
            print(
                f"frontier_eval fraction={fraction:.3f} step={step} "
                f"full={metrics['full_course_success']:.3f} "
                f"mean_gates={metrics['mean_gates']:.2f} "
                f"crash={metrics['crash_rate']:.3f}",
                flush=True,
            )
        summary = {
            "frontier_full_course_mean": float(np.mean([
                item["full_course_success"] for item in summaries
            ])),
            "frontier_full_course_minimum": float(np.min([
                item["full_course_success"] for item in summaries
            ])),
            "frontier_mean_gates": float(np.mean([
                item["mean_gates"] for item in summaries
            ])),
        }
        logger.log_eval(summary, step)
        return summary

    if resume_payload is None:
        baseline = evaluate_policy(
            policy, normalizer, settings, evaluation_stage,
            count=int(settings.get("evaluation_episodes", 96)),
            seed_base=int(settings.get("evaluation_seed", 20260900)), device=device,
        )
        if bool(settings.get("dagger_completion_survival_selection", False)):
            baseline = dagger_completion_survival_metrics(baseline, settings)
        elif bool(settings.get("pace_aware_selection", False)):
            baseline = refinement_metrics(baseline, evaluation_stage, settings)
        logger.log_eval(baseline, 0)
        baseline_reporting = run_reporting_evaluation(0)
        run_frontier_evaluation(0)
        if settings.get('midtrain_pace_probe'):
            from starscream.midtrain_evaluation import evaluate_midtraining
            evaluate_midtraining(policy, normalizer, settings, evaluation_stage, device,
                                 logger, manager.checkpoint_dir, 0, 0)
        completed_round = 0
        environment_steps = 0
    else:
        baseline = copy.deepcopy(dict(resume_payload.get("metrics", {})))
        if "selection_score" not in baseline:
            raise ValueError("DAgger resume checkpoint is missing evaluation metrics")
        baseline_reporting = copy.deepcopy(
            resume_payload.get("reporting_evaluation")
        )
        completed_round = int(resume_payload["round"])
        environment_steps = int(resume_payload.get(
            "environment_steps", resume_payload.get("step", 0)
        ))
    if settings.get('timed_reporting'):
        from starscream.timed_teacher_reporting import report_teacher
        report_teacher(policy, normalizer, settings, evaluation_stage, device, logger,
                       manager.checkpoint_dir, environment_steps, completed_round)
    restore_safe_state = bool(settings.get(
        "dagger_resume_safe_state", True
    ))
    restored_safe = (
        None if resume_payload is None or not restore_safe_state
        else resume_payload.get("dagger_safe_state")
    )
    if isinstance(restored_safe, Mapping):
        safe_policy_state = copy.deepcopy(dict(restored_safe["policy"]))
        safe_optimizer_state = copy.deepcopy(restored_safe["optimizer"])
        safe_evaluation = copy.deepcopy(dict(restored_safe["evaluation"]))
        safe_selection_score = float(safe_evaluation["selection_score"])
        safe_reporting = copy.deepcopy(restored_safe.get("reporting_evaluation"))
    else:
        safe_policy_state = {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        }
        safe_optimizer_state = copy.deepcopy(optimizer.state_dict())
        safe_selection_score = float(baseline["selection_score"])
        safe_evaluation = copy.deepcopy(baseline)
        safe_reporting = copy.deepcopy(baseline_reporting)
    dynamic_sampler_state: dict[str, Any] | None = None
    dynamic_family_weights = dict(base_family_weights)
    dynamic_gate_weights: dict[str, dict[int, float]] = {}
    if dynamic_sampler_enabled:
        restored_sampler = (
            None if resume_payload is None
            else resume_payload.get("dagger_dynamic_sampler_state")
        )
        if isinstance(restored_sampler, Mapping):
            if dynamic_sampler_config.get("behavior_map") and restored_sampler.get(
                "behavior_map_sha256"
            ) != dynamic_sampler_config["behavior_map_sha256"]:
                raise ValueError("Cannot resume sampler from a different behavior map")
            dynamic_sampler_state = copy.deepcopy(dict(restored_sampler))
            dynamic_family_weights = {
                str(name): float(value)
                for name, value in dict(
                    dynamic_sampler_state.get("family_weights", base_family_weights)
                ).items()
            }
            dynamic_gate_weights = {
                str(family): {
                    int(gate): float(weight)
                    for gate, weight in dict(weights).items()
                }
                for family, weights in dict(
                    dynamic_sampler_state.get("gate_weights", {})
                ).items()
            }
        else:
            assert manifest_path is not None
            observed_competence, observed_frontiers = (
                sampler_feedback(
                    baseline, settings, manifest_path,
                    route_horizon=int(dynamic_sampler_config.get(
                        "route_horizon", route_gate_count
                    )),
                )
            )
            dynamic_sampler_state, dynamic_family_weights, dynamic_gate_weights = (
                update_dynamic_dagger_sampler(
                    base_family_weights=base_family_weights,
                    source_by_family=source_by_family,
                    observed_competence=observed_competence,
                    observed_frontiers=observed_frontiers,
                    observed_losses={}, previous_state=None,
                    config=dynamic_sampler_config,
                )
            )
    task_bank_config = settings.get("dagger_learning_progress_task_bank", {})
    task_bank_enabled = bool(
        isinstance(task_bank_config, Mapping)
        and task_bank_config.get("enabled", False)
    )
    if task_bank_enabled and dynamic_sampler_enabled:
        raise ValueError(
            "DAgger learning-progress task bank and dynamic sampler are mutually exclusive"
        )
    task_bank: DaggerLearningProgressBank | None = None
    task_bank_name_by_track: dict[str, str] = {}
    if task_bank_enabled:
        preserve_task_groups = bool(task_bank_config.get(
            "preserve_manifest_families", False
        ))
        task_bank = DaggerLearningProgressBank(
            resolved_tracks,
            active_tasks=int(task_bank_config.get(
                "active_tasks", min(32, len(resolved_tracks))
            )),
            group_by_task=(
                {
                    track: family_by_track[track]
                    for track in resolved_tracks
                }
                if preserve_task_groups else None
            ),
            window=int(task_bank_config.get("window", 8)),
            alpha=float(task_bank_config.get("alpha", 1000.0)),
            seed=seed ^ 0xDA66E2,
        )
        if resume_payload is not None:
            task_bank.load_state_dict(
                resume_payload.get("dagger_learning_progress_task_bank_state")
            )
        task_bank_name_by_track = {
            track: load_track(track).name for track in resolved_tracks
        }
    # A continuation must never lose the incoming policy if every DAgger round
    # is worse. Rank the untouched baseline alongside the learned candidates.
    if resume_payload is None:
        manager.save(
            checkpoint_payload(
                policy, normalizer, stage="dagger", track=",".join(tracks),
                optimizer=optimizer,
                extra={
                    "dynamics_target_mean": dynamics_mean,
                    "dynamics_target_std": dynamics_std,
                    "round": 0,
                    "environment_steps": 0,
                    "initial_checkpoint": str(initial_path),
                    "reporting_evaluation": baseline_reporting,
                    "action_contract": action_contract_metadata(settings),
                    "training_config": config,
                    "dagger_replay": replay_store.checkpoint_metadata(0),
                    "dagger_dynamic_sampler_state": dynamic_sampler_state,
                    "dagger_learning_progress_task_bank_state": (
                        None if task_bank is None else task_bank.state_dict()
                    ),
                    "rng_state": capture_rng_state(),
                    "numpy_rng_state": copy.deepcopy(rng.bit_generator.state),
                    "dagger_safe_state": {
                        "policy": safe_policy_state,
                        "optimizer": safe_optimizer_state,
                        "evaluation": safe_evaluation,
                        "reporting_evaluation": safe_reporting,
                    },
                },
            ),
            step=0, metrics=baseline,
        )
        print(
            f"privileged_dagger actor={initial_path} baseline="
            f"p1={baseline.get('p1',0):.3f}/p2={baseline.get('p2',0):.3f}/"
            f"p3={baseline.get('p3',0):.3f}/full={baseline['full_course_success']:.3f} "
            f"crash={baseline['crash_rate']:.3f}", flush=True,
        )
    else:
        # Seed a continuation's new checkpoint namespace with the untouched
        # incoming policy. This makes a completion- or reliability-ranked
        # continuation non-destructive even if every new round is worse.
        manager.save(
            checkpoint_payload(
                policy, normalizer, stage="dagger", track=",".join(tracks),
                optimizer=optimizer,
                extra={
                    "dynamics_target_mean": dynamics_mean,
                    "dynamics_target_std": dynamics_std,
                    "round": completed_round,
                    "dagger_async_pending": resume_payload.get('dagger_async_pending'),
                    "environment_steps": environment_steps,
                    "initial_checkpoint": str(initial_path),
                    "reporting_evaluation": baseline_reporting,
                    "action_contract": action_contract_metadata(settings),
                    "training_config": config,
                    "dagger_replay": replay_store.checkpoint_metadata(
                        completed_round
                    ),
                    "dagger_dynamic_sampler_state": dynamic_sampler_state,
                    "dagger_learning_progress_task_bank_state": (
                        None if task_bank is None else task_bank.state_dict()
                    ),
                    "rng_state": capture_rng_state(),
                    "numpy_rng_state": copy.deepcopy(rng.bit_generator.state),
                    "dagger_safe_state": {
                        "policy": safe_policy_state,
                        "optimizer": safe_optimizer_state,
                        "evaluation": safe_evaluation,
                        "reporting_evaluation": safe_reporting,
                    },
                },
            ),
            step=environment_steps, metrics=baseline,
        )
        print(
            f"privileged_dagger_resume checkpoint={initial_path} "
            f"completed_round={completed_round} steps={environment_steps} "
            f"full={baseline['full_course_success']:.3f} "
            f"crash={baseline['crash_rate']:.3f} "
            f"optimizer={'restored' if bool(settings.get('dagger_resume_optimizer_state', True)) else 'reset'} ",
            flush=True,
        )
    # Start fork-backed collection workers before materializing the persistent
    # replay.  On a late-run resume the replay is several GiB; forking after
    # restoration makes every worker inherit that address space and ordinary
    # Python bookkeeping eventually turns shared pages into enough private RSS
    # to trip the host OOM killer.  Workers do not consume replay arrays, so
    # creating them here is lifecycle-only and leaves collection semantics
    # unchanged.
    collector_type = (
        ProcessDaggerCollector
        if str(settings.get("collector_backend", "process")) == "process"
        else ParallelDaggerCollector
    )
    async_pipeline = bool(settings.get('dagger_async_pipeline', False))
    if async_pipeline:
        from starscream.dagger_async import AsyncDaggerCollector
        if task_bank is not None:
            raise ValueError('Async DAgger task-bank admission is not yet supported')
        collector_type = AsyncDaggerCollector
    collector = collector_type(policy, normalizer, settings, stage, device)
    if async_pipeline and resume_payload is not None:
        pending_state = resume_payload.get('dagger_async_pending')
        if pending_state is not None:
            if int(pending_state['window']) != completed_round + 1:
                raise ValueError('Async checkpoint pending window must follow its committed round')
            collector.restore_pending(pending_state)
    # Multiprocessing children are non-daemon. If replay restoration,
    # collection, or optimization raises, do not leave them blocking a restart.
    atexit.register(collector.close)

    replay_history_dtype_name = str(settings.get(
        "dagger_replay_history_dtype", "float32"
    ))
    if replay_history_dtype_name not in {"float16", "float32"}:
        raise ValueError("dagger_replay_history_dtype must be float16 or float32")
    replay_history_dtype = (
        np.float16 if replay_history_dtype_name == "float16" else np.float32
    )
    online_histories = np.empty(
        (0, policy.context_steps, policy.input_dim), replay_history_dtype
    )
    online_actions = np.empty((0, 4), np.float32)
    online_previous = np.empty((0, 4), np.float32)
    online_dynamics = np.empty((0, dynamics_replay_dim), np.float32)
    online_dynamics_valid = np.empty((0,), np.bool_)
    online_tracks = np.empty((0,), np.int16)
    online_families = np.empty((0,), np.int16)
    online_gate_indices = np.empty((0,), np.int16)
    online_teacher_modes = np.empty((0,), np.int8)
    online_occupancy_modes = np.empty((0,), np.int8)
    online_trajectory = np.empty((0, quality_width), np.float64)
    online_progress = np.empty((0,), np.float32)
    online_events = np.empty((0,), np.bool_)
    online_speed_commands = np.empty((0,), np.float32)
    online_action_chunks = np.empty(
        (0, len(action_chunk_offsets), 4), np.float32
    )
    online_executed_actions = np.empty((0, 4), np.float32)
    online_reward_components = np.empty((0, reward_aux_dim), np.float32)
    online_topology_targets = np.empty(
        (0, topology_replay_dim), np.float32
    )
    online_topology_valid = np.empty((0,), np.bool_)
    online_groups = np.empty((0,), np.int16)
    online_anchor_mask = np.empty((0,), np.bool_)
    permanent_histories = np.empty(
        (0, policy.context_steps, policy.input_dim), replay_history_dtype
    )
    permanent_actions = np.empty((0, 4), np.float32)
    permanent_previous = np.empty((0, 4), np.float32)
    permanent_dynamics = np.empty((0, dynamics_replay_dim), np.float32)
    permanent_dynamics_valid = np.empty((0,), np.bool_)
    permanent_tracks = np.empty((0,), np.int16)
    permanent_families = np.empty((0,), np.int16)
    permanent_gate_indices = np.empty((0,), np.int16)
    permanent_teacher_modes = np.empty((0,), np.int8)
    permanent_occupancy_modes = np.empty((0,), np.int8)
    permanent_trajectory = np.empty((0, quality_width), np.float64)
    permanent_progress = np.empty((0,), np.float32)
    permanent_events = np.empty((0,), np.bool_)
    permanent_speed_commands = np.empty((0,), np.float32)
    permanent_action_chunks = np.empty(
        (0, len(action_chunk_offsets), 4), np.float32
    )
    permanent_executed_actions = np.empty((0, 4), np.float32)
    permanent_reward_components = np.empty((0, reward_aux_dim), np.float32)
    permanent_topology_targets = np.empty((0, topology_replay_dim), np.float32)
    permanent_topology_valid = np.empty((0,), np.bool_)
    permanent_groups = np.empty((0,), np.int16)
    permanent_anchor_mask = np.empty((0,), np.bool_)
    online_capacity = int(settings.get("online_replay_capacity", 250000))
    permanent_capacity = int(settings.get(
        "dagger_permanent_expert_capacity", 180000
    ))
    if resume_payload is not None and resume_payload.get("dagger_replay") is not None:
        online_restored = replay_store.restore_pool(
            "online",
            {
                "histories": online_histories, "actions": online_actions,
                "previous": online_previous, "dynamics": online_dynamics,
                "dynamics_valid": online_dynamics_valid,
                "tracks": online_tracks, "families": online_families,
                "gate_indices": online_gate_indices,
                "teacher_modes": online_teacher_modes,
                "occupancy_modes": online_occupancy_modes,
                "trajectory": online_trajectory,
                "progress": online_progress, "events": online_events,
                "speed_commands": online_speed_commands,
                "action_chunks": online_action_chunks,
                "executed_actions": online_executed_actions,
                "reward_components": online_reward_components,
                "topology_targets": online_topology_targets,
                "topology_valid": online_topology_valid,
                "groups": online_groups, "anchor_mask": online_anchor_mask,
            },
            capacity=online_capacity, committed_round=completed_round,
        )
        permanent_restored = replay_store.restore_pool(
            "permanent",
            {
                "histories": permanent_histories, "actions": permanent_actions,
                "previous": permanent_previous, "dynamics": permanent_dynamics,
                "dynamics_valid": permanent_dynamics_valid,
                "tracks": permanent_tracks, "families": permanent_families,
                "gate_indices": permanent_gate_indices,
                "teacher_modes": permanent_teacher_modes,
                "occupancy_modes": permanent_occupancy_modes,
                "trajectory": permanent_trajectory,
                "progress": permanent_progress, "events": permanent_events,
                "speed_commands": permanent_speed_commands,
                "action_chunks": permanent_action_chunks,
                "executed_actions": permanent_executed_actions,
                "reward_components": permanent_reward_components,
                "topology_targets": permanent_topology_targets,
                "topology_valid": permanent_topology_valid,
                "groups": permanent_groups,
                "anchor_mask": permanent_anchor_mask,
            },
            capacity=permanent_capacity, committed_round=completed_round,
        )
        online_histories = online_restored["histories"]
        online_actions = online_restored["actions"]
        online_previous = online_restored["previous"]
        online_dynamics = online_restored["dynamics"]
        online_dynamics_valid = online_restored["dynamics_valid"]
        online_tracks = online_restored["tracks"]
        online_families = online_restored["families"]
        online_gate_indices = online_restored["gate_indices"]
        online_teacher_modes = online_restored["teacher_modes"]
        online_occupancy_modes = online_restored["occupancy_modes"]
        online_trajectory = online_restored["trajectory"]
        online_progress = online_restored["progress"]
        online_events = online_restored["events"]
        online_speed_commands = online_restored["speed_commands"]
        online_action_chunks = online_restored["action_chunks"]
        online_executed_actions = online_restored["executed_actions"]
        online_reward_components = online_restored["reward_components"]
        online_topology_targets = online_restored["topology_targets"]
        online_topology_valid = online_restored["topology_valid"]
        online_groups = online_restored["groups"]
        online_anchor_mask = online_restored["anchor_mask"]
        permanent_histories = permanent_restored["histories"]
        permanent_actions = permanent_restored["actions"]
        permanent_previous = permanent_restored["previous"]
        permanent_dynamics = permanent_restored["dynamics"]
        permanent_dynamics_valid = permanent_restored["dynamics_valid"]
        permanent_tracks = permanent_restored["tracks"]
        permanent_families = permanent_restored["families"]
        permanent_gate_indices = permanent_restored["gate_indices"]
        permanent_teacher_modes = permanent_restored["teacher_modes"]
        permanent_occupancy_modes = permanent_restored["occupancy_modes"]
        permanent_trajectory = permanent_restored["trajectory"]
        permanent_progress = permanent_restored["progress"]
        permanent_events = permanent_restored["events"]
        permanent_speed_commands = permanent_restored["speed_commands"]
        permanent_action_chunks = permanent_restored["action_chunks"]
        permanent_executed_actions = permanent_restored["executed_actions"]
        permanent_reward_components = permanent_restored["reward_components"]
        permanent_topology_targets = permanent_restored["topology_targets"]
        permanent_topology_valid = permanent_restored["topology_valid"]
        permanent_groups = permanent_restored["groups"]
        permanent_anchor_mask = permanent_restored["anchor_mask"]
        print(
            f"dagger_replay_restored online={len(online_histories)} "
            f"permanent={len(permanent_histories)} round={completed_round}",
            flush=True,
        )
    frontier_by_track = (
        manifest_track_values(
            settings["track_manifest"], "qualified_speed_mps", cast=float,
        )
        if settings.get("track_manifest") is not None else {}
    )
    permanent_required_families = {
        str(item) for item in settings.get(
            "dagger_permanent_expert_required_families", ()
        )
    }
    if permanent_required_families:
        family_by_track = manifest_track_values(
            settings["track_manifest"], "family", cast=str
        )
        permanent_coverage_tracks = tuple(
            track for track in tracks
            if family_by_track.get(str(Path(track).resolve()))
            in permanent_required_families
        )
        missing_families = permanent_required_families - {
            family_by_track[str(Path(track).resolve())]
            for track in permanent_coverage_tracks
        }
        if missing_families:
            raise ValueError(
                f"permanent expert families absent from training: {missing_families}"
            )
    else:
        permanent_coverage_tracks = tracks
    amp = bool(settings.get("amp", True)) and device.startswith("cuda")
    resume_expert_refresh_rounds = int(settings.get(
        "dagger_resume_expert_refresh_rounds", 0
    ))
    if resume_expert_refresh_rounds < 0:
        raise ValueError("dagger_resume_expert_refresh_rounds must be non-negative")
    if resume_expert_refresh_rounds and resume_payload is None:
        raise ValueError("expert replay refresh is only valid for DAgger continuation")
    resume_expert_refresh_end = completed_round + resume_expert_refresh_rounds
    evaluation_collector = None
    if (settings.get("dagger_persistent_evaluation", False)
            and str(settings.get("evaluation_backend", "thread")) == "process"):
        evaluation_collector = ProcessRaceCollector(
            policy, normalizer, settings, evaluation_stage, device,
            workers=min(int(settings.get("evaluation_workers", 12)),
                        int(settings.get("evaluation_episodes", 96))),
        )
        atexit.register(evaluation_collector.close)
    for round_index in range(
        completed_round + 1, int(settings.get("rounds", 8)) + 1
    ):
        round_started = time.perf_counter()
        round_memory_before = _memory_snapshot()
        if async_pipeline and collector.pending is None:
            collector.settings.update(settings)
        from starscream.dagger_schedule import recovery_anneal
        recovery_schedule = recovery_anneal(round_index, settings)
        if recovery_schedule:
            for key in ('dagger_replay_nominal_fraction', 'dagger_replay_critical_fraction',
                        'dagger_replay_recovery_fraction'):
                settings[key] = recovery_schedule[key]
            print(f"dagger_recovery_anneal round={round_index} {recovery_schedule}", flush=True)
        from starscream.dagger_schedule import speed_fraction_weights
        scheduled_speed_weights = speed_fraction_weights(round_index, settings)
        if scheduled_speed_weights:
            settings["dagger_teacher_speed_fraction_weights"] = scheduled_speed_weights
            collector.settings["dagger_teacher_speed_fraction_weights"] = dict(scheduled_speed_weights)
            print(f"dagger_speed_weights round={round_index} {scheduled_speed_weights}", flush=True)
        if settings.get("dagger_learning_rate_schedule"):
            from starscream.dagger_schedule import apply_round_learning_rate
            lr_factor = apply_round_learning_rate(
                optimizer, round_index, int(settings.get("rounds", 8)),
                settings.get("dagger_learning_rate_schedule"),
            )
            print(f"dagger_lr round={round_index} factor={lr_factor:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
        pre_round_dynamic_sampler_state = copy.deepcopy(dynamic_sampler_state)
        pre_round_task_bank_state = (
            None if task_bank is None else copy.deepcopy(task_bank.state_dict())
        )
        round_tracks = (
            tracks if task_bank is None else tuple(task_bank.active)
        )
        round_tracks = admitted_tracks(fixed_admission, round_index, round_tracks)
        if fixed_admission:
            print(f"dagger_admission round={round_index} active_courses={len(round_tracks)} "
                  f"final_courses={len(tracks)}", flush=True)
        round_permanent_coverage_tracks = tuple(
            track for track in permanent_coverage_tracks
            if str(Path(track).resolve()) in {
                str(Path(item).resolve()) for item in round_tracks
            }
        )
        dynamic_warmup_rounds = int(
            dynamic_sampler_config.get("warmup_rounds", 4)
        ) if dynamic_sampler_enabled else 0
        dynamic_sampler_active = bool(
            dynamic_sampler_enabled and round_index > dynamic_warmup_rounds
        )
        active_family_weights = (
            dynamic_family_weights if dynamic_sampler_active
            else base_family_weights
        )
        active_replay_family_weights = dict(active_family_weights)
        # Replay responds to the latest evaluation. An already-running
        # collection retains the exact weights frozen when it was dispatched.
        pending_collection_settings = (
            collector.pending['settings']
            if async_pipeline and collector.pending is not None else None
        )
        if pending_collection_settings is not None:
            active_family_weights = pending_collection_settings['track_sampling_family_weights']
        collector.settings["track_sampling_family_weights"] = dict(
            active_family_weights
        )
        collector.settings[
            "dagger_transition_start_gate_weights_by_family"
        ] = (
            copy.deepcopy(dynamic_gate_weights)
            if dynamic_sampler_active else {}
        )
        refresh_permanent_expert = bool(
            resume_payload is not None
            and round_index <= resume_expert_refresh_end
        )
        permanent_rounds = int(settings.get("dagger_permanent_expert_rounds", 0))
        beta = dagger_teacher_beta(
            round_index, settings,
            refresh_permanent_expert=refresh_permanent_expert,
        )
        from starscream.dagger_schedule import dart_collection_round
        dart_active = bool(
            dart_collection_round(round_index, settings)
            or (
                refresh_permanent_expert
                and bool(settings.get("dagger_dart_on_expert_refresh", False))
            )
        )
        if dart_active and beta < 1.0 - 1.0e-9:
            raise ValueError(
                "DART expert collection requires beta=1 so only bounded expert "
                "perturbations, not learner actions, define the recovery occupancy"
            )
        successful_episodes_only = dagger_successful_episode_filter(
            round_index, settings,
            refresh_permanent_expert=refresh_permanent_expert,
        )
        # Episode qualification is a host-side replay decision. Workers still
        # report terminal success, so this may change by round without
        # rebuilding persistent simulator/MPCC processes.
        collector.settings["dagger_require_successful_episodes"] = (
            successful_episodes_only
        )
        collector.settings['dagger_pure_expert_collection'] = bool(recovery_schedule.get('pure_expert'))
        if async_pipeline:
            collector.window_collect_seconds = 0.0
            collector.window_provenance = []
            collector.logical_window = round_index
            if collector.pending is None:
                collector.publish(policy, round_index - 1)
            else:
                if int(collector.pending['window']) != round_index:
                    raise ValueError('Async collection window is out of order')
                # This also keeps coverage retries on the same actor and
                # collection distribution as the primary accepted batch.
                collector.settings.update(pending_collection_settings)
        collection_started = time.perf_counter()
        collection_started_unix = time.time()
        collected = collector.collect(
            episodes=int(settings.get("episodes_per_round", 96)),
            beta=beta, seed_base=seed + round_index * 100000,
            dart=dart_active,
            tracks=round_tracks,
        )
        if round_index <= permanent_rounds or refresh_permanent_expert:
            minimum = int(settings.get(
                "dagger_permanent_expert_minimum_episodes_per_track", 1
            ))
            underfilled = {
                track: collected.accepted_episodes_by_track.get(track, 0)
                for track in round_permanent_coverage_tracks
                if collected.accepted_episodes_by_track.get(track, 0) < minimum
            }
            retry_per_track = int(settings.get(
                "dagger_permanent_expert_retry_episodes_per_track", 0
            ))
            if underfilled and retry_per_track > 0:
                supplemental = collector.collect(
                    episodes=retry_per_track * len(underfilled), beta=1.0,
                    seed_base=seed + round_index * 100000 + 70000,
                    tracks=tuple(underfilled),
                    dart=dart_active,
                )
                merge_dagger_batches(collected, supplemental)
                underfilled = {
                    track: collected.accepted_episodes_by_track.get(track, 0)
                    for track in round_permanent_coverage_tracks
                    if collected.accepted_episodes_by_track.get(track, 0) < minimum
                }
                print(
                    f"dagger_permanent_expert_retry round={round_index} "
                    f"episodes={retry_per_track * len(supplemental.accepted_episodes_by_track)} "
                    f"remaining_tracks={len(underfilled)}", flush=True,
                )
            if underfilled and bool(settings.get(
                "dagger_permanent_expert_abort_on_underfilled", True
            )):
                raise RuntimeError(
                    "permanent expert full-lap coverage failed: "
                    f"required={minimum} underfilled={underfilled}"
                )
        collection_seconds = time.perf_counter() - collection_started
        if successful_episodes_only:
            minimum = int(settings.get(
                "dagger_minimum_successful_episodes_per_track", 1
            ))
            underfilled = {
                track: collected.accepted_episodes_by_track.get(track, 0)
                for track in round_tracks
                if collected.accepted_episodes_by_track.get(track, 0) < minimum
            }
            if underfilled:
                retry_per_track = int(settings.get(
                    "dagger_coverage_retry_episodes_per_track", 0
                ))
                if retry_per_track > 0:
                    supplemental = collector.collect(
                        episodes=retry_per_track * len(underfilled),
                        beta=beta,
                        seed_base=seed + round_index * 100000 + 50000,
                        tracks=tuple(underfilled),
                    )
                    merge_dagger_batches(collected, supplemental)
                    print(
                        f"dagger_coverage_retry round={round_index} "
                        f"tracks={len(underfilled)} episodes="
                        f"{retry_per_track * len(underfilled)} accepted="
                        f"{supplemental.accepted_episodes}", flush=True,
                    )
                    underfilled = {
                        track: collected.accepted_episodes_by_track.get(track, 0)
                        for track in round_tracks
                        if collected.accepted_episodes_by_track.get(track, 0) < minimum
                    }
            if underfilled:
                if bool(settings.get("dagger_abort_on_underfilled_tracks", True)):
                    raise RuntimeError(
                        "DAgger successful-label coverage gate failed: "
                        f"required={minimum} underfilled={underfilled}"
                    )
                print(
                    f"dagger_coverage_underfilled round={round_index} "
                    f"required={minimum} skipped={underfilled}",
                    flush=True,
                )
        if not collected.histories:
            raise RuntimeError("DAgger collected no valid MPCC labels")
        # Include mandated coverage retries in collection wall time as well.
        collection_seconds = time.perf_counter() - collection_started
        collection_wait_seconds = collection_seconds
        if async_pipeline:
            # Blocking wait can be near zero after overlap; retain actual
            # collector wall time for utilization/SPS metrics.
            collection_seconds = max(collection_seconds, float(
                collector.window_collect_seconds))
        replay_started_at = time.perf_counter()
        if collected.histories_normalized:
            new_histories = np.asarray(
                collected.histories, dtype=replay_history_dtype
            )
        else:
            # Compatibility path for external/legacy collectors.  Current
            # collectors return normalized histories directly and avoid this
            # full-round float32 copy.
            new_histories = normalizer.numpy(
                np.asarray(collected.histories, np.float32)
            ).astype(replay_history_dtype)
        collected.histories.clear()
        new_actions = np.asarray(collected.actions, np.float32)
        new_previous = np.asarray(collected.previous_actions, np.float32)
        collected_dynamics = np.asarray(collected.dynamics, np.float32).reshape(
            len(new_histories), dynamics_replay_dim
        )
        new_dynamics = (
            (collected_dynamics - dynamics_mean) / dynamics_std
            if dynamics_target_contract == "task_delta_v1"
            else collected_dynamics
        ).astype(np.float32)
        new_dynamics_valid = np.asarray(collected.dynamics_valid, np.bool_)
        new_successful = np.asarray(collected.successful_labels, np.bool_)
        # Resolve each concrete course once, not once per transition (100k+
        # filesystem path traversals per round previously). Preserve aliases.
        resolved_collected_tracks = {
            item: str(Path(item).resolve())
            for item in set(collected.tracks)
        }
        collected_track_ids = {
            item: track_to_id[resolved] for item, resolved in resolved_collected_tracks.items()
        }
        new_tracks = np.fromiter(
            (collected_track_ids[item] for item in collected.tracks),
            dtype=np.int16, count=len(collected.tracks),
        )
        new_families = track_family_ids[new_tracks]
        new_progress = np.asarray(collected.course_progress, np.float32)
        new_gate_phases = np.asarray(collected.gate_phases, np.int8)
        new_gate_indices = (
            np.asarray(collected.active_gate_indices, np.int16)
            if len(collected.active_gate_indices) == len(new_histories)
            else np.zeros(len(new_histories), np.int16)
        )
        new_teacher_modes = (
            np.asarray(collected.teacher_modes, np.int8)
            if len(collected.teacher_modes) == len(new_histories)
            else np.zeros(len(new_histories), np.int8)
        )
        new_occupancy_modes = (
            np.asarray(collected.occupancy_modes, np.int8)
            if len(collected.occupancy_modes) == len(new_histories)
            else np.zeros(len(new_histories), np.int8)
        )
        new_trajectory = (np.asarray(collected.trajectory, np.float64).reshape(len(new_histories), quality_width)
                          if len(collected.trajectory) == len(new_histories) else np.full((len(new_histories), quality_width), -1., np.float64))
        if trajectory_quality:
            if len(collected.trajectory) != len(new_histories):
                raise RuntimeError('quality collector lost metadata')
            # Async handoff arrays are read-only memory maps.
            new_trajectory = new_trajectory.copy()
            new_trajectory[:, 12] += round_index * 10_000_000_000
        new_events = dagger_event_mask(
            new_actions, new_previous, new_dynamics_valid, settings,
            gate_phases=new_gate_phases,
        )
        new_speed_commands = np.asarray(collected.speed_commands, np.float32)
        new_action_chunks = (
            np.asarray(collected.action_chunks, np.float32).reshape(
                len(new_histories), len(action_chunk_offsets), 4
            )
            if action_chunk_weight > 0.0
            else np.empty((len(new_histories), 0, 4), np.float32)
        )
        new_executed_actions = (
            np.asarray(collected.executed_actions, np.float32).reshape(
                len(new_histories), 4
            )
            if reward_aux_dim > 0
            else np.zeros((len(new_histories), 4), np.float32)
        )
        new_reward_components = (
            np.asarray(collected.reward_components, np.float32).reshape(
                len(new_histories), reward_aux_dim
            )
            if reward_aux_dim > 0
            else np.empty((len(new_histories), 0), np.float32)
        )
        new_topology_targets = (
            np.asarray(collected.topology_targets, np.float32).reshape(
                len(new_histories), topology_target_dim
            )
            if topology_weight > 0.0
            else np.empty((len(new_histories), 0), np.float32)
        )
        new_topology_valid = (
            np.asarray(collected.topology_valid, np.bool_)
            if topology_weight > 0.0
            else np.zeros(len(new_histories), np.bool_)
        )
        if new_topology_valid.shape != (len(new_histories),):
            raise RuntimeError("collected topology-valid mask lost alignment")
        new_groups = (
            dagger_topology_group_ids(
                settings,
                np.asarray(collected.tracks, dtype=object),
                new_speed_commands,
                new_progress,
                new_gate_phases,
            )
            if bool(settings.get("dagger_group_balanced_sampling", False))
            else new_tracks.copy()
        )
        anchor_fraction = float(settings.get(
            "dagger_anchor_minimum_frontier_fraction", 0.99
        ))
        new_anchor_mask = np.asarray([
            float(speed) >= anchor_fraction * float(frontier_by_track.get(
                resolved_collected_tracks[track], speed
            ))
            for track, speed in zip(collected.tracks, new_speed_commands)
        ], np.bool_)
        from starscream.dagger_diagnostics import fresh_action_metrics
        fresh_metrics = fresh_action_metrics(policy, new_histories, new_actions,
            new_speed_commands, settings, seed=seed + round_index * 100000 + 7919)
        replay_rows_per_round = int(settings.get(
            "dagger_online_replay_rows_per_round", 0
        ))
        online_keep = stratified_replay_subsample(
            rng, new_tracks, new_gate_indices, new_events,
            new_teacher_modes, new_occupancy_modes,
            replay_rows_per_round,
        )
        if trajectory_quality:
            online_keep = quality_subsample(rng, new_tracks, new_trajectory, replay_rows_per_round, trajectory_quality)
            if not len(online_keep):
                raise RuntimeError('trajectory quality admitted no online labels')
        # Drop per-label Python objects as soon as their compact replay arrays
        # exist. Keeping these lists alive through HDF5 persistence, replay
        # growth, and optimization inflated peak RSS by one additional copy of
        # every small label array (histories were already cleared above).
        for name in (
            "actions", "previous_actions", "dynamics", "dynamics_valid",
            "tracks", "course_progress", "gate_phases",
            "active_gate_indices", "teacher_modes", "occupancy_modes", "trajectory",
            "speed_commands", "action_chunks", "topology_targets",
            "executed_actions", "reward_components",
            "topology_valid",
            "successful_labels",
        ):
            getattr(collected, name).clear()
        del collected_dynamics
        maximum = online_capacity
        if recovery_schedule.get('pure_expert'):
            # Fresh complete expert trajectories only; evict every old row in
            # every aligned array, including after a resume from an older pool.
            maximum = len(online_keep)
            if (not maximum or beta != 1.0 or dart_active
                    or collected.executed_teacher != collected.total_queries):
                raise RuntimeError('pure expert phase requires nonempty unperturbed expert collection')
        if (
            round_index <= permanent_rounds or refresh_permanent_expert
        ) and beta < 1.0 - 1.0e-9:
            raise ValueError(
                "permanent expert replay rounds require teacher_beta=1"
            )
        permanent_keep = (
            np.flatnonzero(new_successful)
            if round_index <= permanent_rounds or refresh_permanent_expert
            else np.empty((0,), np.int64)
        )
        if trajectory_quality:
            permanent_keep = permanent_keep[np.isin(new_trajectory[permanent_keep, 0], [0, 1])]
        replay_store.write_round(
            round_index,
            online={
                "histories": new_histories[online_keep].astype(
                    replay_history_dtype, copy=False
                ),
                "actions": new_actions[online_keep],
                "previous": new_previous[online_keep],
                "dynamics": new_dynamics[online_keep],
                "dynamics_valid": new_dynamics_valid[online_keep],
                "tracks": new_tracks[online_keep],
                "families": new_families[online_keep],
                "gate_indices": new_gate_indices[online_keep],
                "teacher_modes": new_teacher_modes[online_keep],
                "occupancy_modes": new_occupancy_modes[online_keep],
                "trajectory": new_trajectory[online_keep],
                "progress": new_progress[online_keep],
                "events": new_events[online_keep],
                "speed_commands": new_speed_commands[online_keep],
                "action_chunks": new_action_chunks[online_keep],
                "executed_actions": new_executed_actions[online_keep],
                "reward_components": new_reward_components[online_keep],
                "topology_targets": new_topology_targets[online_keep],
                "topology_valid": new_topology_valid[online_keep],
                "groups": new_groups[online_keep],
                "anchor_mask": new_anchor_mask[online_keep],
            },
            permanent={
                "histories": new_histories[permanent_keep].astype(
                    replay_history_dtype, copy=False
                ),
                "actions": new_actions[permanent_keep],
                "previous": new_previous[permanent_keep],
                "dynamics": new_dynamics[permanent_keep],
                "dynamics_valid": new_dynamics_valid[permanent_keep],
                "tracks": new_tracks[permanent_keep],
                "families": new_families[permanent_keep],
                "gate_indices": new_gate_indices[permanent_keep],
                "teacher_modes": new_teacher_modes[permanent_keep],
                "occupancy_modes": new_occupancy_modes[permanent_keep],
                "trajectory": new_trajectory[permanent_keep],
                "progress": new_progress[permanent_keep],
                "events": new_events[permanent_keep],
                "speed_commands": new_speed_commands[permanent_keep],
                "action_chunks": new_action_chunks[permanent_keep],
                "executed_actions": new_executed_actions[permanent_keep],
                "reward_components": new_reward_components[permanent_keep],
                "topology_targets": new_topology_targets[permanent_keep],
                "topology_valid": new_topology_valid[permanent_keep],
                "groups": new_groups[permanent_keep],
                "anchor_mask": new_anchor_mask[permanent_keep],
            },
        )
        online_histories = bounded_replay_append(
            online_histories,
            new_histories[online_keep].astype(replay_history_dtype, copy=False),
            maximum,
        )
        online_actions = bounded_replay_append(
            online_actions, new_actions[online_keep], maximum
        )
        online_previous = bounded_replay_append(
            online_previous, new_previous[online_keep], maximum
        )
        online_dynamics = bounded_replay_append(
            online_dynamics, new_dynamics[online_keep], maximum
        )
        online_dynamics_valid = bounded_replay_append(
            online_dynamics_valid, new_dynamics_valid[online_keep], maximum
        )
        online_tracks = bounded_replay_append(
            online_tracks, new_tracks[online_keep], maximum
        )
        online_families = bounded_replay_append(
            online_families, new_families[online_keep], maximum
        )
        online_gate_indices = bounded_replay_append(
            online_gate_indices, new_gate_indices[online_keep], maximum
        )
        online_teacher_modes = bounded_replay_append(
            online_teacher_modes, new_teacher_modes[online_keep], maximum
        )
        online_occupancy_modes = bounded_replay_append(
            online_occupancy_modes, new_occupancy_modes[online_keep], maximum
        )
        online_trajectory = bounded_replay_append(
            online_trajectory, new_trajectory[online_keep], maximum
        )
        online_progress = bounded_replay_append(
            online_progress, new_progress[online_keep], maximum
        )
        online_events = bounded_replay_append(
            online_events, new_events[online_keep], maximum
        )
        online_speed_commands = bounded_replay_append(
            online_speed_commands, new_speed_commands[online_keep], maximum
        )
        online_action_chunks = bounded_replay_append(
            online_action_chunks, new_action_chunks[online_keep], maximum
        )
        online_executed_actions = bounded_replay_append(
            online_executed_actions, new_executed_actions[online_keep], maximum
        )
        online_reward_components = bounded_replay_append(
            online_reward_components, new_reward_components[online_keep], maximum
        )
        online_topology_targets = bounded_replay_append(
            online_topology_targets, new_topology_targets[online_keep], maximum
        )
        online_topology_valid = bounded_replay_append(
            online_topology_valid, new_topology_valid[online_keep], maximum
        )
        online_groups = bounded_replay_append(
            online_groups, new_groups[online_keep], maximum
        )
        online_anchor_mask = bounded_replay_append(
            online_anchor_mask, new_anchor_mask[online_keep], maximum
        )
        if round_index <= permanent_rounds or refresh_permanent_expert:
            if beta < 1.0 - 1.0e-9:
                raise ValueError(
                    "permanent expert replay rounds require teacher_beta=1"
                )
            keep = permanent_keep
            permanent_maximum = permanent_capacity
            permanent_histories = bounded_replay_append(
                permanent_histories,
                new_histories[keep].astype(replay_history_dtype, copy=False),
                permanent_maximum,
            )
            permanent_actions = bounded_replay_append(
                permanent_actions, new_actions[keep], permanent_maximum
            )
            permanent_previous = bounded_replay_append(
                permanent_previous, new_previous[keep], permanent_maximum
            )
            permanent_dynamics = bounded_replay_append(
                permanent_dynamics, new_dynamics[keep], permanent_maximum
            )
            permanent_dynamics_valid = bounded_replay_append(
                permanent_dynamics_valid, new_dynamics_valid[keep], permanent_maximum
            )
            permanent_tracks = bounded_replay_append(
                permanent_tracks, new_tracks[keep], permanent_maximum
            )
            permanent_families = bounded_replay_append(
                permanent_families, new_families[keep], permanent_maximum
            )
            permanent_gate_indices = bounded_replay_append(
                permanent_gate_indices, new_gate_indices[keep], permanent_maximum
            )
            permanent_teacher_modes = bounded_replay_append(
                permanent_teacher_modes, new_teacher_modes[keep], permanent_maximum
            )
            permanent_occupancy_modes = bounded_replay_append(
                permanent_occupancy_modes, new_occupancy_modes[keep], permanent_maximum
            )
            permanent_trajectory = bounded_replay_append(
                permanent_trajectory, new_trajectory[keep], permanent_maximum
            )
            permanent_progress = bounded_replay_append(
                permanent_progress, new_progress[keep], permanent_maximum
            )
            permanent_events = bounded_replay_append(
                permanent_events, new_events[keep], permanent_maximum
            )
            permanent_speed_commands = bounded_replay_append(
                permanent_speed_commands, new_speed_commands[keep], permanent_maximum
            )
            permanent_action_chunks = bounded_replay_append(
                permanent_action_chunks, new_action_chunks[keep], permanent_maximum
            )
            permanent_executed_actions = bounded_replay_append(
                permanent_executed_actions,
                new_executed_actions[keep], permanent_maximum,
            )
            permanent_reward_components = bounded_replay_append(
                permanent_reward_components,
                new_reward_components[keep], permanent_maximum,
            )
            permanent_topology_targets = bounded_replay_append(
                permanent_topology_targets, new_topology_targets[keep], permanent_maximum
            )
            permanent_topology_valid = bounded_replay_append(
                permanent_topology_valid, new_topology_valid[keep], permanent_maximum
            )
            permanent_groups = bounded_replay_append(
                permanent_groups, new_groups[keep], permanent_maximum
            )
            permanent_anchor_mask = bounded_replay_append(
                permanent_anchor_mask, new_anchor_mask[keep], permanent_maximum
            )
        replay_ready_at = time.perf_counter()
        replay_prepare_seconds = replay_ready_at - replay_started_at
        replay_plans = [None, None]
        if (settings.get("dagger_cached_replay_sampling", False)
                and settings.get("dagger_hierarchical_replay_sampling", False)):
            # Metadata and sampler weights are frozen throughout this update phase.
            # Rebuild once per round; eviction/resume/adaptive quotas cannot stale it.
            for plan_index, (metadata, fraction) in enumerate((
                ((permanent_families, permanent_tracks, permanent_gate_indices,
                  permanent_events, permanent_teacher_modes, permanent_occupancy_modes),
                 float(settings.get("dagger_permanent_expert_fraction", 0.0))),
                ((online_families, online_tracks, online_gate_indices, online_events,
                  online_teacher_modes, online_occupancy_modes),
                 float(settings.get("online_fraction", 0.55))),
            )):
                quota = min(int(round(int(settings.get("batch_size", 1536)) * fraction)), len(metadata[0]))
                if quota:
                    replay_plans[plan_index] = HierarchicalReplayPlan(
                        *metadata, quota,
                        family_weights={family_to_id[str(name)]: float(weight)
                                        for name, weight in active_replay_family_weights.items()
                                        if str(name) in family_to_id},
                        nominal_fraction=float(settings.get("dagger_replay_nominal_fraction", .40)),
                        critical_fraction=float(settings.get("dagger_replay_critical_fraction", .35)),
                        recovery_fraction=float(settings.get("dagger_replay_recovery_fraction", .25)),
                    )
        if trajectory_quality:
            for plan_index, (families, track_ids, gates, telemetry, fraction) in enumerate((
                (permanent_families, permanent_tracks, permanent_gate_indices, permanent_trajectory,
                 float(settings.get('dagger_permanent_expert_fraction', 0))),
                (online_families, online_tracks, online_gate_indices, online_trajectory,
                 float(settings.get('online_fraction', .55))),
            )):
                quota = min(int(round(int(settings.get('batch_size', 1536))*fraction)), len(telemetry))
                if quota:
                    replay_plans[plan_index] = QualityReplayPlan(families, track_ids, gates, telemetry, quota,
                        config=replay_quality_config(trajectory_quality, telemetry, permanent=plan_index == 0,
                            online_override=settings.get('dagger_online_minibatch_quality')), family_weights={family_to_id[str(name)]: float(weight)
                        for name, weight in active_replay_family_weights.items() if str(name) in family_to_id})
        sampling_plan_seconds = time.perf_counter() - replay_ready_at
        if async_pipeline and round_index < int(settings.get('rounds', 8)):
            from starscream.dagger_async import lookahead_request
            next_tracks = admitted_tracks(fixed_admission, round_index + 1, tracks)
            next_collection_settings = copy.deepcopy(settings)
            next_collection_settings['track_sampling_family_weights'] = dict(active_replay_family_weights)
            next_collection_settings['dagger_transition_start_gate_weights_by_family'] = (
                copy.deepcopy(dynamic_gate_weights) if dynamic_sampler_active else {})
            request = lookahead_request(settings, next_collection_settings,
                window=round_index + 1, seed=seed, tracks=next_tracks,
                expert_refresh_end=resume_expert_refresh_end if resume_expert_refresh_rounds else 0)
            if request is not None:
                # The learner has not updated yet: freeze this actor while
                # training the current replay and evaluating the new model.
                collector.publish(policy, round_index - 1)
                frozen, kwargs = request
                collector.begin(kwargs, window=round_index + 1, settings=frozen)
        updates_started_at = time.perf_counter()
        updates_started_unix = time.time()
        policy.train()
        objective_settings: Mapping[str, Any] = settings
        robust_warmup_rounds = int(settings.get(
            "dagger_group_robust_warmup_rounds", 0
        ))
        if robust_warmup_rounds < 0:
            raise ValueError("dagger_group_robust_warmup_rounds cannot be negative")
        if round_index <= robust_warmup_rounds:
            objective_settings = dict(settings)
            objective_settings["dagger_group_robust_weight"] = 0.0
        if settings.get('dagger_static_group_statistics', False):
            if float(objective_settings.get('dagger_group_robust_weight', 0)) != 0:
                raise ValueError('static DAgger statistics require zero group robust weight')
            objective_settings = dict(objective_settings)
            objective_settings['dagger_statistics_group_count'] = max(
                1, int(online_groups.max(initial=-1))+1, int(permanent_groups.max(initial=-1))+1)
        capture_update = bool(settings.get('dagger_capture_updates', False))
        if capture_update:
            unsupported = any(float(settings.get(key, 0)) != 0 for key in (
                'action_chunk_weight', 'topology_weight', 'reward_aux_weight',
                'gate_contrastive_weight', 'dagger_anchor_distillation_weight'))
            stochastic = any(float(getattr(policy, key, 0)) != 0 for key in (
                'transformer_dropout', 'observation_embedding_dropout',
                'history_condition_dropout_probability'))
            if (not device.startswith('cuda') or policy.action_head_type != 'mlp'
                    or not settings.get('dagger_static_group_statistics', False)
                    or not settings.get('dagger_fast_group_statistics', False)
                    or unsupported or stochastic or anchor_policy is not None):
                raise ValueError('DAgger update capture requires deterministic CUDA MLP, static fast statistics and no auxiliary/anchor objectives')
            objective_settings = dict(objective_settings)
            objective_settings['_dagger_physical_scales'] = torch.tensor(
                settings.get('physical_action_scales', [10., 3., 3., 3.]), device=device)
        if capture_update:
            from starscream.imitation_objective import action_dimension_weights
            objective_settings['_dagger_action_dimension_weights'] = action_dimension_weights(
                settings, next(policy.parameters()))
        losses: list[float] = []
        loss_pieces: dict[str, list[float]] = {
            name: [] for name in (
                "action_loss", "physical_action_loss", "dynamics_loss",
                "topology_loss", "plan_state_loss", "action_chunk_loss",
                "reward_aux_loss", "gate_contrastive_loss",
                "anchor_loss", "group_loss_mean",
                "group_loss_max", "group_loss_std", "flow_matching_loss",
                "shortcut_bootstrap_loss", "flow_endpoint_mse",
                "flow_stabilization_loss", "flow_deployment_endpoint_loss",
                "flow_deployment_physical_loss",
                "flow_deployment_collective_mae_mps2",
                "flow_deployment_body_rate_mae_rps",
                "flow_step_size",
                "mixture_mode_loss", "mixture_mode_accuracy",
                "mixture_mode_entropy", "mixture_codebook_mse",
                "mixture_deployed_action_loss",
            )
        }
        batch_size = int(settings.get("batch_size", 1536))
        online_fraction = float(settings.get("online_fraction", 0.55))
        permanent_fraction = float(settings.get(
            "dagger_permanent_expert_fraction", 0.0
        ))
        if not 0.0 <= online_fraction + permanent_fraction <= 1.0:
            raise ValueError("online and permanent DAgger fractions must sum to <= 1")
        def prepare_update_batches():
            for _ in range(int(settings.get("updates_per_round", 800))):
                online_count = min(int(round(batch_size * online_fraction)), len(online_histories))
                permanent_count = min(
                    int(round(batch_size * permanent_fraction)), len(permanent_histories)
                )
                offline_count = batch_size - online_count - permanent_count
                if trajectory_quality and offline_count:
                    raise RuntimeError('quality replay underfilled: collect qualified online/permanent rows; legacy offline fallback is forbidden')
                histories: list[np.ndarray] = []
                actions: list[np.ndarray] = []
                previous: list[np.ndarray] = []
                dynamics: list[np.ndarray] = []
                dynamics_valid: list[np.ndarray] = []
                speed_commands: list[np.ndarray] = []
                action_chunks: list[np.ndarray] = []
                action_chunk_valid: list[np.ndarray] = []
                executed_actions: list[np.ndarray] = []
                reward_components: list[np.ndarray] = []
                topology_targets: list[np.ndarray] = []
                topology_valid: list[np.ndarray] = []
                group_ids: list[np.ndarray] = []
                anchor_masks: list[np.ndarray] = []
                if offline_count:
                    if (
                        data is None or normalized is None
                        or offline_dynamics is None or offline_dynamics_valid is None
                    ):
                        raise ValueError(
                            "DAgger requested offline update rows without offline data"
                        )
                    offline_selected = balanced_choice(
                        rng, train_indices, data.track_ids, offline_count
                    )
                    histories.append(batch_sequences(
                        normalized, data.episode_ids, offline_selected,
                        policy.context_steps,
                    ))
                    actions.append(data.actions[offline_selected])
                    previous.append(data.previous_actions[offline_selected])
                    dynamics.append(offline_dynamics[offline_selected])
                    dynamics_valid.append(offline_dynamics_valid[offline_selected])
                    speed_commands.append(np.full(
                        offline_count,
                        float(settings.get("dagger_offline_speed_command", 15.0)),
                        np.float32,
                    ))
                    action_chunks.append(data.action_chunks[offline_selected])
                    action_chunk_valid.append(np.full(
                        offline_count, action_chunk_weight > 0.0, np.bool_
                    ))
                    executed_actions.append(np.zeros((offline_count, 4), np.float32))
                    reward_components.append(np.empty(
                        (offline_count, reward_aux_dim), np.float32
                    ))
                    topology_targets.append(np.zeros(
                        (offline_count, topology_replay_dim), np.float32
                    ))
                    topology_valid.append(np.zeros(offline_count, np.bool_))
                    group_ids.append(np.full(offline_count, -1, np.int16))
                    anchor_masks.append(np.zeros(offline_count, np.bool_))
                if permanent_count:
                    event_fraction = float(settings.get(
                        "dagger_event_replay_fraction", 0.0
                    ))
                    if bool(settings.get("dagger_hierarchical_replay_sampling", False)):
                        configured_family_weights = {
                            family_to_id[str(name)]: float(weight)
                            for name, weight in active_replay_family_weights.items()
                            if str(name) in family_to_id
                        }
                        permanent_selected = balanced_family_trajectory_choice(
                            rng, permanent_families, permanent_tracks,
                            permanent_gate_indices,
                            permanent_events, permanent_teacher_modes,
                            permanent_occupancy_modes, permanent_count,
                            plan=replay_plans[0],
                            family_weights=configured_family_weights,
                            nominal_fraction=float(settings.get(
                                "dagger_replay_nominal_fraction", 0.40
                            )),
                            critical_fraction=float(settings.get(
                                "dagger_replay_critical_fraction", 0.35
                            )),
                            recovery_fraction=float(settings.get(
                                "dagger_replay_recovery_fraction", 0.25
                            )),
                        )
                    elif event_fraction > 0.0:
                        permanent_selected = balanced_event_progress_choice(
                            rng, permanent_tracks, permanent_progress,
                            permanent_events, permanent_count,
                            event_fraction=event_fraction,
                            late_fraction=float(settings.get(
                                "permanent_expert_late_course_fraction", 0.25
                            )),
                            late_threshold=float(settings.get(
                                "online_late_course_threshold", 0.65
                            )),
                        )
                    else:
                        permanent_selected = balanced_progress_choice(
                            rng, permanent_tracks, permanent_progress, permanent_count,
                            late_fraction=float(settings.get(
                                "permanent_expert_late_course_fraction", 0.45
                            )),
                            late_threshold=float(settings.get(
                                "online_late_course_threshold", 0.65
                            )),
                        )
                    histories.append(permanent_histories[permanent_selected])
                    actions.append(permanent_actions[permanent_selected])
                    previous.append(permanent_previous[permanent_selected])
                    dynamics.append(permanent_dynamics[permanent_selected])
                    dynamics_valid.append(permanent_dynamics_valid[permanent_selected])
                    speed_commands.append(permanent_speed_commands[permanent_selected])
                    action_chunks.append(
                        permanent_action_chunks[permanent_selected]
                    )
                    action_chunk_valid.append(np.full(
                        permanent_count, action_chunk_weight > 0.0, np.bool_
                    ))
                    executed_actions.append(
                        permanent_executed_actions[permanent_selected]
                    )
                    reward_components.append(
                        permanent_reward_components[permanent_selected]
                    )
                    topology_targets.append(
                        permanent_topology_targets[permanent_selected]
                    )
                    topology_valid.append(
                        permanent_topology_valid[permanent_selected]
                    )
                    group_ids.append(permanent_groups[permanent_selected])
                    anchor_masks.append(permanent_anchor_mask[permanent_selected])
                if online_count:
                    event_fraction = float(settings.get(
                        "dagger_event_replay_fraction", 0.0
                    ))
                    if bool(settings.get("dagger_hierarchical_replay_sampling", False)):
                        configured_family_weights = {
                            family_to_id[str(name)]: float(weight)
                            for name, weight in active_replay_family_weights.items()
                            if str(name) in family_to_id
                        }
                        selected = balanced_family_trajectory_choice(
                            rng, online_families, online_tracks,
                            online_gate_indices,
                            online_events, online_teacher_modes,
                            online_occupancy_modes, online_count,
                            plan=replay_plans[1],
                            family_weights=configured_family_weights,
                            nominal_fraction=float(settings.get(
                                "dagger_replay_nominal_fraction", 0.40
                            )),
                            critical_fraction=float(settings.get(
                                "dagger_replay_critical_fraction", 0.35
                            )),
                            recovery_fraction=float(settings.get(
                                "dagger_replay_recovery_fraction", 0.25
                            )),
                        )
                    elif event_fraction > 0.0:
                        selected = balanced_event_progress_choice(
                            rng,
                            (
                                online_groups
                                if bool(settings.get(
                                    "dagger_group_balanced_sampling", False
                                )) else online_tracks
                            ),
                            online_progress, online_events, online_count,
                            event_fraction=event_fraction,
                            late_fraction=float(settings.get(
                                "online_late_course_fraction", 0.25
                            )),
                            late_threshold=float(settings.get(
                                "online_late_course_threshold", 0.65
                            )),
                        )
                    elif float(settings.get("online_late_course_fraction", 0.0)) > 0.0:
                        selected = balanced_progress_choice(
                            rng,
                            (
                                online_groups
                                if bool(settings.get(
                                    "dagger_group_balanced_sampling", False
                                )) else online_tracks
                            ),
                            online_progress, online_count,
                            late_fraction=float(settings["online_late_course_fraction"]),
                            late_threshold=float(settings.get(
                                "online_late_course_threshold", 0.65
                            )),
                        )
                    else:
                        selected = balanced_choice(
                            rng, np.arange(len(online_histories)),
                            (
                                online_groups
                                if bool(settings.get(
                                    "dagger_group_balanced_sampling", False
                                )) else online_tracks
                            ),
                            online_count,
                        )
                    histories.append(online_histories[selected])
                    actions.append(online_actions[selected])
                    previous.append(online_previous[selected])
                    dynamics.append(online_dynamics[selected])
                    dynamics_valid.append(online_dynamics_valid[selected])
                    speed_commands.append(online_speed_commands[selected])
                    action_chunks.append(online_action_chunks[selected])
                    action_chunk_valid.append(np.full(
                        online_count, action_chunk_weight > 0.0, np.bool_
                    ))
                    executed_actions.append(online_executed_actions[selected])
                    reward_components.append(online_reward_components[selected])
                    topology_targets.append(online_topology_targets[selected])
                    topology_valid.append(online_topology_valid[selected])
                    group_ids.append(online_groups[selected])
                    anchor_masks.append(online_anchor_mask[selected])
                yield tuple(np.concatenate(values) for values in (
                    histories, actions, previous, dynamics, dynamics_valid,
                    speed_commands, action_chunks, action_chunk_valid,
                    executed_actions, reward_components, topology_targets,
                    topology_valid, group_ids, anchor_masks,
                ))

        update_transfer = UpdateBatchTransfer(device, settings.get("dagger_packed_update_transfer", False))
        defer_update_metrics = bool(settings.get("defer_update_metrics", False))
        def perform_dagger_update(tensor_batch):
            (history_tensor, action_tensor, previous_tensor, dynamics_tensor,
             dynamics_valid_tensor, speed_command_tensor, action_chunk_tensor,
             action_chunk_valid_tensor, executed_action_tensor,
             reward_component_tensor, topology_target_tensor,
             topology_valid_tensor, group_id_tensor, anchor_mask_tensor) = tensor_batch
            anchor_actions = None
            if anchor_policy is not None:
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=amp
                ):
                    anchor_actions = anchor_policy(
                        history_tensor, speed_command_tensor
                    )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss, pieces = dagger_loss(
                    policy, history_tensor, action_tensor, previous_tensor,
                    dynamics_tensor, objective_settings, dynamics_valid_tensor,
                    speed_command_tensor,
                    topology_target_tensor,
                    topology_valid_tensor,
                    group_id_tensor,
                    anchor_actions,
                    anchor_mask_tensor,
                    action_chunk_tensor,
                    action_chunk_valid_tensor,
                    executed_action_tensor,
                    reward_component_tensor,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings.get("gradient_clip", 2.0)))
            optimizer.step()
            return loss, pieces
        update_step = perform_dagger_update
        if capture_update:
            from starscream.dagger_update_graph import CapturedDaggerUpdate
            update_step = CapturedDaggerUpdate(policy, optimizer, perform_dagger_update)
        with prefetched_batches(prepare_update_batches(),
                                int(settings.get("dagger_batch_prefetch", 0))) as prepared:
            for cpu_batch in prepared:
                loss, pieces = update_step(update_transfer(cpu_batch))
                losses.append(loss.detach() if defer_update_metrics else float(loss.detach()))
                for name in loss_pieces:
                    value = pieces.get(name)
                    loss_pieces[name].append(
                        0.0 if value is None else (
                            value.detach() if defer_update_metrics else float(value.detach()))
                    )
        scalar_lists_to_host({"total_loss": losses, **loss_pieces})
        del update_step  # A fresh graph captures each round's LR/objective.
        updates_seconds = time.perf_counter() - updates_started_at
        # Free index caches before evaluation/collection to bound peak RAM.
        replay_plan_bytes = sum(plan.index_bytes for plan in replay_plans if plan is not None)
        quality_metrics = {
            f'teacher_controller/{key}': value
            for key, value in collected.quality_counts.items() if key.startswith('teacher_')
        }
        if collection_control:
            quality_metrics.update({f'collection/{key.removeprefix("collection_")}': value
                for key, value in collected.quality_counts.items() if key.startswith('collection_')})
            for key in ('misses', 'backward', 'planned_plane_crossings', 'stopped'):
                quality_metrics[f'collection/{key}'] = collected.quality_counts.get(key, 0)
            for cls, name in enumerate(('nominal', 'corrective', 'recovery')):
                quality_metrics[f'collection/valid_{name}_rows'] = int(np.sum(new_trajectory[:, 0] == cls))
                quality_metrics[f'collection/retained_{name}_rows'] = int(np.sum(new_trajectory[online_keep, 0] == cls))
        if trajectory_quality:
            combined = np.zeros(3, np.int64)
            for pool_name, plan in zip(('permanent', 'online'), replay_plans):
                if plan is not None:
                    combined += plan.sample_counts
                    quality_metrics[f'quality/{pool_name}/unique_rows_per_batch'] = float(plan.unique_rows_total/max(plan.sampled_batches, 1))
                    quality_metrics[f'quality/{pool_name}/sampled_precursor'] = float(plan.precursor_draws/max(plan.sample_counts.sum(), 1))
                    quality_metrics[f'quality/{pool_name}/precursor_shortfall_rows_per_batch'] = int(plan.precursor_shortfall)
                    for cls, name in enumerate(('nominal', 'corrective', 'recovery')):
                        stored = int(np.sum(plan.telemetry[:, 0] == cls))
                        quality_metrics[f'quality/{pool_name}/stored_{name}'] = stored
                        quality_metrics[f'quality/{pool_name}/draws_per_stored_{name}'] = float(plan.sample_counts[cls]/max(stored, 1))
                        keys = [12, 9] if cls == 2 else [12, 8]
                        quality_metrics[f'quality/{pool_name}/distinct_{name}_encounters'] = int(len(
                            np.unique(plan.telemetry[plan.telemetry[:, 0] == cls][:, keys], axis=0)))
                        quality_metrics[f'quality/{pool_name}/sampled_{name}'] = float(plan.sample_counts[cls]/max(plan.sample_counts.sum(), 1))
                        quality_metrics[f'quality/{pool_name}/shortfall_{name}_rows_per_batch'] = int(plan.shortfall[cls])
            for cls, name in enumerate(('nominal', 'corrective', 'recovery')):
                quality_metrics[f'quality/combined/sampled_{name}'] = float(combined[cls]/max(combined.sum(), 1))
            quality_metrics.update({f'quality/collection/{k}': v for k,v in collected.quality_counts.items()})
            quality_metrics['quality/collection/unqualified_rows'] = int(np.sum(~np.isin(new_trajectory[:,0], [0,1,2])))
        del replay_plans
        policy.eval()
        observed_source_losses: dict[str, float] = {}
        if dynamic_sampler_enabled and len(online_histories):
            probe_rows = int(dynamic_sampler_config.get(
                "replay_probe_rows_per_source", 384
            ))
            if probe_rows < 1:
                raise ValueError(
                    "dynamic DAgger replay_probe_rows_per_source must be positive"
                )
            probe_rng = np.random.default_rng(
                seed + round_index * 100000 + 0xD19A
            )
            for source in sorted(set(source_by_family.values())):
                source_family_ids = [
                    family_to_id[family]
                    for family, item_source in source_by_family.items()
                    if item_source == source
                ]
                chunks: list[np.ndarray] = []
                per_family = max(1, probe_rows // max(len(source_family_ids), 1))
                for family_id in source_family_ids:
                    pool = np.flatnonzero(online_families == family_id)
                    if not len(pool):
                        continue
                    chunks.append(probe_rng.choice(
                        pool, size=min(per_family, len(pool)), replace=False,
                    ))
                if not chunks:
                    continue
                selected_probe = np.concatenate(chunks)
                probe_histories = torch.from_numpy(
                    online_histories[selected_probe].astype(np.float32)
                ).to(device)
                probe_speed = torch.from_numpy(
                    online_speed_commands[selected_probe]
                ).to(device)
                probe_anchor_actions = None
                if anchor_policy is not None:
                    with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=amp
                    ):
                        probe_anchor_actions = anchor_policy(
                            probe_histories, probe_speed
                        )
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=amp
                ):
                    probe_loss, _ = imitation_loss(
                        policy,
                        probe_histories,
                        torch.from_numpy(online_actions[selected_probe]).to(device),
                        torch.from_numpy(online_previous[selected_probe]).to(device),
                        torch.from_numpy(online_dynamics[selected_probe]).to(device),
                        objective_settings,
                        torch.from_numpy(
                            online_dynamics_valid[selected_probe]
                        ).to(device),
                        probe_speed,
                        torch.from_numpy(
                            online_topology_targets[selected_probe]
                        ).to(device),
                        torch.from_numpy(
                            online_topology_valid[selected_probe]
                        ).to(device),
                        torch.from_numpy(online_groups[selected_probe]).to(device),
                        probe_anchor_actions,
                        torch.from_numpy(
                            online_anchor_mask[selected_probe]
                        ).to(device),
                        torch.from_numpy(
                            online_action_chunks[selected_probe]
                        ).to(device),
                        torch.full(
                            (len(selected_probe),),
                            action_chunk_weight > 0.0,
                            dtype=torch.bool, device=device,
                        ),
                        torch.from_numpy(
                            online_executed_actions[selected_probe]
                        ).to(device),
                        torch.from_numpy(
                            online_reward_components[selected_probe]
                        ).to(device),
                    )
                observed_source_losses[source] = float(probe_loss)
        evaluation_started_at = time.perf_counter()
        evaluation_started_unix = time.time()
        evaluation = evaluate_policy(
            policy, normalizer, settings, evaluation_stage,
            count=int(settings.get("evaluation_episodes", 96)),
            seed_base=int(settings.get("evaluation_seed", 20260900)), device=device,
            collector=evaluation_collector,
        )
        evaluation_seconds = time.perf_counter() - evaluation_started_at
        evaluation['dagger_policy_version'] = float(round_index)
        if bool(settings.get("dagger_completion_survival_selection", False)):
            evaluation = dagger_completion_survival_metrics(evaluation, settings)
        elif bool(settings.get("pace_aware_selection", False)):
            evaluation = refinement_metrics(evaluation, evaluation_stage, settings)
        if dynamic_sampler_enabled:
            assert manifest_path is not None
            observed_competence, observed_frontiers = (
                sampler_feedback(
                    evaluation, settings, manifest_path,
                    route_horizon=int(dynamic_sampler_config.get(
                        "route_horizon", route_gate_count
                    )),
                )
            )
            dynamic_sampler_state, dynamic_family_weights, dynamic_gate_weights = (
                update_dynamic_dagger_sampler(
                    base_family_weights=base_family_weights,
                    source_by_family=source_by_family,
                    observed_competence=observed_competence,
                    observed_frontiers=observed_frontiers,
                    observed_losses=observed_source_losses,
                    previous_state=dynamic_sampler_state,
                    config=dynamic_sampler_config,
                )
            )
        task_bank_metrics: dict[str, float] = {}
        if task_bank is not None:
            # The ordinary DAgger evaluator normally contains the validation
            # split, while the adaptive bank owns training tasks.  Reusing the
            # former would silently assign zero progress to every active task.
            # Evaluate the active bank explicitly under the fixed validation
            # spawn contract, then feed those per-task closed-loop scores to
            # the DAgger proxy for Algorithm 2.
            task_bank_episodes_per_task = int(task_bank_config.get(
                "evaluation_episodes_per_task", 2
            ))
            if task_bank_episodes_per_task < 1:
                raise ValueError(
                    "DAgger task-bank evaluation_episodes_per_task must be positive"
                )
            task_bank_stage = replace(
                evaluation_stage,
                name=f"{evaluation_stage.name}_dagger_task_bank",
                tracks=tuple(task_bank.active),
            )
            task_bank_evaluation = evaluate_policy(
                policy,
                normalizer,
                settings,
                task_bank_stage,
                count=len(task_bank.active) * task_bank_episodes_per_task,
                seed_base=(
                    int(settings.get("evaluation_seed", 20260900))
                    + round_index * 1000003
                    + 0xDA66
                ),
                device=device,
            )
            task_scores = {
                track: float(task_bank_evaluation[
                    f"track/{task_bank_name_by_track[track]}/selection_score"
                ])
                for track in task_bank.active
            }
            task_bank_metrics = task_bank.observe(task_scores)
            task_bank_metrics.update({
                "dagger_task_bank/full_course_success": float(
                    task_bank_evaluation["full_course_success"]
                ),
                "dagger_task_bank/minimum_track_full_course_success": float(
                    task_bank_evaluation["minimum_track_full_course_success"]
                ),
                "dagger_task_bank/performance_weighted_success_score": float(
                    task_bank_evaluation["performance_weighted_success_score"]
                ),
            })
        environment_steps += collected.environment_steps
        reporting = run_reporting_evaluation(
            environment_steps, round_index=round_index
        )
        frontier = run_frontier_evaluation(
            environment_steps, round_index=round_index
        )
        if settings.get('timed_reporting'):
            from starscream.timed_teacher_reporting import report_teacher
            report_teacher(policy, normalizer, settings, evaluation_stage, device, logger,
                           manager.checkpoint_dir, environment_steps, round_index)
        if settings.get('midtrain_pace_probe'):
            from starscream.midtrain_evaluation import evaluate_midtraining
            evaluate_midtraining(policy, normalizer, settings, evaluation_stage, device,
                                 logger, manager.checkpoint_dir, environment_steps, round_index)
        round_seconds = time.perf_counter() - round_started
        train_metrics = {
            **quality_metrics,
            **recovery_schedule,
            "collection_wait_seconds": collection_wait_seconds,
            "async_pipeline": float(async_pipeline),
            "collection_started_unix": collection_started_unix,
            "updates_started_unix": updates_started_unix,
            "evaluation_started_unix": evaluation_started_unix,
            "pure_expert_aborted_episodes": collected.pure_expert_aborted_episodes,
            "learning_rate": float(optimizer.param_groups[0]['lr']),
            "round": round_index,
            "updates_seconds": updates_seconds,
            "evaluation_seconds": evaluation_seconds,
            "sampling_plan_seconds": sampling_plan_seconds,
            "sampling_plan_index_bytes": replay_plan_bytes,
            "replay_prepare_seconds": replay_prepare_seconds,
            **{f"collection_last_call_host/{name}": value
               for name, value in getattr(collector, "last_profile", {}).items()},
            "loss": float(np.mean(losses)),
            **{
                name: float(np.mean(values))
                for name, values in loss_pieces.items()
            },
            **fresh_metrics,
            "new_labels": len(new_histories),
            "new_online_replay_labels": len(online_keep),
            "new_online_replay_retention_fraction": (
                len(online_keep) / max(len(new_histories), 1)
            ),
            "online_replay": len(online_histories),
            "permanent_expert_replay": len(permanent_histories),
            "new_event_label_fraction": float(new_events.mean()),
            "new_crossing_label_fraction": float(
                np.mean(new_gate_phases == 2)
            ),
            "new_post_crossing_label_fraction": float(
                np.mean(new_gate_phases == 3)
            ),
            "new_acquisition_label_fraction": float(
                np.mean(new_gate_phases == 4)
            ),
            "online_event_replay_fraction": float(
                online_events.mean() if len(online_events) else 0.0
            ),
            "permanent_event_replay_fraction": float(
                permanent_events.mean() if len(permanent_events) else 0.0
            ),
            "teacher_valid_fraction": collected.valid_queries / max(collected.total_queries, 1),
            "coherent_chunk_valid_fraction": float(
                (np.abs(new_action_chunks) <= 1.0001).all(axis=(1, 2)).mean()
                if action_chunk_source == 'coherent_teacher' and len(new_action_chunks) else 0.0
            ),
            "topology_target_valid_fraction": float(
                new_topology_valid.mean() if len(new_topology_valid) else 0.0
            ),
            "topology_target_rejected_labels": float(
                np.sum(~new_topology_valid) if topology_weight > 0.0 else 0
            ),
            "group_robust_active": float(round_index > robust_warmup_rounds),
            "teacher_solver_failure_fraction": (
                collected.solver_failures / max(collected.total_queries, 1)
            ),
            "teacher_recovery_fraction": (
                collected.recovery_queries / max(collected.total_queries, 1)
            ),
            "teacher_execution_fraction": collected.executed_teacher / max(collected.total_queries, 1),
            "dart_noisy_execution_fraction": (
                collected.dart_noisy_steps / max(collected.executed_teacher, 1)
            ),
            "targeted_start_episode_fraction": (
                collected.scheduled_targeted_start_episodes
                / max(
                    collected.scheduled_targeted_start_episodes
                    + collected.scheduled_canonical_start_episodes,
                    1,
                )
            ),
            "targeted_start_episodes": float(
                collected.scheduled_targeted_start_episodes
            ),
            "expert_prefix_scheduled_episodes": float(
                collected.scheduled_expert_prefix_episodes
            ),
            "expert_prefix_completed_handoffs": float(
                collected.completed_expert_prefix_handoffs
            ),
            "expert_prefix_handoff_completion_fraction": (
                collected.completed_expert_prefix_handoffs
                / max(collected.scheduled_expert_prefix_episodes, 1)
            ),
            "expert_prefix_forced_teacher_fraction": (
                collected.expert_prefix_forced_steps
                / max(collected.total_queries, 1)
            ),
            "expert_prefix_mean_handoff_speed_mps": (
                collected.expert_prefix_handoff_speed_sum
                / max(collected.completed_expert_prefix_handoffs, 1)
            ),
            "teacher_solver_seconds_per_query": (
                collected.teacher_solve_seconds / max(collected.total_queries, 1)
            ),
            "teacher_wall_seconds_per_query": (
                collected.teacher_wall_seconds / max(collected.total_queries, 1)
            ),
            "environment_step_seconds_per_query": (
                collected.environment_step_seconds
                / max(collected.total_queries, 1)
            ),
            "feature_seconds_per_query": (
                collected.feature_seconds / max(collected.total_queries, 1)
            ),
            "controller_sync_seconds_per_query": (
                collected.controller_wall_breakdown[0]
                / max(collected.total_queries, 1)
            ),
            "controller_reference_seconds_per_query": (
                collected.controller_wall_breakdown[1]
                / max(collected.total_queries, 1)
            ),
            "controller_backend_seconds_per_query": (
                collected.controller_wall_breakdown[2]
                / max(collected.total_queries, 1)
            ),
            "controller_postprocess_seconds_per_query": (
                collected.controller_wall_breakdown[3]
                / max(collected.total_queries, 1)
            ),
            # acados time_tot above is feedback-phase-only under split RTI.
            # Report the complete preparation+feedback interval separately so
            # native solve time is not misclassified as Python wrapper time.
            "backend_prepare_feedback_seconds_per_query": (
                collected.backend_wall_breakdown[1] / max(collected.total_queries, 1)
            ),
            "backend_setup_seconds_per_query": (
                collected.backend_wall_breakdown[0] / max(collected.total_queries, 1)
            ),
            "backend_extraction_seconds_per_query": (
                collected.backend_wall_breakdown[2] / max(collected.total_queries, 1)
            ),
            "backend_diagnostics_seconds_per_query": (
                collected.backend_wall_breakdown[3] / max(collected.total_queries, 1)
            ),
            "collection_worker_busy_fraction": (
                (
                    collected.teacher_wall_seconds
                    + collected.environment_step_seconds
                    + collected.feature_seconds
                )
                / max(
                    collection_seconds * int(settings.get("rollout_envs", 1)),
                    1.0e-6,
                )
            ),
            "successful_episode_label_fraction": (
                collected.accepted_episodes
                / max(collected.accepted_episodes + collected.rejected_episodes, 1)
            ),
            "successful_multilap_episodes": float(
                collected.accepted_multilap_episodes
            ),
            "teacher_speed_command_min": float(new_speed_commands.min()),
            "teacher_speed_command_mean": float(new_speed_commands.mean()),
            "teacher_speed_command_max": float(new_speed_commands.max()),
            "topology_group_count": float(len(np.unique(new_groups))),
            "environment_steps": environment_steps,
            "environment_steps_per_second": (
                collected.environment_steps / max(collection_seconds, 1e-6)
            ),
            "collection_seconds": collection_seconds,
            "round_seconds": round_seconds,
            "active_course_count": float(len(round_tracks)),
            "replay_persisted_round": float(round_index),
            "new_canonical_occupancy_fraction": float(
                np.mean(new_occupancy_modes == 0)
            ),
            "new_cold_reset_occupancy_fraction": float(
                np.mean(new_occupancy_modes == 1)
            ),
            "new_expert_prefix_occupancy_fraction": float(
                np.mean(new_occupancy_modes == 2)
            ),
            "new_recovery_teacher_fraction": float(
                np.mean(new_teacher_modes != 0)
            ),
            **task_bank_metrics,
        }
        if dynamic_sampler_enabled and dynamic_sampler_state is not None:
            sampler_multipliers = dict(
                dynamic_sampler_state.get("multipliers", {})
            )
            sampler_competence = dict(
                dynamic_sampler_state.get("competence", {})
            )
            sampler_losses = dict(dynamic_sampler_state.get("loss", {}))
            sampler_frontiers = dict(
                dynamic_sampler_state.get("frontiers", {})
            )
            train_metrics.update({
                "dynamic_sampler_active": float(dynamic_sampler_active),
                "dynamic_sampler_priority_entropy": float(
                    dynamic_sampler_state.get("priority_entropy", 0.0)
                ),
                "dynamic_sampler_multiplier_min": float(
                    min(sampler_multipliers.values())
                ),
                "dynamic_sampler_multiplier_max": float(
                    max(sampler_multipliers.values())
                ),
                "dynamic_sampler_competence_mean": float(
                    np.mean(list(sampler_competence.values()))
                ),
                "dynamic_sampler_replay_loss_mean": float(
                    np.mean(list(sampler_losses.values()))
                ),
            })
            for source in sorted(sampler_multipliers):
                train_metrics[
                    f"dynamic_sampler/{source}/multiplier"
                ] = float(sampler_multipliers[source])
                train_metrics[
                    f"dynamic_sampler/{source}/competence"
                ] = float(sampler_competence[source])
                train_metrics[
                    f"dynamic_sampler/{source}/replay_loss"
                ] = float(sampler_losses[source])
                if source in sampler_frontiers:
                    train_metrics[
                        f"dynamic_sampler/{source}/frontier_gate"
                    ] = float(sampler_frontiers[source])
            for key, value in dynamic_sampler_state.get("behavior_health", {}).items():
                train_metrics[f"dynamic_sampler_behavior_{key}"] = float(value)
        round_memory_after = _memory_snapshot()
        train_metrics.update(round_memory_after)
        from starscream.memory_pressure import memory_pressure_delta
        train_metrics.update(memory_pressure_delta(round_memory_before, round_memory_after,
                                                   time.perf_counter()-round_started))
        for family_id, family_name in enumerate(family_names):
            train_metrics[
                f"replay_family/{family_name}/new_label_fraction"
            ] = float(np.mean(new_families == family_id))
            train_metrics[
                f"replay_family/{family_name}/online_rows"
            ] = float(np.sum(online_families == family_id))
        if frontier is not None:
            train_metrics.update(frontier)
        selection_regression = bool(
            settings.get("dagger_rollback_on_selection_regression", False)
            and float(evaluation["selection_score"])
            < safe_selection_score - float(settings.get(
                "dagger_rollback_selection_tolerance", 0.0
            ))
        )
        rollback = bool(
            round_index >= int(settings.get("dagger_rollback_start_round", 1))
            and (
                dagger_should_rollback(evaluation, safe_evaluation, settings)
                or selection_regression
            )
        )
        if (
            rollback and dynamic_sampler_enabled
            and pre_round_dynamic_sampler_state is not None
        ):
            dynamic_sampler_state = pre_round_dynamic_sampler_state
            dynamic_family_weights = {
                str(name): float(value)
                for name, value in dict(
                    dynamic_sampler_state.get("family_weights", base_family_weights)
                ).items()
            }
            dynamic_gate_weights = {
                str(family): {
                    int(gate): float(weight)
                    for gate, weight in dict(weights).items()
                }
                for family, weights in dict(
                    dynamic_sampler_state.get("gate_weights", {})
                ).items()
            }
        if rollback and task_bank is not None and pre_round_task_bank_state is not None:
            task_bank.load_state_dict(pre_round_task_bank_state)
        if rollback and async_pipeline:
            # Lookahead is speculative until this model's validation resolves.
            # Discard it before checkpointing the restored actor/optimizer.
            discard_started = time.perf_counter()
            collector.discard_pending()
            discard_wait = time.perf_counter() - discard_started
            round_seconds += discard_wait
            train_metrics['pipeline_discard_wait_seconds'] = discard_wait
            train_metrics['round_seconds'] = round_seconds
        train_metrics["safety_rollback"] = float(rollback)
        train_metrics["rollback_selection_regression"] = float(
            selection_regression
        )
        train_metrics["rollback_reference_full_course_success"] = float(
            safe_evaluation.get("full_course_success", 0.0)
        )
        train_metrics["rollback_full_course_drop"] = float(
            safe_evaluation.get("full_course_success", 0.0)
            - evaluation.get("full_course_success", 0.0)
        )
        logger.log_train(train_metrics, environment_steps)
        logger.log_eval(evaluation, environment_steps)
        if rollback:
            policy.load_state_dict(safe_policy_state)
            # Optimizer.load_state_dict can reuse same-device tensor storage.
            # Keep the safety snapshot immutable across subsequent updates.
            from starscream.dagger_update_graph import restore_optimizer_snapshot
            restore_optimizer_snapshot(optimizer, safe_optimizer_state)
            manager.save(
                checkpoint_payload(
                    policy, normalizer, stage="dagger", track=",".join(tracks),
                    optimizer=optimizer,
                    extra={
                        "dynamics_target_mean": dynamics_mean,
                        "dynamics_target_std": dynamics_std,
                        "round": round_index,
                        "environment_steps": environment_steps,
                        "initial_checkpoint": str(initial_path),
                        "reporting_evaluation": safe_reporting,
                        "rejected_evaluation": evaluation,
                        "dagger_async_pending": (
                            collector.checkpoint_state() if async_pipeline else None),
                        "dagger_collection_requests": (
                            collector.window_provenance if async_pipeline else None),
                        "action_contract": action_contract_metadata(settings),
                        "training_config": config,
                        "dagger_replay": replay_store.checkpoint_metadata(
                            round_index
                        ),
                        "dagger_dynamic_sampler_state": dynamic_sampler_state,
                        "dagger_learning_progress_task_bank_state": (
                            None if task_bank is None else task_bank.state_dict()
                        ),
                        "rng_state": capture_rng_state(),
                        "numpy_rng_state": copy.deepcopy(rng.bit_generator.state),
                        "dagger_safe_state": {
                            "policy": safe_policy_state,
                            "optimizer": safe_optimizer_state,
                            "evaluation": safe_evaluation,
                            "reporting_evaluation": safe_reporting,
                        },
                    },
                ),
                step=environment_steps, metrics=safe_evaluation, rank=False,
            )
        else:
            if float(evaluation["selection_score"]) > safe_selection_score:
                safe_policy_state = {
                    name: value.detach().cpu().clone()
                    for name, value in policy.state_dict().items()
                }
                safe_optimizer_state = copy.deepcopy(optimizer.state_dict())
                safe_selection_score = float(evaluation["selection_score"])
                safe_evaluation = copy.deepcopy(evaluation)
                if reporting is not None:
                    safe_reporting = copy.deepcopy(reporting)
            manager.save(
                checkpoint_payload(
                    policy, normalizer, stage="dagger", track=",".join(tracks),
                    optimizer=optimizer,
                    extra={
                        "dynamics_target_mean": dynamics_mean,
                        "dynamics_target_std": dynamics_std,
                        "round": round_index,
                        "environment_steps": environment_steps,
                        "initial_checkpoint": str(initial_path),
                        "reporting_evaluation": reporting,
                        "dagger_async_pending": (
                            collector.checkpoint_state() if async_pipeline else None),
                        "dagger_collection_requests": (
                            collector.window_provenance if async_pipeline else None),
                        "action_contract": action_contract_metadata(settings),
                        "training_config": config,
                        "dagger_replay": replay_store.checkpoint_metadata(
                            round_index
                        ),
                        "dagger_dynamic_sampler_state": dynamic_sampler_state,
                        "dagger_learning_progress_task_bank_state": (
                            None if task_bank is None else task_bank.state_dict()
                        ),
                        "rng_state": capture_rng_state(),
                        "numpy_rng_state": copy.deepcopy(rng.bit_generator.state),
                        "dagger_safe_state": {
                            "policy": safe_policy_state,
                            "optimizer": safe_optimizer_state,
                            "evaluation": safe_evaluation,
                            "reporting_evaluation": safe_reporting,
                        },
                    },
                ),
                step=environment_steps, metrics=evaluation,
            )
        print(
            f"round={round_index}/{settings.get('rounds',8)} steps={environment_steps} "
            f"beta={beta:.3f} refresh={int(refresh_permanent_expert)} "
            f"labels={len(new_histories)} valid={train_metrics['teacher_valid_fraction']:.3f} "
            f"topo_valid={train_metrics['topology_target_valid_fraction']:.4f} "
            f"loss={train_metrics['loss']:.4f} "
            f"topology={train_metrics.get('topology_loss',0):.4f} "
            f"plan={train_metrics.get('plan_state_loss',0):.4f} "
            f"chunk={train_metrics.get('action_chunk_loss',0):.4f} "
            f"reward_aux={train_metrics.get('reward_aux_loss',0):.4f} "
            f"gate_nce={train_metrics.get('gate_contrastive_loss',0):.4f} "
            f"flow={train_metrics.get('flow_matching_loss',0):.4f} "
            f"shortcut={train_metrics.get('shortcut_bootstrap_loss',0):.4f} "
            f"endpoint={train_metrics.get('flow_endpoint_mse',0):.4f} "
            f"deploy={train_metrics.get('flow_deployment_endpoint_loss',0):.4f} "
            f"deploy_phys={train_metrics.get('flow_deployment_physical_loss',0):.4f} "
            f"deploy_ctbr_mae={train_metrics.get('flow_deployment_collective_mae_mps2',0):.2f}/"
            f"{train_metrics.get('flow_deployment_body_rate_mae_rps',0):.2f} "
            f"mode={train_metrics.get('mixture_mode_loss',0):.4f}/"
            f"{train_metrics.get('mixture_mode_accuracy',0):.3f} "
            f"mix_deploy={train_metrics.get('mixture_deployed_action_loss',0):.4f} "
            f"dyn={train_metrics.get('dynamics_loss',0):.4f} "
            f"group_max={train_metrics.get('group_loss_max',0):.4f} "
            f"anchor={train_metrics.get('anchor_loss',0):.4f} "
            f"events={train_metrics['new_event_label_fraction']:.3f} "
            f"occupancy={train_metrics['new_canonical_occupancy_fraction']:.2f}/"
            f"{train_metrics['new_cold_reset_occupancy_fraction']:.2f}/"
            f"{train_metrics['new_expert_prefix_occupancy_fraction']:.2f} "
            f"targeted_starts={train_metrics['targeted_start_episode_fraction']:.3f} "
            f"prefix_handoffs={collected.completed_expert_prefix_handoffs}/"
            f"{collected.scheduled_expert_prefix_episodes} "
            f"prefix_speed={train_metrics['expert_prefix_mean_handoff_speed_mps']:.1f} "
            f"speed={train_metrics['teacher_speed_command_min']:.1f}/"
            f"{train_metrics['teacher_speed_command_mean']:.1f}/"
            f"{train_metrics['teacher_speed_command_max']:.1f} "
            f"p1={evaluation.get('p1',0):.3f} "
            f"p2={evaluation.get('p2',0):.3f} p3={evaluation.get('p3',0):.3f} "
            f"full={evaluation['full_course_success']:.3f} crash={evaluation['crash_rate']:.3f} "
            f"steps_s={train_metrics['environment_steps_per_second']:.1f} "
            f"solve_ms={1000.0 * train_metrics['teacher_solver_seconds_per_query']:.2f} "
            f"teacher_ms={1000.0 * train_metrics['teacher_wall_seconds_per_query']:.2f} "
            f"env_ms={1000.0 * train_metrics['environment_step_seconds_per_query']:.2f} "
            f"feature_ms={1000.0 * train_metrics['feature_seconds_per_query']:.2f} "
            f"sampler={int(train_metrics.get('dynamic_sampler_active',0))}/"
            f"{train_metrics.get('dynamic_sampler_multiplier_min',1):.2f}/"
            f"{train_metrics.get('dynamic_sampler_multiplier_max',1):.2f} "
            f"mpcc_ms={1000.0 * train_metrics['controller_sync_seconds_per_query']:.1f}/"
            f"{1000.0 * train_metrics['controller_reference_seconds_per_query']:.1f}/"
            f"{1000.0 * train_metrics['controller_backend_seconds_per_query']:.1f}/"
            f"{1000.0 * train_metrics['controller_postprocess_seconds_per_query']:.1f} "
            f"worker_busy={train_metrics['collection_worker_busy_fraction']:.2f} "
            f"collect_s={collection_seconds:.1f} round_s={round_seconds:.1f} "
            f"rollback={int(rollback)}", flush=True,
        )
        # A full collection owns hundreds of thousands of temporary label rows.
        # Keeping the previous round's locals alive until the next assignments,
        # and leaving their freed arenas cached in glibc, makes long DAgger runs
        # grow until the host OOM killer intervenes.  The persistent replay
        # arrays above remain referenced; only consumed per-round buffers go.
        del (
            collected, new_histories, new_actions, new_previous, new_dynamics,
            new_dynamics_valid, new_successful, new_tracks, new_families,
            new_progress, new_gate_phases, new_gate_indices, new_teacher_modes,
            new_occupancy_modes, new_events, new_speed_commands,
            new_action_chunks, new_executed_actions, new_reward_components,
            new_topology_targets, new_topology_valid, new_groups,
            new_anchor_mask, online_keep, permanent_keep,
        )
        release_process_heap()
        if async_pipeline:
            collector.release_results()
    collector.close()
    atexit.unregister(collector.close)
    if evaluation_collector is not None:
        evaluation_collector.close()
        atexit.unregister(evaluation_collector.close)
    logger.finish()


def aggrevate_rollin_beta(
    round_index: int, settings: Mapping[str, Any],
) -> float:
    """Expert roll-in schedule ``alpha_n`` from the AggreVaTe algorithms."""

    start = float(settings.get("rollin_beta_start", 0.90))
    end = float(settings.get("rollin_beta_end", 0.10))
    decay_rounds = max(1, int(settings.get("rollin_beta_decay_rounds", 20)))
    if not 0.0 <= end <= start <= 1.0:
        raise ValueError("AggreVaTe roll-in beta must satisfy 0 <= end <= start <= 1")
    fraction = min(max((int(round_index) - 1) / decay_rounds, 0.0), 1.0)
    schedule = str(settings.get("rollin_beta_schedule", "linear"))
    if schedule == "linear":
        return start + fraction * (end - start)
    if schedule == "exponential":
        if start == 0.0:
            return 0.0
        return max(end, start * (max(end, 1.0e-8) / start) ** fraction)
    raise ValueError("AggreVaTe rollin_beta_schedule must be linear or exponential")


def aggrevate_route_conditioning_probe(
    policy: PrivilegedMLPPolicy,
    replay: AggreVaTeQueryBatch,
    device: str,
    *,
    maximum_rows: int = 256,
) -> dict[str, float]:
    """Probe route sensitivity on the fixed earliest cost-to-go queries.

    Magnitude alone is insufficient: an actor can become highly sensitive to
    future gates while turning in the wrong direction. We therefore match each
    row to the closest state/history from another track, replace only its route
    chain, and compare the induced policy delta with the expert-action delta.
    The earliest replay rows remain fixed as later rounds are appended, making
    the telemetry longitudinal rather than a moving-distribution statistic.
    """

    if policy.input_dim < 28 or (policy.input_dim - 16) % 12:
        return {}
    count = min(int(maximum_rows), len(replay.histories))
    if count < 2:
        return {}
    histories = np.asarray(replay.histories[:count], np.float32)
    expert = np.asarray(replay.expert_actions[:count], np.float32)
    tracks = np.asarray(replay.track_ids[:count], np.int64)
    speeds = np.asarray(replay.speed_commands[:count], np.float32)

    state = histories[:, :, :16].reshape(count, -1)
    state = (state - state.mean(0, keepdims=True)) / state.std(
        0, keepdims=True
    ).clip(1.0e-3)
    squared = np.square(state).sum(1)
    distances = squared[:, None] + squared[None, :] - 2.0 * state @ state.T
    distances[tracks[:, None] == tracks[None, :]] = np.inf
    partner = np.argmin(distances, axis=1)
    valid = np.isfinite(distances[np.arange(count), partner])
    if not np.any(valid):
        return {}

    was_training = policy.training
    policy.eval()
    inputs = torch.from_numpy(histories).to(device)
    commands = torch.from_numpy(speeds).to(device)
    partner_tensor = torch.as_tensor(partner, device=inputs.device)
    with torch.no_grad():
        baseline = policy(inputs, commands)
        swapped_inputs = inputs.clone()
        swapped_inputs[:, :, 16:] = inputs[partner_tensor, :, 16:]
        swapped = policy(swapped_inputs, commands)
    predicted_delta = (swapped - baseline).float().cpu().numpy()[valid]
    expert_delta = (expert[partner] - expert)[valid]
    predicted_norm = np.linalg.norm(predicted_delta, axis=1)
    expert_norm = np.linalg.norm(expert_delta, axis=1)
    cosine = np.sum(predicted_delta * expert_delta, axis=1) / np.maximum(
        predicted_norm * expert_norm, 1.0e-8
    )

    gradient_count = min(64, count)
    gradient_input = inputs[:gradient_count].detach().clone().requires_grad_(True)
    gradient_output = policy(gradient_input, commands[:gradient_count])
    gradient = torch.zeros_like(gradient_input)
    for dimension in range(4):
        component = torch.autograd.grad(
            gradient_output[:, dimension].sum(), gradient_input,
            retain_graph=dimension < 3,
        )[0]
        gradient.add_(component.square())
    saliency = gradient.sqrt().mean(dim=(0, 1))
    state_saliency = saliency[:16].mean()
    route_saliency = saliency[16:].mean()
    if was_training:
        policy.train()
    return {
        "route_probe/policy_action_mse": float(np.square(
            baseline.float().cpu().numpy() - expert
        ).mean()),
        "route_probe/action_rms_delta": float(
            np.sqrt(np.square(predicted_delta).mean())
        ),
        "route_probe/expert_action_rms_delta": float(
            np.sqrt(np.square(expert_delta).mean())
        ),
        "route_probe/delta_cosine": float(cosine.mean()),
        "route_probe/delta_cosine_positive_fraction": float(
            np.mean(cosine > 0.0)
        ),
        "route_probe/delta_magnitude_ratio": float(
            predicted_norm.mean() / max(expert_norm.mean(), 1.0e-8)
        ),
        "route_probe/route_state_jacobian_ratio": float(
            route_saliency / state_saliency.clamp_min(1.0e-12)
        ),
    }


def run_aggrevate(config: dict[str, Any], device: str) -> None:
    """Train a privileged actor with MPCC expert cost-to-go aggregation."""

    if "aggrevate" not in config:
        raise ValueError("configuration has no aggrevate section")
    settings = _merge(
        dict(config.get("dagger", {})), dict(config["aggrevate"])
    )
    algorithm = AggreVaTeConfig.from_mapping(settings.get("algorithm"))
    seed = int(settings.get("seed", 20260901))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    initial_checkpoint = settings.get("initial_checkpoint")
    if not initial_checkpoint:
        raise ValueError("AggreVaTe requires an initial privileged policy checkpoint")
        policy, normalizer, initial, initial_path = load_policy_checkpoint(
        initial_checkpoint, device,
        context_steps=(
            None if settings.get("context_steps_override") is None
            else int(settings["context_steps_override"])
        ),
    )
    if policy.action_head_type != "mlp":
        raise ValueError(
            "base AggreVaTe integration currently requires the exact direct MLP CTBR head"
        )
    if policy.action_mixture_mode_head is not None:
        raise ValueError("AggreVaTe direct-action baseline does not support a mixture head")
    critic_encoder_contract = str(settings.get(
        "critic_state_encoder", "frozen_initial"
    ))
    if critic_encoder_contract not in {"frozen_initial", "current_detached"}:
        raise ValueError(
            "critic_state_encoder must be frozen_initial or current_detached"
        )
    if bool(settings.get("value_gradient_through_encoder", False)):
        raise ValueError(
            "AggreVaTe may only differentiate Q through the policy action; "
            "differentiating through the critic state embedding lets the actor "
            "change the critic input instead of improving control"
        )
    critic_state_encoder = copy.deepcopy(policy).eval()
    for parameter in critic_state_encoder.parameters():
        parameter.requires_grad_(False)
    configured_contract = str(settings.get(
        "observation_contract", policy.observation_contract
    ))
    if configured_contract != policy.observation_contract:
        raise ValueError(
            "AggreVaTe observation contract differs from the policy checkpoint"
        )
    route_gate_count = int(settings.get(
        "route_gate_count", route_gates_from_feature_dim(policy.input_dim)
        if policy.observation_contract == LEGACY_OBSERVATION_CONTRACT else 6
    ))
    expected_input = privileged_feature_dim(
        route_gate_count, configured_contract
    )
    if expected_input != policy.input_dim:
        raise ValueError(
            f"AggreVaTe route/observation width mismatch: expected={expected_input} "
            f"checkpoint={policy.input_dim}"
        )
    tracks = configured_tracks(settings)
    stage = parse_stage(settings["curriculum"])
    evaluation_stage = parse_stage(
        settings.get("evaluation_curriculum", settings["curriculum"])
    )
    if set(stage.tracks) != set(tracks):
        raise ValueError("AggreVaTe curriculum and configured tracks differ")
    if algorithm.query_min_step >= stage.max_steps:
        raise ValueError("AggreVaTe minimum query step lies beyond the curriculum horizon")
    if algorithm.query_horizon < stage.max_steps:
        raise ValueError(
            "base AggreVaTe requires MPCC rollout through the full curriculum "
            "horizon; increase algorithm.query_horizon"
        )
    maximum_success_cost = stage.max_steps * algorithm.step_cost
    if algorithm.failure_base_cost < maximum_success_cost:
        raise ValueError(
            "AggreVaTe failure_base_cost must be at least max_steps*step_cost "
            "so failure cannot be cheaper than completion"
        )

    critic = AggreVaTeQCritic(
        policy.recurrent_dim,
        hidden_dim=algorithm.critic_hidden_dim,
        depth=algorithm.critic_depth,
        ensemble=algorithm.critic_ensemble,
        speed_scale=policy.speed_conditioning_scale,
    ).to(device)
    actor_optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(settings.get("actor_learning_rate", 2.0e-5)),
        weight_decay=float(settings.get("actor_weight_decay", 1.0e-5)),
        fused=device.startswith("cuda"),
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=float(settings.get("critic_learning_rate", 2.0e-4)),
        weight_decay=float(settings.get("critic_weight_decay", 1.0e-5)),
        fused=device.startswith("cuda"),
    )
    replay = AggreVaTeReplayBuffer(int(settings.get("replay_capacity", 20000)))
    imitation_replay = AggreVaTeImitationReplayBuffer(int(settings.get(
        "hybrid_imitation_replay_capacity", 16000
    )))
    hybrid_all_step_labels = bool(
        algorithm.hybrid
        and settings.get("hybrid_collect_all_step_labels", True)
    )
    cost_normalizer: CostNormalizer | None = None
    completed_round = 0
    environment_steps = 0
    resume_checkpoint = settings.get("resume_checkpoint")
    if resume_checkpoint:
        resume_path = resolve_ranked_checkpoint(resume_checkpoint)
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if str(resume.get("stage")) != "aggrevate":
            raise ValueError("AggreVaTe resume checkpoint has the wrong stage")
        policy.load_state_dict(resume["model"])
        critic.load_state_dict(resume["aggrevate_critic"])
        if resume.get("aggrevate_critic_state_encoder") is not None:
            critic_state_encoder.load_state_dict(
                resume["aggrevate_critic_state_encoder"]
            )
        actor_optimizer.load_state_dict(resume["optimizer"])
        critic_optimizer.load_state_dict(resume["aggrevate_critic_optimizer"])
        replay = AggreVaTeReplayBuffer.from_state_dict(resume["aggrevate_replay"])
        imitation_state = resume.get("aggrevate_imitation_replay")
        if imitation_state is not None:
            imitation_replay = AggreVaTeImitationReplayBuffer.from_state_dict(
                imitation_state
            )
        normalizer_state = resume.get("aggrevate_cost_normalizer")
        cost_normalizer = (
            None if normalizer_state is None
            else CostNormalizer.from_state_dict(normalizer_state)
        )
        completed_round = int(resume["round"])
        environment_steps = int(resume.get("environment_steps", 0))
        restore_rng_state(resume.get("rng_state"))
        if resume.get("numpy_rng_state") is not None:
            rng.bit_generator.state = copy.deepcopy(resume["numpy_rng_state"])

    manager = CheckpointManager.from_config(config)
    logger = init_wandb(config)
    collector = ProcessDaggerCollector(policy, normalizer, settings, stage, device)
    atexit.register(collector.close)
    amp = bool(settings.get("amp", True)) and device.startswith("cuda")

    def payload(round_index: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        result = checkpoint_payload(
            policy, normalizer, stage="aggrevate", track=",".join(tracks),
            optimizer=actor_optimizer,
            extra={
                "dynamics_target_mean": np.asarray(
                    initial["dynamics_target_mean"], np.float32
                ),
                "dynamics_target_std": np.asarray(
                    initial["dynamics_target_std"], np.float32
                ),
                "round": int(round_index),
                "environment_steps": int(environment_steps),
                "initial_checkpoint": str(initial_path),
                "aggrevate_algorithm": dict(algorithm.__dict__),
                "aggrevate_critic": critic.state_dict(),
                "aggrevate_critic_state_encoder": (
                    critic_state_encoder.state_dict()
                ),
                "aggrevate_critic_state_encoder_contract": (
                    critic_encoder_contract
                ),
                "aggrevate_critic_optimizer": critic_optimizer.state_dict(),
                "aggrevate_replay": replay.state_dict(),
                "aggrevate_imitation_replay": (
                    imitation_replay.state_dict()
                    if hybrid_all_step_labels else None
                ),
                "aggrevate_cost_normalizer": (
                    None if cost_normalizer is None
                    else cost_normalizer.state_dict()
                ),
                "training_config": config,
                "rng_state": capture_rng_state(),
                "numpy_rng_state": copy.deepcopy(rng.bit_generator.state),
                "metrics": dict(metrics),
            },
        )
        return result

    baseline = evaluate_policy(
        policy, normalizer, settings, evaluation_stage,
        count=int(settings.get("evaluation_episodes", 48)),
        seed_base=int(settings.get("evaluation_seed", seed + 10000)),
        device=device,
    )
    logger.log_eval(baseline, environment_steps)
    if completed_round == 0:
        manager.save(payload(0, baseline), step=0, metrics=baseline)
    print(
        f"aggrevate actor={initial_path} variant={algorithm.variant} "
        f"baseline_full={baseline['full_course_success']:.3f} "
        f"baseline_crash={baseline['crash_rate']:.3f}", flush=True,
    )

    rounds = int(settings.get("rounds", 20))
    critic_batch_size = int(settings.get("critic_batch_size", 512))
    actor_batch_size = int(settings.get("actor_batch_size", 512))
    critic_updates = int(settings.get("critic_updates_per_round", 400))
    actor_updates = int(settings.get("actor_updates_per_round", 200))
    bootstrap_probability = float(settings.get(
        "critic_bootstrap_probability", 0.80
    ))
    if not 0.0 < bootstrap_probability <= 1.0:
        raise ValueError("critic_bootstrap_probability must be in (0,1]")
    freeze_normalizer_round = int(settings.get(
        "freeze_cost_normalizer_after_round", 1
    ))
    if freeze_normalizer_round < 1:
        raise ValueError("freeze_cost_normalizer_after_round must be positive")

    for round_index in range(completed_round + 1, rounds + 1):
        started = time.perf_counter()
        rollin_beta = aggrevate_rollin_beta(round_index, settings)
        query_batch, imitation_batch, collection_metrics = (
            collector.collect_aggrevate(
            queries=int(settings.get("queries_per_round", 96)),
            rollin_beta=rollin_beta,
            seed_base=seed + round_index * 100000,
            config=algorithm,
            maximum_attempt_multiplier=int(settings.get(
                "query_maximum_attempt_multiplier", 4
            )),
            )
        )
        replay.append(query_batch)
        if hybrid_all_step_labels:
            retained_rows = int(settings.get(
                "hybrid_imitation_rows_per_round", 12000
            ))
            retained_rows = min(retained_rows, len(imitation_batch.histories))
            retained = balanced_choice(
                rng,
                np.arange(len(imitation_batch.histories), dtype=np.int64),
                imitation_batch.track_ids,
                retained_rows,
            )
            imitation_replay.append(AggreVaTeImitationBatch(**{
                name: np.asarray(getattr(imitation_batch, name))[retained]
                for name in imitation_batch.__dataclass_fields__
            }))
        environment_steps += int(collection_metrics["environment_steps"])
        replay_arrays = replay.arrays()
        if cost_normalizer is None or round_index <= freeze_normalizer_round:
            cost_normalizer = CostNormalizer.fit(replay_arrays.costs_to_go)

        critic.train()
        policy.eval()
        critic_metrics: dict[str, list[float]] = {}
        for _ in range(critic_updates):
            sample = replay.sample_stratified(
                rng, critic_batch_size, include_query_mode=True,
            )
            histories = torch.from_numpy(sample.histories.astype(np.float32)).to(device)
            actions = torch.from_numpy(sample.query_actions).to(device)
            remaining = torch.from_numpy(sample.remaining_fractions).to(device)
            speeds = torch.from_numpy(sample.speed_commands).to(device)
            targets = torch.from_numpy(sample.costs_to_go).to(device)
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=amp
            ):
                latent = (
                    critic_state_encoder.encode(histories)
                    if critic_encoder_contract == "frozen_initial"
                    else policy.encode(histories)
                )
            targets = cost_normalizer.normalize(targets).clamp(
                -algorithm.critic_target_clip, algorithm.critic_target_clip
            )
            mask = torch.from_numpy(
                (rng.random((len(histories), algorithm.critic_ensemble))
                 < bootstrap_probability).astype(np.float32)
            ).to(device)
            # Every member and every row must supervise at least one estimate.
            mask[:, 0] = 1.0
            mask[0, :] = 1.0
            critic_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                estimates = critic(latent.detach(), actions, remaining, speeds)
                critic_loss, pieces = aggrevate_critic_loss(
                    estimates.float(), targets.float(),
                    huber_beta=algorithm.critic_huber_beta,
                    bootstrap_mask=mask,
                )
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), float(settings.get("critic_gradient_clip", 5.0))
            )
            critic_optimizer.step()
            for name, value in pieces.items():
                critic_metrics.setdefault(name, []).append(float(value))

        critic.eval()
        policy.train()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
        actor_metrics: dict[str, list[float]] = {}
        value_algorithm = (
            replace(algorithm, behavior_cloning_weight=0.0)
            if hybrid_all_step_labels else algorithm
        )
        hybrid_objective_settings = dict(settings)
        hybrid_objective_settings.update(
            topology_weight=0.0,
            action_chunk_weight=0.0,
            dagger_group_robust_weight=0.0,
            dagger_anchor_distillation_weight=0.0,
        )
        dynamics_contract = str(settings.get(
            "dynamics_target_contract", "task_delta_v1"
        ))
        dynamics_mean = np.asarray(initial["dynamics_target_mean"], np.float32)
        dynamics_std = np.asarray(initial["dynamics_target_std"], np.float32)
        teacher_noise = torch.as_tensor(
            algorithm.teacher_noise_std, device=device, dtype=torch.float32
        )
        actor_reference_weight = float(settings.get(
            "actor_reference_weight", 0.0
        ))
        actor_reference_huber_beta = float(settings.get(
            "actor_reference_huber_beta", 0.05
        ))
        if actor_reference_weight < 0.0 or actor_reference_huber_beta <= 0.0:
            raise ValueError("invalid AggreVaTe actor reference regularization")
        for _ in range(actor_updates):
            sample = replay.sample_stratified(
                rng, actor_batch_size, include_query_mode=True,
            )
            histories = torch.from_numpy(sample.histories.astype(np.float32)).to(device)
            expert = torch.from_numpy(sample.expert_actions).to(device)
            sampled = torch.from_numpy(sample.query_actions).to(device)
            remaining = torch.from_numpy(sample.remaining_fractions).to(device)
            speeds = torch.from_numpy(sample.speed_commands).to(device)
            actor_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=amp
            ):
                critic_latent = (
                    critic_state_encoder.encode(histories)
                    if critic_encoder_contract == "frozen_initial"
                    else policy.encode(histories)
                )
                reference_actions = critic_state_encoder(histories, speeds)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                actions = policy(histories, speeds)
                candidate_noise = (
                    torch.randn(
                        len(histories), len(algorithm.candidate_noise_scales), 4,
                        device=device, dtype=actions.dtype,
                    ) * teacher_noise.to(actions)[None, None]
                )
                actor_loss, pieces = aggrevate_actor_loss(
                    policy_actions=actions,
                    latent=critic_latent.detach(),
                    critic=critic,
                    expert_actions=expert,
                    sampled_actions=sampled,
                    remaining_fraction=remaining,
                    speed_command=speeds,
                    config=value_algorithm,
                    candidate_noise=candidate_noise,
                )
                reference_loss = F.smooth_l1_loss(
                    actions, reference_actions,
                    beta=actor_reference_huber_beta,
                )
                actor_loss = (
                    actor_loss + actor_reference_weight * reference_loss
                )
                pieces = dict(pieces)
                pieces.update({
                    "actor_loss": actor_loss.detach(),
                    "actor_reference_loss": reference_loss.detach(),
                    "actor_reference_rms": (
                        actions - reference_actions
                    ).float().square().mean().sqrt().detach(),
                })
                if hybrid_all_step_labels:
                    imitation_sample = imitation_replay.sample(
                        rng, actor_batch_size
                    )
                    imitation_histories = torch.from_numpy(
                        imitation_sample.histories.astype(np.float32)
                    ).to(device)
                    imitation_actions = torch.from_numpy(
                        imitation_sample.expert_actions
                    ).to(device)
                    imitation_previous = torch.from_numpy(
                        imitation_sample.previous_actions
                    ).to(device)
                    imitation_dynamics_array = imitation_sample.dynamics
                    if dynamics_contract == "task_delta_v1":
                        imitation_dynamics_array = (
                            (imitation_dynamics_array - dynamics_mean)
                            / dynamics_std
                        ).astype(np.float32)
                    imitation_dynamics = torch.from_numpy(
                        imitation_dynamics_array
                    ).to(device)
                    imitation_valid = torch.from_numpy(
                        imitation_sample.dynamics_valid
                    ).to(device)
                    imitation_speed = torch.from_numpy(
                        imitation_sample.speed_commands
                    ).to(device)
                    hybrid_loss, hybrid_pieces = imitation_loss(
                        policy,
                        imitation_histories,
                        imitation_actions,
                        imitation_previous,
                        imitation_dynamics,
                        hybrid_objective_settings,
                        imitation_valid,
                        imitation_speed,
                    )
                    actor_loss = (
                        actor_loss
                        + algorithm.behavior_cloning_weight * hybrid_loss
                    )
                    pieces = dict(pieces)
                    pieces.update({
                        "actor_loss": actor_loss.detach(),
                        "hybrid_imitation_loss": hybrid_loss.detach(),
                        "hybrid_action_loss": hybrid_pieces["action_loss"],
                        "hybrid_dynamics_loss": hybrid_pieces["dynamics_loss"],
                    })
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float(settings.get("actor_gradient_clip", 1.0))
            )
            actor_optimizer.step()
            for name, value in pieces.items():
                actor_metrics.setdefault(name, []).append(float(value))
        for parameter in critic.parameters():
            parameter.requires_grad_(True)

        policy.eval()
        route_probe = (
            aggrevate_route_conditioning_probe(
                policy, replay.arrays(), device,
                maximum_rows=int(settings.get("route_probe_rows", 256)),
            )
            if bool(settings.get("route_conditioning_probe", True)) else {}
        )
        evaluation = evaluate_policy(
            policy, normalizer, settings, evaluation_stage,
            count=int(settings.get("evaluation_episodes", 48)),
            seed_base=int(settings.get("evaluation_seed", seed + 10000)),
            device=device,
        )
        round_seconds = time.perf_counter() - started
        metrics = {
            "round": float(round_index),
            "rollin_beta": float(rollin_beta),
            "replay_queries": float(len(replay)),
            "hybrid_imitation_replay": float(len(imitation_replay)),
            "cost_normalizer_mean": cost_normalizer.mean,
            "cost_normalizer_std": cost_normalizer.std,
            "round_seconds": round_seconds,
            "environment_steps": float(environment_steps),
            **{f"collection/{name}": float(value)
               for name, value in collection_metrics.items()},
            **{name: float(np.mean(values))
               for name, values in critic_metrics.items()},
            **{name: float(np.mean(values))
               for name, values in actor_metrics.items()},
            **route_probe,
            **_memory_snapshot(),
        }
        logger.log_train(metrics, environment_steps)
        logger.log_eval(evaluation, environment_steps)
        manager.save(
            payload(round_index, evaluation),
            step=environment_steps, metrics=evaluation,
        )
        print(
            f"aggrevate_round={round_index}/{rounds} variant={algorithm.variant} "
            f"queries={len(query_batch.histories)} replay={len(replay)} "
            f"beta={rollin_beta:.3f} cost={collection_metrics['cost_mean']:.3f}/"
            f"{collection_metrics['cost_std']:.3f} "
            f"critic={metrics.get('critic_loss',0):.4f} "
            f"bias={metrics.get('critic_bias',0):.4f} "
            f"uncertainty={metrics.get('critic_ensemble_std',0):.4f} "
            f"actor={metrics.get('actor_loss',0):.4f} "
            f"adv={metrics.get('expert_advantage',0):.4f} "
            f"q={metrics.get('policy_q',0):.3f}/"
            f"{metrics.get('expert_q',0):.3f} "
            f"shift={metrics.get('actor_reference_rms',0):.3f} "
            f"route={metrics.get('route_probe/delta_magnitude_ratio',0):.3f}/"
            f"{metrics.get('route_probe/delta_cosine',0):.3f} "
            f"full={evaluation['full_course_success']:.3f} "
            f"crash={evaluation['crash_rate']:.3f} "
            f"round_s={round_seconds:.1f}", flush=True,
        )
    collector.close()
    atexit.unregister(collector.close)
    logger.finish()


PRIVILEGED_CRITIC_EPISODE_DIM = 4
PRIVILEGED_CRITIC_COURSE_DIM = 10
PRIVILEGED_CRITIC_ROUTE_DIM = 14
PRIVILEGED_COURSE_CRITIC_ARCHITECTURES = {
    "privileged_course_fusion_v1", "privileged_transformer_v2",
}
_PRIVILEGED_CRITIC_COURSE_CACHE: dict[str, np.ndarray] = {}


def privileged_critic_course_features(
    env: FlightmareEnv, *, target_speed: float,
    target_gates: int, completed_gates: int,
) -> np.ndarray:
    """Return dynamic progress plus cached global course geometry."""

    fingerprint = str(env.track.fingerprint)
    static = _PRIVILEGED_CRITIC_COURSE_CACHE.get(fingerprint)
    if static is None:
        gates = env.track.gates
        positions = np.stack([gate.position for gate in gates]).astype(np.float32)
        segments = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        course_length = float(segments.sum())
        if env.track.loop and len(positions) > 1:
            course_length += float(np.linalg.norm(positions[0] - positions[-1]))
        spans = np.asarray(
            env.track.bounds[:, 1] - env.track.bounds[:, 0], np.float32,
        )
        static = np.asarray([
            float(len(gates)) / 32.0,
            float(env.track.loop),
            course_length / 300.0,
            float(np.ptp(positions[:, 2])) / 20.0,
            float(spans[0]) / 80.0,
            float(spans[1]) / 80.0,
            float(spans[2]) / 30.0,
        ], np.float32)
        _PRIVILEGED_CRITIC_COURSE_CACHE[fingerprint] = static
    remaining = max(int(target_gates) - int(completed_gates), 1)
    return np.concatenate([
        np.asarray([
            float(target_speed) / 25.0,
            float(target_gates) / 32.0,
            float(remaining) / 32.0,
        ], np.float32),
        static,
    ])


def ppo_privileged_critic_context(
    env: FlightmareEnv, state: np.ndarray, episode_context: np.ndarray, *, target_speed: float,
    target_gates: int, completed_gates: int, settings: Mapping[str, Any],
    course_id: int | None = None,
) -> np.ndarray:
    """Give the asymmetric critic the future course, never the actor.

    Route records are vehicle-relative and normalized to stable physical scales.
    The final scalar is an exact integer embedding index derived from the stable
    course fingerprint, so it survives process boundaries and checkpoint resume.
    """
    episode = np.asarray(episode_context, np.float32)
    if episode.shape != (PRIVILEGED_CRITIC_EPISODE_DIM,):
        raise ValueError("critic episode context must contain four values")
    architecture = str(settings.get("critic_architecture", "mlp"))
    if architecture not in PRIVILEGED_COURSE_CRITIC_ARCHITECTURES:
        return episode
    vocabulary = int(settings.get("critic_course_vocab_size", 4096))
    if vocabulary < 2:
        raise ValueError("privileged critic course vocabulary is invalid")
    state = np.asarray(state, np.float32)
    if state.shape != (25,):
        raise ValueError("privileged critic requires the 25-value truth state")
    course = privileged_critic_course_features(
        env, target_speed=target_speed, target_gates=target_gates,
        completed_gates=completed_gates,
    )
    identity = (
        int(env.track.fingerprint[:16], 16) % vocabulary
        if course_id is None else int(course_id)
    )
    if not 0 <= identity < vocabulary:
        raise ValueError(
            f"critic course ID {identity} exceeds vocabulary {vocabulary}"
        )
    if architecture == "privileged_course_fusion_v1":
        result = np.concatenate([
            episode, course, np.asarray([identity], np.float32),
        ]).astype(np.float32)
        expected = (
            PRIVILEGED_CRITIC_EPISODE_DIM + PRIVILEGED_CRITIC_COURSE_DIM + 1
        )
        if result.shape != (expected,) or not np.all(np.isfinite(result)):
            raise ValueError("privileged course critic context is invalid")
        return result

    count = int(settings.get("critic_route_tokens", 24))
    if count < 1:
        raise ValueError("privileged critic route size is invalid")
    remaining = max(int(target_gates) - int(completed_gates), 1)
    preview = env.tracker.relative_gates(
        state[:3], state[3:7], count, remaining=remaining,
    )
    valid = (np.arange(count) < remaining).astype(np.float32)
    route = np.concatenate([
        preview["position"] / float(settings.get("critic_route_position_scale", 30.0)),
        preview["normal"],
        preview["up"],
        preview["size"] / float(settings.get("critic_gate_size_scale", 3.0)),
        preview["distance"][:, None]
        / float(settings.get("critic_route_position_scale", 30.0)),
        preview["enter_from_opposite_side"][:, None].astype(np.float32),
        valid[:, None],
    ], axis=-1).astype(np.float32)
    route[valid == 0.0, :-1] = 0.0
    result = np.concatenate([
        episode, course, route.reshape(-1),
        np.asarray([identity], np.float32),
    ]).astype(np.float32)
    expected = (
        PRIVILEGED_CRITIC_EPISODE_DIM + PRIVILEGED_CRITIC_COURSE_DIM
        + count * PRIVILEGED_CRITIC_ROUTE_DIM + 1
    )
    if result.shape != (expected,) or not np.all(np.isfinite(result)):
        raise ValueError("privileged critic context has an invalid shape or value")
    return result


def critic_input(
    normalized_history: np.ndarray,
    slot: RolloutSlot,
    target_speed: float,
    *,
    include_history: bool = False,
    include_episode_context: bool = True,
    settings: Mapping[str, Any] | None = None,
    course_id: int | None = None,
) -> np.ndarray:
    gates = slot.env.tracker.passed_count - slot.start_passed
    state = np.asarray(slot.observation["state"], np.float32)
    context = np.asarray([
        gates / max(slot.target_gates, 1),
        slot.steps / max(slot.max_steps, 1),
        slot.env.tracker.index / max(len(slot.env.track.gates) - 1, 1),
        np.linalg.norm(state[7:10]) / max(target_speed, 1.0e-3),
    ], np.float32)
    actor_state = (
        normalized_history.reshape(-1)
        if include_history else normalized_history[-1]
    )
    critic_settings = {} if settings is None else settings
    if str(critic_settings.get(
        "critic_architecture", "mlp"
    )) in PRIVILEGED_COURSE_CRITIC_ARCHITECTURES:
        if not include_history or not include_episode_context:
            raise ValueError(
                "privileged Transformer critic requires history and episode context"
            )
        context = ppo_privileged_critic_context(
            slot.env, state, context, target_speed=target_speed,
            target_gates=slot.target_gates, completed_gates=gates,
            settings=critic_settings, course_id=course_id,
        )
    return (
        np.concatenate([actor_state, context]).astype(np.float32)
        if include_episode_context else np.asarray(actor_state, np.float32)
    )


def finish_gae(transitions: list[dict[str, Any]], gamma: float, lam: float) -> None:
    advantage = 0.0
    for item in reversed(transitions):
        delta = item["reward"] + item["discount"] * item["next_value"] - item["value"]
        advantage = delta + item.get("gae_discount", item["discount"] * lam) * advantage
        item["advantage"] = float(advantage)
        item["return"] = float(advantage + item["value"])


def finish_failure_returns(
    transitions: list[dict[str, Any]], *, failed: bool, discount: float,
) -> None:
    """Attach a terminal failure cost return for constrained policy gradients."""

    cost_return = float(bool(failed))
    for item in reversed(transitions):
        item["failure_return"] = cost_return
        cost_return *= float(discount)


def completed_episode_is_terminal(
    done: bool, result: Mapping[str, Any] | None,
    *, timeout_is_terminal: bool = False,
) -> bool:
    """Distinguish true course/crash terminals from collector time limits."""

    return bool(
        done and result is not None and (
            bool(result["crashed"])
            or int(result["gates"]) >= int(result["target_gates"])
            or (timeout_is_terminal and bool(result.get('timed_out',False)))
        )
    )


def normalize_advantages_by_track(
    advantages: torch.Tensor, track_ids: torch.Tensor,
) -> torch.Tensor:
    """Normalize each task independently before forming a shared PPO loss."""

    normalized = torch.empty_like(advantages)
    for track_id in torch.unique(track_ids, sorted=True):
        mask = track_ids == track_id
        values = advantages[mask]
        normalized[mask] = (
            values - values.mean()
        ) / values.std(unbiased=False).clamp_min(1.0e-6)
    return normalized


def normalize_ppo_advantages(
    advantages: torch.Tensor,
    strata_ids: torch.Tensor,
    *,
    per_track: bool,
) -> torch.Tensor:
    """Apply either task-local or conventional joint PPO normalization."""

    groups = strata_ids if per_track else torch.zeros_like(strata_ids)
    return normalize_advantages_by_track(advantages, groups)


def ppo_critic_loss_scales(
    returns: torch.Tensor, track_ids: torch.Tensor, *, per_track: bool,
    minimum_global_fraction: float = 0.1,
) -> torch.Tensor:
    """Balance value regression across courses without changing raw outputs.

    The critic must still emit values in environment-return units for GAE.
    Scaling only its residual gives courses with different reward/lap-time
    variance comparable gradient influence, while the course embedding learns
    each task's raw offset and shape.
    """

    if returns.ndim != 1 or track_ids.shape != returns.shape:
        raise ValueError("critic returns and track IDs must be aligned vectors")
    if not 0.0 <= float(minimum_global_fraction) <= 1.0:
        raise ValueError("critic scale floor fraction must lie in [0,1]")
    global_scale = returns.std(unbiased=False).clamp_min(1.0e-3)
    if not per_track:
        return global_scale
    _, inverse = torch.unique(track_ids.long(), sorted=True, return_inverse=True)
    group_count = int(inverse.max()) + 1
    counts = torch.bincount(inverse, minlength=group_count).to(returns.dtype)
    sums = returns.new_zeros(group_count).scatter_add_(0, inverse, returns)
    means = sums / counts.clamp_min(1.0)
    centered = returns - means[inverse]
    variances = returns.new_zeros(group_count).scatter_add_(
        0, inverse, centered.square(),
    ) / counts.clamp_min(1.0)
    floor = global_scale * float(minimum_global_fraction)
    return variances.clamp_min(0.0).sqrt().clamp_min(floor)[inverse]


def configured_ppo_exploration_correlation(
    settings: Mapping[str, Any],
) -> float:
    """Return the AR(1) correlation used for trajectory-coherent exploration."""

    correlation = float(settings.get("ppo_exploration_correlation", 0.0))
    if not np.isfinite(correlation) or not 0.0 <= correlation < 1.0:
        raise ValueError("ppo_exploration_correlation must lie in [0,1)")
    return correlation


def validated_exploration_log_std(value, parameter, minimum, maximum):
    """Reject invalid scales and silent policy-clamp changes to a recipe."""
    result = torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype)
    if result.ndim == 0:
        result = result.expand_as(parameter)
    if result.shape != parameter.shape:
        raise ValueError('exploration_log_std must be scalar or contain four values')
    if not bool(torch.isfinite(result).all()):
        raise ValueError('exploration_log_std must be finite')
    if bool(((result < minimum) | (result > maximum)).any()):
        raise ValueError('exploration_log_std is outside policy bounds and would be silently clamped')
    return result


def correlated_ppo_distribution(
    base: SquashedGaussian,
    exploration_offset: torch.Tensor,
    correlation: float,
) -> SquashedGaussian:
    """Construct the Gaussian conditional on a supplied exploration offset.

    With fixed scale, the stationary AR residual standard deviation is the
    configured scale; zero-initialized episode residuals have a startup transient.
    Each innovation is scaled by ``sqrt(1-rho^2)``.
    The previous residual is collector state and is treated as fixed when PPO
    scores the conditional action. Matching stored likelihoods verifies this
    conditional calculation, NOT the complete history-marginalized likelihood
    under a changed policy. New temporal exploration designs must also audit
    noise-state evolution and critic conditioning. Current rl.2/rl.3 use rho=0.
    """

    correlation = float(correlation)
    if not np.isfinite(correlation) or not 0.0 <= correlation < 1.0:
        raise ValueError("PPO exploration correlation must lie in [0,1)")
    offset = exploration_offset.to(
        device=base.location.device, dtype=base.location.dtype,
    )
    if offset.shape != base.location.shape:
        raise ValueError("PPO exploration offset must match action location")
    innovation_log_scale = 0.5 * math.log1p(-(correlation * correlation))
    return SquashedGaussian(
        base.location + offset,
        base.log_std + innovation_log_scale,
    )


def squashed_gaussian_log_prob_per_dim(
    location: torch.Tensor, log_std: torch.Tensor,
    action: torch.Tensor, raw: torch.Tensor,
) -> torch.Tensor:
    """Per-dimension terms of ``SquashedGaussian.log_prob`` (same formula)."""

    clipped = action.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    base = -0.5 * ((raw - location) / log_std.exp()).square()
    base = base - log_std - 0.5 * np.log(2.0 * np.pi)
    return base - torch.log(1.0 - clipped.square() + 1.0e-6)


def saturation_masked_log_ratio(
    distribution: "SquashedGaussian", actions: torch.Tensor, raw_actions: torch.Tensor,
    old_location: torch.Tensor, old_log_std: torch.Tensor, threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop saturated action dimensions from the PPO likelihood ratio.

    When the pre-squash mean sits deep in the tanh tail the executed command
    is the same for every noise draw, so the sampled innovation carries no
    information about the outcome. Its score-function term is zero-mean noise
    that only inflates gradient variance. Masking those dimensions (per
    sample, per axis) removes that noise without touching the others.
    Returns the masked log ratio and the masked fraction of dimensions.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError("saturation mask threshold must lie in (0,1)")
    new = squashed_gaussian_log_prob_per_dim(
        distribution.location, distribution.log_std, actions, raw_actions)
    old = squashed_gaussian_log_prob_per_dim(
        old_location.to(actions.dtype), old_log_std.to(actions.dtype), actions, raw_actions)
    keep = (actions.abs() <= threshold).to(new.dtype)
    log_ratio = ((new - old) * keep).sum(-1)
    return log_ratio, 1.0 - keep.mean()


def persistent_ppo_task_indices(
    track_count: int,
    slot_count: int,
    weights: np.ndarray,
    *,
    require_unique: bool,
    balance_tasks: bool,
    rng: np.random.Generator,
    floor_before_weighting: bool = True,
) -> np.ndarray:
    """Assign immutable PPO tasks to simulator slots.

    Green et al. keep one task alive in each environment. A static control
    often has fewer distinct tasks than useful rollout workers, so balanced
    replication gives every task the same trajectory count in every joint
    update instead of accepting multinomial composition noise once at launch.
    """

    track_count = int(track_count)
    slot_count = int(slot_count)
    if track_count < 1 or slot_count < 1:
        raise ValueError("persistent PPO assignment needs tracks and slots")
    if require_unique and slot_count > track_count:
        raise ValueError("persistent PPO assignment cannot make unique replicas")
    probabilities = np.asarray(weights, np.float64)
    if probabilities.shape != (track_count,):
        raise ValueError("persistent PPO weights do not match track count")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
        raise ValueError("persistent PPO weights must be finite and positive")
    probabilities = probabilities / probabilities.sum()
    if balance_tasks and not require_unique:
        # Deterministic largest-remainder allocation preserves an intentional
        # anchor/interior/frontier mixture while guaranteeing every active
        # task at least one persistent simulator whenever slots permit.  The
        # old implementation silently discarded non-uniform weights here.
        if slot_count < track_count:
            return np.asarray(rng.choice(
                track_count, size=slot_count, replace=False, p=probabilities,
            ), np.int64)
        # Historical mode allocates one slot first, diluting requested anchor
        # weights as the archive grows. Opt-in proportional mode apportions
        # the entire slot budget; callers must bound their active archive.
        counts = np.ones(track_count, np.int64) if floor_before_weighting else np.zeros(track_count, np.int64)
        remaining = slot_count - int(counts.sum())
        if remaining:
            raw = remaining * probabilities
            extra = np.floor(raw).astype(np.int64)
            counts += extra
            residual = remaining - int(extra.sum())
            order = np.argsort(-(raw - extra), kind="stable")
            counts[order[:residual]] += 1
        selected = np.repeat(np.arange(track_count, dtype=np.int64), counts)
        rng.shuffle(selected)
        return selected
    return np.asarray(rng.choice(
        track_count,
        size=slot_count,
        replace=not require_unique,
        p=probabilities,
    ), np.int64)


def stratified_minibatches(
    track_ids: torch.Tensor, batch_size: int,
    track_weights: Mapping[int, float] | None = None,
    maximum_batches: int | None = None,
) -> list[torch.Tensor]:
    """Return shuffled minibatches with a configured quota for every track.

    Shorter tracks are reshuffled and sampled again only after all of their
    transitions have been used. The default remains equal-track balancing;
    weighted PPO can deliberately retain a reference-course family without
    allowing different episode lengths to choose the effective mixture.
    """

    unique_tracks = torch.unique(track_ids, sorted=True)
    if len(unique_tracks) == 0:
        return []
    effective_size = max(int(batch_size), len(unique_tracks))
    weights = np.asarray([
        float((track_weights or {}).get(int(track_id), 1.0))
        for track_id in unique_tracks.tolist()
    ], np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("stratified PPO weights must be finite and positive")
    quotas = np.ones(len(unique_tracks), np.int64)
    remaining = effective_size - len(unique_tracks)
    if remaining:
        raw = remaining * weights / weights.sum()
        extra = np.floor(raw).astype(np.int64)
        quotas += extra
        residual = remaining - int(extra.sum())
        order = np.argsort(-(raw - extra), kind="stable")
        quotas[order[:residual]] += 1
    groups = [
        torch.nonzero(track_ids == track_id, as_tuple=False).flatten()
        for track_id in unique_tracks
    ]
    if any(len(group) == 0 for group in groups):
        raise ValueError("stratified PPO requires transitions from every track")
    batch_count = max(
        math.ceil(len(group) / int(quota))
        for group, quota in zip(groups, quotas)
    )
    if maximum_batches is not None:
        if int(maximum_batches) < 1:
            raise ValueError("maximum PPO minibatches must be positive")
        batch_count = min(batch_count, int(maximum_batches))
    draws: list[torch.Tensor] = []
    for group, quota in zip(groups, quotas):
        required = batch_count * int(quota)
        pieces: list[torch.Tensor] = []
        remaining = required
        while remaining > 0:
            shuffled = group[torch.randperm(len(group), device=group.device)]
            pieces.append(shuffled[:remaining])
            remaining -= min(remaining, len(shuffled))
        draws.append(torch.cat(pieces))
    batches: list[torch.Tensor] = []
    for batch_index in range(batch_count):
        indices = torch.cat([
            draw[
                batch_index * int(quota):(batch_index + 1) * int(quota)
            ]
            for draw, quota in zip(draws, quotas)
        ])
        batches.append(indices[torch.randperm(
            len(indices), device=indices.device,
        )])
    return batches


def ppo_minibatch_strata(
    track_ids: torch.Tensor,
    rollout_laps: torch.Tensor | None,
    track_weights: Mapping[int, float] | None,
    lap_weights: Mapping[int, float] | None,
) -> tuple[torch.Tensor, dict[int, float] | None]:
    """Build joint track/lap strata so long episodes cannot choose the loss.

    The observed track/lap combinations become sampling strata. Configured lap
    weights set their relative quota, while the existing track weights retain
    the intended course-family mixture.
    """

    if rollout_laps is None or lap_weights is None:
        return track_ids, None if track_weights is None else dict(track_weights)
    if rollout_laps.shape != track_ids.shape:
        raise ValueError("PPO rollout_laps must align with track_ids")
    rounded_laps = rollout_laps.round().to(torch.long)
    if not torch.allclose(rollout_laps.float(), rounded_laps.float()):
        raise ValueError("PPO rollout_laps must contain integers")
    if torch.any(rounded_laps < 1):
        raise ValueError("PPO rollout_laps must be positive")
    configured = {int(key): float(value) for key, value in lap_weights.items()}
    if any(
        lap < 1 or not np.isfinite(weight) or weight <= 0.0
        for lap, weight in configured.items()
    ):
        raise ValueError("PPO minibatch lap weights must be finite and positive")
    missing = set(rounded_laps.unique().tolist()) - set(configured)
    if missing:
        raise ValueError(f"missing PPO minibatch weights for lap counts {sorted(missing)}")
    radix = int(rounded_laps.max()) + 1
    strata = track_ids.to(torch.long) * radix + rounded_laps
    stratum_records: list[tuple[int, int, int, float]] = []
    for stratum in torch.unique(strata, sorted=True).tolist():
        mask = strata == stratum
        track = int(track_ids[mask][0])
        laps = int(rounded_laps[mask][0])
        stratum_records.append((
            int(stratum), track, laps,
            float((track_weights or {}).get(track, 1.0)),
        ))
    # Normalize within each observed lap count first. Otherwise a lap count
    # represented on more concrete tracks would silently receive more total
    # gradient mass than its configured quota.
    lap_track_weight = {
        laps: sum(weight for _, _, item_laps, weight in stratum_records if item_laps == laps)
        for laps in configured
    }
    weights = {
        stratum: configured[laps] * track_weight / lap_track_weight[laps]
        for stratum, _, laps, track_weight in stratum_records
    }
    return strata, weights


PPO_REWARD_COMPONENTS: tuple[str, ...] = (
    "progress", "center_potential", "heading_potential", "speed_progress",
    "speed_tracking", "gate_pass", "body_rate", "smoothness", "time", "crash",
    "thrust_smoothness", "xy_rate_smoothness", "yaw_rate_smoothness",
    "terminal_success", "terminal_failure",
)


def ppo_missed_directed_gate_crossing(
    previous_position: np.ndarray,
    current_position: np.ndarray,
    gate: Any,
    *,
    gate_passed: bool,
) -> bool:
    """Detect a forward plane crossing rejected by the directed aperture tracker."""
    if gate_passed:
        return False
    previous_side = float((np.asarray(previous_position) - gate.position) @ gate.normal)
    current_side = float((np.asarray(current_position) - gate.position) @ gate.normal)
    return bool(previous_side < 0.0 <= current_side)


def ppo_terminal_reward(
    settings: Mapping[str, Any], *, gates: int, target_gates: int,
    crashed: bool, done: bool,
) -> tuple[float, str | None]:
    """Return a trajectory-level completion signal for PPO credit assignment.

    Per-gate progress can rank two partial trajectories while still failing to
    distinguish a complete lap from a late crash strongly enough.  Applying
    this only at termination lets GAE propagate an explicit whole-course
    objective without changing simulator rewards or evaluation metrics.
    """

    success_bonus = float(settings.get("terminal_success_bonus", 0.0))
    failure_penalty = float(settings.get("terminal_failure_penalty", 0.0))
    if (
        not np.isfinite(success_bonus) or success_bonus < 0.0
        or not np.isfinite(failure_penalty) or failure_penalty < 0.0
    ):
        raise ValueError("PPO terminal rewards must be finite and non-negative")
    if not done:
        return 0.0, None
    success = int(gates) >= int(target_gates) and not bool(crashed)
    return (
        (success_bonus, "terminal_success")
        if success else (-failure_penalty, "terminal_failure")
    )


def ppo_critic_features(
    normalized_history: np.ndarray,
    episode_context: np.ndarray,
    settings: Mapping[str, Any],
) -> np.ndarray:
    """Build an explicitly configured critic input.

    Historical Starscream runs append four rollout-progress values to an
    asymmetric critic.  The Green-2026 state baseline gives actor and critic
    the same state observation, so its reproduction config disables that
    extension and uses only the latest Markov descriptor.
    """

    history = np.asarray(normalized_history, np.float32)
    if str(settings.get(
        "critic_architecture", "mlp"
    )) in PRIVILEGED_COURSE_CRITIC_ARCHITECTURES:
        if not bool(settings.get("critic_include_history", True)):
            raise ValueError("privileged course critic requires critic_include_history")
        if not bool(settings.get("critic_include_episode_context", True)):
            raise ValueError(
                "privileged course critic requires critic_include_episode_context"
            )
    value = (
        history.reshape(-1)
        if bool(settings.get("critic_include_history", False))
        else history[-1]
    )
    if bool(settings.get("critic_include_episode_context", True)):
        value = np.concatenate([
            value, np.asarray(episode_context, np.float32)
        ])
    return np.asarray(value, np.float32)


def build_ppo_critic(
    policy: PrivilegedMLPPolicy, settings: Mapping[str, Any], device: str,
) -> torch.nn.Module:
    """Construct the configured value function without changing actor topology."""
    architecture = str(settings.get("critic_architecture", "mlp"))
    if architecture == "mlp":
        critic_input_dim = (
            policy.context_steps * policy.input_dim
            if bool(settings.get("critic_include_history", False))
            else policy.input_dim
        )
        if bool(settings.get("critic_include_episode_context", True)):
            critic_input_dim += PRIVILEGED_CRITIC_EPISODE_DIM
        return PrivilegedValue(
            input_dim=critic_input_dim,
            hidden_dim=int(settings.get("critic_hidden_dim", 384)),
        ).to(device)
    if architecture == "privileged_course_fusion_v1":
        if not bool(settings.get("critic_include_history", True)):
            raise ValueError("privileged course critic requires critic_include_history")
        if not bool(settings.get("critic_include_episode_context", True)):
            raise ValueError(
                "privileged course critic requires critic_include_episode_context"
            )
        return PrivilegedCourseValue(
            actor_input_dim=policy.input_dim,
            context_steps=policy.context_steps,
            episode_feature_dim=PRIVILEGED_CRITIC_EPISODE_DIM,
            course_feature_dim=PRIVILEGED_CRITIC_COURSE_DIM,
            course_vocab_size=int(settings.get("critic_course_vocab_size", 512)),
            hidden_dim=int(settings.get("critic_fusion_hidden_dim", 256)),
            course_embedding_dim=int(settings.get(
                "critic_course_embedding_dim", 64,
            )),
        ).to(device)
    if architecture != "privileged_transformer_v2":
        raise ValueError(f"unknown critic_architecture {architecture!r}")
    if not bool(settings.get("critic_include_history", True)):
        raise ValueError("privileged Transformer critic requires critic_include_history")
    if not bool(settings.get("critic_include_episode_context", True)):
        raise ValueError(
            "privileged Transformer critic requires critic_include_episode_context"
        )
    return PrivilegedTransformerValue(
        actor_input_dim=policy.input_dim,
        context_steps=policy.context_steps,
        route_tokens=int(settings.get("critic_route_tokens", 24)),
        route_feature_dim=PRIVILEGED_CRITIC_ROUTE_DIM,
        episode_feature_dim=PRIVILEGED_CRITIC_EPISODE_DIM,
        course_feature_dim=PRIVILEGED_CRITIC_COURSE_DIM,
        course_vocab_size=int(settings.get("critic_course_vocab_size", 4096)),
        model_dim=int(settings.get("critic_model_dim", policy.recurrent_dim)),
        depth=int(settings.get("critic_transformer_depth", policy.recurrent_depth)),
        heads=int(settings.get("critic_attention_heads", policy.attention_heads)),
        feedforward_dim=int(settings.get(
            "critic_feedforward_dim", policy.transformer_feedforward_dim
        )),
        dropout=float(settings.get("critic_dropout", 0.0)),
    ).to(device)


def _ppo_env_protocol(
    settings: dict[str, Any],
    stage: RacingCurriculumStage,
    worker_index: int,
) -> None:
    """Cooperative simulator protocol; each lane owns fully independent state."""
    env: FlightmareEnv | None = None
    try:
        worker_track = stage.tracks[worker_index % len(stage.tracks)]
        env = make_env(
            settings, reward=make_reward(settings, stage.target_speed),
            track=worker_track,
        )
        observation: dict[str, Any] | None = None
        start_passed = 0
        steps = 0
        total_return = 0.0
        crashed = False
        speed_sum = 0.0
        forward_speed_sum = 0.0
        maximum_speed = 0.0
        gate_speed_sum = 0.0
        gate_speed_count = 0
        collective_saturation_steps = 0
        body_rate_saturation_steps = 0
        collective_command_sum = 0.0
        maximum_collective_command = 0.0
        absolute_body_rate_command_sum = np.zeros(3, np.float64)
        horizontal_acceleration_sum = 0.0
        maximum_horizontal_acceleration = 0.0
        maximum_motor_utilization = 0.0
        target_speed = float(stage.target_speed)
        course_id = worker_index % len(stage.tracks)
        start_gate_index = 0
        rollout_laps = int(settings.get("rollout_laps", stage.rollout_laps))
        if rollout_laps < 1:
            raise ValueError("rollout_laps must be positive")
        target_gates = completion_gate_count(env.track,stage.target_gates,enabled=settings.get("allow_curriculum_completion_override",False)) * rollout_laps
        lap_crossing_steps: list[int] = []
        last_gate_pass_step = 0
        reward_component_sums = {name: 0.0 for name in PPO_REWARD_COMPONENTS}

        def context() -> np.ndarray:
            assert observation is not None
            gates = env.tracker.passed_count - start_passed
            state = np.asarray(observation["state"], np.float32)
            episode = np.asarray([
                gates / max(target_gates, 1),
                steps / max(stage.max_steps, 1),
                env.tracker.index / max(len(env.track.gates) - 1, 1),
                np.linalg.norm(state[7:10]) / max(target_speed, 1.0e-3),
            ], np.float32)
            return ppo_privileged_critic_context(
                env, state, episode, target_speed=target_speed,
                target_gates=target_gates, completed_gates=gates,
                settings=settings, course_id=course_id,
            )

        yield ("ready", worker_index)
        while True:
            request = yield None
            operation = request[0]
            if operation == "close":
                yield ("closed", worker_index)
                return
            if operation == "reset":
                episode_index, seed = int(request[1]), int(request[2])
                requested_track = str(request[3]) if len(request) > 3 else worker_track
                target_speed = float(request[4]) if len(request) > 4 else float(stage.target_speed)
                rollout_laps = int(request[5]) if len(request) > 5 else int(stage.rollout_laps)
                course_id = int(request[6]) if len(request) > 6 else course_id
                if rollout_laps < 1:
                    raise ValueError("PPO rollout_laps must be positive")
                if requested_track != worker_track:
                    env.close()
                    worker_track = requested_track
                    env = make_env(
                        settings, reward=make_reward(settings, stage.target_speed),
                        track=worker_track,
                    )
                target_gates = (
                    completion_gate_count(env.track,stage.target_gates,enabled=settings.get("allow_curriculum_completion_override",False)) * rollout_laps
                )
                env.reward_function = make_reward(settings, target_speed)
                spawn_episode_index = int(
                    settings.get("evaluation_fixed_start_gate_index", episode_index)
                )
                observation, start_passed = reset_env(
                    env, stage, seed=seed, episode_index=spawn_episode_index,
                    rollout_laps=rollout_laps,
                )
                start_gate_index = int(env.tracker.index)
                gate_events = None
                if settings.get('ppo_reference_aware_gate_events', False):
                    line, _, _, _ = make_gate_reference(env, settings)
                    gate_events = ReferenceGateEvents(line, len(env.track.gates), observation['state'][:3],
                        float(settings.get('gate_event_reference_tolerance', .75)))
                steps = 0
                total_return = 0.0
                crashed = False
                speed_sum = 0.0
                forward_speed_sum = 0.0
                maximum_speed = 0.0
                gate_speed_sum = 0.0
                gate_speed_count = 0
                collective_saturation_steps = 0
                body_rate_saturation_steps = 0
                collective_command_sum = 0.0
                maximum_collective_command = 0.0
                absolute_body_rate_command_sum[:] = 0.0
                horizontal_acceleration_sum = 0.0
                maximum_horizontal_acceleration = 0.0
                maximum_motor_utilization = 0.0
                lap_crossing_steps = []
                last_gate_pass_step = 0
                reward_component_sums = {
                    name: 0.0 for name in PPO_REWARD_COMPONENTS
                }
                yield (
                    "reset", ppo_observation_features(observation, settings), context()
                )
                continue
            if operation != "advance" or observation is None:
                raise RuntimeError(f"invalid PPO worker operation: {operation}")
            normalized_action = np.asarray(request[1], np.float32)
            physical_action = ppo_normalized_to_ctbr(normalized_action, settings)
            gate_before = env.track.gates[env.tracker.index]
            gate_index_before = int(env.tracker.index)
            position_before = np.asarray(observation["state"][:3], np.float32).copy()
            if settings.get('native_vector_step', False):
                env.prepare_step(physical_action)
                raw = yield ('native_step', env)
                observation, reward, terminated, _, info = env.finish_step(raw)
            else:
                observation, reward, terminated, _, info = env.step(physical_action)
            steps += 1
            state = np.asarray(observation["state"], np.float32)
            reference_miss = (gate_events.advance(position_before, state[:3], gate_before,
                gate_index_before, bool(info.get('gate_passed', False))) if gate_events else False)
            missed_gate = bool(
                settings.get("ppo_terminate_on_missed_gate", False)
                and (reference_miss if gate_events else ppo_missed_directed_gate_crossing(
                    position_before, state[:3], gate_before,
                    gate_passed=bool(info.get("gate_passed", False)),
                ))
            )
            gate_dwell_limit = float(settings.get("ppo_max_gate_dwell_seconds", 0.0))
            if gate_dwell_limit < 0 or not np.isfinite(gate_dwell_limit):
                raise ValueError("ppo_max_gate_dwell_seconds must be finite and nonnegative")
            gate_dwell_expired = bool(
                gate_dwell_limit > 0
                and not bool(info.get("gate_passed", False))
                and steps - last_gate_pass_step >= round(gate_dwell_limit * configured_control_hz(settings))
            )
            speed = float(np.linalg.norm(state[7:10]))
            components = dict(info.get("reward_components", {}))
            gate_failure_penalty = float(settings.get("ppo_gate_failure_crash_penalty", 0.0))
            if gate_failure_penalty < 0 or not np.isfinite(gate_failure_penalty):
                raise ValueError("ppo_gate_failure_crash_penalty must be finite and nonnegative")
            if (missed_gate or gate_dwell_expired) and not (
                info.get("ground_contact") or info.get("unity_collision")
            ):
                reward -= gate_failure_penalty
                components["crash"] = float(components.get("crash", 0.0)) - gate_failure_penalty
            total_return += float(reward)
            for name in PPO_REWARD_COMPONENTS:
                reward_component_sums[name] += float(components.get(name, 0.0))
            forward_speed = float(components.get("forward_speed", 0.0))
            speed_sum += speed
            forward_speed_sum += forward_speed
            maximum_speed = max(maximum_speed, speed)
            collective_command_sum += float(physical_action[0])
            maximum_collective_command = max(
                maximum_collective_command, float(physical_action[0])
            )
            absolute_body_rate_command_sum += np.abs(physical_action[1:])
            horizontal_acceleration = float(np.linalg.norm(state[13:15]))
            horizontal_acceleration_sum += horizontal_acceleration
            maximum_horizontal_acceleration = max(
                maximum_horizontal_acceleration, horizontal_acceleration
            )
            maximum_motor_utilization = max(
                maximum_motor_utilization,
                float(np.max(info.get("applied_motor_normalized", 0.0))),
            )
            collective_saturation_steps += int(abs(float(normalized_action[0])) >= 0.95)
            body_rate_saturation_steps += int(np.max(np.abs(normalized_action[1:])) >= 0.95)
            if bool(info.get("gate_passed", False)):
                gate_speed_sum += forward_speed
                gate_speed_count += 1
                last_gate_pass_step = steps
                gate_dwell_expired = False
            gates = env.tracker.passed_count - start_passed
            if (
                bool(info.get("gate_passed", False))
                and gates > 0 and gates % len(env.track.gates) == 0
            ):
                lap_crossing_steps.append(steps)
            crashed = crashed or bool(
                info.get("ground_contact") or info.get("unity_collision")
                or missed_gate or gate_dwell_expired
            )
            done = bool(
                terminated or missed_gate or gate_dwell_expired
                or gates >= target_gates or steps >= stage.max_steps
            )
            terminal_reward, terminal_component = ppo_terminal_reward(
                settings, gates=gates, target_gates=target_gates,
                crashed=crashed, done=done,
            )
            if terminal_component is not None:
                reward += terminal_reward
                total_return += terminal_reward
                reward_component_sums[terminal_component] += terminal_reward
            result = None if not done else {
                "episode_index": episode_index,
                "episode_seed": seed,
                "gates": gates,
                "crashed": crashed,
                "return": total_return,
                "steps": steps,
                "track": env.track.name,
                "target_gates": target_gates,
                "target_speed_mps": target_speed,
                "rollout_laps": rollout_laps,
                "start_gate_index": start_gate_index,
                "final_active_gate_index": int(env.tracker.index),
                "ground_contact": bool(info.get("ground_contact", False)),
                "unity_collision": bool(info.get("unity_collision", False)),
                "timed_out": bool(steps >= stage.max_steps and not terminated and not missed_gate and not gate_dwell_expired),
                "missed_gate_terminated": bool(missed_gate),
                "gate_dwell_terminated": bool(gate_dwell_expired),
                "mean_speed_mps": speed_sum / max(steps, 1),
                "maximum_speed_mps": maximum_speed,
                "mean_forward_speed_mps": forward_speed_sum / max(steps, 1),
                "mean_gate_speed_mps": gate_speed_sum / max(gate_speed_count, 1),
                "collective_saturation_fraction": (
                    collective_saturation_steps / max(steps, 1)
                ),
                "body_rate_saturation_fraction": (
                    body_rate_saturation_steps / max(steps, 1)
                ),
                "mean_collective_command_mps2": (
                    collective_command_sum / max(steps, 1)
                ),
                "maximum_collective_command_mps2": maximum_collective_command,
                "mean_absolute_roll_rate_command_rps": (
                    absolute_body_rate_command_sum[0] / max(steps, 1)
                ),
                "mean_absolute_pitch_rate_command_rps": (
                    absolute_body_rate_command_sum[1] / max(steps, 1)
                ),
                "mean_absolute_yaw_rate_command_rps": (
                    absolute_body_rate_command_sum[2] / max(steps, 1)
                ),
                "mean_horizontal_acceleration_mps2": (
                    horizontal_acceleration_sum / max(steps, 1)
                ),
                "maximum_horizontal_acceleration_mps2": (
                    maximum_horizontal_acceleration
                ),
                "maximum_motor_utilization": maximum_motor_utilization,
                "completed_laps": len(lap_crossing_steps),
                "reward_component_sums": dict(reward_component_sums),
            }
            if result is not None and gate_events is not None:
                result.update(reference_misses=gate_events.misses,
                    raw_plane_crossings=gate_events.raw_crossings,
                    planned_plane_crossings=gate_events.planned_crossings,
                    within_clean_deadline=steps <= int(settings.get('evaluation_clean_deadlines', {}).get(env.track.name, stage.max_steps)))
            if result is not None and len(lap_crossing_steps) >= 2:
                measured_laps = np.diff(np.asarray(lap_crossing_steps, np.float64))
                result["measured_rolling_lap_steps"] = float(measured_laps[-1])
                result["mean_measured_rolling_lap_steps"] = float(measured_laps.mean())
            yield (
                "step", float(reward), ppo_observation_features(observation, settings),
                context(), done, result,
            )
    finally:
        if env is not None:
            env.close()


def _ppo_process_worker(connection, settings, stage, worker_index):
    _terminate_worker_with_parent()
    protocol=_ppo_env_protocol(settings,stage,worker_index)
    try:
        connection.send(next(protocol));next(protocol)
        while True:
            response=protocol.send(connection.recv())
            if response[0] == 'native_step':
                from starscream.env.native_batch import advance_prepared
                response = protocol.send(advance_prepared([response[1]])[0])
            connection.send(response)
            if response[0]=='closed':return
            next(protocol)
    except BaseException:
        connection.send(('error',traceback.format_exc()))
        raise
    finally:
        protocol.close()
        connection.close()


def _ppo_group_worker(connection, settings, stage, first_index, count):
    """Parallel groups; optionally batch native physics within each process.

    Reward/observation finishing is still serial Python within a group.
    """
    _terminate_worker_with_parent()
    protocols=[]
    try:
        ready=[]
        for lane in range(count):
            protocol=_ppo_env_protocol(settings,stage,first_index+lane)
            protocols.append(protocol);ready.append((lane,next(protocol)));next(protocol)
        connection.send(('batch',ready));closed=set()
        while len(closed)<count:
            requests=connection.recv();responses=[]
            native_pending = []
            for lane,request in requests:
                if lane in closed:raise RuntimeError('request to closed simulator')
                response=protocols[lane].send(request)
                if response[0] == 'native_step':
                    native_pending.append((lane, response[1]))
                    continue
                responses.append((lane,response))
                if response[0]=='closed':closed.add(lane)
                else:next(protocols[lane])
            if native_pending:
                from starscream.env.native_batch import advance_prepared
                raw = advance_prepared([e for _, e in native_pending],
                    threads=int(settings.get('native_vector_threads', 1)))
                for (lane, _), state in zip(native_pending, raw):
                    response = protocols[lane].send(state)
                    responses.append((lane, response))
                    next(protocols[lane])
            connection.send(('batch',responses))
    except BaseException:
        connection.send(('error',traceback.format_exc()))
        raise
    finally:
        for protocol in protocols:protocol.close()
        connection.close()


@dataclass
class ProcessPPOSlot:
    connection: Any
    process: Any
    history: CausalHistory
    track: str
    target_speed: float = 0.0
    rollout_laps: int = 1
    context: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    exploration_residual: np.ndarray = field(
        default_factory=lambda: np.zeros(4, np.float32)
    )
    transitions: list[dict[str, Any]] = field(default_factory=list)
    done: bool = True


class ProcessRaceCollector:
    """Persistent process-isolated simulators with batched GPU inference."""

    def __init__(
        self,
        policy: PrivilegedMLPPolicy,
        normalizer: FeatureNormalizer,
        settings: Mapping[str, Any],
        stage: RacingCurriculumStage,
        device: str,
        *,
        workers: int | None = None,
        sampling_prefix: str | None = None,
    ) -> None:
        self.policy = policy
        self.normalizer = normalizer
        self.settings = dict(settings)
        self.stage = stage
        self.device = device
        if (
            str(settings.get("critic_architecture", "mlp"))
            == "privileged_course_fusion_v1"
            and len(stage.tracks) > int(settings.get("critic_course_vocab_size", 512))
        ):
            raise ValueError(
                "active course set exceeds privileged critic vocabulary"
            )
        self.lap_time_constraints = None
        if sampling_prefix == 'ppo' and settings.get('ppo_lap_time', {}).get('enabled', False):
            from starscream.ppo_lap_time import LapTimeConstraints
            self.lap_time_constraints = LapTimeConstraints(settings)
        self.parallel = int(workers or settings.get("rollout_envs", 12))
        if sampling_prefix == 'ppo' and settings.get('ppo_rollout_window_steps', 0):
            from starscream.ppo_windows import validate_window_settings
            validate_window_settings(settings, self.parallel)
        if settings.get('native_vector_step', False):
            from starscream.env.native_batch import require_native_batch
            require_native_batch()  # Fail before starting any worker processes.
        self.inference_policy = inference_callable(policy, settings, self.parallel)
        from starscream.inference_graph import PPOInferenceGraphs, HostPolicyInferenceGraphs
        self.host_evaluation_graphs = (
            HostPolicyInferenceGraphs(policy, self.parallel)
            if settings.get('host_evaluation_graph', False) else None
        )
        self.ppo_inference_graphs = (
            PPOInferenceGraphs(policy, self.parallel)
            if settings.get('cuda_graph_ppo_inference', False) else None
        )
        self.sampling_prefix = sampling_prefix
        self.closed = False
        self.base_track_weights = configured_ppo_track_weights(
            stage.tracks, self.settings,
        ) if sampling_prefix == "ppo" else np.ones(len(stage.tracks), np.float64)
        self.track_weights = self.base_track_weights.copy()
        self.track_names = [Path(track).stem for track in stage.tracks]
        self.track_families = ["unclassified" for _ in stage.tracks]
        self.track_roles = ["unclassified" for _ in stage.tracks]
        manifest_path = settings.get("track_manifest")
        if manifest_path is not None:
            manifest_path = Path(str(manifest_path))
            manifest = read_manifest(manifest_path)
            metadata = {
                str((manifest_path.parent / str(record["path"])).resolve()): (
                    str(record.get("name", Path(str(record["path"])).stem)),
                    str(record.get("family", "unclassified")),
                    str(record.get("rl_role", "unclassified")),
                )
                for record in manifest["records"]
            }
            for index, track in enumerate(stage.tracks):
                name, family, role = metadata.get(
                    str(Path(track).resolve()),
                    (Path(track).stem, "unclassified", "unclassified"),
                )
                self.track_names[index] = name
                self.track_families[index] = family
                self.track_roles[index] = role
        aliases = {
            str(name): str(alias)
            for name, alias in settings.get("reliability_family_aliases", {}).items()
        }
        self.track_families = [aliases.get(family, family) for family in self.track_families]
        role_floors = (
            self.settings.get("ppo_sampling_role_floors", {})
            if sampling_prefix == "ppo" else {}
        )
        if role_floors:
            from starscream.ppo_windows import apply_group_probability_floors
            self.base_track_weights = apply_group_probability_floors(
                self.base_track_weights, self.track_roles, role_floors,
            )
            self.track_weights = self.base_track_weights.copy()
        constraint_scope = str(settings.get(
            "ppo_group_constraint_scope", "family"
        ))
        if constraint_scope not in {"family", "track"}:
            raise ValueError(
                "ppo_group_constraint_scope must be family or track"
            )
        if constraint_scope == "track":
            self.track_families = list(self.track_names)
        self.family_names = tuple(dict.fromkeys(self.track_families))
        self.family_ids = np.asarray(
            [self.family_names.index(family) for family in self.track_families], np.int64
        )
        self._adaptive_competence = np.full(len(stage.tracks), 0.5, np.float64)
        switch_config = settings.get("green2026_adaptive_task_switching", {})
        self._paper_switch_config = (
            dict(switch_config) if isinstance(switch_config, Mapping) else {}
        )
        self._persistent_task_slots = bool(
            sampling_prefix == "ppo"
            and self.settings.get("ppo_persistent_task_slots", False)
        )
        self._paper_switcher: SpearmanTaskSwitcher | None = None
        self._paper_switch_rng = np.random.default_rng(
            int(settings.get("seed", 20260817)) ^ 0x6A7265656E
        )
        if (
            sampling_prefix == "ppo"
            and bool(self._paper_switch_config.get("enabled", False))
        ):
            if bool(settings.get("ppo_adaptive_level_replay", {}).get("enabled", False)):
                raise ValueError(
                    "paper adaptive task switching and PLR sampling are mutually exclusive"
                )
            if len(stage.tracks) < self.parallel and bool(
                self._paper_switch_config.get("require_unique_active_tasks", True)
            ):
                raise ValueError(
                    "paper task switching needs at least one unique track per environment"
                )
            self._paper_switcher = SpearmanTaskSwitcher(
                window=int(self._paper_switch_config.get("window", 600)),
                alpha=float(self._paper_switch_config.get("alpha", 1000.0)),
            )
            self._persistent_task_slots = True
        if getattr(self, "_persistent_task_slots", False):
            require_unique = bool(
                self.settings.get("ppo_persistent_require_unique_tasks", True)
            )
            if len(stage.tracks) < self.parallel and require_unique:
                raise ValueError(
                    "persistent PPO task slots need at least one unique track per environment"
                )
        self._constraint_lambdas = np.zeros(len(self.family_names), np.float64)
        self._constraint_success_ema = np.full(len(self.family_names), np.nan, np.float64)
        self.track_target_speeds: dict[str, float] = {}
        if (
            stage.manifest_speed_scale_range is not None
            or bool(settings.get("evaluation_condition_on_manifest_speed", False))
        ):
            manifest_path = settings.get("track_manifest")
            if manifest_path is None:
                raise ValueError(
                    "evaluation_condition_on_manifest_speed requires track_manifest"
                )
            self.track_target_speeds = manifest_track_values(
                str(manifest_path),
                configured_ppo_manifest_speed_field(settings),
                cast=float,
            )
        if self.sampling_prefix != 'ppo':
            self.track_target_speeds.update(evaluation_track_speed_commands(settings))
        context = mp.get_context("spawn")
        self.slots: list[ProcessPPOSlot] = []
        initial_slot_tracks = list(stage.tracks)
        if getattr(self, "_persistent_task_slots", False):
            require_unique = bool(
                self._paper_switch_config.get("require_unique_active_tasks", True)
                if self._paper_switcher is not None
                else self.settings.get("ppo_persistent_require_unique_tasks", True)
            )
            selected = persistent_ppo_task_indices(
                len(stage.tracks), self.parallel, self.base_track_weights,
                require_unique=require_unique,
                balance_tasks=bool(self.settings.get(
                    "ppo_persistent_balance_tasks", False
                )),
                rng=self._paper_switch_rng,
                floor_before_weighting=bool(self.settings.get(
                    "ppo_persistent_floor_before_weighting", True
                )),
            )
            initial_slot_tracks = [
                stage.tracks[int(track_index)] for track_index in selected
            ]
        try:
            lanes = int(settings.get(
                'ppo_envs_per_worker' if sampling_prefix == 'ppo' else 'evaluation_envs_per_worker', 1))
            if lanes < 1:raise ValueError('ppo_envs_per_worker must be positive')
            for index in range(0, self.parallel, lanes):
                parent, child = context.Pipe()
                count = min(lanes, self.parallel-index)
                if settings.get('ppo_packed_transport', False) and (
                    lanes == 1 or settings.get('ppo_packed_group_transport', False)
                ):
                    from starscream.ppo_collection import PackedPPOPipe
                    parent, child = PackedPPOPipe(parent), PackedPPOPipe(child)
                process = context.Process(
                    target=_ppo_process_worker if lanes==1 else _ppo_group_worker,
                    args=(child, self.settings, stage, index) if lanes==1 else (
                        child,self.settings,stage,index,count),
                    name=f"ppo-env-{index:02d}",
                )
                process.start()
                child.close()
                from starscream.ppo_collection import MultiplexPipe
                mux = MultiplexPipe(parent,count) if lanes>1 else None
                for lane in range(count):
                    self.slots.append(ProcessPPOSlot(
                        parent if mux is None else mux.lane(lane), process,
                        CausalHistory(policy.context_steps),
                        initial_slot_tracks[(index+lane) % len(initial_slot_tracks)],
                    ))
            for slot in self.slots:
                self._receive(slot, "ready")
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    @staticmethod
    def _receive(slot: ProcessPPOSlot, expected: str) -> tuple[Any, ...]:
        response = slot.connection.recv()
        if response[0] == "error":
            raise RuntimeError(f"PPO worker failed:\n{response[1]}")
        if response[0] != expected:
            raise RuntimeError(f"expected PPO worker {expected}, received {response[0]}")
        return response

    def _reset_slot(
        self, slot: ProcessPPOSlot, index: int, seed_base: int, track: str,
    ) -> None:
        episode_seed = seed_base + 1009 * index
        if self.sampling_prefix != 'ppo':
            from starscream.evaluation_suite import selection_episode_seed
            episode_seed = selection_episode_seed(self.settings, track, index, seed_base, len(self.stage.tracks))
        resolved_track = str(Path(track).resolve())
        target_speed = episode_target_speed(self.stage, episode_seed)
        if resolved_track in self.track_target_speeds:
            target_speed = self.track_target_speeds[resolved_track]
            if self.stage.manifest_speed_scale_range is not None:
                low, high = self.stage.manifest_speed_scale_range
                generator = np.random.default_rng(episode_seed ^ 0x4D504343)
                target_speed *= float(generator.uniform(low, high))
            else:
                target_speed *= float(
                    self.settings.get("evaluation_manifest_speed_scale", 1.0)
                )
        rollout_laps = ppo_episode_rollout_laps(
            self.settings, self.stage, episode_seed,
            training=self.sampling_prefix == "ppo",
        )
        slot.connection.send((
            "reset", index, episode_seed, track, target_speed, rollout_laps,
            self.stage.tracks.index(track),
        ))
        response = self._receive(slot, "reset")
        slot.history.reset_feature(response[1])
        slot.context = np.asarray(response[2], np.float32)
        slot.exploration_residual.fill(0.0)
        slot.transitions = []
        slot.track = str(track)
        slot.target_speed = target_speed
        slot.rollout_laps = rollout_laps
        slot.done = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for slot in self.slots:
            if slot.process.is_alive():
                try:
                    slot.connection.send(("close",))
                except (BrokenPipeError, EOFError):
                    pass
        for slot in self.slots:
            if slot.process.is_alive():
                try:
                    self._receive(slot, "closed")
                except (BrokenPipeError, EOFError, RuntimeError):
                    pass
            slot.process.join(timeout=5.0)
            if slot.process.is_alive():
                slot.process.terminate()
                slot.process.join(timeout=2.0)
            if slot.process.is_alive():
                slot.process.kill()
                slot.process.join(timeout=2.0)
            slot.connection.close()
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def _schedule_next(
        self, slot: ProcessPPOSlot, pending: Mapping[str, deque[int]], seed_base: int,
    ) -> bool:
        track = slot.track if pending.get(slot.track) else ""
        if not track:
            track = next((item for item in self.stage.tracks if pending[item]), "")
        if not track:
            slot.done = True
            return False
        self._reset_slot(slot, pending[track].popleft(), seed_base, track)
        return True

    def _schedule(self, episodes: int, seed_base: int) -> dict[str, deque[int]]:
        if getattr(self, "_persistent_task_slots", False):
            if episodes != self.parallel:
                raise ValueError(
                    "persistent PPO task slots require episodes_per_cycle == "
                    "rollout_envs so every task contributes one trajectory per update"
                )
            pending = {track: deque() for track in self.stage.tracks}
            for index, slot in enumerate(self.slots):
                self._reset_slot(slot, index, seed_base, slot.track)
            return pending
        schedule = (
            ppo_episode_tracks(
                self.stage.tracks, episodes, self.settings, seed=seed_base,
                weights_override=self.track_weights,
            )
            if getattr(self, "sampling_prefix", None) == "ppo"
            else [
                self.stage.tracks[index % len(self.stage.tracks)]
                for index in range(episodes)
            ]
        )
        scheduler_settings = getattr(self, "settings", {})
        index_offset = int(scheduler_settings.get("evaluation_episode_index_offset", 0))
        index_stride = int(scheduler_settings.get("evaluation_episode_index_stride", 1))
        if index_offset < 0 or index_stride < 1:
            raise ValueError("evaluation episode index offset/stride is invalid")
        pending = {
            track: deque(
                index_offset + index_stride * index
                for index in range(episodes)
                if schedule[index] == track
            )
            for track in self.stage.tracks
        }
        for slot in self.slots:
            self._schedule_next(slot, pending, seed_base)
        return pending

    def update_paper_task_switching(
        self, results: Sequence[Mapping[str, Any]],
    ) -> dict[str, float]:
        """Apply Algorithm 2 switching after the joint multi-task rollout.

        The configured track manifest is a frozen approximation to the paper's
        asynchronous generator.  A replacement therefore draws a fresh item
        from that pool; it never mutates the held-out evaluation manifest.
        """

        if self._paper_switcher is None:
            return {}
        by_slot = {int(result["episode_index"]): result for result in results}
        if set(by_slot) != set(range(self.parallel)):
            raise RuntimeError(
                "paper task switching expected exactly one completed rollout per slot"
            )
        require_unique = bool(
            self._paper_switch_config.get("require_unique_active_tasks", True)
        )
        switches = 0
        ready = 0
        probabilities: list[float] = []
        correlations: list[float] = []
        for index, slot in enumerate(self.slots):
            key = f"slot-{index:03d}"
            decision = self._paper_switcher.observe(
                key, float(by_slot[index]["return"])
            )
            probabilities.append(decision.probability)
            if decision.correlation is not None:
                correlations.append(decision.correlation)
            ready += int(decision.ready)
            if not decision.ready or self._paper_switch_rng.random() >= decision.probability:
                continue
            occupied = {item.track for item in self.slots if item is not slot}
            candidates = [
                track for track in self.stage.tracks
                if track != slot.track and (not require_unique or track not in occupied)
            ]
            if not candidates:
                raise RuntimeError("adaptive task pool has no legal replacement")
            candidate_indices = np.asarray([
                self.stage.tracks.index(track) for track in candidates
            ], np.int64)
            weights = self.base_track_weights[candidate_indices].astype(np.float64)
            weights /= weights.sum()
            slot.track = str(self._paper_switch_rng.choice(candidates, p=weights))
            self._paper_switcher.replace(key)
            switches += 1
        return {
            "paper_switch/tasks_ready": float(ready),
            "paper_switch/count": float(switches),
            "paper_switch/probability_mean": float(np.mean(probabilities)),
            "paper_switch/absolute_correlation_mean": (
                float(np.mean(np.abs(correlations))) if correlations else 0.0
            ),
        }

    def paper_task_switching_state_dict(self) -> dict[str, Any] | None:
        if self._paper_switcher is None:
            return None
        return {
            "tracks": list(self.stage.tracks),
            "active_tracks": [slot.track for slot in self.slots],
            "switcher": self._paper_switcher.state_dict(),
            "rng_state": copy.deepcopy(
                self._paper_switch_rng.bit_generator.state
            ),
        }

    def load_paper_task_switching_state_dict(
        self, state: Mapping[str, Any] | None,
    ) -> None:
        if state is None:
            return
        if self._paper_switcher is None:
            raise ValueError(
                "checkpoint contains Green-2026 switching state but the sampler is disabled"
            )
        if tuple(str(track) for track in state["tracks"]) != tuple(self.stage.tracks):
            raise ValueError("Green-2026 task pool changed across PPO continuation")
        active = [str(track) for track in state["active_tracks"]]
        if len(active) != len(self.slots) or not set(active) <= set(self.stage.tracks):
            raise ValueError("Green-2026 active task slots are incompatible")
        if (
            bool(self._paper_switch_config.get("require_unique_active_tasks", True))
            and len(active) != len(set(active))
        ):
            raise ValueError("Green-2026 continuation contains duplicate active tasks")
        for slot, track in zip(self.slots, active):
            slot.track = track
        self._paper_switcher.load_state_dict(state.get("switcher"))
        self._paper_switch_rng.bit_generator.state = state["rng_state"]

    def add_tracks(self, paths: Sequence[str]) -> list[int]:
        """Append courses to the live pool between frozen windows (online course bank).

        Existing courses keep their current lane multipliers; new courses enter at the
        pool-mean multiplier with competence 0.5 so the directed sampler observes them
        immediately. Lanes are reassigned by the next window's quota computation.
        """
        new = [str(p) for p in paths if str(p) not in self.stage.tracks]
        if not new:
            return []
        if (
            str(self.settings.get("critic_architecture", "mlp"))
            == "privileged_course_fusion_v1"
            and len(self.stage.tracks) + len(new) > int(
                self.settings.get("critic_course_vocab_size", 512)
            )
        ):
            raise ValueError(
                "online course bank exceeds privileged critic vocabulary"
            )
        old = len(self.stage.tracks)
        self.stage = replace(self.stage, tracks=tuple(self.stage.tracks) + tuple(new))
        base = configured_ppo_track_weights(self.stage.tracks, self.settings)
        current = self.track_weights / self.base_track_weights
        mean_multiplier = float(current.mean()) if len(current) else 1.0
        self.base_track_weights = base
        self.track_weights = np.concatenate([base[:old] * current, base[old:] * mean_multiplier])
        self.track_names = list(self.track_names) + [Path(p).stem for p in new]
        self.track_families = list(self.track_families) + ["online_bank" for _ in new]
        self.track_roles = list(self.track_roles) + ["online_generated" for _ in new]
        role_floors = self.settings.get("ppo_sampling_role_floors", {})
        if role_floors:
            from starscream.ppo_windows import apply_group_probability_floors
            self.base_track_weights = apply_group_probability_floors(
                self.base_track_weights, self.track_roles, role_floors,
            )
            self.track_weights = apply_group_probability_floors(
                self.track_weights, self.track_roles, role_floors,
            )
        self.family_names = tuple(dict.fromkeys(self.track_families))
        self.family_ids = np.asarray(
            [self.family_names.index(family) for family in self.track_families], np.int64
        )
        lambdas = np.zeros(len(self.family_names), np.float64)
        lambdas[:len(self._constraint_lambdas)] = self._constraint_lambdas
        ema = np.full(len(self.family_names), np.nan, np.float64)
        ema[:len(self._constraint_success_ema)] = self._constraint_success_ema
        self._constraint_lambdas, self._constraint_success_ema = lambdas, ema
        self._adaptive_competence = np.concatenate(
            [self._adaptive_competence, np.full(len(new), 0.5, np.float64)]
        )
        stalls = getattr(self, "_adaptive_stall_windows", None)
        if stalls is not None:
            self._adaptive_stall_windows = np.concatenate([stalls, np.zeros(len(new), np.int64)])
        return list(range(old, old + len(new)))

    def update_adaptive_sampling(
        self, results: Sequence[Mapping[str, Any]], rollout: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        """Update next-cycle PLR-style priorities from on-policy outcomes."""

        config = self.settings.get("ppo_adaptive_level_replay", {})
        if not bool(config.get("enabled", False)):
            return {}
        name_to_index = {name: index for index, name in enumerate(self.track_names)}
        metric = str(config.get("competence_metric", "progress"))
        if metric not in {"progress", "success"}:
            raise ValueError("ppo_adaptive_level_replay competence_metric must be progress or success")
        progress_values: list[list[float]] = [[] for _ in self.stage.tracks]
        for result in results:
            index = name_to_index.get(str(result.get("track", "")))
            if index is None:
                continue
            target = max(int(result.get("target_gates", 1)), 1)
            gates = float(result.get("gates", 0))
            if metric == "success":
                # Full-lap completion is the selection objective. Gate progress
                # would call a course that always crashes at its last gate
                # nearly solved and starve it of lanes.
                value = float(gates >= target and not bool(result.get("crashed", False)))
            else:
                value = min(gates / target, 1.0)
            progress_values[index].append(value)
        observed = np.full(len(self.stage.tracks), np.nan, np.float64)
        for index, values in enumerate(progress_values):
            if values:
                observed[index] = float(np.mean(values))
        novelty = np.zeros(len(self.stage.tracks), np.float64)
        track_ids = rollout["track_id"].numpy()
        advantages = rollout["advantage"].numpy()
        for index in range(len(self.stage.tracks)):
            values = advantages[track_ids == index]
            if len(values):
                novelty[index] = float(np.mean(np.abs(values)))
        priority = str(config.get("priority", "frontier"))
        if priority == "frontier":
            weights, competence, multipliers = adaptive_frontier_weights(
                self.base_track_weights, self._adaptive_competence, observed, novelty,
                blend=float(config.get("blend", 0.35)),
                ema=float(config.get("ema", 0.30)),
                target=float(config.get("target_competence", 0.65)),
                width=float(config.get("frontier_width", 0.28)),
                minimum_multiplier=float(config.get("minimum_multiplier", 0.55)),
                maximum_multiplier=float(config.get("maximum_multiplier", 1.80)),
            )
        elif priority == "failure":
            from starscream.ppo_directed_sampling import directed_failure_weights
            stalls = getattr(self, "_adaptive_stall_windows", None)
            if stalls is None:
                stalls = np.zeros(len(self.stage.tracks), np.int64)
            weights, competence, multipliers, stalls = directed_failure_weights(
                self.base_track_weights, self._adaptive_competence, observed, stalls,
                ema=float(config.get("ema", 0.25)),
                power=float(config.get("failure_power", 1.0)),
                minimum_multiplier=float(config.get("minimum_multiplier", 0.4)),
                maximum_multiplier=float(config.get("maximum_multiplier", 2.5)),
                stall_competence=float(config.get("stall_competence", 0.1)),
                stall_patience=int(config.get("stall_patience", 15)),
                stall_factor=float(config.get("stall_factor", 0.5)),
                progress_epsilon=float(config.get("stall_progress_epsilon", 0.02)),
            )
            self._adaptive_stall_windows = stalls
        else:
            raise ValueError("ppo_adaptive_level_replay priority must be frontier or failure")
        role_floors = self.settings.get("ppo_sampling_role_floors", {})
        role_labels = np.asarray(self.track_roles, dtype=object)

        def apply_floors(candidate: np.ndarray) -> np.ndarray:
            if not role_floors:
                return candidate
            from starscream.ppo_windows import apply_group_probability_floors
            return apply_group_probability_floors(
                candidate, role_labels, role_floors,
            )

        weights = apply_floors(weights)
        success_band = self.settings.get("ppo_sampling_success_band")
        sampling_tilt = 0.0
        if success_band is not None:
            if len(success_band) != 2:
                raise ValueError("ppo_sampling_success_band must contain [low, high]")
            low, high = map(float, success_band)
            target_success = float(self.settings.get(
                "ppo_sampling_success_target", 0.5 * (low + high)
            ))
            maximum_tilt = float(self.settings.get(
                "ppo_sampling_success_max_tilt", 3.0
            ))
            if not 0.0 <= low < target_success < high <= 1.0:
                raise ValueError("invalid PPO sampling success band/target")
            if not np.isfinite(maximum_tilt) or maximum_tilt < 1.0:
                raise ValueError("ppo_sampling_success_max_tilt must be at least one")

            def tilted(beta: float) -> np.ndarray:
                centered = competence - target_success
                factor = np.exp(np.clip(beta * centered, -20.0, 20.0))
                return apply_floors(weights * factor)

            def expected(candidate: np.ndarray) -> float:
                return float(np.dot(candidate / candidate.sum(), competence))

            current_success = expected(weights)
            if current_success < low or current_success > high:
                # Positive beta favors easier courses, negative beta favors
                # harder ones. Bisection includes the role-floor projection,
                # so technical coverage cannot be traded away by the guard.
                limit = 2.0 * np.log(maximum_tilt)
                left, right = ((0.0, limit) if current_success < low else (-limit, 0.0))
                best = weights
                for _ in range(24):
                    middle = 0.5 * (left + right)
                    candidate = tilted(middle)
                    candidate_success = expected(candidate)
                    best = candidate
                    if candidate_success < target_success:
                        left = middle
                    else:
                        right = middle
                sampling_tilt = 0.5 * (left + right)
                weights = best
        self.track_weights = weights
        self._adaptive_competence = competence
        probabilities = weights / weights.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1.0e-12)))
        metrics = {
            "adaptive_sampling_multiplier_min": float(multipliers.min()),
            "adaptive_sampling_multiplier_max": float(multipliers.max()),
            "adaptive_sampling_entropy": entropy,
            "adaptive_sampling_observed_tracks": float(np.isfinite(observed).sum()),
            "adaptive_sampling_competence_mean": float(np.mean(competence)),
            "adaptive_sampling_stalled_tracks": float(
                np.sum(getattr(self, "_adaptive_stall_windows", np.zeros(1))
                       > int(config.get("stall_patience", 15)))),
        }
        if success_band is not None:
            metrics.update({
                "adaptive_sampling_expected_success": float(np.dot(
                    weights / weights.sum(), competence
                )),
                "adaptive_sampling_success_target": target_success,
                "adaptive_sampling_success_tilt": sampling_tilt,
                "adaptive_sampling_success_below_band": float(
                    np.dot(weights / weights.sum(), competence) < low
                ),
                "adaptive_sampling_success_above_band": float(
                    np.dot(weights / weights.sum(), competence) > high
                ),
            })
        if role_floors:
            for role in sorted(set(self.track_roles)):
                mask = np.asarray(self.track_roles, dtype=object) == role
                metrics[f"adaptive_sampling_role/{role}/probability"] = float(
                    weights[mask].sum() / weights.sum()
                )
        return metrics

    def update_group_constraints(
        self, results: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        """Dual-ascent reliability constraints pooled by course family."""

        config = self.settings.get("ppo_group_constraints", {})
        if not bool(config.get("enabled", False)):
            return None, {}
        name_to_index = {name: index for index, name in enumerate(self.track_names)}
        outcomes: list[list[float]] = [[] for _ in self.family_names]
        for result in results:
            track_index = name_to_index.get(str(result.get("track", "")))
            if track_index is None:
                continue
            success = float(
                not bool(result.get("crashed", False))
                and int(result.get("gates", 0)) >= int(result.get("target_gates", 1))
            )
            outcomes[int(self.family_ids[track_index])].append(success)
        default_floor = float(config.get("success_floor", 0.55))
        overrides = {
            str(name): float(value)
            for name, value in config.get("family_success_floors", {}).items()
        }
        dual_lr = float(config.get("dual_learning_rate", 0.08))
        maximum = float(config.get("maximum_lambda", 2.0))
        ema = float(config.get("ema", 0.35))
        if not (0.0 <= default_floor <= 1.0 and dual_lr >= 0.0 and maximum >= 0.0 and 0.0 < ema <= 1.0):
            raise ValueError("invalid PPO group-constraint settings")
        metrics: dict[str, float] = {}
        observed_groups = 0
        for family_id, values in enumerate(outcomes):
            if not values:
                continue
            observed_groups += 1
            observed = float(np.mean(values))
            previous = self._constraint_success_ema[family_id]
            smoothed = observed if not np.isfinite(previous) else (1.0 - ema) * previous + ema * observed
            self._constraint_success_ema[family_id] = smoothed
            family = self.family_names[family_id]
            floor = overrides.get(family, default_floor)
            self._constraint_lambdas[family_id] = np.clip(
                self._constraint_lambdas[family_id] + dual_lr * (floor - smoothed),
                0.0,
                maximum,
            )
            metrics[f"constraint/{family}/success_ema"] = smoothed
            metrics[f"constraint/{family}/lambda"] = self._constraint_lambdas[family_id]
        metrics.update({
            "constraint_observed_families": float(observed_groups),
            "constraint_lambda_mean": float(self._constraint_lambdas.mean()),
            "constraint_lambda_max": float(self._constraint_lambdas.max(initial=0.0)),
        })
        return torch.from_numpy(self._constraint_lambdas.astype(np.float32)), metrics

    @torch.no_grad()
    def evaluate_rows(
        self, *, episodes: int, seed_base: int, stochastic: bool = False,
    ) -> list[dict[str, Any]]:
        if stochastic and configured_ppo_exploration_correlation(self.settings) != 0.:
            raise ValueError('stochastic source calibration currently requires IID exploration')
        self.policy.eval()
        pending = self._schedule(episodes, seed_base)
        completed = 0
        results: list[dict[str, Any]] = []
        while completed < episodes:
            active = [slot for slot in self.slots if not slot.done]
            if not active:
                remaining = {track: len(items) for track, items in pending.items() if items}
                raise RuntimeError(
                    f"PPO evaluation scheduler exhausted workers with pending episodes: {remaining}"
                )
            histories = np.stack([slot.history.array() for slot in active])
            normalized = self.normalizer.numpy(histories)
            if stochastic:
                tensor = torch.from_numpy(normalized).to(self.device)
                speeds = torch.as_tensor([slot.target_speed for slot in active], device=self.device, dtype=tensor.dtype)
                actions = self.policy.distribution(tensor, speeds).rsample()[0].float().cpu().numpy()
            elif self.host_evaluation_graphs is not None:
                actions = self.host_evaluation_graphs.predict_numpy(normalized,
                    np.asarray([slot.target_speed for slot in active], np.float32))
            else:
                tensor = torch.from_numpy(normalized).to(self.device)
                speed_commands = torch.as_tensor(
                    [slot.target_speed for slot in active],
                    device=self.device, dtype=tensor.dtype,
                )
                actions = self.inference_policy(tensor, speed_commands).float().cpu().numpy()
            for slot, action in zip(active, actions):
                slot.connection.send(("advance", action))
            from starscream.ppo_collection import flush_lanes
            flush_lanes(active)
            responses = [self._receive(slot, "step") for slot in active]
            for slot, response in zip(active, responses):
                _, _, feature, context, done, result = response
                slot.history.append_feature(feature)
                slot.context = np.asarray(context, np.float32)
                slot.done = bool(done)
                if slot.done:
                    assert result is not None
                    results.append(result)
                    completed += 1
                    self._schedule_next(slot, pending, seed_base)
        return results

    @torch.no_grad()
    def evaluate(self, *, episodes: int, seed_base: int) -> dict[str, float]:
        rows = self.evaluate_rows(episodes=episodes, seed_base=seed_base)
        metrics = multitrack_metrics(
            rows, self.stage.target_gates * self.stage.rollout_laps,
        )
        if self.settings.get("ppo_terminate_on_missed_gate", False):
            metrics["missed_gate_termination_rate"] = float(np.mean([
                bool(row.get("missed_gate_terminated", False)) for row in rows
            ]))
        if self.settings.get("ppo_max_gate_dwell_seconds", 0.0):
            metrics["gate_dwell_termination_rate"] = float(np.mean([
                bool(row.get("gate_dwell_terminated", False)) for row in rows
            ]))
        return metrics

    @torch.no_grad()
    def collect(
        self,
        critic: PrivilegedValue,
        *,
        episodes: int,
        seed_base: int,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], float]:
        self.policy.eval(); critic.eval()
        if self.settings.get('ppo_rollout_window_steps', 0):
            from starscream.ppo_windows import collect_window
            return collect_window(self, critic, episodes=episodes, seed_base=seed_base)
        pending = self._schedule(episodes, seed_base)
        completed = 0
        raw_steps = 0
        cfm_collection_batch_id = 0
        started = time.perf_counter()
        results: list[dict[str, Any]] = []
        all_transitions: list[dict[str, Any]] = []
        packed_episodes = []
        compact = bool(self.settings.get('ppo_compact_rollout_storage',False))
        reuse_bootstrap = bool(self.settings.get('ppo_reuse_bootstrap_values', False))
        if compact and self.policy.action_head_type=='shortcut_flow':
            raise ValueError('compact PPO storage currently supports direct actions only')
        gamma = float(self.settings.get("discount", 0.997))
        lam = float(self.settings.get("gae_lambda", 0.95))
        while completed < episodes:
            active = [slot for slot in self.slots if not slot.done]
            if not active:
                remaining = {track: len(items) for track, items in pending.items() if items}
                raise RuntimeError(
                    f"PPO rollout scheduler exhausted workers with pending episodes: {remaining}"
                )
            raw_histories = np.stack([slot.history.array() for slot in active])
            normalized = self.normalizer.numpy(raw_histories)
            histories = torch.from_numpy(normalized).to(self.device)
            speed_commands = torch.as_tensor(
                [slot.target_speed for slot in active],
                device=self.device, dtype=histories.dtype,
            )
            flow_policy = self.policy.action_head_type == "shortcut_flow"
            critic_numpy = np.stack([
                ppo_critic_features(history, slot.context, self.settings)
                for history, slot in zip(normalized, active)
            ])
            critic_tensor = torch.from_numpy(critic_numpy).to(self.device)
            graph_values = None
            if flow_policy:
                assert self.policy.flow_action_head is not None
                rollout_source_noise = float(self.settings.get(
                    "fpo_rollout_source_noise",
                    self.policy.flow_action_head.source_noise,
                ))
                action_chunks = self.policy.sample_action_chunk(
                    histories, speed_commands, deterministic=False,
                    source_noise=rollout_source_noise,
                ).float()
                actions = action_chunks[:, 0]
                cfm_samples = int(self.settings.get("fpo_cfm_samples", 8))
                epsilon, flow_time, shortcut_step = sample_cfm_conditions(
                    len(active), cfm_samples, self.policy.action_chunk_steps, 4,
                    device=self.device, dtype=action_chunks.dtype,
                    time_beta=float(self.settings.get("fpo_cfm_time_beta", 1.0)),
                    step_size=1.0 / self.policy.flow_sampling_steps,
                )
                epsilon.mul_(rollout_source_noise)
                # The environment depends only on command zero, but the flow
                # distribution samples a coupled four-command latent action.
                # Score the full joint sample: masking commands 1--3 would not
                # produce the marginal likelihood of command zero because the
                # flow decoder couples the chunk through self-attention.
                valid = torch.ones(
                    len(active), self.policy.action_chunk_steps,
                    dtype=torch.bool, device=self.device,
                )
                old_cfm, _ = privileged_flow_cfm_loss(
                    self.policy, histories, speed_commands, action_chunks,
                    epsilon, flow_time, shortcut_step, valid,
                    huber_delta=self.settings.get("fpo_cfm_huber_delta"),
                )
            else:
                if self.ppo_inference_graphs is None:
                    base_distribution = self.policy.distribution(histories, speed_commands)
                else:
                    location, log_std, graph_values = self.ppo_inference_graphs(
                        histories, speed_commands, critic_tensor, critic)
                    base_distribution = SquashedGaussian(location, log_std)
                correlation = configured_ppo_exploration_correlation(self.settings)
                exploration_offsets = torch.as_tensor(
                    np.stack([
                        correlation * slot.exploration_residual for slot in active
                    ]),
                    device=self.device, dtype=base_distribution.location.dtype,
                )
                distribution = correlated_ppo_distribution(
                    base_distribution, exploration_offsets, correlation,
                )
                actions, raw_actions = distribution.rsample()
                log_prob = distribution.log_prob(actions, raw_actions)
            values = critic(critic_tensor) if graph_values is None else graph_values
            if flow_policy:
                value_numpy = values.float().cpu().numpy()
                action_numpy = actions.float().cpu().numpy()
                action_chunk_numpy = action_chunks.cpu().numpy()
                epsilon_numpy = epsilon.cpu().numpy()
                flow_time_numpy = flow_time.cpu().numpy()
                shortcut_step_numpy = shortcut_step.cpu().numpy()
                valid_numpy = valid.cpu().numpy()
                old_cfm_numpy = old_cfm.float().cpu().numpy()
            else:
                if self.settings.get('ppo_fused_rollout_transfer', False):
                    # One device synchronization instead of eight tiny copies.
                    # Pure packing; preserve every FP32 behavior likelihood bit.
                    packed = torch.cat([
                        values.reshape(-1, 1), actions, raw_actions,
                        log_prob.reshape(-1, 1), base_distribution.location,
                        exploration_offsets, distribution.location,
                        distribution.log_std.expand_as(distribution.location),
                    ], dim=-1).float().cpu().numpy()
                    value_numpy, action_numpy = packed[:, 0], packed[:, 1:5]
                    raw_action_numpy, log_prob_numpy = packed[:, 5:9], packed[:, 9]
                    base_location_numpy = packed[:, 10:14]
                    exploration_offset_numpy = packed[:, 14:18]
                    location_numpy, log_std_numpy = packed[:, 18:22], packed[:, 22:26]
                else:
                    value_numpy = values.float().cpu().numpy()
                    action_numpy = actions.float().cpu().numpy()
                    raw_action_numpy = raw_actions.float().cpu().numpy()
                    log_prob_numpy = log_prob.float().cpu().numpy()
                    base_location_numpy = base_distribution.location.float().cpu().numpy()
                    exploration_offset_numpy = exploration_offsets.float().cpu().numpy()
                    location_numpy = distribution.location.float().cpu().numpy()
                    log_std_numpy = distribution.log_std.float().cpu().numpy()
                for index, slot in enumerate(active):
                    slot.exploration_residual = (
                        raw_action_numpy[index] - base_location_numpy[index]
                    ).astype(np.float32)
            if reuse_bootstrap:
                # Critic is frozen for the rollout. V(s[t+1]) of the preceding
                # transition is exactly this step's V(s[t]); don't infer twice.
                for index, slot in enumerate(active):
                    if slot.transitions:
                        slot.transitions[-1]['next_value'] = float(value_numpy[index])
            previous_contexts = [slot.context.copy() for slot in active]
            for slot, action in zip(active, action_numpy):
                slot.connection.send(("advance", action))
            from starscream.ppo_collection import flush_lanes
            flush_lanes(active)
            responses = [self._receive(slot, "step") for slot in active]
            for slot, response in zip(active, responses):
                _, _, feature, context, done, _ = response
                slot.history.append_feature(feature)
                slot.context = np.asarray(context, np.float32)
                slot.done = bool(done)
            bootstrap_indices = list(range(len(active)))
            if reuse_bootstrap:
                # Only nonterminal collector truncations need a separate final
                # value. Never bootstrap from the auto-reset observation.
                bootstrap_indices = [i for i, response in enumerate(responses)
                    if bool(response[4]) and not completed_episode_is_terminal(
                        True, response[5], timeout_is_terminal=bool(
                            self.settings.get('ppo_timeout_is_terminal', False)))]
            next_values = np.full(len(active), np.nan, np.float32)
            if bootstrap_indices:
                bootstrap_slots = [active[i] for i in bootstrap_indices]
                next_normalized = self.normalizer.numpy(
                    np.stack([slot.history.array() for slot in bootstrap_slots]))
                next_critic_numpy = np.stack([
                    ppo_critic_features(history, slot.context, self.settings)
                    for history, slot in zip(next_normalized, bootstrap_slots)])
                next_values[bootstrap_indices] = critic(
                    torch.from_numpy(next_critic_numpy).to(self.device)).cpu().numpy()
            for index, (slot, response) in enumerate(zip(active, responses)):
                _, reward, feature, _, done, result = response
                adjusted_reward = ppo_reliability_reward(
                    float(reward), float(previous_contexts[index][0]),
                    float(slot.context[0]), bool(done), result, self.settings,
                )
                # A finite task deadline terminates; a collector truncation
                # bootstraps. The experiment explicitly selects this contract.
                terminal = completed_episode_is_terminal(bool(done), result,
                    timeout_is_terminal=bool(self.settings.get('ppo_timeout_is_terminal',False)))
                transition = {
                    "history": normalized[index].copy(),
                    "critic_input": critic_numpy[index].copy(),
                    "action": action_numpy[index].copy(),
                    "speed_command": float(slot.target_speed),
                    "reward": adjusted_reward,
                    "discount": 0.0 if terminal else gamma,
                    "gae_discount": 0.0 if terminal else gamma * lam,
                    "value": float(value_numpy[index]),
                    "next_value": 0.0 if terminal else float(next_values[index]),
                    "track_id": self.stage.tracks.index(slot.track),
                    "track_weight": float(self.track_weights[
                        self.stage.tracks.index(slot.track)
                    ]),
                    "rollout_laps": float(slot.rollout_laps),
                    "task_delta": (
                        np.asarray(feature, np.float32)[:TASK_DIM]
                        - raw_histories[index, -1, :TASK_DIM]
                    ),
                    # The legacy privileged task state is expressed in the
                    # active gate frame.  A crossing changes that coordinate
                    # frame discontinuously, so its raw feature delta is not a
                    # physical dynamics target and must never train the
                    # auxiliary head.
                    "dynamics_valid": float(
                        float(slot.context[0])
                        <= float(previous_contexts[index][0]) + 1.0e-7
                    ),
                }
                if flow_policy:
                    transition.update({
                        "action_chunk": action_chunk_numpy[index].copy(),
                        "epsilon": epsilon_numpy[index].copy(),
                        "flow_time": flow_time_numpy[index].copy(),
                        "shortcut_step": shortcut_step_numpy[index].copy(),
                        "valid": valid_numpy[index].copy(),
                        "old_cfm": old_cfm_numpy[index].copy(),
                        "cfm_collection_batch_id": cfm_collection_batch_id,
                        "cfm_collection_batch_position": index,
                    })
                else:
                    transition.update({
                        "raw_action": raw_action_numpy[index].copy(),
                        "old_location": location_numpy[index].copy(),
                        "old_log_std": log_std_numpy[index].copy(),
                        "old_log_prob": float(log_prob_numpy[index]),
                        "exploration_offset": (
                            exploration_offset_numpy[index].copy()
                        ),
                    })
                slot.transitions.append(transition)
                raw_steps += 1
                if done:
                    finish_gae(slot.transitions, gamma, lam)
                    assert result is not None
                    finish_failure_returns(
                        slot.transitions,
                        failed=(
                            bool(result["crashed"])
                            or int(result["gates"]) < int(result["target_gates"])
                        ),
                        discount=float(self.settings.get("ppo_constraint_discount", gamma)),
                    )
                    if compact:
                        from starscream.ppo_collection import pack_episode
                        packed_episodes.append(pack_episode(slot.transitions))
                    else:
                        all_transitions.extend(slot.transitions)
                    result = dict(result)
                    result["ppo_adjusted_return"] = float(sum(
                        item["reward"] for item in slot.transitions
                    ))
                    results.append(result)
                    completed += 1
                    if compact:
                        slot.transitions = []
                    self._schedule_next(slot, pending, seed_base)
            if flow_policy:
                cfm_collection_batch_id += 1
            
        if compact:
            from starscream.ppo_collection import join_episodes
            rollout = {k:torch.from_numpy(v) for k,v in join_episodes(packed_episodes).items()}
            rollout['family_id']=torch.from_numpy(self.family_ids[rollout['track_id'].numpy()]).long()
            return rollout, results, raw_steps / max(time.perf_counter()-started,1e-6)
        common_keys = (
            "history", "critic_input", "action", "advantage", "return",
            "task_delta", "speed_command", "track_weight", "rollout_laps",
            "failure_return", "dynamics_valid",
        )
        policy_keys = (
            ("action_chunk", "epsilon", "flow_time", "shortcut_step", "valid", "old_cfm")
            if self.policy.action_head_type == "shortcut_flow"
            else (
                "raw_action", "old_location", "old_log_std", "old_log_prob",
                "exploration_offset",
            )
        )
        rollout = {
            key: torch.from_numpy(np.asarray(
                [item[key] for item in all_transitions], np.float32
            ))
            for key in common_keys + policy_keys
        }
        rollout["track_id"] = torch.as_tensor(
            [item["track_id"] for item in all_transitions], dtype=torch.long,
        )
        rollout["family_id"] = torch.as_tensor(
            [self.family_ids[item["track_id"]] for item in all_transitions],
            dtype=torch.long,
        )
        if self.policy.action_head_type == "shortcut_flow":
            rollout["cfm_collection_batch_id"] = torch.as_tensor(
                [item["cfm_collection_batch_id"] for item in all_transitions],
                dtype=torch.long,
            )
            rollout["cfm_collection_batch_position"] = torch.as_tensor(
                [item["cfm_collection_batch_position"] for item in all_transitions],
                dtype=torch.long,
            )
        return rollout, results, raw_steps / max(time.perf_counter() - started, 1.0e-6)


def ppo_reward_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Summarize the reward actually optimized without metric explosion."""

    if not results:
        return {}
    steps = max(sum(int(result.get("steps", 0)) for result in results), 1)
    environment_returns = np.asarray(
        [float(result.get("environment_return", result.get("return", 0.0))) for result in results], np.float64,
    )
    adjusted_returns = np.asarray([
        float(result.get("ppo_adjusted_return", result.get("return", 0.0)))
        for result in results
    ], np.float64)
    output = {
        "reward/environment_return_mean": float(environment_returns.mean()),
        "reward/adjusted_return_mean": float(adjusted_returns.mean()),
        "reward/adjustment_mean": float((adjusted_returns - environment_returns).mean()),
        "reward/episode_steps_mean": float(np.mean([
            int(result.get("steps", 0)) for result in results
        ])),
        "reward/missed_gate_termination_rate": float(np.mean([
            bool(result.get("missed_gate_terminated", False)) for result in results
        ])),
        "reward/gate_dwell_termination_rate": float(np.mean([
            bool(result.get("gate_dwell_terminated", False)) for result in results
        ])),
    }
    for name in PPO_REWARD_COMPONENTS:
        total = sum(
            float(result.get("reward_component_sums", {}).get(name, 0.0))
            for result in results
        )
        output[f"reward/{name}_per_step"] = total / steps
    return output


@torch.no_grad()
def collect_ppo(
    policy: PrivilegedMLPPolicy,
    critic: PrivilegedValue,
    normalizer: FeatureNormalizer,
    settings: Mapping[str, Any],
    stage: RacingCurriculumStage,
    *,
    episodes: int,
    seed_base: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], float]:
    policy.eval(); critic.eval()
    parallel = min(int(settings.get("rollout_envs", 16)), episodes)
    all_transitions: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    raw_steps = 0
    started = time.perf_counter()
    episode_start = 0
    while episode_start < episodes:
        batch_count = min(parallel, episodes - episode_start)
        slots: list[RolloutSlot] = []
        try:
            for offset in range(batch_count):
                index = episode_start + offset
                episode_seed = seed_base + 1009 * index
                target_speed = episode_target_speed(stage, episode_seed)
                env = make_env(
                    settings, reward=make_reward(settings, target_speed),
                    track=stage.tracks[index % len(stage.tracks)],
                )
                observation, start = reset_env(
                    env, stage, seed=episode_seed, episode_index=index
                )
                history = CausalHistory(policy.context_steps)
                history.reset_feature(ppo_observation_features(observation, settings))
                slots.append(RolloutSlot(
                    env, observation, history, start, index,
                    completion_gate_count(env.track, stage.target_gates,
                        enabled=bool(settings.get("allow_curriculum_completion_override", False))), stage.max_steps,
                    target_speed,
                ))
            with ThreadPoolExecutor(max_workers=batch_count) as executor:
                while any(not slot.done for slot in slots):
                    active = [slot for slot in slots if not slot.done]
                    raw_history = np.stack([slot.history.array() for slot in active])
                    normalized_history = normalizer.numpy(raw_history)
                    histories = torch.from_numpy(normalized_history).to(device)
                    speed_commands = torch.as_tensor(
                        [slot.target_speed for slot in active],
                        device=device, dtype=histories.dtype,
                    )
                    distribution = policy.distribution(histories, speed_commands)
                    actions, raw_actions = distribution.rsample()
                    old_log_prob = distribution.log_prob(actions, raw_actions)
                    critic_numpy = np.stack([
                        critic_input(
                            history, slot, slot.target_speed,
                            include_history=bool(settings.get(
                                "critic_include_history", False
                            )),
                            include_episode_context=bool(settings.get(
                                "critic_include_episode_context", True
                            )),
                            settings=settings,
                            course_id=stage.tracks.index(slot.env.track.name),
                        )
                        for history, slot in zip(normalized_history, active)
                    ])
                    values = critic(torch.from_numpy(critic_numpy).to(device))
                    action_numpy = actions.float().cpu().numpy()
                    raw_action_numpy = raw_actions.float().cpu().numpy()
                    location_numpy = distribution.location.float().cpu().numpy()
                    log_std_numpy = distribution.log_std.float().cpu().numpy()
                    futures = {
                        index: executor.submit(
                            slot.env.step,
                            ppo_normalized_to_ctbr(action_numpy[index], settings),
                        )
                        for index, slot in enumerate(active)
                    }
                    step_records: list[tuple[RolloutSlot, dict[str, Any], float, bool, dict[str, Any]]] = []
                    for index, slot in enumerate(active):
                        observation, reward, terminated, _, info = futures[index].result()
                        slot.observation = observation
                        slot.history.append_feature(
                            ppo_observation_features(observation, settings)
                        )
                        slot.steps += 1; raw_steps += 1
                        slot.total_return += float(reward)
                        gates = slot.env.tracker.passed_count - slot.start_passed
                        slot.crashed = bool(info.get("ground_contact") or info.get("unity_collision"))
                        slot.done = bool(
                            terminated or gates >= slot.target_gates
                            or slot.steps >= slot.max_steps
                        )
                        terminal_reward, _ = ppo_terminal_reward(
                            settings, gates=gates, target_gates=slot.target_gates,
                            crashed=slot.crashed, done=slot.done,
                        )
                        reward = float(reward) + terminal_reward
                        slot.total_return += terminal_reward
                        step_records.append((
                            slot, observation, reward, bool(terminated), info,
                        ))
                    next_raw = np.stack([slot.history.array() for slot in active])
                    next_normalized = normalizer.numpy(next_raw)
                    next_critic_numpy = np.stack([
                        critic_input(
                            history, slot, slot.target_speed,
                            include_history=bool(settings.get(
                                "critic_include_history", False
                            )),
                            include_episode_context=bool(settings.get(
                                "critic_include_episode_context", True
                            )),
                            settings=settings,
                            course_id=stage.tracks.index(slot.env.track.name),
                        )
                        for history, slot in zip(next_normalized, active)
                    ])
                    next_values = critic(torch.from_numpy(next_critic_numpy).to(device)).cpu().numpy()
                    for index, (slot, next_observation, reward, terminated, _) in enumerate(step_records):
                        gates = slot.env.tracker.passed_count - slot.start_passed
                        terminal = bool(terminated or gates >= slot.target_gates)
                        slot.transitions.append({
                            "history": normalized_history[index].copy(),
                            "critic_input": critic_numpy[index].copy(),
                            "action": action_numpy[index].copy(),
                            "raw_action": raw_action_numpy[index].copy(),
                            "old_location": location_numpy[index].copy(),
                            "old_log_std": log_std_numpy[index].copy(),
                            "old_log_prob": float(old_log_prob[index]),
                            "speed_command": float(slot.target_speed),
                            "reward": reward,
                            "discount": 0.0 if terminal else float(settings.get("discount", 0.997)),
                            "gae_discount": (
                                0.0 if terminal else float(settings.get("discount", 0.997))
                                * float(settings.get("gae_lambda", 0.95))
                            ),
                            "value": float(values[index]),
                            "next_value": 0.0 if terminal else float(next_values[index]),
                            "track_id": stage.tracks.index(slot.env.track.name),
                            "task_delta": (
                                np.asarray(next_observation["task_state"], np.float32)
                                - raw_history[index, -1, :TASK_DIM]
                            ),
                        })
            for slot in slots:
                finish_gae(
                    slot.transitions, float(settings.get("discount", 0.997)),
                    float(settings.get("gae_lambda", 0.95)),
                )
                all_transitions.extend(slot.transitions)
                results.append({
                    "gates": slot.env.tracker.passed_count - slot.start_passed,
                    "crashed": slot.crashed,
                    "return": slot.total_return,
                    "steps": slot.steps,
                    "track": slot.env.track.name,
                    "target_gates": slot.target_gates,
                    "target_speed_mps": slot.target_speed,
                })
        finally:
            for slot in slots:
                slot.env.close()
        episode_start += batch_count
    rollout = {
        key: torch.from_numpy(np.asarray([item[key] for item in all_transitions], np.float32))
        for key in (
            "history", "critic_input", "action", "raw_action",
            "old_location", "old_log_std", "old_log_prob",
            "advantage", "return", "task_delta", "speed_command",
        )
    }
    rollout["track_id"] = torch.as_tensor(
        [item["track_id"] for item in all_transitions], dtype=torch.long,
    )
    return rollout, results, raw_steps / max(time.perf_counter() - started, 1e-6)


def gaussian_kl(current, reference) -> torch.Tensor:
    variance_ratio = current.std.square() / reference.std.square()
    mean_term = (current.location - reference.location).square() / reference.std.square()
    return 0.5 * (
        variance_ratio + mean_term - 1.0
        + 2.0 * (reference.log_std - current.log_std)
    ).sum(-1)


def normalize_dynamics_targets(
    task_deltas: torch.Tensor,
    mean: np.ndarray | torch.Tensor,
    std: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Normalize on-policy transitions with the original offline contract."""

    target_mean = torch.as_tensor(mean, dtype=task_deltas.dtype)
    target_std = torch.as_tensor(std, dtype=task_deltas.dtype).clamp_min(1.0e-4)
    return (task_deltas - target_mean) / target_std


def privileged_flow_cfm_loss(
    policy: PrivilegedMLPPolicy,
    histories: torch.Tensor,
    speed_commands: torch.Tensor | None,
    action_chunks: torch.Tensor,
    epsilon: torch.Tensor,
    flow_time: torch.Tensor,
    shortcut_step: torch.Tensor,
    valid_steps: torch.Tensor,
    *,
    huber_delta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-condition CFM loss for a privileged receding-horizon policy.

    The sampled four-command chunk is the joint latent action while command zero
    is the deterministic plant action.  Scoring the joint sample gives a valid
    score-function estimator for rewards that depend only on command zero;
    masking its coupled tail would not yield the command-zero marginal.
    """

    if policy.action_head_type != "shortcut_flow" or policy.flow_action_head is None:
        raise ValueError("privileged FPO++ requires a shortcut-flow policy")
    if action_chunks.ndim != 3:
        raise ValueError("FPO action chunks must have shape [B,H,A]")
    batch, horizon, action_dim = action_chunks.shape
    if (horizon, action_dim) != (policy.action_chunk_steps, 4):
        raise ValueError("FPO action chunk does not match the policy contract")
    if epsilon.ndim != 4 or epsilon.shape[0] != batch or epsilon.shape[2:] != (
        horizon, action_dim
    ):
        raise ValueError("FPO epsilon must have shape [B,N,H,A]")
    samples = epsilon.shape[1]
    if flow_time.shape != (batch, samples) or shortcut_step.shape != (
        batch, samples
    ):
        raise ValueError("FPO time conditions must have shape [B,N]")
    if valid_steps.shape != (batch, horizon) or not bool(
        torch.all(valid_steps.any(dim=1))
    ):
        raise ValueError("every FPO latent action needs a valid command")

    context, _, _ = policy._flow_context(histories, speed_commands)
    noisy = epsilon + flow_time[..., None, None] * (
        action_chunks[:, None] - epsilon
    )
    target = action_chunks[:, None] - epsilon
    expanded_context = context[:, None].expand(
        batch, samples, *context.shape[1:]
    )
    predicted = policy.flow_action_head.velocity_from_context(
        noisy.flatten(0, 1),
        flow_time.flatten(),
        shortcut_step.flatten(),
        expanded_context.flatten(0, 1),
    ).reshape(batch, samples, horizon, action_dim)
    difference = predicted.float() - target.float()
    if huber_delta is None:
        element = difference.square()
    else:
        delta = float(huber_delta)
        if delta <= 0.0:
            raise ValueError("fpo_cfm_huber_delta must be positive")
        absolute = difference.abs()
        element = torch.where(
            absolute <= delta,
            0.5 * difference.square(),
            delta * (absolute - 0.5 * delta),
        )
    per_step = element.mean(-1)
    loss = (per_step * valid_steps[:, None].float()).sum(-1)
    return loss, predicted


@torch.no_grad()
def privileged_fpo_contract_error(
    rollout: Mapping[str, torch.Tensor],
    policy: PrivilegedMLPPolicy,
    *,
    device: str,
    maximum_samples: int = 2048,
    inference_batch_size: int = 64,
    huber_delta: float | None = None,
) -> float:
    """Recompute stored behavior CFM losses before any optimizer update."""

    required = {
        "history", "speed_command", "action_chunk", "epsilon",
        "flow_time", "shortcut_step", "valid", "old_cfm",
    }
    if not required.issubset(rollout):
        raise ValueError("FPO rollout is missing its behavior-flow contract")
    count = min(len(rollout["history"]), int(maximum_samples))
    if count < 1:
        raise ValueError("FPO contract check requires transitions")
    collection_batches = rollout.get("cfm_collection_batch_id")
    collection_positions = rollout.get("cfm_collection_batch_position")
    grouped_indices: list[torch.Tensor] | None = None
    if collection_batches is not None and collection_positions is not None:
        batch_ids = torch.unique(collection_batches.long(), sorted=True)
        group_counts = torch.stack([
            (collection_batches == batch_id).sum() for batch_id in batch_ids
        ])
        maximum_group = max(int(group_counts.max()), 1)
        group_limit = max(count // maximum_group, 1)
        chosen = batch_ids[torch.linspace(
            0, len(batch_ids) - 1,
            min(len(batch_ids), group_limit), dtype=torch.long,
        )]
        grouped_indices = []
        for batch_id in chosen:
            group = torch.nonzero(
                collection_batches == batch_id, as_tuple=False
            ).flatten()
            order = torch.argsort(collection_positions[group])
            grouped_indices.append(group[order])
    else:
        indices = torch.linspace(
            0, len(rollout["history"]) - 1, count, dtype=torch.long,
        )
        grouped_indices = list(indices.split(max(int(inference_batch_size), 1)))
    differences: list[torch.Tensor] = []
    for chunk in grouped_indices:
        current, _ = privileged_flow_cfm_loss(
            policy,
            rollout["history"][chunk].to(device),
            rollout["speed_command"][chunk].to(device),
            rollout["action_chunk"][chunk].to(device),
            rollout["epsilon"][chunk].to(device),
            rollout["flow_time"][chunk].to(device),
            rollout["shortcut_step"][chunk].to(device),
            rollout["valid"][chunk].to(device),
            huber_delta=huber_delta,
        )
        differences.append(
            (current - rollout["old_cfm"][chunk].to(device)).abs().flatten()
        )
    return float(torch.cat(differences).max().cpu())


@torch.no_grad()
def ppo_log_prob_contract_error(
    rollout: Mapping[str, torch.Tensor],
    policy: PrivilegedMLPPolicy,
    *,
    device: str,
    maximum_samples: int = 4096,
    inference_batch_size: int = 16,
    exploration_correlation: float = 0.0,
    update_kernel: bool = False,
) -> tuple[float, float, float, float]:
    """Verify that a rollout's stored behavior likelihood is reproducible.

    PPO must evaluate the exact pre-squash Gaussian sample used at collection.
    Inverting a float32 ``tanh`` action is not equivalent near saturation.
    """

    required = {"raw_action", "old_location", "old_log_std"}
    if not required.issubset(rollout):
        raise ValueError("PPO rollout is missing stored behavior parameters")
    count = min(len(rollout["history"]), int(maximum_samples))
    if count < 1:
        raise ValueError("PPO likelihood validation requires transitions")
    indices = torch.linspace(
        0, len(rollout["history"]) - 1, count, dtype=torch.long,
        device=rollout["history"].device,
    )
    actions = rollout["action"][indices].to(device)
    raw_actions = rollout["raw_action"][indices].to(device)
    behavior = SquashedGaussian(
        rollout["old_location"][indices].to(device),
        rollout["old_log_std"][indices].to(device),
    )
    stored_log_prob = behavior.log_prob(actions, raw_actions)
    recorded_log_prob = rollout["old_log_prob"][indices].to(device)
    storage_error = (stored_log_prob - recorded_log_prob).abs()
    current_parts: list[torch.Tensor] = []
    batch_size = max(int(inference_batch_size), 1)
    for chunk in indices.split(batch_size):
        histories = rollout["history"][chunk].to(device)
        speed_commands = rollout.get("speed_command")
        speed_commands = (
            speed_commands[chunk].to(device) if speed_commands is not None else None
        )
        with torch.set_grad_enabled(update_kernel):
            current_parts.append(policy.distribution(histories, speed_commands).location.detach())
    current_location = torch.cat(current_parts)
    base = SquashedGaussian(
        current_location,
        policy.log_std_parameter.clamp(
            policy.minimum_log_std, policy.maximum_log_std,
        ),
    )
    offsets = rollout.get("exploration_offset")
    if offsets is None:
        if float(exploration_correlation) != 0.0:
            raise ValueError(
                "correlated PPO rollout is missing exploration_offset"
            )
        selected_offsets = torch.zeros_like(current_location)
    else:
        selected_offsets = offsets[indices].to(device)
    current = correlated_ppo_distribution(
        base, selected_offsets, exploration_correlation,
    )
    current_log_prob = current.log_prob(
        actions, raw_actions,
    )
    difference = current_log_prob - recorded_log_prob
    error = difference.abs()
    ratio = difference.clamp(-10.0, 10.0).exp()
    approximate_kl = ((ratio - 1.0) - difference.clamp(-10.0, 10.0)).mean()
    return (
        float(storage_error.max().cpu()),
        float(error.mean().cpu()),
        float(approximate_kl.cpu()),
        float(gaussian_kl(current, behavior).mean().cpu()),
    )


def ppo_update_rollout(
    rollout: Mapping[str, torch.Tensor], settings: Mapping[str, Any], device: str,
) -> dict[str, torch.Tensor]:
    """Stage one PPO window on the accelerator once instead of per minibatch.

    Window-mode PPO revisits the same immutable behavior data for several actor
    and critic epochs.  Copying each indexed minibatch separately repeats host
    indexing and H2D synchronization hundreds of times.  The complete compact
    window is small relative to the training activations on supported GPUs, so
    an opt-in resident copy amortizes transport while preserving every dtype and
    value.  The CPU rollout remains owned by the collector and is released at
    the normal cycle boundary.
    """

    if not bool(settings.get("ppo_device_resident_updates", False)):
        return dict(rollout)
    target = torch.device(device)
    if target.type != "cuda":
        return dict(rollout)
    byte_limit = int(settings.get("ppo_device_resident_max_bytes", 2 * 1024**3))
    total = sum(tensor.numel() * tensor.element_size() for tensor in rollout.values())
    if byte_limit < 1 or total > byte_limit:
        raise MemoryError(
            "PPO device-resident rollout exceeds configured budget: "
            f"required={total:,} limit={byte_limit:,}"
        )
    # One copy per dense field and one synchronization for the whole window.
    # non_blocking is effective when a future collector provides pinned tensors,
    # and harmless for the current NumPy-backed CPU storage.
    staged = {
        name: tensor.to(target, non_blocking=True)
        for name, tensor in rollout.items()
    }
    torch.cuda.current_stream(target).synchronize()
    return staged


def ppo_epoch_kl_minibatches(
    strata: torch.Tensor,
    batch_size: int,
    weights: Mapping[int, float] | None,
    settings: Mapping[str, Any],
) -> list[torch.Tensor]:
    """Return a bounded, stratified trust-region audit sample.

    The behavior-contract audit still validates exact likelihood storage.  This
    sample is only the post-epoch stopping/LR statistic; evaluating all 200k+
    transitions after every epoch duplicates a large actor inference pass with
    little statistical benefit.
    """

    requested = settings.get("ppo_epoch_kl_samples")
    if requested is None:
        return stratified_minibatches(strata, batch_size, weights)
    count = int(requested)
    if count < 1:
        raise ValueError("ppo_epoch_kl_samples must be positive")
    return stratified_minibatches(
        strata, min(count, len(strata)), weights, maximum_batches=1,
    )


def update_ppo(
    rollout: dict[str, torch.Tensor],
    policy: PrivilegedMLPPolicy,
    reference: PrivilegedMLPPolicy,
    critic: PrivilegedValue,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    settings: Mapping[str, Any],
    *,
    device: str,
    anchor_weight: float,
    constraint_lambdas: torch.Tensor | None = None,
) -> dict[str, float]:
    from starscream.ppo_update import (PPOBatchPlan, UpdateTimer, configure_critic_acceleration,
                                      configure_actor_acceleration)
    timer = UpdateTimer(bool(settings.get('ppo_profile_updates', False)), device)
    configure_critic_acceleration(critic, settings)
    configure_actor_acceleration(policy, settings)
    epoch_mode = str(settings.get('ppo_epoch_mode', 'stratified'))
    if epoch_mode not in {'stratified', 'weighted_single_pass'}:
        raise ValueError('ppo_epoch_mode must be stratified or weighted_single_pass')
    weighted_pass = epoch_mode == 'weighted_single_pass'
    if weighted_pass and settings.get('actor_early_stop_scope', 'epoch') == 'minibatch':
        raise ValueError('weighted_single_pass requires complete epochs before KL stopping')
    if weighted_pass and settings.get('ppo_minibatches_per_epoch') is not None:
        raise ValueError('weighted_single_pass cannot truncate a complete epoch')
    if weighted_pass and (anchor_weight > 0 or constraint_lambdas is not None
                          or float(settings.get('ppo_dynamics_weight', 0)) > 0):
        raise ValueError('weighted_single_pass currently requires plain unanchored PPO')
    rollout = ppo_update_rollout(rollout, settings, device)
    timer.mark('staging_seconds')
    for key in ('advantage','return','old_log_prob'):
        if not bool(torch.isfinite(rollout[key]).all()):
            raise FloatingPointError(f'nonfinite PPO rollout {key}')
    critic_loss_kind = str(settings.get('critic_loss','huber'))
    if critic_loss_kind not in {'huber','mse'}:
        raise ValueError('critic_loss must be mse or huber')
    raw_advantages = rollout["advantage"]
    track_ids = rollout["track_id"]
    raw_track_weights = rollout.get("track_weight")
    track_weights: dict[int, float] | None = None
    if raw_track_weights is not None:
        track_weights = {}
        for track_id in torch.unique(track_ids, sorted=True):
            mask = track_ids == track_id
            values = raw_track_weights[mask]
            if not torch.allclose(values, values[:1]):
                raise ValueError("PPO track weights must be constant within each track")
            track_weights[int(track_id)] = float(values[0])
    stratification_ids, stratification_weights = ppo_minibatch_strata(
        track_ids,
        rollout.get("rollout_laps") if bool(
            settings.get("ppo_stratify_minibatches_by_lap", False)
        ) else None,
        track_weights,
        settings.get("ppo_minibatch_lap_weights"),
    )
    batch_plan = (PPOBatchPlan(stratification_ids, stratification_weights)
                  if weighted_pass or settings.get('ppo_cached_minibatches', False) else None)

    def epoch_batches(size, maximum):
        with timer.phase('sampling_seconds'):
            if weighted_pass:
                return batch_plan.single_pass(size)
            if batch_plan is not None:
                return batch_plan.stratified(size, maximum)
            return stratified_minibatches(stratification_ids, size, stratification_weights,
                                          maximum_batches=maximum)

    def weighted_mean(value, weights):
        return value.mean() if weights is None else (value * weights).mean()

    raw_advantage_std = raw_advantages.std(unbiased=False).clamp_min(1e-6)
    advantages = normalize_ppo_advantages(
        raw_advantages, stratification_ids,
        per_track=bool(settings.get("normalize_advantages_per_track", False)),
    )
    constraint_advantages: torch.Tensor | None = None
    family_ids = rollout.get("family_id")
    device_constraint_lambdas: torch.Tensor | None = None
    if constraint_lambdas is not None:
        if family_ids is None or "failure_return" not in rollout:
            raise ValueError("group-constrained PPO requires family IDs and failure returns")
        if int(family_ids.max()) >= len(constraint_lambdas):
            raise ValueError("PPO constraint lambda vector does not cover rollout families")
        constraint_advantages = normalize_advantages_by_track(
            rollout["failure_return"], family_ids,
        )
        device_constraint_lambdas = constraint_lambdas.to(device)
    count = len(advantages)
    batch_size = min(int(settings.get("minibatch_size", 1024)), count)
    maximum_minibatches = settings.get("ppo_minibatches_per_epoch")
    maximum_minibatches = (
        None if maximum_minibatches is None else int(maximum_minibatches)
    )
    clip = float(settings.get("clip_epsilon", 0.20))
    target_kl = float(settings.get("target_kl", 0.02))
    dynamics_weight = float(settings.get("ppo_dynamics_weight", 0.0))
    if dynamics_weight > 0.0 and "dynamics_target" not in rollout:
        raise ValueError("ppo_dynamics_weight requires rollout dynamics_target")
    (
        contract_storage_error,
        contract_mean_error,
        contract_approximate_kl,
        contract_behavior_kl,
        ) = (
        ppo_log_prob_contract_error(
        rollout, policy, device=device,
        maximum_samples=int(settings.get("ppo_likelihood_check_samples", 4096)),
        inference_batch_size=int(settings.get("rollout_envs", 16)),
        exploration_correlation=configured_ppo_exploration_correlation(settings),
        )
    )
    storage_tolerance = float(settings.get(
        "ppo_likelihood_storage_tolerance", 2.0e-5,
    ))
    kl_tolerance = float(settings.get(
        "ppo_likelihood_check_kl_tolerance", 1.0e-4,
    ))
    if (
        not all(np.isfinite(x) for x in (contract_storage_error,contract_mean_error,
            contract_approximate_kl,contract_behavior_kl))
        or contract_storage_error > storage_tolerance
        or contract_behavior_kl > kl_tolerance
    ):
        raise RuntimeError(
            "PPO behavior likelihood contract is not reproducible: "
            f"storage_error={contract_storage_error:.3e} "
            f"mean_error={contract_mean_error:.3e} "
            f"approximate_kl={contract_approximate_kl:.3e} "
            f"behavior_kl={contract_behavior_kl:.3e}"
        )
    metrics: dict[str, list[float]] = {
        "raw_advantage_mean": [float(raw_advantages.mean())],
        "raw_advantage_std": [float(raw_advantage_std)],
        "return_std": [float(rollout["return"].std(unbiased=False))],
        "likelihood_contract_storage_error": [contract_storage_error],
        "likelihood_contract_mean_error": [contract_mean_error],
        "likelihood_contract_approximate_kl": [contract_approximate_kl],
        "likelihood_contract_behavior_kl": [contract_behavior_kl],
    }
    if settings.get('ppo_audit_update_kernel', False) and int(settings.get('actor_epochs', 6)) > 0:
        previous_mode = policy.training
        policy.train()
        try:
            _, update_error, _, update_kl = ppo_log_prob_contract_error(
                rollout, policy, device=device, maximum_samples=batch_size,
                inference_batch_size=batch_size,
                exploration_correlation=configured_ppo_exploration_correlation(settings),
                update_kernel=True)
        finally:
            policy.train(previous_mode)
        if not np.isfinite(update_kl) or update_kl > kl_tolerance:
            raise RuntimeError(f'PPO update-kernel behavior KL is not reproducible: {update_kl:.3e}')
        metrics['likelihood_update_kernel_behavior_kl'] = [update_kl]
        metrics['likelihood_update_kernel_mean_error'] = [update_error]
    stop = False
    completed_actor_epochs = 0
    completed_actor_updates = 0
    actor_samples = 0
    critic_samples = 0
    critic_updates = 0
    if 'old_value' in rollout:
        value_error = rollout['old_value'] - rollout['return']
        variance = rollout['return'].var(unbiased=False)
        metrics['critic_prefit_gae_target_rmse'] = [float(value_error.square().mean().sqrt())]
        if float(variance) > 1e-8:
            metrics['critic_prefit_gae_target_explained_variance'] = [float(
                1.0-value_error.var(unbiased=False)/variance)]
    from starscream.ppo_windows import actor_early_stop_scope as _early_stop_scope
    early_stop_scope = _early_stop_scope(settings)
    defer_update_metrics = bool(settings.get("defer_update_metrics", False))
    timer.mark('setup_and_contract_seconds')
    policy.train()
    for _ in range(int(settings.get("actor_epochs", 6))):
        for indices in epoch_batches(batch_size, maximum_minibatches):
            sample_weights = batch_plan.loss_weights[indices] if weighted_pass else None
            histories = rollout["history"][indices].to(device)
            speed_commands = rollout.get("speed_command")
            speed_commands = (
                speed_commands[indices].to(device) if speed_commands is not None else None
            )
            actions = rollout["action"][indices].to(device)
            raw_actions = rollout["raw_action"][indices].to(device)
            old_log_prob = rollout["old_log_prob"][indices].to(device)
            advantage = advantages[indices].to(device)
            constraint_penalty = torch.zeros_like(advantage)
            if constraint_advantages is not None:
                assert family_ids is not None and device_constraint_lambdas is not None
                multipliers = device_constraint_lambdas[family_ids[indices]]
                constraint_penalty = (
                    float(settings.get("ppo_constraint_advantage_scale", 1.0))
                    * multipliers
                    * constraint_advantages[indices].to(device)
                )
                advantage = advantage - constraint_penalty
            actor_optimizer.zero_grad(set_to_none=True)
            base_distribution = policy.distribution(histories, speed_commands)
            offsets = rollout.get("exploration_offset")
            if offsets is None:
                if configured_ppo_exploration_correlation(settings) != 0.0:
                    raise ValueError(
                        "correlated PPO rollout is missing exploration_offset"
                    )
                exploration_offsets = torch.zeros_like(
                    base_distribution.location
                )
            else:
                exploration_offsets = offsets[indices].to(device)
            distribution = correlated_ppo_distribution(
                base_distribution,
                exploration_offsets,
                configured_ppo_exploration_correlation(settings),
            )
            log_prob = distribution.log_prob(actions, raw_actions)
            if not bool(torch.isfinite(log_prob).all()):
                raise FloatingPointError('nonfinite PPO action likelihood')
            saturation_threshold = settings.get("ppo_saturation_mask_threshold")
            if saturation_threshold is None:
                log_ratio = (log_prob - old_log_prob).clamp(-10.0, 10.0)
                masked_fraction = torch.zeros((), device=device)
            else:
                log_ratio, masked_fraction = saturation_masked_log_ratio(
                    distribution, actions, raw_actions,
                    rollout["old_location"][indices].to(device),
                    rollout["old_log_std"][indices].to(device),
                    float(saturation_threshold),
                )
                log_ratio = log_ratio.clamp(-10.0, 10.0)
            ratio = log_ratio.exp()
            approximate_kl = weighted_mean((ratio - 1.0) - log_ratio, sample_weights)
            clip_fraction = weighted_mean(((ratio - 1.0).abs() > clip).float(), sample_weights)
            # A minibatch-scoped trust-region guard must run before the next
            # optimizer step. The old ordering detected an already-violating
            # policy and then applied one additional update before stopping,
            # which allowed a single cycle to overshoot the configured KL by
            # several times on long sequence-policy batches.
            if (
                early_stop_scope == "minibatch" and target_kl > 0
                and float(approximate_kl.detach()) > target_kl
            ):
                metrics.setdefault("approximate_kl", []).append(
                    float(approximate_kl.detach())
                )
                metrics.setdefault("clip_fraction", []).append(
                    float(clip_fraction.detach())
                )
                stop = True
                break
            surrogate = torch.minimum(
                ratio * advantage,
                ratio.clamp(1.0 - clip, 1.0 + clip) * advantage,
            )
            kl_anchor = torch.zeros((),device=device)
            if anchor_weight > 0.0:
                with torch.no_grad():
                    prior = reference.distribution(histories, speed_commands)
                kl_anchor = gaussian_kl(base_distribution, prior).mean()
            entropy = weighted_mean(distribution.base_entropy(), sample_weights)
            dynamics_loss = torch.zeros((), device=device)
            if dynamics_weight > 0.0:
                dynamics_prediction = policy.predict_dynamics(histories)
                dynamics_per_sample = F.smooth_l1_loss(
                    dynamics_prediction,
                    rollout["dynamics_target"][indices].to(device),
                    beta=float(settings.get("ppo_dynamics_beta", 0.10)),
                    reduction="none",
                ).mean(-1)
                dynamics_mask = rollout.get("dynamics_valid")
                dynamics_mask = (
                    torch.ones_like(dynamics_per_sample)
                    if dynamics_mask is None
                    else dynamics_mask[indices].to(
                        device=device, dtype=dynamics_per_sample.dtype,
                    )
                )
                dynamics_loss = (
                    dynamics_per_sample * dynamics_mask
                ).sum() / dynamics_mask.sum().clamp_min(1.0)
            loss = (
                -weighted_mean(surrogate, sample_weights)
                + anchor_weight * kl_anchor
                + dynamics_weight * dynamics_loss
                - float(settings.get("entropy_weight", 0.001)) * entropy
            )
            loss.backward()
            if (
                "actor_backbone_gradient_clip" in settings
                or "actor_head_gradient_clip" in settings
            ):
                backbone_gradient = torch.nn.utils.clip_grad_norm_(
                    actor_optimizer.param_groups[0]["params"],
                    float(settings.get("actor_backbone_gradient_clip", 1.0)),
                )
                head_parameters = [
                    parameter
                    for group in actor_optimizer.param_groups[1:]
                    for parameter in group["params"]
                ]
                head_gradient = torch.nn.utils.clip_grad_norm_(
                    head_parameters,
                    float(settings.get("actor_head_gradient_clip", 1.0)),
                )
                gradient = torch.maximum(
                    torch.as_tensor(backbone_gradient), torch.as_tensor(head_gradient)
                )
            else:
                gradient = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float(settings.get("actor_gradient_clip", 1.0))
                )
                backbone_gradient = gradient
                head_gradient = gradient
            actor_optimizer.step()
            completed_actor_updates += 1
            actor_samples += len(indices)
            if settings.get('ppo_noise_bounds') and bool(settings.get('train_log_std', True)):
                from starscream.ppo_exploration import project_noise_bounds
                project_noise_bounds(settings, policy)
            values = {
                "policy_loss": -weighted_mean(surrogate, sample_weights), "anchor_kl": kl_anchor,
                "constraint_advantage_penalty": constraint_penalty.mean(),
                "dynamics_loss": dynamics_loss,
                "entropy": entropy, "ratio": ratio.mean(),
                "ratio_std": ratio.std(unbiased=False), "approximate_kl": approximate_kl,
                "clip_fraction": clip_fraction,
                "saturation_masked_fraction": masked_fraction,
                "actor_gradient_norm": gradient,
                "actor_backbone_gradient_norm": backbone_gradient,
                "actor_head_gradient_norm": head_gradient,
            }
            for name, value in values.items():
                metrics.setdefault(name, []).append(
                    value.detach() if defer_update_metrics else float(value.detach()))
        if stop:
            break
        completed_actor_epochs += 1
        if early_stop_scope == "epoch" and target_kl > 0:
            epoch_kl_values: list[float] = []
            policy.eval()
            with torch.no_grad():
                if batch_plan is not None and settings.get('ppo_epoch_kl_samples') is not None:
                    probe_count = int(settings['ppo_epoch_kl_samples'])
                    if probe_count < 1:
                        raise ValueError('ppo_epoch_kl_samples must be positive')
                    probe_batches = batch_plan.stratified(min(probe_count, count), 1)
                else:
                    probe_batches = ppo_epoch_kl_minibatches(
                        stratification_ids, batch_size, stratification_weights, settings)
                for indices in probe_batches:
                    histories = rollout["history"][indices].to(device)
                    speed_commands = rollout.get("speed_command")
                    speed_commands = (
                        speed_commands[indices].to(device)
                        if speed_commands is not None else None
                    )
                    actions = rollout["action"][indices].to(device)
                    raw_actions = rollout["raw_action"][indices].to(device)
                    old_log_prob = rollout["old_log_prob"][indices].to(device)
                    base_distribution = policy.distribution(
                        histories, speed_commands,
                    )
                    offsets = rollout.get("exploration_offset")
                    exploration_offsets = (
                        torch.zeros_like(base_distribution.location)
                        if offsets is None
                        else offsets[indices].to(device)
                    )
                    distribution = correlated_ppo_distribution(
                        base_distribution,
                        exploration_offsets,
                        configured_ppo_exploration_correlation(settings),
                    )
                    saturation_threshold = settings.get(
                        "ppo_saturation_mask_threshold"
                    )
                    if saturation_threshold is None:
                        log_prob = distribution.log_prob(actions, raw_actions)
                        log_ratio = (log_prob - old_log_prob).clamp(-10.0, 10.0)
                    else:
                        log_ratio, _ = saturation_masked_log_ratio(
                            distribution, actions, raw_actions,
                            rollout["old_location"][indices].to(device),
                            rollout["old_log_std"][indices].to(device),
                            float(saturation_threshold),
                        )
                        log_ratio = log_ratio.clamp(-10.0, 10.0)
                    ratio = log_ratio.exp()
                    epoch_kl_values.append(float(
                        (((ratio - 1.0) - log_ratio).mean()).cpu()
                    ))
            policy.train()
            epoch_kl = float(np.mean(epoch_kl_values))
            metrics.setdefault("epoch_approximate_kl", []).append(epoch_kl)
            if epoch_kl > target_kl:
                stop = True
                break
    critic.train()
    timer.mark('actor_and_kl_seconds')
    critic_batch_size = int(settings.get("critic_minibatch_size", batch_size))
    if critic_batch_size < 1:
        raise ValueError("critic_minibatch_size must be positive")
    normalize_critic = bool(settings.get(
        "critic_normalize_loss_by_return_std", False,
    ))
    per_track_critic_scale = bool(settings.get(
        "critic_normalize_loss_per_track", False,
    ))
    critic_return_scale = (
        ppo_critic_loss_scales(
            rollout["return"], track_ids, per_track=per_track_critic_scale,
            minimum_global_fraction=float(settings.get(
                "critic_per_track_scale_floor_fraction", 0.1,
            )),
        )
        if normalize_critic or per_track_critic_scale
        else torch.ones((), dtype=rollout["return"].dtype)
    )
    for _ in range(int(settings.get("critic_epochs", 6))):
        for indices in epoch_batches(critic_batch_size, maximum_minibatches):
            critic_optimizer.zero_grad(set_to_none=True)
            predicted = critic(rollout["critic_input"][indices].to(device))
            target = rollout["return"][indices].to(device)
            scale = (
                critic_return_scale[indices].to(device)
                if critic_return_scale.ndim else critic_return_scale.to(device)
            )
            normalized_error = (predicted - target) / scale
            loss = (
                normalized_error.square()
                if critic_loss_kind == 'mse'
                else F.smooth_l1_loss(
                    normalized_error, torch.zeros_like(normalized_error), reduction='none'
                )
            )
            loss = weighted_mean(loss, batch_plan.loss_weights[indices] if weighted_pass else None)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite PPO critic loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), float(settings.get("critic_gradient_clip", 5.0))
            )
            critic_optimizer.step()
            critic_samples += len(indices)
            critic_updates += 1
            metrics.setdefault("critic_loss", []).append(
                loss.detach() if defer_update_metrics else float(loss.detach()))
            metrics.setdefault("critic_gradient_norm", []).append(
                gradient.detach() if defer_update_metrics else float(gradient))
    timer.mark('critic_seconds')
    critic.eval()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for indices in torch.arange(
            count, device=rollout["critic_input"].device,
        ).split(batch_size):
            predictions.append(
                critic(rollout["critic_input"][indices].to(device)).to(
                    rollout["return"].device
                )
            )
    predicted_values = torch.cat(predictions)
    return_variance = rollout["return"].var(unbiased=False)
    explained_variance = (
        1.0 - (rollout["return"] - predicted_values).var(unbiased=False)
        / return_variance.clamp_min(1.0e-8)
    )
    metrics["critic_explained_variance"] = [float(explained_variance)]
    metrics["critic_postfit_gae_target_explained_variance"] = [float(explained_variance)]
    critic_error = predicted_values - rollout["return"]
    metrics["critic_postfit_rmse"] = [float(critic_error.square().mean().sqrt())]
    metrics["critic_postfit_bias"] = [float(critic_error.mean())]
    metrics["critic_prediction_std"] = [float(predicted_values.std(unbiased=False))]
    metrics["critic_return_scale"] = [float(critic_return_scale.mean())]
    metrics["critic_return_scale_min"] = [float(critic_return_scale.min())]
    metrics["critic_return_scale_max"] = [float(critic_return_scale.max())]
    per_track_ev = []
    for track_id in torch.unique(track_ids, sorted=True):
        mask = track_ids == track_id
        if int(mask.sum()) < 2:
            continue
        variance = rollout["return"][mask].var(unbiased=False)
        if float(variance) > 1.0e-8:
            per_track_ev.append(float(
                1.0 - critic_error[mask].var(unbiased=False) / variance
            ))
    metrics["critic_per_track_explained_variance_mean"] = [
        float(np.mean(per_track_ev)) if per_track_ev else 0.0
    ]
    metrics["actor_early_stop"] = [float(stop)]
    metrics["actor_epochs_completed"] = [float(completed_actor_epochs)]
    metrics["actor_updates_completed"] = [float(completed_actor_updates)]
    metrics['actor_sample_presentations'] = [float(actor_samples)]
    metrics['critic_sample_presentations'] = [float(critic_samples)]
    metrics['critic_updates_completed'] = [float(critic_updates)]
    metrics['actor_sample_passes'] = [actor_samples / count]
    metrics['critic_sample_passes'] = [critic_samples / count]
    if weighted_pass:
        weights = batch_plan.loss_weights
        metrics['objective_weight_max'] = [float(weights.max())]
        metrics['objective_weight_ess_fraction'] = [float(weights.sum().square() / weights.square().sum() / count)]
    for index, name in enumerate(('collective', 'roll', 'pitch', 'yaw')):
        metrics[f'exploration_{name}_log_std'] = [float(policy.log_std_parameter[index].detach())]
    metrics["rollout_transitions"] = [float(count)]
    for track_id in torch.unique(track_ids, sorted=True):
        metrics[f"rollout_track_{int(track_id)}_transitions"] = [
            float((track_ids == track_id).sum())
        ]
    if "rollout_laps" in rollout:
        for laps in torch.unique(rollout["rollout_laps"], sorted=True):
            metrics[f"rollout_lap_{int(laps)}_transitions"] = [
                float((rollout["rollout_laps"] == laps).sum())
            ]
    metrics["advantage_mean"] = [float(advantages.mean())]
    metrics["return_mean"] = [float(rollout["return"].mean())]
    scalar_lists_to_host(metrics)
    timer.mark('diagnostics_seconds')
    output = {name: float(np.mean(values)) for name, values in metrics.items()}
    output.update({f'update_profile/{key}': value for key, value in timer.metrics.items()})
    if metrics.get('epoch_approximate_kl'):
        output['epoch_approximate_kl_last'] = float(metrics['epoch_approximate_kl'][-1])
    output["approximate_kl_max"] = float(max(metrics.get("approximate_kl", [0.0])))
    return output


def update_privileged_fpo(
    rollout: dict[str, torch.Tensor],
    policy: PrivilegedMLPPolicy,
    reference: PrivilegedMLPPolicy,
    critic: PrivilegedValue,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    settings: Mapping[str, Any],
    *,
    device: str,
    anchor_weight: float,
    constraint_lambdas: torch.Tensor | None = None,
) -> dict[str, float]:
    """FPO++/ASPO update for the privileged four-command flow policy."""

    if policy.action_head_type != "shortcut_flow":
        raise ValueError("privileged FPO++ requires a shortcut-flow checkpoint")
    raw_advantages = rollout["advantage"]
    track_ids = rollout["track_id"]
    raw_track_weights = rollout.get("track_weight")
    track_weights: dict[int, float] | None = None
    if raw_track_weights is not None:
        track_weights = {}
        for track_id in torch.unique(track_ids, sorted=True):
            mask = track_ids == track_id
            values = raw_track_weights[mask]
            if not torch.allclose(values, values[:1]):
                raise ValueError("FPO track weights must be constant within a track")
            track_weights[int(track_id)] = float(values[0])
    stratification_ids, stratification_weights = ppo_minibatch_strata(
        track_ids,
        rollout.get("rollout_laps") if bool(
            settings.get("ppo_stratify_minibatches_by_lap", False)
        ) else None,
        track_weights,
        settings.get("ppo_minibatch_lap_weights"),
    )
    advantages = normalize_ppo_advantages(
        raw_advantages, stratification_ids,
        per_track=bool(settings.get("normalize_advantages_per_track", False)),
    )
    constraint_advantages: torch.Tensor | None = None
    family_ids = rollout.get("family_id")
    if constraint_lambdas is not None:
        if family_ids is None or "failure_return" not in rollout:
            raise ValueError("group-constrained FPO requires family failure returns")
        if int(family_ids.max()) >= len(constraint_lambdas):
            raise ValueError("FPO constraint vector does not cover rollout families")
        constraint_advantages = normalize_advantages_by_track(
            rollout["failure_return"], family_ids
        )
    contract_error = privileged_fpo_contract_error(
        rollout, policy, device=device,
        maximum_samples=int(settings.get("fpo_likelihood_check_samples", 2048)),
        inference_batch_size=int(settings.get("rollout_envs", 16)),
        huber_delta=settings.get("fpo_cfm_huber_delta"),
    )
    tolerance = float(settings.get("fpo_likelihood_storage_tolerance", 2.0e-5))
    if contract_error > tolerance:
        raise RuntimeError(
            "FPO behavior CFM contract is not reproducible: "
            f"maximum_error={contract_error:.3e} tolerance={tolerance:.3e}"
        )

    count = len(advantages)
    batch_size = min(int(settings.get("minibatch_size", 1024)), count)
    maximum_minibatches = settings.get("ppo_minibatches_per_epoch")
    maximum_minibatches = (
        None if maximum_minibatches is None else int(maximum_minibatches)
    )
    target_kl = float(settings.get("target_kl", 0.02))
    dynamics_weight = float(settings.get("ppo_dynamics_weight", 0.0))
    if dynamics_weight > 0.0 and "dynamics_target" not in rollout:
        raise ValueError("ppo_dynamics_weight requires rollout dynamics_target")
    metrics: dict[str, list[float]] = {
        "raw_advantage_mean": [float(raw_advantages.mean())],
        "raw_advantage_std": [float(raw_advantages.std(unbiased=False))],
        "return_std": [float(rollout["return"].std(unbiased=False))],
        "likelihood_contract_storage_error": [contract_error],
    }
    from starscream.ppo_windows import actor_early_stop_scope as _early_stop_scope
    early_stop_scope = _early_stop_scope(settings)
    stop = False
    completed_actor_epochs = 0
    completed_actor_updates = 0
    policy.train()
    for _ in range(int(settings.get("actor_epochs", 4))):
        for indices in stratified_minibatches(
            stratification_ids, batch_size, stratification_weights,
            maximum_batches=maximum_minibatches,
        ):
            histories = rollout["history"][indices].to(device)
            speed_commands = rollout["speed_command"][indices].to(device)
            action_chunks = rollout["action_chunk"][indices].to(device)
            epsilon = rollout["epsilon"][indices].to(device)
            flow_time = rollout["flow_time"][indices].to(device)
            shortcut_step = rollout["shortcut_step"][indices].to(device)
            valid = rollout["valid"][indices].to(device)
            old_cfm = rollout["old_cfm"][indices].to(device)
            advantage = advantages[indices].to(device)
            constraint_penalty = torch.zeros_like(advantage)
            if constraint_advantages is not None:
                assert family_ids is not None and constraint_lambdas is not None
                multipliers = constraint_lambdas[family_ids[indices]].to(device)
                constraint_penalty = (
                    float(settings.get("ppo_constraint_advantage_scale", 1.0))
                    * multipliers
                    * constraint_advantages[indices].to(device)
                )
                advantage = advantage - constraint_penalty
            actor_optimizer.zero_grad(set_to_none=True)
            current_cfm, current_velocity = privileged_flow_cfm_loss(
                policy, histories, speed_commands, action_chunks,
                epsilon, flow_time, shortcut_step, valid,
                huber_delta=settings.get("fpo_cfm_huber_delta"),
            )
            fpo = fpo_plus_plus_loss(
                old_cfm, current_cfm, advantage,
                clip_epsilon=float(settings.get("clip_epsilon", 0.05)),
                trust_region=str(settings.get("fpo_trust_region", "aspo")),
                cfm_loss_clamp=float(settings.get("fpo_cfm_loss_clamp", 20.0)),
                negative_cfm_clamp=float(settings.get(
                    "fpo_negative_cfm_clamp", 20.0
                )),
                log_ratio_clamp=float(settings.get("fpo_log_ratio_clamp", 10.0)),
                log_ratio_gain=float(settings.get("fpo_log_ratio_gain", 1.0)),
            )
            if (
                early_stop_scope == "minibatch" and target_kl > 0.0
                and float(fpo.approximate_kl) > target_kl
            ):
                metrics.setdefault("approximate_kl", []).append(
                    float(fpo.approximate_kl)
                )
                metrics.setdefault("clip_fraction", []).append(
                    float(fpo.clip_fraction)
                )
                stop = True
                break
            with torch.no_grad():
                _, reference_velocity = privileged_flow_cfm_loss(
                    reference, histories, speed_commands, action_chunks,
                    epsilon, flow_time, shortcut_step, valid,
                    huber_delta=settings.get("fpo_cfm_huber_delta"),
                )
            mask = valid[:, None, :, None].to(current_velocity.dtype)
            anchor_denominator = (
                mask.sum()
                * current_velocity.shape[1]
                * current_velocity.shape[-1]
            ).clamp_min(1.0)
            anchor_loss = (
                (current_velocity.float() - reference_velocity.float()).square()
                * mask
            ).sum() / anchor_denominator
            dynamics_loss = torch.zeros((), device=device)
            if dynamics_weight > 0.0:
                dynamics_per_sample = F.smooth_l1_loss(
                    policy.predict_dynamics(histories),
                    rollout["dynamics_target"][indices].to(device),
                    beta=float(settings.get("ppo_dynamics_beta", 0.10)),
                    reduction="none",
                ).mean(-1)
                dynamics_mask = rollout.get("dynamics_valid")
                dynamics_mask = (
                    torch.ones_like(dynamics_per_sample)
                    if dynamics_mask is None
                    else dynamics_mask[indices].to(
                        device=device, dtype=dynamics_per_sample.dtype,
                    )
                )
                dynamics_loss = (
                    dynamics_per_sample * dynamics_mask
                ).sum() / dynamics_mask.sum().clamp_min(1.0)
            loss = (
                fpo.loss + anchor_weight * anchor_loss
                + dynamics_weight * dynamics_loss
            )
            loss.backward()
            backbone_gradient = torch.nn.utils.clip_grad_norm_(
                actor_optimizer.param_groups[0]["params"],
                float(settings.get("actor_backbone_gradient_clip", 1.0)),
            )
            head_gradient = torch.nn.utils.clip_grad_norm_(
                actor_optimizer.param_groups[1]["params"],
                float(settings.get("actor_head_gradient_clip", 1.0)),
            )
            actor_optimizer.step()
            completed_actor_updates += 1
            values = {
                "policy_loss": fpo.loss,
                "anchor_loss": anchor_loss,
                "constraint_advantage_penalty": constraint_penalty.mean(),
                "dynamics_loss": dynamics_loss,
                "ratio": fpo.ratio_mean,
                "ratio_std": fpo.ratio_std,
                "approximate_kl": fpo.approximate_kl,
                "clip_fraction": fpo.clip_fraction,
                "old_cfm_loss": fpo.old_cfm_loss,
                "current_cfm_loss": fpo.current_cfm_loss,
                "actor_backbone_gradient_norm": torch.as_tensor(backbone_gradient),
                "actor_head_gradient_norm": torch.as_tensor(head_gradient),
            }
            for name, value in values.items():
                metrics.setdefault(name, []).append(float(value.detach()))
        if stop:
            break
        completed_actor_epochs += 1
        if early_stop_scope == "epoch" and target_kl > 0.0:
            epoch_values: list[float] = []
            policy.eval()
            with torch.no_grad():
                for indices in stratified_minibatches(
                    stratification_ids, batch_size, stratification_weights,
                    maximum_batches=maximum_minibatches,
                ):
                    current, _ = privileged_flow_cfm_loss(
                        policy,
                        rollout["history"][indices].to(device),
                        rollout["speed_command"][indices].to(device),
                        rollout["action_chunk"][indices].to(device),
                        rollout["epsilon"][indices].to(device),
                        rollout["flow_time"][indices].to(device),
                        rollout["shortcut_step"][indices].to(device),
                        rollout["valid"][indices].to(device),
                        huber_delta=settings.get("fpo_cfm_huber_delta"),
                    )
                    probe = fpo_plus_plus_loss(
                        rollout["old_cfm"][indices].to(device), current,
                        advantages[indices].to(device),
                        clip_epsilon=float(settings.get("clip_epsilon", 0.05)),
                        trust_region=str(settings.get("fpo_trust_region", "aspo")),
                        cfm_loss_clamp=float(settings.get(
                            "fpo_cfm_loss_clamp", 20.0
                        )),
                        negative_cfm_clamp=float(settings.get(
                            "fpo_negative_cfm_clamp", 20.0
                        )),
                        log_ratio_clamp=float(settings.get(
                            "fpo_log_ratio_clamp", 10.0
                        )),
                        log_ratio_gain=float(settings.get("fpo_log_ratio_gain", 1.0)),
                    )
                    epoch_values.append(float(probe.approximate_kl))
            policy.train()
            epoch_kl = float(np.mean(epoch_values))
            metrics.setdefault("epoch_approximate_kl", []).append(epoch_kl)
            if epoch_kl > target_kl:
                stop = True
                break

    critic.train()
    critic_batch_size = int(settings.get("critic_minibatch_size", batch_size))
    if critic_batch_size < 1:
        raise ValueError("critic_minibatch_size must be positive")
    critic_return_scale = (
        rollout["return"].std(unbiased=False).clamp_min(1.0e-3)
        if bool(settings.get("critic_normalize_loss_by_return_std", False))
        else torch.ones((), dtype=rollout["return"].dtype)
    )
    for _ in range(int(settings.get("critic_epochs", 4))):
        for indices in stratified_minibatches(
            stratification_ids, critic_batch_size, stratification_weights,
            maximum_batches=maximum_minibatches,
        ):
            critic_optimizer.zero_grad(set_to_none=True)
            predicted = critic(rollout["critic_input"][indices].to(device))
            target = rollout["return"][indices].to(device)
            normalized_error = (predicted - target) / critic_return_scale.to(device)
            critic_loss = F.smooth_l1_loss(
                normalized_error, torch.zeros_like(normalized_error)
            )
            critic_loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), float(settings.get("critic_gradient_clip", 5.0))
            )
            critic_optimizer.step()
            metrics.setdefault("critic_loss", []).append(float(critic_loss.detach()))
            metrics.setdefault("critic_gradient_norm", []).append(float(gradient))
    critic.eval()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for indices in torch.arange(count).split(batch_size):
            predictions.append(
                critic(rollout["critic_input"][indices].to(device)).cpu()
            )
    predicted_values = torch.cat(predictions)
    return_variance = rollout["return"].var(unbiased=False)
    explained_variance = (
        1.0 - (rollout["return"] - predicted_values).var(unbiased=False)
        / return_variance.clamp_min(1.0e-8)
    )
    metrics["critic_explained_variance"] = [float(explained_variance)]
    critic_error = predicted_values - rollout["return"]
    metrics["critic_postfit_rmse"] = [float(critic_error.square().mean().sqrt())]
    metrics["critic_postfit_bias"] = [float(critic_error.mean())]
    metrics["critic_prediction_std"] = [float(predicted_values.std(unbiased=False))]
    metrics["critic_return_scale"] = [float(critic_return_scale)]
    metrics["actor_early_stop"] = [float(stop)]
    metrics["actor_epochs_completed"] = [float(completed_actor_epochs)]
    metrics["actor_updates_completed"] = [float(completed_actor_updates)]
    metrics["rollout_transitions"] = [float(count)]
    metrics["advantage_mean"] = [float(advantages.mean())]
    metrics["return_mean"] = [float(rollout["return"].mean())]
    output = {name: float(np.mean(values)) for name, values in metrics.items()}
    output["approximate_kl_max"] = float(max(metrics.get("approximate_kl", [0.0])))
    return output


def update_policy_preserving_dynamics(
    rollout: Mapping[str, torch.Tensor],
    policy: PrivilegedMLPPolicy,
    settings: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, float]:
    """Run a PPG-style dynamics phase while constraining behavior drift."""

    config = settings.get("ppo_phasic_dynamics", {})
    if not bool(config.get("enabled", False)):
        return {}
    if "dynamics_target" not in rollout:
        raise ValueError("phasic dynamics requires normalized dynamics targets")
    behavior = copy.deepcopy(policy).eval().requires_grad_(False)
    parameters = (
        list(policy.step_embedding.parameters())
        + list(policy.recurrent.parameters())
        + list(policy.temporal_norm.parameters())
    )
    if policy.temporal_position is not None:
        parameters.append(policy.temporal_position)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.get("learning_rate", 2.0e-6)),
        weight_decay=float(config.get("weight_decay", 1.0e-5)),
        fused=device.startswith("cuda"),
    )
    count = len(rollout["history"])
    batch_size = min(int(config.get("batch_size", 2048)), count)
    dynamics_losses: list[float] = []
    behavior_kls: list[float] = []
    gradients: list[float] = []
    policy.train()
    for _ in range(int(config.get("epochs", 1))):
        for indices in torch.randperm(count).split(batch_size):
            histories = rollout["history"][indices].to(device)
            targets = rollout["dynamics_target"][indices].to(device)
            speed_commands = rollout.get("speed_command")
            speed_commands = (
                speed_commands[indices].to(device) if speed_commands is not None else None
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = policy.predict_dynamics(histories)
            dynamics_per_sample = F.smooth_l1_loss(
                prediction, targets,
                beta=float(config.get("beta", 0.10)),
                reduction="none",
            ).mean(-1)
            dynamics_mask = rollout.get("dynamics_valid")
            dynamics_mask = (
                torch.ones_like(dynamics_per_sample)
                if dynamics_mask is None
                else dynamics_mask[indices].to(
                    device=device, dtype=dynamics_per_sample.dtype,
                )
            )
            dynamics_loss = (
                dynamics_per_sample * dynamics_mask
            ).sum() / dynamics_mask.sum().clamp_min(1.0)
            current = policy.distribution(histories, speed_commands)
            with torch.no_grad():
                prior = behavior.distribution(histories, speed_commands)
            behavior_kl = gaussian_kl(current, prior).mean()
            loss = dynamics_loss + float(config.get("behavior_kl_weight", 1.0)) * behavior_kl
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                parameters, float(config.get("gradient_clip", 0.25))
            )
            optimizer.step()
            dynamics_losses.append(float(dynamics_loss.detach()))
            behavior_kls.append(float(behavior_kl.detach()))
            gradients.append(float(gradient))
    return {
        "phasic_dynamics_loss": float(np.mean(dynamics_losses)),
        "phasic_behavior_kl": float(np.mean(behavior_kls)),
        "phasic_gradient_norm": float(np.mean(gradients)),
        "phasic_updates": float(len(dynamics_losses)),
    }


def build_actor_optimizer(
    policy: PrivilegedMLPPolicy,
    settings: Mapping[str, Any],
    device: str,
) -> torch.optim.Optimizer:
    """Use a smaller representation LR so PPO refines rather than erases BC."""

    learning_rate = float(settings.get("actor_learning_rate", 3.0e-4))
    backbone_scale = float(settings.get("actor_backbone_lr_scale", 1.0))
    backbone = (
        list(policy.step_embedding.parameters())
        + list(policy.recurrent.parameters())
        + list(policy.temporal_norm.parameters())
    )
    if policy.temporal_position is not None:
        backbone.append(policy.temporal_position)
    head = list(policy.mean_network.parameters())
    if policy.action_chunk_queries is not None:
        head.append(policy.action_chunk_queries)
    if policy.action_chunk_decoder is not None:
        head += list(policy.action_chunk_decoder.parameters())
    if policy.action_chunk_output is not None:
        head += list(policy.action_chunk_output.parameters())
    if policy.speed_conditioner is not None:
        head += list(policy.speed_conditioner.parameters())
    groups: list[dict[str, Any]] = [
        {"params": backbone, "lr": learning_rate * backbone_scale, "lr_scale": backbone_scale},
        {"params": head, "lr": learning_rate, "lr_scale": 1.0},
    ]
    if bool(settings.get("train_log_std", True)):
        from starscream.ppo_exploration import noise_bounds, noise_optimizer_group, prepare_noise_bounds
        low, high = noise_bounds(settings, policy)
        configured = policy.log_std_parameter.detach().cpu().numpy()
        if np.any(configured < low-1e-7) or np.any(configured > high+1e-7):
            raise ValueError('initial exploration scale is outside configured noise bounds')
        if settings.get('ppo_noise_bounds'):
            prepare_noise_bounds(settings, policy)
        groups.append(noise_optimizer_group(settings, policy.log_std_parameter, learning_rate))
    else:
        policy.log_std_parameter.requires_grad_(False)
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        weight_decay=float(settings.get("actor_weight_decay", 1.0e-5)),
        fused=device.startswith("cuda"),
    )


def build_privileged_flow_optimizer(
    policy: PrivilegedMLPPolicy,
    settings: Mapping[str, Any],
    device: str,
) -> torch.optim.Optimizer:
    """Separate the causal representation and flow decoder learning rates."""

    if policy.action_head_type != "shortcut_flow" or policy.flow_action_head is None:
        raise ValueError("privileged FPO optimizer requires a shortcut-flow policy")
    learning_rate = float(settings.get("actor_learning_rate", 5.0e-6))
    backbone_scale = float(settings.get("actor_backbone_lr_scale", 0.05))
    backbone = (
        list(policy.step_embedding.parameters())
        + list(policy.recurrent.parameters())
        + list(policy.temporal_norm.parameters())
    )
    if policy.temporal_position is not None:
        backbone.append(policy.temporal_position)
    flow = list(policy.flow_action_head.parameters())
    if policy.speed_conditioner is not None:
        flow += list(policy.speed_conditioner.parameters())
    groups = [
        {
            "params": backbone,
            "lr": learning_rate * backbone_scale,
            "lr_scale": backbone_scale,
        },
        {"params": flow, "lr": learning_rate, "lr_scale": 1.0},
    ]
    selected = {id(parameter) for group in groups for parameter in group["params"]}
    for parameter in policy.parameters():
        parameter.requires_grad_(id(parameter) in selected)
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=tuple(settings.get("adam_betas", [0.9, 0.95])),
        weight_decay=float(settings.get("actor_weight_decay", 1.0e-5)),
        fused=device.startswith("cuda"),
    )


def set_actor_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = base_learning_rate * float(group.get("lr_scale", 1.0))


def lap_timing_metrics(metrics: Mapping[str, float], control_hz: float) -> dict[str, float]:
    """Expose successful lap distributions in seconds, including each track.

    Aggregate lap times mix course lengths and successful-course composition;
    per-track timing and successful episode counts must accompany comparisons.
    """
    names = {
        "successful_mean_steps": "successful_lap_time_seconds",
        "successful_minimum_steps": "fastest_lap_time_seconds",
        "successful_median_steps": "median_lap_time_seconds",
        "successful_p10_steps": "p10_lap_time_seconds",
        "successful_p90_steps": "p90_lap_time_seconds",
    }
    return {
        key.rsplit("/", 1)[0] + "/" + names[key.rsplit("/", 1)[-1]]
        if "/" in key else names[key]: float(value) / control_hz
        for key, value in metrics.items()
        if key.rsplit("/", 1)[-1] in names
    }


def refinement_metrics(
    metrics: dict[str, float],
    stage: RacingCurriculumStage,
    settings: Mapping[str, Any],
) -> dict[str, float]:
    """Rank accuracy lexicographically before successful-lap pace."""

    metrics.update(track_family_metrics(metrics, settings))
    metrics.update(lap_timing_metrics(metrics, configured_control_hz(settings)))

    successful_steps = float(metrics.get("successful_mean_steps", float("nan")))
    if np.isfinite(successful_steps):
        lap_time = successful_steps / configured_control_hz(settings)
        pace = max(0.0, 1.0 - successful_steps / max(stage.max_steps, 1))
    else:
        lap_time = float("nan")
        pace = 0.0
    full = float(metrics.get("full_course_success", 0.0))
    metrics["successful_lap_time_seconds"] = lap_time
    metrics["pace_score"] = full * pace
    pace_seconds = (
        full * max(
            0.0,
            float(settings.get("selection_lap_time_reference_seconds", 0.0))
            - lap_time,
        )
        if np.isfinite(lap_time)
        else 0.0
    )
    metrics["pace_seconds_score"] = pace_seconds
    raw_pace_seconds = (
        max(
            0.0,
            float(settings.get("selection_lap_time_reference_seconds", 0.0))
            - lap_time,
        )
        if np.isfinite(lap_time)
        else 0.0
    )
    pace_qualified = full >= float(settings.get(
        "selection_minimum_full_course_success", 0.0
    ))
    metrics["pace_qualified"] = float(pace_qualified)
    metrics["qualified_pace_seconds_score"] = (
        raw_pace_seconds if pace_qualified else 0.0
    )
    aggregate_score = (
        float(settings.get("selection_full_weight", 100.0)) * full
        + 10.0 * float(metrics.get("p3", 0.0))
        + 3.0 * float(metrics.get("p2", 0.0))
        + 0.5 * float(metrics.get("p1", 0.0))
        - 10.0 * float(metrics.get("crash_rate", 0.0))
        + float(settings.get("selection_pace_weight", 1.0)) * metrics["pace_score"]
        + float(settings.get("selection_pace_seconds_weight", 0.0)) * pace_seconds
        + float(settings.get("selection_qualified_pace_seconds_weight", 0.0))
        * metrics["qualified_pace_seconds_score"]
    )
    metrics["selection_score"] = aggregate_score

    # Multi-track pace must be normalized by each course's baseline duration;
    # absolute lap seconds would systematically favor shorter tracks. A speed
    # checkpoint is qualified only while every track clears its own reliability
    # floor, preventing aggregate success from hiding a damaged hard course.
    # Config inheritance deep-merges mappings.  Experiments that change track
    # distributions therefore need an explicit replacement channel, otherwise
    # stale parent-course references silently poison multitrack ranking.
    reference_source = settings.get(
        "selection_track_lap_time_reference_seconds_exclusive",
        settings.get("selection_track_lap_time_reference_seconds", {}),
    )
    references = dict(reference_source)
    if references:
        thresholds = dict(settings.get("selection_track_minimum_success", {}))
        track_full: list[float] = []
        track_improvement: list[float] = []
        track_qualified: list[bool] = []
        for track, raw_reference in references.items():
            prefix = f"track/{track}/"
            track_success = float(metrics.get(prefix + "full_course_success", 0.0))
            track_steps = float(metrics.get(prefix + "successful_mean_steps", float("nan")))
            reference = float(raw_reference)
            track_lap = (
                track_steps / configured_control_hz(settings)
                if np.isfinite(track_steps) else float("nan")
            )
            improvement = (
                (reference - track_lap) / max(reference, 1.0e-6)
                if np.isfinite(track_lap) else -1.0
            )
            floor = float(thresholds.get(track, settings.get(
                "selection_default_track_minimum_success", 0.80
            )))
            metrics[prefix + "successful_lap_time_seconds"] = track_lap
            metrics[prefix + "pace_improvement_fraction"] = improvement
            metrics[prefix + "pace_qualified"] = float(track_success >= floor)
            track_full.append(track_success)
            track_improvement.append(improvement)
            track_qualified.append(track_success >= floor)
        minimum_track_full = min(track_full)
        mean_track_improvement = float(np.mean(track_improvement))
        multitrack_qualified = bool(
            full >= float(settings.get("selection_minimum_full_course_success", 0.0))
            and all(track_qualified)
        )
        metrics["minimum_track_full_course_success"] = minimum_track_full
        metrics["mean_track_pace_improvement_fraction"] = mean_track_improvement
        metrics["multitrack_pace_qualified"] = float(multitrack_qualified)
        metrics["selection_score"] = (
            aggregate_score
            + float(settings.get("selection_minimum_track_full_weight", 0.0))
            * minimum_track_full
            + (
                float(settings.get("selection_mean_track_pace_weight", 0.0))
                * mean_track_improvement
                if multitrack_qualified else 0.0
            )
        )
    if settings.get('ppo_lap_time', {}).get('enabled', False):
        from starscream.ppo_lap_time import selection_metrics
        metrics.update(selection_metrics(metrics, settings['ppo_lap_time']))
    return metrics


def dagger_completion_survival_metrics(
    metrics: dict[str, float], settings: Mapping[str, Any],
) -> dict[str, float]:
    """Rank DAgger checkpoints by deep-course survival and family coverage.

    Exact real courses remain reporting-only. This score operates solely on
    the frozen held-out augmentation split and prevents early-gate averages
    from outranking policies that survive the Swift and A2RL topology cliffs.
    """

    metrics.update(track_family_metrics(metrics, settings))
    gates = tuple(int(item) for item in settings.get(
        "dagger_selection_survival_gates", (3, 5, 7, 10, 12)
    ))
    weights = tuple(float(item) for item in settings.get(
        "dagger_selection_survival_weights", (0.25, 0.75, 1.0, 1.0, 1.0)
    ))
    if len(gates) != len(weights) or not gates or any(item < 1 for item in gates):
        raise ValueError("invalid DAgger survival gate/weight configuration")
    if any(not np.isfinite(item) or item < 0.0 for item in weights):
        raise ValueError("DAgger survival weights must be finite and non-negative")

    survival_score = float(sum(
        weight * float(metrics.get(f"p{gate}", 0.0))
        for gate, weight in zip(gates, weights)
    ))
    family_successes = [
        float(value) for key, value in metrics.items()
        if key.startswith("family/") and key.endswith("/full_course_success")
    ]
    minimum_family_success = min(family_successes) if family_successes else 0.0
    family_survival_gates = tuple(int(item) for item in settings.get(
        "dagger_selection_family_survival_gates", gates
    ))
    family_survival_weights = tuple(float(item) for item in settings.get(
        "dagger_selection_family_survival_weights",
        [1.0] * len(family_survival_gates),
    ))
    if len(family_survival_gates) != len(family_survival_weights):
        raise ValueError("family survival gates and weights must align")
    family_names = sorted({
        key.split("/", 2)[1]
        for key in metrics
        if key.startswith("family/") and key.endswith("/episodes")
    })
    family_survival_scores = [
        sum(
            weight * float(metrics.get(f"family/{family}/p{gate}", 0.0))
            for gate, weight in zip(
                family_survival_gates, family_survival_weights
            )
        )
        for family in family_names
    ]
    minimum_family_survival = (
        min(family_survival_scores) if family_survival_scores else 0.0
    )
    full = float(metrics.get("full_course_success", 0.0))
    crash = float(metrics.get("crash_rate", 0.0))
    score = (
        float(settings.get("dagger_selection_full_course_weight", 12.0)) * full
        + survival_score
        + float(settings.get("dagger_selection_minimum_family_weight", 3.0))
        * minimum_family_success
        + float(settings.get(
            "dagger_selection_minimum_family_survival_weight", 0.0
        )) * minimum_family_survival
        - float(settings.get("dagger_selection_crash_weight", 0.25)) * crash
    )
    metrics["completion_survival_score"] = survival_score
    metrics["minimum_family_full_course_success"] = minimum_family_success
    metrics["minimum_family_survival_score"] = minimum_family_survival
    metrics["selection_score"] = score
    return metrics


def track_family_metrics(
    metrics: Mapping[str, float], settings: Mapping[str, Any],
) -> dict[str, float]:
    """Pool concrete-track outcomes into statistically useful families.

    Held-out suites commonly have only a handful of episodes per concrete
    track.  A hard minimum over those Bernoulli estimates is dominated by
    sampling noise.  Family pooling retains topology-specific protection while
    giving each constraint enough trials to support a rollback decision.
    """

    manifest_path = settings.get("track_manifest")
    if manifest_path is None:
        return {}
    manifest = read_manifest(str(manifest_path))
    family_by_name = {
        str(record["name"]): str(record.get("family", "unclassified"))
        for record in manifest["records"]
    }
    family_aliases = {
        str(name): str(alias)
        for name, alias in settings.get("reliability_family_aliases", {}).items()
    }
    totals: dict[str, list[float]] = {}
    survival_totals: dict[str, dict[str, float]] = {}
    suffix = "/full_course_success"
    for key, raw_success in metrics.items():
        if not key.startswith("track/") or not key.endswith(suffix):
            continue
        track = key[len("track/"):-len(suffix)]
        family = family_by_name.get(track)
        if family is None:
            continue
        family = family_aliases.get(family, family)
        episodes = float(metrics.get(f"track/{track}/episodes", 0.0))
        if episodes <= 0.0 or not np.isfinite(episodes):
            continue
        successes = float(raw_success) * episodes
        bucket = totals.setdefault(family, [0.0, 0.0])
        bucket[0] += successes
        bucket[1] += episodes
        survival = survival_totals.setdefault(family, {})
        for gate in range(1, 13):
            key_name = f"p{gate}"
            value = metrics.get(f"track/{track}/{key_name}")
            if value is not None:
                survival[key_name] = survival.get(key_name, 0.0) + (
                    float(value) * episodes
                )
    output: dict[str, float] = {}
    for family, (successes, episodes) in sorted(totals.items()):
        output[f"family/{family}/episodes"] = episodes
        output[f"family/{family}/full_course_success"] = successes / episodes
        for key_name, total in survival_totals.get(family, {}).items():
            output[f"family/{family}/{key_name}"] = total / episodes
    if totals:
        output["reliability_family_count"] = float(len(totals))
    return output


def reliability_floors(
    baseline: Mapping[str, float],
    settings: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    """Freeze aggregate and topology safety constraints at a stage baseline."""

    aggregate = max(
        float(settings.get("rollback_full_course_threshold", 0.0)),
        float(baseline.get("full_course_success", 0.0))
        * float(settings.get("rollback_full_course_fraction_of_baseline", 0.0)),
    )
    scope = str(settings.get("rollback_reliability_scope", "track"))
    if scope not in {"track", "family"}:
        raise ValueError("rollback_reliability_scope must be track or family")
    relative = float(settings.get(
        f"rollback_{scope}_fraction_of_baseline",
        settings.get("rollback_track_fraction_of_baseline", 0.0),
    ))
    absolute = dict(settings.get(
        f"rollback_{scope}_minimum_success",
        settings.get("rollback_track_minimum_success", {}),
    ))
    metric_prefix = f"{scope}/"
    observed_tracks = sorted({
        key.split("/")[1]
        for key in baseline
        if key.startswith(metric_prefix) and key.endswith("/full_course_success")
    })
    if scope == "family" and not observed_tracks:
        raise ValueError("family reliability requested but no family metrics exist")
    tracks = tuple(observed_tracks) or configured_tracks(settings)
    per_track = {
        (f"family:{track}" if scope == "family" else track): max(
            float(absolute.get(track, 0.0)),
            float(baseline.get(
                f"{scope}/{track}/full_course_success", 0.0,
            )) * relative,
        )
        for track in tracks
    }
    return aggregate, per_track


def reliability_constraint_metrics(
    metrics: Mapping[str, float],
    aggregate_floor: float,
    track_floors: Mapping[str, float],
    *,
    confidence_z: float = 0.0,
) -> dict[str, float]:
    """Report whether a candidate remains in the reliability set.

    ``confidence_z`` grants a one-sided binomial standard-error tolerance.
    It is intentionally zero by default for backwards compatibility.  Family
    constraints can opt into a small tolerance without allowing a material
    aggregate collapse.
    """

    if confidence_z < 0.0 or not np.isfinite(confidence_z):
        raise ValueError("confidence_z must be finite and non-negative")

    def adjusted_margin(rate: float, floor: float, episodes: float) -> tuple[float, float]:
        raw = rate - floor
        if confidence_z == 0.0 or episodes <= 0.0:
            return raw, raw
        variance = max(floor * (1.0 - floor), 1.0e-6) / episodes
        return raw, raw + confidence_z * float(np.sqrt(variance))

    # Aggregate reliability is the hard global constraint. Sampling tolerance
    # applies only to topology-family estimates; it must never excuse collapse
    # of the suite as a whole.
    aggregate_raw = (
        float(metrics.get("full_course_success", 0.0)) - aggregate_floor
    )
    aggregate_margin = aggregate_raw
    raw_margins = [aggregate_raw]
    margins = [aggregate_margin]
    for track, floor in track_floors.items():
        if track.startswith("family:"):
            prefix = f"family/{track.split(':', 1)[1]}/"
        else:
            prefix = f"track/{track}/"
        raw, adjusted = adjusted_margin(
            float(metrics.get(prefix + "full_course_success", 0.0)),
            floor,
            float(metrics.get(prefix + "episodes", 0.0)),
        )
        raw_margins.append(raw)
        margins.append(adjusted)
    violations = sum(margin < -1.0e-12 for margin in margins)
    return {
        "reliability_constraints_met": float(violations == 0),
        "reliability_minimum_margin": float(min(margins)),
        "reliability_minimum_raw_margin": float(min(raw_margins)),
        "reliability_violation_count": float(violations),
        "reliability_aggregate_floor": float(aggregate_floor),
        "reliability_confidence_z": float(confidence_z),
    }


def ppo_rank_curriculum_stage(
    settings: Mapping[str, Any], stage_index: int, stage_count: int,
) -> bool:
    """Return whether a PPO curriculum stage may enter the global top-k.

    Short-horizon curriculum success is not numerically comparable with
    full-course success. Experiments can therefore retain every candidate in
    ``latest.pt`` while limiting ranked checkpoints to explicitly named stage
    indices (normally the full-course reliability and speed stages).
    """

    configured = settings.get("checkpoint_rank_curriculum_stages")
    if configured is None:
        return True
    if not isinstance(configured, (list, tuple)) or not configured:
        raise ValueError(
            "checkpoint_rank_curriculum_stages must be a non-empty sequence"
        )
    indices = {int(index) for index in configured}
    if min(indices) < 0 or max(indices) >= int(stage_count):
        raise ValueError(
            "checkpoint_rank_curriculum_stages contains an invalid stage index"
        )
    return int(stage_index) in indices


def online_course_bank_anchors(
    stages: Sequence[RacingCurriculumStage],
) -> tuple[str, ...]:
    """Return the stable anchor roster required by a multi-stage online bank.

    Bank competence, retirement, and replay multipliers are positional.  A
    curriculum may change rewards, speed commands, and spawn randomization,
    but changing or reordering its anchor tracks would silently attach that
    state to the wrong courses after a stage transition.
    """

    if not stages:
        raise ValueError("ppo_online_course_bank requires a curriculum stage")
    anchors = tuple(stages[0].tracks)
    for index, stage in enumerate(stages[1:], start=1):
        if tuple(stage.tracks) != anchors:
            raise ValueError(
                "ppo_online_course_bank requires identical ordered anchor "
                f"tracks across curriculum stages; stage {index} differs"
            )
    return anchors


def online_bank_resume_progress(settings, initial, stages):
    """Validate an explicit bank continuation, including legacy phase-zero saves."""
    if not settings.get("resume_online_course_bank", False):
        return None
    if (initial.get("stage") != "ppo"
            or not settings.get("ppo_online_course_bank", {}).get("enabled")
            or not initial.get("online_course_bank_state")
            or not settings.get("restore_actor_optimizer")
            or not settings.get("restore_critic_optimizer")):
        raise ValueError("online bank resume requires a PPO bank and both optimizers")
    for key in ("actor_optimizer_initial_checkpoint", "critic_initial_checkpoint"):
        if settings.get(key) != settings.get("initial_checkpoint"):
            raise ValueError("online bank resume must use the same actor/critic checkpoint")
    old = initial["training_config"]["ppo"]
    if old["curriculum"] != settings["curriculum"]:
        raise ValueError("online bank resume requires the same curriculum")
    index = int(initial["curriculum_stage"])
    if not 0 <= index < len(stages):
        raise ValueError("invalid resumed curriculum stage")
    steps = int(initial["environment_steps"])
    phase_steps = initial.get("stage_environment_steps")
    if phase_steps is None:
        if index != 0:
            raise ValueError("legacy later-phase checkpoint lacks stage_environment_steps")
        phase_steps = steps
    lr = initial.get("actor_base_learning_rate")
    if lr is None:
        group = initial["optimizer"]["param_groups"][0]
        lr = float(group["lr"]) / float(group.get("lr_scale", 1.0))
    return int(initial["cycle"]), steps, index, int(phase_steps), float(lr)


def run_ppo(
    config: dict[str, Any], device: str, *, settings_key: str = "ppo"
) -> None:
    settings = config[settings_key]
    flow_fpo = settings_key == "fpo"
    if settings.get('ppo_lap_time', {}).get('enabled', False):
        if (flow_fpo or not settings.get('ppo_rollout_window_steps', 0)
                or settings.get('collector_backend') != 'process'
                or len(settings['curriculum']) != 1):
            raise ValueError('lap-time PPO requires one frozen stage and process rollout windows')
    if settings.get('ppo_rollout_window_steps', 0):
        from starscream.ppo_windows import validate_window_settings
        validate_window_settings(settings, int(settings.get('rollout_envs',12)))
        if flow_fpo or settings.get('collector_backend','thread') != 'process':
            raise ValueError('fixed windows require the direct-action process PPO collector')
        if int(settings.get('episodes_per_cycle',96)) != int(settings.get('rollout_envs',12)):
            raise ValueError('fixed windows require episodes_per_cycle equal to rollout_envs (lane slots)')
    if device.startswith("cuda") and bool(
        settings.get("ppo_strict_likelihood_math", False)
    ):
        # Correlated exploration has a narrower conditional innovation than
        # its marginal action noise.  TF32's batch-shape-dependent rounding can
        # then look like a nonzero behavior KL before the first update.  PPO's
        # ratio is a mathematical contract, so prefer reproducibility here.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    seed = int(settings.get("seed", 20260817))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    policy, normalizer, initial, initial_path = load_policy_checkpoint(
        settings["initial_checkpoint"], device
    )
    configured_observation_contract = str(settings.get(
        "observation_contract", LEGACY_OBSERVATION_CONTRACT
    ))
    if policy.observation_contract != configured_observation_contract:
        raise ValueError(
            "PPO observation contract does not match its initial checkpoint: "
            f"checkpoint={policy.observation_contract} "
            f"config={configured_observation_contract}"
        )
    if flow_fpo and policy.action_head_type != "shortcut_flow":
        raise ValueError("privileged FPO++ requires a shortcut-flow checkpoint")
    if not flow_fpo and policy.action_head_type == "shortcut_flow":
        raise ValueError("shortcut-flow checkpoints must use --stage fpo, not PPO")
    exploration_correlation = configured_ppo_exploration_correlation(settings)
    if exploration_correlation > 0.0:
        if str(settings.get("collector_backend", "process")) != "process":
            raise ValueError(
                "correlated PPO exploration requires the persistent process collector"
            )
        if bool(settings.get("train_log_std", True)):
            raise ValueError(
                "correlated PPO exploration requires fixed marginal log_std so "
                "the AR(1) latent process remains exogenous"
            )
    policy.set_exact_likelihood_mode(True)
    stored_action_contract = initial.get("action_contract")
    configured_action_contract = action_contract_metadata(settings)
    if (
        stored_action_contract is not None
        and dict(stored_action_contract) != configured_action_contract
    ):
        raise ValueError(
            "PPO action contract does not match its initial checkpoint: "
            f"checkpoint={stored_action_contract}, "
            f"configured={configured_action_contract}"
        )
    if bool(settings.get("actor_speed_conditioning", False)):
        policy.enable_speed_conditioning(
            scale=float(settings.get("speed_conditioning_scale", 20.0)),
            default_command=float(settings.get("default_speed_command", 13.0)),
        )
    if not flow_fpo and "exploration_log_std" in settings:
        with torch.no_grad():
            configured_log_std = validated_exploration_log_std(
                settings['exploration_log_std'], policy.log_std_parameter,
                policy.minimum_log_std, policy.maximum_log_std)
            policy.log_std_parameter.copy_(configured_log_std)
    if (
        float(settings.get("ppo_dynamics_weight", 0.0)) > 0.0
        or bool(settings.get("ppo_phasic_dynamics", {}).get("enabled", False))
    ):
        if policy.observation_contract == GREEN2026_OBSERVATION_CONTRACT:
            raise ValueError(
                "Green-2026 PPO uses the paper reward without Starscream's "
                "legacy gate-frame task-delta objective; keep PPO dynamics disabled"
            )
        # Keep the offline-trained readout fixed. Its on-policy prediction loss
        # then acts as a probe that prevents the temporal representation from
        # discarding task dynamics while PPO changes the command head.
        policy.dynamics_head.requires_grad_(False)
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    policy.flatten_backbone_parameters()
    reference.flatten_backbone_parameters()
    critic = build_ppo_critic(policy, settings, device)
    from starscream.training_acceleration import configure_policy_acceleration
    configure_policy_acceleration(policy, settings)
    actor_optimizer = (
        build_privileged_flow_optimizer(policy, settings, device)
        if flow_fpo else build_actor_optimizer(policy, settings, device)
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=float(settings.get("critic_learning_rate", 3.0e-4)),
        weight_decay=float(settings.get("critic_weight_decay", 1.0e-5)),
        fused=device.startswith("cuda"),
    )
    actor_optimizer_initial_checkpoint = settings.get(
        "actor_optimizer_initial_checkpoint"
    )
    if actor_optimizer_initial_checkpoint:
        actor_optimizer_path = resolve_ranked_checkpoint(
            actor_optimizer_initial_checkpoint
        )
        actor_optimizer_payload = torch.load(
            actor_optimizer_path, map_location="cpu", weights_only=False
        )
        if bool(settings.get("restore_actor_optimizer", True)):
            actor_optimizer.load_state_dict(actor_optimizer_payload["optimizer"])
            set_actor_learning_rate(
                actor_optimizer,
                float(settings.get("actor_learning_rate", 3.0e-4)),
            )
        print(
            f"ppo_actor_optimizer_continuation checkpoint={actor_optimizer_path} "
            f"optimizer={int(bool(settings.get('restore_actor_optimizer', True)))}",
            flush=True,
        )
    critic_initial_checkpoint = settings.get("critic_initial_checkpoint")
    if critic_initial_checkpoint:
        critic_path = resolve_ranked_checkpoint(critic_initial_checkpoint)
        critic_payload = torch.load(
            critic_path, map_location="cpu", weights_only=False
        )
        stored_architecture = str(critic_payload.get("critic_architecture", "mlp"))
        configured_architecture = str(settings.get("critic_architecture", "mlp"))
        if stored_architecture != configured_architecture:
            raise ValueError(
                "critic continuation architecture differs: "
                f"checkpoint={stored_architecture} configured={configured_architecture}"
            )
        stored_normalizer = critic_payload.get("normalizer", {})
        if not (
            np.allclose(stored_normalizer.get("mean"), normalizer.mean)
            and np.allclose(stored_normalizer.get("std"), normalizer.std)
        ):
            raise ValueError(
                "critic continuation checkpoint uses a different feature normalizer"
            )
        critic.load_state_dict(critic_payload["critic"])
        if bool(settings.get("restore_critic_optimizer", True)):
            critic_optimizer.load_state_dict(critic_payload["critic_optimizer"])
        print(
            f"ppo_critic_continuation checkpoint={critic_path} "
            f"optimizer={int(bool(settings.get('restore_critic_optimizer', True)))}",
            flush=True,
        )
    stages = [parse_stage(item) for item in settings["curriculum"]]
    raw_evaluation = settings.get("evaluation_curriculum")
    if raw_evaluation is None:
        evaluation_stages = stages
    elif isinstance(raw_evaluation, list):
        evaluation_stages = [parse_stage(item) for item in raw_evaluation]
    else:
        evaluation_stages = [parse_stage(raw_evaluation) for _ in stages]
    if len(evaluation_stages) != len(stages):
        raise ValueError("PPO evaluation curriculum must align with training stages")
    raw_reporting = settings.get("reporting_evaluation_curriculum")
    if raw_reporting is None:
        reporting_stages: list[RacingCurriculumStage] | None = None
    elif isinstance(raw_reporting, list):
        reporting_stages = [parse_stage(item) for item in raw_reporting]
    else:
        reporting_stages = [parse_stage(raw_reporting) for _ in stages]
    if reporting_stages is not None and len(reporting_stages) != len(stages):
        raise ValueError("PPO reporting curriculum must align with training stages")
    checkpoint_metric_source = str(
        settings.get("checkpoint_metric_source", "active")
    ).lower()
    if checkpoint_metric_source not in {"active", "reporting"}:
        raise ValueError("checkpoint_metric_source must be active or reporting")
    if checkpoint_metric_source == "reporting":
        if reporting_stages is None:
            raise ValueError(
                "reporting checkpoint ranking requires a reporting curriculum"
            )
    online_raw = settings.get("online_manifold_curriculum", {})
    online_manager: OnlineManifestCurriculum | None = None
    if isinstance(online_raw, Mapping) and bool(online_raw.get("enabled", False)):
        if len(stages) != 1 or len(evaluation_stages) != 1:
            raise ValueError(
                "online manifold PPO currently requires one mutable curriculum stage"
            )
        online_manager = OnlineManifestCurriculum(
            online_raw,
            restored_state=(
                initial.get("online_manifold_curriculum_state")
                if bool(online_raw.get("restore_state_from_checkpoint", True))
                else None
            ),
        )
        settings["track_manifest"] = str(online_manager.active_manifest_path)
        settings["ppo_track_sampling_weights"] = (
            online_manager.track_sampling_weights()
        )
        settings.pop("ppo_track_sampling_family_weights", None)
        stages[0] = replace(stages[0], tracks=online_manager.rollout_paths)
        evaluation_stages[0] = replace(
            evaluation_stages[0], tracks=online_manager.active_paths,
        )
    pool_manager = None
    task_blocks = settings.get('ppo_task_blocks')
    if task_blocks and (len(stages)!=1 or settings.get('ppo_pool_curriculum')
                        or online_manager is not None):
        raise ValueError('task blocks require one stage and no generation curriculum')
    block_tracks = tuple(stages[0].tracks)
    if task_blocks:
        from starscream.ppo_task_blocks import task_block
        stages[0]=replace(stages[0],tracks=task_block(block_tracks,1,**task_blocks))
    pool_raw = settings.get("ppo_pool_curriculum")
    if pool_raw:
        import json
        from starscream.behavior_pool_rl import PoolCurriculum
        if online_manager is not None or len(stages) != 1:
            raise ValueError("pool curriculum requires one stage and no goal-tube manager")
        pool_manager = PoolCurriculum(pool_raw, settings['curriculum'][0],
            initial.get('ppo_pool_curriculum_state'))
    # Validate the ranking contract before constructing collectors or running
    # an expensive baseline evaluation.
    ppo_rank_curriculum_stage(settings, 0, len(stages))
    probe_env = make_env(settings, track=stages[0].tracks[0])
    course_gate_count = len(probe_env.track.gates)
    probe_env.close()
    stage_index = 0
    stage_environment_steps = 0
    total_environment_steps = int(settings.get("initial_environment_steps", 0))
    bank_resume = online_bank_resume_progress(settings, initial, stages)
    restored_online_run = bool(online_manager is not None
        and online_raw.get("restore_state_from_checkpoint", True)
        and initial.get("online_manifold_curriculum_state"))
    restored_online_run = restored_online_run or bool(
        pool_manager is not None and initial.get('ppo_pool_curriculum_state'))
    restored_online_run = restored_online_run or bank_resume is not None
    if settings.get('resume_task_blocks',False):
        old_config=initial.get('training_config',{})
        old_settings=old_config.get('ppo',old_config)
        if (not task_blocks or initial.get('stage')!='ppo'
            or old_settings.get('ppo_task_blocks')!=task_blocks
            or tuple(old_settings['curriculum'][0]['tracks'])!=block_tracks
            or not settings.get('restore_actor_optimizer')
            or not settings.get('restore_critic_optimizer')):
            raise ValueError('task-block resume requires matching roster/schedule and optimizers')
        restored_online_run=True
    initial_cycle = int(initial.get("cycle", 0)) if restored_online_run else 0
    if restored_online_run:
        total_environment_steps = int(initial["environment_steps"])
    if total_environment_steps < 0:
        raise ValueError("initial_environment_steps must be non-negative")
    manager = CheckpointManager.from_config(config)
    logger = init_wandb(config)
    actor_base_learning_rate = float(settings.get("actor_learning_rate", 3.0e-4))
    if bank_resume is not None:
        (initial_cycle, total_environment_steps, stage_index,
         stage_environment_steps, actor_base_learning_rate) = bank_resume
        set_actor_learning_rate(actor_optimizer, actor_base_learning_rate)
    process_rollouts = str(settings.get("collector_backend", "process")) == "process"
    collector: ProcessRaceCollector | None = None
    collector_stage_index = -1
    # Preserve dual/EMA history by task identity across active-pool rebuilds.
    constraint_history = dict(initial.get("ppo_group_constraint_state", {})) if restored_online_run else {}

    def constraint_state_dict():
        if collector is not None:
            constraint_history.update({name: (
                float(collector._constraint_lambdas[i]),
                float(collector._constraint_success_ema[i]),
            ) for i, name in enumerate(collector.family_names)})
        return dict(constraint_history)

    bank = None

    def replace_collector(index: int) -> None:
        nonlocal collector, collector_stage_index
        switch_state = (collector.paper_task_switching_state_dict()
                        if pool_manager is not None and collector is not None else None)
        if collector is not None:
            constraint_history.update({name: (
                float(collector._constraint_lambdas[i]),
                float(collector._constraint_success_ema[i]),
            ) for i, name in enumerate(collector.family_names)})
            collector.close()
        collector_settings = dict(settings)
        if online_manager is not None:
            collector_settings["track_manifest"] = str(
                online_manager.active_manifest_path
            )
            collector_settings["ppo_track_sampling_weights"] = (
                online_manager.track_sampling_weights()
            )
        reward_override = settings.get("stage_reward_overrides", {}).get(
            stages[index].name
        )
        if reward_override is not None:
            collector_settings["reward"] = _merge(
                dict(settings.get("reward", {})), dict(reward_override)
            )
        collector = ProcessRaceCollector(
            policy, normalizer, collector_settings, stages[index], device,
            workers=int(settings.get("rollout_envs", 12)),
            sampling_prefix="ppo",
        ) if process_rollouts else None
        if collector is not None:
            if switch_state is not None:
                # Only append-only bank updates may preserve these active slots.
                if not set(switch_state['tracks']) <= set(stages[index].tracks):
                    raise ValueError('pool update removed rehearsed tasks')
                switch_state['tracks'] = list(stages[index].tracks)
                collector.load_paper_task_switching_state_dict(switch_state)
            for i, name in enumerate(collector.family_names):
                if name in constraint_history:
                    collector._constraint_lambdas[i], collector._constraint_success_ema[i] = constraint_history[name]
            if collector_stage_index >= 0 and bank is not None:
                collector.add_tracks([c["path"] for c in bank.courses if c["origin"] != "anchor"])
            if bank is not None:
                bank.begin_stage(f"{index}:{stages[index].name}",
                                 reset_competence=collector_stage_index >= 0)
        collector_stage_index = index

    def evaluate_current(index: int, seed_base: int) -> dict[str, float]:
        if settings.get('ppo_lap_time', {}).get('enabled', False):
            import json
            evaluator = ProcessRaceCollector(policy, normalizer, settings,
                evaluation_stages[index], device, workers=int(settings.get('evaluation_workers', 32)))
            try:
                rows = evaluator.evaluate_rows(episodes=int(settings['evaluation_episodes']), seed_base=seed_base)
            finally:
                evaluator.close()
            directory = manager.checkpoint_dir / 'evaluation_episodes'
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f'step-{total_environment_steps:09d}.json').write_text(json.dumps(rows, indent=2) + '\n')
            return multitrack_metrics(rows, evaluation_stages[index].target_gates)
        if bool(settings.get("evaluation_matched_track_seeds", False)):
            return evaluate_policy_matched_tracks(
                policy, normalizer, settings, evaluation_stages[index],
                count_per_track=int(settings.get(
                    "evaluation_episodes_per_track", 12,
                )),
                seed_base=seed_base, device=device,
            )
        weighted_training = bool(
            settings.get("ppo_track_sampling_family_weights")
            or settings.get("ppo_track_sampling_weights")
        )
        if (
            collector is not None
            and evaluation_stages[index] == stages[index]
            and not weighted_training
        ):
            return collector.evaluate(
                episodes=int(settings.get("evaluation_episodes", 96)),
                seed_base=seed_base,
            )
        return evaluate_policy(
            policy, normalizer, settings, evaluation_stages[index],
            count=int(settings.get("evaluation_episodes", 96)),
            seed_base=seed_base, device=device,
        )

    def evaluate_reporting(index: int, step: int) -> dict[str, float] | None:
        if reporting_stages is None:
            return None
        reporting_stage = reporting_stages[index]
        reporting_settings = dict(settings)
        if bool(settings.get("reporting_nominal_environment", False)):
            reporting_settings["dynamics_randomization"] = {"enabled": False}
            reporting_settings.pop("state_estimator_randomization", None)
            reporting_settings.pop("flight_plan_randomization", None)
            reporting_settings["policy_state_source"] = "truth"
            reporting_settings.pop("action_delay_range", None)
            reporting_settings["action_delay"] = float(
                settings.get("reporting_action_delay", 0.011)
            )
        reporting_seed = (
            int(settings.get("reporting_evaluation_seed", 20262000))
            + index * int(settings.get("reporting_evaluation_stage_seed_stride", 10000))
        )
        if bool(settings.get("reporting_evaluation_matched_track_seeds", False)):
            metrics = evaluate_policy_matched_tracks(
                policy, normalizer, reporting_settings, reporting_stage,
                count_per_track=int(settings.get(
                    "reporting_evaluation_episodes_per_track", 24,
                )),
                seed_base=reporting_seed, device=device,
            )
        else:
            metrics = evaluate_policy(
                policy, normalizer, reporting_settings, reporting_stage,
                count=int(settings.get("reporting_evaluation_episodes", 48)),
                seed_base=reporting_seed, device=device,
            )
        metrics.update(lap_timing_metrics(metrics, configured_control_hz(settings)))
        logger.log_eval({
            f"reporting/{reporting_stage.name}/{key}": value
            for key, value in metrics.items()
        }, step)
        print(
            f"ppo_reporting_eval stage={reporting_stage.name} step={step} "
            f"full={metrics['full_course_success']:.3f} "
            f"mean_gates={metrics['mean_gates']:.2f} "
            f"crash={metrics['crash_rate']:.3f} "
            f"lap={float(metrics['successful_mean_steps']) / configured_control_hz(settings):.3f}",
            flush=True,
        )
        return metrics

    bank = None
    bank_restored_tracks: list[str] = []
    bank_config = dict(settings.get("ppo_online_course_bank", {}))
    if bool(bank_config.get("enabled", False)):
        if not settings.get("ppo_rollout_window_steps", 0) or not process_rollouts:
            raise ValueError("ppo_online_course_bank requires window-mode process rollouts")
        bank_anchors = online_course_bank_anchors(stages)
        from starscream.ppo_online_bank import OnlineCourseBank, subprocess_screen_launcher
        protected: tuple[str, ...] = ()
        if bank_config.get("protected_track_suite"):
            protected, _ = load_active_real_course_suite(bank_config["protected_track_suite"])
        bank = OnlineCourseBank(
            {k: v for k, v in bank_config.items() if k != "protected_track_suite"},
            anchors=bank_anchors, bank_dir=manager.checkpoint_dir / "bank",
            protected_tracks=protected, seed=seed,
            screen_launcher=(
                subprocess_screen_launcher() if bank_config.get("raceability") == "nominal_mpcc" else None
            ),
        )
        initialize_bank = bool(settings.get("initialize_online_course_bank_from_checkpoint", False))
        if initialize_bank and restored_online_run:
            raise ValueError("bank initialization and exact run resume are mutually exclusive")
        if initialize_bank and not initial.get("online_course_bank_state"):
            raise ValueError("bank initialization requires a checkpoint with an online bank")
        if (restored_online_run or initialize_bank) and initial.get("online_course_bank_state"):
            bank_restored_tracks = bank.load_state_dict(initial["online_course_bank_state"])
            if initialize_bank:
                bank.begin_stage(f"{stage_index}:{stages[stage_index].name}", reset_competence=True)
    replace_collector(stage_index)
    if collector is not None and collector.lap_time_constraints is not None:
        collector.lap_time_constraints.load_state_dict(initial.get('ppo_lap_time_constraint_state'))
    if collector is not None and bank_restored_tracks:
        collector.add_tracks(bank_restored_tracks)
    initialize_sampling = bool(settings.get('initialize_adaptive_sampling_from_checkpoint', False))
    if collector is not None and (bank_resume is not None or initialize_sampling):
        sampling = initial.get("ppo_adaptive_sampling_state")
        if sampling is not None:
            if list(collector.stage.tracks) != sampling["tracks"]:
                raise ValueError("resumed adaptive sampling roster differs")
            collector._adaptive_competence = np.asarray(sampling["competence"], np.float64)
            collector._adaptive_stall_windows = np.asarray(sampling["stalls"], np.int64)
            collector.track_weights = np.asarray(sampling["weights"], np.float64)
        else:
            if initialize_sampling:
                raise ValueError('sampling initialization requires checkpoint adaptive sampling state')
            print("ppo_bank_resume legacy checkpoint: sampling EMAs and live episodes restart; admitted bank preserved", flush=True)
            bank.begin_stage(f"{stage_index}:{stages[stage_index].name}", reset_competence=True)

    def continuation_state():
        return {
            "ppo_lap_time_constraint_state": (None if collector is None or collector.lap_time_constraints is None
                                             else collector.lap_time_constraints.state_dict()),
            "stage_environment_steps": stage_environment_steps,
            "actor_base_learning_rate": actor_base_learning_rate,
            "ppo_adaptive_sampling_state": None if collector is None else {
                "tracks": list(collector.stage.tracks),
                "competence": collector._adaptive_competence.tolist(),
                "stalls": getattr(collector, "_adaptive_stall_windows",
                                  np.zeros(len(collector.stage.tracks), np.int64)).tolist(),
                "weights": collector.track_weights.tolist(),
            },
        }
    if collector is not None:
        collector.load_paper_task_switching_state_dict(
            initial.get("green2026_adaptive_task_switching_state")
        )
    baseline = refinement_metrics(evaluate_current(
        stage_index, int(settings.get("evaluation_seed", 20261000))
        + stage_index * int(settings.get("evaluation_stage_seed_stride", 10000))
    ), evaluation_stages[stage_index], settings)
    baseline["curriculum_stage"] = float(stage_index)
    baseline["cycle"] = float(initial_cycle)
    baseline["environment_steps"] = float(total_environment_steps)
    baseline["safety_rollback"] = 0.0
    safe_state = {
        name: value.detach().cpu().clone() for name, value in policy.state_dict().items()
    }
    safe_score = float(baseline["selection_score"])
    rollback_full_course_threshold, rollback_track_thresholds = reliability_floors(
        baseline, settings
    )
    baseline.update(reliability_constraint_metrics(
        baseline, rollback_full_course_threshold, rollback_track_thresholds,
        confidence_z=float(settings.get("rollback_confidence_z", 0.0)),
    ))
    logger.log_eval(baseline, total_environment_steps)
    baseline_reporting = evaluate_reporting(stage_index, total_environment_steps)
    manager.save(
        checkpoint_payload(
            policy, normalizer, stage=settings_key, track=str(settings["track"]),
            optimizer=actor_optimizer,
            extra={
                "critic": critic.state_dict(),
                "critic_architecture": str(settings.get("critic_architecture", "mlp")),
                "critic_optimizer": critic_optimizer.state_dict(),
                "cycle": initial_cycle,
                **continuation_state(),
                "environment_steps": total_environment_steps,
                "curriculum_stage": stage_index,
                "initial_checkpoint": str(initial_path),
                "dynamics_target_mean": initial["dynamics_target_mean"],
                "dynamics_target_std": initial["dynamics_target_std"],
                "training_config": config,
                "reporting_evaluation": baseline_reporting,
                "ppo_group_constraint_state": constraint_state_dict(),
                "online_course_bank_state": (None if bank is None else bank.state_dict()),
                "online_manifold_curriculum_state": (
                    None if online_manager is None else online_manager.state_dict()
                ),
            },
        ),
        step=total_environment_steps,
        metrics=(
            baseline_reporting
            if checkpoint_metric_source == "reporting"
            else baseline
        ),
        rank=(ppo_rank_curriculum_stage(settings, stage_index, len(stages))
              and bool(baseline.get('lap_completion_qualified', 1.))),
    )
    print(
        f"privileged_{'fpo' if flow_fpo else 'ppo'} actor={initial_path} "
        f"contract={'cfm_joint_chunk_ratio_execute_action0' if flow_fpo else 'exact_tanh_gaussian_full_actor'} "
        f"track={settings['track']} context={policy.context_steps} "
        f"actor_parameters={sum(p.numel() for p in policy.parameters()):,} "
        f"baseline_full={baseline['full_course_success']:.3f} "
        f"baseline_lap_s={baseline['successful_lap_time_seconds']:.3f} "
        f"rollback_full={rollback_full_course_threshold:.3f} "
        f"rollback_tracks={rollback_track_thresholds}", flush=True,
    )
    for cycle in range(initial_cycle + 1, int(settings.get("cycles", 200)) + 1):
        cycle_started = time.perf_counter()
        if task_blocks:
            from starscream.ppo_task_blocks import task_block
            tracks=task_block(block_tracks,cycle,**task_blocks)
            if tracks != stages[0].tracks:
                stages[0]=replace(stages[0],tracks=tracks)
                replace_collector(0)
            print(f"ppo_task_block cycle={cycle} tracks={','.join(Path(t).stem for t in tracks)}",flush=True)
        if pool_manager is not None:
            from starscream.behavior_pool_rl import reachable
            from starscream.course_model.training import atomic_json
            def probe_pool_course(row):
                probe_settings = dict(settings)
                probe_settings['evaluation_workers'] = 2
                probe_stage = replace(stages[0], tracks=(row['path'],),
                    allow_archived_task_resets=False, random_gate=False,
                    fixed_start_gate_index=0, target_speed=16.5)
                result = evaluate_policy(policy, normalizer, probe_settings,
                    probe_stage, count=4,
                    seed_base=int(settings['seed'])+700000+total_environment_steps,
                    device=device)
                accepted = reachable(result, len(load_track(row['path']).gates))
                print(f"pool_probe name={row['name']} full={result['full_course_success']:.3f} "
                      f"gates={result['mean_gates']:.3f} accepted={int(accepted)}", flush=True)
                return accepted
            changed, raw_stage, weights, phase = pool_manager.update(
                total_environment_steps, probe_pool_course)
            if changed or cycle == initial_cycle+1:
                stages[0] = parse_stage(raw_stage)
                settings['curriculum'][0] = raw_stage
                settings['ppo_track_sampling_weights'] = weights
                settings.pop('ppo_track_sampling_family_weights', None)
                settings['ppo_group_constraints']['success_floor'] = phase['success_floor']
                records = json.loads(Path(pool_raw['core_manifest']).read_text())['records']
                manifest = Path(pool_raw['active_manifest'])
                atomic_json(manifest, {'schema':'starscream-procedural-track-manifest-v1',
                    'records':records+pool_manager.state['accepted']})
                settings['track_manifest'] = str(manifest)
                replace_collector(0)
                print(f"pool_phase={phase['stage']['name']} courses={len(stages[0].tracks)} "
                      f"steps={total_environment_steps}", flush=True)
        stage = stages[stage_index]
        if collector_stage_index != stage_index:
            replace_collector(stage_index)
            if bool(settings.get("recalibrate_safety_each_stage", True)):
                stage_baseline = refinement_metrics(evaluate_current(
                    stage_index,
                    int(settings.get("evaluation_seed", 20261000))
                    + stage_index * int(settings.get("evaluation_stage_seed_stride", 10000)),
                ), evaluation_stages[stage_index], settings)
                safe_state = {
                    name: value.detach().cpu().clone()
                    for name, value in policy.state_dict().items()
                }
                safe_score = float(stage_baseline["selection_score"])
                stage_settings = _merge(
                    dict(settings), dict(settings["curriculum"][stage_index])
                )
                rollback_full_course_threshold, rollback_track_thresholds = (
                    reliability_floors(stage_baseline, stage_settings)
                )
                stage_baseline.update(reliability_constraint_metrics(
                    stage_baseline,
                    rollback_full_course_threshold,
                    rollback_track_thresholds,
                    confidence_z=float(settings.get("rollback_confidence_z", 0.0)),
                ))
                stage_baseline.update({
                    "curriculum_stage": float(stage_index),
                    "environment_steps": float(total_environment_steps),
                    "safety_rollback": 0.0,
                    "stage_safety_recalibration": 1.0,
                })
                logger.log_eval(stage_baseline, total_environment_steps)
                print(
                    f"stage_safety_baseline stage={stage.name} "
                    f"full={stage_baseline['full_course_success']:.3f} "
                    f"rollback_full={rollback_full_course_threshold:.3f}",
                    flush=True,
                )
        if collector is not None:
            rollout, results, steps_per_second = collector.collect(
                critic,
                episodes=int(settings.get("episodes_per_cycle", 96)),
                seed_base=seed + cycle * 100000,
            )
            constraint_lambdas, constraint_metrics = (
                collector.update_group_constraints(results)
            )
            if collector.lap_time_constraints is not None:
                constraint_metrics.update(collector.lap_time_constraints.update(results))
            adaptive_sampling_metrics = collector.update_adaptive_sampling(
                results, rollout,
            )
            if bank is not None:
                bank_added, bank_multipliers, bank_metrics = bank.step(
                    collector._adaptive_competence, cycle,
                )
                collector.track_weights = collector.track_weights * bank_multipliers
                collector.add_tracks(bank_added)
                role_floors = settings.get("ppo_sampling_role_floors", {})
                if role_floors:
                    from starscream.ppo_windows import apply_group_probability_floors
                    collector.track_weights = apply_group_probability_floors(
                        collector.track_weights, collector.track_roles, role_floors,
                    )
                    role_labels = np.asarray(collector.track_roles, dtype=object)
                    for role in sorted(set(collector.track_roles)):
                        mask = role_labels == role
                        bank_metrics[
                            f"bank_role/{role}/next_probability"
                        ] = float(
                            collector.track_weights[mask].sum()
                            / collector.track_weights.sum()
                        )
                adaptive_sampling_metrics.update(bank_metrics)
            adaptive_sampling_metrics.update(
                collector.update_paper_task_switching(results)
            )
            if settings.get('ppo_rollout_window_steps', 0):
                window = collector.last_window_metrics
                adaptive_sampling_metrics.update({f'window_{key}': value for key, value in window.items()
                                                 if isinstance(value, (int, float))})
                adaptive_sampling_metrics.update({f'window_track_{i}_transitions': count
                    for i, count in enumerate(window['course_transition_counts'])})
        else:
            rollout, results, steps_per_second = collect_ppo(
                policy, critic, normalizer, settings, stage,
                episodes=int(settings.get("episodes_per_cycle", 96)),
                seed_base=seed + cycle * 100000, device=device,
            )
            constraint_lambdas = None
            constraint_metrics = {}
            adaptive_sampling_metrics = {}
        if (
            float(settings.get("ppo_dynamics_weight", 0.0)) > 0.0
            or bool(settings.get("ppo_phasic_dynamics", {}).get("enabled", False))
        ):
            rollout["dynamics_target"] = normalize_dynamics_targets(
                rollout["task_delta"],
                initial["dynamics_target_mean"],
                initial["dynamics_target_std"],
            )
        cycle_steps = len(rollout["advantage"])
        collected_at = time.perf_counter()
        if task_blocks:
            from collections import Counter
            episode_counts = Counter(item['track'] for item in results)
            transition_counts = torch.bincount(rollout['track_id'],minlength=len(stage.tracks))
            if (len(episode_counts)!=len(stage.tracks)
                or len(set(episode_counts.values()))!=1):
                raise RuntimeError(f'unbalanced dense PPO episodes: {episode_counts}')
            print(f"ppo_collection cycle={cycle} steps={cycle_steps} "
                f"episodes_per_course={min(episode_counts.values())} "
                f"transitions_per_course={transition_counts.tolist()} "
                f"sps={steps_per_second:.1f}",flush=True)
        total_environment_steps += cycle_steps
        stage_environment_steps += cycle_steps
        if pool_manager is not None:
            from starscream.course_model.training import atomic_json
            atomic_json(Path(pool_raw['progress_path']), {
                'steps':total_environment_steps, 'cycle':cycle,
                'phase':pool_manager.state['phase'],
                'accepted':len(pool_manager.state['accepted'])})
        progress = total_environment_steps / max(int(settings.get("target_environment_steps", 5000000)), 1)
        anchor = max(
            float(settings.get("anchor_weight_end", 0.002)),
            float(settings.get("anchor_weight_start", 0.04)) * (1.0 - min(progress, 1.0)),
        )
        from starscream.ppo_windows import critic_warmup_settings
        update_settings, critic_warmup = critic_warmup_settings(settings, cycle)
        if critic_warmup and (flow_fpo or not settings.get('ppo_rollout_window_steps', 0)
                              or settings.get('ppo_phasic_dynamics', {}).get('enabled', False)):
            raise ValueError('critic-only warmup requires plain fixed-window PPO without phasic updates')
        update = (
            update_privileged_fpo(
                rollout, policy, reference, critic,
                actor_optimizer, critic_optimizer, settings,
                device=device, anchor_weight=anchor,
                constraint_lambdas=constraint_lambdas,
            )
            if flow_fpo else update_ppo(
                rollout, policy, reference, critic,
                actor_optimizer, critic_optimizer, update_settings,
                device=device, anchor_weight=anchor,
                constraint_lambdas=constraint_lambdas,
            )
        )
        phasic_metrics: dict[str, float] = {}
        phasic_config = settings.get("ppo_phasic_dynamics", {})
        phasic_interval = max(int(phasic_config.get("interval", 4)), 1)
        if (
            not flow_fpo
            and bool(phasic_config.get("enabled", False))
            and cycle % phasic_interval == 0
        ):
            phasic_metrics = update_policy_preserving_dynamics(
                rollout, policy, settings, device=device,
            )
        kl_statistic = str(settings.get("kl_adaptation_statistic", "max"))
        if kl_statistic not in {"mean", "max", "epoch", "epoch_last"}:
            raise ValueError("kl_adaptation_statistic must be mean, max, epoch, or epoch_last")
        kl_key = {
            "mean": "approximate_kl",
            "max": "approximate_kl_max",
            "epoch": "epoch_approximate_kl",
            "epoch_last": "epoch_approximate_kl_last",
        }[kl_statistic]
        approximate_kl = float(update.get(kl_key, update.get("approximate_kl", 0.0)))
        target_kl = float(settings.get("target_kl", 0.02))
        if not critic_warmup and bool(settings.get("adaptive_actor_learning_rate", True)):
            if target_kl > 0 and approximate_kl > 1.5 * target_kl:
                actor_base_learning_rate *= float(settings.get("kl_lr_backoff", 0.5))
            elif target_kl > 0 and approximate_kl < 0.30 * target_kl:
                actor_base_learning_rate *= float(settings.get("kl_lr_growth", 1.02))
        actor_base_learning_rate = float(np.clip(
            actor_base_learning_rate,
            float(settings.get("actor_learning_rate_min", 1.0e-6)),
            float(settings.get("actor_learning_rate_max", settings.get("actor_learning_rate", 3.0e-4))),
        ))
        set_actor_learning_rate(actor_optimizer, actor_base_learning_rate)
        # A short persistent window may finish no episodes. That is missing
        # outcome data, not zero success; never manufacture a failure-rate point.
        train_outcomes = refinement_metrics(
            multitrack_metrics(results, stage.target_gates), stage, settings
        ) if results else {}
        reward_metrics = ppo_reward_metrics(results) if results else {}
        train_metrics = {
            **update,
            **phasic_metrics,
            **constraint_metrics,
            **adaptive_sampling_metrics,
            **reward_metrics,
            **{f"outcome_{key}": value for key, value in train_outcomes.items()},
            "cycle": cycle,
            "critic_only_warmup": float(critic_warmup),
            "curriculum_stage": stage_index,
            "environment_steps": total_environment_steps,
            "environment_steps_per_second": steps_per_second,
            "anchor_weight": anchor,
            "actor_learning_rate": actor_base_learning_rate,
            "kl_adaptation_value": approximate_kl,
            "rollout_collection_seconds": collected_at-cycle_started,
            "rollout_update_seconds": time.perf_counter()-collected_at,
            "rollout_episodes": len(results),
            "rollout_transitions": cycle_steps,
        }
        if task_blocks:
            train_metrics.update(
                rollout_episodes_per_course_min=min(episode_counts.values()),
                rollout_episodes_per_course_max=max(episode_counts.values()),
                rollout_transitions_per_course_min=int(transition_counts.min()),
                rollout_transitions_per_course_max=int(transition_counts.max()),
                rollout_active_courses=len(stage.tracks))
        logger.log_train(train_metrics, total_environment_steps)
        # Neither evaluation nor checkpointing needs training samples. Release
        # them before spawning evaluators and before allocating the next window.
        if settings.get("ppo_release_host_memory", False):
            from starscream.ppo_windows import release_rollout_storage
            release_rollout_storage(rollout)
        del rollout
        post_update_started = time.perf_counter()
        evaluation: dict[str, float] = {}
        reporting_evaluation: dict[str, float] | None = None
        final_cycle = (total_environment_steps >= int(settings.get("target_environment_steps", 5000000))
                       or cycle == int(settings.get("cycles", 200)))
        stage_budget_due = bool(
            settings.get("evaluation_on_stage_budget", False)
            and stage_index < len(stages) - 1
            and stage_environment_steps >= int(settings["curriculum"][stage_index].get("minimum_environment_steps", 0))
        )
        if cycle == 1 or final_cycle or stage_budget_due or cycle % int(settings.get("evaluation_interval", 2)) == 0:
            evaluated_stage_index = stage_index
            evaluation = evaluate_current(
                stage_index,
                int(settings.get("evaluation_seed", 20261000)) + stage_index * int(settings.get("evaluation_stage_seed_stride", 10000)),
            )
            evaluation = refinement_metrics(
                evaluation, evaluation_stages[stage_index], settings
            )
            evaluation["curriculum_stage"] = float(stage_index)
            evaluation["cycle"] = float(cycle)
            evaluation["environment_steps"] = float(total_environment_steps)
            evaluation["safety_rollback"] = 0.0
            reliability = reliability_constraint_metrics(
                evaluation,
                rollback_full_course_threshold,
                rollback_track_thresholds,
                confidence_z=float(settings.get("rollback_confidence_z", 0.0)),
            )
            evaluation.update(reliability)
            threshold = float(settings["curriculum"][stage_index].get("advancement_success", 1.0))
            minimum = int(settings["curriculum"][stage_index].get("minimum_environment_steps", 0))
            maximum_lap_time = settings["curriculum"][stage_index].get(
                "advancement_lap_time_seconds"
            )
            pace_ready = bool(
                maximum_lap_time is None
                or (
                    np.isfinite(evaluation["successful_lap_time_seconds"])
                    and evaluation["successful_lap_time_seconds"]
                    <= float(maximum_lap_time)
                )
            )
            evaluation["advancement_pace_ready"] = float(pace_ready)
            configured_advancement_metric = settings["curriculum"][stage_index].get(
                "advancement_metric"
            )
            target_key = (
                str(configured_advancement_metric)
                if configured_advancement_metric is not None
                else (
                    "full_course_success"
                    if stage.target_gates >= course_gate_count
                    else f"p{stage.target_gates}"
                )
            )
            if target_key not in evaluation:
                raise ValueError(
                    f"curriculum advancement metric {target_key!r} is unavailable"
                )
            competence = float(evaluation.get(target_key, 0.0))
            evaluation["advancement_competence"] = competence
            evaluation["advancement_threshold"] = threshold
            rollback = bool(
                bool(settings.get("enable_safety_rollback", True))
                and stage.target_gates >= course_gate_count
                and not bool(reliability["reliability_constraints_met"])
            )
            # Reporting and candidate archival must observe the policy that was
            # actually evaluated.  In v1 both happened after restoration, which
            # made Swift telemetry look frozen and discarded useful frontier
            # policies before they could be characterized.
            if (
                cycle == 1 or final_cycle or stage_budget_due
                or cycle % int(settings.get(
                    "reporting_evaluation_interval",
                    settings.get("evaluation_interval", 2),
                )) == 0
            ):
                reporting_evaluation = evaluate_reporting(
                    evaluated_stage_index, total_environment_steps
                )
            evaluation["safety_rollback"] = float(rollback)
            evaluation["candidate_preserved"] = 1.0
            if (
                online_manager is not None
                and not rollback
                and cycle >= int(online_raw.get("warmup_cycles", 5))
                and cycle % max(
                    int(online_raw.get("decision_interval_cycles", 5)), 1
                ) == 0
                and (
                    online_manager.candidate_probe_names()
                    or bool(online_raw.get("ordered_geometry_ladder", False))
                )
            ):
                probed_names = online_manager.candidate_probe_names()
                probe_names = (online_manager.source_name, *probed_names)
                probe_stage = replace(
                    evaluation_stages[evaluated_stage_index],
                    name="online_manifold_frozen_policy_probe",
                    tracks=tuple(
                        online_manager.tasks[name].path for name in probe_names
                    ),
                    target_speed=float(
                        online_raw.get("probe_target_speed", 16.5)
                    ),
                    target_speed_range=None,
                    manifest_speed_scale_range=None,
                )
                episodes_per_track = int(
                    online_raw.get("probe_episodes_per_track", 12)
                )
                if episodes_per_track < 1:
                    raise ValueError(
                        "online manifold probe_episodes_per_track must be positive"
                    )
                probe_seed = (
                    int(online_raw.get("probe_seed", 2026094701))
                    + cycle * 100003
                )
                if bool(online_raw.get("matched_track_seeds", False)):
                    probe_metrics = evaluate_policy_matched_tracks(
                        policy, normalizer, settings, probe_stage,
                        count_per_track=episodes_per_track,
                        seed_base=probe_seed, device=device,
                    )
                else:
                    probe_metrics = evaluate_policy(
                        policy, normalizer, settings, probe_stage,
                        count=episodes_per_track * len(probe_names),
                        seed_base=probe_seed, device=device,
                    )
                online_decision, online_changed = online_manager.decide(
                    probe_metrics, evaluation, cycle=cycle,
                    reporting_metrics=reporting_evaluation,
                    environment_steps=total_environment_steps,
                )
                evaluation.update({
                    "online_manifold/generation": float(
                        online_manager.generation
                    ),
                    "online_manifold/active_tracks": float(
                        len(online_manager.active_names)
                    ),
                    "online_manifold/source_retention": float(
                        probe_metrics.get(
                            f"track/{online_manager.source_name}/full_course_success",
                            0.0,
                        )
                    ),
                    "online_manifold/action_expand": float(
                        online_decision.action == "expand"
                    ),
                    "online_manifold/action_rehearse": float(
                        online_decision.action == "rehearse"
                    ),
                    "online_manifold/action_retreat": float(
                        online_decision.action == "retreat"
                    ),
                })
                for name in probed_names:
                    for key in (
                        "full_course_success", "mean_gates", "crash_rate"
                    ):
                        evaluation[
                            f"online_manifold/probe/{name}/{key}"
                        ] = float(
                            probe_metrics.get(f"track/{name}/{key}", 0.0)
                        )
                if online_changed:
                    settings["track_manifest"] = str(
                        online_manager.active_manifest_path
                    )
                    settings["ppo_track_sampling_weights"] = (
                        online_manager.track_sampling_weights()
                    )
                    stages[0] = replace(
                        stages[0], tracks=online_manager.rollout_paths,
                    )
                    evaluation_stages[0] = replace(
                        evaluation_stages[0], tracks=online_manager.active_paths,
                    )
                    # The completed rollout and update used the old immutable
                    # generation. Workers are replaced only for the next one.
                    collector_stage_index = -1
                print(
                    f"online_manifold cycle={cycle} "
                    f"generation={online_manager.generation} "
                    f"action={online_decision.action} "
                    f"selected={online_decision.selected_name} "
                    f"frontier={online_manager.frontier_name} "
                    f"active={len(online_manager.active_names)} "
                    f"reason={online_decision.reason}",
                    flush=True,
                )
            ranking_metrics = (
                reporting_evaluation
                if checkpoint_metric_source == "reporting"
                else evaluation
            )
            manager.save(
                checkpoint_payload(
                    policy, normalizer, stage=settings_key, track=str(settings["track"]),
                    optimizer=actor_optimizer,
                    extra={
                        "critic": critic.state_dict(),
                        "critic_architecture": str(settings.get("critic_architecture", "mlp")),
                        "critic_optimizer": critic_optimizer.state_dict(),
                        "cycle": cycle,
                        **continuation_state(),
                        "environment_steps": total_environment_steps,
                        "curriculum_stage": stage_index,
                        "initial_checkpoint": str(initial_path),
                        "dynamics_target_mean": initial["dynamics_target_mean"],
                        "dynamics_target_std": initial["dynamics_target_std"],
                        "training_config": config,
                        "reporting_evaluation": reporting_evaluation,
                        "candidate_pre_rollback": True,
                        "ppo_group_constraint_state": constraint_state_dict(),
                        "online_course_bank_state": (None if bank is None else bank.state_dict()),
                        "green2026_adaptive_task_switching_state": (
                            None if collector is None
                            else collector.paper_task_switching_state_dict()
                        ),
                        "online_manifold_curriculum_state": (
                            None if online_manager is None
                            else online_manager.state_dict()
                        ),
                    },
                ),
                step=total_environment_steps,
                metrics=(ranking_metrics or evaluation),
                rank=(
                    ranking_metrics is not None
                    and bool(evaluation.get('lap_completion_qualified', 1.))
                    and ppo_rank_curriculum_stage(
                        settings, evaluated_stage_index, len(stages)
                    )
                ),
            )
            if rollback:
                if online_manager is not None and online_manager.task_selector is not None:
                    # A restored policy invalidates the preceding transfer-response window.
                    online_manager.task_selector.previous = None
                policy.load_state_dict(safe_state)
                actor_base_learning_rate = max(
                    float(settings.get("actor_learning_rate_min", 1.0e-6)),
                    actor_base_learning_rate * float(settings.get("rollback_lr_backoff", 0.5)),
                )
                actor_optimizer = (
                    build_privileged_flow_optimizer(policy, settings, device)
                    if flow_fpo else build_actor_optimizer(policy, settings, device)
                )
                set_actor_learning_rate(actor_optimizer, actor_base_learning_rate)
                print(
                    f"safety_rollback cycle={cycle} full="
                    f"{evaluation.get('full_course_success',0):.3f} "
                    f"violations={int(reliability['reliability_violation_count'])} "
                    f"margin={reliability['reliability_minimum_margin']:.3f} "
                    f"restored_score={safe_score:.3f} lr={actor_base_learning_rate:.2e}",
                    flush=True,
                )
            elif (bool(settings.get("safety_use_latest_accepted", False))
                  or float(evaluation["selection_score"]) > safe_score):
                safe_state = {
                    name: value.detach().cpu().clone()
                    for name, value in policy.state_dict().items()
                }
                safe_score = float(evaluation["selection_score"])
            if (
                not rollback and stage_index + 1 < len(stages)
                and stage_environment_steps >= minimum
                and competence >= threshold
                and pace_ready
            ):
                stage_index += 1
                stage_environment_steps = 0
            logger.log_eval(evaluation, total_environment_steps)
        rollback = bool(evaluation.get("safety_rollback", 0.0))
        metrics = evaluation or train_outcomes
        checkpoint_started = time.perf_counter()
        manager.save(
            checkpoint_payload(
                policy, normalizer, stage=settings_key, track=str(settings["track"]),
                optimizer=actor_optimizer,
                extra={
                    "critic": critic.state_dict(),
                    "critic_architecture": str(settings.get("critic_architecture", "mlp")),
                    "critic_optimizer": critic_optimizer.state_dict(),
                    "cycle": cycle,
                    **continuation_state(),
                    "environment_steps": total_environment_steps,
                    "curriculum_stage": stage_index,
                    "initial_checkpoint": str(initial_path),
                    "dynamics_target_mean": initial["dynamics_target_mean"],
                    "dynamics_target_std": initial["dynamics_target_std"],
                    "training_config": config,
                    "reporting_evaluation": (
                        None if rollback else reporting_evaluation
                    ),
                    "candidate_reporting_evaluation": (
                        reporting_evaluation if rollback else None
                    ),
                    "restored_after_rollback": rollback,
                    "ppo_pool_curriculum_state": (
                        None if pool_manager is None else pool_manager.state),
                    "ppo_group_constraint_state": constraint_state_dict(),
                    "online_course_bank_state": (None if bank is None else bank.state_dict()),
                    "green2026_adaptive_task_switching_state": (
                        None if collector is None
                        else collector.paper_task_switching_state_dict()
                    ),
                    "online_manifold_curriculum_state": (
                        None if online_manager is None
                        else online_manager.state_dict()
                    ),
                },
            ),
            step=total_environment_steps, metrics=metrics,
            # Evaluated candidates were already ranked before any restoration.
            rank=False,
        )
        cycle_finished = time.perf_counter()
        # Collection SPS alone omits optimization, evaluation and checkpoint IO.
        logger.log_train({
            "environment_steps": total_environment_steps,
            "performance/full_cycle_seconds": cycle_finished - cycle_started,
            "performance/full_cycle_steps_per_second": cycle_steps / max(cycle_finished-cycle_started, 1e-6),
            "performance/evaluation_and_ranking_seconds": checkpoint_started - post_update_started,
            "performance/checkpoint_seconds": cycle_finished - checkpoint_started,
        }, total_environment_steps)
        print(
            f"cycle={cycle} stage={stage.name} steps={total_environment_steps} "
            f"train_p1={train_outcomes.get('p1',0):.3f} "
            f"train_p2={train_outcomes.get('p2',0):.3f} "
            f"train_lap1={train_outcomes.get('lap_count/1/full_course_success',float('nan')):.3f} "
            f"train_lap2={train_outcomes.get('lap_count/2/full_course_success',float('nan')):.3f} "
            f"eval_p2={evaluation.get('p2',float('nan')):.3f} "
            f"eval_full={evaluation.get('full_course_success',float('nan')):.3f} "
            f"eval_lap={evaluation.get('successful_lap_time_seconds',float('nan')):.3f} "
            f"kl={update.get('approximate_kl',0):.5f} "
            f"kl_max={update.get('approximate_kl_max',0):.5f} "
            f"clip={update.get('clip_fraction',0):.3f} "
            f"epoch_kl={update.get('epoch_approximate_kl',float('nan')):.5f} "
            f"epochs={update.get('actor_epochs_completed',0):.0f} "
            f"updates={update.get('actor_updates_completed',0):.0f} "
            f"reward={reward_metrics.get('reward/adjusted_return_mean',float('nan')):.2f} "
            f"speed_r={reward_metrics.get('reward/speed_progress_per_step',float('nan')):.4f} "
            f"gate_r={reward_metrics.get('reward/gate_pass_per_step',float('nan')):.4f} "
            f"crash_r={reward_metrics.get('reward/crash_per_step',float('nan')):.4f} "
            f"dual={constraint_metrics.get('lap_constraint_dual_max', constraint_metrics.get('constraint_lambda_max',0)):.3f} "
            f"plr={adaptive_sampling_metrics.get('adaptive_sampling_multiplier_max',1):.2f} "
            f"lr={actor_base_learning_rate:.2e} steps_s={steps_per_second:.1f}", flush=True,
        )
        if total_environment_steps >= int(settings.get("target_environment_steps", 5000000)):
            break
    if collector is not None:
        collector.close()
    logger.finish()


def main() -> None:
    args, config = parse_args()
    torch.set_float32_matmul_precision("high")
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if args.stage == "bc":
        run_bc(config, args.device)
    elif args.stage == "dagger":
        run_dagger(config, args.device)
    elif args.stage == "aggrevate":
        run_aggrevate(config, args.device)
    elif args.stage == "fpo":
        run_ppo(config, args.device, settings_key="fpo")
    else:
        run_ppo(config, args.device)


if __name__ == "__main__":
    main()
