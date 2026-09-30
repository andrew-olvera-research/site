"""Method-preserving DAgger scheduling utilities (no controller or loss math)."""
from __future__ import annotations

from typing import Mapping
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import numpy as np


def fixed_group_statistics(values, groups, count):
    """Logging-only group statistics without dynamic shapes or scalar reads."""
    import torch
    if count < 1:
        raise ValueError('group count must be positive')
    with torch.no_grad():
        valid = groups >= 0
        indices = groups.clamp_min(0).long()
        sums = values.new_zeros(count).scatter_add_(0, indices, torch.where(valid, values.detach(), 0.))
        sizes = values.new_zeros(count).scatter_add_(0, indices, valid.to(values.dtype))
        present = sizes > 0
        means = sums / sizes.clamp_min(1)
        n = present.sum().clamp_min(1)
        average = means.sum()/n
        variance = ((means-average).square()*present).sum()/n
        maximum = torch.where(present,means,-float('inf')).max()
        any_present = present.any()
        fallback = values.detach().mean()
        return (torch.where(any_present,average,fallback),
                torch.where(any_present,maximum,fallback),
                torch.where(any_present,variance.sqrt(),values.new_ones(())))


@contextmanager
def prefetched_batches(batches, depth=0):
    """One ordered CPU producer; bounded lookahead, errors and shutdown propagate.

    The producer exclusively owns the sampling RNG until context exit. It must
    not touch the policy/CUDA or mutate replay. Depth zero is the reference path.
    """
    if not 0 <= depth <= 8:
        raise ValueError("DAgger batch prefetch depth must be between zero and eight")
    iterator = iter(batches)
    if depth == 0:
        yield iterator
        return
    sentinel = object()
    def advance():
        return next(iterator, sentinel)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="dagger-batch") as executor:
        futures = deque(executor.submit(advance) for _ in range(depth))
        def consume():
            while futures:
                result = futures.popleft().result()
                if result is sentinel:
                    return
                futures.append(executor.submit(advance))
                yield result
        try:
            yield consume()
        finally:
            for future in futures:
                future.cancel()


class HierarchicalReplayPlan:
    """Compile immutable replay metadata into an exact RNG-equivalent draw plan.

    Rebuild after every replay append/eviction or change in weights, fractions,
    or batch quota. Never cache by array identity: replay can change in place.
    Only index pools are retained, not observation/action data.
    """

    def __init__(self, family_ids, track_ids, gate_indices, event_mask,
                 teacher_modes, occupancy_modes, size, *, family_weights=None,
                 nominal_fraction=.40, critical_fraction=.35, recovery_fraction=.25):
        arrays = [np.asarray(x).reshape(-1) for x in (
            family_ids, track_ids, gate_indices, np.asarray(event_mask, bool),
            teacher_modes, occupancy_modes)]
        if len({len(x) for x in arrays}) != 1 or not len(arrays[0]):
            raise ValueError("hierarchical DAgger replay arrays must be aligned and nonempty")
        if size < 0:
            raise ValueError("hierarchical DAgger replay size must be non-negative")
        fractions = np.asarray([nominal_fraction, critical_fraction, recovery_fraction], np.float64)
        if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1):
            raise ValueError("DAgger trajectory fractions must be non-negative and sum to one")
        families = np.unique(arrays[0])
        configured = dict(family_weights or {})
        weights = np.asarray([float(configured.get(int(f), 1.)) for f in families], np.float64)
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("DAgger family replay weights must be finite and positive")
        self.size = size
        self.weights = weights
        self.fractions = fractions
        self.tree = []
        for family in families:
            fp = np.flatnonzero(arrays[0] == family)
            tracks = []
            for track in np.unique(arrays[1][fp]):
                pool = fp[arrays[1][fp] == track]
                critical = arrays[3][pool]
                recovery = (arrays[4][pool] != 0) | (arrays[5][pool] == 1)
                strata = (pool[~critical & ~recovery], pool[critical], pool[~critical & recovery])
                tracks.append([
                    [source[arrays[2][source] == gate] for gate in np.unique(arrays[2][source])]
                    for source in (stratum if len(stratum) else pool for stratum in strata)
                ])
            self.tree.append(tracks)

    @staticmethod
    def quotas(rng, size, weights):
        # Randomized systematic rounding is unbiased even when quota < groups.
        weights = np.asarray(weights, np.float64)
        cumulative = np.cumsum(weights / weights.sum()) * size
        cumulative[-1] = size
        return np.bincount(np.searchsorted(cumulative, np.arange(size) + rng.random(),
                                          side='right'), minlength=len(weights))

    @property
    def index_bytes(self):
        return sum(pool.nbytes for tracks in self.tree for strata in tracks
                   for groups in strata for pool in groups)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        selected = []
        for tracks, family_count in zip(self.tree, self.quotas(rng, self.size, self.weights)):
            if not family_count:
                continue
            for strata, track_count in zip(tracks, self.quotas(rng, family_count, np.ones(len(tracks)))):
                if not track_count:
                    continue
                for groups, count in zip(strata, self.quotas(rng, track_count, self.fractions)):
                    if not count:
                        continue
                    for pool, quota in zip(groups, self.quotas(rng, count, np.ones(len(groups)))):
                        if quota:
                            selected.append(pool[rng.integers(0, len(pool), size=int(quota))])
        result = np.concatenate(selected) if selected else np.empty(0, np.int64)
        if len(result) != self.size:
            raise RuntimeError('hierarchical replay quota mismatch')
        rng.shuffle(result)
        return result


# Deliberately excludes batch size, episode/step counts, teacher math, precision,
# data retention, family/course weights and all observation/loss settings.
HARDWARE_KEYS = frozenset({
    "collection_compact_observation", "native_vector_step", "native_vector_threads",
    "rollout_envs", "dagger_worker_cpu_threads", "dagger_process_start_method",
    "dagger_async_collection", "dagger_async_min_inference_batch", "dagger_async_batch_wait_ms",
    "evaluation_workers", "dagger_persistent_evaluation", "dagger_cached_replay_sampling",
    "compile_policy_backbone", "compile_policy_inference", "compile_threads",
    "dagger_batch_prefetch",
    "dagger_packed_update_transfer", "defer_update_metrics", "host_evaluation_graph",
    "cuda_graph_policy_inference",
    "current_route_projection_only", "dagger_fast_group_statistics",
    "evaluation_envs_per_worker", "dagger_static_group_statistics", "dagger_capture_updates",
    "dagger_overlap_teacher_inference",
    "dagger_dispatch_teacher_first", "dagger_chunked_labels",
    "dagger_event_inference",
    "dagger_host_inference_graph", "dagger_packed_transport", "dagger_lazy_teacher_actions",
    "mpcc_native_reference", "mpcc_native_projection", "mpcc_cache_runtime_vehicle",
    "mpcc_native_model_step", "mpcc_prediction_backend", "mpcc_native_stage_updates",
})

PPO_HARDWARE_KEYS = frozenset({
    "ppo_window_max_host_bytes",
    "collection_compact_observation", "native_vector_step", "native_vector_threads",
    "ppo_packed_group_transport",
    "defer_update_metrics", "host_evaluation_graph",
    "rollout_envs", "ppo_envs_per_worker", "ppo_packed_transport",
    "ppo_fused_rollout_transfer", "ppo_reuse_bootstrap_values",
    "cuda_graph_ppo_inference", "cuda_graph_policy_inference",
    "current_route_projection_only", "evaluation_workers", "evaluation_envs_per_worker",
    "compile_policy_backbone", "compile_policy_inference", "compile_threads",
})


def hardware_overlay(settings: Mapping, overrides: Mapping, *, stage="dagger") -> dict:
    if stage not in {"dagger", "ppo"}:
        raise ValueError('hardware profiles support DAgger and PPO only')
    unknown = set(overrides) - (HARDWARE_KEYS if stage == "dagger" else PPO_HARDWARE_KEYS)
    if unknown:
        raise ValueError(f"not hardware-only {stage} settings: {sorted(unknown)}")
    merged = {**settings, **overrides}
    if ('mpcc_prediction_backend' in overrides
            and overrides['mpcc_prediction_backend'] not in {'numpy', 'native_exact'}):
        raise ValueError('hardware profiles require an exact MPCC prediction backend')
    switching = merged.get('green2026_adaptive_task_switching', {})
    persistent = bool(merged.get('ppo_rollout_window_steps', 0)) or merged.get('ppo_persistent_task_slots', False) or (
        isinstance(switching, Mapping) and switching.get('enabled', False))
    if stage == 'ppo' and persistent and 'rollout_envs' in overrides:
        if int(overrides['rollout_envs']) != int(settings.get('rollout_envs', 12)):
            raise ValueError('persistent PPO active task count is part of the method; '
                             'omit rollout_envs from this hardware profile and tune '
                             'ppo_envs_per_worker instead')
    return merged
