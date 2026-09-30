#!/usr/bin/env python3
"""Fit scratch-DAgger units from an expert-only collection.

This is a normalization bootstrap, not behavior pretraining.  It executes the
configured MPCC teacher with beta=1 under the exact DAgger environment,
reconstructs raw final-step features using an identity normalizer, balances
rows by concrete track, and saves only feature/dynamics statistics.  No actor
weights or replay transitions enter the output artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_privileged_racing import (
    ProcessDaggerCollector,
    merge_dagger_batches,
    configured_tracks,
    load_config,
    parse_stage,
)
from starscream.agile_generalization import LEGACY_OBSERVATION_CONTRACT
from starscream.privileged_racing import (
    FeatureNormalizer,
    initialize_scratch_policy,
    privileged_feature_dim,
)


def balanced_track_indices(
    tracks: np.ndarray, seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return an equal-count deterministic sample from every concrete track."""

    tracks = np.asarray(tracks, dtype=str)
    if tracks.ndim != 1 or not len(tracks):
        raise ValueError("track labels must be a non-empty vector")
    names, counts = np.unique(tracks, return_counts=True)
    if np.any(counts < 1):
        raise ValueError("every concrete track must contain at least one row")
    count = int(counts.min())
    rng = np.random.default_rng(seed)
    selected = np.concatenate([
        rng.choice(np.flatnonzero(tracks == name), size=count, replace=False)
        for name in names
    ]).astype(np.int64)
    rng.shuffle(selected)
    return selected, {str(name): count for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026090310)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--coverage-retry-episodes-per-track", type=int, default=0)
    args = parser.parse_args()
    if args.episodes < 3:
        raise ValueError("expert bootstrap requires at least three episodes")
    if args.coverage_retry_episodes_per_track < 0:
        raise ValueError("coverage retry budget must be non-negative")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"statistics output already exists: {args.output}")

    config = load_config(args.config)
    settings = dict(config["dagger"])
    tracks = configured_tracks(settings)
    if args.episodes % len(tracks):
        raise ValueError("episodes must divide evenly across configured tracks")
    route_gate_count = int(settings.get("route_gate_count", 3))
    observation_contract = str(settings.get(
        "observation_contract", LEGACY_OBSERVATION_CONTRACT,
    ))
    feature_dim = privileged_feature_dim(route_gate_count, observation_contract)
    model_config = dict(settings["model"])
    model_config["input_dim"] = feature_dim
    model_config["observation_contract"] = observation_contract
    policy = initialize_scratch_policy(model_config, args.device)
    identity = FeatureNormalizer(
        np.zeros(feature_dim, np.float32), np.ones(feature_dim, np.float32),
    )

    # Preserve raw features exactly enough to fit units.  The production run
    # returns to compact fp16 collection/replay after this one-time preflight.
    settings["dagger_collection_history_dtype"] = "float32"
    settings["dagger_family_total_balanced_sampling"] = True
    stage = parse_stage(settings["curriculum"])
    collector = ProcessDaggerCollector(
        policy, identity, settings, stage, args.device,
    )
    try:
        collected_episodes = args.episodes
        batch = collector.collect(
            episodes=args.episodes,
            beta=1.0,
            seed_base=args.seed,
            tracks=tracks,
            dart=True,
        )
        missing = tuple(track for track in tracks
                        if not batch.accepted_episodes_by_track.get(track, 0))
        if missing and args.coverage_retry_episodes_per_track:
            retry_episodes = len(missing) * args.coverage_retry_episodes_per_track
            print(f"statistics coverage retry: tracks={len(missing)} episodes={retry_episodes}", flush=True)
            retry = collector.collect(episodes=retry_episodes, beta=1.0,
                                      seed_base=args.seed + 50000, tracks=missing, dart=True)
            merge_dagger_batches(batch, retry)
            collected_episodes += retry_episodes
    finally:
        collector.close()
    if not batch.histories or not batch.histories_normalized:
        raise RuntimeError("expert bootstrap returned no identity-normalized rows")

    histories = np.asarray(batch.histories, np.float32)
    features = histories[:, -1]
    row_tracks = np.asarray(batch.tracks, dtype=str)
    dynamics = np.asarray(batch.dynamics, np.float32)
    dynamics_valid = np.asarray(batch.dynamics_valid, np.bool_)
    if features.shape[1:] != (feature_dim,):
        raise ValueError(
            f"bootstrap feature width mismatch: {features.shape} vs {feature_dim}"
        )
    if not (
        len(features) == len(row_tracks) == len(dynamics) == len(dynamics_valid)
    ):
        raise ValueError("expert bootstrap arrays lost alignment")
    expected_tracks = {str(Path(track).resolve()) for track in tracks}
    observed_tracks = {str(Path(track).resolve()) for track in row_tracks}
    if observed_tracks != expected_tracks:
        raise ValueError(
            "expert bootstrap did not produce accepted rows for every track: "
            f"missing={sorted(expected_tracks - observed_tracks)} "
            f"unexpected={sorted(observed_tracks - expected_tracks)}"
        )
    selected, track_counts = balanced_track_indices(row_tracks, args.seed)
    normalizer = FeatureNormalizer.fit(features[selected])
    selected_valid = selected[dynamics_valid[selected]]
    if not len(selected_valid):
        raise ValueError("expert bootstrap contains no valid dynamics targets")
    dynamics_values = dynamics[selected_valid]

    payload = {
        "contract": "starscream-dagger-statistics-v1",
        "normalizer": normalizer.state_dict(),
        "dynamics_target_mean": dynamics_values.mean(0).astype(np.float32),
        "dynamics_target_std": np.maximum(
            dynamics_values.std(0), 1.0e-4,
        ).astype(np.float32),
        "feature_dim": feature_dim,
        "dynamics_dim": int(dynamics_values.shape[1]),
        "observation_contract": observation_contract,
        "route_gate_count": route_gate_count,
        "previous_action_feature_mapping": settings.get('previous_action_feature_mapping', 'legacy_linear'),
        "source_config": str(args.config.resolve()),
        "source": "balanced_expert_only_dagger_collection",
        "source_beta": 1.0,
        "source_dart": True,
        "source_histories_normalized_by_identity": True,
        "fit_transition": "causal_history_final_step",
        "episodes": int(collected_episodes),
        "accepted_episodes": int(batch.accepted_episodes),
        "rejected_episodes": int(batch.rejected_episodes),
        "valid_queries": int(batch.valid_queries),
        "total_queries": int(batch.total_queries),
        "teacher_solver_failures": int(batch.solver_failures),
        "track_counts": track_counts,
        "rows": int(len(selected)),
        "valid_dynamics_rows": int(len(selected_valid)),
        "seed": int(args.seed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{args.output.name}.", suffix=".tmp",
        dir=args.output.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({
        key: value for key, value in payload.items()
        if key not in {"normalizer", "dynamics_target_mean", "dynamics_target_std"}
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
