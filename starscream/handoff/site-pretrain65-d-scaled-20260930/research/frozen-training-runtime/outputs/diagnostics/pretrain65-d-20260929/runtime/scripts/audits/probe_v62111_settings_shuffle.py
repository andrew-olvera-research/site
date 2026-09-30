#!/usr/bin/env python3
"""Paired action-sensitivity check for the v6.21.1.1 privileged plant token."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.privileged_racing import load_policy_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=62111)
    args = parser.parse_args()

    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.replay, "r") as replay:
        online = replay["online"]
        indices = np.sort(rng.choice(len(online["actions"]),
                                     size=min(args.samples, len(online["actions"])),
                                     replace=False))
        histories = online["histories"][indices].astype(np.float32)
        speeds = online["speed_commands"][indices].astype(np.float32)
        labels = online["actions"][indices].astype(np.float32)
        tracks = online["tracks"][indices]
    assert histories.shape[1:] == (3, 167)
    assert np.isfinite(histories).all()
    assert len(np.unique(histories[:, -1, 103:152], axis=0)) > 1

    policy, normalizer, state, resolved = load_policy_checkpoint(args.checkpoint, "cpu")
    policy.eval()
    speed_tensor = torch.from_numpy(speeds)

    def predict(raw: np.ndarray) -> np.ndarray:
        normalized = normalizer.numpy(raw)
        with torch.inference_mode():
            return policy(torch.from_numpy(np.asarray(normalized, dtype=np.float32)),
                          speed_tensor).numpy()

    original = predict(histories)
    shuffled = histories.copy()
    shuffled[:, :, 103:167] = histories[rng.permutation(len(histories)), :, 103:167]
    shuffled_action = predict(shuffled)
    constants_only = histories.copy()
    constants_only[:, :, 103:152] = histories[rng.permutation(len(histories)), :, 103:152]
    constants_action = predict(constants_only)
    within_track = histories.copy()
    for track in np.unique(tracks):
        group = np.flatnonzero(tracks == track)
        if len(group) > 1:
            within_track[group, :, 103:152] = histories[rng.permutation(group), :, 103:152]
    within_track_action = predict(within_track)

    def summarize(actions: np.ndarray) -> dict:
        difference = np.abs(actions - original)
        return {
            "mean_abs_action_change": float(difference.mean()),
            "median_l2_action_change": float(np.median(np.linalg.norm(actions-original, axis=1))),
            "p90_l2_action_change": float(np.quantile(np.linalg.norm(actions-original, axis=1), .9)),
            "fraction_l2_change_above_0_05": float(np.mean(np.linalg.norm(actions-original, axis=1) > .05)),
            "mean_abs_change_by_action": difference.mean(axis=0).tolist(),
            "teacher_action_mse": float(np.square(actions-labels).mean()),
        }

    output = {
        "checkpoint": str(resolved),
        "checkpoint_step": state.get("environment_steps"),
        "replay": str(args.replay),
        "samples": len(histories),
        "distinct_tracks": int(len(np.unique(tracks))),
        "seed": args.seed,
        "original_teacher_action_mse": float(np.square(original-labels).mean()),
        "original_mean_abs_action": float(np.abs(original).mean()),
        "original_action_std_by_component": original.std(axis=0).tolist(),
        "full_settings_shuffle": summarize(shuffled_action),
        "episode_constants_shuffle": summarize(constants_action),
        "within_track_episode_constants_shuffle": summarize(within_track_action),
        "interpretation": "Action sensitivity on fixed replay states; this does not measure closed-loop benefit or causal improvement in validation success.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
