#!/usr/bin/env python3
"""Paired selection-suite probe with wrong episode plant constants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import train_privileged_racing as trainer
from starscream.evaluation_suite import configure_selection_suite, selection_metrics
from starscream.privileged_racing import load_policy_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-track", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=62112)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.replay, "r") as replay:
        data = replay["online/histories"]
        indices = np.sort(rng.choice(len(data), size=min(2048,len(data)), replace=False))
        plant_bank = data[indices,-1,103:152].astype(np.float32)
    settings = json.loads(Path("configs/exp/v6.21.1.1/plant_dagger.yaml").read_text())["dagger"]
    settings["evaluation_suite_episodes_per_track"] = args.episodes_per_track
    settings["evaluation_workers"] = args.workers
    configure_selection_suite(settings)
    policy, normalizer, state, resolved = load_policy_checkpoint(args.checkpoint, "cuda")
    policy.eval()
    stage = trainer.parse_stage(settings["evaluation_curriculum"])
    results = {}
    for arm in ("original", "shuffled_constants"):
        collector = trainer.ProcessRaceCollector(
            policy,normalizer,settings,stage,"cuda",workers=args.workers,
            sampling_prefix="evaluation")
        assignments = {}
        changed_batches = 0
        if arm == "shuffled_constants":
            reset_original = collector._reset_slot

            def reset(slot,index,seed_base,track):
                result = reset_original(slot,index,seed_base,track)
                actual = slot.history.array()[-1,103:152]
                for _ in range(20):
                    fake = plant_bank[rng.integers(len(plant_bank))]
                    if not np.array_equal(actual,fake):
                        break
                assignments[id(slot)] = fake
                return result

            collector._reset_slot = reset
            if collector.host_evaluation_graphs is None:
                raise RuntimeError("Expected host evaluation graphs")
            predict_original = collector.host_evaluation_graphs.predict_numpy

            def predict(histories,speeds,*,asynchronous=False):
                nonlocal changed_batches
                active = [slot for slot in collector.slots if not slot.done]
                if len(active) != len(histories):
                    raise AssertionError("Active-slot inference batch is misaligned")
                modified = histories.copy()
                for index,slot in enumerate(active):
                    modified[index,:,103:152] = assignments[id(slot)]
                changed_batches += 1
                return predict_original(modified,speeds,asynchronous=asynchronous)

            collector.host_evaluation_graphs.predict_numpy = predict
        started = time.perf_counter()
        try:
            rows = collector.evaluate_rows(
                episodes=settings["evaluation_episodes"],
                seed_base=settings["evaluation_seed"])
        finally:
            collector.close()
        metrics = selection_metrics(trainer.multitrack_metrics(rows,stage.target_gates),settings,stage)
        results[arm] = {
            "seconds": time.perf_counter()-started,
            "episodes": len(rows),
            "inference_batches_with_wrong_constants": changed_batches,
            "selection_suite_success": metrics["selection_suite_success"],
            "full_course_success": metrics["full_course_success"],
            "mean_gates": metrics["mean_gates"],
            "mean_return": metrics["mean_return"],
            "per_course_success": {
                name:metrics[f"track/{name}/full_course_success"]
                for name in settings["evaluation_suite_names"]},
        }
        print(arm,results[arm]["selection_suite_success"],results[arm]["seconds"],flush=True)
    output = {
        "checkpoint":str(resolved),"checkpoint_step":state.get("environment_steps"),
        "suite_sha256":settings["evaluation_suite_sha256"],
        "episodes_per_track":args.episodes_per_track,"seed":args.seed,
        "plant_bank_replay":str(args.replay),
        "note":"Same policy, courses, commands, and episode seeds. Only 49 static plant constants are replaced with a different plausible episode plant; two episodes/course make this a small diagnostic, not a benchmark.",
        "arms":results,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"original":results["original"]["selection_suite_success"],
                      "shuffled":results["shuffled_constants"]["selection_suite_success"]}))


if __name__ == "__main__":
    main()
