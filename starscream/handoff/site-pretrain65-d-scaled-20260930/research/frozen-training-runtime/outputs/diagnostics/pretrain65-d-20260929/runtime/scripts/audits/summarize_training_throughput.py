"""Read local event logs without loading models or touching running jobs.

Example: python scripts/audits/summarize_training_throughput.py EVENTS.jsonl
  --actor-batch-size 8192 --output outputs/audits/throughput.json
Phase timers can overlap or exclude startup/checkpoint work; see their call sites.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics


def summarize(path, actor_batch_size=None):
    rows = collections.defaultdict(dict)
    with Path(path).open() as stream:
        for line in stream:
            event = json.loads(line)
            rows[event["step"]].update(event.get("metrics", {}))
    selected = {
        "rollout_collection_seconds", "rollout_update_seconds", "collection_seconds",
        "round_seconds", "updates_seconds", "evaluation_seconds", "replay_prepare_seconds",
        "sampling_plan_seconds", "actor_epochs_completed", "actor_updates_completed",
        "environment_steps_per_second", "new_labels", "memory_available_gib",
        "swap_used_gib", "cgroup_memory_gib", "window_rollout_buffer_bytes",
        "critic_prefit_gae_target_explained_variance", "critic_explained_variance",
        "window_critic_mc_prefit_explained_variance",
    }
    keys = sorted({key for row in rows.values() for key in row
                   if key in {"train/" + name for name in selected}
                   or key.startswith(("train/performance/", "train/collection_last_call_host/"))})
    metrics = {}
    for key in keys:
        values = [row[key] for row in rows.values()
                  if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
        if values:
            metrics[key] = dict(n=len(values), total=sum(values), mean=statistics.mean(values),
                                median=statistics.median(values), minimum=min(values), maximum=max(values))
    result = dict(source=str(path), distinct_steps=len(rows), maximum_step=max(rows), metrics=metrics)
    actor_rows = [row for row in rows.values() if row.get("train/actor_epochs_completed", 0) > 0]
    if actor_rows:
        batches = [row["train/actor_updates_completed"] / row["train/actor_epochs_completed"]
                   for row in actor_rows]
        result["actor_batches_per_epoch"] = dict(mean=statistics.mean(batches),
            median=statistics.median(batches), minimum=min(batches), maximum=max(batches))
        if actor_batch_size is not None:
            draws = sum(row["train/actor_updates_completed"] * actor_batch_size for row in actor_rows)
            single_pass_draws = sum(row["train/actor_epochs_completed"] * row["train/rollout_transitions"]
                                    for row in actor_rows)
            result["actor_sample_presentations"] = dict(batch_size=actor_batch_size,
                actual=draws, single_pass_epoch_equivalent=single_pass_draws,
                expansion=draws / single_pass_draws)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--actor-batch-size", type=int,
                        help="Verified run batch size; assumes full-size stratified batches.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.events, args.actor_batch_size)
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(args.output)
    else:
        print(output, end="")
