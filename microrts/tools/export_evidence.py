
"""Export the source records used in the MicroRTS report. Run from the research repo."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

AP = argparse.ArgumentParser(description=__doc__)
AP.add_argument("repo", type=Path, help="AO-OpenResearch checkout containing the original outputs")
args = AP.parse_args()
repo = args.repo
base = repo / "outputs"
campaign = base / "oracle-science-tuned-abcd-20261104"
out = Path(__file__).resolve().parents[1] / "assets/microrts-report/data"
out.mkdir(parents=True, exist_ok=True)

def load(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_csv(name, rows):
    with (out / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

design = load(campaign / "protocol/design.json")
games, losses, runs = [], [], []
for arm in "ABCD":
    for seed in (0, 1):
        run = campaign / "runs" / f"{arm}-s{seed}"
        receipt = load(run / "COMPLETE.json")
        meta = load(run / "run.json")
        evaluations = run / "evaluations.json"
        assert receipt["status"] == "complete" and receipt["step"] == 262144
        assert digest(evaluations) == receipt["files"]["evaluations.json"]
        runs.append(dict(arm=arm, seed=seed, initial_actor_sha256=meta["initial_actor_sha256"],
                         final_checkpoint_sha256=receipt["files"]["latest.pt"],
                         evaluations_sha256=digest(evaluations)))
        for evaluation in load(evaluations):
            assert len(evaluation["results"]) == evaluation["games"] == 16
            for game in evaluation["results"]:
                assert game["terminated"] != game["truncated"]
                outcome = next((x for x in ("win", "loss", "draw") if game[x]), "timeout")
                assert (outcome == "timeout") == game["truncated"]
                games.append(dict(arm=arm, seed=seed, transitions=evaluation["step"],
                                  map=Path(game["map"]).stem, opponent=game["bot"],
                                  seat=game["seat"], outcome=outcome))
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        losses.extend(dict(arm=arm, seed=seed, transitions=row["step"],
                           critic_loss=row["real_value_loss"])
                      for row in metrics if row.get("real_value_loss") is not None)

behavior_path = campaign / "analysis/offline_behavior.json"
behavior = load(behavior_path)
actions = [dict(actor=actor, **record["action_counts"])
           for actor, record in behavior["results"].items()]
historical_path = base / "tells-audit-20261015/historical-metrics.json"
historical = load(historical_path)
with_belief = next(r["metrics"] for r in historical if "xcomocpv" in r["path"])
without_belief = next(r["metrics"] for r in historical if "uz5fp5r1" in r["path"])
tells = []
for horizon in ("h1", "h8", "h32", "h128", "episode"):
    prefix = f"val/return/{horizon}/"
    matched = with_belief[prefix + "z_loss"]
    control = without_belief[prefix + "no_z_loss"]
    tells.append(dict(horizon=horizon, belief_loss=matched, state_only_loss=control,
                      relative_reduction_pct=100 * (control - matched) / control))

benchmark_path = base / "pmpo-infra-20261104/repeat_sweep.json"
benchmark = [dict(payload="structured" if row["structured_only"] else "full",
                  transitions_per_s=row["transitions_per_s"],
                  unhidden_jvm_wait_s=row["wait_s"]) for row in load(benchmark_path)]
qual_path = base / "oracle-pmpo-package-20261104/qualification.json"
qualification = load(qual_path)
reward = qualification["comparisons"][qualification["selected"]]
summary = {
    "factorial": {
        "arms": design["arms"],
        "budget": design["budget"],
        "evaluation": design["evaluation"],
        "protocol_sha256": digest(campaign / "protocol/design.json"),
        "runs": runs,
        "means_critic_loss_last_quarter": {
            arm: mean(row["critic_loss"] for row in losses
                      if row["arm"] == arm and row["transitions"] > 196608)
            for arm in "ABCD"
        },
    },
    "action_probe": {
        "games": behavior["games_per_map"] * behavior["maps"],
        "trajectories": behavior["trajectories"], "states": behavior["states"],
        "actionable_unit_opportunities": behavior["actionable_unit_opportunities"],
        "multi_choice_opportunities": behavior["multi_choice_opportunities"],
        "source_sha256": digest(behavior_path),
        "seat_1_no_op": {actor: row["by_seat"]["1"]["action_shares"]["none"]
                         for actor, row in behavior["results"].items()},
    },
    "reward_probe": {
        "reward_mae": reward["reward_mae"], "labels": int(reward["labels"]),
        "terminal_reward_mae": reward["terminal_mae"],
        "terminal_labels": int(reward["terminal_labels"]),
        "windows": qualification["panel_windows"],
        "world_model_sha256": qualification["package_sha256"],
        "source_sha256": digest(qual_path),
    },
    "throughput": {
        "learner_pairs": 128, "cpu_threads": 2, "ticks_per_run": 80,
        "source_sha256": digest(benchmark_path),
    },
    "tells": {
        "with_belief_run": "xcomocpv", "state_only_run": "uz5fp5r1",
        "summary_source_sha256": digest(historical_path),
    },
}
assert len(games) == 640 and len(runs) == 8
assert len({row["initial_actor_sha256"] for row in runs}) == 1
assert all(sum(row[key] for key in row if key != "actor") == 1335 for row in actions)
for name, rows in [("match-outcomes.csv", games), ("critic-loss.csv", losses),
                   ("action-counts.csv", actions), ("tells-return-loss.csv", tells),
                   ("throughput-repeats.csv", benchmark)]:
    write_csv(name, rows)
(out / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
