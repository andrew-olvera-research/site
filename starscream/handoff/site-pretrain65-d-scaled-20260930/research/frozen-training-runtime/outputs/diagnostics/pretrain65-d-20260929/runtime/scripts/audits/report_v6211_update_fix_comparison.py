#!/usr/bin/env python3
"""Build the v6.21 / v6.21.1 update-fix / v6.22.2 benchmark comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HZ = 130.0


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text())
    metrics = data["metrics"]
    episodes = int(metrics["episodes"])
    completion = float(metrics["full_course_success"])
    track_prefix = "track/"
    suffix = "/successful_minimum_steps"
    fastest_steps = [
        float(value)
        for key, value in metrics.items()
        if key.startswith(track_prefix)
        and key.endswith(suffix)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ]
    return {
        "source": str(path.relative_to(ROOT)),
        "checkpoint": data["checkpoint"],
        "episodes": episodes,
        "completed_laps": int(round(completion * episodes)),
        "total_score": completion,
        "fastest_lap_steps": int(min(fastest_steps)) if fastest_steps else None,
        "fastest_lap_seconds": min(fastest_steps) / HZ if fastest_steps else None,
        "performance_weighted_success_hz90": metrics.get("performance_weighted_success_hz90"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="outputs/evals/v6211-update-fix-final")
    args = parser.parse_args()
    out = ROOT / args.eval_dir
    inputs = {
        "real60": {
            "v6.21": ROOT / "outputs/evals/v6222-crosscompare/v621-v6_22_real60-e8.json",
            "v6.21.1 update-fix": out / "update-fix-real60-e8.json",
            "v6.22.2": ROOT / "outputs/evals/v6222-crosscompare/v6222-v6_22_real60-e8.json",
        },
        "real100-hard-v2": {
            "v6.21": ROOT / "outputs/evals/v6222-crosscompare/v621-v6_22_real100_hard_v2-e8.json",
            "v6.21.1 update-fix": out / "update-fix-real100-hard-v2-e8.json",
            "v6.22.2": ROOT / "outputs/evals/v6222-crosscompare/v6222-v6_22_real100_hard_v2-e8.json",
        },
    }
    summary = {suite: {model: summarize(path) for model, path in models.items()} for suite, models in inputs.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# v6.21.1 update-fix final benchmark comparison",
        "",
        "Matched randomized evaluation: 8 episodes/course, seed 2034091462, 6,000-step cap, 130 Hz.",
        "Existing v6.21 and v6.22.2 results were reused; only update-fix was evaluated.",
        "",
        "| Suite | Model | Total score | Completed laps | Fastest lap |",
        "|---|---|---:|---:|---:|",
    ]
    for suite, models in summary.items():
        for model, row in models.items():
            fastest = "n/a" if row["fastest_lap_seconds"] is None else f'{row["fastest_lap_seconds"]:.3f} s'
            lines.append(f'| {suite} | {model} | {row["total_score"]:.1%} | {row["completed_laps"]}/{row["episodes"]} | {fastest} |')
    (out / "comparison.md").write_text("\n".join(lines) + "\n")

    import matplotlib.pyplot as plt

    models = ["v6.21", "v6.21.1 update-fix", "v6.22.2"]
    colors = ["#718096", "#2B6CB0", "#805AD5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    width = 0.24
    x = range(2)
    suites = ["real60", "real100-hard-v2"]
    for index, model in enumerate(models):
        positions = [value + (index - 1) * width for value in x]
        scores = [summary[suite][model]["total_score"] * 100 for suite in suites]
        bars = axes[0].bar(positions, scores, width, label=model, color=colors[index])
        axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        laps = [summary[suite][model]["fastest_lap_seconds"] for suite in suites]
        bars = axes[1].bar(positions, laps, width, label=model, color=colors[index])
        axes[1].bar_label(bars, fmt="%.2fs", padding=3, fontsize=9)
    for axis in axes:
        axis.set_xticks(list(x), ["real60", "100-v2 hard"])
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Total score (completion rate)")
    axes[0].set_ylabel("Completed laps (%) — higher is better")
    axes[1].set_title("Fastest successful lap")
    axes[1].set_ylabel("Seconds — lower is better")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Starscream benchmark comparison · matched 8-episode protocol", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
