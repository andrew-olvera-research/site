
"""Render portfolio figures from the bundled data. Requires matplotlib."""
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ASSETS = Path(__file__).resolve().parents[1] / "assets/microrts-report"
DATA = ASSETS / "data"
FIGURES = ASSETS / "figures"
FIGURES.mkdir(exist_ok=True)
ARMS = ("A", "B", "C", "D")
COLORS = dict(zip(ARMS, ("#2364a3", "#007867", "#9a4a14", "#7655a6")))
INK, MUTED, GRID = "#182838", "#485b6c", "#dce4eb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

def rows(name):
    with (DATA / name).open() as file:
        return list(csv.DictReader(file))

def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=175, bbox_inches="tight",
                pad_inches=0.23, facecolor="white")
    plt.close(fig)

games = rows("match-outcomes.csv")
meta = json.loads((DATA / "provenance.json").read_text())
checkpoints = sorted({int(row["transitions"]) for row in games})
wins = {arm: [sum(row["arm"] == arm and int(row["transitions"]) == step
                  and row["outcome"] == "win" for row in games)
              for step in checkpoints] for arm in ARMS}
assert wins == {"A": [6, 6, 5, 4, 5], "B": [6, 5, 3, 5, 3],
                "C": [6, 6, 4, 4, 1], "D": [7, 6, 2, 3, 5]}
labels = ["A · Fog", "B · Full state", "C · Fog + imagination", "D · Full state + imagination"]
fig, (left, right) = plt.subplots(2, 1, figsize=(7.4, 7.5), layout="constrained")
left.set_title("Critic fit during the final quarter", loc="left", pad=12)
left.barh(ARMS[::-1], [meta["factorial"]["means_critic_loss_last_quarter"][arm]
                         for arm in ARMS[::-1]], color=[COLORS[arm] for arm in ARMS[::-1]],
          height=.48, zorder=3)
for i, arm in enumerate(ARMS[::-1]):
    value = meta["factorial"]["means_critic_loss_last_quarter"][arm]
    left.text(value + .06, i, f"{value:.2f}", va="center", fontweight="bold")
left.set(xlim=(0, 3.25), xlabel="Mean bootstrapped value loss")
left.grid(axis="x", color=GRID)
left.tick_params(axis="y", length=0)
right.set_title("Evaluation against scripted opponents", loc="left", pad=12)
for arm, label, marker in zip(ARMS, labels, ("o", "s", "^", "D")):
    right.plot(range(5), wins[arm], label=label, color=COLORS[arm],
               marker=marker, markeredgecolor="white", markersize=8, lw=2.1)
right.set(xticks=range(5), xticklabels=["0", "65k", "131k", "197k", "262k"],
          yticks=[0, 2, 4, 6, 8], ylim=(0, 8.1),
          ylabel="Wins per 32 evaluation games", xlabel="Real learner transitions per run")
right.grid(axis="y", color=GRID)
right.legend(loc="upper center", bbox_to_anchor=(.5, -.27), ncols=2, frameon=False, fontsize=10)
save(fig, "critic-and-control")

counts = {row["actor"]: row for row in rows("action-counts.csv")}
denominator = meta["action_probe"]["actionable_unit_opportunities"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True, layout="constrained")
for ax, action, title in zip(axes, ("move", "none"), ("Move", "No action")):
    initial = int(counts["initial"][action]) / denominator
    ax.axvline(initial, color=MUTED, linestyle="--", linewidth=1.3, zorder=1)
    for i, arm in enumerate(ARMS):
        final = mean(int(counts[f"{arm}-s{seed}"][action]) for seed in (0, 1)) / denominator
        ax.plot([initial, final], [i, i], color=COLORS[arm], lw=2.5)
        ax.scatter(final, i, s=57, color=COLORS[arm], zorder=3)
        leftward = final < initial
        ax.text(final + (-.009 if leftward else .009), i, f"{final:.1%}",
                ha="right" if leftward else "left", va="center", fontsize=10)
    ax.set(title=title, yticks=range(4), yticklabels=ARMS, ylim=(3.55, -.68),
           xlim=(.255, .474), xticks=[.3, .35, .4, .45])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", color=GRID)
    ax.tick_params(axis="y", length=0)
    ax.text(initial, -.49, f"{initial:.1%}", ha="center", va="center", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
fig.supxlabel("Share of actionable unit choices · dashed line marks the initial actor",
              fontsize=11, color=MUTED)
save(fig, "action-shift")

tells = rows("tells-return-loss.csv")
values = [float(row["relative_reduction_pct"]) for row in tells]
fig, ax = plt.subplots(figsize=(7.4, 3.45), layout="constrained")
positions = range(len(tells))
ax.barh(positions, values, color=["#9a4a14" if x < 0 else "#007867" for x in values],
        height=.49, zorder=3)
for i, value in enumerate(values):
    ax.text(value + (.14 if value > 0 else -.14), i, f"{value:+.1f}%",
            va="center", ha="left" if value > 0 else "right", fontweight="bold", fontsize=10)
ax.axvline(0, color=MUTED, lw=1, zorder=2)
ax.set(yticks=positions, yticklabels=["1 step", "8 steps", "32 steps", "128 steps", "Episode"],
       ylim=(4.48, -.48), xlim=(-2.6, 7.2),
       xlabel="Validation loss reduction vs. state-only head")
ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.grid(axis="x", color=GRID)
ax.tick_params(axis="y", length=0)
save(fig, "return-prediction")
