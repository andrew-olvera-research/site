"""Compiles every number the Starscream page charts into results.json.

  python build.py      (stdlib only)

All values are read from the scaled-D handoff under ../../handoff/ (rates are fractions, times seconds).
Nothing is typed in by hand; the page's figures are drawn from this file.
"""
import csv, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "..", "handoff", "site-pretrain65-d-scaled-20260930")
BEH = os.path.join(PKG, "evals/behavior/outputs/diagnostics/pretrain65-d-behavior-20260929")
RAW = os.path.join(PKG, "evals/raw/outputs/evals/pretrain65-d-scaled-final-20260929")
OLD = os.path.join(PKG, "evals/historical/outputs/evals/v6211-update-fix-final")
load = lambda *p: json.load(open(os.path.join(*p)))
r4 = lambda v: None if v is None else round(v, 4)

front = load(PKG, "evals/frontend-data.json")
r100 = front["real100_v2"]
full = load(BEH, "full-analysis.json")
laps = load(PKG, "evals/lap-distributions.json")
hist = load(PKG, "evals/historical-comparison.json")

# ---- learning curves on the frozen selection25 panel (family-weighted) ----
sel = load(BEH, "selection-history.json")
snap = load(PKG, "research/sources/outputs/diagnostics/pretrain65-d-20260929/learning-analysis-snapshot.json")
learning = {
    "scaled_d": [[r["step"], r4(r["selection_suite_success"]), r4(r["selection_suite_timely_success"]),
                  r4(r["selection_suite_clean_success"]), r4(r["selection_suite_clean_timely_success"])] for r in sel],
    "earlier": [[0, 0.0]] + [[r["step"], r4(r["selection_suite_success"])] for r in snap["plant"] if "selection_suite_success" in r],
    "selected_step": front["checkpoint_steps"], "selected_round": front["checkpoint_round"],
}

# ---- old (v6.21.1 update-fix, 8 starts) per-track eventual ----
def old_tracks(fn):
    m = load(OLD, fn)["metrics"]
    return {k.split("/")[1]: m[k] for k in m if re.fullmatch(r"track/[^/]+/full_course_success", k)}
old100, old60 = old_tracks("update-fix-real100-hard-v2-e8.json"), old_tracks("update-fix-real60-e8.json")

rows = list(csv.DictReader(open(os.path.join(BEH, "courses.csv"))))
courses = []
for c in r100["courses"]:
    courses.append({"slot": c["slot"], "name": c["name"], "family": c["family"],
                    "eventual": r4(c["eventual_sr"]), "timely": r4(c["timely_sr"]),
                    "clean_timely": r4(c["clean_timely_sr"]), "old": r4(old100.get(c["name"])),
                    "median_s": r4(c["success_lap_seconds"].get("median")),
                    "reference_s": r4(c["reference_seconds"])})
assert sum(c["old"] is None for c in courses) == 0, [c["name"] for c in courses if c["old"] is None]

families = []
for f, v in r100["families"].items():
    cs = [c for c in courses if c["family"] == f]
    families.append({"family": f, "courses": len(cs), "episodes": v["episodes"],
                     "eventual": r4(v["eventual_sr"]), "timely": r4(v["timely_sr"]),
                     "clean_timely": r4(v["clean_timely_sr"]),
                     "old": r4(sum(c["old"] for c in cs) / len(cs)),
                     "median_s": r4(v["success_lap_seconds"].get("median"))})

# ---- real60: eventual only (no frozen timed protocol) ----
n60 = load(RAW, "real60-e32.json")["metrics"]
new60 = {k.split("/")[1]: n60[k] for k in n60 if re.fullmatch(r"track/[^/]+/full_course_success", k)}
real60 = [{"name": k, "eventual": r4(v), "old": r4(old60.get(k))} for k, v in sorted(new60.items())]

speed = [{k: (float(v) if k != "command" else float(v)) for k, v in r.items()}
         for r in csv.DictReader(open(os.path.join(BEH, "speed-summary.csv")))]
speed.sort(key=lambda r: r["command"])

agg = r100["aggregate"]
out = {
    "checkpoint": {"round": front["checkpoint_round"], "steps": front["checkpoint_steps"]},
    "suites": [{"suite": h["suite"], "old": r4(h["old_eventual_sr"]), "new": r4(h["new_eventual_sr"]),
                "old_starts": h["old_starts_per_course"], "new_starts": h["new_starts_per_course"],
                "new_total": h["new_total_starts"]} for h in hist],
    "real100": {"episodes": agg["episodes"], "completed": agg["completed"],
                "eventual": agg["eventual_sr"], "timely": agg["timely_sr"], "clean": agg["clean_sr"],
                "clean_timely": agg["clean_timely_sr"], "crash": agg["crash_sr"],
                "outcomes": r100["outcome_counts"], "termination": full["aggregate"]["termination"],
                "saturation": full["saturation"]},
    "real60": {"eventual": front["real60"]["eventual_sr"], "starts": front["real60"]["total_starts"],
               "courses": real60},
    "families": families, "courses": courses, "learning": learning,
    "laps": {"hist": laps["successful_lap_histogram"], "ratio_hist": laps["successful_reference_ratio_histogram"],
             "all": laps["all_success"], "ratio": laps["successful_reference_ratio"], "by_outcome": laps["by_outcome"]},
    "tails": {"recovery_over_s": full["aggregate"]["successes_with_recovery_over_s"],
              "no_pass_over_s": full["aggregate"]["successes_with_no_pass_over_s"],
              "multiple_recovered_gates": full["successes_multiple_recovered_gates"],
              "repeat_miss": full["successes_repeat_miss_recovery"]},
    "speed": speed,
}
json.dump(out, open(os.path.join(HERE, "results.json"), "w"), separators=(",", ":"))
print("families", [(f["family"], f["courses"], f["old"], f["eventual"]) for f in families])
print("real60 matched old", sum(c["old"] is not None for c in real60), "of", len(real60))
print("check old100 mean", sum(c["old"] for c in courses) / 100, "old60 mean", sum(c["old"] for c in real60 if c["old"] is not None) / 60)
print(os.path.getsize(os.path.join(HERE, "results.json")), "bytes")
