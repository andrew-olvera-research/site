"""Splits the handoff's trajectories.jsonl into one scene file per rollout and computes behaviour metrics.

  python build.py      (stdlib only)

Inputs (from the scaled-D handoff under ../../handoff/):
  generated/trajectories.jsonl   frontend trajectories (metres, seconds, quat w,x,y,z), one per gallery asset
  generated/manifest.json        gallery metadata, one entry per asset, same order
  evals/raw/.../real100-v2-e32.json  the frozen benchmark; each gallery asset is one of its episodes

Outputs next to this file:
  <slug>.json   scene for scene3d.js; recovery legs carry trajectory.highlight + gates[k].mark
  index.json    per-rollout metrics used by the page

Misses come from the benchmark's own ordered-reference event list (matched by slot and seed, and
checked against the replay's step count), not from a geometric filter on the states. A recovery leg
runs from a miss of gate k to the pass of gate k. Detour = leg path length / straight-line distance.
"""
import json, math, os, re, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "..", "handoff", "site-pretrain65-d-scaled-20260930")
RAW = os.path.join(PKG, "evals/raw/outputs/evals/pretrain65-d-scaled-final-20260929/real100-v2-e32.json")
DECIMATE = 2            # 130 Hz -> 65 Hz is plenty for drawing
SMOOTH = 7              # frames for speed smoothing

NAMES = {
    "vertical_chain": "Vertical chain", "compound_reversal": "Compound reversal",
    "diving_hairpin": "Diving hairpin", "slalom": "Slalom", "long_braking": "Long braking",
    "radius_switch": "Changing-radius turns", "ordered_3d": "Ordered 3D",
    "wrong_side_incidence": "Oblique gate approaches", "flow": "Flow", "long_low": "Long low",
    "go_around": "Go-around", "long_low_braking": "Long low braking",
    "stacked_reversal": "Stacked reversal", "hairpin_chain": "Hairpin chain",
}


def pretty(track):
    if track.startswith("a2rl_"):
        return "A2RL S2 2026"
    if track.startswith("multigp_"):
        return "MultiGP " + track.split("_")[-1].capitalize()
    m = re.match(r"v622_hard_(?:v2b_)?(.+?)_(\d+)_(\d+)", track)
    return f"{NAMES.get(m.group(1), m.group(1).replace('_', ' ').capitalize())} {m.group(2)}-{m.group(3)}"


def path_len(p, a, b):
    return sum(math.dist(p[i], p[i + 1]) for i in range(a, b))


def analyse(row, meta, episode):
    track, tag = [s.strip() for s in row["name"].split(" - ")]
    t = row["trajectory"]
    p, q, dt = t["pos"], t["quat"], t["dt"]
    n = len(p)
    assert n == episode["steps"] + 1, (track, n, episode["steps"])

    raw = [math.dist(p[i], p[i + 1]) / dt for i in range(n - 1)]
    h = SMOOTH // 2
    speed = [statistics.fmean(raw[max(0, i - h):i + h + 1]) for i in range(len(raw))]
    tilt = [math.degrees(math.acos(max(-1, min(1, 1 - 2 * (x * x + y * y))))) for _, x, y, _ in q]

    passes = {e["gate"]: e["step"] for e in episode["events"] if e["kind"] == "pass"}
    misses = {}
    for e in episode["events"]:
        if e["kind"] == "miss":
            misses.setdefault(e["gate"], []).append(e["step"])

    legs, a = [], 0
    for k in range(len(row["gates"])):
        b = passes.get(k)
        end = b if b is not None else n - 1          # the failed leg runs to the crash
        L = path_len(p, a, end)
        m = misses.get(k, [])
        legs.append({"gate": k + 1, "a": a, "b": end, "passed": b is not None,
                     "time": (end - a) * dt, "path": L,
                     "detour": L / max(math.dist(p[a], p[end]), 1e-3),
                     "min_speed": min(speed[a:end] or [0]), "misses": m})
        if b is None:
            break
        a = b

    rec = [l for l in legs if l["misses"]]
    T = (n - 1) * dt
    return {
        "slug": re.sub(r"[^a-z0-9]+", "-", f"{track}-{tag}".lower()).strip("-"),
        "name": pretty(track), "track": track, "tag": tag, "family": meta["family"],
        "slot": meta["benchmark_slot"], "seed": meta["seed"],
        "gates": len(row["gates"]), "passed": meta["passed_gates"],
        "completed": meta["completed"], "timely": meta["timely_success"], "clean": meta["clean_success"],
        "course_sr": meta["benchmark_success_rate"],
        "time": round(T, 2), "path": round(path_len(p, 0, n - 1), 1),
        "mean_speed": round(statistics.fmean(raw), 2), "peak_speed": round(max(speed), 2),
        "peak_tilt": round(max(tilt), 1),
        "mp4": meta["mp4"], "poster": meta["poster"],
        "legs": [{"gate": l["gate"], "time": round(l["time"], 3), "detour": round(l["detour"], 3),
                  "min_speed": round(l["min_speed"], 2), "recovery": bool(l["misses"]),
                  "passed": l["passed"]} for l in legs],
        "recoveries": [{"gate": l["gate"], "passed": l["passed"],
                        "miss_to_pass": round(((l["b"]) - l["misses"][0]) * dt, 2),
                        "time": round(l["time"], 2), "detour": round(l["detour"], 2),
                        "extra_path": round(l["path"] - math.dist(p[l["a"]], p[l["b"]]), 1),
                        "min_speed": round(l["min_speed"], 2),
                        "a": l["misses"][0], "b": l["b"]} for l in rec],
    }


def scene(row, meta):
    t = row["trajectory"]
    idx = list(range(0, len(t["pos"]), DECIMATE))
    if idx[-1] != len(t["pos"]) - 1:
        idx.append(len(t["pos"]) - 1)
    marked = {r["gate"] - 1 for r in meta["recoveries"]}
    gates = [{**g, **({"mark": True} if k in marked else {})} for k, g in enumerate(row["gates"])]
    return {
        "name": meta["name"], "gates": gates,
        "trajectory": {
            "dt": t["dt"] * DECIMATE,
            "pos": [[round(v, 3) for v in t["pos"][i]] for i in idx],
            "quat": [[round(v, 4) for v in t["quat"][i]] for i in idx],
            "passed": [t["passed"][i] for i in idx],
            "highlight": [[r["a"] // DECIMATE, -(-r["b"] // DECIMATE)] for r in meta["recoveries"]],
        },
    }


episodes = {(e["slot"], e["seed"]): e for e in json.load(open(RAW))["episodes"]}
assets = json.load(open(os.path.join(PKG, "generated", "manifest.json")))["assets"]
index = []
with open(os.path.join(PKG, "generated", "trajectories.jsonl")) as fh:
    for row, meta in zip(map(json.loads, fh), assets):
        assert row["name"].startswith(meta["name"])
        info = analyse(row, meta, episodes[(meta["benchmark_slot"], meta["seed"])])
        with open(os.path.join(HERE, info["slug"] + ".json"), "w") as out:
            json.dump(scene(row, info), out, separators=(",", ":"))
        for r in info["recoveries"]:
            del r["a"], r["b"]
        index.append(info)
        print(f"{info['tag']:<14} {info['name']:<30} {info['time']:6.2f}s  "
              f"recoveries={[(r['gate'], r['miss_to_pass']) for r in info['recoveries']]}")

with open(os.path.join(HERE, "index.json"), "w") as fh:
    json.dump(index, fh, indent=1)
