"""Splits trajectories.jsonl into one scene file per rollout and computes behaviour metrics.

  python build.py      (stdlib only)

Outputs next to this file:
  <slug>.json   scene for scene3d.js; recovery legs carry trajectory.highlight + gates[k].mark
  index.json    per-rollout metrics used by the page

A recovery leg is a gate-to-gate leg, inside a rollout tagged recovery*, that flies at least
RECOVERY_EXTRA m further than the straight line and either takes RECOVERY_FACTOR x that rollout's
median leg time or has a detour of at least RECOVERY_DETOUR. Detour = leg path length / straight-line distance.
"""
import json, math, os, re, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DECIMATE = 2            # 130 Hz -> 65 Hz is plenty for drawing
RECOVERY_FACTOR = 2.0
RECOVERY_DETOUR = 3.0
RECOVERY_EXTRA = 5.0      # m
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
    if track.startswith("multigp_"):
        return "MultiGP " + track.split("_")[-1].capitalize()
    m = re.match(r"v622_hard_(?:v2b_)?(.+?)_(\d+)_(\d+)", track)
    return f"{NAMES.get(m.group(1), m.group(1).replace('_', ' ').capitalize())} {m.group(2)}-{m.group(3)}"


def path_len(p, a, b):
    return sum(math.dist(p[i], p[i + 1]) for i in range(a, b))


def analyse(row):
    track, tag = [s.strip() for s in row["name"].split(" - ")]
    t = row["trajectory"]
    p, q, passed, dt = t["pos"], t["quat"], t["passed"], t["dt"]
    n = len(p)

    raw = [math.dist(p[i], p[i + 1]) / dt for i in range(n - 1)]
    h = SMOOTH // 2
    speed = [statistics.fmean(raw[max(0, i - h):i + h + 1]) for i in range(len(raw))]
    tilt = [math.degrees(math.acos(max(-1, min(1, 1 - 2 * (x * x + y * y))))) for _, x, y, _ in q]

    events = [i for i in range(1, n) if passed[i] != passed[i - 1]]
    starts = [0] + events[:-1]
    legs = []
    for k, (a, b) in enumerate(zip(starts, events)):
        L = path_len(p, a, b)
        legs.append({"gate": k + 1, "a": a, "b": b, "time": (b - a) * dt, "path": L,
                     "detour": L / max(math.dist(p[a], p[b]), 1e-3),
                     "min_speed": min(speed[a:b] or [0]),
                     "offset": math.dist(p[b], row["gates"][k]["pos"])})

    median = statistics.median(l["time"] for l in legs)
    recovering = tag.startswith("recovery")
    rec = [l for l in legs if recovering and l["path"] - math.dist(p[l["a"]], p[l["b"]]) >= RECOVERY_EXTRA
           and (l["time"] >= RECOVERY_FACTOR * median or l["detour"] >= RECOVERY_DETOUR)]
    T = (n - 1) * dt
    return track, tag, {
        "slug": re.sub(r"[^a-z0-9]+", "-", f"{track}-{tag}".lower()).strip("-"),
        "name": pretty(track), "track": track, "tag": tag,
        "gates": len(row["gates"]), "time": round(T, 2), "path": round(path_len(p, 0, n - 1), 1),
        "mean_speed": round(statistics.fmean(raw), 2), "peak_speed": round(max(speed), 2),
        "peak_tilt": round(max(tilt), 1), "mean_offset": round(statistics.fmean(l["offset"] for l in legs), 2),
        "legs": [{"gate": l["gate"], "time": round(l["time"], 3), "detour": round(l["detour"], 3),
                  "recovery": l in rec} for l in legs],
        "recoveries": [{"gate": l["gate"], "time": round(l["time"], 2), "detour": round(l["detour"], 2),
                        "extra_path": round(l["path"] - math.dist(p[l["a"]], p[l["b"]]), 1),
                        "min_speed": round(l["min_speed"], 2), "a": l["a"], "b": l["b"]} for l in rec],
        "recovery_share": round(sum(l["time"] for l in rec) / T, 3),
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


index = []
with open(os.path.join(HERE, "trajectories.jsonl")) as fh:
    for line in fh:
        row = json.loads(line)
        _, _, meta = analyse(row)
        with open(os.path.join(HERE, meta["slug"] + ".json"), "w") as out:
            json.dump(scene(row, meta), out, separators=(",", ":"))
        for r in meta["recoveries"]:
            del r["a"], r["b"]
        index.append(meta)
        print(f"{meta['tag']:<18} {meta['name']:<32} {meta['time']:6.2f}s  recoveries={[(r['gate'], r['time']) for r in meta['recoveries']]}")

with open(os.path.join(HERE, "index.json"), "w") as fh:
    json.dump(index, fh, indent=1)
