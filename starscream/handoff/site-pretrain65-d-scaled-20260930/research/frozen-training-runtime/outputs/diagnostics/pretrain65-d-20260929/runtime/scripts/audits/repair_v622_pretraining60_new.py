"""Replace bridge rows that failed MPCC admission with simpler realizations."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint, save_track_yaml
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex, generate, requirement_cells, validate_geometry
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track
from scripts.audits.review_v622_geometry import resampled
from scripts.audits.build_v622_pretraining50 import _cell_axes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/v622-pretraining60"


def resolve(path):
    p = Path(path)
    return ROOT / p.relative_to("/workspace") if str(p).startswith("/workspace/") else p


def main():
    mp = OUT / "manifest.json"
    manifest = json.loads(mp.read_text())
    failed = json.loads((OUT / "qualification-progress-new.json").read_text())["failed"]
    if len(failed) != 3:
        raise ValueError(f"expected three failed bridge rows, found {failed}")
    base = json.loads((ROOT / "outputs/v622-pretraining50/mpcc-frontier-v3/manifest.json").read_text())
    protected_paths = [resolve(r["path"]) for r in base["records"]]
    for p in (ROOT / "configs/eval/v6_22_real60.manifest.json", ROOT / "configs/eval/v6_22_real100_hard_v2.manifest.json"):
        protected_paths += [resolve(r["path"]) for r in json.loads(p.read_text())["records"]]
    exact = CloneIndex([load_track(p) for p in protected_paths])
    continuous = CloneIndex([resampled(load_track(p)) for p in protected_paths])
    for r in manifest["records"]:
        if r["name"] not in failed:
            t = load_track(resolve(r["path"]))
            exact.add(t); continuous.add(resampled(t))
    replacements = []
    for ordinal, old_name in enumerate(failed):
        old = next(r for r in manifest["records"] if r["name"] == old_name)
        label, family = old["label"], old["family"]
        found = None
        for trial in range(240):
            seed = 87200000 + ordinal * 100003 + trial
            rng = np.random.default_rng(seed)
            # Shorter realizations reduce pathological recovery chains while
            # retaining the missing behavior primitive.
            count = int(rng.integers(8, 10))
            name = f"v622_pretrain60_repair_{label}_{ordinal:02d}_{trial:03d}"
            t = generate(family, count, seed, name, hard=False)
            if validate_geometry(t):
                continue
            ex, cont = exact.distance(t), continuous.distance(resampled(t))
            if ex < .12 or cont < .06:
                continue
            found = (t, seed, cont); break
        if found is None:
            raise RuntimeError(f"no independent repair for {old_name}")
        t, seed, cont = found
        path = save_track_yaml(t, OUT / "tracks" / f"{t.name}.yaml")
        row = dict(old)
        row.update({"name": t.name, "path": "/workspace/" + str(path.relative_to(ROOT)),
                    "source": "v622-pretraining-behavior-program-v2-repair",
                    "seed": int(seed), "geometry_fingerprint": geometry_fingerprint(t),
                    "track_fingerprint": t.fingerprint, "fingerprint": geometry_fingerprint(t),
                    "cells": sorted(requirement_cells(t)), "coarse_cells": sorted(_cell_axes(t)),
                    "geometry": geometry_record(t),
                    "behavior_signature": dict((t.metadata or {}).get("behavior_signature", {})),
                    "minimum_protected_clone_distance": float(cont),
                    "qualified_speed_mps": None, "qualification": None,
                    "mpcc_admission": {"contract": "starscream-mpcc-admission-v2",
                                       "rl_eligible": False, "dagger_eligible": False}})
        replacements.append(row); exact.add(t); continuous.add(resampled(t))
        print("REPAIR", old_name, "->", row["name"], flush=True)
    out = []
    repl = {old: new for old, new in zip(failed, replacements)}
    for r in manifest["records"]:
        out.append(repl[r["name"]] if r["name"] in repl else r)
    out.sort(key=lambda r: int(r.get("rank", 0)))
    manifest["records"] = out
    manifest["admission"] = {"status": "new_bridge_qualification_pending",
                              "repaired": len(replacements), "reused_pretrain50": 50}
    atomic_json(mp, manifest)
    atomic_json(OUT / "repair-ledger.json", {"replaced": failed,
                                              "replacements": [r["name"] for r in replacements]})


if __name__ == "__main__": main()
