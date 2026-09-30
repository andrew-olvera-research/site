"""Find an MPCC-admitted go-around bridge for the train60 panel."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
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
from scripts.audits.qualify_v622_benchmark_protocol import job, protocol_contract

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/v622-pretraining60"
FAILED = "v622_pretrain60_repair_go_around_bridge_01_000"


def resolve(path):
    p = Path(path)
    return ROOT / p.relative_to("/workspace") if str(p).startswith("/workspace/") else p


def make_row(track, seed, old):
    path = save_track_yaml(track, OUT / "tracks" / f"{track.name}.yaml")
    row = dict(old)
    row.update({
        "name": track.name, "path": "/workspace/" + str(path.relative_to(ROOT)),
        "source": "v622-pretraining-behavior-program-v2-go-around-repair",
        "seed": int(seed), "geometry_fingerprint": geometry_fingerprint(track),
        "track_fingerprint": track.fingerprint, "fingerprint": geometry_fingerprint(track),
        "cells": sorted(requirement_cells(track)), "coarse_cells": sorted(_cell_axes(track)),
        "geometry": geometry_record(track),
        "behavior_signature": dict((track.metadata or {}).get("behavior_signature", {})),
        "minimum_protected_clone_distance": None, "qualified_speed_mps": None,
        "qualification": None,
        "mpcc_admission": {"contract": "starscream-mpcc-admission-v2", "rl_eligible": False,
                           "dagger_eligible": False},
    })
    return row


def main():
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    old = next(r for r in manifest["records"] if r["name"] == FAILED)
    protected_paths = [resolve(r["path"]) for r in manifest["records"] if r["name"] != FAILED]
    for mpth in (ROOT / "configs/eval/v6_22_real60.manifest.json",
                 ROOT / "configs/eval/v6_22_real100_hard_v2.manifest.json"):
        protected_paths += [resolve(r["path"]) for r in json.loads(mpth.read_text())["records"]]
    exact = CloneIndex([load_track(p) for p in protected_paths])
    continuous = CloneIndex([resampled(load_track(p)) for p in protected_paths])
    candidates = []
    for trial in range(80):
        seed = 88310000 + trial
        rng = np.random.default_rng(seed)
        count = int(rng.integers(7, 10))
        name = f"v622_pretrain60_go_around_bridge_repair_{trial:03d}"
        track = generate("go_around", count, seed, name, hard=False)
        if validate_geometry(track):
            continue
        ex, cont = exact.distance(track), continuous.distance(resampled(track))
        if ex < .12 or cont < .06:
            continue
        # Prefer shorter, wider and less alternating realizations: this keeps
        # the go-around primitive while avoiding recovery-heavy chains.
        g = geometry_record(track)
        score = (2.0 * cont + 0.01 * len(_cell_axes(track))
                 - 0.04 * g["alternating_pairs"] - 0.01 * len(track.gates))
        candidates.append((score, track, seed, cont))
    if not candidates:
        raise RuntimeError("no independent go-around candidates")
    candidates.sort(key=lambda x: (-x[0], x[1].name))
    cfg, contract, payload = protocol_contract()
    args = [(make_row(t, s, old), cfg, contract, payload["base"])
            for _, t, s, _ in candidates[:24]]
    print("testing", len(args), "candidates", flush=True)
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(job, a): a[0] for a in args}
        winner = None
        for f in as_completed(futures):
            row = futures[f]
            result = f.result()
            print("CANDIDATE", row["name"], result.get("qualified"), flush=True)
            if result.get("qualified") and winner is None:
                winner = (row, result)
        if winner is None:
            raise SystemExit("no go-around candidate qualified")
    row, result = winner
    row["qualification"] = result
    row["qualified_speed_mps"] = result.get("speed_command")
    row["mpcc_admission"] = {"contract": "starscream-mpcc-admission-v2", "rl_eligible": True,
                              "dagger_eligible": True, "qualification_contract": contract}
    row["minimum_protected_clone_distance"] = float(
        continuous.distance(resampled(load_track(resolve(row["path"]))))
    )
    records = [row if r["name"] == FAILED else r for r in manifest["records"]]
    manifest["records"] = sorted(records, key=lambda r: int(r.get("rank", 0)))
    manifest["admission"] = {"status": "new_bridge_qualification_pending", "reused_pretrain50": 50,
                              "new_bridge_courses": 10, "replacement": row["name"]}
    atomic_json(manifest_path, manifest)
    atomic_json(OUT / "repair-ledger.json", {"replaced": [FAILED], "replacements": [row["name"]]})
    print(json.dumps({"winner": row["name"], "speed": row["qualified_speed_mps"],
                      "contract": contract}, indent=2))


if __name__ == "__main__":
    main()
