"""Install a previously admitted go-around candidate into train60."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint
from starscream.env.racing_manifold.benchmark_v22 import requirement_cells
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track
from scripts.audits.build_v622_pretraining50 import _cell_axes
from scripts.audits.review_v622_geometry import resampled
from scripts.audits.build_v622_benchmarks import ROOT
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex

OUT = ROOT / "outputs/v622-pretraining60"
OLD = "v622_pretrain60_repair_go_around_bridge_01_000"
NEW = "v622_pretrain60_go_around_bridge_repair_057"

def resolve(path):
    p = Path(path)
    return ROOT / p.relative_to("/workspace") if str(p).startswith("/workspace/") else p

def main():
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    old = next(r for r in manifest["records"] if r["name"] == OLD)
    track = load_track(OUT / "tracks" / f"{NEW}.yaml")
    result_path = Path("/workspace/outputs/course-pools/v622-benchmarks/benchmark-qualification") / NEW / "5a965f44a24e" / "result.json"
    result = json.loads(resolve(result_path).read_text())
    if not result.get("qualified"):
        raise RuntimeError("candidate is not qualified")
    protected = []
    for r in manifest["records"]:
        if r["name"] != OLD:
            protected.append(load_track(resolve(r["path"])))
    for name in ("v6_22_real60.manifest.json", "v6_22_real100_hard_v2.manifest.json"):
        protected += [load_track(resolve(r["path"])) for r in json.loads((ROOT / "configs/eval" / name).read_text())["records"]]
    continuous = CloneIndex([resampled(t) for t in protected])
    row = dict(old)
    row.update({"name": NEW, "path": "/workspace/outputs/v622-pretraining60/tracks/" + NEW + ".yaml",
        "source": "v622-pretraining-behavior-program-v2-go-around-repair", "seed": 88310057,
        "geometry_fingerprint": geometry_fingerprint(track), "track_fingerprint": track.fingerprint,
        "fingerprint": geometry_fingerprint(track), "cells": sorted(requirement_cells(track)),
        "coarse_cells": sorted(_cell_axes(track)), "geometry": geometry_record(track),
        "behavior_signature": dict((track.metadata or {}).get("behavior_signature", {})),
        "minimum_protected_clone_distance": float(continuous.distance(resampled(track))),
        "qualified_speed_mps": None, "qualification": None,
        "mpcc_admission": {"contract": "starscream-mpcc-admission-v2", "rl_eligible": False, "dagger_eligible": False}})
    records = [row if r["name"] == OLD else r for r in manifest["records"]]
    manifest["records"] = sorted(records, key=lambda r: int(r.get("rank", 0)))
    manifest["admission"] = {"status": "new_bridge_qualification_pending", "reused_pretrain50": 50,
                              "new_bridge_courses": 10, "replacement": NEW}
    atomic_json(manifest_path, manifest)
    atomic_json(OUT / "repair-ledger.json", {"replaced": [OLD], "replacements": [NEW]})
    print(json.dumps({"replacement": NEW, "qualified_result": str(result_path),
                      "continuous_distance": row["minimum_protected_clone_distance"]}, indent=2))
if __name__ == "__main__": main()
