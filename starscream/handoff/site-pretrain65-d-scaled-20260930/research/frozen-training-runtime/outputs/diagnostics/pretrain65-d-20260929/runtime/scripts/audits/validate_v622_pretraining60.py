"""Release checks for the reused-plus-bridged v6.22 train60 panel."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track
from scripts.audits.review_v622_geometry import resampled
from scripts.audits.build_v622_pretraining50 import _cell_axes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/v622-pretraining60"

def resolve(path):
    p = Path(path)
    return ROOT / p.relative_to("/workspace") if str(p).startswith("/workspace/") else p

def load_manifest(path):
    return json.loads(path.read_text())

def stats(rows):
    gs = [geometry_record(load_track(resolve(r["path"]))) for r in rows]
    return {"gate_count": {"min": int(min(x["gate_count"] for x in gs)), "median": float(np.median([x["gate_count"] for x in gs])), "mean": float(np.mean([x["gate_count"] for x in gs])), "max": int(max(x["gate_count"] for x in gs))},
            "alternating_pairs": {"min": int(min(x["alternating_pairs"] for x in gs)), "median": float(np.median([x["alternating_pairs"] for x in gs])), "mean": float(np.mean([x["alternating_pairs"] for x in gs])), "max": int(max(x["alternating_pairs"] for x in gs))}}

def main():
    raw = load_manifest(OUT / "manifest.json")
    tuned = load_manifest(OUT / "mpcc-frontier-v3/manifest.json")
    real60 = load_manifest(ROOT / "configs/eval/v6_22_real60.manifest.json")
    real100 = load_manifest(ROOT / "configs/eval/v6_22_real100_hard_v2.manifest.json")
    rows = tuned["records"]
    assert len(rows) == 60 and len({r["name"] for r in rows}) == 60
    assert all(r.get("qualified_speed_mps") and r.get("qualification") for r in rows)
    train_paths = {str(resolve(r["path"])) for r in rows}
    eval_paths = {str(resolve(r["path"])) for r in real60["records"] + real100["records"]}
    assert all(Path(p).exists() for p in train_paths | eval_paths)
    assert not train_paths & eval_paths
    train_fp = {r["fingerprint"] for r in rows}
    eval_fp = {r["fingerprint"] for r in real60["records"] + real100["records"]}
    assert not train_fp & eval_fp
    protected = [load_track(resolve(r["path"])) for r in real60["records"] + real100["records"]]
    idx = CloneIndex([resampled(t) for t in protected])
    distances = [idx.distance(resampled(load_track(resolve(r["path"])))) for r in rows]
    assert min(distances) >= .06
    family_counts = Counter(r["family"] for r in rows)
    weights = {str(k): float(v) for k, v in json.loads((ROOT / "configs/exp/v6.22/pretraining_behavior60_dagger.yaml").read_text())["dagger"]["track_sampling_family_weights"].items()}
    weighted = {f: family_counts[f] * weights[f] for f in family_counts}
    total = sum(weighted.values())
    speeds = np.asarray([float(r["qualified_speed_mps"]) for r in rows])
    norm = torch.load(OUT / "normalization.pt", map_location="cpu", weights_only=False)
    assert norm["episodes"] == 120 and norm["accepted_episodes"] == 120
    report = {"schema": "starscream-v622-pretraining60-validation-v1", "status": "PASS", "courses": 60,
        "reused_pretrain50": 50, "new_bridges": 10,
        "new_bridge_qualification": json.loads((OUT / "qualification-summary.json").read_text()),
        "family_counts": dict(sorted(family_counts.items())), "family_weights": weights,
        "weighted_sampling_share": {f: weighted[f] / total for f in sorted(weighted)},
        "real60_family_counts": dict(Counter(r["family"] for r in real60["records"])),
        "real100_family_counts": dict(Counter(r["family"] for r in real100["records"])),
        "train_geometry": stats(rows), "real60_geometry": stats(real60["records"]),
        "real100_geometry": stats(real100["records"]),
        "minimum_continuous_distance_to_eval": float(min(distances)),
        "qualified_speed_mps": {"min": float(speeds.min()), "median": float(np.median(speeds)), "mean": float(speeds.mean()), "max": float(speeds.max())},
        "dart_exemptions": sum(not bool(r.get("dart_eligible", False)) for r in rows),
        "normalization": {"episodes": norm["episodes"], "accepted_episodes": norm["accepted_episodes"], "rows": norm["rows"], "valid_dynamics_rows": norm["valid_dynamics_rows"], "teacher_solver_failures": norm["teacher_solver_failures"], "sha256": hashlib.sha256((OUT / "normalization.pt").read_bytes()).hexdigest()},
        "evaluation": {"real60_courses": len(real60["records"]), "real100_courses": len(real100["records"]), "eval_manifest_disjoint": True},
        "config": {"path": "/workspace/configs/exp/v6.22/pretraining_behavior60_dagger.yaml", "rounds": 75, "episodes_per_round": 480, "evaluation_episodes": 180, "reporting_evaluation_episodes": 200}}
    atomic_json(OUT / "validation-report.json", report)
    print(json.dumps({"status": report["status"], "courses": 60, "min_continuous_distance": report["minimum_continuous_distance_to_eval"], "speed_mean": report["qualified_speed_mps"]["mean"], "dart_exemptions": report["dart_exemptions"], "normalization_rows": norm["rows"]}, indent=2))

if __name__ == "__main__": main()
