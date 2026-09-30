"""Run the established v6.22 MPCC admission contract on pretraining60."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint
from starscream.env.tracks import load_track
from scripts.audits.qualify_v622_benchmark_protocol import job, protocol_contract
from scripts.audits.build_v622_benchmarks import OUT as CANONICAL_OUT, ROOT

OUR_OUT = ROOT / "outputs/v622-pretraining60"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    manifest_path = OUR_OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = list(manifest["records"])
    for row in rows:
        row.setdefault("suite", "pretraining50")
        if geometry_fingerprint(load_track(ROOT / row["path"])) != row["fingerprint"]:
            raise ValueError(f"fingerprint drift: {row['name']}")
    cfg, contract, payload = protocol_contract()
    progress_path = OUR_OUT / "qualification-progress.json"
    progress = {"contract": contract, "total": len(rows), "qualified": [], "failed": [], "pending": [r["name"] for r in rows]}
    atomic_json(progress_path, progress)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as pool:
        futures = {pool.submit(job, (row, cfg, contract, payload["base"])): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            result = future.result()
            name = row["name"]
            progress["pending"] = [x for x in progress["pending"] if x != name]
            if result.get("qualified"):
                progress["qualified"].append(name)
            else:
                progress["failed"].append(name)
            atomic_json(progress_path, progress)
            print("ADMISSION", name, result.get("qualified"), len(progress["qualified"]), "/", len(rows), flush=True)
    # Keep a self-contained copy of the canonical evidence.  The canonical
    # cache remains shared with benchmark admission, but the training corpus
    # carries its own contract/result paths and can be reviewed independently.
    outcome_by_name = {}
    for row in rows:
        src = CANONICAL_OUT / "benchmark-qualification" / row["name"] / contract[:12] / "result.json"
        if not src.exists():
            raise FileNotFoundError(src)
        result = json.loads(src.read_text())
        dst = OUR_OUT / "qualification" / row["name"] / contract[:12] / "result.json"
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        if result["fingerprint"] != row["fingerprint"]:
            raise ValueError(f"qualification geometry drift: {row['name']}")
        outcome_by_name[row["name"]] = result
        row["qualification"] = result
        row["qualified_speed_mps"] = result.get("speed_command")
        row["mpcc_admission"] = {"contract": "starscream-mpcc-admission-v2", "rl_eligible": bool(result.get("qualified")), "dagger_eligible": bool(result.get("qualified")), "qualification_contract": contract}
    manifest["records"] = rows
    manifest["admission"] = {"contract": contract, "all_courses_qualified": len(progress["qualified"]) == len(rows), "qualified_count": len(progress["qualified"]), "failed_count": len(progress["failed"]), "protocol": "canonical-nominal2-randomized2-dart2-expert-prefix-all1"}
    atomic_json(manifest_path, manifest)
    atomic_json(OUR_OUT / "qualification-summary.json", {"contract": contract, "total": len(rows), "qualified": len(progress["qualified"]), "failed": sorted(progress["failed"])})
    if len(progress["qualified"]) != len(rows):
        raise SystemExit("pretraining50 has MPCC admission failures")
    print(json.dumps({"total": len(rows), "qualified": len(progress["qualified"]), "contract": contract}, indent=2))


if __name__ == "__main__":
    main()
