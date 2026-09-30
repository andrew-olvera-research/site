"""Tune the ten new train60 bridges and publish one combined frontier."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from scripts.audits import tune_v622_pretraining50_mpcc as tune

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/v622-pretraining60/manifest.json"
BASE_DIR = ROOT / "outputs/v622-pretraining50/mpcc-frontier-v3"
OUT = ROOT / "outputs/v622-pretraining60/mpcc-frontier-v3"
CONFIG = ROOT / "configs/exp/v6.22/pretraining_behavior50_dagger.yaml"

def main():
    source = json.loads(SOURCE.read_text())
    base_manifest = json.loads((BASE_DIR / "manifest.json").read_text())
    base_summary = json.loads((BASE_DIR / "summary.json").read_text())
    base_profiles = json.loads((BASE_DIR / "teacher-profiles.json").read_text())
    base_names = {r["name"] for r in base_manifest["records"]}
    base_rows = {r["name"]: r for r in base_manifest["records"]}
    new_rows = [r for r in source["records"] if r["name"] not in base_names]
    if len(source["records"]) != 60 or len(new_rows) != 10:
        raise ValueError(f"expected 60 rows with 10 new bridges, got {len(source['records'])}/{len(new_rows)}")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text())
    settings = cfg["dagger"]
    source_hashes = {"config": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                     "manifest": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                     "tuner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "base_frontier": hashlib.sha256((BASE_DIR / "manifest.json").read_bytes()).hexdigest()}
    contract = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    atomic_json(OUT / "contract.json", {"schema": "starscream-v622-pretraining60-frontier-v1",
                                         "contract": contract, "source_hashes": source_hashes})
    results = []
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(tune._job, (settings, row, str(OUT), contract)) for row in new_rows]
        for f in as_completed(futures):
            result = f.result(); results.append(result)
            print(f"{len(results)}/{len(new_rows)} {result['name']} {result['admitted_speed_mps']:g} -> {result['selected']['speed']:g} m/s", flush=True)
    if len(results) != 10:
        raise RuntimeError("frontier did not complete")
    new_by_name = {r["name"]: r for r in results}
    base_by_name = {r["name"]: r for r in base_summary["records"]}
    combined_records = []
    controllers = dict(base_profiles["controller_profiles"])
    planners = dict(base_profiles["planner_profiles"])
    for row in source["records"]:
        row = deepcopy(row)
        result = base_by_name[row["name"]] if row["name"] in base_by_name else new_by_name[row["name"]]
        selected = result["selected"]
        profile = "v622-frontier-" + row["fingerprint"][:12]
        row["qualified_speed_mps"] = selected["speed"]
        # The pretrain50 frontier's stronger DART evidence is authoritative
        # for reused rows.  The merged summary only carries the pace search,
        # so replacing this flag with its weaker result would erase the
        # established exemptions.  New bridges use their own frontier result.
        row["dart_eligible"] = bool(
            base_rows[row["name"]]["dart_eligible"]
            if row["name"] in base_rows else result["dart_eligible"]
        )
        row["dart_exemption_reason"] = None if row["dart_eligible"] else "MPCC pace passes nominal/randomized but not the stronger perturbed cohort"
        row["v622_pace_qualification"] = result
        q = deepcopy(row.get("qualification") or {})
        q["selected"] = dict(q.get("selected") or {})
        q["selected"]["teacher_profile"] = profile
        row["qualification"] = q
        controllers[profile] = selected["controller"]
        planners[profile] = selected["planner"]
        combined_records.append(row)
    combined = deepcopy(source)
    combined["records"] = combined_records
    combined["pace_frontier"] = {"schema": "starscream-v622-pretraining60-frontier-v1", "contract": contract,
                                  "search_ceiling_mps": 36., "grid_resolution_mps": 2.,
                                  "reused_pretrain50_frontier": str(BASE_DIR / "manifest.json")}
    atomic_json(OUT / "manifest.json", combined)
    atomic_json(OUT / "teacher-profiles.json", {"schema": "starscream-v622-pretraining60-frontier-v1",
                                                "contract": contract, "controller_profiles": controllers,
                                                "planner_profiles": planners})
    atomic_json(OUT / "summary.json", {"schema": "starscream-v622-pretraining60-frontier-v1", "contract": contract,
                                       "complete": True, "reused_pretrain50": 50, "new_bridges": 10,
                                       "records": [base_by_name[r["name"]] if r["name"] in base_by_name else new_by_name[r["name"]] for r in source["records"]]})
    atomic_json(OUT / "frontier-report.json", {"schema": "starscream-v622-pretraining60-frontier-report-v1",
        "contract": contract, "courses": [{"name": r["name"], "maximum_acceptable_speed_mps": r["qualified_speed_mps"],
        "dart_eligible": r["dart_eligible"], "teacher_profile": r["qualification"]["selected"]["teacher_profile"]} for r in combined_records]})
    print(json.dumps({"courses": 60, "reused": 50, "new": 10, "contract": contract,
                      "dart_eligible": sum(bool(r["dart_eligible"]) for r in combined_records),
                      "mean_speed_mps": sum(float(r["qualified_speed_mps"]) for r in combined_records) / 60}, indent=2))

if __name__ == "__main__": main()
