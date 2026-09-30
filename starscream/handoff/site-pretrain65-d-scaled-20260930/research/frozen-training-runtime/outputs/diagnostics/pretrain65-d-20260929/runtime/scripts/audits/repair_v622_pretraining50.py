"""Replace pretraining50 rows rejected by MPCC without changing quotas."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.env.procedural_tracks import geometry_fingerprint, save_track_yaml
from starscream.env.racing_manifold.benchmark_hard_v2 import generate_hard_v2
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex, requirement_cells, validate_geometry, generate
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track
from scripts.audits.review_v622_geometry import resampled
from scripts.audits.build_v622_pretraining50 import ROOT, OUT, _cell_axes, _protected_tracks


def main():
    progress = json.loads((OUT / "qualification-progress.json").read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    failed = set(progress["failed"])
    if not failed:
        print("no failed rows")
        return
    kept = [r for r in manifest["records"] if r["name"] not in failed]
    protected = _protected_tracks() + [load_track(ROOT / r["path"]) for r in kept]
    exact = CloneIndex(protected); continuous = CloneIndex([resampled(t) for t in protected])
    replacements = []
    for ordinal, old in enumerate([r for r in manifest["records"] if r["name"] in failed]):
        label = old["label"]
        found = None
        for trial in range(80):
            seed = 83200000 + ordinal * 100003 + trial
            rng = np.random.default_rng(seed)
            name = f"v622_pretrain50_repair_{label}_{ordinal:02d}_{trial:02d}"
            if label.startswith("technical_"):
                # Not expected for current failures, but preserve the repair
                # tool's ability to keep quotas if a future run rejects one.
                st = {"technical_vertical":"vertical_chain","technical_compound":"compound_reversal","technical_wrong_side":"wrong_side_incidence"}[label]
                track = generate_hard_v2(st, int(rng.integers(15,19)), seed, name)
            elif label == "long_low_braking":
                track = generate_hard_v2("long_low_braking", int(rng.integers(13,18)), seed, name)
            elif label == "radius_switch":
                track = generate_hard_v2("radius_switch", int(rng.integers(13,18)), seed, name)
            else:
                family = {"ordered_3d_bridge":"ordered_3d", "stacked_reversal_bridge":"stacked_reversal", "flow_bridge":"flow", "slalom_bridge":"slalom"}[label]
                # The first panel intentionally used hard=True.  A rejected
                # bridge is replaced by a softer independent realization of
                # the same grammar, never by dropping the cell.
                track = generate(family, int(rng.integers(8,12)), seed, name, hard=False)
            if validate_geometry(track):
                continue
            ex = exact.distance(track); cont = continuous.distance(resampled(track))
            if ex < .12 or cont < .06:
                continue
            found = (track, seed, ex, cont); break
        if found is None:
            raise RuntimeError(f"could not repair {old['name']}")
        track, seed, ex, cont = found
        path = save_track_yaml(track, OUT / "tracks" / f"{track.name}.yaml")
        row = {
            "name": track.name, "path": "/workspace/" + str(path.relative_to(ROOT)), "split":"train", "suite":"pretraining50", "family": old["family"], "source":"v622-pretraining-behavior-program-repair", "seed": int(seed), "rank": old["rank"], "label": label,
            "geometry_fingerprint": geometry_fingerprint(track), "track_fingerprint": track.fingerprint, "fingerprint": geometry_fingerprint(track),
            "cells": sorted(requirement_cells(track)), "coarse_cells": sorted(_cell_axes(track)), "geometry": geometry_record(track), "behavior_signature": dict((track.metadata or {}).get("behavior_signature", {})), "minimum_protected_clone_distance": float(cont), "qualified_speed_mps": None, "qualification": None,
        }
        replacements.append(row); exact.add(track); continuous.add(resampled(track))
        print("REPAIR", old["name"], "->", row["name"], flush=True)
    manifest["records"] = kept + replacements
    manifest["records"].sort(key=lambda r: int(r.get("rank", 0)))
    manifest["admission"] = {"status":"repair_pending", "replaced_rejected_rows": sorted(failed), "repair_count": len(replacements)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Keep the evaluator suite in exact manifest order.
    import yaml
    suite = {"schema":"starscream-real-course-suite-v1", "description":"v6.22 50-course behavior-cell pretraining panel; bridge repairs pending MPCC admission", "active":[{"name":r["name"],"track":r["path"],"geometry_fingerprint":r["geometry_fingerprint"],"exposure":"fresh training geometry","family":r["family"],"label":r["label"]} for r in manifest["records"]]}
    (ROOT / "configs/eval/v6_22_pretraining50.yaml").write_text(yaml.safe_dump(suite, sort_keys=False))
    (OUT / "repair-ledger.json").write_text(json.dumps({"replaced":sorted(failed),"records":replacements}, indent=2) + "\n")
    print(json.dumps({"repaired":len(replacements),"total":len(manifest["records"])}, indent=2))


if __name__ == "__main__": main()
