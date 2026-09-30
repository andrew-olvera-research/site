"""Diagnose why A2RL/UTT remain unsupported by v6.21 train and online bank."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from starscream.env.racing_manifold.atlas import ManifoldAtlas
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.racing_manifold.descriptors import analyze_track_geometry
from starscream.env.racing_manifold.transition_cells import (
    arrival_cells, describe_cell, transition_cells,
)
from starscream.env.real_course_suite import load_active_real_course_suite
from starscream.env.tracks import load_track


def features(track):
    r = geometry_record(track)
    t = r["transitions"]
    return dict(
        gates=len(t), length_m=sum(x["incoming_m"] for x in t),
        incoming_ge_15=sum(x["incoming_m"] >= 15 for x in t),
        incoming_ge_25=sum(x["incoming_m"] >= 25 for x in t),
        incoming_ge_40=sum(x["incoming_m"] >= 40 for x in t),
        hard_ge_120=sum(x["turn_deg"] >= 120 for x in t),
        hard_ge_150=sum(x["turn_deg"] >= 150 for x in t),
        vertical_ge_2=sum(abs(x["height_change_m"]) >= 2 for x in t),
        low_center=sum(x["gate_center_height_m"] <= 1.2 for x in t),
        low_center_08=sum(x["gate_center_height_m"] <= .8 for x in t),
        narrow_165=sum(x["width_m"] <= 1.65 for x in t),
        hard_after_long=sum(x["incoming_m"] >= 15 and x["turn_deg"] >= 120 for x in t),
        hard_after_very_long=sum(x["incoming_m"] >= 25 and x["turn_deg"] >= 120 for x in t),
        vertical_hard=sum(abs(x["height_change_m"]) >= 2 and x["turn_deg"] >= 90 for x in t),
        max_incoming=max(x["incoming_m"] for x in t),
        min_center=min(x["gate_center_height_m"] for x in t),
        min_width=min(x["width_m"] for x in t),
    )


def summaries(rows):
    keys = [k for k in rows[0] if k not in ("name", "source")]
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "p90": float(np.quantile([r[k] for r in rows], .9)),
                "max": float(np.max([r[k] for r in rows]))} for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    manifest = json.loads(Path("outputs/course-pools/v621-additions/manifest.json").read_text())
    train_paths = [r["path"] for r in manifest["records"] if r["split"] == "train"]
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    bank_paths = [r["path"] for r in payload["online_course_bank_state"]["courses"]]
    suite_paths, _ = load_active_real_course_suite("configs/eval/v6_21_holdout_eval_50.yaml")
    names = [Path(x).stem for x in suite_paths]
    targets = [p for p, n in zip(suite_paths, names)
               if n.startswith("a2rl_") or n.startswith("multigp_utt")]
    evaluation = json.loads(a.eval.read_text())["metrics"]

    train_tracks = [load_track(p) for p in train_paths]
    bank_tracks = [load_track(p) for p in bank_paths]
    target_tracks = [load_track(p) for p in targets]
    train_profiles = [analyze_track_geometry(t) for t in train_tracks]
    bank_profiles = [analyze_track_geometry(t) for t in bank_tracks]
    target_profiles = [analyze_track_geometry(t) for t in target_tracks]
    atlas = ManifoldAtlas(train_profiles + target_profiles)

    train_cells = Counter(c for t in train_tracks for c in transition_cells(t))
    bank_cells = Counter(c for t in bank_tracks for c in transition_cells(t))
    target_cells = Counter(c for t in target_tracks for c in transition_cells(t))
    rows = []
    failure_cells = Counter()
    failure_transitions = []
    for track in target_tracks:
        record = geometry_record(track); ts = record["transitions"]; n = len(ts)
        prefix = f"track/{track.name}/"
        p = [evaluation.get(prefix + f"p{i}", 0.0) for i in range(1, n + 1)]
        # The single-lap evaluation does not observe last->first survival, so
        # exclude that geometric closure exactly as analyze_v621_gaps does.
        for index in range(1, n):
            transition = ts[index]
            reached = p[index - 1]
            survived = p[index]
            loss = max(reached - survived, 0.0) * 32
            cells = arrival_cells(transition, ts[index - 1], ts[(index + 1) % n])
            for cell in cells: failure_cells[cell] += loss
            failure_transitions.append(dict(
                course=track.name, gate=index, reached=reached, survival=(survived / reached if reached else None),
                episodes_lost=loss, incoming_m=transition["incoming_m"], turn_deg=transition["turn_deg"],
                height_change_m=transition["height_change_m"], width_m=transition["width_m"],
                gate_center_height_m=transition["gate_center_height_m"],
                train_nearest_chain3=atlas.transition_chain_support_profiles(
                    analyze_track_geometry(track), train_profiles, horizon=3
                )["per_start_gate_distance"][index],
                bank_nearest_chain3=atlas.transition_chain_support_profiles(
                    analyze_track_geometry(track), bank_profiles, horizon=3
                )["per_start_gate_distance"][index], cells=sorted(cells),
            ))
        profile = analyze_track_geometry(track)
        item = features(track); item.update(name=track.name)
        item["success"] = evaluation[prefix + "full_course_success"]
        item["gate_fraction"] = evaluation[prefix + "mean_gates"] / n
        item["train_support"] = atlas.support_against_profiles(profile, train_profiles).to_mapping()
        item["bank_support"] = atlas.support_against_profiles(profile, bank_profiles).to_mapping()
        for h in (1, 2, 3, 4, 6):
            item[f"train_chain_{h}"] = atlas.transition_chain_support_profiles(profile, train_profiles, horizon=h)
            item[f"bank_chain_{h}"] = atlas.transition_chain_support_profiles(profile, bank_profiles, horizon=h)
        rows.append(item)

    ranked_cells = [{"cell": c, "meaning": describe_cell(c), "episodes_lost": lost,
                     "target_courses": target_cells[c], "train_courses": train_cells[c],
                     "bank_courses": bank_cells[c]}
                    for c, lost in failure_cells.most_common() if lost > 0]
    output = dict(
        checkpoint_steps=payload["environment_steps"],
        rosters={"pretrain": len(train_tracks), "bank": len(bank_tracks), "targets": len(target_tracks)},
        distribution={"pretrain": summaries([features(t) for t in train_tracks]),
                      "bank": summaries([features(t) for t in bank_tracks]),
                      "targets": summaries([features(t) for t in target_tracks])},
        targets=rows, ranked_failure_cells=ranked_cells,
        failure_transitions=sorted(failure_transitions, key=lambda x: -x["episodes_lost"]),
        cell_coverage={"target_unique": len(target_cells),
                       "pretrain_covered": sum(c in train_cells for c in target_cells),
                       "bank_covered": sum(c in bank_cells for c in target_cells)},
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"cell_coverage": output["cell_coverage"], "targets": [
        {"name": r["name"], "success": r["success"], "gate_fraction": r["gate_fraction"],
         "max_incoming": r["max_incoming"], "min_center": r["min_center"],
         "train_chain3_p90": r["train_chain_3"]["p90_distance"],
         "bank_chain3_p90": r["bank_chain_3"]["p90_distance"]} for r in rows],
         "top_failures": ranked_cells[:12]}, indent=2))


if __name__ == "__main__":
    main()
