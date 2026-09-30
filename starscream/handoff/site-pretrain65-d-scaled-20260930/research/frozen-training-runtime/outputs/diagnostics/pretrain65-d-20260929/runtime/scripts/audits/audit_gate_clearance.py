#!/usr/bin/env python3
"""Audit physical gate aperture clearance, independent of camera projection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import lsq_linear
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starscream.env.tracks import load_track

SUITES = ('v6_22_real60', 'v6_22_real100_hard_v2')
MIN_CLEARANCE = .75
MIN_CENTER_DISTANCE = 1.5


def aperture_distance(a, b):
    """Exact convex distance between the two finite, oriented aperture rectangles."""
    basis = np.column_stack((a.lateral, a.up, -b.lateral, -b.up)).astype(float)
    bounds = np.r_[a.size, b.size].astype(float) * .5
    result = lsq_linear(basis, np.asarray(b.position-a.position, float),
                        bounds=(-bounds, bounds), method='bvls', tol=1e-11)
    if not result.success:
        raise RuntimeError(result.message)
    return float(np.linalg.norm(basis @ result.x - (b.position-a.position)))


def crowded_pairs(track):
    pairs = []
    for i, a in enumerate(track.gates):
        if not a.render or a.kind != "gate":
            continue
        for j in range(i+1, len(track.gates)):
            b = track.gates[j]
            if not b.render or b.kind != "gate":
                continue
            center = float(np.linalg.norm(a.position-b.position))
            if center > .5*(np.linalg.norm(a.size)+np.linalg.norm(b.size))+MIN_CLEARANCE:
                continue
            clearance = aperture_distance(a,b)
            if clearance < MIN_CLEARANCE-1e-6 or center < MIN_CENTER_DISTANCE-1e-6:
                pairs.append(dict(gates_1based=[i+1,j+1], center_distance_m=center,
                                  aperture_clearance_m=clearance))
    return pairs


def audit():
    courses=[]
    for suite in SUITES:
        active=yaml.safe_load((ROOT/'configs/eval'/f'{suite}.yaml').read_text())['active']
        for record in active:
            track=load_track(ROOT/record['track'])
            pairs=crowded_pairs(track)
            exceptions=[]
            if track.metadata.get("provenance") == "exact-author-released-track-yaml":
                # Documented Swift split-S: distinct apertures, 0.53 m clear gap.
                exceptions=[p for p in pairs if p["gates_1based"] == [4,5]
                            and p["aperture_clearance_m"] >= .5
                            and "stacked split-S" in track.metadata.get("description", "")]
                pairs=[p for p in pairs if p not in exceptions]
            if track.metadata.get("benchmark_key") == "multigp_utt06":
                # Official Fury route visits the central physical gate twice;
                # route normals differ because of the intervening over/under maneuver.
                reused=[p for p in pairs if p["gates_1based"] == [10,18]
                        and p["center_distance_m"] < 1e-5
                        and track.gates[9].name == "obstacle_3_gate"
                        and track.gates[17].name == "obstacle_5_through_gate"]
                exceptions.extend(reused)
                pairs=[p for p in pairs if p not in reused]
            courses.append(dict(suite=suite,name=record['name'],path=record['track'],
                                gate_count=len(track.gates),pairs=pairs,reviewed_intentional_pairs=exceptions))
    return dict(minimum_aperture_clearance_m=MIN_CLEARANCE,
                minimum_center_distance_m=MIN_CENTER_DISTANCE,
                courses_checked=len(courses),
                affected_courses=sum(bool(r['pairs']) for r in courses),
                crowded_pairs=sum(len(r['pairs']) for r in courses),courses=courses)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='courses'},indent=2))
    for r in result['courses']:
        if r['pairs']:print(r['suite'],r['name'],r['pairs'])
    raise SystemExit(1 if result["crowded_pairs"] else 0)
