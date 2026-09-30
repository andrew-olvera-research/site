"""Executed behavior coverage, separate from geometric requirement proxies.

Compare a fixed common teacher profile as well as per-course tuned profiles;
otherwise changing the teacher could masquerade as a geometry effect.
"""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.audits.build_v622_benchmarks import OUT, teacher_contract, usable
from starscream.course_model.training import atomic_json


def trace_metrics(path):
    with np.load(path) as t:
        state = t['states']; nxt = t['next_states']; hz = float(t['control_hz'])
        if len(state) < 2: return []
        velocity = state[:, 7:10]; acceleration = (nxt[:, 7:10]-velocity)*hz
        speed = np.linalg.norm(velocity, axis=1)
        tangent = (velocity*acceleration).sum(1)/np.maximum(speed, .5)
        curvature = np.linalg.norm(np.cross(velocity, acceleration), axis=1)/np.maximum(speed**3, 1.)
        result = []
        for phase in np.unique(t['phase']):
            mask = t['phase'] == phase
            turning = mask & (speed > 3.) & (curvature > .02)
            result.append(dict(phase=int(phase), arrival_speed=float(speed[mask][-1]),
                approach_peak_speed=float(speed[mask].max()), minimum_speed=float(speed[mask].min()),
                braking_p90=float(np.quantile(np.maximum(-tangent[mask], 0), .9)),
                turning_radius_p10=float(np.quantile(1/curvature[turning], .1)) if turning.any() else None))
        return result


def main():
    _, contract, _ = teacher_contract()
    data = json.loads((OUT/'candidates.json').read_text())
    groups = {}
    for row in data['records']:
        if row.get('rank', 0): continue
        root = OUT/'qualification'/row['name']/contract[:12]
        if not (root/'screen.json').exists(): continue
        result = json.loads((root/'screen.json').read_text())
        if not result['screen_passed']: continue
        for comparison, profile in [('common-base', 'base'), ('tuned', result['selected_profile'])]:
            folder = root/f'{profile}-nominal-screen'
            report = json.loads((folder/'report.json').read_text())['report']
            rows = []
            for index, episode in enumerate(report['episodes']):
                if usable(episode): rows += trace_metrics(folder/f'episode-{index:03d}.npz')
            if not rows: continue
            # Each course contributes at most one witness to a measured cell.
            cells = {f'speed-brake:{np.searchsorted([3,6,9,12,16,20],t["arrival_speed"])}:'
                     f'{np.searchsorted([2,5,10,15,20],t["braking_p90"])}' for t in rows}
            cells.update(f'speed-radius:{np.searchsorted([3,6,9,12,16,20],t["arrival_speed"])}:'
                         f'{np.searchsorted([2,4,8,16,32],t["turning_radius_p10"])}'
                         for t in rows if t['turning_radius_p10'] is not None)
            key = row['suite']+'/'+comparison
            groups.setdefault(key, []).append(dict(name=row['name'], profile=profile, cells=sorted(cells), transitions=rows))
    output = {}
    for key, courses in groups.items():
        cells = Counter(c for r in courses for c in r['cells'])
        output[key] = dict(courses=len(courses), course_witnesses=dict(cells),
                           cells_with_three_courses=sum(n >= 3 for n in cells.values()), records=courses)
    atomic_json(OUT/'executed-behavior-audit.json', dict(contract=contract,
        scope='nominal successful rank-zero proposals; not final selection or randomized reliability', groups=output))
    print({key:(r['courses'], len(r['course_witnesses']), r['cells_with_three_courses']) for key,r in output.items()})


if __name__ == '__main__':
    main()
