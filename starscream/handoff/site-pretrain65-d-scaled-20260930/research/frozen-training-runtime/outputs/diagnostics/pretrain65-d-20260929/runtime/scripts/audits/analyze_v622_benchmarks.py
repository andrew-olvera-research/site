"""Compare measured benchmark requirements with real50/pretraining/online banks.

No policy-dependent admission and no use of real60 to emit training targets.
"""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.audits.build_v622_benchmarks import OUT, ROOT, resolve
from starscream.course_model.training import atomic_json
from starscream.env.tracks import load_track
from starscream.env.procedural_tracks import geometry_fingerprint
from starscream.env.racing_manifold.benchmark_v22 import requirement_cells
from starscream.env.racing_manifold.corpus_coverage import geometry_record


def summarize(tracks):
    records = [geometry_record(t) for t in tracks]
    witnesses = Counter(c for t in tracks for c in requirement_cells(t))
    values = {}
    for key in ('incoming_m', 'turn_deg', 'height_change_m', 'gate_center_height_m', 'width_m'):
        v = [r[key] for record in records for r in record['transitions']]
        values[key] = dict(zip(('min', 'p10', 'median', 'p90', 'max'), map(float, np.quantile(v, [0, .1, .5, .9, 1]))))
    courses = []
    for t, record in zip(tracks, records):
        ts = record['transitions']
        courses.append(dict(name=t.name, route_checkpoints=len(t.gates), physical_crossings=sum(g.render for g in t.gates),
            length=sum(x['incoming_m'] for x in ts),
            long_low=sum(x['incoming_m'] >= 25 and x['gate_center_height_m'] <= 1.2 for x in ts),
            long_hard=sum(x['incoming_m'] >= 25 and x['turn_deg'] >= 120 for x in ts),
            vertical_hard=sum(abs(x['height_change_m']) >= 2 and x['turn_deg'] >= 90 for x in ts),
            wrong_side=sum(x['preceding_gate_on_exit_side'] for x in ts),
            left=record['left_turns'], right=record['right_turns']))
    return dict(courses=len(tracks), transitions=sum(len(t.gates) for t in tracks),
        cells=len(witnesses), cells_with_3_courses=sum(n >= 3 for n in witnesses.values()),
        course_witnesses=dict(witnesses), transition_quantiles=values, per_course=courses,
        independent_witness_warning='Singleton chain4 cells are diagnostics, not evidence of supported behavioral modes.')


def main():
    data = json.loads((OUT/'candidates.json').read_text())
    groups = {}
    suite = yaml.safe_load((ROOT/'configs/eval/v6_21_holdout_eval_50.yaml').read_text())
    groups['real50'] = [load_track(resolve(r['track'])) for r in suite['active']]
    manifest = json.loads((ROOT/'outputs/course-pools/v621-additions/manifest.json').read_text())
    groups['pretrain45'] = [load_track(resolve(r['path'])) for r in manifest['records'] if r['split'] == 'train']
    for split in ('real60', 'real100-hard'):
        released = ROOT/'configs/eval'/f'v6_22_{split.replace("-", "_")}.manifest.json'
        if released.exists():
            selected = json.loads(released.read_text())['records']
            label = 'released'
        else:
            plan = json.loads((OUT/'independence-selection.json').read_text())
            names = {choices[0]['name'] for choices in plan['preferences'].values() if choices}
            selected = [r for r in data['records'] if r['suite'] == 'public_reference' or r['name'] in names]
            selected = [r for r in selected if r['suite'] in (split, 'public_reference')]
            label = 'proposed'
        groups[split+'-'+label] = [load_track(resolve(r['path'])) for r in selected]
        groups[split+'-generated'] = [load_track(resolve(r['path'])) for r in selected if r['suite'] == split]
    banks = list((ROOT/'outputs/checkpoints').glob('*/bank/**/*.yaml'))
    if banks:
        # Union is exposure protection, not a claim that all were simultaneously
        # admitted or active in a particular checkpoint.
        unique = {}
        for path in banks:
            track = load_track(path)
            unique[geometry_fingerprint(track)] = track
        groups['saved-bank-geometry-union'] = list(unique.values())
    result = {name:summarize(tracks) for name, tracks in groups.items()}
    atomic_json(OUT/'distribution-audit.json', result)
    print(json.dumps({name:{k:v for k,v in r.items() if k not in ('course_witnesses', 'per_course')} for name,r in result.items()}, indent=2))


if __name__ == '__main__':
    main()
