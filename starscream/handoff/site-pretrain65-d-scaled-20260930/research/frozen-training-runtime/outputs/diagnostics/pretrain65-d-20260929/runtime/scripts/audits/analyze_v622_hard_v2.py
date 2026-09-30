"""Compare hard-v2 behavior/shape support against frozen real60 and hard."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex, requirement_cells
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track
from scripts.audits.review_v622_geometry import resampled

ROOT = Path(__file__).resolve().parents[2]


def load(stem, suite):
    d = json.loads((ROOT / 'configs/eval' / f'{stem}.manifest.json').read_text())
    rows = [r for r in d['records'] if r['suite'] == suite]
    return rows, [load_track(ROOT / r['path']) for r in rows]


def quantile(xs):
    return dict(zip(('min','p10','median','p90','max'), np.quantile(xs, [0,.1,.5,.9,1]).tolist())) if xs else {}


def stats(tracks):
    rec = [geometry_record(t) for t in tracks]
    trans = [x for r in rec for x in r['transitions']]
    return dict(courses=len(tracks), gates=sum(len(t.gates) for t in tracks),
        unique_cells=len(set().union(*(requirement_cells(t) for t in tracks))),
        gate_count=quantile([len(t.gates) for t in tracks]),
        incoming=quantile([x['incoming_m'] for x in trans]),
        turn=quantile([x['turn_deg'] for x in trans]),
        abs_height_change=quantile([abs(x['height_change_m']) for x in trans]),
        gate_height=quantile([x['gate_center_height_m'] for x in trans]),
        long_low=sum(x['incoming_m'] >= 25 and x['gate_center_height_m'] <= 1.2 for x in trans),
        long_turn=sum(x['incoming_m'] >= 25 and x['turn_deg'] >= 120 for x in trans),
        vertical_turn=sum(abs(x['height_change_m']) >= 2 and x['turn_deg'] >= 90 for x in trans),
        wrong_side=sum(x['preceding_gate_on_exit_side'] for x in trans),
        reverse_entry=sum(x['reverse_entry'] for x in trans),
        narrow=sum(min(x['width_m'], x['height_m']) <= 1.65 for x in trans))


def nearest(source, target):
    index = CloneIndex([resampled(t) for t in target])
    target_cells = [requirement_cells(t) for t in target]
    result = []
    for t in source:
        proxy = resampled(t); d = index.distance(proxy)
        js = [len(requirement_cells(t) & c) / max(1, len(requirement_cells(t) | c)) for c in target_cells]
        result.append((d, max(js)))
    return dict(distance=quantile([x[0] for x in result]), max_cell_jaccard=quantile([x[1] for x in result]))


def main():
    _, real60 = load('v6_22_real60', 'real60')
    _, baseline = load('v6_22_real100_hard', 'real100-hard')
    v2_rows, v2 = load('v6_22_real100_hard_v2', 'real100-hard-v2')
    old_rows, old_tracks = load('v6_22_real100_hard', 'real100-hard')
    replacement = [r for r in v2_rows if r.get('source') == 'v622-hard-v2-behavior-program']
    output = dict(schema='starscream-v622-hard-v2-distribution-v1',
        suites=dict(real60=stats(real60), baseline_real100_hard=stats(baseline), hard_v2=stats(v2)),
        nearest_real60=dict(baseline=nearest(baseline, real60), hard_v2=nearest(v2, real60)),
        replacement_panel=dict(count=len(replacement), strata=dict(Counter(r['behavior_signature']['stratum'] for r in replacement)),
            minimum_continuous_distance=min(r.get('continuous_distance', 9) for r in replacement),
            median_continuous_distance=float(np.median([r.get('continuous_distance', 9) for r in replacement])),
            minimum_exact_distance=min(r.get('exact_distance', 9) for r in replacement),
            old_success=quantile([r['old_policy_success'] for r in replacement]),
            old_nearest_real60=quantile([r['old_nearest_real60_distance'] for r in replacement])),
        interpretation=['The v2 panel is not selected by a new policy score; the 30 replacements are the frozen RL-easy/real60-near tail.',
            'Behavior-program strata add long-low braking, vertical chains, wrong-side incidence, compound reversals, and radius switching as independent axes.',
            'The MPCC contract proves labeler admissibility; randomized fast/DART failures are retained as a design diagnostic during candidate iteration, not silently counted as geometry validity.'])
    out = ROOT / 'outputs/v622-hard-v2/distribution-analysis.json'; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2)); print('WROTE', out)


if __name__ == '__main__': main()
