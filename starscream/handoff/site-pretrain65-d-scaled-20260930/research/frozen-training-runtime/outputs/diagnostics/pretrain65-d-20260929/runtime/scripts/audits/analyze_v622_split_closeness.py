"""Measure how much independent real100-hard resembles real60."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.audits.review_v622_geometry import resampled
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex, requirement_cells
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track


ROOT = Path(__file__).resolve().parents[2]


def q(values):
    values = np.asarray(values, float)
    return dict(zip(('min','p10','median','p90','max'), np.quantile(values,[0,.1,.5,.9,1]).tolist())) if len(values) else {}


def load_split(stem):
    manifest = json.loads((ROOT/'configs/eval'/f'{stem}.manifest.json').read_text())
    suite = 'real100-hard' if 'real100' in stem else 'real60'
    generated = [r for r in manifest['records'] if r['suite'] == suite]
    return manifest, generated, [load_track(ROOT/r['path']) for r in generated]


def main():
    real60_manifest, real60_rows, real60 = load_split('v6_22_real60')
    hard_manifest, hard_rows, hard = load_split('v6_22_real100_hard')
    real60_proxies = [resampled(t) for t in real60]; hard_proxies = [resampled(t) for t in hard]
    index = CloneIndex(real60_proxies)
    real60_cells = [requirement_cells(t) for t in real60]; hard_cells = [requirement_cells(t) for t in hard]
    nearest = []
    for row, track, proxy, cells in zip(hard_rows, hard, hard_proxies, hard_cells):
        distances = [CloneIndex([other]).distance(proxy) for other in real60_proxies]
        jaccards = [len(cells & other)/max(len(cells | other),1) for other in real60_cells]
        i = int(np.argmin(distances)); nearest.append(dict(name=row['name'],family=row['family'],
            distance=float(distances[i]), nearest_real60=real60_rows[i]['name'],
            nearest_family=real60_rows[i]['family'], cell_jaccard=float(jaccards[i]),
            independent_cells=len(cells), nearest_cells=len(real60_cells[i])))
    def geometry_stats(tracks):
        records=[geometry_record(t) for t in tracks]; ts=[x for r in records for x in r['transitions']]
        keys=('incoming_m','turn_deg','height_change_m','gate_center_height_m','width_m','incoming_alignment')
        return dict(courses=len(tracks), gates=sum(len(t.gates) for t in tracks),
            cells=len(set().union(*(requirement_cells(t) for t in tracks))),
            cell_witnesses=sum(n>=3 for n in Counter(c for t in tracks for c in requirement_cells(t)).values()),
            gate_count=q([len(t.gates) for t in tracks]),
            transitions={key:q([x[key] for x in ts]) for key in keys},
            special=dict(long_low=sum(x['incoming_m']>=25 and x['gate_center_height_m']<=1.2 for x in ts),
                long_turn=sum(x['incoming_m']>=25 and x['turn_deg']>=120 for x in ts),
                vertical_turn=sum(abs(x['height_change_m'])>=2 and x['turn_deg']>=90 for x in ts),
                wrong_side=sum(x['preceding_gate_on_exit_side'] for x in ts),
                reverse_entry=sum(x['reverse_entry'] for x in ts),
                narrow=sum(x['width_m']<=1.65 for x in ts)))
    output=dict(schema='starscream-v622-split-closeness-v1',
        scope='fresh generated geometry only; public references excluded',
        real60=geometry_stats(real60), real100_hard=geometry_stats(hard),
        nearest_hard_to_real60=dict(distance=q([x['distance'] for x in nearest]),
            cell_jaccard=q([x['cell_jaccard'] for x in nearest]),
            same_family_fraction=float(np.mean([x['family']==x['nearest_family'] for x in nearest])),
            records=nearest),
        interpretation=['Clone distance is whole-course shape similarity after cyclic/reflection normalization, not behavioral similarity.',
            'Cell Jaccard compares factorized requirement witnesses and can be high for courses with different precise coordinates.',
            'Neither metric establishes policy difficulty or causal transfer.'])
    path=ROOT/'outputs/course-pools/v622-benchmarks/split-closeness.json'; path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(real60=output['real60'],real100_hard=output['real100_hard'],nearest=output['nearest_hard_to_real60'],output=str(path)),indent=2))


if __name__=='__main__': main()
