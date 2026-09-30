"""Independent whole-course, different-count clone and reference semantic audit."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.audits.build_v622_benchmarks import OUT, protected_tracks
from starscream.course_model.training import atomic_json
from starscream.env.tracks import load_track
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex


def resampled(track, count=64):
    p = np.asarray([g.position for g in track.gates], float)
    p = np.concatenate((p, p[:1]))
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    positions = np.stack([np.interp(np.linspace(0, length[-1], count, endpoint=False), length, p[:, j])
                          for j in range(3)], axis=1)
    return SimpleNamespace(gates=[SimpleNamespace(position=p) for p in positions])


def reference_review(track):
    problems = []
    virtual = []
    for i, g in enumerate(track.gates):
        if not g.render:
            virtual.append(dict(index=i, kind=g.kind, size=g.size.tolist()))
            continue
        extent = abs(g.lateral[2])*g.size[0]/2 + abs(g.up[2])*g.size[1]/2
        if g.position[2]-extent < -1e-5:
            problems.append(f'physical_aperture_below_ground:{i}')
    p = np.array([g.position for g in track.gates])
    repeats = [(i, j) for i in range(len(p)) for j in range(i)
               if np.linalg.norm(p[i]-p[j]) < .4]
    return dict(name=track.name, physical_crossings=sum(g.render for g in track.gates),
                virtual_checkpoints=virtual, repeated_locations=repeats, errors=problems,
                source_metadata=track.metadata)


def main():
    data = json.loads((OUT/'candidates.json').read_text())
    protected = protected_tracks()
    index = CloneIndex([resampled(t) for t in protected])
    records, selected = [], []
    references = []
    for row in data['records']:
        if row.get('rank', 0): continue
        t = load_track(row['path'])
        if row['suite'] == 'public_reference':
            references.append(reference_review(t)); continue
        proxy = resampled(t)
        distance = index.distance(proxy)
        records.append(dict(name=t.name, suite=row['suite'], minimum_resampled_distance=distance,
                            review_flag=distance < .08))
        index.add(proxy); selected.append(t)
    result = dict(scope='rank-zero proposals; rerun on final selection before release',
                  metric='64-point arc-length resampling, cyclic yaw/reflection normalized RMS',
                  review_threshold=.08, records=records, references=references)
    atomic_json(OUT/'geometry-review.json', result)
    print(json.dumps(dict(flagged=[r for r in records if r['review_flag']],
                          reference_errors={r['name']:r['errors'] for r in references}), indent=2))


if __name__ == '__main__':
    main()
