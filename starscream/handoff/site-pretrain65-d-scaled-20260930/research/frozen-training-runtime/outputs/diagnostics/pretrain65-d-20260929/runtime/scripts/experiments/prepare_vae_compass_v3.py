"""Freeze the predeclared radius-.5 VAE/random/classical comparison."""
from pathlib import Path
import json
from starscream.course_model.training import atomic_json


def main():
    root=Path('outputs/course-model/v3/matched-panel')
    names=('source_exact_canonical','target_exact_canonical','anchored_r0.5',
           'anchored_random_r0.5','classical_r0.5')
    records={r['name']:r for r in json.loads((root/'panel.json').read_text())['records']}
    selected=[records[n] for n in names]
    if not all(r['static_valid'] for r in selected):
        raise ValueError([(r['name'],r['reasons']) for r in selected if not r['static_valid']])
    atomic_json(root/'comparison.json',dict(records=selected,
        scope='development-target comparison; radius selected before policy/MPCC evaluation',
        matched_quantity='Frobenius norm of world gate-position displacement'))
    print([(r['name'],r['target_curve_distance_m']) for r in selected])


if __name__=='__main__':main()
