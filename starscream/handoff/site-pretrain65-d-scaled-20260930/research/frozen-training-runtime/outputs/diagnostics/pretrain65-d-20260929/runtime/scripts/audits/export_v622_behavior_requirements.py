"""Export training-design requirements from released real100-hard only.

Outputs cells and witness targets, never benchmark gate coordinates or labels.
Does not generate/train a model or consult real60 results.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.audits.build_v622_benchmarks import ROOT, OUT
from starscream.course_model.training import atomic_json


def requirements(payload, source_sha256):
    if payload.get('training_distribution_design_allowed') is not True:
        raise ValueError('This benchmark is protected from training-distribution design.')
    records = payload['records']
    if len(records) != 100 or any(r['suite'] not in ('real100-hard', 'public_reference') for r in records):
        raise ValueError('Expected the independent real100-hard release, not real60 or a candidate pool.')
    witnesses = Counter(c for row in records for c in set(row['cells']))
    return dict(schema='starscream-v622-training-requirements-v1',
        source_manifest_sha256=source_sha256,
        scope='benchmark-informed training design; real100-hard is subsequently a development benchmark',
        target_witnesses_per_independent_split=3,
        independent_train_development_geometry=True,
        protected_suites=['configs/eval/v6_22_real60.yaml','configs/eval/v6_22_real100_hard.yaml'],
        clone_limits=dict(same_count=.12, resampled_centerline=.06),
        requirements=[dict(cell=c, source_course_witnesses=n,
            priority='core' if n >= 3 else 'edge_composition',
            minimum_independent_train_courses=3, minimum_independent_development_courses=3)
            for c,n in sorted(witnesses.items())],
        teacher_requirements=dict(per_course_pace_search=True,
            stored_successful_prefix_trajectories=True,
            per_geometry_speed_commands='multiple caps including one below the curvature-limited plateau',
            measured_coordinates=['arrival_speed','braking_p90','turning_radius_p10','minimum_speed'],
            admission='separate geometry coverage, teacher feasibility, and pilot learnability'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'configs/eval/v6_22_real100_hard.manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'configs/exp/v6.22/behavior_requirements.json')
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    result = requirements(manifest, hashlib.sha256(raw).hexdigest())
    measured_path = OUT/'admitted-behavior-audit.json'
    measured_raw = measured_path.read_bytes()
    measured = json.loads(measured_raw)
    if not measured['complete'] or measured['contract'] != manifest['benchmark_admission_contract_sha256']:
        raise ValueError('Run the complete admitted behavior audit for the released contract first.')
    # Select only hard/public records: protected real60 behavior is never used
    # to derive training targets, even though the diagnostic file contains it.
    records = [r for group in ('real100-hard','public_reference')
               for r in measured['groups'][group]['records']]
    expected = {r['name']:r['fingerprint'] for r in manifest['records']}
    if {r['name']:r['fingerprint'] for r in records} != expected or len(records) != 100:
        raise ValueError('Executed behavior does not match the frozen hard benchmark.')
    counts = Counter(c for r in records for c in set(r['cells']))
    result['executed_behavior'] = dict(source_audit_sha256=hashlib.sha256(measured_raw).hexdigest(),
        interpretation='Physical-aperture, accepted-teacher behavior; separate from intrinsic geometry requirements and not an optimal-speed claim.',
        requirements=[dict(cell=c,source_course_witnesses=n,
            minimum_independent_train_courses=3,minimum_independent_development_courses=3)
            for c,n in sorted(counts.items())])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(args.output, len(result['requirements']), 'requirement cells')


if __name__ == '__main__':
    main()
