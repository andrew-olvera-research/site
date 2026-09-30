"""Materialize portable local benchmark suites only after complete qualification.

This writes repository files, not an external publication. Source licensing is
still a separate review before distributing public-reference adaptations.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.audits.build_v622_benchmarks import OUT, ROOT, protected_tracks, teacher_contract
from scripts.audits.qualify_v622_benchmark_protocol import protocol_contract
from scripts.audits.review_v622_geometry import resampled, reference_review
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint, save_track_yaml
from starscream.env.tracks import load_track
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex, validate_geometry, requirement_cells
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.mpcc.racing_line import RacingLinePlanner, RacingLinePlannerConfig


def main():
    data = json.loads((OUT/'candidates.json').read_text())
    plan = json.loads((OUT/'independence-selection.json').read_text())
    _, contract, contract_payload = teacher_contract()
    _, benchmark_contract, benchmark_payload = protocol_contract()
    source = ROOT/'scripts/audits/tune_v622_reference_teachers.py'
    search_contract = hashlib.sha256((contract+source.read_text()).encode()).hexdigest()
    rows = {r['name']:r for r in data['records']}
    selected, missing = [], []
    for slot, choices in plan['preferences'].items():
        found = False
        for choice in choices:
            row = rows[choice['name']]
            path = OUT/'benchmark-qualification'/row['name']/benchmark_contract[:12]/'result.json'
            if path.exists() and json.loads(path.read_text())['qualified']:
                selected.append((row, json.loads(path.read_text()))); found = True; break
        if not found: missing.append(slot)
    for row in rows.values():
        if row['suite'] != 'public_reference': continue
        candidates = []
        path = OUT/'benchmark-qualification'/row['name']/benchmark_contract[:12]/'result.json'
        if path.exists() and json.loads(path.read_text())['qualified']:
            candidates.append(json.loads(path.read_text()))
        if candidates:
            selected.append((row, candidates[0]))
        else: missing.append(row['name'])
    status = dict(qualified=len(selected), required=152, missing=missing, release_ready=False)
    atomic_json(OUT/'release-status.json', status)
    print(json.dumps(status, indent=2), flush=True)
    if missing:
        raise SystemExit('No suites published: qualification incomplete.')
    protected = protected_tracks()
    exact = CloneIndex(protected); continuous = CloneIndex([resampled(t) for t in protected])
    planner = RacingLinePlanner(RacingLinePlannerConfig(sample_count=1400, aperture_fraction=.2,
        aperture_margin=.18, offset_iterations=30, cache_directory=str(OUT/'racing-lines')))
    if len({r['slot'] for r,e in selected}) != 152:
        raise ValueError('Repeated or missing independent slot')
    for split, expected in [('real60',52), ('real100-hard',92), ('public_reference',8)]:
        if sum(r['suite'] == split for r,e in selected) != expected:
            raise ValueError(f'Wrong portfolio size: {split}')
    # Gate all invariants before writing any published file.
    for row, evidence in selected:
        t = load_track(row['path'])
        if geometry_fingerprint(t) != row['fingerprint'] or evidence['fingerprint'] != row['fingerprint']:
            raise ValueError(f'geometry drift: {row["name"]}')
        if evidence['contract'] != benchmark_contract:
            raise ValueError(f'admission contract drift: {row["name"]}')
        line = planner.plan(t)
        if (np.any(line.position[:,:2].min(0)-2 < t.bounds[:2,0])
                or np.any(line.position[:,:2].max(0)+2 > t.bounds[:2,1])):
            raise ValueError(f'arena clips reference maneuver envelope: {row["name"]}')
        if row['suite'] == 'public_reference':
            if reference_review(t)['errors']: raise ValueError('physical reference geometry invalid')
        else:
            if validate_geometry(t): raise ValueError(f'geometry invalid: {row["name"]}')
            g = geometry_record(t)
            sign = int(np.sign(g['left_turns']-g['right_turns']))
            if sign not in (0, plan['dominant_handedness_targets'][row['slot']]):
                raise ValueError(f'handedness drift: {row["name"]}')
            if exact.distance(t) < .12 or continuous.distance(resampled(t)) < .06:
                raise ValueError(f'clone at release: {row["name"]}')
            exact.add(t); continuous.add(resampled(t))
    for split in ('real60', 'real100-hard'):
        stem = 'v6_22_'+split.replace('-', '_')
        for suffix in ('.yaml', '.manifest.json'):
            if (ROOT/'configs/eval'/f'{stem}{suffix}').exists():
                raise FileExistsError('Frozen suite exists; publish a new version instead.')
    for split, count in [('real60', 60), ('real100-hard', 100)]:
        members = [(r, e) for r,e in selected if r['suite'] in (split, 'public_reference')]
        assert len(members) == count
        records, active = [], []
        for row, evidence in members:
            t = load_track(row['path'])
            dest = ROOT/'starscream/assets/tracks/v622'/split/f'{row["name"]}.yaml'
            save_track_yaml(t, dest)
            if geometry_fingerprint(load_track(dest)) != row['fingerprint']:
                raise ValueError(f'published serialization drift: {dest}')
            record = deepcopy(row); record['path'] = str(dest.relative_to(ROOT))
            record['cells'] = sorted(requirement_cells(t)); record['qualification'] = evidence
            record['source_metadata'] = t.metadata
            records.append(record)
            active.append(dict(name=t.name, track=record['path'], geometry_fingerprint=row['fingerprint'],
                exposure=row['prior_exposure'], family=row['family']))
        stem = 'v6_22_'+split.replace('-', '_')
        suite_path = ROOT/'configs/eval'/f'{stem}.yaml'
        if suite_path.exists(): raise FileExistsError('Frozen suite exists; publish a new version instead.')
        suite_path.write_text(yaml.safe_dump(dict(schema='starscream-real-course-suite-v1',
            description=f'{split}: shared public anchors and independently generated courses; see manifest exposure policy',
            active=active, pending_geometry=[]), sort_keys=False))
        atomic_json(ROOT/'configs/eval'/f'{stem}.manifest.json', dict(schema='starscream-v622-benchmark-v1',
            records=records, teacher_contract=contract_payload,
            qualification_contract_sha256=contract, reference_search_contract_sha256=search_contract,
            benchmark_admission_contract_sha256=benchmark_contract, benchmark_admission_contract=benchmark_payload,
            validation_scope='Headless Flightmare dynamics, directed aperture crossings, ground-contact termination and MPCC cohorts; no swept-rotor gate-frame clearance certification or official leaderboard timing claim.',
            source_mix=dict(Counter(r['source'] for r in records)),
            family_counts=dict(Counter(r['family'] for r in records)),
            independent_course_witnesses=dict(Counter(c for r in records for c in r['cells'])),
            training_distribution_design_allowed=split == 'real100-hard',
            evaluation_protocol=dict(version='v622-randomized-canonical-v1',
                randomized_environment=True, matched_track_seeds=True,
                episodes_per_course=32, seed=62422001, max_steps=6000,
                separate_generated_and_public_reference_metrics=True,
                checkpoint_selection_on_real60_allowed=False),
            exposure_policy='Public references are previously exposed. Only fresh real60 geometry is protected from future training design and checkpoint selection.'))
        print('PUBLISHED-LOCALLY', suite_path, count, flush=True)
    status['release_ready'] = True
    atomic_json(OUT/'release-status.json', status)


if __name__ == '__main__':
    main()
