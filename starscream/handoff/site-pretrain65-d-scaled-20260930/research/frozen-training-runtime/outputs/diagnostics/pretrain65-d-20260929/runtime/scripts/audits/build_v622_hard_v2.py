"""Build a behavior-programmed replacement panel for real100-hard.

The released v6.22 suites are immutable.  This script starts from the frozen
real100-hard manifest, identifies the policy-easy/real60-near cells, replaces
only those slots with courses generated from independent behavior programs,
and materializes a versioned hard-v2 candidate.  Geometry is admitted without
using policy scores for candidate selection; the old policy scores are used
only to define which old slots are being repaired.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint, save_track_yaml
from starscream.env.racing_manifold.benchmark_hard_v2 import STRATA, generate_hard_v2
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex, requirement_cells, validate_geometry
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track
from scripts.audits.build_v622_benchmarks import ROOT, teacher_contract
from scripts.audits.qualify_v622_benchmark_protocol import protocol_contract
from scripts.audits.review_v622_geometry import resampled

BASE_MANIFEST = ROOT / 'configs/eval/v6_22_real100_hard.manifest.json'
REAL60_MANIFEST = ROOT / 'configs/eval/v6_22_real60.manifest.json'
RL_EVAL = ROOT / 'outputs/evals/v622-frozen/rl74-real100-hard-8ep.json'
CLOSENESS = ROOT / 'outputs/course-pools/v622-benchmarks/split-closeness.json'
# Existing course-pools artifacts were materialized by the container as root;
# keep this candidate in a user-writable sibling while preserving the same
# repository-relative provenance in its manifest.
OUT = ROOT / 'outputs/v622-hard-v2'
SUITE_STEM = 'v6_22_real100_hard_v2'


def _cell_jaccard(a, b):
    u = len(a | b)
    return 0.0 if not u else len(a & b) / u


def _metric_rows(manifest, metrics, closeness):
    out = []
    near = {r['name']: r for r in closeness['nearest_hard_to_real60']['records']}
    for row in manifest['records']:
        if row['suite'] != 'real100-hard':
            continue
        name = row['name']
        out.append(dict(row, policy_success=float(metrics.get(f'track/{name}/full_course_success', 0.0)),
                        nearest_real60_distance=float(near.get(name, {}).get('distance', 9.0)),
                        nearest_real60=row.get('nearest_real60', near.get(name, {}).get('nearest_real60'))))
    return out


def _protected_tracks():
    """Use all real60 geometry plus retained hard geometry as the novelty bank."""
    tracks = []
    for path in (REAL60_MANIFEST, BASE_MANIFEST):
        data = json.loads(path.read_text())
        tracks.extend(load_track(ROOT / r['path']) for r in data['records']
                      if r['suite'] != 'public_reference')
    # Public anchors are also protected from accidental duplication, although
    # they do not count as fresh generated slots.
    data = json.loads(BASE_MANIFEST.read_text())
    tracks.extend(load_track(ROOT / r['path']) for r in data['records']
                   if r['suite'] == 'public_reference')
    return tracks


def _choose_replacements(rows, limit=30):
    # The failure mode is precisely the policy-easy/real60-near tail.  Sort by
    # completion first, then normalized shape distance; this keeps the repair
    # criterion explicit and reproducible rather than hand-picked by family.
    # High completion is the first signal for "not hard enough"; among those,
    # repair the courses that are also nearest to real60. (The previous draft
    # accidentally sorted ascending and selected already-hard failures.)
    rows = sorted(rows, key=lambda r: (-r['policy_success'], r['nearest_real60_distance'], r['name']))
    return rows[:limit]


def _program_assignment(index):
    # Six strata would underrepresent the long-low/vertical extremes.  The
    # 30-slot panel is deliberately weighted 8/6/6/5/5 and interleaved in
    # slot order so no family is mapped to one contiguous section.
    weights = [8, 6, 6, 5, 5]
    plan = []
    for stratum, count in zip(STRATA, weights):
        plan.extend([stratum] * count)
    return plan[index]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE_MANIFEST.read_text())
    real60 = json.loads(REAL60_MANIFEST.read_text())
    metrics = json.loads(RL_EVAL.read_text())['metrics']
    closeness = json.loads(CLOSENESS.read_text())
    generated = _metric_rows(base, metrics, closeness)
    targets = _choose_replacements(generated, 30)
    target_names = {r['name'] for r in targets}

    # The novelty bank includes every untouched hard course. Candidate search
    # uses 64-point arc-length shapes so differing gate counts cannot evade the
    # split check. New candidates are required to clear a stricter .12 shape
    # distance and a .06 resampled distance (the frozen release used .06).
    protected = _protected_tracks()
    exact = CloneIndex(protected)
    continuous = CloneIndex([resampled(t) for t in protected])
    retained = [load_track(ROOT / r['path']) for r in base['records']
                if r['suite'] != 'public_reference' and r['name'] not in target_names]
    exact_retained = CloneIndex(retained)
    continuous_retained = CloneIndex([resampled(t) for t in retained])
    candidate_rows = []
    selected = {}
    candidate_records = []
    candidate_dir = OUT / 'candidates'
    candidate_dir.mkdir(parents=True, exist_ok=True)

    for i, old in enumerate(targets):
        stratum = _program_assignment(i)
        old_track = load_track(ROOT / old['path'])
        n = int(np.clip(len(old_track.gates) + ((i * 7) % 5) - 2, 8, 18))
        choices = []
        for trial in range(24):
            seed = 72200000 + i * 1000 + trial
            # Include the realization revision in the name so an admission
            # cache can never silently reuse evidence for an earlier, softer
            # or failed geometry revision.
            name = f'v622_hard_v2b_{stratum}_{i:03d}_{trial:02d}'
            track = generate_hard_v2(stratum, n, seed, name)
            static = validate_geometry(track)
            if static:
                continue
            xdist = exact.distance(track)
            cdist = continuous.distance(resampled(track))
            # Compare requirement cells against both real60 and the retained
            # hard bank. This is a diagnostic, not a policy-derived score.
            cells = requirement_cells(track)
            old_cells = set(old.get('cells', []))
            real60_cells = set(c for r in real60['records'] for c in r.get('cells', []))
            j60 = _cell_jaccard(cells, real60_cells)
            jold = _cell_jaccard(cells, old_cells)
            candidate_records.append(dict(name=name, slot=old['slot'], old_name=old['name'],
                stratum=stratum, seed=seed, gate_count=len(track.gates),
                exact_distance=xdist, continuous_distance=cdist,
                real60_cell_jaccard=j60, old_cell_jaccard=jold, static_reasons=static,
                geometry=geometry_record(track), behavior_signature=track.metadata['behavior_signature']))
            if xdist < .12 or cdist < .06:
                continue
            # Prefer large distance and low overlap, but keep a small
            # gate-count penalty so a single 18-gate outlier does not dominate
            # the replacement panel.
            objective = cdist + .35 * xdist + .25 * (1. - j60) + .10 * min(1., abs(n - len(old_track.gates))/8.)
            choices.append((objective, track, xdist, cdist, j60, jold))
        if not choices:
            raise RuntimeError(f'no independent realization for {old["name"]}')
        choices.sort(key=lambda x: (-x[0], x[1].name))
        _, track, xdist, cdist, j60, jold = choices[0]
        dest = candidate_dir / f'{track.name}.yaml'
        save_track_yaml(track, dest)
        selected[old['name']] = dict(old_name=old['name'], old_slot=old['slot'],
            old_family=old['family'], old_policy_success=old['policy_success'],
            old_nearest_real60_distance=old['nearest_real60_distance'],
            name=track.name, path=str(dest.relative_to(ROOT)), family=f'behavior:{stratum}',
            source='v622-hard-v2-behavior-program', stratum=stratum,
            seed=int(track.metadata['behavior_signature']['program_seed']),
            fingerprint=geometry_fingerprint(track), cells=sorted(requirement_cells(track)),
            exact_distance=float(xdist), continuous_distance=float(cdist),
            real60_cell_jaccard=float(j60), old_cell_jaccard=float(jold),
            behavior_signature=track.metadata['behavior_signature'])
        exact.add(track); continuous.add(resampled(track))
        candidate_rows.append(selected[old['name']])

    # Compose v2 in the exact frozen order. Public anchors are carried over
    # unchanged and remain explicitly labelled; only the 30 weak slots change.
    replacements = {k: v for k, v in selected.items()}
    records, active = [], []
    for old in base['records']:
        if old['name'] in replacements:
            new = deepcopy(replacements[old['name']])
            new.update(suite='real100-hard-v2', slot=old['slot'], rank=old.get('rank', 0),
                       prior_exposure='fresh geometry', qualification_pending=True)
            records.append(new)
            active.append(dict(name=new['name'], track=new['path'],
                               geometry_fingerprint=new['fingerprint'],
                               exposure='fresh geometry', family=new['family']))
        else:
            new = deepcopy(old)
            if new['suite'] == 'real100-hard': new['suite'] = 'real100-hard-v2'
            records.append(new)
            active.append(dict(name=new['name'], track=new['path'],
                               geometry_fingerprint=new['fingerprint'],
                               exposure=new.get('prior_exposure', 'public reference'), family=new['family']))
    assert len(records) == 100 and len(selected) == 30
    cfg = dict(schema='starscream-real-course-suite-v1',
               description='real100-hard-v2: 30 policy-easy/real60-near slots replaced by independent behavior-program courses; v6.22 hard baseline is frozen.',
               active=active, pending_geometry=[])
    (ROOT / 'configs/eval' / f'{SUITE_STEM}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    cfg_contract, benchmark_contract, benchmark_payload = protocol_contract()
    teacher_cfg, teacher_sha, teacher_payload = teacher_contract()
    manifest = dict(schema='starscream-v622-benchmark-v2-candidate', records=records,
        teacher_contract=teacher_payload, qualification_contract_sha256=benchmark_contract,
        benchmark_admission_contract_sha256=benchmark_contract,
        benchmark_admission_contract=benchmark_payload,
        validation_scope='Headless Flightmare dynamics, directed aperture crossings, ground-contact termination and MPCC cohorts; candidate pending final qualification.',
        source_mix=dict(Counter(r.get('source', 'frozen-v622') for r in records)),
        family_counts=dict(Counter(r['family'] for r in records)),
        independent_course_witnesses=dict(Counter(c for r in records for c in r.get('cells', []))),
        training_distribution_design_allowed=False,
        evaluation_protocol=dict(version='v622-randomized-canonical-v1', episodes_per_course=32,
            seed=62422001, max_steps=6000, separate_generated_and_public_reference_metrics=True,
            checkpoint_selection_on_real60_allowed=False),
        exposure_policy='Versioned hard-v2 candidate; not eligible for training or checkpoint selection until qualification and review.',
        replacement_policy=dict(replaced_count=30, selection='lowest RL completion then nearest real60 normalized shape distance',
            strata=dict(Counter(v['stratum'] for v in selected.values())),
            old_baseline='configs/eval/v6_22_real100_hard.manifest.json',
            protected_real60='configs/eval/v6_22_real60.manifest.json'))
    atomic_json(ROOT / 'configs/eval' / f'{SUITE_STEM}.manifest.json', manifest)
    atomic_json(OUT / 'candidate-records.json', dict(records=candidate_records, selected=list(selected.values())))
    atomic_json(OUT / 'replacement-ledger.json', dict(targets=targets, replacements=list(selected.values())))
    print(json.dumps(dict(replaced=len(selected), strata=Counter(v['stratum'] for v in selected.values()),
                          suite=str(ROOT / 'configs/eval' / f'{SUITE_STEM}.yaml'),
                          manifest=str(ROOT / 'configs/eval' / f'{SUITE_STEM}.manifest.json')), indent=2))


if __name__ == '__main__':
    main()
