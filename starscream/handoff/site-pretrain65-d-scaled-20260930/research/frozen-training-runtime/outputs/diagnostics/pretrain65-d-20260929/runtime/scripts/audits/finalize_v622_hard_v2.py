"""Finalize the qualified hard-v2 candidate without touching frozen suites."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex, requirement_cells, validate_geometry
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track
from scripts.audits.build_v622_hard_v2 import ROOT, OUT
from scripts.audits.review_v622_geometry import resampled


def main():
    path = ROOT / 'configs/eval/v6_22_real100_hard_v2.manifest.json'
    manifest = json.loads(path.read_text())
    summary = json.loads((OUT / 'qualification-summary.json').read_text())
    if summary['qualified'] != summary['total'] or summary['failed']:
        raise SystemExit('qualification is incomplete')
    contract = summary['contract']
    generated, references = [], []
    for row in manifest['records']:
        track = load_track(ROOT / row['path'])
        if geometry_fingerprint(track) != row['fingerprint']:
            raise ValueError(f'fingerprint drift: {row["name"]}')
        if row['suite'] == 'public_reference':
            references.append(track); continue
        if validate_geometry(track):
            raise ValueError(f'geometry invalid: {row["name"]}')
        generated.append(track)
        if row.get('qualification_pending'):
            evidence_path = OUT / 'qualification' / row['name'] / contract[:12] / 'result.json'
            evidence = json.loads(evidence_path.read_text())
            if not evidence.get('qualified'):
                raise ValueError(f'qualification failed: {row["name"]}')
            row['qualification'] = evidence
            row.pop('qualification_pending', None)
        row['cells'] = sorted(requirement_cells(track))
        row['geometry'] = geometry_record(track)
        row['source_metadata'] = track.metadata
    if len(generated) != 92 or len(references) != 8:
        raise ValueError('portfolio count drift')
    # A final all-pair novelty pass catches accidental collisions introduced by
    # the two edge-case repairs, independently of the candidate ledger.
    index = CloneIndex()
    continuous = CloneIndex()
    minima = []
    for track in generated:
        minima.append(dict(name=track.name, exact_distance=index.distance(track),
                           continuous_distance=continuous.distance(resampled(track))))
        if index.distance(track) < .12 or continuous.distance(resampled(track)) < .06:
            raise ValueError(f'final clone threshold failed: {track.name}')
        index.add(track); continuous.add(resampled(track))
    manifest['schema'] = 'starscream-v622-benchmark-v2'
    manifest['training_distribution_design_allowed'] = False
    manifest['exposure_policy'] = 'Independent hard-v2 challenge candidate. Frozen real60 and v6.22 hard remain unchanged; do not use this suite for training or checkpoint selection until explicitly promoted.'
    manifest['release_status'] = dict(qualified_replacements=92, public_references=8,
        all_replacements_admitted=True, admission_contract=contract,
        clone_thresholds=dict(exact=.12, continuous=.06),
        generated_novelty=dict(min_exact=min(x['exact_distance'] for x in minima),
                               min_continuous=min(x['continuous_distance'] for x in minima)),
        source='scripts/audits/finalize_v622_hard_v2.py')
    atomic_json(path, manifest)
    atomic_json(OUT / 'final-novelty.json', dict(records=minima, generated=len(generated), references=len(references)))
    atomic_json(OUT / 'release-status.json', manifest['release_status'])
    print(json.dumps(manifest['release_status'], indent=2))


if __name__ == '__main__': main()
