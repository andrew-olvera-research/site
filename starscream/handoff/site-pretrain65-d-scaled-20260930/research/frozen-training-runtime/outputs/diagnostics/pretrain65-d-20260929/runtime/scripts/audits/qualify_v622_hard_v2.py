"""Run the established MPCC/Flightmare admission protocol on hard-v2 replacements."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint
from starscream.env.tracks import load_track
from scripts.audits.build_v622_hard_v2 import OUT, ROOT
from scripts.audits.qualify_v622_benchmark_protocol import job, protocol_contract
from scripts.audits.build_v622_benchmarks import OUT as BASE_OUT


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/eval/v6_22_real100_hard_v2.manifest.json').read_text())
    cfg, contract, payload = protocol_contract()
    rows = [r for r in manifest['records'] if r.get('qualification_pending')]
    rows.sort(key=lambda r: r['slot'])
    progress = {'contract': contract, 'attempted': {}, 'qualified': [], 'pending': [r['name'] for r in rows]}
    atomic_json(OUT / 'qualification-progress.json', progress)
    # The protocol job owns the canonical benchmark-qualification cache.  It
    # is run in the container as root, then copied into the user-writable v2
    # artifact directory for self-contained review.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             max_tasks_per_child=1) as pool:
        futures = {pool.submit(job, (row, cfg, contract, payload['base'])): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            result = future.result()
            progress['attempted'][row['name']] = result
            if result.get('qualified'):
                progress['qualified'].append(row['name'])
            progress['pending'] = [n for n in progress['pending'] if n != row['name']]
            atomic_json(OUT / 'qualification-progress.json', progress)
            print('ADMISSION', row['name'], result.get('qualified'),
                  'qualified', len(progress['qualified']), '/', len(rows), flush=True)
    # Copy canonical evidence into the v2 directory without changing its
    # provenance or the frozen v6.22 baseline.
    for row in rows:
        src = BASE_OUT / 'benchmark-qualification' / row['name'] / contract[:12] / 'result.json'
        if not src.exists():
            raise FileNotFoundError(src)
        dst = OUT / 'qualification' / row['name'] / contract[:12] / 'result.json'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if json.loads(dst.read_text())['fingerprint'] != geometry_fingerprint(load_track(ROOT / row['path'])):
            raise ValueError(f'qualification fingerprint drift: {row["name"]}')
    atomic_json(OUT / 'qualification-summary.json', dict(contract=contract,
        total=len(rows), qualified=len(progress['qualified']),
        failed=sorted(set(progress['pending']) | {n for n, r in progress['attempted'].items() if not r.get('qualified')}),
        source='scripts/audits/qualify_v622_benchmark_protocol.py'))
    print(json.dumps(dict(total=len(rows), qualified=len(progress['qualified']),
                          failed=len(rows)-len(progress['qualified']), contract=contract), indent=2))
    if len(progress['qualified']) != len(rows):
        raise SystemExit('hard-v2 candidate is not release-ready: admission failures')


if __name__ == '__main__':
    main()
