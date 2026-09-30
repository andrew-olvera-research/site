"""Validate regressions, actual collector/update/resume evidence, and freeze B-E."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starscream.dagger_collection import collection_config
from starscream.dagger_replay import DaggerReplayStore

OUT = ROOT / 'outputs/diagnostics/collection-ablation-20260929'
TESTS = ['dagger_collection', 'dagger_quality', 'multifix', 'dagger_replay',
         'dagger_throughput', 'dagger_transport', 'dagger_async', 'dagger_schedule',
         'native_reference', 'privileged_rl_recipe', 'dagger_update_graph',
         'route_conditioning_ablation', 'plant_privileged']


def main():
    result = subprocess.run([sys.executable, '-m', 'pytest',
        *[f'tests/test_{name}.py' for name in TESTS], '-q', '--disable-warnings'],
        cwd=ROOT, capture_output=True, text=True)
    (OUT / 'tests.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Regression failure; see tests.log')
    baseline_path = ROOT / 'configs/exp/v6.21.1.1/mini_slalom_aug2_fixes.yaml'
    baseline = json.loads(baseline_path.read_text())['dagger']
    allowed = {'run_name', 'tags', 'mpcc_build_root', 'dagger_collection_control'}
    report = dict(tests=result.stdout.strip(), arms={}, comparison={}, validated_unix=time.time())
    paths = [baseline_path, Path(baseline['dagger_initialization_stats_checkpoint'])]
    builds = set()
    for arm in 'bcde':
        config_path = ROOT / f'configs/exp/v6.21.1.1/mini_slalom_aug2_collection_{arm}.yaml'
        settings = json.loads(config_path.read_text())['dagger']
        collection_config(settings)
        difference = {k: {'a': baseline.get(k), arm: settings.get(k)}
            for k in baseline.keys() | settings.keys() if baseline.get(k) != settings.get(k)}
        if set(difference) - allowed:
            raise RuntimeError(f'Unexpected method confound: {arm}: {set(difference)-allowed}')
        assert settings['initial_checkpoint'] is None and settings['resume_checkpoint'] is None
        builds.add(settings['mpcc_build_root'])
        report['comparison'][arm] = difference
        payload = torch.load(OUT / f'smoke/checkpoints/collection-smoke-{arm}/latest.pt',
                             map_location='cpu', weights_only=False)
        assert payload['round'] == 5 and payload['optimizer']['state']
        metadata = payload['dagger_replay']
        assert metadata['contract']['collection_control_v1'] == settings['dagger_collection_control']
        store = DaggerReplayStore.create(OUT / f'restore-check-{arm}', metadata['contract'], resume_metadata=metadata)
        shards = store._committed_shards(5)
        with h5py.File(shards[-1]) as h:
            templates = {k: np.empty((0, *v.shape[1:]), v.dtype) for k,v in h['online'].items()}
        actual = store.restore_pool('online', templates, capacity=800, committed_round=5)
        for key in templates:
            chunks = []
            for path in shards:
                with h5py.File(path) as h:
                    chunks.append(h['online'][key][:])
            np.testing.assert_array_equal(actual[key], np.concatenate(chunks)[-800:])
        assert actual['trajectory'].shape == (800, 18)
        assert np.isfinite(actual['trajectory']).all()
        events = [json.loads(l)['metrics'] for l in (OUT / f'collection-smoke-{arm}.full.events.jsonl').read_text().splitlines()]
        metrics = [e for e in events if e.get('train/round') == 5][-1]
        counts = {k.removeprefix('train/collection/'):v for k,v in metrics.items() if k.startswith('train/collection/')}
        assert counts['active_steps'] > 0
        assert sum(counts[f'retained_{k}_rows'] for k in ('nominal','corrective','recovery')) == 400
        if arm in 'de':
            assert counts['active_teacher_steps'] == 0
        if arm == 'b':
            assert counts['recovery_steps'] > 0 and counts['learner_recovery_steps'] == 0
        if arm in 'ce':
            assert counts['recovery_steps'] > 0 and counts['learner_recovery_steps'] == counts['recovery_steps']
        report['arms'][arm] = dict(round=5, steps=payload['environment_steps'],
            restored_rows=800, verified_arrays=list(templates), collection=counts)
        paths.append(config_path)
    assert len(builds) == 4
    # Real plant segment truncation with a competent existing policy; this is
    # a validation probe, never an ablation result or initialization for B-E.
    report['handoff_probes'] = {}
    for arm in 'de':
        rows = [json.loads(l)['metrics'] for l in (OUT / f'collection-handoff-probe-{arm}.full.events.jsonl').read_text().splitlines()]
        metrics = [e for e in rows if e.get('train/round') == 2][-1]
        assert metrics['train/collection/segment_completed'] > 0
        assert metrics['train/collection/active_teacher_steps'] == 0
        report['handoff_probes'][arm] = {k:v for k,v in metrics.items() if k.startswith('train/collection/')}
    runtime = OUT / 'runtime'
    if runtime.exists():
        raise FileExistsError('Runtime already frozen; do not replace launch evidence')
    for folder in ('scripts', 'starscream'):
        shutil.copytree(ROOT / folder, runtime / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    hashes = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(runtime.rglob('*')) if p.is_file()}
    (OUT / 'runtime-sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    paths += [ROOT / 'scripts/train_privileged_racing.py', ROOT / 'starscream/dagger_collection.py',
              ROOT / 'scripts/audits/launch_collection_ablation.py', Path(__file__)]
    paths += [ROOT / f'tests/test_{name}.py' for name in TESTS]
    paths += [OUT / 'runtime-sha256.json']
    report['sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report['status'] = 'passed'
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(result.stdout.strip())
    print('Validated all four methods, actual segment handoffs, 5-round resume, and every replay array; runtime frozen.')


if __name__ == '__main__':
    main()
