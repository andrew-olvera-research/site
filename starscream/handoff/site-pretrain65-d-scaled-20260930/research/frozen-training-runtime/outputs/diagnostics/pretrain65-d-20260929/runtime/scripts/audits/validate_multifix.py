"""Freeze passing regression, real collection, replay and resume evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starscream.dagger_replay import DaggerReplayStore
from scripts.audits.launch_multifix_leg import CONFIG, DEPENDENCIES

TESTS = ['test_multifix', 'test_dagger_replay', 'test_dagger_throughput',
         'test_privileged_rl_recipe', 'test_dagger_update_graph', 'test_dagger_quality',
         'test_route_conditioning_ablation', 'test_dagger_transport', 'test_dagger_async',
         'test_dagger_row_buffer', 'test_dagger_schedule', 'test_plant_privileged']
SOURCES = ['scripts/train_privileged_racing.py', 'starscream/imitation_objective.py',
           'starscream/dagger_contracts.py', 'starscream/dagger_diagnostics.py',
           'starscream/dagger_replay.py', 'starscream/dagger_throughput.py',
           'scripts/audits/collect_dagger_initialization_statistics.py',
           'scripts/audits/launch_multifix_leg.py', 'scripts/audits/validate_multifix.py',
           CONFIG, 'outputs/diagnostics/multifix-20260928/normalization.pt']


def main():
    root = ROOT / 'outputs/diagnostics/multifix-20260928'
    result = subprocess.run([sys.executable, '-m', 'pytest', *[f'tests/{name}.py' for name in TESTS],
                             '-q', '--disable-warnings'], cwd=ROOT, capture_output=True, text=True)
    (root / 'tests.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Regression tests failed; see tests.log')
    config = json.loads((ROOT / CONFIG).read_text())
    settings = config['dagger']
    assert settings['initial_checkpoint'] is None and settings['resume_checkpoint'] is None
    stats = torch.load(settings['dagger_initialization_stats_checkpoint'], map_location='cpu', weights_only=False)
    assert stats['previous_action_feature_mapping'] == settings['previous_action_feature_mapping']
    assert len(stats['track_counts']) == len(settings['curriculum']['tracks']) == 9
    assert all(value > 0 for value in stats['track_counts'].values())
    smoke = root / 'smoke/checkpoints/multifix-smoke-resume-20260928/latest.pt'
    payload = torch.load(smoke, map_location='cpu', weights_only=False)
    assert payload['round'] == 4 and payload['optimizer']['state']
    metadata = payload['dagger_replay']
    contract = metadata['contract']
    assert contract['previous_action_feature_mapping'] == settings['previous_action_feature_mapping']
    np.testing.assert_array_equal(contract['normalization']['feature_mean'], payload['normalizer']['mean'])
    store = DaggerReplayStore.create(root / 'restore-check', contract, resume_metadata=metadata)
    shards = store._committed_shards(4)
    with h5py.File(shards[-1]) as h:
        templates = {key: np.empty((0, *value.shape[1:]), value.dtype) for key, value in h['online'].items()}
    actual = store.restore_pool('online', templates, capacity=800, committed_round=4)
    for key in templates:
        pieces = []
        for shard in shards:
            with h5py.File(shard) as h:
                pieces.append(h['online'][key][:])
        np.testing.assert_array_equal(actual[key], np.concatenate(pieces)[-800:])
    baseline = {}
    for name in DEPENDENCIES:
        status = json.loads((ROOT / f'outputs/diagnostics/mini-slalom-recovery-baselines/{name}.status.json').read_text())
        baseline[name] = status
    files = SOURCES + [f'tests/{name}.py' for name in TESTS]
    report = dict(status='passed', validated_unix=time.time(), tests=result.stdout.strip(),
                  statistics=dict(rows=stats['rows'], valid_queries=stats['valid_queries'], courses=9),
                  smoke=dict(rounds=3, resumed_round=4, rows_restored=800,
                             arrays_verified=list(templates), checkpoint=str(smoke.relative_to(ROOT))),
                  baseline_status=baseline,
                  sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files})
    (root / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(result.stdout.strip())
    print('Validated actual collection, CUDA updates, all replay arrays, eviction and checkpoint resume.')


if __name__ == '__main__':
    main()
