"""Validate and freeze the missing bounded-with-fixes scratch control."""
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
OUT = ROOT / 'outputs/diagnostics/fixes-bounded-20260929'
sys.path.insert(0, str(ROOT))
from starscream.dagger_replay import DaggerReplayStore
from starscream.dagger_quality import quality_config

BASE = ROOT / 'configs/exp/v6.21.1.1/mini_slalom_aug2_fixes.yaml'
OLD = ROOT / 'configs/exp/v6.21.1.1/mini_slalom_aug2_scratch_bounded.yaml'
NEW = ROOT / 'configs/exp/v6.21.1.1/mini_slalom_aug2_fixes_bounded.yaml'
TESTS = ('test_dagger_quality', 'test_dagger_collection', 'test_dagger_replay',
         'test_dagger_throughput', 'test_dagger_transport', 'test_dagger_async',
         'test_native_reference', 'test_multifix', 'test_privileged_rl_recipe')


def main():
    a = json.loads(BASE.read_text())['dagger']
    old = json.loads(OLD.read_text())['dagger']
    b = json.loads(NEW.read_text())['dagger']
    if b['dagger_trajectory_quality'] != old['dagger_trajectory_quality']:
        raise RuntimeError('bounded package differs from completed bounded control')
    quality_config(b)
    allowed = {'dagger_trajectory_quality', 'run_name', 'tags', 'mpcc_build_root'}
    differences = {key:dict(a=a.get(key), bounded=b.get(key)) for key in a.keys() | b.keys()
                   if a.get(key) != b.get(key)}
    if set(differences) != allowed:
        raise RuntimeError(f'comparison has unexpected differences: {set(differences) - allowed}')
    if b['initial_checkpoint'] is not None or b['resume_checkpoint'] is not None:
        raise RuntimeError('bounded fixes control must start from scratch')
    stats = torch.load(b['dagger_initialization_stats_checkpoint'], map_location='cpu', weights_only=False)
    if stats['previous_action_feature_mapping'] != b['previous_action_feature_mapping']:
        raise RuntimeError('normalization and previous-action contract differ')
    if len(stats['track_counts']) != 9 or len(b['curriculum']['tracks']) != 9:
        raise RuntimeError('normalization course coverage differs')
    result = subprocess.run([sys.executable, '-m', 'pytest',
        *[f'tests/{name}.py' for name in TESTS], '-q', '--disable-warnings'],
        cwd=ROOT, capture_output=True, text=True)
    (OUT / 'tests.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('bounded/fixes regression tests failed')
    checkpoint = OUT / 'smoke/checkpoints/fixes-bounded-smoke-20260929/latest.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['round'] != 4 or not payload['optimizer']['state']:
        raise RuntimeError('real simulator smoke/resume checkpoint incomplete')
    metadata = payload['dagger_replay']
    if metadata['contract']['trajectory_quality_v1'] != b['dagger_trajectory_quality']:
        raise RuntimeError('bounded replay contract lost trajectory quality')
    store = DaggerReplayStore.create(OUT / 'restore-check', metadata['contract'], resume_metadata=metadata)
    shards = store._committed_shards(4)
    with h5py.File(shards[-1]) as h:
        templates = {key:np.empty((0,*value.shape[1:]),value.dtype) for key,value in h['online'].items()}
    restored = store.restore_pool('online', templates, capacity=800, committed_round=4)
    for key in templates:
        pieces=[]
        for shard in shards:
            with h5py.File(shard) as h:
                pieces.append(h['online'][key][:])
        np.testing.assert_array_equal(restored[key],np.concatenate(pieces)[-800:])
    if restored['trajectory'].shape != (800,18) or not np.isfinite(restored['trajectory']).all():
        raise RuntimeError('bounded replay lost finite trajectory telemetry')
    events = [json.loads(line)['metrics'] for line in (OUT/'smoke.full.events.jsonl').read_text().splitlines()]
    train = [event for event in events if 'train/round' in event]
    if len(train)!=4 or train[-1]['train/round']!=4:
        raise RuntimeError('smoke training telemetry incomplete')
    if train[-1].get('train/quality/collection/failed_segments',0)<=0:
        raise RuntimeError('bounded handoff path was not exercised')
    runtime = OUT/'runtime'
    if runtime.exists():
        raise FileExistsError('runtime already frozen')
    for folder in ('scripts','starscream'):
        shutil.copytree(ROOT/folder, runtime/folder,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    runtime_hashes = {str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(runtime.rglob('*')) if p.is_file()}
    (OUT/'runtime-sha256.json').write_text(json.dumps(runtime_hashes,indent=2)+'\n')
    paths=[BASE,OLD,NEW,Path(b['dagger_initialization_stats_checkpoint']),
           Path(b['dagger_trajectory_quality']['calibration_source']),
           ROOT/'scripts/train_privileged_racing.py',ROOT/'starscream/dagger_quality.py',
           ROOT/'starscream/dagger_replay.py',ROOT/'starscream/dagger_collection.py',
           ROOT/'scripts/audits/launch_fixes_bounded.py',Path(__file__),OUT/'runtime-sha256.json']
    paths += [ROOT/'tests'/f'{name}.py' for name in TESTS]
    report = dict(status='passed',validated_unix=time.time(),tests=result.stdout.strip(),
        difference_keys=sorted(differences),statistics=dict(rows=stats['rows'],courses=len(stats['track_counts'])),
        smoke=dict(round=4,environment_steps=payload['environment_steps'],restored_rows=800,
                   arrays_verified=list(templates),failed_segments=train[-1]['train/quality/collection/failed_segments']),
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(result.stdout.strip())
    print('Validated bounded package, actual collection/update/resume, replay arrays and frozen runtime.')


if __name__=='__main__':
    main()
