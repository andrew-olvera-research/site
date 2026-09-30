"""Regression, simulator, resume and frozen-runtime validation for next screen."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace')
OUT = ROOT/'outputs/diagnostics/collection-next-20260929'
sys.path.insert(0, str(ROOT))

def main():
    import h5py
    import numpy as np
    import torch
    from starscream.dagger_replay import DaggerReplayStore
    report = dict(state='running', pid=os.getpid(), started=time.time(), smoke={})
    def save(): (OUT/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        tests = ['dagger_collection', 'collection_next', 'dagger_quality', 'multifix', 'dagger_replay',
                 'dagger_transport', 'dagger_async', 'dagger_schedule', 'native_reference', 'dagger_update_graph']
        with (OUT/'tests.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'pytest', *[f'tests/test_{x}.py' for x in tests],
                '-q', '--disable-warnings'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report['tests'] = (OUT/'tests.log').read_text().strip()
        for arm in ('full_learner','time_2s','full_expert20','recovery_4s'):
            path = OUT/f'smoke-{arm}.json'
            cfg = json.loads(path.read_text())
            latest = OUT/'smoke/checkpoints'/cfg['dagger']['run_name']/'latest.pt'
            for phase in ('initial','resume'):
                if phase == 'resume':
                    cfg['dagger'].update(rounds=4, resume_checkpoint=str(latest))
                    path = OUT/f'smoke-{arm}-resume.json'
                    path.write_text(json.dumps(cfg, indent=2)+'\n')
                with (OUT/f'smoke-{arm}-{phase}.log').open('w') as log:
                    subprocess.run([sys.executable, '-u', 'scripts/train_privileged_racing.py',
                        '--config', str(path), '--stage','dagger','--device','cuda'], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
            payload = torch.load(latest, map_location='cpu', weights_only=False)
            assert payload['round'] == 4
            metadata = payload['dagger_replay']
            assert metadata['contract']['collection_control_v1'] == cfg['dagger']['dagger_collection_control']
            store = DaggerReplayStore.create(OUT/f'restore-{arm}', metadata['contract'], resume_metadata=metadata)
            shards = store._committed_shards(4)
            with h5py.File(shards[-1]) as h:
                templates = {k:np.empty((0,*v.shape[1:]), v.dtype) for k,v in h['online'].items()}
            actual = store.restore_pool('online', templates, capacity=800, committed_round=4)
            for key in templates:
                chunks=[]
                for shard in shards:
                    with h5py.File(shard) as h: chunks.append(h['online'][key][:])
                np.testing.assert_array_equal(actual[key], np.concatenate(chunks)[-800:])
            report['smoke'][arm] = dict(round=4, arrays=len(templates), restored_rows=len(actual['trajectory']), steps=payload['environment_steps'])
            save()
        runtime = OUT/'runtime'
        for folder in ('scripts','starscream'):
            shutil.copytree(ROOT/folder, runtime/folder, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        hashes = {str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in runtime.rglob('*') if p.is_file()}
        (OUT/'runtime-sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
        jobs=json.loads((OUT/'jobs.json').read_text())
        report['config_sha256']={j['config']:hashlib.sha256(Path(j['config']).read_bytes()).hexdigest() for j in jobs}
        report.update(state='passed', finished=time.time())
        save()
    except BaseException as exc:
        report.update(state='failed', error=repr(exc), finished=time.time()); save(); raise

if __name__ == '__main__': main()
