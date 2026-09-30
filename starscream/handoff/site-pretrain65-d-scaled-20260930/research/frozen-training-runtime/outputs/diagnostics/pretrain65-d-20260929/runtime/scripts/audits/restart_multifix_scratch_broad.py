"""One explicit repair of the broad control's diagnosed shared-build startup failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/diagnostics/multifix-scratch-controls-20260928'
STATUS = ROOT / 'outputs/diagnostics/mini-slalom-recovery-baselines'


def save(path, value):
    temp = path.with_suffix('.repair.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    name = 'mini_slalom_aug2_scratch_broad'
    path = STATUS / f'{name}.status.json'
    log = ROOT / f'outputs/logs/{name}.console.log'
    config = ROOT / f'configs/exp/v6.21.1.1/{name}.yaml'
    if path.exists() or log.exists():
        raise FileExistsError('Repaired broad control already reserved')
    c = json.loads(config.read_text())
    assert c['dagger']['mpcc_build_root'] == '/tmp/starscream-mini-slalom-scratch-broad-20260928'
    runtime = OUT / 'baseline-runtime'
    run = dict(name=name, state='launching', config=str(config.relative_to(ROOT)),
               log=str(log.relative_to(ROOT)), runtime=str(runtime), started_unix=time.time(),
               repair='Isolated compiler output; no completed training rounds discarded')
    # Exclusive reservation prevents duplicate repair launches.
    with path.open('x') as stream:
        json.dump(run, stream)
    with log.open('x') as stream:
        p = subprocess.Popen([sys.executable, '-u', str(runtime / 'scripts/train_privileged_racing.py'),
            '--config', str(config), '--stage', 'dagger', '--device', 'cuda'], cwd=ROOT,
            env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                 'OPENBLAS_NUM_THREADS': '1', 'PYTHONPATH': str(runtime)},
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        run.update(state='running', pid=p.pid); save(path, run)
        code = p.wait()
    run.update(state='complete' if code == 0 else 'failed', exit_code=code, finished_unix=time.time())
    save(path, run)
    # The original pair coordinator owns bounded. Reconcile its historical
    # startup failure only after it has finished writing the bounded status.
    bounded = STATUS / 'mini_slalom_aug2_scratch_bounded.status.json'
    while json.loads(bounded.read_text())['state'] in {'running', 'launching'}:
        time.sleep(10)
    states = [json.loads(p.read_text()) for p in [path, bounded]]
    save(OUT / 'pair.status.json', dict(state='complete' if all(s.get('exit_code') == 0 for s in states) else 'failed',
         finished_unix=time.time(), runs=states, startup_repairs=['explicit W&B env path', 'isolated broad compiler directory']))
    return code


if __name__ == '__main__':
    sys.exit(main())
