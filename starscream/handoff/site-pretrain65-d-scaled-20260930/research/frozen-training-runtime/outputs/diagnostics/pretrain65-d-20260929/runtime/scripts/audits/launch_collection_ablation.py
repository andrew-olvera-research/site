"""One-shot B/C then D/E execution. No monitoring automation or retries.

Each child wrapper records its own exit status. The queue waits for both child
wrappers to exit successfully before starting the next pair. All artifacts are
reserved once; a failure stops the queue and preserves the evidence.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace')
OUT = ROOT / 'outputs/diagnostics/collection-ablation-20260929'
RUNTIME = OUT / 'runtime'


def save(path, payload):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(path)


def verify():
    report = json.loads((OUT / 'validation.json').read_text())
    if report['status'] != 'passed':
        raise RuntimeError('Collection validation has not passed')
    for name, digest in report['sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Validated artifact changed: {name}')
    for name, digest in json.loads((OUT / 'runtime-sha256.json').read_text()).items():
        if hashlib.sha256((RUNTIME / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen runtime changed: {name}')


def run_arm(arm):
    path = OUT / f'{arm}.status.json'
    status = json.loads(path.read_text())
    if status['state'] != 'queued':
        raise RuntimeError(f'Arm {arm} is already reserved/running')
    environment = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                   'OPENBLAS_NUM_THREADS': '1', 'PYTHONPATH': str(RUNTIME), 'PYTHONUNBUFFERED': '1'}
    try:
        with (ROOT / status['log']).open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', str(RUNTIME / 'scripts/train_privileged_racing.py'),
                '--config', str(ROOT / status['config']), '--stage', 'dagger', '--device', 'cuda'],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        status.update(state='running', pid=child.pid, wrapper_pid=os.getpid(), started_unix=time.time())
        save(path, status)
        code = child.wait()
        if code == 0:
            import torch
            payload = torch.load(ROOT / status['checkpoint'], map_location='cpu', weights_only=False)
            if payload['round'] != 24:
                raise RuntimeError(f'Arm {arm} exited without completing round 24')
            status.update(round=24, environment_steps=payload['environment_steps'])
        status.update(state='complete' if code == 0 else 'failed', exit_code=code, finished_unix=time.time())
        save(path, status)
        return code
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), finished_unix=time.time())
        save(path, status)
        raise


def main():
    os.chdir(ROOT)
    if len(sys.argv) == 3 and sys.argv[1] == '--arm' and sys.argv[2] in 'bcde':
        return run_arm(sys.argv[2])
    with (OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = OUT / 'queue.status.json'
        if state_path.exists():
            raise FileExistsError('Collection queue already reserved; inspect its evidence')
        verify()
        # Only race-training processes count; do not match this launcher's name.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                args = (proc / 'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if any(a.endswith(b'/train_privileged_racing.py') or a == b'train_privileged_racing.py' for a in args):
                raise RuntimeError(f'Existing racing trainer PID {proc.name}; queue not started')
        jobs = {}
        for arm in 'bcde':
            config = f'configs/exp/v6.21.1.1/mini_slalom_aug2_collection_{arm}.yaml'
            c = json.loads((ROOT / config).read_text())
            name = c['dagger']['run_name']
            checkpoint = f'outputs/checkpoints/{name}/latest.pt'
            log = f'outputs/logs/mini_slalom_aug2_collection_{arm}.console.log'
            if ((OUT / f'{arm}.status.json').exists() or (ROOT / log).exists()
                    or (ROOT / checkpoint).parent.exists()):
                raise FileExistsError(f'Arm {arm} already has artifacts')
            if c['dagger']['initial_checkpoint'] or c['dagger']['resume_checkpoint']:
                raise ValueError('These arms must start from scratch')
            jobs[arm] = dict(arm=arm, state='queued', config=config, log=log, checkpoint=checkpoint,
                             runtime=str(RUNTIME))
        state = dict(state='running', queue_pid=os.getpid(), started_unix=time.time(), pairs=['bc','de'])
        save(state_path, state)
        for arm, job in jobs.items():
            save(OUT / f'{arm}.status.json', job)
        try:
            for pair in ('bc', 'de'):
                verify()
                state.update(active_pair=pair)
                save(state_path, state)
                children = [subprocess.Popen([sys.executable, '-u', __file__, '--arm', arm], cwd=ROOT) for arm in pair]
                codes = [child.wait() for child in children]
                if any(codes):
                    state.update(state='failed', failed_pair=pair, exit_codes=codes, finished_unix=time.time())
                    save(state_path, state)
                    return 1
            state.update(state='complete', finished_unix=time.time())
            save(state_path, state)
        except BaseException as exc:
            state.update(state='failed', error=repr(exc), finished_unix=time.time())
            save(state_path, state)
            raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
