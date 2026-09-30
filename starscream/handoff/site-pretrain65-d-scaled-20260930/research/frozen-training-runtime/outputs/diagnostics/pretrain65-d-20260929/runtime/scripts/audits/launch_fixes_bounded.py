"""Launch the single corrected-input bounded collection control once."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace')
OUT = ROOT / 'outputs/diagnostics/fixes-bounded-20260929'
RUNTIME = OUT / 'runtime'
CONFIG = ROOT / 'configs/exp/v6.21.1.1/mini_slalom_aug2_fixes_bounded.yaml'
STATUS = OUT / 'fixes_bounded.status.json'
LOG = ROOT / 'outputs/logs/mini_slalom_aug2_fixes_bounded.console.log'


def save(payload):
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(STATUS)


def main():
    os.chdir(ROOT)
    with (OUT / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = json.loads((OUT / 'validation.json').read_text())
        if report['status'] != 'passed':
            raise RuntimeError('bounded control validation has not passed')
        for name,digest in report['sha256'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'validated input changed: {name}')
        for name,digest in json.loads((OUT / 'runtime-sha256.json').read_text()).items():
            if hashlib.sha256((RUNTIME / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'frozen runtime changed: {name}')
        config = json.loads(CONFIG.read_text())
        settings = config['dagger']
        output = Path(config['output_root']) / 'checkpoints' / settings['run_name']
        if STATUS.exists() or LOG.exists() or output.exists():
            raise FileExistsError('bounded fixes arm already has status/log/checkpoint')
        if settings['initial_checkpoint'] or settings['resume_checkpoint']:
            raise ValueError('bounded fixes arm must start from scratch')
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                args = (proc / 'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if any(a.endswith(b'/train_privileged_racing.py') or a == b'train_privileged_racing.py' for a in args):
                raise RuntimeError(f'existing racing trainer PID {proc.name}')
        run = dict(state='reserved', config=str(CONFIG.relative_to(ROOT)),
                   log=str(LOG.relative_to(ROOT)), checkpoint=str(output.relative_to(ROOT) / 'latest.pt'),
                   runtime=str(RUNTIME), launcher_pid=os.getpid(), started_unix=time.time())
        save(run)
        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
               'OPENBLAS_NUM_THREADS': '1', 'PYTHONPATH': str(RUNTIME), 'PYTHONUNBUFFERED': '1'}
        try:
            with LOG.open('x') as stream:
                child = subprocess.Popen([sys.executable, '-u', str(RUNTIME / 'scripts/train_privileged_racing.py'),
                    '--config', str(CONFIG), '--stage', 'dagger', '--device', 'cuda'],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            run.update(state='running', pid=child.pid)
            save(run)
            code = child.wait()
            if code == 0:
                import torch
                payload = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
                if payload['round'] != 24:
                    raise RuntimeError('trainer exited before round-24 checkpoint')
                run.update(round=24, environment_steps=payload['environment_steps'])
            run.update(state='complete' if code == 0 else 'failed', exit_code=code, finished_unix=time.time())
            save(run)
            return code
        except BaseException as exc:
            run.update(state='failed', error=repr(exc), finished_unix=time.time())
            save(run)
            raise


if __name__ == '__main__':
    sys.exit(main())
