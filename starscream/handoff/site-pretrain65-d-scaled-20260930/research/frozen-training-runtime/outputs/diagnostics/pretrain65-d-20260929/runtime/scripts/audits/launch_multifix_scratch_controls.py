"""Bootstrap shared legacy units, then launch both authorized scratch controls."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/diagnostics/multifix-scratch-controls-20260928'
RUNTIME = OUT / 'baseline-runtime'
STATUS_DIR = ROOT / 'outputs/diagnostics/mini-slalom-recovery-baselines'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    os.chdir(ROOT)
    with (OUT / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = OUT / 'pair.status.json'
        if state_path.exists():
            raise FileExistsError('Scratch pair already reserved; inspect its status')
        for rel, digest in json.loads((OUT / 'runtime-sha256.json').read_text()).items():
            if hashlib.sha256((RUNTIME / rel).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Frozen runtime changed: {rel}')
        jobs = []
        for kind in ('broad', 'bounded'):
            name = f'mini_slalom_aug2_scratch_{kind}'
            config = ROOT / f'configs/exp/v6.21.1.1/{name}.yaml'
            status = STATUS_DIR / f'{name}.status.json'
            log = ROOT / f'outputs/logs/{name}.console.log'
            c = json.loads(config.read_text())
            output = Path(c['output_root']) / 'checkpoints' / c['dagger']['run_name']
            if status.exists() or log.exists() or output.exists():
                raise FileExistsError(f'Scratch control already exists: {name}')
            assert c['dagger']['initial_checkpoint'] is None
            assert c['dagger']['resume_checkpoint'] is None
            jobs.append((name, config, status, log))
        state = dict(state='bootstrapping', started_unix=time.time(), launcher_pid=os.getpid())
        save(state_path, state)
        environment = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                       'OPENBLAS_NUM_THREADS': '1', 'PYTHONPATH': str(RUNTIME), 'PYTHONUNBUFFERED': '1'}
        try:
            if not (OUT / 'normalization-legacy.pt').exists():
                with (OUT / 'bootstrap.log').open('x') as log:
                    subprocess.run([sys.executable, '-u', str(RUNTIME / 'scripts/audits/collect_dagger_initialization_statistics.py'),
                        '--config', str(OUT / 'bootstrap.json'), '--output', str(OUT / 'normalization-legacy.pt'),
                        '--episodes', '36', '--coverage-retry-episodes-per-track', '2', '--device', 'cpu'],
                        cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            import torch
            stats = torch.load(OUT / 'normalization-legacy.pt', map_location='cpu', weights_only=False)
            if stats['previous_action_feature_mapping'] != 'legacy_linear' or len(stats['track_counts']) != 9:
                raise RuntimeError('Legacy normalization coverage/contract mismatch')
            active = []
            for name, config, status, log in jobs:
                with log.open('x') as stream:
                    process = subprocess.Popen([sys.executable, '-u', str(RUNTIME / 'scripts/train_privileged_racing.py'),
                        '--config', str(config), '--stage', 'dagger', '--device', 'cuda'], cwd=ROOT,
                        env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                run = dict(name=name, state='running', pid=process.pid, config=str(config.relative_to(ROOT)),
                           log=str(log.relative_to(ROOT)), runtime=str(RUNTIME), started_unix=time.time())
                save(status, run)
                active.append((process, status, run))
            state.update(state='running', pids=[p.pid for p, _, _ in active])
            save(state_path, state)
            failed = False
            while active:
                for process, status, run in active[:]:
                    code = process.poll()
                    if code is not None:
                        run.update(state='complete' if code == 0 else 'failed', exit_code=code, finished_unix=time.time())
                        save(status, run)
                        failed |= code != 0
                        active.remove((process, status, run))
                if active:
                    time.sleep(10)
            state.update(state='failed' if failed else 'complete', finished_unix=time.time())
            save(state_path, state)
            return int(failed)
        except Exception as exc:
            state.update(state='failed', error=str(exc), finished_unix=time.time())
            save(state_path, state)
            raise


if __name__ == '__main__':
    sys.exit(main())
