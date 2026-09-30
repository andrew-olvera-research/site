"""Launch the validated fix pilot once both named baseline arms exit cleanly."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / 'outputs/diagnostics/multifix-20260928/validation.json'
STATUS = ROOT / 'outputs/diagnostics/mini-slalom-recovery-baselines/mini_slalom_aug2_fixes.status.json'
CONFIG = 'configs/exp/v6.21.1.1/mini_slalom_aug2_fixes.yaml'
LOG = ROOT / 'outputs/logs/mini_slalom_aug2_fixes.console.log'
DEPENDENCIES = ('mini_slalom_aug2_broad', 'mini_slalom_aug2_bounded')


def dependencies_ready(directory):
    for name in DEPENDENCIES:
        path = directory / f'{name}.status.json'
        if not path.exists():
            return False
        status = json.loads(path.read_text())
        if status.get('state') == 'failed':
            raise RuntimeError(f'Baseline failed: {name}; fix arm was not launched')
        if status.get('state') != 'complete':
            return False
        if status.get('exit_code') != 0:
            raise RuntimeError(f'Baseline lacks successful exit: {name}')
    return True


def active_trainers():
    result = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
            if any(arg.endswith(b'/train_privileged_racing.py') for arg in argv):
                result.append(int(path.parent.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return result


def validate_artifact():
    report = json.loads(ARTIFACT.read_text())
    if report.get('status') != 'passed':
        raise RuntimeError('Fix validation has not passed')
    for name, digest in report['sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Validated artifact changed: {name}')
    return report


def save_status(run):
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(run, indent=2) + '\n')
    temporary.replace(STATUS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    validate_artifact()
    if not args.launch:
        print(json.dumps({'validated': True, 'dependencies_ready': dependencies_ready(STATUS.parent),
                          'active_trainers': active_trainers()}))
        return 0
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATUS.exists() or LOG.exists():
            raise FileExistsError('Fix arm already reserved; inspect existing status, never duplicate it')
        run = dict(name='mini_slalom_aug2_fixes', state='waiting', launcher_pid=os.getpid(),
                   config=CONFIG, log=str(LOG.relative_to(ROOT)), queued_unix=time.time(),
                   dependencies=list(DEPENDENCIES), validation=str(ARTIFACT.relative_to(ROOT)))
        save_status(run)
        try:
            while not dependencies_ready(STATUS.parent) or active_trainers():
                time.sleep(20)
            validate_artifact()
            config = json.loads((ROOT / CONFIG).read_text())
            output = Path(config['output_root']) / 'checkpoints' / config['dagger']['run_name']
            if output.exists():
                raise FileExistsError('Fresh fix-arm output already exists')
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with LOG.open('x') as stream:
                process = subprocess.Popen([sys.executable, '-u', 'scripts/train_privileged_racing.py',
                    '--config', CONFIG, '--stage', 'dagger', '--device', 'cuda'], cwd=ROOT,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                         'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'})
                run.update(state='running', pid=process.pid, started_unix=time.time())
                save_status(run)
                code = process.wait()
            run.update(state='complete' if code == 0 else 'failed', exit_code=code,
                       finished_unix=time.time())
            save_status(run)
            return code
        except Exception as exc:
            run.update(state='failed', error=str(exc), finished_unix=time.time())
            save_status(run)
            raise


if __name__ == '__main__':
    sys.exit(main())
