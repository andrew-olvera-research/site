#!/usr/bin/env python3
"""Run only the two requested base students, sequentially; stop on failure."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ('base_raw_interface_3m', 'base_ekf_3m')


def main():
    os.chdir(ROOT)
    output = ROOT/'outputs/vision-distillation'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'first-two.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configs = [json.loads((ROOT/f'configs/exp/v6.21.1.distill/{v}.json').read_text()) for v in VARIANTS]
        for variant, config in zip(VARIANTS, configs):
            for target in (ROOT/config['output'], output/f'{variant}.log'):
                if target.exists():
                    raise FileExistsError(target)
            subprocess.run(['bash', 'scripts/launchers/current/run_v6211_vision_distillation.sh', variant], check=True)
        state = dict(supervisor_pid=os.getpid(), variants=list(VARIANTS), completed=[], started_at=time.time())
        def publish(**fields):
            state.update(fields)
            temporary = output/'first-two.status.tmp'
            temporary.write_text(json.dumps(state, indent=2)+'\n')
            temporary.replace(output/'first-two.status.json')
        for variant in VARIANTS:
            with (output/f'{variant}.log').open('x') as log:
                child = subprocess.Popen(['bash', 'scripts/launchers/current/run_v6211_vision_distillation.sh',
                    variant, '--launch'], stdout=log, stderr=subprocess.STDOUT)
                publish(status='running', current=variant, child_pid=child.pid)
                code = child.wait()
                if code:
                    publish(status='failed', exit_code=code, finished_at=time.time())
                    raise SystemExit(code)
                state['completed'].append(variant)
        publish(status='complete', current=None, child_pid=None, finished_at=time.time())


if __name__ == '__main__':
    main()
