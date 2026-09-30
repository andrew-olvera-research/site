"""Run one isolated config and sample GPU/memory telemetry until it exits."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.memory_pressure import memory_pressure_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if Path(args.tag).name != args.tag:
        raise ValueError('Tag must be a simple file name')
    logs = Path('outputs/logs')
    with (logs/f'{args.tag}.log').open('w') as console, (logs/f'{args.tag}.system.jsonl').open('w') as telemetry:
        process = subprocess.Popen([sys.executable, 'scripts/train_privileged_racing.py',
            '--config', args.config, '--stage', 'dagger', '--device', 'cuda'],
            stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                sample = dict(unix=time.time(), **memory_pressure_snapshot())
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
                    values = [float(x.strip()) for x in result.stdout.splitlines()[0].split(',')]
                    sample.update(zip(('gpu_utilization_percent', 'gpu_memory_utilization_percent',
                                       'gpu_memory_mib', 'gpu_power_watts'), values))
                except (subprocess.SubprocessError, ValueError, IndexError, OSError) as error:
                    sample['gpu_error'] = str(error)
                telemetry.write(json.dumps(sample)+'\n')
                telemetry.flush()
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f'Training exited {process.returncode}; see {console.name}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == '__main__':
    main()
