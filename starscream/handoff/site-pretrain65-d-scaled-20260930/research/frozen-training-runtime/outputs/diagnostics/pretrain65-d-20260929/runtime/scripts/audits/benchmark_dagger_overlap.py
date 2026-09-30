"""Compare warmed real collection/update workloads serially and concurrently.

Independent CUDA contexts and immutable collector weights deliberately isolate
correctness from scheduling. This is a contention experiment, not an async trainer.
No production checkpoints or replay are written.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', default='serial,overlap,serial')
    parser.add_argument('--episodes', type=int, default=65)
    parser.add_argument('--updates', type=int, default=1200)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    dest = root / 'outputs/dagger-throughput'
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix='dagger-overlap-') as temp:
        for index, mode in enumerate(args.modes.split(',')):
            if mode not in ('serial', 'overlap'):
                raise ValueError(mode)
            tag = f'overlap-{index}-{mode}'
            gates = Path(temp) / tag
            gates.mkdir()
            collection_file = dest / f'{tag}-collection.json'
            commands = {
                'collect': [sys.executable, 'scripts/audits/benchmark_dagger_collection.py',
                    '--output', str(collection_file), '--episodes', str(args.episodes),
                    '--workers', str(args.workers), '--variants', '1:0:chunk',
                    '--warmup-episodes', str(args.episodes)],
                'learn': [sys.executable, 'scripts/audits/benchmark_dagger_pipeline.py',
                    '--mode', 'graph', '--steps', str(args.updates + 20), '--tag', tag],
            }
            processes, logs = {}, []
            try:
                # Warm sequentially to avoid concurrent compilation/capture and
                # exclude startup from both modes. Both processes stay resident.
                for name, command in commands.items():
                    log = (dest / f'{tag}-{name}.log').open('w')
                    logs.append(log)
                    command += ['--ready-file', str(gates / f'{name}.ready'),
                                '--start-file', str(gates / f'{name}.start')]
                    processes[name] = subprocess.Popen(command, cwd=root, stdout=log,
                                                       stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + 240
                    while not (gates / f'{name}.ready').exists():
                        if processes[name].poll() is not None:
                            raise RuntimeError(f'{name} exited during warmup; see {log.name}')
                        if time.monotonic() > deadline:
                            raise TimeoutError(f'{name} warmup')
                        time.sleep(.1)
                started = time.perf_counter()
                (gates / 'collect.start').touch()
                if mode == 'serial':
                    if processes['collect'].wait(timeout=240):
                        raise RuntimeError('Collector failed')
                (gates / 'learn.start').touch()
                for name, process in processes.items():
                    if process.wait(timeout=240):
                        raise RuntimeError(f'{name} failed')
                wall = time.perf_counter() - started
                collection = json.loads(collection_file.read_text())[0]
                learner = json.loads((dest / f'pipeline-{tag}.json').read_text())
                learning_seconds = learner['warm_seconds'] * args.updates
                result = dict(mode=mode, workers=args.workers,
                    collect_seconds=collection['seconds'], learn_seconds=learning_seconds,
                    collection_sps=collection['steps_s'], updates_s=1/learner['warm_seconds'],
                    workload_seconds=(collection['seconds'] + learning_seconds if mode == 'serial'
                                      else max(collection['seconds'], learning_seconds)),
                    process_wall_seconds=wall, labels=collection['labels'],
                    accepted=collection['accepted'], learner_index_digest=learner['index_digest'],
                    note='Workload time excludes warmup, teardown and parameter serialization; '
                         'two CUDA contexts, fixed actor, bounded replay shards, no weight publication.')
                results.append(result)
                (dest / 'overlap-summary.json').write_text(json.dumps(results, indent=2) + '\n')
                print(json.dumps(result), flush=True)
            finally:
                for process in processes.values():
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                for log in logs:
                    log.close()


if __name__ == '__main__':
    main()
