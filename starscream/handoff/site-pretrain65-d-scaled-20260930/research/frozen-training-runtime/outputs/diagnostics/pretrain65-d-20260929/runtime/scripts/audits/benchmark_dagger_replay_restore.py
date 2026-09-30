"""Compare direct replay restore with the former concatenate path, in fresh processes."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
from starscream.dagger_replay import DaggerReplayStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['direct', 'concatenate'], required=True)
    args = parser.parse_args()
    root = Path('outputs/checkpoints/starscream-v6.21.1.update-fix-pretrain65-dagger/dagger-replay')
    with h5py.File(root/'round-00218.h5') as archive:
        contract = json.loads(archive.attrs['contract_json'])
        templates = {k: np.empty((0, *v.shape[1:]), v.dtype) for k,v in archive['online'].items()}
    store = DaggerReplayStore(root, contract, [{'root': str(root), 'min_round': 1, 'max_round': 218}])
    start = time.perf_counter()
    capacity = 3949799
    if args.mode == 'direct':
        result = store.restore_pool('online', templates, capacity=capacity, committed_round=218)
    else:
        pieces = {k: [] for k in templates}
        remaining = capacity
        for path in reversed(store._committed_shards(218)):
            if remaining <= 0:
                break
            with h5py.File(path) as archive:
                group = archive['online']
                rows = int(group.attrs['rows'])
                take = min(rows, remaining)
                if not take:
                    continue
                for name in templates:
                    pieces[name].append(group[name][rows-take:rows])
                remaining -= take
        result = {k: np.concatenate(v[::-1]) for k,v in pieces.items()}
    elapsed = time.perf_counter()-start
    digest = hashlib.sha256()
    for name in sorted(result):
        digest.update(name.encode())
        if result[name].size:
            digest.update(memoryview(result[name]).cast('B'))
    record = dict(mode=args.mode, seconds=elapsed,
                  max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                  rows=len(result['histories']), bytes=sum(v.nbytes for v in result.values()),
                  digest=digest.hexdigest())
    Path(f'outputs/dagger-throughput/restore-{args.mode}.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
