"""Isolate exact hierarchical draw vectorization from GPU update time."""
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
from starscream.dagger_throughput import HierarchicalReplayPlan


def main():
    root = Path('outputs/checkpoints/starscream-v6.21.1.update-fix-pretrain65-dagger/dagger-replay')
    plans = []
    for filename, name, size in [('round-00004.h5','permanent',538),('round-00218.h5','online',998)]:
        with h5py.File(root/filename) as archive:
            metadata = [archive[name][k][:] for k in ['families','tracks','gate_indices','events','teacher_modes','occupancy_modes']]
        plans.append(HierarchicalReplayPlan(*metadata,size))
    legacy = []
    for plan in plans:
        draws = []
        for pool, lengths, offsets in plan.draws:
            starts, counts = np.unique(offsets, return_counts=True)
            draws.append([(pool[start:start+lengths[np.searchsorted(offsets,start)]], int(count))
                          for start,count in zip(starts,counts)])
        legacy.append(draws)
    def old_sample(draws, rng):
        parts = []
        for draw in draws:
            rows = np.concatenate([rng.choice(pool,size=count,replace=True) for pool,count in draw])
            rng.shuffle(rows)
            parts.append(rows)
        result = np.concatenate(parts)
        rng.shuffle(result)
        return result
    records = []
    states = []
    for variant in ['choice','vectorized']:
        rng = np.random.default_rng(2036092404)
        digest = hashlib.sha256()
        start = time.perf_counter()
        for _ in range(200):
            for plan, draws in zip(plans,legacy):
                rows = plan.sample(rng) if variant == 'vectorized' else old_sample(draws,rng)
                digest.update(rows.tobytes())
        records.append(dict(variant=variant,seconds_per_batch=(time.perf_counter()-start)/200,digest=digest.hexdigest()))
        states.append(rng.bit_generator.state)
    assert records[0]['digest'] == records[1]['digest'] and states[0] == states[1]
    Path('outputs/dagger-throughput/sampler.json').write_text(json.dumps(records,indent=2)+'\n')
    print(records)


if __name__ == '__main__':
    main()
