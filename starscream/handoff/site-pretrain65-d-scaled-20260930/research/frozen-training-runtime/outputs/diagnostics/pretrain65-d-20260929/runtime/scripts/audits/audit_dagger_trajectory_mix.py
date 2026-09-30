"""Reproduce legacy replay-stratum exposure without loading policies or CUDA.

Draws use the production cached sampler on each supplied shard in isolation.
These are diagnostic draws, NOT reconstructed historical training minibatches.
Missing trajectory-quality metadata is reported as unknown, never nominal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.dagger_throughput import HierarchicalReplayPlan

KEYS = ('families', 'tracks', 'gate_indices', 'events', 'teacher_modes', 'occupancy_modes')


def proportions(events, teachers, occupancy, indices):
    event = events[indices].astype(bool)
    proxy = (teachers[indices] != 0) | (occupancy[indices] == 1)
    return dict(legacy_nominal=float(np.mean(~event & ~proxy)),
                legacy_critical=float(np.mean(event)),
                legacy_recovery=float(np.mean(~event & proxy)),
                recovery_proxy_including_events=float(np.mean(proxy)))


def audit_shard(path, batches, size, seed):
    results = []
    with h5py.File(path, 'r') as archive:
        for name in ('online', 'permanent'):
            group = archive[name]
            if not len(group['events']):
                continue
            arrays = [group[key][:] for key in KEYS]
            plan = HierarchicalReplayPlan(*arrays, size)
            rng = np.random.default_rng(seed)
            sampled = np.concatenate([plan.sample(rng) for _ in range(batches)])
            results.append(dict(pool=name, rows=len(arrays[0]),
                fields=sorted(group.keys()),
                stored=proportions(*arrays[3:], np.arange(len(arrays[0]))),
                diagnostic_sampled=proportions(*arrays[3:], sampled),
                sampled_rows=len(sampled), unique_sampled_rows=len(np.unique(sampled)),
                unique_tracks=int(len(np.unique(arrays[1]))),
                teacher_modes=np.unique(arrays[4]).tolist(),
                occupancy_modes=np.unique(arrays[5]).tolist()))
    return dict(path=str(path), pools=results)


def audit_timed(path):
    report = json.loads(path.read_text())
    episodes = [e for track in report['tracks'] for e in track['episodes']]
    return dict(path=str(path), episodes=len(episodes),
        any_miss=sum(e['missed_crossings'] > 0 for e in episodes),
        multiple_misses=sum(e['missed_crossings'] > 1 for e in episodes),
        any_recovery=sum(e['recovered_gates'] > 0 for e in episodes),
        completions=sum(bool(e['success']) for e in episodes),
        completions_after_miss=sum(bool(e['success']) and e['missed_crossings'] > 0 for e in episodes),
        clean_completions=sum(bool(e['clean_success']) for e in episodes),
        crashes=sum(bool(e['crashed']) for e in episodes),
        timeouts=sum(bool(e['timeout']) for e in episodes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=Path, action='append', default=[])
    parser.add_argument('--timed', type=Path, action='append', default=[])
    parser.add_argument('--batches', type=int, default=100)
    parser.add_argument('--size', type=int, default=998)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batches < 1 or args.size < 1 or not (args.shard or args.timed):
        parser.error('require positive batches/size and at least one shard or timed report')
    report = dict(schema='dagger-trajectory-mix-audit-v1',
        sampling=dict(batches=args.batches, size=args.size, seed=args.seed,
            requested_fractions=[.40, .35, .25], family_weights='equal',
            scope='Each shard/pool independently; not historical minibatch reconstruction'),
        limitations=['Legacy event/teacher/start-mode tags do not identify clean, corrective or retry states.',
                    'Historical active replay, dynamic family weights and RNG are not reconstructed.',
                    'Timed counts are raw episode counts; official metrics may weight courses/families.'],
        shards=[audit_shard(p, args.batches, args.size, args.seed) for p in args.shard],
        timed=[audit_timed(p) for p in args.timed])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output),
        shards=[dict(path=s['path'], pools=[{k: p[k] for k in ('pool', 'rows', 'stored', 'diagnostic_sampled')} for p in s['pools']]) for s in report['shards']],
        timed=report['timed']), indent=2))


if __name__ == '__main__':
    main()
