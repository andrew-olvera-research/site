"""Capture/verify fixed real replay inputs, hierarchical draws and actor outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import torch
from scripts import train_privileged_racing as t
from starscream.dagger_throughput import HierarchicalReplayPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--source-run', default='starscream-v6.21.1.update-fix-pretrain65-dagger')
    parser.add_argument('--permanent-round', type=int, default=4)
    parser.add_argument('--online-round', type=int, default=218)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    root = Path('outputs/checkpoints') / args.source_run
    if args.verify:
        hashes = json.loads((args.output/'sha256.json').read_text())
        for name, digest in hashes.items():
            if hashlib.sha256((args.output/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Reference fixture was modified: {name}')
        manifest = json.loads((args.output/'manifest.json').read_text())
        checkpoint = args.output/'actor.pt'
        arrays = dict(np.load(args.output/'inputs.npz', allow_pickle=False))
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        checkpoint = args.output/'actor.pt'
        policy, normalizer, _, _ = t.load_policy_checkpoint(root/'latest.pt', 'cpu')
        torch.save(t.checkpoint_payload(policy, normalizer, stage='dagger', track='reference'), checkpoint)
        arrays = {}
        metadata = ['families', 'tracks', 'gate_indices', 'events', 'teacher_modes', 'occupancy_modes']
        for name, filename in [('permanent', f'round-{args.permanent_round:05d}.h5'), ('online', f'round-{args.online_round:05d}.h5')]:
            with h5py.File(root/'dagger-replay'/filename) as file:
                pool = file[name]
                for key in metadata:
                    arrays[f'{name}/{key}'] = pool[key][:]
                selected = np.linspace(0, len(pool['histories'])-1, 512, dtype=int)
                for key in ('histories', 'speed_commands', 'actions', 'previous', 'dynamics', 'dynamics_valid'):
                    arrays[f'{name}/input/{key}'] = pool[key][selected]
        manifest = dict(schema=1, seed=2036092404, draws=100, quotas=[538, 998],
                        source_run=args.source_run, permanent_round=args.permanent_round, online_round=args.online_round,
                        note='Fixed actor and real replay inputs; scalar family weighting variant included.')
        np.savez(args.output/'inputs.npz', **arrays)
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    outputs = {}
    for weighted in (False, True):
        rng = np.random.default_rng(manifest['seed'])
        for name, quota in zip(('permanent', 'online'), manifest['quotas']):
            plan = HierarchicalReplayPlan(*(arrays[f'{name}/{key}'] for key in (
                'families', 'tracks', 'gate_indices', 'events', 'teacher_modes', 'occupancy_modes')),
                quota, family_weights={0: .5, 1: 2.} if weighted else None)
            outputs[f'{name}/indices/{weighted}'] = np.stack([plan.sample(rng) for _ in range(manifest['draws'])])
    policy, _, _, _ = t.load_policy_checkpoint(checkpoint, 'cuda')
    policy.eval()
    with torch.no_grad():
        for name in ('permanent', 'online'):
            h = torch.from_numpy(arrays[f'{name}/input/histories'].astype(np.float32)).cuda()
            s = torch.from_numpy(arrays[f'{name}/input/speed_commands']).cuda()
            outputs[f'{name}/actor'] = policy(h, s).cpu().numpy()
    if args.verify:
        expected = dict(np.load(args.output/'outputs.npz', allow_pickle=False))
        for name in outputs:
            np.testing.assert_array_equal(outputs[name], expected[name], err_msg=name)
        print('Exact reference parity: hierarchical indices and actor outputs', flush=True)
    else:
        np.savez(args.output/'outputs.npz', **outputs)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()}
        (args.output/'sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
        print(f'Captured reference suite in {args.output}', flush=True)


if __name__ == '__main__':
    main()
