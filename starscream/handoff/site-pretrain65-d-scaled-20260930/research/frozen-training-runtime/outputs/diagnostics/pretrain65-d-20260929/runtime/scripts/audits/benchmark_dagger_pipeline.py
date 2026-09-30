"""Bounded 35/65 replay update pipeline benchmark with production dtypes/fields."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
import torch
from scripts import train_privileged_racing as t
from starscream.dagger_throughput import HierarchicalReplayPlan, prefetched_batches
from starscream.training_acceleration import configure_policy_acceleration
from starscream.update_transport import UpdateBatchTransfer
from starscream.dagger_update_graph import CapturedDaggerUpdate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['baseline', 'static', 'prefetch', 'packed-prefetch', 'graph'], required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--tag', help='Artifact suffix when comparing sampler revisions')
    p.add_argument('--ready-file', type=Path)
    p.add_argument('--start-file', type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    root = Path('outputs/checkpoints/starscream-v6.21.1.update-fix-pretrain65-dagger')
    policy, _, payload, _ = t.load_policy_checkpoint(root/'latest.pt', 'cuda')
    settings = copy.deepcopy(payload['training_config']['dagger'])
    pools = []
    for file, pool in [('round-00004.h5', 'permanent'), ('round-00218.h5', 'online')]:
        with h5py.File(root/'dagger-replay'/file) as archive:
            pools.append({k:v[:] for k,v in archive[pool].items()})
            pools[-1]['action_chunk_valid'] = np.zeros(len(pools[-1]['histories']), dtype=bool)
    if args.mode != 'baseline':
        settings['dagger_statistics_group_count'] = 1+max(int(x['groups'].max()) for x in pools)
    if args.mode == 'graph':
        settings['_dagger_physical_scales'] = torch.tensor(settings.get('physical_action_scales',[10.,3.,3.,3.]),device='cuda')
    configure_policy_acceleration(policy, settings)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(settings['learning_rate']), fused=True)
    optimizer.load_state_dict(payload['optimizer'])
    plans = [HierarchicalReplayPlan(*(data[k] for k in ['families','tracks','gate_indices','events','teacher_modes','occupancy_modes']), size)
             for data,size in zip(pools,[538,998])]
    keys = ['histories','actions','previous','dynamics','dynamics_valid','speed_commands',
            'action_chunks','action_chunk_valid','executed_actions','reward_components','topology_targets','topology_valid','groups','anchor_mask']
    rng = np.random.default_rng(2036092404)
    torch.manual_seed(2036092404)
    digest = hashlib.sha256()
    def batches():
        for _ in range(args.steps):
            selected = [plan.sample(rng) for plan in plans]
            for indices in selected:
                digest.update(indices.tobytes())
            yield tuple(np.concatenate([data[k][indices] for data,indices in zip(pools,selected)]) for k in keys)
    transfer = UpdateBatchTransfer('cuda', args.mode == 'packed-prefetch')
    policy.train()
    losses = []
    def update(tensors):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, pieces = t.imitation_loss(policy,tensors['histories'],tensors['actions'],tensors['previous'],
                tensors['dynamics'],settings,tensors['dynamics_valid'],tensors['speed_commands'],
                topology_targets=tensors['topology_targets'],topology_valid=tensors['topology_valid'],
                group_ids=tensors['groups'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings['gradient_clip']))
        optimizer.step()
        return loss, pieces
    captured = CapturedDaggerUpdate(policy,optimizer,lambda batch: update(dict(zip(keys,batch)))) if args.mode == 'graph' else None
    with prefetched_batches(batches(), 2 if 'prefetch' in args.mode else 0) as prepared:
        for i, arrays in enumerate(prepared):
            tensors = dict(zip(keys, transfer(arrays)))
            if args.mode == 'graph':
                loss, _ = captured(tuple(tensors.values()))
            else:
                loss, _ = update(tensors)
            losses.append(loss.detach())
            if i == 19:
                torch.cuda.synchronize()
                if args.ready_file:
                    if args.start_file is None:
                        raise ValueError('--ready-file requires --start-file')
                    args.ready_file.touch()
                    deadline = time.monotonic() + 300
                    while not args.start_file.exists():
                        if time.monotonic() > deadline:
                            raise TimeoutError('Benchmark start gate timed out')
                        time.sleep(.01)
                start = time.perf_counter()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
    result = dict(mode=args.mode, steps=args.steps, warm_seconds=elapsed/(args.steps-20),
                  index_digest=digest.hexdigest(),losses=torch.stack(losses).cpu().tolist(),
                  note='Real permanent round 4 and online round 218, 538/998 rows per batch; bounded shard locality, not full resident replay.')
    dest = Path('outputs/dagger-throughput')
    tag = args.tag or args.mode
    (dest/f'pipeline-{tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    torch.save(policy.state_dict(),dest/f'pipeline-{tag}-parameters.pt')
    print({k:v for k,v in result.items() if k!='losses'},flush=True)


if __name__ == '__main__':
    main()
