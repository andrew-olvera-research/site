"""Compare validation batching under live MPCC collection, with fixed actor/seeds."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from scripts import train_privileged_racing as t
from starscream.dagger_async import AsyncDaggerCollector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', default='32:4,64:8,32:4')
    p.add_argument('--episodes', type=int, default=220)
    p.add_argument('--output', type=Path, default=Path('outputs/dagger-throughput/evaluation-batching.json'))
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    settings = t.load_config(Path('configs/exp/v6.21.1/update_fix_dagger.yaml'))['dagger']
    settings.update(dagger_event_inference=True,
                    dagger_require_successful_episodes=t.dagger_successful_episode_filter(219, settings))
    policy, normalizer, _, _ = t.load_policy_checkpoint(settings['resume_checkpoint'], 'cuda')
    policy.eval()
    stage = t.parse_stage(settings['curriculum'])
    evaluation_stage = t.parse_stage(settings.get('evaluation_curriculum', settings['curriculum']))
    collector = AsyncDaggerCollector(policy, normalizer, settings, stage, 'cuda')
    fixture = np.load('outputs/dagger-throughput/async-reference-v1/inputs.npz', allow_pickle=False)
    histories = fixture['permanent/input/histories'].astype(np.float32)
    speed = fixture['permanent/input/speed_commands']
    results = []
    try:
        collector.publish(policy, 218)
        warm = collector.collect(episodes=65, beta=.35, seed_base=2036092420)
        del warm
        collector.release_results()
        gc.collect()
        collector.begin(dict(episodes=1040, beta=.35, seed_base=2036092421), window=220)
        pending_path = Path(collector.directory.name)/collector.pending['token']/'metadata.pkl'
        for spec in args.variants.split(','):
            workers, per_worker = map(int, spec.split(':'))
            candidate = copy.deepcopy(settings)
            candidate.update(evaluation_workers=workers, evaluation_envs_per_worker=per_worker)
            evaluator = t.ProcessRaceCollector(policy, normalizer, candidate, evaluation_stage, 'cuda', workers=workers)
            try:
                for size in range(1, workers+1):
                    evaluator.host_evaluation_graphs.predict_numpy(histories[:size], speed[:size])
                torch.cuda.synchronize()
                before = pending_path.exists()
                started = time.perf_counter()
                metrics = t.evaluate_policy(policy, normalizer, candidate, evaluation_stage,
                    count=args.episodes, seed_base=int(settings['evaluation_seed']),
                    device='cuda', collector=evaluator)
                seconds = time.perf_counter()-started
                result = dict(variant=spec, seconds=seconds, metrics=metrics,
                    collection_active_for_entire_eval=not before and not pending_path.exists(),
                    main_cuda_reserved_mib=torch.cuda.memory_reserved()/1024**2)
                results.append(result)
                args.output.write_text(json.dumps(results, indent=2)+'\n')
                print({key: value for key, value in result.items() if key != 'metrics'} |
                      {'completion': metrics['full_course_success'], 'crash': metrics['crash_rate']}, flush=True)
            finally:
                evaluator.close()
                del evaluator
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        # This is a load generator; its speculative data never enters replay.
        collector.close()


if __name__ == '__main__':
    main()
