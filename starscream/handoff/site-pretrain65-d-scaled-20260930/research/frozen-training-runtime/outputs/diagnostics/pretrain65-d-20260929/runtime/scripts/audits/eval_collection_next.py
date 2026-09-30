"""Fresh paired evaluation of fixed broad, learner segments and fixed bounded."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace')
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/diagnostics/collection-next-20260929'

def save(name, data):
    path = OUT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str) + '\n')
    tmp.replace(path)

def main():
    import torch
    from scripts.train_privileged_racing import load_config, parse_stage, ProcessRaceCollector
    from starscream.privileged_racing import load_policy_checkpoint
    from starscream.racing_evaluation import canonical_inference
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    save('eval.status.json', dict(state='running', pid=os.getpid(), started=time.time()))
    try:
        for arm, suffix in [('a', 'fixes'), ('d', 'collection-d'), ('bounded', 'fixes-bounded')]:
            config = ROOT / ('configs/exp/v6.21.1.1/mini_slalom_aug2_' + suffix.replace('-', '_') + '.yaml')
            settings = load_config(config)['dagger']
            checkpoint = ROOT / 'outputs/checkpoints' / settings['run_name'] / 'latest.pt'
            policy, normalizer, payload, _ = load_policy_checkpoint(checkpoint, 'cuda')
            stage = parse_stage(settings['evaluation_curriculum'])
            collector = ProcessRaceCollector(policy, normalizer, settings, stage, 'cuda', workers=16)
            try:
                with canonical_inference():
                    rows = collector.evaluate_rows(episodes=100, seed_base=2046092900)
            finally:
                collector.close()
            save(f'eval-{arm}-related.json', dict(checkpoint=str(checkpoint), round=payload['round'], seed=2046092900, episodes=rows))
            del policy, normalizer, collector, payload
            torch.cuda.empty_cache()
            command = [sys.executable, 'scripts/audits/eval_privileged_real100_v2_dual.py',
                '--config', str(config), '--checkpoint', str(checkpoint), '--output', str(OUT/f'eval-{arm}-real100.json'),
                '--episodes', '1', '--seed', '2046092900', '--workers', '16', '--reference-aware',
                '--max-steps', '6000', '--retry-steps', '6001', '--dwell-steps', '6001']
            subprocess.run(command, cwd=ROOT, check=True)
            results[arm] = dict(related_episodes=len(rows), real100=json.loads((OUT/f'eval-{arm}-real100.json').read_text())['aggregate'])
            save('eval.progress.json', results)
        save('eval.status.json', dict(state='complete', finished=time.time(), results=results))
    except BaseException as exc:
        save('eval.status.json', dict(state='failed', error=repr(exc), finished=time.time()))
        raise

if __name__ == '__main__':
    main()
