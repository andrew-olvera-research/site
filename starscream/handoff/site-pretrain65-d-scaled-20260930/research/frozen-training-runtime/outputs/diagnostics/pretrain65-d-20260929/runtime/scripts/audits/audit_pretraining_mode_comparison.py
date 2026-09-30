"""Read-only matched checkpoint screen; never trains or modifies source checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/diagnostics/pretraining-mode-decision-20260928'
CASES = {
    'v621': ('configs/exp/v6.21/scratch_5m_dagger_r218.yaml', 'starscream-v6.21-scratch-5m-dagger-r218/best-step-060968436-full_course_success-0.72857143.pt'),
    'pre_fix65': ('configs/exp/v6.21.1/pretrain65_dagger.yaml', 'starscream-v6.21.1-pretrain65-dagger/best-step-062273909-full_course_success-0.37272727.pt'),
    'update_fix': ('configs/exp/v6.21.1/update_fix_dagger.yaml', 'starscream-v6.21.1.update-fix-pretrain65-dagger/best-step-121343545-full_course_success-0.63181818.pt'),
    # Its original manifest was pruned. Use the common legacy103 evaluation
    # environment contract; model and normalizer still come from its checkpoint.
    'v6222': ('configs/exp/v6.21/scratch_5m_dagger_r218.yaml', 'starscream-v6.22.2-pretrain45/best-step-062343135-full_course_success-0.60555556.pt'),
    'plant': ('configs/exp/v6.21.1.1/plant_dagger.yaml', 'starscream-v6.21.1.1-plant-selection25-dagger/best-step-150819355-selection_suite_success-0.57875.pt'),
    'bounded': ('configs/exp/v6.21.1.1/plant_recovery_fix.yaml', 'starscream-v6.21.1.1-plant-selection25-dagger-recovery-fix/best-step-125516154-selection_suite_success-0.29125.pt'),
    'bounded_scale': ('configs/exp/v6.21.1.1/plant_recovery_fix_scale_continuation.yaml', 'starscream-v6.21.1.1-plant-selection25-dagger-recovery-fix-scale/best-step-141977167-selection_suite_success-0.2625.pt'),
    'v3-best': ('configs/exp/v6.21.1.1/plant_recovery_support_v3.yaml', 'starscream-v6.21.1.1-plant-recovery-support-v3/best-step-042289694-selection_suite_clean_timely_success-0.0225.pt'),
    'v3-latest': ('configs/exp/v6.21.1.1/plant_recovery_support_v3.yaml', 'starscream-v6.21.1.1-plant-recovery-support-v3/latest.pt'),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--cases', nargs='+', default=list(CASES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.cases:
        config, checkpoint = CASES[name]
        checkpoint = ROOT / 'outputs/checkpoints' / checkpoint
        output = OUT / f'{name}-e{args.episodes}.json'
        cmd = [sys.executable, '-u', 'scripts/audits/eval_privileged_real100_v2_dual.py',
               '--config', config, '--checkpoint', str(checkpoint), '--output', str(output),
               '--episodes', str(args.episodes), '--workers', '8', '--reference-aware',
               '--max-steps', '6000', '--retry-steps', '6001', '--dwell-steps', '6001']
        record = dict(name=name, command=cmd, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256((ROOT/config).read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256((ROOT/'configs/eval/real100_v2_timed_protocol_v1.json').read_bytes()).hexdigest(),
            evaluator_sha256=hashlib.sha256((ROOT/'scripts/audits/eval_privileged_real100_v2_dual.py').read_bytes()).hexdigest())
        print(f'START {name}', flush=True)
        with (OUT / f'{name}-e{args.episodes}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record['returncode'] = result.returncode
        if result.returncode == 0:
            report = json.loads(output.read_text())
            record.update(aggregate=report['aggregate'], round=report['checkpoint_round'], steps=report['checkpoint_steps'])
            successes = [r for r in report['episodes'] if r['success']]
            record['successes_with_miss'] = sum(r['missed_crossings'] > 0 for r in successes)
            record['successful_retry_fraction'] = record['successes_with_miss']/len(successes) if successes else None
            record['late_fraction_of_successes'] = sum(r['late_success'] for r in successes)/len(successes) if successes else None
        records.append(record)
        (OUT / f'summary-{"-".join(args.cases)}-e{args.episodes}.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
