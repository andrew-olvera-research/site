"""Report-only matched real100 outcomes for a bounded DAgger continuation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from starscream.checkpoint_manager import capture_rng_state, restore_rng_state
from starscream.privileged_racing import checkpoint_payload


def report_bounded(policy, normalizer, settings, device, logger, directory, step, round_index):
    from scripts.audits.eval_privileged_real100_v2_dual import evaluate
    spec = settings['timed_reporting']
    folder = Path(directory)/'real100-behavior'
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder/f'round-{round_index:05d}.pt'
    payload = checkpoint_payload(policy, normalizer, stage='dagger', track='reporting-only',
        extra=dict(round=round_index, environment_steps=step, control_hz=130,
                   reporting_only=True, parent_checkpoint_sha256=spec['parent_sha256'],
                   training_config={'dagger': settings}))
    rng, training, threads = capture_rng_state(), policy.training, torch.get_num_threads()
    try:
        # Independent inference snapshot; full resumable states remain in latest/top-k.
        temporary = snapshot.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(snapshot)
        args = SimpleNamespace(config=Path(spec['config']), checkpoint=snapshot,
            protocol=Path(spec['protocol']), output=folder/f'round-{round_index:05d}.json',
            episodes=int(spec['episodes_per_course']), seed=int(settings['evaluation_seed']),
            workers=8, device=str(device), max_steps=6000, retry_steps=6001, dwell_steps=6001,
            reference_aware=True, reference_tolerance=.75, limit_courses=0, save_states=None,
            max_states=0)
        report = evaluate(args, loaded=(policy, normalizer, payload, snapshot))
        report.update(role='reporting_only', checkpoint_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            parent_checkpoint_sha256=spec['parent_sha256'],
            protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest())
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        logger.log_eval({f'real100/{k}':v for k,v in report['aggregate'].items()
                         if isinstance(v, (float, int))}, step)
        print(f'bounded_real100_report round={round_index} path={args.output}', flush=True)
        return report
    finally:
        restore_rng_state(rng)
        policy.train(training)
        torch.set_num_threads(threads)
