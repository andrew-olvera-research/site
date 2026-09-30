"""Opt-in timed reports without changing DAgger selection or replay feedback."""
import json
from pathlib import Path

from starscream.racing_evaluation import evaluate_racing
from starscream.checkpoint_manager import capture_rng_state, restore_rng_state


def report_due(round_index, spec):
    start=int(spec['start_round']);interval=int(spec['interval'])
    if interval<1:raise ValueError('timed report interval must be positive')
    return round_index>=start and ((round_index-start)%interval==0 or round_index==int(spec['end_round']))


def report_teacher(policy, normalizer, settings, stage, device, logger, directory, step, round_index):
    spec=settings.get('timed_reporting')
    if not spec or not report_due(round_index,spec):return None
    if spec.get('evaluator') == 'bounded_real100_dual':
        from starscream.bounded_continuation_reporting import report_bounded
        return report_bounded(policy, normalizer, settings, device, logger, directory, step, round_index)
    suite=json.loads(Path(spec['suite']).read_text())
    config=dict(evaluation_protocol=spec['protocol'],evaluation_protocol_sha256=spec['protocol_sha256'],
        evaluation_workers=16,evaluation_episodes=int(spec['episodes_per_course']),
        evaluation_seed=int(settings['evaluation_seed']),speed_command=16.5,
        observation='ekf',image_delay=1/30,model={})
    rng=capture_rng_state();was_training=policy.training
    try:
        report=evaluate_racing(config,settings,stage,suite,policy,normalizer,policy,device,teacher_only=True)
        report.update(round=round_index,environment_steps=step,role='reporting_only',
                      parent_checkpoint_sha256=spec['parent_sha256'])
        path=Path(directory)/'timed-evaluation'/f'round-{round_index:05d}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
        logger.log_eval({f'timed/{k}':report[k] for k in
            ('timely_success','time_weighted_success','crashed','timeout','clean_success')},step)
        print(f'timed_teacher_report round={round_index} timely={report["timely_success"]:.4f} '
              f'crash={report["crashed"]:.4f} path={path}',flush=True)
        return report
    finally:
        restore_rng_state(rng)
        policy.train(was_training)
