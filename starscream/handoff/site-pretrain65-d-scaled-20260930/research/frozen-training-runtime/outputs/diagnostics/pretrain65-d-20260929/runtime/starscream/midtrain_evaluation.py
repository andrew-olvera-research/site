"""Fixed-cohort per-course pace diagnostics for expert specialization."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch


def finite_metrics(metrics):
    return {k: float(v) if np.isfinite(v) else None for k,v in metrics.items()}


def collect_pace_cohort(policy,normalizer,settings,stage,device,count,seed):
    from scripts.train_privileged_racing import ProcessRaceCollector, multitrack_metrics
    collector=ProcessRaceCollector(policy,normalizer,settings,stage,device,
        workers=min(int(settings.get('evaluation_workers',12)),count))
    try:
        rows=collector.evaluate_rows(episodes=count,seed_base=seed)
    finally:
        collector.close()
    return multitrack_metrics(rows,stage.target_gates*stage.rollout_laps),rows


def paired_success_laps(baseline,current,control_hz):
    def successful(rows):
        return {(r['track'],r['episode_index']):float(r['steps'])/control_hz for r in rows
                if not r['crashed'] and r['gates']>=r['target_gates']}
    old,new=successful(baseline),successful(current)
    by_track={}
    for key in old.keys() & new.keys():
        by_track.setdefault(key[0],[]).append(new[key]/old[key])
    return by_track


def evaluate_midtraining(policy, normalizer, settings, stage, device, logger,
                         checkpoint_dir, step, round_index):
    spec=settings.get('midtrain_pace_probe')
    if not spec or (round_index and round_index % int(spec.get('interval',5))
                    and round_index != int(settings['rounds'])):
        return
    from scripts.train_privileged_racing import lap_timing_metrics
    from starscream.course_model.training import atomic_json
    from starscream.privileged_racing import checkpoint_payload
    root=Path(checkpoint_dir)/'pace-probes'; root.mkdir(parents=True,exist_ok=True)
    for panel,manifest in spec['panels'].items():
        probe=dict(settings,track_manifest=manifest,evaluation_condition_on_manifest_speed=True)
        probe_stage=replace(stage,tracks=tuple(settings['curriculum']['tracks']))
        metrics,rows=collect_pace_cohort(policy,normalizer,probe,probe_stage,device,
            len(probe_stage.tracks)*int(spec.get('episodes_per_course',16)),
            int(spec.get('seed',2076091721)))
        metrics.update(lap_timing_metrics(metrics,float(settings['control_hz'])))
        baseline_path=root/f'{panel}-round-000.json'
        if round_index:
            baseline_document=json.loads(baseline_path.read_text())
            baseline=baseline_document['metrics']
            paired=paired_success_laps(baseline_document['episodes'],rows,float(settings['control_hz']))
            prefixes=[k[:-len('successful_episodes')] for k,v in baseline.items()
                      if k.startswith('track/') and k.endswith('/successful_episodes') and v>=3]
            ratios=[]; eligible=True
            for prefix in prefixes:
                current=float(metrics.get(prefix+'successful_episodes',0))
                if current<3:
                    eligible=False; continue
                mean=float(metrics[prefix+'successful_lap_time_seconds'])
                ratios.append(mean/float(baseline[prefix+'successful_lap_time_seconds']))
                metrics[prefix+'mean_over_base_fastest']=mean/float(baseline[prefix+'fastest_lap_time_seconds'])
                metrics[prefix+'mean_over_base_p10']=mean/float(baseline[prefix+'p10_lap_time_seconds'])
                metrics[prefix+'mean_to_fastest_gap_seconds']=mean-float(metrics[prefix+'fastest_lap_time_seconds'])
                pairs=paired.get(prefix.split('/')[1],[])
                metrics[prefix+'paired_successes']=len(pairs)
                metrics[prefix+'paired_success_lap_ratio']=float(np.mean(pairs)) if pairs else float('nan')
            metrics['paired_course_mean_lap_ratio']=float(np.mean(ratios)) if ratios else 1.
            metrics['pace_comparable_courses']=len(ratios)
            metrics['pace_guard_passed']=float(bool(eligible and ratios and
                metrics['full_course_success']>=baseline['full_course_success']-.05))
            score=(1.-metrics['paired_course_mean_lap_ratio']) if metrics['pace_guard_passed'] else -1.
            best_path=root/f'{panel}-best.json'
            best=json.loads(best_path.read_text()) if best_path.exists() else {'score':0.}
            if score>best['score']:
                payload=checkpoint_payload(policy,normalizer,stage='dagger',track=','.join(probe_stage.tracks),
                    extra={'round':round_index,'environment_steps':step,'metrics':metrics,
                           'initial_checkpoint':settings['initial_checkpoint'],'pace_panel':panel,
                           'control_hz':settings['control_hz'],'training_config':{'dagger':settings}})
                destination=Path(checkpoint_dir)/f'pace-best-{panel}.pt'
                temporary=destination.with_suffix('.tmp')
                torch.save(payload,temporary); temporary.replace(destination)
                atomic_json(best_path,dict(round=round_index,step=step,score=score,checkpoint=str(destination)))
        from scripts.audits.prepare_v620_transitions import finite_report
        atomic_json(root/f'{panel}-round-{round_index:03d}.json',dict(round=round_index,step=step,
                    metrics=finite_metrics(metrics),episodes=finite_report(rows)))
        logger.log_eval({f'midtrain/{panel}/{k}':v for k,v in metrics.items()},step)
        print(f"midtrain_pace panel={panel} round={round_index} full={metrics['full_course_success']:.4f} "
              f"mean_s={metrics['successful_lap_time_seconds']:.4f} "
              f"paired_ratio={metrics.get('paired_course_mean_lap_ratio',1.):.4f}",flush=True)
