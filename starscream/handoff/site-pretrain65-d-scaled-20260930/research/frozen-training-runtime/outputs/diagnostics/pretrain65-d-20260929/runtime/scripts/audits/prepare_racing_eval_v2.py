"""Freeze empirical course references, rescore saved laps, and prepare unlaunched student arms."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from starscream.racing_evaluation import digest,deadline_steps,score_episode

MANIFEST=Path('configs/eval/v6_22_real100_hard_v2.manifest.json')
PAIRED=Path('outputs/evals/v62111-plant-student-final/best-real100-hard-v2-e8-paired.json')
SOLO=Path('outputs/evals/v62111-plant-final/best-real100-hard-v2-e8.json')
PROTOCOL=Path('configs/eval/real100_v2_timed_protocol_v1.json')
OUT=Path('outputs/diagnostics/racing-eval-v2')


def reference_candidates(qualification):
    result=[]
    def add(summary,path):
        starts=summary.get('by_start_gate',{})
        t=summary.get('successful_lap_time_seconds')
        if (summary.get('success_rate')==1 and set(starts)=={'0'} and
            starts['0'].get('episodes',0)>=2 and t is not None and np.isfinite(t) and t>0):
            result.append((float(t),path))
    if not qualification.get('qualified'):return result
    for profile,cohorts in qualification.get('attempted_profiles',{}).items():
        if 'nominal' in cohorts:add(cohorts['nominal'],f'qualification/attempted_profiles/{profile}/nominal')
    for profile,summary in qualification.get('screens',{}).items():add(summary,f'qualification/screens/{profile}')
    for cohort,row in qualification.get('cohorts',{}).items():
        if cohort in ('nominal','nominal-screen'):add(row.get('summary',{}),f'qualification/cohorts/{cohort}/summary')
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads(MANIFEST.read_text());paired=json.loads(PAIRED.read_text());solo=json.loads(SOLO.read_text())
    teacher={r['slot']:r for r in paired['teacher']['tracks']}
    records=[]
    for r in source['records']:
        candidates=reference_candidates(r['qualification'])
        if candidates:
            reference,field=min(candidates);kind='mpcc_nominal_cold_start_mean';evidence=str(MANIFEST)
        else:
            times=[e['lap_seconds'] for e in teacher[r['slot']]['episodes'] if e['success']]
            assert len(times)>=2,'missing current-geometry reference; do not invent a deadline'
            # Use independent saved teacher reports, never the student's outcomes.
            reference=min(times);field=f'teacher/tracks/{r["slot"]}/minimum_successful_lap';evidence=str(PAIRED)
            prior=solo['metrics'].get(f'track/{r["name"]}/successful_minimum_steps')
            if prior is not None and np.isfinite(prior) and prior>0 and prior/130<reference:
                reference=prior/130;field=f'metrics/track/{r["name"]}/successful_minimum_steps';evidence=str(SOLO)
            kind='provisional_frozen_pretrain_fastest_lap'
        records.append(dict(slot=r['slot'],name=r['name'],family=r['family'],path=r['path'],
            track_sha256=digest(r['path']),geometry_fingerprint=r['fingerprint'],
            reference_seconds=reference,reference_kind=kind,evidence=evidence,evidence_sha256=digest(evidence),evidence_field=field,
            reference_is_certified_optimal=False,deadline_steps=deadline_steps(reference)))
    protocol=dict(schema='starscream-racing-evaluation-v2',status='frozen-development-protocol',
        control_hz=130,speed_command=16.5,source_manifest=str(MANIFEST),source_manifest_sha256=digest(MANIFEST),
        deadline_multiplier=1.5,recovery_allowance_seconds=3.,
        deadline_formula='floor(130 * max(1.5 * reference_seconds, reference_seconds + 3))',
        inference=dict(batch_size=16,precision='fp32_math_no_tf32',slots='stable; zero padded inactive lanes'),
        selection='lexicographic family-weighted timely success, time-weighted success, negative crash rate',
        reference_note='85 current qualified courses use nominal cold-start MPCC means. Geometry-revised courses without current MPCC evidence use explicitly provisional, frozen teacher fast laps; these are not optima or new qualifications.',
        records=records)
    PROTOCOL.write_text(json.dumps(protocol,indent=2)+'\n')
    refs={r['slot']:r for r in records}
    grids=[]
    for multiple in (1.25,1.5,2.):
        for allowance in (0.,3.):
            row=dict(multiplier=multiple,minimum_recovery_allowance_seconds=allowance)
            for arm in ('teacher','student'):
                scored=[score_episode(e['success'],e['steps'],refs[r['slot']]['reference_seconds'],
                    deadline_steps(refs[r['slot']]['reference_seconds'],multiple,allowance))
                    for r in paired[arm]['tracks'] for e in r['episodes']]
                row[arm]=dict(timely_success=float(np.mean([e['timely_success'] for e in scored])),
                             time_weighted_success=float(np.mean([e['time_weighted_success'] for e in scored])))
            row['retention']=row['student']['timely_success']/row['teacher']['timely_success']
            grids.append(row)
    summary=dict(reference_counts={k:sum(r['reference_kind']==k for r in records) for k in sorted({r['reference_kind'] for r in records})},
        deadline_step_range=[min(r['deadline_steps'] for r in records),max(r['deadline_steps'] for r in records)],
        deadline_step_quantiles=np.quantile([r['deadline_steps'] for r in records],[0,.25,.5,.75,1]).tolist(),
        grid=grids,note='Post-hoc development rescore of legacy trajectories, not corrected-batching evaluation; crash status absent from paired source, so crash rate is not inferred.')
    (OUT/'deadline-sensitivity.json').write_text(json.dumps(summary,indent=2)+'\n')
    config=json.loads(Path('configs/exp/v6.21.1.1/plant_ekf_readout_student_async.json').read_text())
    config.update(schema='starscream-vision-distillation-v2',evaluation_protocol=str(PROTOCOL),
                  evaluation_protocol_sha256=digest(PROTOCOL),evaluation_workers=16)
    config.pop('replay_capacity',None)
    changes={
        'control':{},
        'alignment':dict(readout_predictor='mlp',readout_predictor_width=512,readout_weight=.1,readout_regression_weight=.02),
        'physical':dict(physical_action_weight=.25,physical_action_scales=[10.,3.,3.,3.]),
        'memory':dict(model=dict(config['model'],memory_steps=32,memory_stride=2,memory_width=64,feedforward=488)),
    }
    changes['combined']={**changes['alignment'],**changes['physical'],**changes['memory'],
        'permanent_teacher_only':True,'recovery_success_fraction':.5,
        'learning_rate_schedule':dict(warmup_rounds=2,final_fraction=.25)}
    folder=Path('configs/exp/v6.21.1.student-v2');folder.mkdir(exist_ok=True)
    for name,change in changes.items():
        arm=deepcopy(config);arm.update(change)
        run=f'v621-plant-ekf-v2-{name}-seed1';arm['output']=f'outputs/vision-distillation/{run}'
        arm['wandb']['run_name']=run;arm['wandb']['local_event_path']=arm['output']+'/wandb-events.jsonl'
        arm['wandb']['tags']=['student-v2',name,'timed-eval-v2','prepared-not-launched']
        arm['wandb']['train_metric_allowlist']+=['physical_action','readout_regression']
        arm['wandb']['eval_metric_allowlist']+=['time_weighted_success','crash_rate','timeout_rate','clean_success_rate']
        (folder/f'{name}.json').write_text(json.dumps(arm,indent=2)+'\n')
    (folder/'experiment-plan.json').write_text(json.dumps(dict(status='prepared-not-launched',
        recommended_first='combined',arms=list(changes),control='Same original learning recipe, revised evaluator and checkpoint bookkeeping.',
        combined_note='Combined is a practical new recipe, not an isolated causal ablation. Alignment, physical, and memory arms each change only their named training factor relative to control.',
        deployment_core_budget=3000000,teacher_only_during_training=True),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
