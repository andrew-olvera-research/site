"""Freeze a fast warm-start broad/bounded screen over two augmented parents."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.train_privileged_racing import configured_tracks,load_config,stage_config
from starscream.dagger_quality import quality_config
from starscream.evaluation_suite import configure_selection_suite

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/exp/v6.21.1.1'
OUT=ROOT/'outputs/diagnostics/mini-slalom-aug2'
AUG=ROOT/'outputs/course-pools/mini-slalom-aug2'
SOURCE=CONFIG/'mini_slalom_2_train_manifest.json'
SOURCE_EVAL=ROOT/'configs/eval/v62111_mini_slalom_selection2.json'
SOURCE_PROTOCOL=ROOT/'configs/eval/real100_v2_timed_protocol_v1.json'
SOURCE_QUALITY=CONFIG/'plant_quality_v1.yaml'
CHECKPOINT='/workspace/outputs/checkpoints/starscream-v6.21.1.1-plant-selection25-dagger/best-step-150819355-selection_suite_success-0.57875.pt'

def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')

def main():
    parent_manifest=json.loads(SOURCE.read_text())
    parents={r['family']:r for r in parent_manifest['records']}
    generated=json.loads((AUG/'manifest.json').read_text())['records']
    audits={r['name']:r for r in json.loads((OUT/'qualification.json').read_text())['records']}
    assert len(generated)==12 and len(audits)==12
    accepted=[r for r in generated if audits[r['name']]['qualified']]
    assert len([r for r in accepted if r['split']=='train'])>=6
    assert len([r for r in accepted if r['split']=='family_validation'])>=3

    train_records=deepcopy(parent_manifest['records'])
    for row in accepted:
        if row['split']!='train':continue
        parent=parents[row['family']]
        clone=deepcopy(parent)
        clone.update(name=row['name'],path=str(AUG/row['path']),slot=row['name'],
                     fingerprint=row['track_fingerprint'],
                     geometry_fingerprint=row['geometry_fingerprint'],
                     augmentation_parent=parent['name'],augmentation_seed=row['seed'],
                     augmentation_qualification=audits[row['name']])
        train_records.append(clone)
    train_manifest=dict(parent_manifest,records=train_records)
    train_manifest_path=OUT/'train-manifest.json'
    write(train_manifest_path,train_manifest)
    train_paths=[r['path'] for r in train_records]

    original_cal=json.loads(Path(json.loads(SOURCE_QUALITY.read_text())['dagger']['dagger_trajectory_quality']['calibration_source']).read_text())
    quality=deepcopy(json.loads(SOURCE_QUALITY.read_text())['dagger']['dagger_trajectory_quality'])
    for row in train_records[2:]:
        parent=parents[row['family']]
        original_cal['track_bounds'][row['path']]=deepcopy(original_cal['track_bounds'][parent['path']])
    calibration_path=OUT/'parent-transferred-quality-calibration.json'
    write(calibration_path,original_cal)
    quality['track_bounds']=original_cal['track_bounds']
    quality['calibration_source']=str(calibration_path)
    quality['calibration_sha256']=hashlib.sha256(calibration_path.read_bytes()).hexdigest()

    source_suite=json.loads(SOURCE_EVAL.read_text())
    hard=[]
    for r in source_suite['records']:
        hard.append(dict(r,family='hard_slalom'))
    val=[];deadlines={}
    for row in accepted:
        if row['split']!='family_validation':continue
        audit=audits[row['name']]['nominal']
        reference=float(audit['successful_lap_time_seconds'])
        assert reference>0
        family='aug_10_slalom' if row['family']=='10_slalom' else 'aug_4_slalom'
        val.append(dict(name=row['name'],path=str(AUG/row['path']),slot=row['name'],
                        family=family,fingerprint=row['geometry_fingerprint'],cells=[]))
        deadlines[row['name']]=math.floor(130*max(1.5*reference,reference+3.0))
    protocol=json.loads(SOURCE_PROTOCOL.read_text())
    hard_deadlines={r['name']:r['deadline_steps'] for r in protocol['records']
                    if r['slot'] in {'real100-hard-010','real100-hard-019'}}
    deadlines.update(hard_deadlines)
    assert len(val)==3 and len(hard_deadlines)==2
    families=sorted({r['family'] for r in val})
    weights={family:(0.4 if family.startswith('aug_') else 0.2) for family in families+['hard_slalom']}
    assert abs(sum(weights.values())-1)<1e-9
    suite=dict(schema='starscream-vision-selection-v1',source=source_suite['source'],
               source_sha256=source_suite['source_sha256'],
               method='disjoint teacher-qualified variants of two pretrain parents plus two hard slaloms',
               family_quotas={f:sum(r['family']==f for r in val+hard) for f in weights},
               family_weights=weights,records=val+hard)
    suite_path=OUT/'validation-suite.json'
    write(suite_path,suite)
    suite_sha=hashlib.sha256(suite_path.read_bytes()).hexdigest()

    common={
        'inherits':'mini_slalom_2_common.yaml',
        'dagger':{
            'rounds':24,'episodes_per_round':64,'updates_per_round':626,
            'dagger_online_replay_rows_per_round':8334,
            'online_replay_capacity':260000,
            'dagger_permanent_expert_capacity':52086,
            'dagger_permanent_expert_rounds':1,
            'dagger_dart_expert_rounds':1,
            'teacher_beta_schedule_rounds':3,
            'initial_checkpoint':CHECKPOINT,
            'dagger_initialization_stats_checkpoint':None,
            'resume_checkpoint':None,
            'learning_rate':3e-5,
            'dagger_learning_rate_schedule':{'__replace__':True,'type':'constant'},
            'track_manifest':str(train_manifest_path),
            'curriculum':{'tracks':train_paths},
            'evaluation_suite_manifest':str(suite_path),
            'evaluation_suite_sha256':suite_sha,
            'evaluation_suite_episodes_per_track':4,
            'evaluation_curriculum':{'tracks':[r['path'] for r in val+hard],
                                     'max_steps':6000},
            'evaluation_clean_deadlines':deadlines,
            'evaluation_workers':4,
            'evaluation_envs_per_worker':6,
        },
        'experiment_notes':{'augmented_two_course':{
            'parents':[r['name'] for r in parent_manifest['records']],
            'train_augmentations':len(train_records)-2,'heldout_augmentations':len(val),
            'hard_external_courses':[r['name'] for r in hard],
            'warm_start':CHECKPOINT,'fresh_replay':True,
            'budget':'24 rounds x 64 episodes; 626 updates/round; beta .35 by r3',
            'qualification':'nominal and randomized canonical MPCC pass; rejected variants excluded',
        }},
    }
    common_path=CONFIG/'mini_slalom_aug2_common.yaml'
    write(common_path,common)
    settings=[]
    for kind in ('broad','bounded'):
        run=f'starscream-v6.21.1.1-mini-slalom-aug2-{kind}-r24'
        dagger={'run_name':run,'tags':['mini-slalom-aug2',kind,'warm-start','async-dagger']}
        if kind=='bounded':dagger['dagger_trajectory_quality']=quality
        path=CONFIG/f'mini_slalom_aug2_{kind}.yaml'
        write(path,{'inherits':common_path.name,'dagger':dagger,
            'wandb':{'group':'v62111-mini-slalom-aug2-baselines',
                     'name':run,
                     'local_event_path':f'/workspace/outputs/logs/{run}.events.jsonl',
                     'local_full_event_path':f'/workspace/outputs/logs/{run}.full.events.jsonl',
                     'eval_metric_allowlist':['selection_suite_success','selection_suite_timely_success',
                        'selection_suite_clean_success','selection_suite_clean_timely_success',
                        'full_course_success','crash_rate','mean_gates','mean_steps','dagger_policy_version']}})
        d=stage_config(load_config(path),'dagger')['dagger']
        configure_selection_suite(d)
        assert configured_tracks(d)==tuple(train_paths)
        assert d['evaluation_episodes']==4*len(val+hard)
        assert (quality_config(d) is not None)==(kind=='bounded')
        assert d['dagger_async_pipeline'] and d['dagger_capture_updates']
        assert d['dagger_packed_transport'] and d['dagger_overlap_teacher_inference']
        settings.append(d)
    differing={k for k in settings[0].keys()|settings[1].keys()
               if settings[0].get(k)!=settings[1].get(k)}
    assert differing=={'run_name','tags','dagger_trajectory_quality'},differing
    print(json.dumps(dict(train=len(train_records),validation=len(val+hard),
        warm_start=CHECKPOINT,deadlines=deadlines,
        only_differences=sorted(differing)),indent=2))

if __name__=='__main__':main()
