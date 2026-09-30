"""Build the scaled run only from a hash-locked, fully qualified corpus.

No policy launch. Keep v6.19 architecture/loss and scale collection, updates and
retained per-round rows together. A round count is not a per-course step floor.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.audits.prepare_v619_pool import digest
from starscream.course_model.training import atomic_json


def build_config(base, manifest, pool, rounds, episodes_per_course, hardware):
    c=deepcopy(base);s=c['dagger']
    train=[r for r in manifest['records'] if r['split']=='train']
    val=[r for r in manifest['records'] if r['split']=='validation']
    report=[r for r in manifest['records'] if r['split']=='report']
    if len(train)!=35 or len(val)!=35:raise ValueError('requires complete 35/35 corpus')
    if rounds<1 or episodes_per_course<1:raise ValueError('positive training budget required')
    name=f'starscream-v6.20.1-transition-corpus-dagger-r{rounds}'
    volume=episodes_per_course*len(train)
    ratio=volume/base['dagger']['episodes_per_round']
    weights={r['family']:1. for r in train}
    for k in list(s):
        if k.startswith(('mpcc_family_', 'offline_', 'selection_track_')):s.pop(k)
    for k in ('track','tracks','data','track_families','minimum_qualified_speed'):s.pop(k,None)
    s.update(run_name=name,rounds=rounds,episodes_per_round=volume,
        updates_per_round=math.ceil(base['dagger']['updates_per_round']*ratio),
        dagger_online_replay_rows_per_round=math.ceil(base['dagger']['dagger_online_replay_rows_per_round']*ratio),
        online_replay_capacity=math.ceil(base['dagger']['online_replay_capacity']*ratio),
        dagger_permanent_expert_capacity=math.ceil(base['dagger']['dagger_permanent_expert_capacity']*ratio),
        initial_checkpoint=None,resume_checkpoint=None,dagger_anchor_checkpoint=None,
        seed=2026091462,evaluation_seed=2034091462,reporting_evaluation_seed=2035091462,
        monitor='full_course_success',monitor_mode='max',top_k=5,
        track_manifest=str(pool/'manifest.json'),track_split='train',qualified_tracks_only=True,
        dagger_initialization_stats_checkpoint=str(pool/'normalization.pt'),
        track_sampling_family_weights=weights,dagger_replay_family_weights=weights,
        dagger_permanent_expert_required_families=list(weights),
        reliability_family_aliases={r['family']:r['grammar'] for r in train},
        dagger_transition_start_gate_indices={},
        mpcc_build_root='/tmp/starscream-v6201-training-runtime51',
        racing_line_cache=str(pool/'racing-lines'),
        evaluation_episodes=4*len(val),reporting_evaluation_episodes=16*len(report),
        reporting_evaluation_interval=10,
        tags=['v6.20.1','scratch','transition-corpus','runtime-plant-fixed',
              'legacy-privileged103','h3','route6','130hz','macro-completion-selection'])
    s.update(deepcopy(manifest['mandatory_teacher_overrides']))
    s.update(hardware)
    for key,rows in [('curriculum',train),('evaluation_curriculum',val),('reporting_evaluation_curriculum',report)]:
        stage=s[key]
        for k in ('track_manifest','track_split','track_families','qualified_tracks_only',
                  'minimum_qualified_speed','real_course_suite','track_limit'):
            stage.pop(k,None)
        stage.update(name='v6201_'+key,tracks=[r['path'] for r in rows],target_gates=64,
            max_steps=3000,rollout_laps=1,random_gate=False,fixed_start_gate_index=0,
            manifest_speed_scale_range=None,allow_archived_task_resets=False)
    c['checkpoint'].update(run_name=name,monitor='full_course_success',mode='max',top_k=5)
    c['wandb'].update(run_name=name,group='starscream-v6.20.1')
    c['wandb']['eval_metric_allowlist']=['full_course_success','crash_rate','mean_gates',
        'p*','track/*','family/*','minimum_family*','successful_*','mean_speed*','reporting/*']
    c['wandb']['train_metric_allowlist']=list(dict.fromkeys(c['wandb']['train_metric_allowlist']+
        ['updates_seconds','evaluation_seconds','replay_prepare_seconds','sampling_plan*','collection_last_call_host/*']))
    return c


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pool',type=Path,default=Path('/workspace/outputs/course-pools/v6201-transitions-r6'))
    p.add_argument('--base',type=Path,default=Path('/workspace/configs/exp/v6.19/green_inspired_dagger_r60.yaml'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=240)
    p.add_argument('--episodes-per-course',type=int,default=8)
    a=p.parse_args()
    lock=json.loads((a.pool/'manifest.lock.json').read_text())
    for path,sha in lock.items():
        if digest(Path(path))!=sha:raise ValueError(f'changed locked data: {path}')
    review=json.loads((a.pool/'review.json').read_text())
    if not review['freeze_ready']:raise ValueError('corpus not ready')
    base=yaml.safe_load(a.base.read_text())
    manifest=json.loads((a.pool/'manifest.json').read_text())
    hardware=yaml.safe_load(Path('/workspace/configs/hardware/dagger_local_16c.yaml').read_text())
    c=build_config(base,manifest,a.pool,a.rounds,a.episodes_per_course,hardware)
    from scripts.train_privileged_racing import configured_tracks
    from starscream.privileged_racing import PrivilegedMLPPolicy,FeatureNormalizer
    import torch
    import numpy as np
    s=c['dagger']
    assert len(configured_tracks(s))==35
    assert s['model']==base['dagger']['model'] and s['dynamics_weight']==.15
    stats=torch.load(a.pool/'normalization.pt',weights_only=False,map_location='cpu')
    assert not {'model','optimizer'}&stats.keys()
    norm=FeatureNormalizer.from_state_dict(stats['normalizer'])
    actor=PrivilegedMLPPolicy(**s['model']).eval();actor.bind_feature_normalizer(norm)
    assert actor.input_dim==103 and actor.context_steps==3 and actor.action_chunk_steps==1
    assert np.isfinite(norm.std).all()
    n=35;total=n*1200000
    audit=dict(dataset_lock_sha256=digest(a.pool/'manifest.lock.json'),
        primary_metric='full_course_success (equal 4 episodes/course = macro completion)',
        rounds=a.rounds,episodes_per_round=s['episodes_per_round'],updates_per_round=s['updates_per_round'],
        target_collected_steps=total,target_collected_steps_per_course=1200000,
        required_mean_steps_per_episode=total/(a.rounds*s['episodes_per_round']),
        warning='No guarantee of 1.2M per course: measure actual exposure and valid labels; dynamic sampling and early crashes change episode lengths.',
        parameters=sum(p.numel() for p in actor.parameters()),scratch_weights=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(yaml.safe_dump(c,sort_keys=False))
    audit['config_sha256']=digest(a.output)
    atomic_json(a.output.with_suffix('.preflight.json'),audit)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
