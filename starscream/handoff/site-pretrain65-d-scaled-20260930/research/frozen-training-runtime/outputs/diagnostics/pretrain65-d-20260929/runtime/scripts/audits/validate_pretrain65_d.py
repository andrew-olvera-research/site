"""Fail-closed audit of scaled D lineage, splits, statistics and live smoke evidence."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch
import h5py

ROOT=Path('/workspace')
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/diagnostics/pretrain65-d-20260929'
CONFIG=ROOT/'configs/exp/v6.21.1.1/pretrain65_d_scaled.json'

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    from scripts.train_privileged_racing import stage_config, dagger_episode_tracks, dagger_episode_start_gates, dagger_episode_start_modes
    from starscream.evaluation_suite import configure_selection_suite
    from starscream.dagger_collection import collection_config, collection_beta
    c=json.loads(CONFIG.read_text()); s=c['dagger']
    d=json.loads((ROOT/'configs/exp/v6.21.1.1/collection_next_d_repeat.json').read_text())['dagger']
    old=json.loads((ROOT/'configs/exp/v6.21.1/update_fix_dagger.yaml').read_text())['dagger']
    allowed={'run_name','seed','rounds','episodes_per_round','updates_per_round','mpcc_build_root',
        'dagger_initialization_stats_checkpoint','evaluation_workers','evaluation_envs_per_worker',
        'evaluation_episodes','evaluation_clean_deadlines','top_k','tags','curriculum','track_manifest',
        'track_sampling_family_weights','dagger_replay_family_weights','dagger_permanent_expert_required_families',
        'mpcc_manifest_teacher_profile_controller_configs','mpcc_manifest_teacher_profile_planner_configs',
        'dagger_online_replay_rows_per_round','online_replay_capacity','dagger_permanent_expert_capacity',
        'evaluation_curriculum','evaluation_track_speed_commands'}
    changed={k for k in s.keys()|d.keys() if s.get(k)!=d.get(k)}
    assert not {k for k in changed-allowed if not k.startswith('evaluation_suite_')},changed-allowed
    assert s['initial_checkpoint'] is None and s['resume_checkpoint'] is None
    assert s['curriculum']==old['curriculum'] and s['track_manifest']==old['track_manifest']
    train=set(s['curriculum']['tracks']);assert len(train)==65
    manifest=json.loads(Path(s['track_manifest']).read_text())
    rows=[r for r in manifest['records'] if r['split']=='train']
    assert {r['path'] for r in rows}==train
    families={r['family'] for r in rows};assert len(families)==65
    for k in ('track_sampling_family_weights','dagger_replay_family_weights','dagger_permanent_expert_required_families'):
        assert set(s[k])==families
    assert s['model']==d['model'] and s['dagger_collection_control']==d['dagger_collection_control']
    assert collection_beta(.99,collection_config(s))==0
    assert all(s[k]==old[k] for k in ('rounds','episodes_per_round','updates_per_round','batch_size'))
    for key in ('dagger_online_replay_rows_per_round','online_replay_capacity','dagger_permanent_expert_capacity'):
        assert s[key]==math.ceil(d[key]*520/64)
    schedule=dagger_episode_tracks(tuple(s['curriculum']['tracks']),520,s,s['seed'])
    assert set(Counter(schedule).values())=={8}
    gates=dagger_episode_start_gates(schedule,s,s['seed'])
    modes=dagger_episode_start_modes(schedule,gates,s,s['seed'])
    assert .4<modes.count('expert_prefix')/520<.6
    resolved=stage_config(c,'dagger')['dagger'];configure_selection_suite(resolved)
    val=set(resolved['evaluation_curriculum']['tracks']);assert len(val)==25 and not train&val
    from starscream.env.procedural_tracks import geometry_fingerprint
    from starscream.env.tracks import load_track
    train_geometry={geometry_fingerprint(load_track(p)) for p in train}
    val_geometry={geometry_fingerprint(load_track(p)) for p in val}
    assert len(train_geometry)==65 and len(val_geometry)==25 and not train_geometry&val_geometry
    suite=json.loads(Path(s['evaluation_suite_manifest']).read_text())
    protocol=json.loads((ROOT/'configs/eval/real100_v2_timed_protocol_v1.json').read_text())
    timed={r['name']:r for r in protocol['records']}
    for r in suite['records']:
        assert digest(ROOT/r['path'])==timed[r['name']]['track_sha256']
        assert s['evaluation_clean_deadlines'][r['name']]==timed[r['name']]['deadline_steps']
    assert resolved['evaluation_episodes']==200
    assert set(s['evaluation_clean_deadlines'])==set(resolved['evaluation_suite_names'])
    assert resolved['evaluation_curriculum']['max_steps']==6000
    assert c['checkpoint']['monitor']==s['monitor']=='selection_suite_timely_success'
    stats=torch.load(s['dagger_initialization_stats_checkpoint'],map_location='cpu',weights_only=False)
    assert set(stats['track_counts'])==train and len(set(stats['track_counts'].values()))==1
    assert stats['feature_dim']==167 and stats['observation_contract']==s['observation_contract']
    assert stats['previous_action_feature_mapping']==s['previous_action_feature_mapping']
    assert stats['source_beta']==1 and stats['episodes']>=260 and stats['valid_dynamics_rows']>0
    for v in list(stats['normalizer'].values())+[stats['dynamics_target_mean'],stats['dynamics_target_std']]:
        assert np.isfinite(v).all()
    assert (np.asarray(stats['normalizer']['std'])>0).all()
    assert (OUT/'tests.exit').read_text().strip()=='0'
    assert json.loads((OUT/'smoke.complete.json').read_text())['state']=='passed'
    payload=torch.load(OUT/'smoke/checkpoints/pretrain65-d-smoke/latest.pt',map_location='cpu',weights_only=False)
    assert payload['round']==3 and payload['optimizer']['state']
    assert payload['dagger_replay']['contract']['collection_control_v1']==s['dagger_collection_control']
    np.testing.assert_array_equal(payload['normalizer']['mean'],stats['normalizer']['mean'])
    from starscream.dagger_replay import DaggerReplayStore
    metadata=payload['dagger_replay']
    store=DaggerReplayStore.create(OUT/'restore-check',metadata['contract'],resume_metadata=metadata)
    shards=store._committed_shards(3)
    with h5py.File(shards[-1]) as h:
        templates={k:np.empty((0,*v.shape[1:]),v.dtype) for k,v in h['online'].items()}
    actual=store.restore_pool('online',templates,capacity=8000,committed_round=3)
    for key in templates:
        chunks=[]
        for shard in shards:
            with h5py.File(shard) as h:chunks.append(h['online'][key][:])
        np.testing.assert_array_equal(actual[key],np.concatenate(chunks)[-8000:])
    events=[json.loads(l) for l in (OUT/'smoke.full.events.jsonl').read_text().splitlines()]
    metrics=[e['metrics'] for e in events if 'train/round' in e.get('metrics',{})]
    assert [r['train/round'] for r in metrics]==[1,2,3]
    evaluation=[e['metrics'] for e in events if 'eval/dagger_policy_version' in e.get('metrics',{})]
    assert [r['eval/dagger_policy_version'] for r in evaluation]==[1,2,3]
    for r in evaluation:
        assert all(f'eval/track/{name}/full_course_success' in r for name in resolved['evaluation_suite_names'])
        assert all(np.isfinite(r[f'eval/selection_suite_{measure}']) for measure in ('success','timely_success','clean_timely_success'))
    assert all(np.isfinite(r['train/loss']) for r in metrics)
    active=metrics[1:]
    assert sum(r.get('train/collection/active_steps',0) for r in active)>0
    assert sum(r.get('train/collection/active_teacher_steps',0) for r in active)==0
    files=[CONFIG,Path(s['track_manifest']),Path(s['evaluation_suite_manifest']),Path(s['dagger_initialization_stats_checkpoint']),ROOT/'configs/eval/real100_v2_timed_protocol_v1.json']
    files += [Path(p) for p in train|val]
    files += list((ROOT/'starscream').rglob('*.py')) + list((ROOT/'scripts').rglob('*.py'))
    report=dict(state='passed',train_courses=65,selection_courses=25,evaluation_episodes=200,
        evaluation_cadence='every round, fixed paired seeds',changed_D_keys=sorted(changed),
        normalization={k:stats[k] for k in ('rows','episodes','accepted_episodes','rejected_episodes','valid_dynamics_rows','feature_dim')},
        smoke=dict(rounds=3,resume=True,all_train_courses=True,all_selection_courses=True,replay_arrays_verified=len(templates)),
        tests=(OUT/'tests.log').read_text().strip(),sha256={str(p):digest(p) for p in files})
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sha256'},indent=2))

if __name__=='__main__':main()
