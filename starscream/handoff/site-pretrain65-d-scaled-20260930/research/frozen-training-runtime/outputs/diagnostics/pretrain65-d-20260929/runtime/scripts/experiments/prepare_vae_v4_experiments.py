"""Freeze six step/direction/cadence ablations after physical qualification."""
from copy import deepcopy
from pathlib import Path
import json
import yaml
import argparse
from scripts.train_privileged_racing import load_config,stage_config,parse_stage
from starscream.course_model.training import atomic_json
from starscream.env.tracks import load_track
from starscream.env.procedural_tracks import geometry_fingerprint

ROOT=Path('outputs/course-model/v4/rays')
SOURCE='/workspace/outputs/checkpoints/starscream-v6.11-policy-transition-ppo-3m/latest.pt'
SWIFT='/workspace/starscream/assets/tracks/swift_champion_2022_exact.yaml'
ARMS=(('small','vae',1,False,False),('hyper','vae',1,True,True),
      ('far_vae','vae',4,False,False),('far_classical','classical',4,False,False),
      ('online_vae','vae',1,True,False),('online_classical','classical',1,True,False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank-only',action='store_true');args=parser.parse_args()
    rows=json.loads((ROOT/'screen.json').read_text())['records']
    expert={r['name']:r for r in json.loads(Path('outputs/audits/vae-v4-mpcc/manifest.json').read_text())['rows']}
    source_expert=json.loads(Path('outputs/audits/vae-v4-source-mpcc/manifest.json').read_text())['rows'][0]
    behavior={r['name']:r['metrics'] for r in json.loads(Path('outputs/audits/vae-v4-policy.json').read_text())['rows']}
    source_row=dict(name='swift_champion_2022_exact',path=SWIFT,family='source',split='train',valid=True,
                    operation='identity',geometry_fingerprint=geometry_fingerprint(load_track(SWIFT)))
    records=[source_row]+rows
    for row in records:
        audit=source_expert if row['name']==source_row['name'] else expert[row['name']]
        assert audit['fingerprint']==row['geometry_fingerprint'],row['name']
        assert audit['summary']['success_rate']>=.75,(row['name'],audit['summary'])
        row.update(qualified_speed_mps=16.5,operation=row.get('method','identity'),curriculum_completion_gates=7,
            mpcc_admission=dict(rl_eligible=True,teacher_labelable=False,robustly_labelable=False,
                evidence=audit['trace_directory'],scope='pilot canonical-start randomized4; not robust certification'),
            dynamic_qualification=audit['summary'])
    bank=ROOT/'qualified_bank.json';atomic_json(bank,dict(schema='starscream-goal-conditioned-task-manifest-v1',records=records))
    if args.bank_only:return
    indexed={r['name']:r for r in records}
    base=stage_config(load_config(Path('configs/exp/course_model/compass_v3_vae_ppo.yaml')),'ppo')
    files=[]
    for arm,method,scale,online,hyper in ARMS:
        cfg=deepcopy(base);s=cfg['ppo'];run=f'starscream-vae-v4-{arm}-ppo-524k'
        frontier=f'v4_{method}_x{scale}'
        assert behavior[frontier]['full_course_success']>=.25,frontier
        tracks=[SWIFT,indexed[frontier]['path']]
        s.update(run_name=run,track=run,tracks=tracks,initial_checkpoint=SOURCE,
            actor_optimizer_initial_checkpoint=SOURCE,critic_initial_checkpoint=SOURCE,
            initial_environment_steps=0,target_environment_steps=524288,cycles=10000,
            seed=2026090911,track_manifest=None,qualified_tracks_only=False,
            evaluation_condition_on_manifest_speed=False,mpcc_use_manifest_speed=False,
            ppo_track_sampling_weights={'swift_champion_2022_exact':.5,frontier:.5},
            ppo_track_sampling_family_weights={},evaluation_interval=2 if hyper else 6,
            evaluation_episodes_per_track=16,reporting_evaluation_interval=100000,
            checkpoint_metric_source='active',top_k=1)
        stage=deepcopy(s['curriculum'][0]);stage.update(name=run,tracks=tracks,
            minimum_environment_steps=999999999,allow_archived_task_resets=False)
        s['curriculum']=[deepcopy(stage)];s['evaluation_curriculum']=[deepcopy(stage)]
        s['online_manifold_curriculum']={'enabled':False}
        if online:
            # Geometry is generated and frozen before training; admission is
            # policy-dependent online, only between rollout/update transactions.
            s['online_manifold_curriculum']=dict(enabled=True,restore_state_from_checkpoint=False,
                ordered_geometry_ladder=True,goal_conditioned_manifest=True,
                source_name='swift_champion_2022_exact',target_name='multigp_cdra_2026_reconstructed',
                target_track='/workspace/starscream/assets/tracks/multigp_cdra_2026_reconstructed.yaml',
                active_manifest_path=f'/workspace/outputs/online-manifold/{run}/active/manifest.json',
                manifests=[str(bank.resolve())],initial_active_names=['swift_champion_2022_exact',frontier],
                initial_frontier_name=frontier,candidate_names=[f'v4_{method}_x4',f'v4_{method}_x12'],
                candidate_lookahead=2 if hyper else 1,maximum_active_tracks=4,
                required_consecutive_passes=1,warmup_cycles=2 if hyper else 6,
                decision_interval_cycles=2 if hyper else 6,probe_episodes_per_track=16,
                matched_track_seeds=True,probe_seed=2026090912,probe_target_speed=16.5,
                prune_preceding_candidates_on_expand=True,
                thresholds=dict(frontier_success_floor=.125 if hyper else .25,
                    frontier_success_ceiling=.25 if hyper else .5,mastery_success=.25 if hyper else .5,
                    minimum_gate_fraction=.35 if hyper else .5,source_retention_floor=.65,
                    anchor_weight=.4,interior_weight=.1,frontier_weight=.5))
        cfg['wandb'].update(run_name=run,group='vae-v4-step-direction-cadence')
        cfg['checkpoint'].update(run_name=run,top_k=1)
        for key in ('model','observation_contract','control_hz','route_gate_count','reward',
                    'dynamics_randomization','ppo_dynamics_weight','ppo_strict_likelihood_math','rollout_envs'):
            assert s[key]==base['ppo'][key],key
        assert len(parse_stage(stage).tracks)==2
        path=Path(f'configs/exp/course_model/compass_v4_{arm}.yaml')
        if path.exists():raise FileExistsError(path)
        path.write_text(yaml.safe_dump(cfg,sort_keys=False));files.append(str(path))
    atomic_json(ROOT/'experiments.json',dict(configs=files,source=SOURCE,steps_per_arm=524288,
        purpose='frozen generator banks; online cadence; hyper is a combined stress arm'))
    print('prepared',files)

if __name__=='__main__':main()
