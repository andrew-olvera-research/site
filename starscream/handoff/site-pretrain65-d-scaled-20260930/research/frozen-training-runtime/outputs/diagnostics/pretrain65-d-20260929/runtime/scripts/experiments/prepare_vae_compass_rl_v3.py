"""Fail-closed preparation of equal-budget, frozen-proposal PPO branches."""
from copy import deepcopy
from pathlib import Path
import json
import yaml
from scripts.train_privileged_racing import load_config, stage_config, parse_stage
from starscream.course_model.training import atomic_json

ROOT=Path('outputs/course-model/v3/matched-panel')
SOURCE='/workspace/outputs/checkpoints/starscream-v6.11-policy-transition-ppo-3m/latest.pt'
ARMS={'source':None,'vae':'anchored_r0.5','random':'anchored_random_r0.5','classical':'classical_r0.5'}


def main():
    records={r['name']:r for r in json.loads((ROOT/'comparison.json').read_text())['records']}
    expert={r['name']:r['summary'] for r in json.loads(Path('outputs/audits/vae-v3-mpcc-randomized/manifest.json').read_text())['rows']}
    behavior={r['name']:r['metrics'] for r in json.loads(Path('outputs/audits/vae-v3-policy-randomized.json').read_text())['rows']}
    for name in ('source_exact_canonical',*[n for n in ARMS.values() if n]):
        assert records[name]['static_valid'],name
        assert expert[name]['success_rate']>=.75,(name,'MPCC threshold',expert[name])
        assert behavior[name]['full_course_success']>=.5,(name,'policy competence',behavior[name])
    base=stage_config(load_config(Path('configs/exp/v6.11/policy_transition_ppo_3m.yaml')),'ppo')
    files=[]
    for arm,candidate in ARMS.items():
        cfg=deepcopy(base);s=cfg['ppo'];run=f'starscream-vae-v3-compass-{arm}-ppo-196k'
        swift='/workspace/starscream/assets/tracks/swift_champion_2022_exact.yaml'
        tracks=[swift] if candidate is None else [swift,records[candidate]['path']]
        s.update(run_name=run,initial_checkpoint=SOURCE,actor_optimizer_initial_checkpoint=SOURCE,
                 critic_initial_checkpoint=SOURCE,initial_environment_steps=0,
                 target_environment_steps=196608,cycles=10000,seed=2026090711,
                 track_manifest=None,ppo_track_sampling_family_weights={},
                 evaluation_condition_on_manifest_speed=False,mpcc_use_manifest_speed=False,
                 ppo_track_sampling_weights=({'swift_champion_2022_exact':1.} if candidate is None
                     else {'swift_champion_2022_exact':.5,candidate:.5}),
                 evaluation_interval=6,reporting_evaluation_interval=100000,
                 evaluation_episodes_per_track=16,checkpoint_metric_source='active',top_k=1)
        # No target geometry, target suffix reset or online generator in rollouts.
        stage=deepcopy(s['reporting_evaluation_curriculum'])
        if isinstance(stage,list):stage=stage[0]
        for key in ('track_manifest','real_course_suite','track_families'):
            stage.pop(key,None)
        stage.update(name='frozen_compass_local_test',tracks=tracks,allow_archived_task_resets=False)
        s['curriculum']=[deepcopy(stage)];s['evaluation_curriculum']=[deepcopy(stage)]
        s['online_manifold_curriculum']={'enabled':False}
        cfg['wandb'].update(run_name=run,group='vae-v3-equal-budget-transfer')
        cfg['checkpoint'].update(run_name=run,top_k=1)
        for key in ('model','observation_contract','control_hz','route_gate_count','reward',
                    'dynamics_randomization','ppo_dynamics_weight','rollout_envs',
                    'ppo_strict_likelihood_math','collector_backend'):
            assert s[key]==base['ppo'][key],key
        assert s['control_hz']==130 and s['route_gate_count']==6
        assert s['collector_backend']=='process' and s['dynamics_randomization']['enabled']
        assert len(parse_stage(stage).tracks)==len(tracks)
        path=Path(f'configs/exp/course_model/compass_v3_{arm}_ppo.yaml')
        if path.exists():raise FileExistsError(path)
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        files.append(str(path))
    atomic_json(ROOT/'rl_preflight.json',dict(source=SOURCE,configs=files,
        steps_per_arm=196608,expert_minimum=.75,policy_minimum=.5,
        interpretation='pilot; exact CDRA is a development target, never used for checkpoint selection'))
    print('validated',files)


if __name__=='__main__':main()
