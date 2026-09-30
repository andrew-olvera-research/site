"""Prepare an explicit continuation; keep protected geometry outside training."""
import copy,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.train_privileged_racing import load_config,parse_stage,stage_config
from starscream.env.real_course_suite import load_active_real_course_suite
ROOT=Path('/workspace')
def build(lanes=280,group=20,horizon=768):
 c=stage_config(load_config(ROOT/'configs/exp/v6.21.rl.7/b_time_mild.yaml'),'ppo');s=c['ppo']
 name='starscream-v6.21.rl.7.4-unified-real50-100m';source=ROOT/'outputs/checkpoints/starscream-v6.21.rl.7.3-b-time-mild/best-step-012042240-selection_score-124.05945.pt'
 assert source.is_file()
 s.update(run_name=name,track='v621-train45-bank-real50-validation',seed=2026091708,initial_checkpoint=str(source),actor_optimizer_initial_checkpoint=str(source),restore_actor_optimizer=True,critic_initial_checkpoint=str(source),restore_critic_optimizer=True,resume_online_course_bank=False,initialize_online_course_bank_from_checkpoint=True,ppo_critic_warmup_cycles=2,initial_environment_steps=0,target_environment_steps=100_000_000,cycles=math.ceil(100_000_000/(lanes*horizon))+1,actor_learning_rate=2e-6,rollout_envs=lanes,episodes_per_cycle=lanes,ppo_envs_per_worker=group,ppo_rollout_window_steps=horizon,ppo_window_max_host_bytes=3*1024**3,host_evaluation_graph=True,evaluation_workers=32,evaluation_envs_per_worker=4,ppo_release_host_memory=True,evaluation_episodes=400,reporting_evaluation_episodes=400,evaluation_interval=math.ceil(10_000_000/(lanes*horizon)),reporting_evaluation_interval=2*math.ceil(10_000_000/(lanes*horizon)),evaluation_on_stage_budget=True,evaluation_seed=2036091708,reporting_evaluation_seed=2036091708,evaluation_stage_seed_stride=0,reporting_evaluation_stage_seed_stride=0,evaluation_condition_on_manifest_speed=False,evaluation_matched_track_seeds=False,checkpoint_rank_curriculum_stages=[0,1,2],monitor='full_course_success',monitor_mode='max',top_k=5)
 s['ppo_fused_window_inputs']=True; s['stage_reward_overrides']={};s['reward']['time_penalty_per_second']=.1
 stages=[]
 for label,command,band,budget in [('reliability',16.5,[14.,18.],50_000_000),('bridge',18.,[14.,20.],25_000_000),('speed_with_retention',20.,[16.,22.],25_000_000)]:
  stage=copy.deepcopy(s['curriculum'][0]);stage.update(name=label,target_speed=command,target_speed_range=band,manifest_speed_scale_range=None,minimum_environment_steps=budget);stages.append(stage)
 s['curriculum']=stages;s['ppo_online_course_bank']['stage_warmup_windows']=5
 tracks,metadata=load_active_real_course_suite(ROOT/'configs/eval/v6_21_holdout_eval_50.yaml');assert len(tracks)==50
 ev=copy.deepcopy(s['evaluation_curriculum'])
 for k in ['track_manifest','track_split','qualified_tracks_only','target_speed_range']:ev.pop(k,None)
 ev.update(name='real50_validation_command16_5',tracks=[str(p) for p in tracks],target_speed=16.5,max_steps=6000,manifest_speed_scale_range=None)
 s['evaluation_curriculum']=ev;s['reporting_evaluation_curriculum']={**copy.deepcopy(ev),'name':'real50_diagnostic_command20','target_speed':20.}
 c['experiment_notes']=dict(recipe='B reward throughout; continuation from best B actor/critic/optimizers and admitted bank; 50/25/25M overlapping command curriculum; real50 is validation, not untouched test.',production_mapping='300M -> 150/75/75M; 400M -> 200/100/100M. Hypothesis, not established optimal scaling.',normalization='Preserve source actor normalizer. No data edits in this RL research run.')
 c['wandb'].update(enabled=True,mode='online',group='v6.21.rl.7.4',run_name=name,local_event_path=f'/workspace/outputs/logs/{name}.events.jsonl')
 c=stage_config(c,'ppo');path=ROOT/'configs/exp/v6.21.rl.7/unified_real50_100m.yaml';path.write_text(json.dumps(c,indent=2)+'\n');return c
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--lanes',type=int,default=280);ap.add_argument('--group',type=int,default=20);ap.add_argument('--horizon',type=int,default=768);a=ap.parse_args();c=build(a.lanes,a.group,a.horizon);print(c['ppo']['run_name'])


