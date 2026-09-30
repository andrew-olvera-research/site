"""Closed-loop bitwise qualification of packed window inputs."""
import copy,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/workspace')
from scripts.train_privileged_racing import load_config,stage_config,load_policy_checkpoint,parse_stage,ProcessRaceCollector,PrivilegedValue
from starscream.ppo_windows import release_rollout_storage

def main():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 s=stage_config(load_config(Path('configs/exp/v6.21.rl.7/unified_real50_100m.yaml')),'ppo')['ppo'];s.update(rollout_envs=40,episodes_per_cycle=40,ppo_envs_per_worker=10,ppo_rollout_window_steps=64,ppo_inference_cohorts=1)
 actor,norm,payload,_=load_policy_checkpoint(s['initial_checkpoint'],'cuda');actor.eval().set_exact_likelihood_mode(True)
 critic=PrivilegedValue(input_dim=actor.context_steps*actor.input_dim+4,hidden_dim=s['critic_hidden_dim']).cuda().eval();critic.load_state_dict(payload['critic']);outputs=[]
 for enabled in [False,True]:
  settings={**s,'ppo_fused_window_inputs':enabled};collector=ProcessRaceCollector(actor,norm,settings,parse_stage(s['curriculum'][0]),'cuda',workers=40,sampling_prefix='ppo')
  try:
   torch.manual_seed(2026091708);np.random.seed(2026091708)
   rollout,results,_=collector.collect(critic,episodes=40,seed_base=2027091708)
   outputs.append({k:v.clone() for k,v in rollout.items() if isinstance(v,torch.Tensor)})
   release_rollout_storage(rollout)
  finally:collector.close()
 assert outputs[0].keys()==outputs[1].keys()
 differences={k:float((outputs[0][k]-outputs[1][k]).abs().max()) for k in outputs[0] if not torch.equal(outputs[0][k],outputs[1][k])}
 assert not differences,differences
 out=Path('outputs/diagnostics/v621-rl74-throughput/input-transfer-equivalence.json');out.write_text(json.dumps(dict(transitions=2560,fields=list(outputs[0]),bitwise_equal=True),indent=2));print('All rollout fields bitwise equal across 2560 closed-loop transitions')
if __name__=='__main__':main()
