"""Compare teacher and student simulator contracts on identical reset/action steps."""
import json
from pathlib import Path
import numpy as np
import torch

from scripts.train_privileged_racing import make_env, make_reward, reset_env, ppo_observation_features, ppo_normalized_to_ctbr
from scripts.train_vision_student import environment, load_setup, parse_stage, sample_curriculum_spawn

c=json.loads(Path('configs/exp/v6.21.1.1/plant_ekf_readout_student_async.json').read_text())
_,settings,source,_,policy,normalizer,_=load_setup(c)
stage=parse_stage(source['evaluation_curriculum']); track=stage.tracks[2];seed=2034091462+123
a=make_env(settings,track=track,reward=make_reward(settings,16.5))
b=environment(track,c,settings,stage)
try:
    oa,_=reset_env(a,stage,seed=seed,episode_index=0)
    spawn=sample_curriculum_spawn(b.track,stage,seed=seed,episode_index=0)
    ob,_=b.reset(seed=seed,options=dict(state=spawn.state,gate_index=spawn.gate_index,
        route_plan_total_gates=min(stage.target_gates,len(b.track.gates))))
    policy.cuda().eval()
    differences=[]
    for i in range(3000):
        fa=ppo_observation_features(oa,settings);fb=ppo_observation_features(ob,settings)
        differences.append(float(np.max(np.abs(fa-fb))))
        with torch.inference_mode():
            x=torch.as_tensor((fa-normalizer.mean)/normalizer.std,device='cuda').float().reshape(1,1,-1).expand(1,3,-1)
            action=policy(x,torch.tensor([16.5],device='cuda'))[0].cpu().numpy()
        command=ppo_normalized_to_ctbr(action,settings)
        oa,_,terminated_a,truncated_a,_=a.step(command)
        ob,_,terminated_b,truncated_b,_=b.step(command)
        assert (terminated_a,truncated_a,a.tracker.passed_count)==(terminated_b,truncated_b,b.tracker.passed_count)
        if terminated_a or truncated_a or a.tracker.passed_count==len(a.track.gates):
            break
    print(json.dumps(dict(feature_dim=len(fa),steps=len(differences),
                          maximum_feature_difference=max(differences),
                          gates=a.tracker.passed_count,terminated=terminated_a,
                          plant_a=oa['privileged']['plant_settings'][:5].tolist(),
                          plant_b=ob['privileged']['plant_settings'][:5].tolist())))
finally:
    a.close();b.close()
