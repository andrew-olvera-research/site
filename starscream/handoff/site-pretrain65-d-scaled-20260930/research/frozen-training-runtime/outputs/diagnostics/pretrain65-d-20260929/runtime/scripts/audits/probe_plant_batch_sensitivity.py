"""600-step diagnostic: identical plant/seed, teacher batch shape 1 versus 16."""
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_vision_student import load_setup, environment, parse_stage, sample_curriculum_spawn, SensorHistory, ppo_normalized_to_ctbr


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--second-batch',type=int,default=16);args=parser.parse_args()
    torch.set_num_threads(1)
    config=json.loads(Path('configs/exp/v6.21.1.1/plant_ekf_readout_student_async.json').read_text())
    _,settings,source,_,teacher,norm,_=load_setup(config)
    teacher.cuda().eval()
    stage=parse_stage(source['evaluation_curriculum'])
    record=next(r for r in json.loads(Path('configs/eval/v6_22_real100_hard_v2.manifest.json').read_text())['records'] if r['slot']=='real100-hard-021')
    seed=config['evaluation_seed']+int(hashlib.sha256(record['slot'].encode()).hexdigest()[:8],16)%1000000
    envs=[];sensors=[];rows=[]
    mean=torch.as_tensor(norm.mean,device='cuda');std=torch.as_tensor(norm.std,device='cuda')
    try:
        for _ in range(2):
            env=environment(record['path'],config,settings,stage);envs.append(env)
            spawn=sample_curriculum_spawn(env.track,stage,seed=seed,episode_index=0)
            obs,_=env.reset(seed=seed,options=dict(state=spawn.state,gate_index=spawn.gate_index,route_plan_total_gates=len(env.track.gates)))
            sensor=SensorHistory(seed,'ekf',settings);sensor.append(obs);sensors.append(sensor)
        for step in range(600):
            truths=[sensor.arrays()[2] for sensor in sensors]
            actions=[]
            with torch.inference_mode():
                for t,batch in zip(truths,(1,args.second_batch)):
                    x=(torch.as_tensor(t[None],device='cuda')-mean)/std
                    actions.append(teacher(x.repeat(batch,1,1),torch.full((batch,),16.5,device='cuda'))[0].cpu().numpy())
                x=(torch.as_tensor(truths[0][None],device='cuda')-mean)/std
                same16=teacher(x.repeat(16,1,1),torch.full((16,),16.5,device='cuda'))[0].cpu().numpy()
            rows.append(dict(step=step,same_state_batch_action_max=float(np.max(np.abs(actions[0]-same16))),
                             state_max=float(np.max(np.abs(truths[0][-1,:19]-truths[1][-1,:19]))),
                             position_difference_m=float(np.linalg.norm(truths[0][-1,:3]-truths[1][-1,:3])),
                             velocity_difference_mps=float(np.linalg.norm(truths[0][-1,3:6]-truths[1][-1,3:6])),
                             action_max=float(np.max(np.abs(actions[0]-actions[1])))))
            done=False
            for env,sensor,action in zip(envs,sensors,actions):
                obs,_,term,trunc,_=env.step(ppo_normalized_to_ctbr(action,settings));sensor.append(obs);done|=term or trunc
            if done:break
        result=dict(slot=record['slot'],seed=seed,steps=len(rows),second_batch=args.second_batch,
                    same_state_batch_action_max=max(r['same_state_batch_action_max'] for r in rows),
                    milestones=rows[::100]+rows[-1:],gates=[e.tracker.passed_count for e in envs])
        Path(f'outputs/diagnostics/plant-student-saved-audit/batch-sensitivity-{args.second_batch}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        for env in envs:env.close()


if __name__=='__main__':main()
