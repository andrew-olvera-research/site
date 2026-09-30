"""Two bounded diagnostic episodes, with explicit miss/retry and dwell telemetry."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_vision_student import (load_setup, environment, parse_stage,
    sample_curriculum_spawn, SensorHistory, ppo_normalized_to_ctbr)


def main():
    torch.set_num_threads(1)
    root=Path('outputs/vision-distillation/v62111-plant-ekf-readout-async-48r')
    saved=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    config=saved['config']
    _,settings,source,_,teacher,norm,student=load_setup(config)
    student.load_state_dict(saved['model']);student.cuda().eval();teacher.cuda().eval()
    stage=parse_stage(source['evaluation_curriculum'])
    records=json.loads(Path('configs/eval/v6_22_real100_hard_v2.manifest.json').read_text())['records']
    record=next(r for r in records if r['slot']=='real100-hard-021')
    seed=config['evaluation_seed']+int(hashlib.sha256(record['slot'].encode()).hexdigest()[:8],16)%1000000
    out=Path('outputs/diagnostics/plant-student-saved-audit');out.mkdir(exist_ok=True,parents=True)
    summary=[]
    for arm in ('teacher','student'):
        env=environment(record['path'],config,settings,stage)
        try:
            spawn=sample_curriculum_spawn(env.track,stage,seed=seed,episode_index=0)
            obs,_=env.reset(seed=seed,options=dict(state=spawn.state,gate_index=spawn.gate_index,route_plan_total_gates=len(env.track.gates)))
            sensor=SensorHistory(seed,'ekf',settings);sensor.append(obs)
            rows=[];events=[];last_pass=0;max_dwell=0
            mean=torch.as_tensor(norm.mean,device='cuda');std=torch.as_tensor(norm.std,device='cuda')
            speed=torch.tensor([16.5],device='cuda')
            for step in range(10000):
                f,m,t=sensor.arrays()
                with torch.inference_mode():
                    if arm=='teacher':action=teacher((torch.as_tensor(t[None],device='cuda')-mean)/std,speed)[0].cpu().numpy()
                    else:action=student(torch.as_tensor(f[None],device='cuda'),torch.as_tensor(m[None],device='cuda').float(),speed)['action'][0].cpu().numpy()
                gate=env.track.gates[env.tracker.index];state=np.asarray(obs['state']);passed=env.tracker.passed_count
                side=float((state[:3]-gate.position)@gate.normal)
                rows.append([step,passed,env.tracker.index,*state[:3],*state[7:10],side,*action])
                obs,_,terminated,truncated,info=env.step(ppo_normalized_to_ctbr(action,settings))
                new_side=float((np.asarray(obs['state'])[:3]-gate.position)@gate.normal)
                changed=env.tracker.passed_count!=passed
                if changed:
                    max_dwell=max(max_dwell,step+1-last_pass);last_pass=step+1
                    events.append(dict(step=step+1,gate=int(gate.index) if hasattr(gate,'index') else rows[-1][2],kind='pass'))
                elif side<0<=new_side:events.append(dict(step=step+1,gate=env.tracker.index,kind='miss_plane'))
                elif side>=0>new_side:events.append(dict(step=step+1,gate=env.tracker.index,kind='return_plane'))
                if terminated or truncated or env.tracker.passed_count>=len(env.track.gates):break
                sensor.append(obs)
            max_dwell=max(max_dwell,len(rows)-last_pass)
            data=np.array(rows)
            np.savez_compressed(out/f'recovery-{arm}-021-e0.npz',rows=data,columns=np.array(['step','passed','gate','x','y','z','vx','vy','vz','side','thrust','roll','pitch','yaw']))
            item=dict(arm=arm,slot=record['slot'],seed=seed,steps=len(rows),passed=env.tracker.passed_count,
                      success=env.tracker.passed_count>=len(env.track.gates),terminated=bool(terminated),truncated=bool(truncated),
                      max_gate_dwell_seconds=max_dwell/130,events=events,
                      last_1000_position_std=data[-1000:,3:6].std(0).tolist(),last_1000_speed_mean=float(np.linalg.norm(data[-1000:,6:9],axis=1).mean()))
            summary.append(item);print(json.dumps(item),flush=True)
        finally:env.close()
    (out/'recovery-probe.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
