"""Read-only checkpoint and live frame-equivalence audit; no training changes."""
from pathlib import Path
import sys, json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.train_privileged_racing import load_config, parse_stage, make_env, ppo_observation_features, dynamics_observation_task
from starscream.privileged_racing import FeatureNormalizer
from starscream.world_privileged import world_feature_scales

def main():
    torch.set_num_threads(2)
    configs={k:load_config(Path(f'configs/exp/v6.15/{v}.yaml'))['dagger'] for k,v in [('b','b_magnitude'),('d','d_world')]}
    norms={}
    result={'checkpoints':{},'max_frame_errors':{},'samples':0}
    for k,s in configs.items():
        ck=torch.load(f"outputs/checkpoints/{s['run_name']}/latest.pt",map_location='cpu',weights_only=False)
        norms[k]=FeatureNormalizer.from_state_dict(ck['normalizer'])
        result['checkpoints'][k]={'contract':ck['model_config']['observation_contract'],
            'position_std':norms[k].std[:3].tolist(),'velocity_std':norms[k].std[3:6].tolist(),
            'dynamics_std':np.asarray(ck['dynamics_target_std']).tolist(),
            'dynamics_contract':s.get('dynamics_target_contract','task_delta_v1')}
        if k=='d':
            np.testing.assert_allclose(norms[k].std,world_feature_scales())
            np.testing.assert_allclose(norms[k].mean,0)
    stage=parse_stage(configs['b']['curriculum']); env=make_env(configs['d'],track=stage.tracks[0])
    samples={k:[] for k in configs}; errors={}
    def check(name,a,b):
        error=float(np.max(np.abs(a-b))); errors[name]=max(errors.get(name,0),error)
        np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)
    try:
        rng=np.random.default_rng(615)
        for index in range(len(env.track.gates)):
            obs,_=env.reset(seed=615+index,options={'gate_index':index})
            state=obs['privileged']['state'].copy()
            state[:3]+=rng.normal(0,.25,3)
            quat=rng.normal(size=4);state[3:7]=quat/np.linalg.norm(quat)
            state[7:10]=rng.normal(0,4,3);state[10:13]=rng.normal(0,1,3)
            obs,_=env.reset(seed=615+index,options={'gate_index':index,'state':state})
            for step in range(30):
                b=ppo_observation_features(obs,configs['b']); d=ppo_observation_features(obs,configs['d'])
                gate=env.track.gates[env.tracker.index]; q=gate.directed_rotation.T
                check('position',q@(d[:3]-gate.position),b[:3])
                check('velocity',q@d[3:6],b[3:6])
                check('attitude',(q@d[6:12].reshape(2,3).T).T.reshape(-1),b[6:12])
                check('body_motor',d[12:19],b[12:19])
                wr=d[19:97].reshape(6,13); br=b[19:97].reshape(6,13)
                check('route_position',(wr[:,:3]-gate.position)@q.T,br[:,:3])
                check('route_normal',wr[:,3:6]@q.T,br[:,3:6])
                check('route_up',wr[:,6:9]@q.T,br[:,6:9])
                check('route_flags',wr[:,9:],br[:,9:])
                check('action_age_valid',d[97:],b[97:])
                check('world_dynamics_input',dynamics_observation_task(obs,configs['d']),d[:19])
                for k,x in [('b',b),('d',d)]:samples[k].append(norms[k].numpy(x))
                result['samples']+=1
                obs,_,terminated,truncated,_=env.step(np.array([9.81,0.,0.,0.],np.float32))
                if terminated or truncated:break
    finally:env.close()
    result['max_frame_errors']=errors
    result['normalized_groups']={}
    for k,rows in samples.items():
        x=np.stack(rows); result['normalized_groups'][k]={}
        for name,lo,hi in [('position',0,3),('velocity',3,6),('attitude',6,12),('rates',12,15),('motor',15,19),('routes',19,97),('control',97,103)]:
            y=x[:,lo:hi];result['normalized_groups'][k][name]={'rms':float(np.sqrt(np.mean(y*y))),'max_abs':float(np.abs(y).max())}
    out=Path('outputs/diagnostics/v615-world-contract-audit.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
