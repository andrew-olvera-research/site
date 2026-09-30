"""Diagnostic: share of replay rows / Huber force / fresh error by signed distance to the active gate plane."""
import sys, numpy as np, torch, h5py
sys.path.insert(0,'/workspace')
from starscream.privileged_racing import load_policy_checkpoint
from starscream.racing_evaluation import canonical_inference
ckpt, shard = sys.argv[1:3]
policy, normalizer, payload, _ = load_policy_checkpoint(ckpt, 'cuda'); policy.eval()
h=h5py.File(shard)['online']
H=h['histories'][:]; A=h['actions'][:]; S=h['speed_commands'][:]; T=h['trajectory'][:]
with canonical_inference():
    P=np.concatenate([policy(torch.from_numpy(H[i:i+4096].astype(np.float32)).cuda(),torch.from_numpy(S[i:i+4096]).cuda()).float().cpu().numpy() for i in range(0,len(H),4096)])
e=P-A; hub=np.where(np.abs(e)<.05,.5*e**2/.05,np.abs(e)-.025).mean(-1); force=np.clip(np.abs(e)/.05,0,1).mean(-1)
side=T[:,13]; q=T[:,0]
print('rows',len(A))
for lo,hi in [(-99,-8),(-8,-4),(-4,-2),(-2,-1),(-1,0),(0,2),(2,99)]:
    m=(side>=lo)&(side<hi)&(q>=0)
    print(f"gate-plane side [{lo:>4},{hi:>3}) m: rows {m.mean():6.1%}  force share {force[m].sum()/force[q>=0].sum():6.1%}  fresh huber {hub[m].mean():.3f}  |err| dims {np.round(np.abs(e[m]).mean(0),3)}")
