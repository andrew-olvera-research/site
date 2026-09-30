"""Diagnostic: where fresh-state action error comes from (saturated labels, sign flips, per dim)."""
import sys, numpy as np, torch, h5py
sys.path.insert(0,'/workspace')
from starscream.privileged_racing import load_policy_checkpoint
from starscream.racing_evaluation import canonical_inference
ckpt, shard = sys.argv[1:3]
policy, normalizer, payload, _ = load_policy_checkpoint(ckpt, 'cuda'); policy.eval()
h=h5py.File(shard)['online']; H=h['histories'][:]; A=h['actions'][:]; S=h['speed_commands'][:]; Pv=h['previous'][:]
with canonical_inference():
    P=np.concatenate([policy(torch.from_numpy(H[i:i+4096].astype(np.float32)).cuda(),torch.from_numpy(S[i:i+4096]).cuda()).float().cpu().numpy() for i in range(0,len(H),4096)])
e=P-A; hub=np.where(np.abs(e)<.05,.5*e**2/.05,np.abs(e)-.025)
tot=hub.sum()
names=['collective','roll','pitch','yaw']
for k in range(4):
    sat=np.abs(A[:,k])>.9; flip=(np.sign(P[:,k])!=np.sign(A[:,k]))&(np.abs(e[:,k])>.5); big=np.abs(e[:,k])>.5
    print(f"{names[k]:10s} loss share {hub[:,k].sum()/tot:5.1%} | saturated-label rows {sat.mean():5.1%} carry {hub[sat,k].sum()/hub[:,k].sum():5.1%} | |err|>0.5 rows {big.mean():5.1%} carry {hub[big,k].sum()/hub[:,k].sum():5.1%} | large sign flips {flip.mean():5.1%} carry {hub[flip,k].sum()/hub[:,k].sum():5.1%} | median |err| {np.median(np.abs(e[:,k])):.3f}")
# is the label predictable from where the label will be in the future? proxy: label vs previous executed within-dim change
d=np.abs(A-Pv)
print('label jump |A - previous executed| > 0.5 frac per dim', np.round((d>.5).mean(0),3))
tm=h['teacher_modes'][:]; T=h['trajectory'][:]
for v in np.unique(tm):
    m=tm==v; print('teacher_mode',v,'rows',f"{m.mean():.1%}",'huber',hub[m].mean().round(3),'per-dim |err|',np.round(np.abs(e[m]).mean(0),3))
m=T[:,16]>0; print('teacher_missed_gate_recovery rows',f"{m.mean():.1%}",'huber',hub[m].mean().round(3) if m.any() else None)
