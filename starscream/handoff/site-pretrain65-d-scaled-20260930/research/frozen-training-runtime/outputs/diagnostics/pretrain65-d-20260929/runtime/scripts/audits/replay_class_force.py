"""Diagnostic: per-class action residual and Huber gradient 'force' of a checkpoint on replay shards."""
import sys, numpy as np, torch, h5py
sys.path.insert(0,'/workspace')
from starscream.privileged_racing import load_policy_checkpoint
from starscream.racing_evaluation import canonical_inference
ckpt, shards = sys.argv[1], sys.argv[2:]
policy, normalizer, payload, _ = load_policy_checkpoint(ckpt, 'cuda'); policy.eval()
beta=.05
rows=[]
for sh in shards:
    h=h5py.File(sh)['online']
    H=h['histories'][:]; A=h['actions'][:]; S=h['speed_commands'][:]; T=h['trajectory'][:]; E=h['executed_actions'][:]
    preds=[]
    with canonical_inference():
        for i in range(0,len(H),4096):
            x=torch.from_numpy(H[i:i+4096].astype(np.float32)).cuda()
            preds.append(policy(x, torch.from_numpy(S[i:i+4096]).cuda()).float().cpu().numpy())
    rows.append((np.concatenate(preds),A,T,E))
P,A,T,E=[np.concatenate(x) for x in zip(*rows)]
err=P-A
force=np.clip(np.abs(err)/beta,0,1).mean(-1)
lin=(np.abs(err)>beta).mean(-1)
q=T[:,0].astype(int); prec=(T[:,15].astype(int)&16)>0; teach=T[:,5]>0; dist=T[:,4]
def summ(mask,name):
    n=mask.sum()
    if n==0: return
    print(f"{name:32s} rows {n:7d} ({n/len(mask):6.1%}) |err| {np.abs(err[mask]).mean():.4f} dims {np.round(np.abs(err[mask]).mean(0),3)} lin {lin[mask].mean():.2f} force/row {force[mask].mean():.3f} share {force[mask].sum()/force.sum():6.1%}")
print('checkpoint round',payload['round'],'rows',len(P))
for c,nm in [(-1,'unknown'),(0,'nominal'),(1,'corrective'),(2,'recovery')]: summ(q==c,nm)
summ(prec,'precursor')
summ((q==0)&teach,'nominal teacher-executed'); summ((q==0)&~teach,'nominal learner-executed')
for lo,hi in [(0,.25),(.25,.5),(.5,1),(1,1.5),(1.5,3),(3,99)]:
    summ((dist>=lo)&(dist<hi)&(q>=0),f'line dist {lo}-{hi} m')
for c,nm in [(0,'nominal'),(1,'corrective'),(2,'recovery')]:
    m=q==c
    if m.any(): print(nm,'label |a|',np.round(np.abs(A[m]).mean(0),3),'sat>.95',np.round((np.abs(A[m])>.95).mean(0),3))
hub=np.where(np.abs(err)<beta,.5*err**2/beta,np.abs(err)-.5*beta).mean(-1)
print('huber action loss mean (compare trainer train/action_loss)',hub.mean(), 'median |err|',np.median(np.abs(err)))
