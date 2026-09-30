"""Diagnostic: teacher label temporal consistency vs policy error on a fresh shard."""
import sys, numpy as np, torch, h5py
sys.path.insert(0,'/workspace')
from starscream.privileged_racing import load_policy_checkpoint
from starscream.racing_evaluation import canonical_inference
ckpt, shard = sys.argv[1:3]
policy, normalizer, payload, _ = load_policy_checkpoint(ckpt, 'cuda'); policy.eval()
h=h5py.File(shard)['online']
H=h['histories'][:]; A=h['actions'][:]; S=h['speed_commands'][:]; T=h['trajectory'][:]; Pv=h['previous'][:]; E=h['executed_actions'][:]
with canonical_inference():
    P=np.concatenate([policy(torch.from_numpy(H[i:i+4096].astype(np.float32)).cuda(),torch.from_numpy(S[i:i+4096]).cuda()).float().cpu().numpy() for i in range(0,len(H),4096)])
ep,st=T[:,12],T[:,7]
cons=np.r_[False,(ep[1:]==ep[:-1])&(st[1:]==st[:-1]+1)]
hub=lambda e: np.where(np.abs(e)<.05,.5*e**2/.05,np.abs(e)-.025).mean(-1)
print('rows',len(A),'consecutive pairs',cons.sum())
print('policy error |P-A|      dims',np.round(np.abs(P-A).mean(0),3),'huber',hub(P-A).mean())
print('prev executed |prev-A|  dims',np.round(np.abs(Pv-A).mean(0),3),'huber',hub(Pv-A).mean())
c=np.flatnonzero(cons)
print('teacher label jitter |A_t-A_t-1| dims',np.round(np.abs(A[c]-A[c-1]).mean(0),3),'huber',hub(A[c]-A[c-1]).mean())
print('policy pred jitter  |P_t-P_t-1| dims',np.round(np.abs(P[c]-P[c-1]).mean(0),3))
# second difference of labels = high-frequency component (noise proxy)
c2=c[np.isin(c-1,c)]
print('label 2nd-diff |A_t-2A_t-1+A_t-2| dims',np.round(np.abs(A[c2]-2*A[c2-1]+A[c2-2]).mean(0),3))
# error vs a smoothed label (3-tap moving avg) -> how much of the error is high-frequency label content
sm=A.copy(); sm[c2-1]=(A[c2-2]+A[c2-1]+A[c2])/3
print('policy error vs smoothed label (interior rows)',np.round(np.abs(P[c2-1]-sm[c2-1]).mean(0),3),' vs raw',np.round(np.abs(P[c2-1]-A[c2-1]).mean(0),3))
teach=T[:,5]>0
for m,n in [(teach,'teacher-executed'),(~teach,'learner-executed')]:
    cc=c[m[c]]
    print(n,'label jitter',np.round(np.abs(A[cc]-A[cc-1]).mean(0),3),'policy err',np.round(np.abs(P[m]-A[m]).mean(0),3))
dart=T[:,6]>0
for m,n in [(teach&~dart,'teacher-exec no-DART: |prev-A| = teacher step jitter'),(~teach,'learner-exec: |prev-A| = DAgger correction size')]:
    print(n,np.round(np.abs(Pv[m]-A[m]).mean(0),3),'huber',hub(Pv[m]-A[m]).mean(),'| policy err',np.round(np.abs(P[m]-A[m]).mean(0),3),'huber',hub(P[m]-A[m]).mean(), '| |P-prev|',np.round(np.abs(P[m]-Pv[m]).mean(0),3))
q=np.abs(Pv-A)
print('teacher-exec |prev-A| quantiles per dim (50/90/99):',[np.round(np.quantile(q[teach&~dart][:,k],[.5,.9,.99]),3).tolist() for k in range(4)])
