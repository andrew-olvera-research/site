"""Diagnostic: nearest-neighbour label consistency (Bayes-noise proxy) vs policy error on fresh rows."""
import sys, numpy as np, torch, h5py
sys.path.insert(0,'/workspace')
from starscream.privileged_racing import load_policy_checkpoint
from starscream.racing_evaluation import canonical_inference
ckpt, query, *bank = sys.argv[1:]
policy, normalizer, payload, _ = load_policy_checkpoint(ckpt, 'cuda'); policy.eval()
def load(f, n=None, seed=0):
    h=h5py.File(f)['online']; N=len(h['actions'])
    idx=np.arange(N) if n is None else np.sort(np.random.default_rng(seed).choice(N,n,replace=False))
    return (torch.from_numpy(h['histories'][idx].astype(np.float32)).cuda(), h['actions'][idx], h['speed_commands'][idx], h['tracks'][idx], h['trajectory'][idx,12])
Hq,Aq,Sq,Tq,Eq=load(query,4096)
with canonical_inference(): Pq=policy(Hq,torch.from_numpy(Sq).cuda()).float().cpu().numpy()
B=[load(f) for f in bank]
Hb=torch.cat([b[0] for b in B]); Ab=np.concatenate([b[1] for b in B]); Tb=np.concatenate([b[3] for b in B]); Sb=np.concatenate([b[2] for b in B])
xq=torch.cat([Hq.flatten(1), torch.from_numpy(Sq).cuda()[:,None]/5],1); xb=torch.cat([Hb.flatten(1), torch.from_numpy(Sb).cuda()[:,None]/5],1)
best=torch.full((len(xq),),1e9,device='cuda'); arg=torch.zeros(len(xq),dtype=torch.long,device='cuda')
for i in range(0,len(xb),65536):
    d=torch.cdist(xq,xb[i:i+65536]); m,a=d.min(1); upd=m<best; best[upd]=m[upd]; arg[upd]=a[upd]+i
best=best.cpu().numpy(); arg=arg.cpu().numpy()
dA=np.abs(Aq-Ab[arg]); dP=np.abs(Pq-Aq)
print('bank rows',len(Ab),'query rows',len(Aq),'same-track NN frac',np.mean(Tq==Tb[arg]))
qs=np.quantile(best,[0,.05,.2,.5,.8,1])
for lo,hi in zip(qs[:-1],qs[1:]):
    m=(best>=lo)&(best<=hi)
    print(f"NN obs dist [{lo:6.2f},{hi:6.2f}] n={m.sum():4d}  |label - NN label| dims {np.round(dA[m].mean(0),3)} mean {dA[m].mean():.3f}   policy |P-A| mean {dP[m].mean():.3f}")
