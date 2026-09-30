#!/usr/bin/env python3
"""Paired latent and action probe on the same future DAgger shard."""
import json
from pathlib import Path
import numpy as np
import h5py
import torch
from starscream.privileged_racing import load_policy_checkpoint

ROOT=Path("/workspace")
SHARD=ROOT/"outputs/checkpoints/starscream-v6.21.1.1-plant-selection25-dagger-recovery-fix/dagger-replay/round-00258.h5"
OLD=ROOT/"outputs/checkpoints/starscream-v6.21.1.1-plant-selection25-dagger/best-step-150819355-selection_suite_success-0.57875.pt"
NEW=ROOT/"outputs/checkpoints/starscream-v6.21.1.1-plant-selection25-dagger-recovery-fix/best-step-125516154-selection_suite_success-0.29125.pt"
OUT=ROOT/"outputs/diagnostics/plant-representation-paired-best-r210-vs-r254.json"

def subset_indices(quality, rng, per_class=2400):
    pieces=[]
    for cls in (0,1,2):
        indices=np.flatnonzero(quality==cls)
        pieces.append(np.sort(rng.choice(indices,size=min(per_class,len(indices)),replace=False)))
    return np.sort(np.concatenate(pieces))

def ridge_r2(latent,target,train,test,alpha=10.):
    mean=latent[train].mean(0); std=latent[train].std(0)
    x=np.clip((latent-mean)/np.maximum(std,1e-6),-10,10).astype(np.float64)
    y=target.astype(np.float64)
    ym=y[train].mean(0); ys=np.maximum(y[train].std(0),1e-6)
    z=(y-ym)/ys
    xt=np.concatenate([x[train],np.ones((train.sum(),1))],axis=1)
    xv=np.concatenate([x[test],np.ones((test.sum(),1))],axis=1)
    gram=xt.T@xt; gram.flat[::len(gram)+1]+=alpha
    weight=np.linalg.solve(gram,xt.T@z[train])
    pred=xv@weight
    denom=np.sum((z[test]-z[test].mean(0))**2,axis=0)
    score=1-np.sum((z[test]-pred)**2,axis=0)/np.maximum(denom,1e-9)
    return float(np.mean(score)),[float(v) for v in score]

def rank_stats(latent):
    centered=latent-latent.mean(0)
    eig=np.linalg.eigvalsh(centered.T@centered/len(centered))
    eig=np.maximum(eig,0)
    total=eig.sum()
    return dict(participation_rank=float(total**2/np.sum(eig**2)),
                top10_variance=float(eig[-10:].sum()/total),
                effective_rank=float(np.exp(-np.sum((eig/total)*np.log(np.maximum(eig/total,1e-12))))))

def encode_and_act(path,raw,speed):
    policy,normalizer,payload,_=load_policy_checkpoint(path,"cuda")
    policy.eval()
    embeddings=[]; actions=[]
    with torch.inference_mode():
        for start in range(0,len(raw),256):
            x=torch.from_numpy(raw[start:start+256]).cuda()
            v=torch.from_numpy(speed[start:start+256]).cuda()
            embeddings.append(policy.encode(x).float().cpu().numpy())
            actions.append(policy(x,v).float().cpu().numpy())
    return np.concatenate(embeddings),np.concatenate(actions),int(payload["round"])

def main():
    torch.set_num_threads(1)
    rng=np.random.default_rng(20260927)
    with h5py.File(SHARD) as file:
        group=file["online"]
        quality=group["trajectory"][:,0].astype(np.int8)
        idx=subset_indices(quality,rng)
        raw=group["histories"][idx].astype(np.float32)
        expert=group["actions"][idx].astype(np.float32)
        speed=group["speed_commands"][idx].astype(np.float32)
        episode=group["trajectory"][idx,12].astype(np.int64)
        cls=quality[idx]
    unique=np.unique(episode)
    test_ids=set(rng.choice(unique,size=max(1,int(.2*len(unique))),replace=False).tolist())
    test=np.asarray([x in test_ids for x in episode],bool);train=~test
    _,norm,_,_=load_policy_checkpoint(OLD,"cpu")
    physical=raw*norm.std+norm.mean
    target=np.concatenate([physical[:,-1,19:22]-physical[:,-1,0:3],
                           physical[:,-1,32:35]-physical[:,-1,0:3]],axis=1)
    result=dict(schema="starscream-representation-paired-probe-v1",
                shard=str(SHARD),rows=len(raw),train_rows=int(train.sum()),test_rows=int(test.sum()),
                split="held-out episode IDs",class_rows={str(i):int(np.sum(cls==i)) for i in (0,1,2)},
                target="current and next gate relative world positions (six coordinates)",
                checkpoints={})
    latent={}
    for name,path in [("old",OLD),("recovery_fix",NEW)]:
        z,action,round_idx=encode_and_act(path,raw,speed)
        latent[name]=z.astype(np.float64)
        r2,coordinates=ridge_r2(z,target,train,test)
        row=dict(path=str(path),round=round_idx,latent_dim=z.shape[1],
                 gate_probe_r2=r2,gate_probe_coordinates=coordinates,
                 latent_rank=rank_stats(z))
        for class_id,label in ((0,"nominal"),(1,"corrective"),(2,"qualified_recovery")):
            mask=test&(cls==class_id)
            row[label+"_test_rows"]=int(mask.sum())
            row[label+"_action_mae"]=float(np.mean(np.abs(action[mask]-expert[mask]))) if mask.any() else None
        result["checkpoints"][name]=row
    a=latent["old"]-latent["old"].mean(0)
    b=latent["recovery_fix"]-latent["recovery_fix"].mean(0)
    numerator=np.sum((a.T@b)**2)
    denominator=np.sqrt(np.sum((a.T@a)**2)*np.sum((b.T@b)**2))
    result["linear_cka"]=float(numerator/denominator)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()

