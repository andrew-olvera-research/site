"""Diagnostic: trainer imitation_loss vs direct forward on one replay shard (per quality class)."""
import sys, numpy as np, torch, h5py
sys.path.insert(0,'/workspace')
from starscream.privileged_racing import load_policy_checkpoint
from scripts.train_privileged_racing import imitation_loss, load_config, stage_config
ckpt, shard, cfg = sys.argv[1:4]
from pathlib import Path
settings=dict(stage_config(load_config(Path(cfg)),'dagger')['dagger'])
policy, normalizer, payload, _ = load_policy_checkpoint(ckpt, 'cuda'); policy.eval()
import os; h=h5py.File(shard)[os.environ.get('GROUP','online')]
idx=np.sort(np.random.default_rng(0).choice(len(h['actions']),8192,replace=False))
g=lambda k: torch.from_numpy(np.asarray(h[k][idx]).astype(np.float32)).cuda()
T=None
with torch.no_grad():
    tot,p=imitation_loss(policy,g('histories'),g('actions'),g('previous'),g('dynamics'),settings,
        dynamics_valid=torch.from_numpy(h['dynamics_valid'][idx]).cuda(),speed_commands=g('speed_commands'),
        executed_actions=g('executed_actions'))
    print('trainer action_loss',float(p['action_loss']),'physical',float(p['physical_action_loss']))
    direct=policy(g('histories'),g('speed_commands')).float()
    e=(direct-g('actions')).abs()
    print('direct forward huber',float(torch.where(e<.05,.5*e**2/.05,e-.025).mean()))
    pred=policy.training_predictions_with_chunk_and_encoding(g('histories'),g('speed_commands'))[0]
    e2=(pred-g('actions')).abs()
    print('training_predictions huber',float(torch.where(e2<.05,.5*e2**2/.05,e2-.025).mean()),'max diff direct vs train',float((pred-direct).abs().max()))
