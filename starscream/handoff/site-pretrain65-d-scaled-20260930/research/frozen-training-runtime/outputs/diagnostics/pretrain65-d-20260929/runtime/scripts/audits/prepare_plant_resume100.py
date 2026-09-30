"""Build and validate the round-218 to round-318 teacher continuation; never launch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch

from scripts.train_privileged_racing import load_policy_checkpoint, dagger_teacher_beta
from starscream.dagger_schedule import apply_round_learning_rate
from starscream.dagger_replay import DaggerReplayStore
from starscream.racing_evaluation import load_protocol

RUN='starscream-v6.21.1.1-plant-resume100-r318'
CONFIG=Path('configs/exp/v6.21.1.1/plant_resume100_r318.json')
CERT=Path('outputs/diagnostics/plant-resume100/preflight.json')
PARENT=Path('outputs/checkpoints/starscream-v6.21.1.1-plant-selection25-dagger/latest.pt')
PINNED=Path(f'outputs/resume-inputs/{RUN}/round-218.pt')
SOURCES=['starscream/dagger_schedule.py','starscream/timed_teacher_reporting.py',
         'starscream/racing_evaluation.py','scripts/train_privileged_racing.py',
         'scripts/audits/prepare_plant_resume100.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.verify:
        cert=json.loads(CERT.read_text())
        for path,fingerprint in cert['fingerprints'].items():
            if sha(path)!=fingerprint:raise ValueError(f'Launch input changed: {path}')
        print('Resume launch certificate verified.');return
    torch.set_num_threads(1)
    if CONFIG.exists() or CERT.exists():raise FileExistsError('Prepared continuation already exists; use --verify')
    if not PINNED.exists():
        PINNED.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PARENT,PINNED)
    assert sha(PARENT)==sha(PINNED)
    policy,normalizer,payload,_=load_policy_checkpoint(str(PINNED),'cpu')
    assert payload['stage']=='dagger' and payload['round']==218
    assert payload['dagger_replay']['committed_round']==218
    assert payload['dagger_async_pending'] is None
    assert payload['dagger_dynamic_sampler_state'] and payload['numpy_rng_state'] and payload['rng_state']
    config=copy.deepcopy(payload['training_config']);settings=config['dagger']
    assert settings['rounds']==218 and settings['learning_rate']==.00012
    settings.update(run_name=RUN,rounds=318,resume_checkpoint=str(PINNED.resolve()),initial_checkpoint=None,
        dagger_resume_optimizer_state=True,dagger_resume_require_replay_state=True,
        dagger_resume_safe_state=True,dagger_resume_expert_refresh_rounds=0)
    settings['dagger_learning_rate_schedule']['horizon_rounds']=218
    protocol='configs/eval/real100_v2_timed_protocol_v1.json';load_protocol(protocol)
    settings['timed_reporting']=dict(protocol=protocol,protocol_sha256=sha(protocol),
        suite='configs/eval/v6211_vision_selection25.json',episodes_per_course=4,
        interval=10,start_round=218,end_round=318,parent_sha256=sha(PINNED))
    settings['tags']=[t for t in settings.get('tags',[]) if t!='scratch']+['resume100','fixed-schedule-horizon']
    config['checkpoint']['run_name']=RUN
    config['wandb'].update(run_name=RUN,group='v6.21.1.1-plant-continuation',tags=settings['tags'],
        local_event_path=f'/workspace/outputs/logs/{RUN}.events.jsonl',
        local_full_event_path=f'/workspace/outputs/logs/{RUN}.full.events.jsonl')
    for key in ('id','run_id','resume'):config['wandb'].pop(key,None)
    config['wandb']['eval_metric_allowlist']+=['timed/*','successful_median_steps','successful_p90_steps']
    config['experiment_notes']=dict(parent=str(PARENT),parent_sha256=sha(PINNED),
        purpose='100 additional rounds, identical learning recipe, terminal LR floor preserved; timed reports do not drive sampling or selection')
    optimizer=torch.optim.AdamW(policy.parameters(),lr=.00012,weight_decay=1e-5)
    optimizer.load_state_dict(payload['optimizer'])
    assert len(optimizer.state)>0
    steps=[float(s['step']) for s in optimizer.state.values()]
    assert min(steps)>0
    for state in optimizer.state.values():
        for key in ('exp_avg','exp_avg_sq'):assert torch.isfinite(state[key]).all()
    assert all(abs(g['lr']-3e-5)<1e-12 for g in optimizer.param_groups)
    curve=[]
    for round_index in range(219,319):
        apply_round_learning_rate(optimizer,round_index,318,settings['dagger_learning_rate_schedule'])
        beta=dagger_teacher_beta(round_index,settings)
        assert all(abs(g['lr']-3e-5)<1e-12 for g in optimizer.param_groups)
        assert abs(beta-.35)<1e-12
        curve.append(dict(round=round_index,lr=optimizer.param_groups[0]['lr'],teacher_beta=beta))
    metadata=payload['dagger_replay']
    store=DaggerReplayStore(Path(metadata['sources'][-1]['root']),metadata['contract'],metadata['sources'])
    shards=store._committed_shards(218)
    counts={'online':0,'permanent':0}
    for path in shards:
        with h5py.File(path,'r') as archive:
            assert archive.attrs['contract_fingerprint']==store.fingerprint
            for pool in counts:
                group=archive[pool];rows=int(group.attrs['rows']);counts[pool]+=rows
                assert all(a.shape[0]==rows for a in group.values())
                if rows:
                    for key in ('histories','actions','speed_commands'):
                        assert np.isfinite(group[key][-1]).all()
    retained=dict(online=min(counts['online'],settings['online_replay_capacity']),
                  permanent=min(counts['permanent'],settings['dagger_permanent_expert_capacity']))
    assert all(v>0 for v in retained.values())
    changed={k for k in settings if settings[k]!=payload['training_config']['dagger'].get(k)}
    allowed={'run_name','rounds','resume_checkpoint','initial_checkpoint','tags','dagger_learning_rate_schedule',
        'dagger_resume_optimizer_state','dagger_resume_require_replay_state','dagger_resume_safe_state',
        'dagger_resume_expert_refresh_rounds','timed_reporting'}
    assert changed<=allowed,changed-allowed
    CONFIG.write_text(json.dumps(config,indent=2)+'\n')
    cert=dict(status='validated-not-launched',run=RUN,parent_round=218,first_round=219,last_round=318,
        added_rounds=len(curve),parent_environment_steps=payload['environment_steps'],
        optimizer_step_range=[min(steps),max(steps)],restored_replay_rows=retained,validated_shards=len(shards),
        changed_dagger_keys=sorted(changed),schedule=curve,
        fingerprints={p:sha(p) for p in [str(CONFIG),str(PINNED),protocol,settings['timed_reporting']['suite'],*SOURCES]})
    CERT.parent.mkdir(parents=True,exist_ok=True);CERT.write_text(json.dumps(cert,indent=2)+'\n')
    print(json.dumps({k:v for k,v in cert.items() if k not in ('schedule','fingerprints')},indent=2))


if __name__=='__main__':main()
