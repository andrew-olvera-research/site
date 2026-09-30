"""Validate and launch the authorized r287 + 50-round bounded continuation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train_privileged_racing import load_config, stage_config, dagger_teacher_beta
from starscream.dagger_quality import quality_config, online_minibatch_quality_config
from starscream.dagger_replay import replay_contract_fingerprint
from starscream.timed_teacher_reporting import report_due

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    os.chdir(ROOT)
    path=Path('configs/exp/v6.21.1.1/plant_bounded_r287_recovery10_r337.yaml')
    config=stage_config(load_config(path),'dagger');settings=config['dagger']
    source=Path(settings['resume_checkpoint']);p=torch.load(source,map_location='cpu',weights_only=False)
    assert p['round']==287 and p['environment_steps']==141977167
    assert settings['rounds']-p['round']==50
    assert hashlib.sha256(source.read_bytes()).hexdigest()==settings['timed_reporting']['parent_sha256']
    old=stage_config(p['training_config'],'dagger')['dagger']
    changed={k for k in set(old)|set(settings) if old.get(k)!=settings.get(k)}
    allowed={'rounds','run_name','tags','resume_checkpoint','dagger_online_minibatch_quality','timed_reporting','dagger_dynamic_sampling'}
    assert not changed-allowed,changed-allowed
    # The trainer materializes these map-derived fields before checkpointing.
    dynamic=dict(old['dagger_dynamic_sampling'])
    for k in ('behavior_gate_weights','behavior_course_requirements','behavior_family_weights',
              'behavior_competence','behavior_health','behavior_training_mix'):
        dynamic.pop(k,None)
    assert dynamic==settings['dagger_dynamic_sampling'], sorted(set(dynamic)^set(settings['dagger_dynamic_sampling']))
    map_path=Path(dynamic['behavior_map'])
    assert hashlib.sha256(map_path.read_bytes()).hexdigest()==dynamic['behavior_map_sha256']
    q=quality_config(settings)
    assert q==p['dagger_replay']['contract']['trajectory_quality_v1']
    assert replay_contract_fingerprint(p['dagger_replay']['contract'])==p['dagger_replay']['contract_fingerprint']
    assert online_minibatch_quality_config(q,settings['dagger_online_minibatch_quality'])['recovery_fraction']==.1
    assert q['recovery_fraction']==.05 and q['implementation_version']==2
    assert settings['dagger_learning_rate_schedule']['horizon_rounds']==258
    assert settings['learning_rate']*settings['dagger_learning_rate_schedule']['final_fraction']==3e-5
    assert all(g['lr']==3e-5 for g in p['optimizer']['param_groups'])
    assert dagger_teacher_beta(288,settings)==.35
    assert settings['updates_per_round']==5081 and settings['batch_size']==1536
    assert settings['episodes_per_round']==520 and settings['dagger_resume_require_replay_state']
    assert not settings.get('dagger_dart_refresh_interval',0)
    missing=[]
    for s in p['dagger_replay']['sources']:
        for r in range(int(s['min_round']),min(int(s['max_round']),287)+1):
            f=Path(s['root'])/f'round-{r:05d}.h5'
            if not f.is_file(): missing.append(str(f))
    assert not missing, missing
    spec=settings['timed_reporting']
    assert hashlib.sha256(Path(spec['protocol']).read_bytes()).hexdigest()==spec['protocol_sha256']
    rounds=[r for r in range(287,338) if report_due(r,spec)]
    assert rounds==[287,297,307,317,327,337]
    run=settings['run_name']; destination=ROOT/'outputs/checkpoints'/run
    assert not destination.exists(), 'Destination exists; explicit resume needed, never overwrite.'
    active=subprocess.run(['pgrep','-af','[t]rain_privileged_racing.py .*--stage (dagger|ppo|fpo)'],capture_output=True,text=True)
    assert active.returncode==1, f'Active trainer or process inspection error: {active.stdout} {active.stderr}'
    output=ROOT/'outputs/diagnostics/bounded-r287-recovery10'
    output.mkdir(parents=True,exist_ok=True)
    files=[path,Path('scripts/train_privileged_racing.py'),Path('starscream/dagger_quality.py'),
           Path('starscream/bounded_continuation_reporting.py'),Path('starscream/timed_teacher_reporting.py'),
           Path('scripts/audits/eval_privileged_real100_v2_dual.py')]
    result=dict(passed=True,parent_round=287,end_round=337,additional_rounds=50,
        changed_settings=sorted(changed),derived_dynamic_map_fields_only=True,
        optimizer_learning_rate=3e-5,updates_per_round=5081,batch_size=1536,
        online_draw_counts=[649,250,99],permanent_draw_counts=[403,135,0],
        actual_combined_recovery=99/1536,retention_recovery_cap=.05,
        report_rounds=rounds,source_checkpoint_sha256=spec['parent_sha256'],
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (output/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'resolved-config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if not args.launch:return
    command=[sys.executable,'-u','scripts/train_privileged_racing.py','--config',str(path),'--stage','dagger','--device','cuda']
    log=ROOT/'outputs/logs'/f'{run}.console.log'
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',WANDB_CONSOLE='redirect')
    with log.open('xb') as handle:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(pid=child.pid,command=command,log=str(log),user_authorized=True,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_name=run)
    (output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
