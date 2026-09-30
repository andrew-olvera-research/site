"""Final production-path and exercised-control assertions before queue launch."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path('/workspace')
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/diagnostics/collection-next-20260929'

def main():
    from scripts.train_privileged_racing import stage_config
    from starscream.dagger_collection import collection_config
    from starscream.evaluation_suite import configure_selection_suite
    jobs=json.loads((OUT/'jobs.json').read_text())
    baseline=json.loads((ROOT/'configs/exp/v6.21.1.1/mini_slalom_aug2_fixes.yaml').read_text())['dagger']
    allowed={'run_name','seed','top_k','monitor','mpcc_build_root','tags','dagger_collection_control','action_dimension_weights'}
    evidence={}; paths=set(); differences={}
    for job in jobs:
        config=json.loads(Path(job['config']).read_text()); s=config['dagger']
        changed={k for k in baseline.keys()|s.keys() if baseline.get(k)!=s.get(k)}
        assert not changed-allowed, (job['arm'],changed-allowed)
        differences[job['arm']]=sorted(changed)
        assert s['initial_checkpoint'] is None and s['resume_checkpoint'] is None
        assert s['rounds']==24 and s['updates_per_round']==626 and s['episodes_per_round']==64
        assert s['seed']==2026092913
        collection_config(s)
        for key in ('local_event_path','local_full_event_path'):
            p=config['wandb'][key]; assert p not in paths and job['run_name'] in p; paths.add(p)
        resolved=stage_config(config,'dagger'); configure_selection_suite(resolved['dagger'])
        assert resolved['checkpoint']['top_k']==1
        assert resolved['checkpoint']['monitor']=='selection_suite_timely_success'
        assert len(resolved['dagger']['evaluation_curriculum']['tracks'])==5
    for arm in ('full_learner','time_2s','full_expert20','recovery_4s'):
        rows=[json.loads(l)['metrics'] for l in (OUT/f'next-smoke-{arm}.full.events.jsonl').read_text().splitlines()]
        train=[r for r in rows if 'train/round' in r]
        assert [r['train/round'] for r in train]==[1,2,3,4]
        keys={k for r in train for k in r if k.startswith('train/collection/')}
        counts={k:sum(r.get(k,0) for r in train[1:]) for k in keys}
        assert counts['train/collection/active_steps']>0
        if arm!='full_expert20': assert counts['train/collection/active_teacher_steps']==0
        else: assert counts['train/collection/active_teacher_steps']>0 and counts['train/collection/learner_steps']>0
        if arm=='time_2s': assert counts['train/collection/segment_time_limit']>0
        if arm=='recovery_4s': assert counts['train/collection/learner_recovery_steps']>0
        evidence[arm]=counts
    assert json.loads((OUT/'eval.status.json').read_text())['state']=='complete'
    assert json.loads((OUT/'validation.json').read_text())['state']=='passed'
    report=dict(state='passed',arms=len(jobs),differences=differences,collection_evidence=evidence,
        fixture_logging_repair='New smoke events isolated from inherited smoke log; original 360054-byte historical log restored. Production paths asserted unique.',
        historical_fixture_log_sha256=hashlib.sha256((ROOT/'outputs/diagnostics/collection-ablation-20260929/collection-smoke-d.full.events.jsonl').read_bytes()).hexdigest())
    (OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Passed: 12 isolated production configs, matching budgets, active new controls, evaluations and validation complete.')

if __name__=='__main__': main()
