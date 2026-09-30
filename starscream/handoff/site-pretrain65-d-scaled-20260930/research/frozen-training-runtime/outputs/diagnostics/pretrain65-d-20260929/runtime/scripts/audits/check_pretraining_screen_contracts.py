"""Verify common plant and reference-planner settings for the checkpoint screen."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train_privileged_racing import load_config, stage_config, read_manifest, _merge
from scripts.audits.audit_pretraining_mode_comparison import CASES, OUT
from starscream.env.tracks import load_track

protocol=json.loads((ROOT/'configs/eval/real100_v2_timed_protocol_v1.json').read_text())
names=[load_track(str(ROOT/r['path'])).name for r in protocol['records']]
result={}
for name,(config,_) in CASES.items():
    s=stage_config(load_config(ROOT/config),'dagger')['dagger']
    manifest=read_manifest(str(s.get('track_manifest') or s.get('mpcc_track_manifest')))
    plans={}
    for track in names:
        match=next((r for r in manifest['records'] if r['name']==track),None)
        p=dict(s.get('mpcc_planner_config',{}))
        if match:
            p=_merge(p,s.get('mpcc_family_planner_configs',{}).get(match.get('family',''),{}))
        if s.get('mpcc_use_manifest_teacher_profile',False) and match:
            profile=((match.get('qualification') or {}).get('selected') or {}).get('teacher_profile','')
            p=_merge(p,s['mpcc_manifest_teacher_profile_planner_configs'][profile])
        p.setdefault('offset_iterations',int(s.get('racing_line_iterations',30)))
        p.pop('cache_directory',None)
        plans[track]=p
    result[name]=dict(planner=plans,plant={k:v for k,v in s.items() if k in ['dynamics_randomization',
        'observation_randomization','flight_plan_randomization','state_estimation','actuator_delay_steps',
        'action_contract','control_hz','body_rate_limit','collective_min','collective_max']})
base=result['v621']
comparison={name:dict(planner_differences=[t for t in names if r['planner'][t]!=base['planner'][t]],
    plant_equal=r['plant']==base['plant']) for name,r in result.items()}
(OUT/'contract-check.json').write_text(json.dumps(dict(comparison=comparison,resolved=result),indent=2)+'\n')
print(json.dumps(comparison,indent=2))
