"""Build additive v6.21.1 pretraining data from v6.21 + v6.22.x findings.

The 45 v6.21 training courses are immutable anchors. Twenty already-qualified,
evaluation-clone-protected v6.22.x courses add UTT-relevant long/low/narrow,
radius-switch, long-braking, and ordered technical chains. No policy score is
used for admission or selection.
"""
from collections import Counter
import copy, hashlib, json, math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train_privileged_racing import load_config, stage_config
from starscream.course_model.training import atomic_json
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'outputs/course-pools/v621-additions/manifest.json'
V622 = ROOT/'outputs/v622-pretraining60/mpcc-frontier-v3/manifest.json'
PROFILES = ROOT/'outputs/v622-pretraining60/mpcc-frontier-v3/teacher-profiles.json'
OUT = ROOT/'outputs/course-pools/v6211-pretrain'
CONFIG = ROOT/'configs/exp/v6.21.1/pretrain65_dagger.yaml'

QUOTAS = {
    'behavior:long_low_braking': 10,
    'behavior:radius_switch': 4,
    'long_braking': 2,
    'a2rl_technical:compound_reversal': 2,
    'a2rl_technical:wrong_side_incidence': 2,
}

def resolve(path):
    p=Path(path)
    return ROOT/p.relative_to('/workspace') if str(p).startswith('/workspace/') else p

def utt_utility(row):
    """Geometry-only preference within a fixed family quota."""
    g=row['geometry']; ts=g['transitions']
    long=sum(t['incoming_m'] >= 20 for t in ts)
    very_long=sum(t['incoming_m'] >= 30 for t in ts)
    hard_after_long=sum(t['incoming_m'] >= 20 and t['turn_deg'] >= 105 for t in ts)
    low=sum(t.get('gate_center_height_m',99) <= 1.2 for t in ts)
    narrow=sum(min(t.get('width_m',99),t.get('height_m',99)) <= 1.65 for t in ts)
    return (2*hard_after_long + long + 2*very_long + low + narrow
            + .5*g['gate_count'] + .01*sum(t['incoming_m'] for t in ts))

def build_manifest():
    old=json.loads(OLD.read_text()); v622=json.loads(V622.read_text())
    anchors=[copy.deepcopy(r) for r in old['records'] if r['split']=='train']
    validation=[copy.deepcopy(r) for r in old['records'] if r['split']!='train']
    assert len(anchors)==45 and len(validation)==38
    additions=[]; ledger=[]
    for family,quota in QUOTAS.items():
        candidates=[copy.deepcopy(r) for r in v622['records'] if r['family']==family]
        candidates.sort(key=lambda r:(-utt_utility(r),r['name']))
        if len(candidates)<quota: raise RuntimeError(f'{family}: need {quota}, have {len(candidates)}')
        for r in candidates[:quota]:
            r['split']='train'; r['suite']='v6.21.1-pretrain'
            r['lineage_role']='v6.22.x_utt_support_addition'
            # The v6.21 collector and replay are balanced by ``family``.  The
            # v6.22 source manifest groups many courses into a family, which
            # would give each member only a fraction of a v6.21 course's
            # collection, replay, and permanent-expert quota.  Keep the
            # behavior class separately and make every admitted course a
            # singleton replay family, exactly like the 45 anchors.
            r['behavior_family']=family
            r['family']=r['name']
            r['source_family']=r['name']
            r['source_stratum']='canonical'
            r['flight_mimic_start_gates']=list(range(int(r['geometry']['gate_count'])))
            r['qualified']=True
            r['dart_eligible']=True
            additions.append(r)
            ledger.append(dict(name=r['name'],family=family,geometry_utility=utt_utility(r),
                gate_count=r['geometry']['gate_count'],qualified_speed_mps=r['qualified_speed_mps'],
                source_manifest=str(V622.relative_to(ROOT))))
    for r in anchors:r['lineage_role']='v6.21_immutable_anchor'
    records=anchors+additions+validation
    manifest={
      'schema':'starscream-procedural-track-manifest-v1','seed':62110065,
      'records':records,'family_weights':{f:1. for f in sorted({r['family'] for r in records if r['split']=='train'})},
      'source':'v6.21.1-pretrain65: all v6.21 train45 plus 20 qualified v6.22.x UTT-support courses',
      'lineage':{
        'base_version':'v6.21','base_manifest':str(OLD.relative_to(ROOT)),
        'base_manifest_sha256':hashlib.sha256(OLD.read_bytes()).hexdigest(),
        'findings_version':'v6.22.x','qualified_source_manifest':str(V622.relative_to(ROOT)),
        'qualified_source_manifest_sha256':hashlib.sha256(V622.read_bytes()).hexdigest(),
        'selection_evidence':['outputs/diagnostics/v621-rl7-tuning/technical-support.json',
                              'outputs/evals/v6222-crosscompare'],
      },
      'admission':{'status':'ready','v621_anchors':45,'v622x_additions':20,
                   'all_additions_prequalified':True,'policy_scores_used':False},
      'protected_suites':['configs/eval/v6_22_real60.manifest.json','configs/eval/v6_22_real100_hard_v2.manifest.json'],
      'normalization':'outputs/course-pools/v6211-pretrain/normalization.pt',
    }
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/'manifest.json',manifest)
    atomic_json(OUT/'selection.json',{'schema':'starscream-v6211-utt-selection-v1','quotas':QUOTAS,
      'rule':'fixed family quotas, then geometry-only UTT utility; no policy outcomes', 'selected':ledger})
    return manifest

def build_config(manifest):
    c=stage_config(load_config(ROOT/'configs/exp/v6.21/scratch_5m_dagger_r218.yaml'),'dagger');s=c['dagger']
    train=[r for r in manifest['records'] if r['split']=='train']
    additions=[r for r in train if r['lineage_role']=='v6.22.x_utt_support_addition']
    profiles=json.loads(PROFILES.read_text())
    v622_settings=json.loads((ROOT/'configs/exp/v6.22/pretraining_behavior60_dagger.yaml').read_text())['dagger']
    s.update(run_name='starscream-v6.21.1-pretrain65-dagger',seed=2026092111,
      track_manifest='/workspace/outputs/course-pools/v6211-pretrain/manifest.json',track_split='train',
      dagger_initialization_stats_checkpoint='/workspace/outputs/course-pools/v6211-pretrain/normalization.pt',
      racing_line_cache='/workspace/outputs/course-pools/v6211-pretrain/racing-lines',
      mpcc_build_root='/tmp/starscream-v6211-pretrain65',episodes_per_round=520,
      mpcc_use_manifest_planner_profile=True,
      # Preserve v6.21's 218-round learning schedule and scale the per-round
      # optimizer/replay budgets with course count.  This keeps both rollout
      # episodes and batch presentations per course at the v6.21 target.
      rounds=218, updates_per_round=math.ceil(2026*65/45),
      online_replay_capacity=math.ceil(1575001*65/45),
      dagger_online_replay_rows_per_round=math.ceil(27001*65/45),
      dagger_permanent_expert_capacity=math.ceil(675000*65/45),
      dagger_condition_on_teacher_speed=True, evaluation_condition_on_manifest_speed=True,
      tags=['v6.21.1','pretrain65','v6.21-preserved','v6.22.x-utt-support','130hz','route6'])
    s['curriculum']['name']='v6211_pretrain65';s['curriculum']['tracks']=[r['path'] for r in train]
    s['curriculum']['max_steps']=10000
    # Preserve v6.21's four evaluation episodes per competence source and put
    # every new source into the adaptive-sampling feedback loop.  The legacy
    # 35 validation counterparts remain unchanged; the source pool does not
    # contain separate counterparts for these additive courses.
    s['evaluation_curriculum']['tracks'] += [r['path'] for r in additions]
    s['evaluation_curriculum']['max_steps']=10000
    s['evaluation_episodes']=4*len(s['evaluation_curriculum']['tracks'])
    family_weights={f:1. for f in sorted({r['family'] for r in train})}
    s['track_sampling_family_weights']=family_weights
    s['dagger_replay_family_weights']=family_weights
    s['dagger_permanent_expert_required_families']=sorted(family_weights)
    for r in additions:
        s['reliability_family_aliases'][r['family']]=v622_settings['reliability_family_aliases'][r['behavior_family']]
    # Do not inherit v6.22's per-track DART exemptions: v6.21 applies the same
    # four-round DART bootstrap to every course.
    s.pop('dagger_dart_exempt_tracks',None)
    s['mpcc_manifest_teacher_profile_controller_configs'].update(profiles['controller_profiles'])
    s['mpcc_manifest_teacher_profile_planner_configs'].update(profiles['planner_profiles'])
    c['checkpoint']['run_name']=s['run_name'];c['wandb'].update(
      run_name=s['run_name'],group='v6.21.1-pretrain',
      local_event_path='/workspace/outputs/logs/starscream-v6.21.1-pretrain65-dagger.events.jsonl')
    c['experiment_notes']={
      'data_version':'v6.21.1-pretrain','base':'v6.21 train45 retained byte-for-byte',
      'additions':'20 MPCC-qualified v6.22.x courses selected for UTT support',
      'intent':'add long-low-narrow and multi-transition chain support; do not remove v6.21 behavior modes',
      'evaluation_hygiene':'no held-out geometry copied; v6.22 clone-protection inherited; selection uses geometry requirements, not policy outcomes',
      'a2rl':'compound/wrong-side support included, but A2RL completion is not an admission claim'}
    CONFIG.parent.mkdir(parents=True,exist_ok=True);CONFIG.write_text(json.dumps(c,indent=2)+'\n')

if __name__=='__main__':
    m=build_manifest();build_config(m)
    print(json.dumps({'manifest':str(OUT/'manifest.json'),'config':str(CONFIG),
      'train':sum(r['split']=='train' for r in m['records']),'validation':sum(r['split']=='validation' for r in m['records']),
      'report':sum(r['split']=='report' for r in m['records'])},indent=2))
