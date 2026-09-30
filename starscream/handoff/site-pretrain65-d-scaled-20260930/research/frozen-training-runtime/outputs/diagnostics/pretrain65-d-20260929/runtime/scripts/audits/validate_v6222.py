"""Fail-closed v6.22.2 recipe, metadata, and actual scheduling preflight."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train_privileged_racing import (dagger_episode_tracks, dagger_episode_start_gates,
    dagger_episode_start_modes, dagger_family_survival_statistics, load_config)
from starscream.env.tracks import load_track
from starscream.course_model.training import atomic_json

BASE=ROOT/'configs/exp/v6.21/scratch_5m_dagger_r218.yaml'
CONFIG=ROOT/'configs/exp/v6.22.2/pretrain45_dagger.yaml'
OUT=ROOT/'outputs/v6222-pretrain45'
ALLOWED={'run_name','track_manifest','dagger_initialization_stats_checkpoint','racing_line_cache',
    'mpcc_build_root','evaluation_episodes','reporting_evaluation_episodes','tags',
    'track_sampling_family_weights','dagger_replay_family_weights',
    'dagger_permanent_expert_required_families','reliability_family_aliases',
    'curriculum','evaluation_curriculum','reporting_evaluation_curriculum','reporting_evaluation_seed'}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(require_normalization=True):
    base=load_config(BASE)['dagger']; cfg=load_config(CONFIG); s=cfg['dagger']
    changed={k for k in set(base)|set(s) if base.get(k)!=s.get(k)}
    assert changed<=ALLOWED, f'learning recipe drift: {changed-ALLOWED}'
    for key in ('curriculum','evaluation_curriculum','reporting_evaluation_curriculum'):
        assert {k:v for k,v in s[key].items() if k not in ('tracks','name')}=={k:v for k,v in base[key].items() if k not in ('tracks','name')}
    records=json.loads(Path(s['track_manifest']).read_text())['records']
    train=[r for r in records if r['split']=='train']; val=[r for r in records if r['split']=='validation']
    assert len(train)==len(val)==45
    assert not set(r['path'] for r in train)&set(r['path'] for r in val)
    weights=s['track_sampling_family_weights']
    assert len(weights)==45 and set(weights.values())=={1.}
    assert Counter(r['family'] for r in train)==Counter(r['family'] for r in val)
    assert s['dagger_dynamic_sampling']['enabled'] and not s.get('dagger_dart_exempt_tracks')
    lock=json.loads((OUT/'manifest.lock.json').read_text())
    for r in records:
        assert digest(r['path'])==lock[r['path']]
        assert r['qualified'] and r['dart_eligible']
        assert r['source_family']==r['family'] and r['source_stratum']=='canonical'
        assert r['flight_mimic_start_gates']==list(range(len(load_track(r['path']).gates)))
    tracks=tuple(s['curriculum']['tracks'])
    schedule=dagger_episode_tracks(tracks,360,s,2026091922)
    assert set(Counter(schedule).values())=={8}
    targets=dagger_episode_start_gates(schedule,s,2026091922)
    modes=dagger_episode_start_modes(schedule,targets,s,2026091922)
    fraction=sum(x is not None for x in targets)/len(targets)
    assert .40<fraction<.60 and modes.count('expert_prefix')==sum(x is not None for x in targets)
    # Ensure the real reduction discovers every source; catches absent records,
    # wrong names, and source_stratum omissions before collecting a single row.
    synthetic={f'track/{r["name"]}/p{i}':max(0.,1-i*.08) for r in val for i in range(1,7)}
    competence,frontiers=dagger_family_survival_statistics(synthetic,s['track_manifest'],route_horizon=6)
    assert set(competence)==set(weights)==set(frontiers)
    normalization=Path(s['dagger_initialization_stats_checkpoint'])
    norm_info=None
    if require_normalization:
        import torch
        n=torch.load(normalization,map_location='cpu',weights_only=False)
        assert n['train_courses']==45 and len(n['traces'])==135 and n['rows']==34560
        assert n['source_recipe_sha256']==digest(ROOT/'scripts/audits/freeze_v621_normalization.py')
        norm_info={k:n[k] for k in ('train_courses','rows','provenance')}
    report=dict(status='PASS',config_sha256=digest(CONFIG),manifest_sha256=digest(s['track_manifest']),
        lock_sha256=digest(OUT/'manifest.lock.json'),hardware_sha256=digest(ROOT/'configs/hardware/dagger_local_16c_v625.yaml'),
        normalization_sha256=digest(normalization) if require_normalization else None,
        train=45,validation=45,episodes_per_round=360,updates_per_round=s['updates_per_round'],
        targeted_start_fraction=fraction,mapped_competence_sources=len(competence),
        changed_dagger_keys=sorted(changed),normalization=norm_info)
    if require_normalization:atomic_json(CONFIG.with_suffix('.preflight.json'),report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    validate('--without-normalization' not in sys.argv)
