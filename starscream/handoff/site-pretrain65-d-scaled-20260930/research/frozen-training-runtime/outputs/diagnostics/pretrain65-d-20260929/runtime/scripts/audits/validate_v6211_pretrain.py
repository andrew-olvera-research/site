"""Fail-closed release validation for the v6.21.1-pretrain data version."""
from collections import Counter
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json
from starscream.env.procedural_tracks import geometry_fingerprint
from starscream.env.racing_manifold.benchmark_v22 import CloneIndex
from starscream.env.racing_manifold.corpus_coverage import geometry_record
from starscream.env.tracks import load_track
from scripts.audits.review_v622_geometry import resampled
from scripts.train_privileged_racing import (dagger_episode_start_gates,
    dagger_episode_start_modes, dagger_episode_tracks, dagger_family_survival_statistics,
    load_config)

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/course-pools/v6211-pretrain'
def resolve(path):
 p=Path(path);return ROOT/p.relative_to('/workspace') if str(p).startswith('/workspace/') else p

def main():
 m=json.loads((OUT/'manifest.json').read_text()); old=json.loads((ROOT/'outputs/course-pools/v621-additions/manifest.json').read_text())
 rows=[r for r in m['records'] if r['split']=='train']; anchors=[r for r in rows if r['lineage_role']=='v6.21_immutable_anchor']; added=[r for r in rows if r['lineage_role']=='v6.22.x_utt_support_addition']
 assert len(rows)==65 and len(anchors)==45 and len(added)==20 and len({r['name'] for r in rows})==65
 assert len({r['family'] for r in rows})==65
 assert all(r['source_family']==r['family'] and r['source_stratum']=='canonical' for r in rows)
 assert all(r.get('dart_eligible') for r in added)
 assert all(r['flight_mimic_start_gates']==list(range(r['geometry']['gate_count'])) for r in added)
 old_train=[r for r in old['records'] if r['split']=='train']
 assert [(r['name'],r['path'],r['fingerprint']) for r in anchors]==[(r['name'],r['path'],r['fingerprint']) for r in old_train]
 assert all(resolve(r['path']).is_file() for r in m['records'])
 assert all(r.get('qualified_speed_mps') and r.get('qualification',{}).get('qualified') for r in added)
 eval_rows=[]
 for p in ('configs/eval/v6_22_real60.manifest.json','configs/eval/v6_22_real100_hard_v2.manifest.json'):
  eval_rows += json.loads((ROOT/p).read_text())['records']
 eval_paths={str(resolve(r['path'])) for r in eval_rows};eval_fp={r['fingerprint'] for r in eval_rows}
 assert not {str(resolve(r['path'])) for r in rows}&eval_paths
 assert not {r['fingerprint'] for r in added}&eval_fp
 protected=[load_track(resolve(r['path'])) for r in eval_rows];idx=CloneIndex([resampled(t) for t in protected])
 distances=[idx.distance(resampled(load_track(resolve(r['path'])))) for r in added]
 assert min(distances)>=.06
 normalization_path=OUT/'normalization.pt'
 assert normalization_path.is_file()
 normalization=torch.load(normalization_path,map_location='cpu',weights_only=False)
 track_counts=normalization['track_counts']
 assert set(track_counts)=={r['path'] for r in rows}
 assert len(set(track_counts.values()))==1
 assert normalization['accepted_episodes']>0
 for key,value in normalization['normalizer'].items():
  if isinstance(value,(np.ndarray,torch.Tensor)): assert np.isfinite(value).all(),key
 assert np.isfinite(normalization['dynamics_target_mean']).all()
 assert np.isfinite(normalization['dynamics_target_std']).all()
 s=load_config(ROOT/'configs/exp/v6.21.1/pretrain65_dagger.yaml')['dagger']
 assert s['rounds']==218 and s['episodes_per_round']==8*65
 assert s['updates_per_round']==int(np.ceil(2026*65/45))
 assert s['dagger_online_replay_rows_per_round']==int(np.ceil(27001*65/45))
 assert s['online_replay_capacity']==int(np.ceil(1575001*65/45))
 assert s['dagger_permanent_expert_capacity']==int(np.ceil(675000*65/45))
 assert set(s['track_sampling_family_weights'])=={r['family'] for r in rows}
 assert set(s['dagger_replay_family_weights'])=={r['family'] for r in rows}
 assert set(s['dagger_permanent_expert_required_families'])=={r['family'] for r in rows}
 assert not s.get('dagger_dart_exempt_tracks')
 tracks=tuple(s['curriculum']['tracks'])
 local_settings=dict(s);local_settings['track_manifest']=str(OUT/'manifest.json')
 schedule=dagger_episode_tracks(tracks,s['episodes_per_round'],local_settings,s['seed'])
 assert set(Counter(schedule).values())=={8}
 targets=dagger_episode_start_gates(schedule,local_settings,s['seed'])
 modes=dagger_episode_start_modes(schedule,targets,local_settings,s['seed'])
 assert .40 < sum(x is not None for x in targets)/len(targets) < .60
 assert modes.count('expert_prefix')==sum(x is not None for x in targets)
 synthetic={f'track/{r["name"]}/p{i}':max(0.,1-i*.08) for r in rows for i in range(1,7)}
 competence,frontiers=dagger_family_survival_statistics(synthetic,OUT/'manifest.json',route_horizon=6)
 assert set(competence)==set(frontiers)=={r['source_family'] for r in rows}
 geo=[geometry_record(load_track(resolve(r['path']))) for r in rows]
 def count(pred):return sum(sum(pred(t) for t in g['transitions']) for g in geo)
 report={'schema':'starscream-v6211-pretrain-validation-v1','status':'PASS','train_courses':65,
  'v621_immutable_anchors':45,'v622x_qualified_additions':20,
  'family_counts':dict(Counter(r['family'] for r in rows)),
  'geometry':{'gate_count_min':min(g['gate_count'] for g in geo),'gate_count_max':max(g['gate_count'] for g in geo),
    'courses_ge_13_gates':sum(g['gate_count']>=13 for g in geo),
    'long_incoming_ge20':count(lambda t:t['incoming_m']>=20),'very_long_incoming_ge30':count(lambda t:t['incoming_m']>=30),
    'hard_after_long':count(lambda t:t['incoming_m']>=20 and t['turn_deg']>=105),
    'low_gate_centers':count(lambda t:t.get('gate_center_height_m',99)<=1.2),
    'narrow_gates':count(lambda t:min(t.get('width_m',99),t.get('height_m',99))<=1.65)},
  'minimum_new_course_continuous_distance_to_v622_evals':float(min(distances)),
  'recipe':{'rounds':s['rounds'],'episodes_per_round':s['episodes_per_round'],
    'episodes_per_course_per_round':8,'updates_per_round':s['updates_per_round'],
    'singleton_collection_and_replay_families':len(s['track_sampling_family_weights']),
    'targeted_start_fraction':sum(x is not None for x in targets)/len(targets),
    'dart_exempt_courses':0,'mapped_competence_sources':len(competence)},
  'lineage':m['lineage'],'manifest_sha256':hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest(),
  'normalization':{'ready':True,'sha256':hashlib.sha256(normalization_path.read_bytes()).hexdigest(),
    'contract':normalization['contract'],'episodes_requested':normalization['episodes'],
    'episodes_accepted':normalization['accepted_episodes'],
    'episodes_rejected':normalization['rejected_episodes'],
    'rows':normalization['rows'],'rows_per_course':next(iter(track_counts.values())),
    'courses':len(track_counts),'valid_dynamics_rows':normalization['valid_dynamics_rows'],
    'teacher_solver_failures':normalization['teacher_solver_failures']},
  'config':'configs/exp/v6.21.1/pretrain65_dagger.yaml'}
 atomic_json(OUT/'validation-report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
