"""Audit MPCC labels for train/held-out variants of two fixed slalom parents."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.train_privileged_racing import load_config
from scripts.audits.audit_dagger_teacher_labels import audit_track

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'configs/exp/v6.21.1.1/mini_slalom_2_train_manifest.json'
AUG_ROOT=ROOT/'outputs/course-pools/mini-slalom-aug2'
OUT=ROOT/'outputs/diagnostics/mini-slalom-aug2'
CONFIG=ROOT/'configs/exp/v6.21.1.1/mini_slalom_2_broad.yaml'

def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    parent_manifest=json.loads(SOURCE.read_text())
    parents={r['family']:r for r in parent_manifest['records']}
    generated=json.loads((AUG_ROOT/'manifest.json').read_text())['records']
    records=list(parent_manifest['records'])
    for row in generated:
        parent=parents[row['family']]
        clone=deepcopy(parent)
        clone.update(name=row['name'],path=str(AUG_ROOT/row['path']),
                     slot=row['name'],split='train' if row['split']=='train' else 'family_validation',
                     fingerprint=row['track_fingerprint'],geometry_fingerprint=row['geometry_fingerprint'],
                     qualified=True,augmentation_parent=parent['name'],
                     augmentation_seed=row['seed'])
        records.append(clone)
    candidate=dict(parent_manifest,records=records)
    candidate_path=OUT/'candidate-manifest.json'
    write(candidate_path,candidate)
    settings=load_config(CONFIG)['dagger']
    settings['track_manifest']=str(candidate_path)
    stage=deepcopy(settings['curriculum'])
    stage['tracks']=[r['path'] for r in records]
    results=[]
    for i,row in enumerate(generated):
        path=str(AUG_ROOT/row['path'])
        parent=parents[row['family']]
        report=dict(name=row['name'],split=row['split'],path=path,parent=parent['name'])
        for domain in ('nominal','randomized'):
            effective=deepcopy(settings)
            if domain=='nominal':effective['dynamics_randomization']={'enabled':False}
            result=audit_track(effective,stage,path,i,len(generated),1,
                2026092912 if domain=='nominal' else 2026092913,
                start_mode='canonical',repeats_per_start=1,
                start_perturbation_scale=1.0,dart_action_noise_scale=0.0,
                dart_episode_fraction=0.0,speed_fractions=(1.0,),
                frontier_speed=float(parent['qualified_speed_mps']))
            report[domain]=result['summary']
        report['qualified']=all(report[d]['success_rate']==1.0 and
             report[d]['solver_failure_fraction']<0.1 for d in ('nominal','randomized'))
        results.append(report)
        write(OUT/'qualification.json',dict(records=results))
        print(f"qualified {i+1}/{len(generated)} {row['name']} {report['qualified']} "
              f"lap={report['nominal']['successful_lap_time_seconds']}",flush=True)
    if not all(r['qualified'] for r in results):
        raise RuntimeError('Some augmented tracks failed teacher audit; inspect qualification.json')

if __name__=='__main__':main()
