"""Prepare matched 2/4/6-course slalom pilots and bounded comparisons."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train_privileged_racing import configured_tracks, load_config, stage_config
from starscream.dagger_quality import quality_config
from starscream.evaluation_suite import configure_selection_suite

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT/'configs/exp/v6.21.1.1'
SOURCE = ROOT/'outputs/diagnostics/v62111-train65-teacher-v4/frozen/approved/manifest.json'
PICKS = ('v6201_train_10_slalom_0','v6201_train_4_slalom_1',
         'v6201_train_5_slalom_0','v6201_train_6_slalom_1',
         'v6201_train_7_slalom_0','v6201_train_8_slalom_7')

def write(path, data): path.write_text(json.dumps(data,indent=2)+'\n')

def main():
    source=json.loads(SOURCE.read_text())
    byname={r['name']:r for r in source['records']}
    assert all(byname[n]['qualified'] and byname[n]['split']=='train' for n in PICKS)
    quality=json.loads((CONFIG_DIR/'plant_quality_v1.yaml').read_text())['dagger']['dagger_trajectory_quality']
    outcome={}
    for count in (2,4,6):
        names=PICKS[:count]
        records=[byname[n] for n in names]
        tracks=[r['path'] for r in records]
        families=[r['family'] for r in records]
        assert len(set(families))==count
        assert all(t in quality['track_bounds'] for t in tracks)
        manifest=dict(source,records=records,family_weights={f:1.0 for f in families})
        manifest_path=CONFIG_DIR/f'mini_slalom_{count}_train_manifest.json'
        write(manifest_path,manifest)
        # The one-course completed run collected 1.157M steps from 32x32.
        # Scale episodes and optimizer draws per course, not update rounds.
        common={
            'inherits':'mini_slalom_common.yaml',
            'dagger':{
                'episodes_per_round':32*count,
                'updates_per_round':313*count,
                'dagger_online_replay_rows_per_round':4167*count,
                'online_replay_capacity':130000*count,
                'dagger_permanent_expert_capacity':26043*count,
                'track_manifest':'/workspace/'+str(manifest_path.relative_to(ROOT)),
                'curriculum':{'tracks':tracks},
                'track_sampling_family_weights':{'__replace__':True,**{f:1.0 for f in families}},
                'dagger_replay_family_weights':{'__replace__':True,**{f:1.0 for f in families}},
                'dagger_permanent_expert_required_families':families,
            },
            'experiment_notes':{'mini_recovery_baseline':{
                'course_count':count,'training_courses':names,
                'expected_steps':1157386*count,
                'budget':'32 rounds x 32 episodes per course; actual steps are authoritative',
            }},
        }
        common_path=CONFIG_DIR/f'mini_slalom_{count}_common.yaml'
        write(common_path,common)
        paths=[]
        for kind in ('broad','bounded'):
            run=f'starscream-v6.21.1.1-mini-slalom{count}-{kind}-r32'
            dagger={'run_name':run,'tags':['mini-slalom',f'{count}-train-two-val',kind,
                                             'async-dagger','scratch']}
            if kind=='bounded':dagger['dagger_trajectory_quality']=quality
            config_path=CONFIG_DIR/f'mini_slalom_{count}_{kind}.yaml'
            write(config_path,{'inherits':common_path.name,'dagger':dagger,
                 'wandb':{'group':'v62111-mini-slalom-recovery-scaling',
                          'name':run,
                          'local_event_path':f'/workspace/outputs/logs/{run}.events.jsonl',
                          'local_full_event_path':f'/workspace/outputs/logs/{run}.full.events.jsonl',
                          'eval_metric_allowlist':['selection_suite_success','selection_suite_timely_success',
                                                   'selection_suite_clean_success','selection_suite_clean_timely_success',
                                                   'full_course_success','crash_rate','mean_gates','mean_steps',
                                                   'dagger_policy_version']}})
            d=stage_config(load_config(config_path),'dagger')['dagger']
            configure_selection_suite(d)
            assert configured_tracks(d)==tuple(tracks)
            assert d['episodes_per_round']==32*count and d['updates_per_round']==313*count
            assert d['evaluation_episodes']==16
            assert (quality_config(d) is not None)==(kind=='bounded')
            assert d['dagger_async_pipeline'] and d['dagger_capture_updates']
            assert d['dagger_packed_transport'] and d['dagger_overlap_teacher_inference']
            paths.append((config_path,d))
        differing={k for k in paths[0][1].keys()|paths[1][1].keys()
                   if paths[0][1].get(k)!=paths[1][1].get(k)}
        assert differing=={'run_name','tags','dagger_trajectory_quality'},differing
        outcome[count]={'courses':names,'families':families,'configs':[str(p.relative_to(ROOT)) for p,_ in paths],
                        'episodes_per_round':32*count,'updates_per_round':313*count,
                        'expected_steps':1157386*count,'only_differences':sorted(differing)}
    print(json.dumps(outcome,indent=2))

if __name__=='__main__':main()
