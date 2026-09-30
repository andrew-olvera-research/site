"""Compare a tuned runtime reference with its published manifest configuration."""
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts import train_privileged_racing as t
from starscream.course_model.training import atomic_json


def main():
    root=Path('outputs/diagnostics/v62111-train65-teacher-v4')
    available=[json.loads(p.read_text()) for p in root.glob('*/selection.json')]
    result=next(r for r in available if r['selected']['name']!='baseline')
    selected=result['selected']
    settings=json.loads(Path('configs/exp/v6.21.1.1/plant_dagger.yaml').read_text())['dagger']
    manifest=json.loads(Path(settings['track_manifest']).read_text())
    record=next(r for r in manifest['records'] if r['name']==result['name'])
    record['qualified_speed_mps']=selected['speed']
    record['qualification']['selected']['teacher_profile']='wiring-smoke'
    output=root/'wiring-smoke';output.mkdir(exist_ok=True)
    atomic_json(output/'manifest.json',manifest)
    published=deepcopy(settings)
    published['track_manifest']=str((output/'manifest.json').resolve())
    published['mpcc_manifest_teacher_profile_controller_configs']['wiring-smoke']=selected['controller']
    published['mpcc_manifest_teacher_profile_planner_configs']['wiring-smoke']=selected['planner']
    runtime=deepcopy(settings)
    runtime.update(mpcc_use_manifest_teacher_profile=False,mpcc_use_manifest_speed=False,
        mpcc_nominal_speed=16.5,mpcc_config=selected['controller'],mpcc_planner_config=selected['planner'],
        mpcc_family_configs={},mpcc_family_planner_configs={},mpcc_family_nominal_speeds={},mpcc_speed_frontier_profile='')
    traces=[]
    for i,s in enumerate((runtime,published)):
        s['mpcc_build_root']=str((output/f'backend-{i}').resolve())
        env=t.make_env(s,track=record['path'])
        try:
            expert=t.RoutedDaggerTeacher(env,s,worker_index=0)
            observation,start=t.reset_env(env,t.parse_stage(s['curriculum']),seed=621112600,episode_index=0)
            expert.set_nominal_speed(selected['speed']);expert.reset()
            trace=[]
            for step in range(s['curriculum']['max_steps']):
                command=expert(observation)
                action=np.asarray(command.action.as_array(),np.float32)
                expert.observe_executed_action(action)
                observation,_,terminated,_,info=env.step(action)
                trace.append(np.concatenate([action,np.asarray(observation['state']),[command.solver_status]]))
                if terminated or env.tracker.passed_count-start>=len(env.track.gates):break
            traces.append(np.asarray(trace))
        finally:env.close()
    np.testing.assert_allclose(traces[0],traces[1],rtol=0,atol=1e-6)
    report=dict(course=result['name'],profile=selected['name'],steps=len(traces[0]),
        max_absolute_difference=float(np.max(np.abs(traces[0]-traces[1]))),
        published_manifest_matches_tuner=True)
    atomic_json(output/'report.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
