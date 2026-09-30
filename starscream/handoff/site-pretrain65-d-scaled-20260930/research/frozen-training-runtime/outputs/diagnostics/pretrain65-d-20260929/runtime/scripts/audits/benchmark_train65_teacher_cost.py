"""Warm single-worker CPU cost check on low/middle/high-gain teacher cases."""
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.audits.audit_dagger_teacher_labels import audit_track
from starscream.course_model.training import atomic_json


def main():
    root=Path('outputs/diagnostics/v62111-train65-teacher-v4/frozen/approved')
    search=json.loads((root/'search-summary.json').read_text())
    settings=json.loads(Path('configs/exp/v6.21.1.1/plant_dagger.yaml').read_text())['dagger']
    records={r['name']:r for r in json.loads(Path(settings['track_manifest']).read_text())['records']}
    eligible=[r for r in search['records'] if r['selected']['name']!='baseline'
              and r['confirmations']['baseline']['randomized']['usable']==6]
    eligible.sort(key=lambda r:1-r['selected_median']/r['baseline_median'])
    selected=[eligible[i] for i in sorted({0,len(eligible)//2,len(eligible)-1})]
    cache=[];rows=[]
    def run(record,candidate,repeats):
        s=deepcopy(settings)
        s.update(mpcc_use_manifest_teacher_profile=False,mpcc_use_manifest_speed=False,
            mpcc_speed_frontier_profile='',mpcc_nominal_speed=16.5,
            mpcc_config=candidate['controller'],mpcc_planner_config=candidate['planner'],
            mpcc_family_configs={},mpcc_family_planner_configs={},mpcc_family_nominal_speeds={},
            mpcc_build_root='/tmp/starscream-train65-cost-check')
        started=time.perf_counter()
        result=audit_track(s,s['curriculum'],record['path'],0,1,repeats,621112600,
            start_mode='canonical',repeats_per_start=repeats,start_perturbation_scale=1.,
            dart_action_noise_scale=0.,dart_episode_fraction=0.,speed_fractions=(1.,),
            frontier_speed=candidate['speed'],backend_cache=cache)
        seconds=time.perf_counter()-started
        episodes=result['episodes']
        assert all(r['success'] for r in episodes)
        steps=sum(r['steps'] for r in episodes)
        return dict(wall_seconds=seconds,steps=steps,wall_seconds_per_step=seconds/steps,
            mean_physical_lap_seconds=statistics.mean(r['lap_time_seconds'] for r in episodes),
            solver_p95_median_ms=statistics.median(r['solve_time_p95_ms'] for r in episodes))
    for result in selected:
        record=records[result['name']]
        # Warm every immutable solver and path before recording either side.
        run(record,result['baseline'],1);run(record,result['selected'],1)
        measured={}
        for side in ('baseline','selected'):
            measured[side]=run(record,result[side],2)
        rows.append(dict(name=result['name'],profile=result['selected']['name'],**measured))
    report=dict(protocol='one worker, warmed solver/path, two matched randomized episodes; low/middle/high gain among improved courses with 6/6 baseline usable',
        scope='Diagnostic CPU teacher cost, not a full parallel DAgger throughput benchmark',courses=rows)
    atomic_json(root/'cpu-cost-check.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
