"""Matched stronger-DART recovery check for the riskiest tuned teacher changes.

The production qualification already covers every course at its actual DART
noise. This supplementary check doubles that noise on high-gain and recovery-
sensitive winners. It is a bounded robustness probe, not arbitrary-state rescue.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from scripts.audits.audit_dagger_teacher_labels import audit_track
from starscream.course_model.training import atomic_json


def main():
    torch.set_num_threads(1)
    root = Path('outputs/diagnostics/v62111-train65-teacher-v4/frozen')
    search = json.loads((root/'search-summary.json').read_text())
    settings = json.loads(Path('configs/exp/v6.21.1.1/plant_dagger.yaml').read_text())['dagger']
    manifest = json.loads(Path(settings['track_manifest']).read_text())
    records = {r['name']:r for r in manifest['records']}
    winners = [r for r in search['records'] if r['selected']['name'] != 'baseline']
    def gain(r):
        return 1-r['selected_median']/r['baseline_median'] if r['baseline_median'] else 0.
    def recovery(r):
        raw = json.loads((Path(r['evidence_root'])/r['name']/r['selected']['name']/'dart.json').read_text())
        return max(e['recovery_fraction'] for e in raw['episodes'])
    chosen = {r['name']:r for r in sorted(winners,key=gain,reverse=True)[:3]}
    chosen.update({r['name']:r for r in sorted(winners,key=recovery,reverse=True)[:3]})
    for family in ('braking','reversal','go_around','hairpin'):
        options = [r for r in winners if family in r['name']]
        if options:
            r = max(options,key=gain); chosen[r['name']] = r
    protocol = dict(search_contract=search['contract'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seeds=[621113900+1009*i for i in range(4)],dart_action_noise_scale=2.,
        selection='top three pace gains, top three DART recovery fractions, highest gain braking/reversal/go-around/hairpin')
    contract = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    output=root/'recovery-stress';output.mkdir(exist_ok=True)
    cache=[];results=[]
    for name,result in sorted(chosen.items()):
        sides={}
        for side in ('baseline','selected'):
            candidate=result[side]
            path=output/f'{name}-{side}.json'
            if path.exists():
                data=json.loads(path.read_text())
                assert data['contract']==contract and data['candidate']==candidate
            else:
                s=deepcopy(settings)
                s.update(mpcc_use_manifest_teacher_profile=False,mpcc_use_manifest_planner_profile=False,
                    mpcc_use_manifest_speed=False,mpcc_speed_frontier_profile='',mpcc_nominal_speed=16.5,
                    mpcc_config=candidate['controller'],mpcc_planner_config=candidate['planner'],
                    mpcc_family_configs={},mpcc_family_planner_configs={},mpcc_family_nominal_speeds={},
                    mpcc_build_root='/tmp/starscream-train65-recovery-check')
                report=audit_track(s,s['curriculum'],records[name]['path'],0,1,4,621113900,
                    start_mode='canonical',repeats_per_start=4,start_perturbation_scale=1.,
                    dart_action_noise_scale=2.,dart_episode_fraction=1.,speed_fractions=(1.,),
                    frontier_speed=candidate['speed'],backend_cache=cache)
                data=dict(contract=contract,candidate=candidate,episodes=report['episodes'])
                atomic_json(path,data)
            sides[side]=data['episodes']
        regressions=[]
        for before,after in zip(sides['baseline'],sides['selected']):
            assert before['seed']==after['seed']
            if before['success'] and (not after['success'] or
                    after['solver_failure_fraction'] > max(.06,before['solver_failure_fraction'])):
                regressions.append(after['seed'])
        row=dict(name=name,baseline_success=sum(e['success'] for e in sides['baseline']),
            selected_success=sum(e['success'] for e in sides['selected']),regression_seeds=regressions,
            baseline_recovery_max=max(e['recovery_fraction'] for e in sides['baseline']),
            selected_recovery_max=max(e['recovery_fraction'] for e in sides['selected']))
        results.append(row)
        atomic_json(output/'report.json',dict(contract=contract,protocol=protocol,complete=len(results)==len(chosen),
            passed=all(not r['regression_seeds'] for r in results),courses=results))
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
