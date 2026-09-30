"""Prepare a fixed-budget, second-seed collection screen and integration fixtures."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path('/workspace')
sys.path.insert(0, str(ROOT))
from starscream.dagger_collection import collection_config
OUT = ROOT / 'outputs/diagnostics/collection-next-20260929'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT/'configs/exp/v6.21.1.1/mini_slalom_aug2_fixes.yaml').read_text())
    d = json.loads((ROOT/'configs/exp/v6.21.1.1/mini_slalom_aug2_collection_d.yaml').read_text())['dagger']['dagger_collection_control']
    full = {**d, 'segment_gates':None, 'coherent_learner':True}
    arms = [
        ('a_repeat', None, {}, 'Broad per-step mixture; second-seed control'),
        ('d_repeat', d, {}, 'Learner 2–4 accepted-gate segments; second-seed control'),
        ('full_learner', full, {}, 'Coherent learner full episodes'),
        ('gates_1_2', {**d, 'segment_gates':[1,2]}, {}, 'Short learner gate segments'),
        ('gates_4_6', {**d, 'segment_gates':[4,6]}, {}, 'Long learner gate segments'),
        ('time_2s', {**full, 'segment_seconds':2.}, {}, 'Two-second learner windows after prefix'),
        ('time_4s', {**full, 'segment_seconds':4.}, {}, 'Four-second learner windows after prefix'),
        ('full_expert20', {**full, 'expert_episode_probability':.2}, {}, 'Episode-coherent expert/learner mixture'),
        ('segments_expert20', {**d, 'coherent_learner':True, 'expert_episode_probability':.2}, {}, 'Coherent expert/learner gate segments'),
        ('recovery_4s', {**full, 'recovery_control':'learner', 'maximum_distance':50., 'stop_on_invalid_label':False, 'recovery_seconds':4.}, {}, 'Learner recovery clock; relaxed distance and no invalid-query stop'),
        ('recovery_8s', {**full, 'recovery_control':'learner', 'maximum_distance':50., 'stop_on_invalid_label':False, 'recovery_seconds':8.}, {}, 'Eight-second counterpart to recovery_4s'),
        ('d_equal_axes', d, {'action_dimension_weights':[1.,1.,1.,1.]}, 'D collection with equal action-axis weights'),
    ]
    jobs = []
    for arm, control, patch, description in arms:
        c = copy.deepcopy(base)
        s = c['dagger']
        name = f'starscream-collection-next-{arm}-r24'
        s.update(run_name=name, seed=2026092913, top_k=1,
                 monitor='selection_suite_timely_success', mpcc_build_root=f'/tmp/collection-next-{arm}', **patch)
        if control is not None:
            s['dagger_collection_control'] = control
        collection_config(s)
        c['checkpoint'].update(run_name=name, top_k=1, monitor='selection_suite_timely_success')
        c['wandb'].update(run_name=name, name=name, group='collection-next-20260929', env_file='/workspace/.env',
            local_event_path=f'/workspace/outputs/logs/{name}.events.jsonl',
            local_full_event_path=f'/workspace/outputs/logs/{name}.full.events.jsonl')
        s['tags']=['collection-next', 'scratch', arm, 'seed2']
        c.pop('experiment_notes', None)
        c['ablation_next'] = dict(description=description, seed=2026092913,
            retention='One best timely checkpoint and latest; replay retained during run, latest three plus permanent shards after completion')
        path = ROOT / f'configs/exp/v6.21.1.1/collection_next_{arm}.json'
        path.write_text(json.dumps(c, indent=2)+'\n')
        jobs.append(dict(arm=arm, config=str(path), run_name=name, description=description, rounds=24))
    (OUT/'jobs.json').write_text(json.dumps(jobs, indent=2)+'\n')
    # Reuse an already validated small simulator fixture; preserve production methods.
    template = json.loads((ROOT/'outputs/diagnostics/collection-ablation-20260929/smoke-d.json').read_text())
    for arm in ('full_learner', 'time_2s', 'full_expert20', 'recovery_4s'):
        source = next(j for j in jobs if j['arm']==arm)
        cfg = json.loads(Path(source['config']).read_text())
        c = copy.deepcopy(template)
        name = f'next-smoke-{arm}'
        c['output_root'] = str(OUT/'smoke')
        s = c['dagger']
        s.update(run_name=name, rounds=3, episodes_per_round=9, updates_per_round=3,
                 dagger_permanent_expert_capacity=1, top_k=1,
                 dagger_collection_control=cfg['dagger']['dagger_collection_control'],
                 mpcc_build_root=f'/tmp/collection-next-smoke-{arm}')
        c['checkpoint'].update(run_name=name, top_k=1)
        c['wandb'].update(enabled=False, run_name=name, event_path=None,
            local_event_path=str(OUT/f'{name}.events.jsonl'),
            local_full_event_path=str(OUT/f'{name}.full.events.jsonl'))
        (OUT/f'smoke-{arm}.json').write_text(json.dumps(c, indent=2)+'\n')
    print(json.dumps(jobs, indent=2))

if __name__ == '__main__':
    main()
