"""Build D at pretrain65 update-fix scale, retaining the corrected D recipe."""
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/workspace')
OUT = ROOT/'outputs/diagnostics/pretrain65-d-20260929'
CONFIG = ROOT/'configs/exp/v6.21.1.1/pretrain65_d_scaled.json'
NAME = 'starscream-pretrain65-d-scaled-20260929'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT/'configs/exp/v6.21.1/update_fix_dagger.yaml').read_text())['dagger']
    c = json.loads((ROOT/'configs/exp/v6.21.1.1/collection_next_d_repeat.json').read_text())
    s = c['dagger']
    for key in ('curriculum', 'track_manifest', 'track_split', 'track_sampling_family_weights',
                'dagger_replay_family_weights', 'dagger_permanent_expert_required_families',
                'mpcc_manifest_teacher_profile_controller_configs', 'mpcc_manifest_teacher_profile_planner_configs'):
        s[key] = copy.deepcopy(old[key])
    for key in list(s):
        if key.startswith('evaluation_suite_') or key == 'evaluation_track_speed_commands':
            del s[key]
    suite = ROOT/'configs/eval/v6211_vision_selection25.json'
    records = json.loads(suite.read_text())['records']
    timed = json.loads((ROOT/'configs/eval/real100_v2_timed_protocol_v1.json').read_text())
    deadlines = {r['name']:r['deadline_steps'] for r in timed['records']}
    s.update(run_name=NAME, seed=2026092931, rounds=218, episodes_per_round=520,
        updates_per_round=5081, initial_checkpoint=None, resume_checkpoint=None,
        mpcc_build_root='/tmp/pretrain65-d-scaled',
        dagger_initialization_stats_checkpoint=str(OUT/'normalization.pt'),
        evaluation_suite_manifest=str(suite),
        evaluation_suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
        evaluation_suite_episodes_per_track=8, evaluation_workers=16, evaluation_envs_per_worker=4,
        evaluation_episodes=200, evaluation_clean_deadlines={r['name']:deadlines[r['name']] for r in records},
        top_k=3, tags=['pretrain65', 'selection25', 'D', 'scratch', 'scaled', 'corrected-input-loss'])
    # Keep D's collection/update ratio and 15-round online replay horizon.
    for key in ('dagger_online_replay_rows_per_round', 'online_replay_capacity', 'dagger_permanent_expert_capacity'):
        s[key] = math.ceil(s[key]*520/64)
    s['evaluation_curriculum']['tracks'] = [str(ROOT/r['path']) if not Path(r['path']).is_absolute() else r['path'] for r in records]
    c['checkpoint'].update(run_name=NAME, top_k=3)
    c['wandb'].update(run_name=NAME, name=NAME, group='pretrain65-d-20260929',
        local_event_path=f'/workspace/outputs/logs/{NAME}.events.jsonl',
        local_full_event_path=f'/workspace/outputs/logs/{NAME}.full.events.jsonl')
    c['wandb']['train_metric_allowlist'] += ['collection/*', 'fresh_*', 'online_replay_*', 'permanent_expert_*']
    c['wandb']['eval_metric_allowlist'] += ['track/*', 'selection_family/*', 'timely_success', 'clean_timely_success']
    c.pop('ablation_next',None)
    c['experiment_notes'] = dict(source_D='collection_next_d_repeat.json', scale_source='v6.21.1/update_fix_dagger.yaml',
        budget='218 rounds, 520 episodes/round, 5081 updates/round, batch 1536; actual D labels/steps are not assumed equal to broad collection',
        replay='D capacities and retention scaled by 520/64; 90% online, 10% permanent; 15-round online horizon',
        learning_rate='Preserve tested D constant 1.2e-4; no untested cosine change',
        bootstrap='Preserve D one expert/DART round and expert prefixes; all 65 required families',
        validation='Frozen selection25, eight paired starts/course every round; family-weighted timely checkpoint selection; report clean/timely and eventual too')
    CONFIG.write_text(json.dumps(c,indent=2)+'\n')
    bootstrap=copy.deepcopy(c)
    bootstrap['dagger'].update(rollout_envs=8, dagger_async_pipeline=False, dagger_event_inference=False,
        dagger_host_inference_graph=False, cuda_graph_policy_inference=False, mpcc_build_root='/tmp/pretrain65-d-norm')
    (OUT/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
    print(CONFIG)

if __name__=='__main__': main()
