"""Freeze the v6.21.1.1 plant-aware scratch experiment and normalization schema."""
import copy
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from starscream.plant_privileged import (
    PLANT_OBSERVATION_CONTRACT, PLANT_SETTINGS_DIM, PLANT_SETTINGS_NAMES, PLANT_STATIC_SCALE)
from starscream.evaluation_suite import configure_selection_suite
from starscream.behavior_feedback import build_behavior_map
import hashlib


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-frontier',type=Path,
        help='Completed frozen train65 MPCC pace directory')
    args=parser.parse_args()
    original = json.loads(Path('configs/exp/v6.21.1/update_fix_dagger.yaml').read_text())
    config = copy.deepcopy(original)
    run = 'starscream-v6.21.1.1-plant-selection25-dagger'
    if (Path('outputs/checkpoints')/run).exists():
        raise FileExistsError('Experiment already exists; do not rewrite its recipe')
    root = Path('outputs/course-pools/v62111-plant')
    root.mkdir(parents=True, exist_ok=True)
    stats = torch.load(original['dagger']['dagger_initialization_stats_checkpoint'],
                       map_location='cpu', weights_only=False)
    stats = copy.deepcopy(stats)
    stats['normalizer']['mean'] = np.concatenate([stats['normalizer']['mean'], np.zeros(PLANT_SETTINGS_DIM, np.float32)])
    stats['normalizer']['std'] = np.concatenate([stats['normalizer']['std'], np.ones(PLANT_SETTINGS_DIM, np.float32)])
    stats['observation_contract'] = PLANT_OBSERVATION_CONTRACT
    stats['route_gate_count'] = 6
    stats['plant_settings_schema'] = dict(version=PLANT_OBSERVATION_CONTRACT,
        names=list(PLANT_SETTINGS_NAMES), static_scale=PLANT_STATIC_SCALE.tolist())
    torch.save(stats, root/'normalization.pt')
    (root/'plant-schema.json').write_text(json.dumps(stats['plant_settings_schema'], indent=2)+'\n')
    for section in ('dagger', 'checkpoint', 'wandb'):
        config[section]['run_name'] = run
    settings = config['dagger']
    if args.teacher_frontier:
        frontier=args.teacher_frontier.resolve()
        manifest=json.loads((frontier/'manifest.json').read_text())
        profiles=json.loads((frontier/'teacher-profiles.json').read_text())
        if manifest['teacher_pace_frontier']['contract']!=profiles['contract']:
            raise ValueError('Teacher profiles and manifest contracts differ')
        if {r['path'] for r in manifest['records'] if 'teacher_pace_evidence' in r}!=set(settings['curriculum']['tracks']):
            raise ValueError('Teacher tuning must preserve all training geometries')
        settings['track_manifest']=str(frontier/'manifest.json')
        settings['mpcc_manifest_teacher_profile_controller_configs'].update(profiles['controller_profiles'])
        settings['mpcc_manifest_teacher_profile_planner_configs'].update(profiles['planner_profiles'])
        settings['mpcc_speed_frontier_profile']=''
    settings.update(resume_checkpoint=None, initial_checkpoint=None,
        dagger_initialization_stats_checkpoint=str((root/'normalization.pt').resolve()),
        observation_contract=PLANT_OBSERVATION_CONTRACT,
        dagger_async_pipeline=True, dagger_event_inference=True,
        dagger_pipeline_max_handoff_bytes=2*1024**3, dagger_pipeline_timeout_seconds=900,
        evaluation_workers=64, evaluation_envs_per_worker=8)
    settings['model'].update(input_dim=103+PLANT_SETTINGS_DIM, observation_contract=PLANT_OBSERVATION_CONTRACT)
    settings.update(evaluation_suite_manifest='configs/eval/v6211_vision_selection25.json',
        evaluation_suite_sha256=hashlib.sha256(Path('configs/eval/v6211_vision_selection25.json').read_bytes()).hexdigest(),
        evaluation_suite_episodes_per_track=4, evaluation_seed=2034091462,
        reporting_evaluation_curriculum=None,
        dagger_completion_survival_selection=False,
        pace_aware_selection=False, monitor='selection_suite_success')
    config['checkpoint']['monitor'] = 'selection_suite_success'
    configure_selection_suite(settings)
    behavior_path = root/'real100-behavior-map.json'
    behavior_path.write_text(json.dumps(build_behavior_map(settings),indent=2)+'\n')
    settings['dagger_dynamic_sampling'].update(behavior_map=str(behavior_path.resolve()),
        behavior_map_sha256=hashlib.sha256(behavior_path.read_bytes()).hexdigest())
    settings['tags'] = ['v6.21.1.1', 'scratch', 'plant-privileged', 'async-dagger', 'selection25', 'real100-behavior-feedback']
    if args.teacher_frontier:
        settings['tags'].append('mpcc-pace-tuned')
    wb = config['wandb']
    wb.update(event_path=None, local_event_path=f'/workspace/outputs/logs/{run}.events.jsonl',
              local_full_event_path=f'/workspace/outputs/logs/{run}.full.events.jsonl',
              group='v6.21.1.1-plant', tags=settings['tags'])
    wb['eval_metric_allowlist'] = ['selection_suite_success', 'full_course_success', 'crash_rate', 'mean_gates',
        'mean_return', 'mean_steps', 'selection_score', 'p1', 'p3', 'p6', 'p10', 'p16',
        'successful_mean_speed_mps', 'successful_mean_steps', 'dagger_policy_version',
        'minimum_family_full_course_success']
    wb['train_metric_allowlist'] = [
        'round', 'loss', 'action_loss', 'physical_action_loss', 'dynamics_loss',
        'group_loss_mean', 'group_loss_max', 'group_loss_std', 'new_labels',
        'online_replay', 'permanent_expert_replay', 'teacher_valid_fraction',
        'teacher_solver_failure_fraction', 'teacher_execution_fraction',
        'successful_episode_label_fraction', 'targeted_start_episode_fraction',
        'environment_steps', 'environment_steps_per_second', 'round_seconds',
        'collection_seconds', 'collection_wait_seconds', 'updates_seconds',
        'evaluation_seconds', 'replay_prepare_seconds', 'replay_persisted_round',
        'new_online_replay_retention_fraction', 'learning_rate', 'safety_rollback',
        'memory_available_gib', 'swap_used_gib', 'memory_swap_in_bytes_per_second',
        'memory_swap_out_bytes_per_second', 'sampling_plan*',
        'dynamic_sampler_active', 'dynamic_sampler_priority_entropy',
        'dynamic_sampler_multiplier_min', 'dynamic_sampler_multiplier_max',
        'dynamic_sampler_competence_mean', 'dynamic_sampler_behavior_*',
        'collection_last_call_host/pipeline_*',
        'collection_last_call_host/inference_*', 'collection_last_call_host/wait_seconds',
        'pipeline_discard_wait_seconds']
    config['experiment_notes']['plant_experiment'] = dict(
        parent='v6.21.1-update-fix', scratch=True, fresh_replay=True,
        comparison='Parent training curriculum, supervision, replay and update budgets; plant prefix + zero FiLM, async execution, selection25 validation and real100 behavior feedback.',
        original_resume_removed=original['dagger']['resume_checkpoint'],
        settings_schema=str((root/'plant-schema.json').resolve()))
    if args.teacher_frontier:
        config['experiment_notes']['plant_experiment']['teacher_frontier']=str(args.teacher_frontier.resolve())
        config['experiment_notes']['plant_experiment']['comparison']=(
            'Parent training geometry, loss, replay and update budgets; plant prefix + zero FiLM, '
            'async execution, selection25 validation, real100 behavior feedback, and newly qualified '
            'per-course MPCC pace profiles. This is not an isolated observation ablation.')
    destination = Path('configs/exp/v6.21.1.1/plant_dagger.yaml')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2)+'\n')
    differences = {key:dict(old=original['dagger'].get(key),new=value)
        for key,value in settings.items() if original['dagger'].get(key) != value}
    (root/'recipe-diff.json').write_text(json.dumps(differences, indent=2)+'\n')
    print(destination)


if __name__ == '__main__':
    main()
