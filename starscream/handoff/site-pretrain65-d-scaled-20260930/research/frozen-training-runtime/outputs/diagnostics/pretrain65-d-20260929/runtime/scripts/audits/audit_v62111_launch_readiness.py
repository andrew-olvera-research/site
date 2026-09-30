"""Validate the final scratch recipe and write a local launch certificate.

This checks evidence and configuration; it never starts policy training.
"""
import hashlib
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts import train_privileged_racing as t
from starscream.evaluation_suite import configure_selection_suite
from starscream.behavior_feedback import sampler_feedback
from starscream.memory_pressure import memory_pressure_snapshot
from starscream.course_model.training import atomic_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true',help='Verify an existing certificate without rewriting it')
    args=parser.parse_args()
    path=Path('configs/exp/v6.21.1.1/plant_dagger.yaml')
    config=json.loads(path.read_text());settings=config['dagger']
    frontier=Path(settings['track_manifest']).parent
    if args.verify:
        certificate=json.loads((frontier/'launch-readiness.json').read_text())
        assert certificate['ready'] and certificate['run']==settings['run_name']
        for field,file in dict(configuration_sha256=path,manifest_sha256=settings['track_manifest'],
                teacher_report_sha256=frontier/'report.json',
                recovery_audit_sha256=frontier/'recovery-stress/report.json',
                behavior_map_sha256=settings['dagger_dynamic_sampling']['behavior_map'],
                selection_suite_sha256=settings['evaluation_suite_manifest'],
                validation_audit_sha256='outputs/dagger-throughput/selection25-native-audit.json').items():
            if sha(file)!=certificate[field]:raise ValueError(f'Launch evidence changed: {file}')
        for file,digest in certificate['source_sha256'].items():
            if sha(file)!=digest:raise ValueError(f'Launch source changed: {file}')
        print('Launch certificate matches the qualified experiment.')
        return
    original=json.loads(Path('configs/exp/v6.21.1/update_fix_dagger.yaml').read_text())['dagger']
    report=json.loads((frontier/'report.json').read_text())
    assert report['courses']==report['refreshed_qualified']==65
    recovery=json.loads((frontier/'recovery-stress/report.json').read_text())
    assert recovery['complete'] and recovery['passed']
    assert recovery['protocol']['search_contract']==report['contract']
    assert settings['curriculum']==original['curriculum']
    for key in ('rounds','episodes_per_round','updates_per_round','batch_size',
                'online_replay_capacity','teacher_beta_start','teacher_beta_end',
                'teacher_beta_schedule_rounds','dagger_dart_action_noise_std_normalized',
                'dynamics_randomization','control_hz'):
        assert settings.get(key)==original.get(key),key
    assert not settings['resume_checkpoint'] and not settings['initial_checkpoint']
    run=settings['run_name']
    assert config['checkpoint']['run_name']==config['wandb']['run_name']==run
    assert not (Path('outputs/checkpoints')/run).exists()
    assert settings['dagger_async_pipeline'] and settings['dagger_event_inference']
    assert settings['model']['input_dim']==167
    assert settings['model']['speed_conditioning']
    stats=torch.load(settings['dagger_initialization_stats_checkpoint'],map_location='cpu',weights_only=False)
    np.testing.assert_array_equal(stats['normalizer']['mean'][103:],np.zeros(64))
    np.testing.assert_array_equal(stats['normalizer']['std'][103:],np.ones(64))
    manifest=json.loads(Path(settings['track_manifest']).read_text())
    records=[r for r in manifest['records'] if r['path'] in settings['curriculum']['tracks']]
    assert len(records)==65
    for r in records:
        profile=r['qualification']['selected']['teacher_profile']
        assert profile in settings['mpcc_manifest_teacher_profile_controller_configs']
        assert profile in settings['mpcc_manifest_teacher_profile_planner_configs']
        assert r['qualification']['refreshed_qualified']
    configure_selection_suite(settings)
    validation=json.loads(Path('outputs/dagger-throughput/selection25-native-audit.json').read_text())
    assert validation['suite_sha256']==settings['evaluation_suite_sha256']
    assert validation['conditioning']['commands']==settings['evaluation_track_speed_commands']
    assert validation['conditioning']['batched_commands_match_episode_labels']
    assert validation['conditioning']['plant_constants_stable_in_history']
    assert len(validation['rows'])==100
    sampler_feedback(validation['metrics'],settings,settings['track_manifest'],6)
    assert settings['monitor']==config['checkpoint']['monitor']=='selection_suite_success'
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # Emit an explicit, reviewable per-course command label catalog, preserving
    # the student suite's command and seed contract.
    suite=json.loads(Path(settings['evaluation_suite_manifest']).read_text())
    commands=[dict(name=r['name'],slot=r['slot'],family=r['family'],path=track,
        speed_command_mps=settings['evaluation_track_speed_commands'][track])
        for r,track in zip(suite['records'],settings['evaluation_curriculum']['tracks'])]
    atomic_json(frontier/'validation-command-labels.json',dict(suite_sha256=settings['evaluation_suite_sha256'],
        conditioning='Per-episode command, collated into policy speed conditioning alongside privileged plant inputs',
        records=commands))
    sources=list(Path('starscream').rglob('*.py'))+[Path('scripts/train_privileged_racing.py'),
        Path(settings['dagger_initialization_stats_checkpoint'])]
    sources.extend(Path(r['path']) for r in records)
    sources.extend(Path(track) for track in settings['evaluation_curriculum']['tracks'])
    certificate=dict(ready=True,run=run,training_started=False,configuration_sha256=sha(path),
        source_sha256={str(file):sha(file) for file in sources},
        teacher_contract=report['contract'],teacher_report_sha256=sha(frontier/'report.json'),
        recovery_audit_sha256=sha(frontier/'recovery-stress/report.json'),
        manifest_sha256=sha(settings['track_manifest']),
        behavior_map_sha256=sha(settings['dagger_dynamic_sampling']['behavior_map']),
        selection_suite_sha256=settings['evaluation_suite_sha256'],
        validation_audit_sha256=sha('outputs/dagger-throughput/selection25-native-audit.json'),
        qualified_teachers=65,validation_courses=25,validation_episodes=100,
        bf16_supported=True,memory=memory_pressure_snapshot(),
        limitations=['Bounded teacher search, not global time optimality',
            'Recovery stress is targeted; arbitrary learner states are not guaranteed recoverable',
            'Functional smoke evaluation does not establish learned policy quality',
            'Full production replay residency and throughput still require live telemetry'])
    atomic_json(frontier/'launch-readiness.json',certificate)
    print(json.dumps(certificate,indent=2))


if __name__=='__main__':main()
