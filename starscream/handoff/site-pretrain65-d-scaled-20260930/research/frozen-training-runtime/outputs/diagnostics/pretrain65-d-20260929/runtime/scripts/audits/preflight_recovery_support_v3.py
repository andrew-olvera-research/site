"""Verify the reviewable recovery-support recipe; never starts training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import train_privileged_racing as t
from starscream.dagger_quality import quality_config, track_quality_config
from starscream.dagger_schedule import dart_collection_round
from starscream.racing_evaluation import load_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/diagnostics/recovery-support-v3/preflight.json'))
    args = parser.parse_args()
    path = Path('configs/exp/v6.21.1.1/plant_recovery_support_v3.yaml')
    config = t.load_config(path); settings = config['dagger']
    base = t.load_config(Path('configs/exp/v6.21.1.1/plant_recovery_fix.yaml'))['dagger']
    allowed = {'run_name', 'monitor', 'tags', 'dagger_dart_refresh_interval',
               'ppo_reference_aware_gate_events', 'gate_event_reference_tolerance',
               'dagger_trajectory_quality', 'evaluation_clean_deadlines', 'timed_reporting'}
    changed = sorted(k for k in set(base) | set(settings) if base.get(k) != settings.get(k))
    if set(changed)-allowed:
        raise ValueError(f'unreviewed recipe changes: {set(changed)-allowed}')
    if any(settings.get(k) for k in ('initial_checkpoint', 'resume_checkpoint', 'dagger_anchor_checkpoint')):
        raise ValueError('candidate requires fresh weights/replay')
    q = quality_config(settings)
    assert q and q['implementation_version'] == 3 and not q.get('audit_only')
    assert [q[k] for k in ('nominal_fraction','corrective_fraction','recovery_fraction')] == [.35,.4,.25]
    assert q['precursor_seconds'] == 1 and q['precursor_fraction'] == .5
    assert q['permanent_corrective_max_multiplier'] == 2
    assert q['balance_corrective_encounters']
    tracks = t.configured_tracks(settings)
    assert len(tracks) == 65
    assert set(map(lambda p: str(Path(p).resolve()), tracks)) == set(q['track_bounds'])
    for track in tracks:
        track_quality_config(q, track)
    assert q['track_bounds'] == base['dagger_trajectory_quality']['track_bounds']
    assert settings['monitor'] == 'selection_suite_clean_timely_success'
    assert settings['ppo_reference_aware_gate_events'] and settings['gate_event_reference_tolerance'] == .75
    spec = settings['timed_reporting']
    protocol = load_protocol(spec['protocol'], spec['protocol_sha256'])
    expected = {r['name']:r['deadline_steps'] for r in protocol['records'] if r['name'] in settings['evaluation_suite_names']}
    assert len(expected) == 25 and settings['evaluation_clean_deadlines'] == expected
    assert 'quality/*' in config['wandb']['train_metric_allowlist']
    assert settings['dagger_dart_refresh_interval'] == 8
    dart_rounds = [r for r in range(1, settings['rounds']+1) if dart_collection_round(r, settings)]
    for r in dart_rounds:
        assert t.dagger_teacher_beta(r, settings) == 1
    root = Path('outputs/diagnostics/recovery-support-v3')
    suites = ET.parse(root/'tests.xml').getroot().iter('testsuite')
    test_count = 0
    for suite in suites:
        assert int(suite.get('failures', 0)) == int(suite.get('errors', 0)) == 0
        test_count += int(suite.get('tests', 0))
    probe = json.loads((root/'train65/report.json').read_text())['results'][0]
    assert probe['counters']['qualified_segments'] > 0 and probe['counters']['precursor_rows'] > 0
    assert probe['quota_shortfalls'] == [0,0,0]
    assert probe['sampled_fractions'][2] <= .25
    integration = json.loads((root/'integration-validation.json').read_text())
    assert integration['passed'] and integration['round'] == 4
    selection = json.loads((root/'selection-smoke.json').read_text())
    assert 'selection_suite_clean_timely_success' in selection['metrics']
    sources = [path, Path('configs/exp/v6.21.1.1/plant_recovery_fix.yaml'),
        Path('configs/exp/v6.21.1.1/plant_quality_v1.yaml'), Path('configs/exp/v6.21.1.1/plant_dagger.yaml'),
        *map(Path, ['scripts/train_privileged_racing.py', 'starscream/dagger_quality.py',
            'starscream/dagger_async.py', 'starscream/dagger_schedule.py', 'starscream/evaluation_suite.py',
            'starscream/racing_evaluation.py', 'scripts/audits/eval_privileged_real100_v2_dual.py'])]
    result = dict(passed=True, launched=False, run_name=settings['run_name'], changed_dagger_keys=changed,
        training_courses=65, rounds=settings['rounds'], episodes_per_round=settings['episodes_per_round'],
        updates_per_round=settings['updates_per_round'], batch_size=settings['batch_size'],
        fresh_weights=True, fresh_replay=True, teacher_profiles_and_bounds_unchanged=True,
        online_recovery_fraction=.25, combined_recovery_cap=.25*settings['online_fraction'],
        periodic_dart_rounds=dart_rounds, test_count=test_count,
        collector_probe=probe['counters'], integration=integration,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        limitation='Mechanical readiness only; no claim of improved learning or target success rate.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    (root/'resolved-config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','integration')},indent=2))


if __name__ == '__main__':
    main()
