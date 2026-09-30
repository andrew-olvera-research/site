"""Frozen selection-suite paths, matched seeds, and population-family scoring."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def repository_path(value):
    path = Path(value)
    if path.is_absolute():
        # Saved container paths remain readable in a host-side checkout.
        if str(path).startswith('/workspace/') and not path.exists():
            path = ROOT / str(path)[len('/workspace/'):]
        return path
    return ROOT / path


def configure_selection_suite(settings):
    path = settings.get('evaluation_suite_manifest')
    if not path:
        return
    path = repository_path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = settings.get('evaluation_suite_sha256')
    if not expected or digest != expected:
        raise ValueError('Selection suite hash is missing or differs from the frozen manifest')
    suite = json.loads(path.read_text())
    records = suite['records']
    if len({r['name'] for r in records}) != len(records) or len({r['slot'] for r in records}) != len(records):
        raise ValueError('Selection names and slots must be unique')
    tracks = [str(repository_path(row['path']).resolve()) for row in records]
    if not tracks or len(set(tracks)) != len(tracks):
        raise ValueError('Selection suite needs distinct courses')
    if any(not Path(track).is_file() for track in tracks):
        raise FileNotFoundError('Selection suite contains missing course files')
    train = {str(repository_path(track).resolve()) for track in settings['curriculum']['tracks']}
    if train.intersection(tracks):
        raise ValueError('Training/selection course overlap')
    weights = {family:float(weight) for family,weight in suite['family_weights'].items()}
    if set(weights) != {row['family'] for row in records} or any(w <= 0 for w in weights.values()) or not np.isclose(sum(weights.values()),1):
        raise ValueError('Selection family weights must cover the suite and sum to one')
    count = int(settings.get('evaluation_suite_episodes_per_track', 4))
    if count < 1:
        raise ValueError('Selection episodes per course must be positive')
    if settings.get('evaluation_episode_index_offset',0) or settings.get('evaluation_episode_index_stride',1) != 1:
        raise ValueError('Matched selection suite cannot use index offsets/strides')
    curriculum = dict(settings['evaluation_curriculum'])
    curriculum.update(name='selection25_real100_v2', tracks=tracks,
                      target_speed=16.5, manifest_speed_scale_range=None,
                      rollout_laps=1, random_gate=False, fixed_start_gate_index=0)
    settings['evaluation_curriculum'] = curriculum
    settings['evaluation_episodes'] = len(tracks)*count
    settings['evaluation_fixed_start_gate_index'] = 0
    settings['evaluation_suite_track_slots'] = dict(zip(tracks, (row['slot'] for row in records)))
    settings['evaluation_suite_names'] = {row['name']:row['family'] for row in records}
    settings['evaluation_suite_family_weights'] = weights
    settings['evaluation_track_speed_commands'] = {track:16.5 for track in tracks}
    settings['evaluation_manifest_speed_scale'] = 1.0
    settings['evaluation_suite_resolved_sha256'] = digest


def selection_episode_seed(settings, track, index, seed_base, course_count):
    slot = settings.get('evaluation_suite_track_slots', {}).get(str(Path(track).resolve()))
    if slot is None:
        return seed_base + 1009*index
    offset = int(hashlib.sha256(slot.encode()).hexdigest()[:8],16) % 1_000_000
    return seed_base + offset + index//course_count


def selection_metrics(metrics, settings, stage):
    if stage.name != 'selection25_real100_v2' or not settings.get('evaluation_suite_manifest'):
        return metrics
    families = {}
    for name,family in settings['evaluation_suite_names'].items():
        key = f'track/{name}/full_course_success'
        if key not in metrics:
            raise ValueError(f'Selection course was not evaluated: {name}')
        if not np.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
            raise ValueError(f'Invalid selection success for {name}')
        families.setdefault(family,[]).append(float(metrics[key]))
    metrics['selection_suite_success'] = sum(settings['evaluation_suite_family_weights'][f]*float(np.mean(v)) for f,v in families.items())
    metrics['selection_score'] = metrics['selection_suite_success']
    for family,values in families.items():
        metrics[f'selection_family/{family}/full_course_success'] = float(np.mean(values))
    if settings.get('ppo_reference_aware_gate_events', False):
        for measure in ('timely_success', 'clean_success', 'clean_timely_success'):
            clean_families = {}
            for name, family in settings['evaluation_suite_names'].items():
                key = f'track/{name}/{measure}'
                if key not in metrics:
                    raise ValueError(f'Missing reference-aware evaluation for {name}')
                clean_families.setdefault(family, []).append(float(metrics[key]))
            key = f'selection_suite_{measure}'
            metrics[key] = sum(settings['evaluation_suite_family_weights'][f]
                * float(np.mean(values)) for f, values in clean_families.items())
            if settings.get('monitor') == key:
                metrics['selection_score'] = metrics[key]
    return metrics
