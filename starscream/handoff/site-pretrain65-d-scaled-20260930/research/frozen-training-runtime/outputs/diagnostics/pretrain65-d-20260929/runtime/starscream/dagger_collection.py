"""Collection-only controls. Admission, teacher labels and replay quotas stay separate.

The optional controls are disabled for beta=1 expert/bootstrap collections.
Gate events use the pre-step active gate and the shared reference-aware plane
classifier in the worker; only the environment can declare a gate passed.
"""
from __future__ import annotations

import math
import numpy as np

from starscream.dagger_quality import NOMINAL, CORRECTIVE, RECOVERY


def collection_config(settings):
    raw = settings.get('dagger_collection_control')
    if not raw:
        return None
    c = dict(raw)
    allowed = {'version', 'recovery_control', 'segment_gates', 'maximum_distance',
               'no_progress_seconds', 'recovery_seconds', 'recovery_gate_passes',
               'nominal_distance', 'intervention_distance', 'track_bounds',
               'plane_tolerance', 'coherent_learner', 'segment_seconds',
               'expert_episode_probability', 'stop_on_invalid_label'}
    if set(c) - allowed or c.get('version') != 1:
        raise ValueError('invalid collection control schema/version')
    if c.get('recovery_control') not in ('none', 'teacher', 'learner'):
        raise ValueError('invalid recovery control')
    for key in ('coherent_learner', 'stop_on_invalid_label'):
        if key in c and not isinstance(c[key], bool):
            raise ValueError(f'{key} must be boolean')
    seconds = c.get('segment_seconds')
    if seconds is not None and (not math.isfinite(float(seconds)) or float(seconds) <= 0):
        raise ValueError('segment_seconds must be positive and finite')
    probability = float(c.get('expert_episode_probability', 0))
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('invalid expert episode probability')
    if probability and (not c.get('coherent_learner') or c['recovery_control'] != 'none'):
        raise ValueError('expert episodes require coherent learner and no recovery takeover')
    if not c.get('stop_on_invalid_label', True) and c['recovery_control'] == 'teacher':
        raise ValueError('teacher recovery requires uninterrupted valid labels')
    for key in ('maximum_distance', 'no_progress_seconds', 'recovery_seconds',
                'nominal_distance', 'intervention_distance', 'plane_tolerance'):
        if not math.isfinite(float(c[key])) or float(c[key]) <= 0:
            raise ValueError(f'invalid collection {key}')
    if not c['nominal_distance'] < c['intervention_distance'] < c['maximum_distance']:
        raise ValueError('collection distance bounds must be ordered')
    if c['recovery_gate_passes'] != int(c['recovery_gate_passes']) or c['recovery_gate_passes'] < 1:
        raise ValueError('invalid recovery gate count')
    segment = c.get('segment_gates')
    if segment is not None and (len(segment) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in segment)
                                or not 1 <= segment[0] <= segment[1]):
        raise ValueError('segment_gates must be an inclusive positive integer range')
    for bounds in c.get('track_bounds', {}).values():
        if set(bounds) - {'nominal_distance', 'intervention_distance', 'maximum_distance',
                          'no_progress_seconds', 'recovery_seconds'}:
            raise ValueError('invalid per-track collection override')
        collection_config({**settings, 'dagger_collection_control': {**c, 'track_bounds': {}, **bounds}})
    if settings.get('collector_backend', 'process') != 'process':
        raise ValueError('collection controls require process collection')
    if (settings.get('dagger_trajectory_quality', {}).get('enabled')
            or settings.get('dagger_require_successful_episodes')
            or settings.get('midtrain_quality') or settings.get('aggrevate', {}).get('enabled')
            or settings.get('dagger_recovery_anneal') or settings.get('dagger_pure_expert_collection')):
        raise ValueError('collection controls require ordinary valid-row DAgger admission')
    return c


def track_collection_config(config, track):
    from pathlib import Path
    bounds = config.get('track_bounds', {})
    if not bounds:
        return config
    key = str(Path(track).resolve())
    if key not in bounds:
        raise ValueError(f'missing collection bounds for {key}')
    return {**config, **bounds[key]}


def collection_beta(beta, config):
    # Keep the exact expert/DART bootstrap; coherent learner segments start
    # only when the ordinary schedule leaves expert-only collection.
    coherent = config and (config.get('segment_gates') or config.get('coherent_learner'))
    return 0. if coherent and beta < 1. else beta


class CollectionControl:
    def __init__(self, config, dt, seed, *, enabled):
        self.config, self.dt, self.seed, self.enabled = config, dt, seed, enabled
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('invalid collection timestep')
        self.steps = self.last_pass = self.recovery_steps = self.passes = self.segment = 0
        self.total_steps = 0
        self.learner_passes = 0
        self.recovering = False
        self.trigger = 0
        self.expert_episode = bool(np.random.default_rng(seed ^ 0xE91A).random()
                                   < config.get('expert_episode_probability', 0))
        self.limit = (int(np.random.default_rng(seed ^ 0x5E6A).integers(*(
            config['segment_gates'][0], config['segment_gates'][1] + 1)))
            if config.get('segment_gates') else None)

    def _start(self, trigger):
        self.recovering = True
        self.trigger = trigger
        self.segment += 1
        self.recovery_steps = self.passes = 0

    def before(self, distance, teacher_recovery=False, *, prefix=False):
        self.distance = float(distance)
        self.teacher_recovery = bool(teacher_recovery)
        self.active = self.enabled and not prefix
        self.invalid = self.active and self.config['recovery_control'] != 'none' and (
            not math.isfinite(self.distance) or self.distance > self.config['maximum_distance'])
        mode = self.config['recovery_control']
        # A reference controller's recovery signal alone is not a geometric
        # miss. It may occur on an intended wrong-side reference approach.
        # B deliberately permits that teacher intervention; C/E require a
        # genuine observed miss or a stalled gate encounter.
        trigger = 0
        if self.active and mode != 'none':
            if (self.steps - self.last_pass) * self.dt >= self.config['no_progress_seconds']:
                trigger |= 4
            if mode == 'teacher':
                trigger |= 2 if self.distance > self.config['intervention_distance'] else 0
                trigger |= 8 if teacher_recovery else 0
            if trigger and not self.recovering and not self.invalid:
                self._start(trigger)
        self.row_class = (RECOVERY if self.recovering else CORRECTIVE
                          if self.distance > self.config['nominal_distance'] else NOMINAL)
        if self.active and self.expert_episode:
            return True, False
        return (self.active and self.recovering and mode == 'teacher',
                self.active and (self.recovering and mode == 'learner'
                                 or self.config.get('coherent_learner', False)))

    def after(self, *, pre_side, post_side, passed, gate_occurrence, teacher,
              dart, valid, crashed, done, finished, planned_crossing=False):
        self.total_steps += 1
        miss = bool(not passed and pre_side < 0 <= post_side and not planned_crossing)
        backward = bool(not passed and pre_side >= 0 > post_side)
        was_recovering, row_segment, row_trigger = self.recovering, self.segment, self.trigger
        events = {'active_steps': int(self.active), 'teacher_steps': int(teacher),
                  'learner_steps': int(not teacher), 'prefix_steps': int(not self.active and self.enabled),
                  'recovery_steps': int(self.active and was_recovering),
                  'active_teacher_steps': int(self.active and teacher),
                  'learner_recovery_steps': int(self.active and was_recovering and not teacher)}
        qualification, stop = 0, False
        if self.active:
            self.steps += 1
            self.learner_passes += int(passed)
            stop = bool(crashed or (self.config['recovery_control'] != 'none'
                                   and (self.invalid or (not valid and self.config.get('stop_on_invalid_label', True)))))
            if self.invalid:
                events['invalid_state_stops'] = 1
            if was_recovering:
                self.recovery_steps += 1
                self.passes += int(passed)
                mode = self.config['recovery_control']
                if mode == 'teacher' and (not teacher or dart):
                    stop = True
                complete = bool(passed and (mode == 'learner' or (
                    self.distance <= self.config['intervention_distance'] and
                    (finished or self.passes >= self.config['recovery_gate_passes']))))
                within = self.recovery_steps * self.dt <= self.config['recovery_seconds'] + 1e-9
                if complete and within and not stop:
                    qualification = 1
                    self.recovering = False
                    self.trigger = 0
                    events['recovery_completed'] = 1
                elif stop or done or self.recovery_steps * self.dt >= self.config['recovery_seconds'] - 1e-9:
                    qualification, stop = -1, True
                    events['recovery_failed'] = 1
                    if not within or self.recovery_steps * self.dt >= self.config['recovery_seconds'] - 1e-9:
                        events['recovery_timeout'] = 1
            if passed:
                self.last_pass = self.steps
            if miss and not self.recovering and not stop and not done and self.config['recovery_control'] != 'none':
                self._start(1)
            # Only actual accepted passes after the prefix count. Segment
            # completion is a truncation, never a full-course success label.
            if self.limit and self.learner_passes >= self.limit and not stop and not done:
                stop = True
                events['segment_completed'] = 1
            if (self.config.get('segment_seconds') is not None and not stop and not done
                    and self.steps * self.dt >= self.config['segment_seconds'] - 1e-9):
                stop = True
                events['segment_time_limit'] = 1
        row = np.asarray([self.row_class, miss, backward,
            (self.steps-self.last_pass)*self.dt, self.distance if math.isfinite(self.distance) else -1., teacher, dart,
            self.total_steps, gate_occurrence, row_segment, qualification, stop,
            self.seed, pre_side, post_side, row_trigger, self.teacher_recovery,
            planned_crossing], dtype=np.float64)
        return row, stop, events
