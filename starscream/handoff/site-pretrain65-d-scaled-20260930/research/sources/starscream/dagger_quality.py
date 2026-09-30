"""Trajectory-aware DAgger admission and bounded recovery exposure.

Quality is independent of control-event tags and reset provenance. Recovery is
qualified by uninterrupted, noise-free teacher execution in the actual plant.
The distance measurement is MPCC's ordered projection, not gate-center distance.
"""
from __future__ import annotations

import numpy as np

UNKNOWN, NOMINAL, CORRECTIVE, RECOVERY, PENDING = -1, 0, 1, 2, 3
FIELDS = ('quality', 'miss', 'backward', 'gate_dwell_seconds', 'projection_distance',
          'executed_teacher', 'dart', 'step', 'gate_occurrence', 'segment',
          'qualification', 'stopped', 'episode', 'pre_gate_side', 'post_gate_side', 'trigger',
          'teacher_missed_gate_recovery', 'planned_plane_crossing')
WIDTH = len(FIELDS)
PRECURSOR = 16  # trigger bit; telemetry width and old replay remain unchanged


def mark_precursors(rows, indices, config, control_hz, terminal_row=None):
    """Tag qualified pre-action states before a miss/takeover, even on failure.

    Called on one completed slot, before replay retention. Never turns failed
    recovery, unknown labels, or a previous gate's states into corrective data.
    The bit is sampling metadata only, never an actor input or action target.
    """
    horizon = float(config.get('precursor_seconds', 0)) * control_hz
    if horizon <= 0 or not indices:
        return 0
    data = np.asarray([rows[i] for i in indices])
    starts = np.r_[True, np.diff(data[:, 9]) != 0]
    events = list(data[(data[:, 1] > 0) | (starts & (data[:, 15] > 0))])
    # Invalid final teacher queries have telemetry but no stored action label.
    # Their miss event must still preserve useful earlier supervision.
    if terminal_row is not None and (terminal_row[1] > 0 or terminal_row[15] > 0):
        events.append(terminal_row)
    marked = np.zeros(len(data), bool)
    for event in events:
        marked |= ((data[:, 7] >= event[7] - horizon)
                   & (data[:, 7] <= event[7])
                   & (data[:, 8] == event[8])
                   & (data[:, 12] == event[12])
                   & np.isin(data[:, 0], [NOMINAL, CORRECTIVE]))
    for local in np.flatnonzero(marked):
        row = rows[indices[local]]
        row[0] = CORRECTIVE
        row[15] = int(row[15]) | PRECURSOR
    return int(marked.sum())


def online_minibatch_quality_config(config, override=None):
    """Change fitting proportions without changing admission or retained replay."""
    result = dict(config)
    if override is None:
        return result
    keys = {'nominal_fraction', 'corrective_fraction', 'recovery_fraction'}
    if set(override) != keys:
        raise ValueError('online minibatch override requires exactly three class fractions')
    values = np.asarray([override[k] for k in sorted(keys)], dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.isclose(values.sum(), 1)
            or not 0 < float(override['recovery_fraction']) < .5):
        raise ValueError('invalid online minibatch quality fractions')
    result.update({k: float(override[k]) for k in keys})
    return result


def replay_quality_config(config, telemetry, *, permanent=False, online_override=None):
    """Bound sparse permanent corrective reuse relative to its actual support."""
    result = dict(config) if permanent else online_minibatch_quality_config(config, online_override)
    if permanent and 'permanent_corrective_max_multiplier' in config:
        prevalence = float(np.mean(np.asarray(telemetry)[:, 0] == CORRECTIVE))
        fraction = min(config.get('permanent_corrective_max_fraction', .1),
                       prevalence * config['permanent_corrective_max_multiplier'])
        result.update(nominal_fraction=1-fraction, corrective_fraction=fraction,
                      recovery_fraction=0., precursor_fraction=0.)
    return result


def quality_config(settings):
    raw = settings.get('dagger_trajectory_quality', {})
    if not raw or not raw.get('enabled', False):
        return None
    config = dict(raw)
    if config.get('implementation_version', 2) not in (2, 3):
        raise ValueError('unsupported trajectory quality implementation version')
    config.setdefault('implementation_version', 2)
    for key, default, upper in (
        ('precursor_seconds', 0., 5.), ('precursor_fraction', 0., 1.),
        ('permanent_corrective_max_multiplier', 1., 10.),
        ('permanent_corrective_max_fraction', .1, 1.),
    ):
        value = float(config.get(key, default))
        if not np.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f'invalid trajectory quality {key}')
    if config.get('precursor_fraction', 0) and not config.get('precursor_seconds', 0):
        raise ValueError('precursor quota requires a positive precursor horizon')
    if config.get('calibration_source'):
        import hashlib
        import json
        from pathlib import Path
        payload = Path(config['calibration_source']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != config.get('calibration_sha256'):
            raise ValueError('trajectory calibration fingerprint mismatch')
        calibration = json.loads(payload)
        if calibration.get('missing') or calibration.get('minimum_clean_episodes', 0) < 2:
            raise ValueError('trajectory calibration lacks clean training references')
        if config.get('track_bounds') != calibration.get('track_bounds'):
            raise ValueError('trajectory bounds differ from frozen calibration')
    required = ('nominal_distance', 'intervention_distance', 'maximum_distance',
                'maximum_gate_seconds', 'recovery_seconds', 'recovery_gate_passes',
                'nominal_fraction', 'corrective_fraction', 'recovery_fraction')
    if any(k not in config for k in required):
        raise ValueError('trajectory quality requires explicit calibrated bounds and quotas')
    values = np.asarray([config[k] for k in required], float)
    if not np.isfinite(values).all():
        raise ValueError('trajectory quality bounds must be finite')
    if not 0 < config['nominal_distance'] < config['intervention_distance'] < config['maximum_distance']:
        raise ValueError('trajectory distance bounds must be positive and ordered')
    if min(config['maximum_gate_seconds'], config['recovery_seconds']) <= 0:
        raise ValueError('trajectory time bounds must be positive')
    if int(config['recovery_gate_passes']) != config['recovery_gate_passes'] or config['recovery_gate_passes'] < 1:
        raise ValueError('recovery_gate_passes must be a positive integer')
    fractions = values[-3:]
    if (fractions < 0).any() or not np.isclose(fractions.sum(), 1) or not 0 < fractions[2] < .5:
        raise ValueError('quality fractions must sum to one with a positive minority recovery quota')
    if settings.get('collector_backend', 'process') != 'process':
        raise ValueError('trajectory quality currently requires the process collector')
    if settings.get('dagger_require_successful_episodes', False):
        raise ValueError('trajectory quality admits segments; disable whole-episode admission')
    if settings.get('midtrain_quality') or settings.get('aggrevate', {}).get('enabled', False):
        raise ValueError('trajectory quality cannot be combined with episode/adaptive-query admission')
    if settings.get('action_chunk_weight', 0) or settings.get('dagger_pure_expert_collection', False):
        raise ValueError('trajectory quality does not support action chunks or pure-expert abort mode')
    if not settings.get('dagger_hierarchical_replay_sampling', False):
        raise ValueError('trajectory quality requires hierarchical replay sampling')
    if settings.get('dagger_recovery_anneal'):
        raise ValueError('trajectory quality cannot use legacy recovery annealing')
    if not np.isclose(float(settings.get('online_fraction', 0)) + float(settings.get('dagger_permanent_expert_fraction', 0)), 1):
        raise ValueError('trajectory quality requires fully qualified online/permanent replay (no legacy offline pool)')
    return config


class TrajectoryGuard:
    def __init__(self, config, dt, episode):
        self.config, self.dt, self.episode = config, dt, episode
        self.steps = self.last_pass = self.segment = self.recovery_steps = self.passes = 0
        self.recovering = False
        self.trigger = 0

    def before(self, distance, teacher_recovery=False):
        """Classify s[t], before selecting a[t]; starts takeover before stepping."""
        c = self.config
        self.distance = float(distance)
        self.teacher_recovery = bool(teacher_recovery)
        self.invalid = not np.isfinite(distance) or distance > c['maximum_distance']
        dwell = (self.steps - self.last_pass) * self.dt
        trigger = ((2 if distance > c['intervention_distance'] else 0)
                   | (4 if dwell >= c['maximum_gate_seconds'] else 0)
                   | (8 if teacher_recovery else 0))
        if c.get('audit_only', False):
            self.row_class = RECOVERY if self.recovering or teacher_recovery else CORRECTIVE if distance > c['nominal_distance'] else NOMINAL
            self.trigger = trigger
            return False
        if trigger and not self.recovering and not self.invalid:
            self._start(trigger)
        self.row_class = UNKNOWN if self.invalid else PENDING if self.recovering else (
            CORRECTIVE if distance > c['nominal_distance'] else NOMINAL)
        return self.recovering

    def _start(self, trigger):
        self.recovering = True
        self.recovery_steps = self.passes = 0
        self.segment += 1
        self.trigger = trigger

    def after(self, *, pre_side, post_side, passed, gate_occurrence, teacher,
              dart, valid, crashed, done, finished, planned_crossing=False):
        self.steps += 1
        miss = bool(not passed and pre_side < 0 <= post_side and not planned_crossing)
        backward = bool(not passed and pre_side >= 0 > post_side)
        dwell = (self.steps - self.last_pass) * self.dt
        qualification, stop = 0, bool(self.invalid or crashed or not valid)
        row_segment = self.segment
        row_trigger = self.trigger if self.recovering else 0
        if self.config.get('audit_only', False):
            if passed:
                self.last_pass = self.steps
                self.recovering = False
            elif miss:
                self.recovering = True
            return np.asarray([self.row_class, miss, backward, dwell, self.distance,
                teacher, dart, self.steps, gate_occurrence, self.segment, 0, 0,
                self.episode, pre_side, post_side, self.trigger, self.teacher_recovery,
                planned_crossing], np.float64), False
        if self.recovering:
            self.recovery_steps += 1
            self.passes += int(passed)
            # Qualification requires a valid uninterrupted teacher continuation,
            # not just the mixed-policy episode eventually completing.
            stop |= not teacher or dart
            within_budget = self.recovery_steps * self.dt <= self.config['recovery_seconds'] + 1e-9
            complete = (passed and self.distance <= self.config['intervention_distance']
                        and (finished or self.passes >= self.config['recovery_gate_passes']))
            if not stop and within_budget and complete:
                qualification = 1
                self.recovering = False
                self.trigger = 0
            elif stop or done or not within_budget or self.recovery_steps * self.dt >= self.config['recovery_seconds']:
                qualification, stop = -1, True
        if passed:
            self.last_pass = self.steps
        if miss and not self.recovering and not stop and not done:
            # The missed transition's pre-state keeps its own classification.
            # The next state begins the explicit recovery segment.
            self._start(1)
        if not valid or (crashed and teacher):
            self.row_class = UNKNOWN
        row = np.asarray([self.row_class, miss, backward, dwell, self.distance,
                          teacher, dart, self.steps, gate_occurrence, row_segment,
                          qualification, stop, self.episode, pre_side, post_side,
                          row_trigger, self.teacher_recovery, planned_crossing], np.float64)
        return row, stop


def planned_plane_crossing(line, gate, gate_index, gate_count, position, hint, tolerance):
    """Recognize a local, intended crossing of the infinite plane outside a gate.

    Only an ordered incoming reference segment can exempt a crossing. The nearby
    reference must itself cross forward outside the aperture, and the vehicle
    must be near that intersection. A mere wrong-side position is insufficient.
    This is observational: never mutates the controller's progress/warm start.
    """
    projection = line.project(np.asarray(position), hint_progress=hint, search_radius=8.)
    if projection.gate_index != (gate_index-1) % gate_count or projection.distance > tolerance:
        return False
    radius = max(.5, 2*tolerance)
    points = np.asarray([line.evaluate(s)['position'] for s in
                         np.linspace(projection.progress-radius, projection.progress+radius, 17)])
    local = (points - gate.position) @ gate.directed_rotation
    for a, b, pa, pb in zip(local[:-1], local[1:], points[:-1], points[1:]):
        if a[0] < 0 <= b[0]:
            fraction = -a[0]/(b[0]-a[0])
            hit = a + fraction*(b-a)
            outside = abs(hit[1]) > gate.size[0]/2 or abs(hit[2]) > gate.size[1]/2
            if outside and np.linalg.norm(np.asarray(position)-(pa+fraction*(pb-pa))) <= tolerance:
                return True
    return False


class ReferenceGateEvents:
    """Shared evaluation/RL event classifier; never changes plant gate acceptance."""
    def __init__(self, line, gate_count, position, tolerance=.75):
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('gate event reference tolerance must be positive')
        self.line, self.gate_count, self.tolerance = line, gate_count, tolerance
        self.progress = line.project(np.asarray(position)).progress
        self.raw_crossings = self.planned_crossings = self.misses = 0

    def advance(self, previous, current, gate, gate_index, passed):
        self.progress = self.line.project(np.asarray(current), hint_progress=self.progress,
                                          search_radius=8.).progress
        raw = bool(not passed and (np.asarray(previous)-gate.position) @ gate.normal < 0
                   <= (np.asarray(current)-gate.position) @ gate.normal)
        planned = bool(raw and planned_plane_crossing(self.line, gate, gate_index,
            self.gate_count, current, self.progress, self.tolerance))
        self.raw_crossings += int(raw)
        self.planned_crossings += int(planned)
        self.misses += int(raw and not planned)
        return raw and not planned


def track_quality_config(config, track):
    """Frozen training-only per-course calibration; identity stays out of policy."""
    from pathlib import Path
    if config is None:
        return None
    bounds = config.get('track_bounds', {})
    if not bounds:
        return config
    key = str(Path(track).resolve())
    if key not in bounds:
        raise ValueError(f'missing trajectory bounds for training course {key}')
    allowed = {'nominal_distance', 'intervention_distance', 'maximum_distance',
               'maximum_gate_seconds', 'recovery_seconds'}
    if set(bounds[key]) - allowed:
        raise ValueError('per-course quality override may only change geometric/time bounds')
    result = {**config, **bounds[key]}
    vals = [result[k] for k in allowed]
    if not np.isfinite(vals).all() or min(vals) <= 0 or not (
            result['nominal_distance'] < result['intervention_distance'] < result['maximum_distance']):
        raise ValueError('invalid per-course trajectory bounds')
    return result


def _quotas(size, weights):
    weights = np.asarray(weights, float)
    raw = size * weights / weights.sum()
    counts = np.floor(raw).astype(int)
    counts[np.argsort(-(raw-counts), kind='stable')[:size-int(counts.sum())]] += 1
    return counts


class QualityReplayPlan:
    """Class-first quotas: recovery is capped globally, even with many tracks.

    Empty classes only fall back to available nominal/corrective states. Unknown
    rows are never eligible. Within each class balance family, track, then gate
    (or independent recovery segment) to avoid weighting long retries by dwell.
    """
    def __init__(self, families, tracks, gates, telemetry, size, *, config, family_weights=None):
        self.size = int(size)
        self.telemetry = np.asarray(telemetry)
        self.draws = []
        self.sample_counts = np.zeros(3, np.int64)
        self.sampled_batches = self.unique_rows_total = 0
        self.precursor_draws = 0
        self.precursor_shortfall = 0
        self.requested = np.asarray([config[k] for k in ('nominal_fraction', 'corrective_fraction', 'recovery_fraction')])
        if self.telemetry.shape != (len(families), WIDTH) or not all(len(a) == len(families) for a in (tracks, gates)):
            raise ValueError('quality replay metadata lost alignment')
        quality = self.telemetry[:, 0]
        if not np.isin(quality, [NOMINAL, CORRECTIVE, RECOVERY]).all():
            raise ValueError('quality replay contains unknown/unqualified rows')
        if size < 0 or not len(quality):
            raise ValueError('quality replay requires nonempty data and nonnegative size')
        # Floor recovery separately: per-track rounding must never inflate it.
        recovery_count = int(np.floor(size * self.requested[2]))
        counts = np.r_[_quotas(size-recovery_count, self.requested[:2]), recovery_count]
        available = np.asarray([np.any(quality == i) for i in range(3)])
        if not available[:2].any():
            raise ValueError('quality replay has no nominal/corrective coverage')
        self.shortfall = np.where(available, 0, counts)
        counts[~available] = 0
        fallback = 0 if available[0] else 1
        counts[fallback] += self.shortfall.sum()
        self.realized_counts = counts.copy()
        families, tracks, gates = map(np.asarray, (families, tracks, gates))
        self.trees = []
        strata = [(cls, int(count), np.flatnonzero(quality == cls)) for cls, count in enumerate(counts)]
        precursor_share = float(config.get('precursor_fraction', 0.))
        if precursor_share:
            _, count, pool = strata[CORRECTIVE]
            mask = (self.telemetry[pool, 15].astype(np.int64) & PRECURSOR) != 0
            wanted = int(np.floor(count * precursor_share))
            if mask.any() and (~mask).any():
                strata[CORRECTIVE] = (CORRECTIVE, count-wanted, pool[~mask])
                strata.append((CORRECTIVE, wanted, pool[mask]))
            elif not mask.any():
                self.precursor_shortfall = wanted
        for cls, count, pool in strata:
            if not count:
                continue
            fs = np.unique(families[pool])
            fw = np.asarray([float((family_weights or {}).get(int(f), 1.)) for f in fs])
            if not np.isfinite(fw).all() or min(fw) <= 0:
                raise ValueError('quality family weights must be finite and positive')
            tree = []
            for f in fs:
                fp = pool[families[pool] == f]
                track_groups = []
                for t in np.unique(tracks[fp]):
                    tp = fp[tracks[fp] == t]
                    if cls == RECOVERY:
                        keys = self.telemetry[tp][:, [12, 9]]
                    elif cls == CORRECTIVE and config.get('balance_corrective_encounters', False):
                        keys = self.telemetry[tp][:, [12, 8]]
                    else:
                        keys = gates[tp, None]
                    _, groups = np.unique(keys, axis=0, return_inverse=True)
                    track_groups.append([tp[groups == g] for g in np.unique(groups)])
                tree.append(track_groups)
            self.trees.append((int(count), fw, tree))

    @property
    def index_bytes(self):
        return sum(pool.nbytes for _, _, families in self.trees
                   for tracks in families for groups in tracks for pool in groups)

    @staticmethod
    def _random_quotas(size, weights, rng):
        # Systematic rounding gives every nonempty group positive opportunity.
        # Fixed largest-remainder ties would permanently starve later gates when
        # the group count exceeds the small per-class minibatch quota.
        weights = np.asarray(weights, float)
        cumulative = np.cumsum(weights / weights.sum()) * size
        cumulative[-1] = size
        return np.bincount(np.searchsorted(cumulative, np.arange(size)+rng.random(), side='right'),
                           minlength=len(weights))

    def sample(self, rng):
        selected = []
        for count, weights, families in self.trees:
            for family, n in zip(families, self._random_quotas(count, weights, rng)):
                if not n:
                    continue
                for groups, tn in zip(family, self._random_quotas(n, np.ones(len(family)), rng)):
                    if not tn:
                        continue
                    for pool, gn in zip(groups, self._random_quotas(tn, np.ones(len(groups)), rng)):
                        if gn:
                            selected.append(rng.choice(pool, size=int(gn), replace=True))
        rows = np.concatenate(selected) if selected else np.empty(0, np.int64)
        if len(rows) != self.size:
            raise RuntimeError('quality replay quota mismatch')
        rng.shuffle(rows)
        self.sample_counts += np.bincount(self.telemetry[rows, 0].astype(int), minlength=3)
        self.precursor_draws += int(np.sum((self.telemetry[rows, 15].astype(np.int64) & PRECURSOR) != 0))
        self.sampled_batches += 1
        self.unique_rows_total += len(np.unique(rows))
        return rows


def quality_subsample(rng, tracks, telemetry, maximum, config):
    """Unique retention with a hard recovery cap, no unqualified fallback."""
    telemetry = np.asarray(telemetry)
    size = len(telemetry) if maximum <= 0 else min(maximum, len(telemetry))
    eligible = np.flatnonzero(np.isin(telemetry[:, 0], [0, 1, 2]))
    pools = [eligible[telemetry[eligible, 0] == cls] for cls in range(3)]
    rmax = int(np.floor(size * config['recovery_fraction']))
    nr = min(rmax, len(pools[2]))
    targets = list(_quotas(size-nr, [config['nominal_fraction'], config['corrective_fraction']])) + [nr]
    targets = np.minimum(targets, [len(p) for p in pools])
    missing = size-int(targets.sum())
    for cls in (0, 1):
        add = min(missing, len(pools[cls])-targets[cls])
        targets[cls] += add
        missing -= add
    # A severe non-recovery shortage must not turn the requested 5% into 100%.
    non_recovery = int(targets[:2].sum())
    targets[2] = min(targets[2], int(np.floor(non_recovery * config['recovery_fraction'] / (1-config['recovery_fraction']))))
    selected = []
    strata = [(cls, pool, int(n)) for cls, (pool, n) in enumerate(zip(pools, targets))]
    if config.get('precursor_fraction', 0):
        cls, pool, n = strata[CORRECTIVE]
        mask = (telemetry[pool, 15].astype(np.int64) & PRECURSOR) != 0
        wanted = min(int(np.floor(n * config['precursor_fraction'])), int(mask.sum()))
        other = min(n-wanted, int((~mask).sum()))
        wanted = min(n-other, int(mask.sum()))
        strata[CORRECTIVE] = (cls, pool[~mask], other)
        strata.append((cls, pool[mask], wanted))
    for cls, pool, n in strata:
        if not n:
            continue
        if cls == RECOVERY:
            keys = np.c_[np.asarray(tracks)[pool], telemetry[pool, 12], telemetry[pool, 9]]
        elif cls == CORRECTIVE and config.get('balance_corrective_encounters', False):
            keys = np.c_[np.asarray(tracks)[pool], telemetry[pool, 12], telemetry[pool, 8]]
        else:
            keys = np.asarray(tracks)[pool, None]
        _, group = np.unique(keys, axis=0, return_inverse=True)
        # Shuffle within groups; select balanced ranks without replacement.
        order = rng.permutation(len(pool))
        order = order[np.argsort(group[order], kind='stable')]
        starts = np.r_[0, np.flatnonzero(np.diff(group[order]))+1]
        ranks = np.arange(len(pool)) - np.repeat(starts, np.diff(np.r_[starts, len(pool)]))
        tie = rng.random(len(pool))
        selected.append(pool[order[np.lexsort((tie, ranks))[:n]]])
    result = np.concatenate(selected) if selected else np.empty(0, np.int64)
    rng.shuffle(result)
    return result
