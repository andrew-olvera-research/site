"""Per-round learning-rate schedule for DAgger pretraining (v6.21).

The v6.20.1 r240 run trained at a constant 1e-4 and kept improving slowly in its last
100 rounds; a warmup-then-cosine schedule consolidates that tail. The factor multiplies
each optimizer group's initial learning rate at the start of every round.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


def dart_collection_round(round_index: int, settings: Mapping[str, Any]) -> bool:
    """Periodic teacher-only noisy collection refreshes online support, not anchors."""
    every = settings.get('dagger_dart_refresh_interval', 0)
    if not isinstance(every, int) or every < 0:
        raise ValueError('DART refresh interval must be a nonnegative integer')
    return (round_index <= int(settings.get('dagger_dart_expert_rounds', 0))
            or bool(every and round_index % every == 0))


def weighted_speed_fractions(fractions, weights, count, *, rng=None):
    """Course-local quotas, optionally unbiased across small episode budgets.

    A seeded systematic draw has less than one episode of rounding error per
    command and preserves the requested proportions in expectation. Unlike the
    legacy coverage floor, it need not include every command in every round.
    """
    if count < 0:
        raise ValueError('negative episode count')
    normalized={float(k):float(v) for k,v in weights.items()}
    if set(normalized)!=set(fractions) or any(not math.isfinite(v) or v<=0 for v in normalized.values()):
        raise ValueError('speed fraction weights must cover the configured fractions and be positive')
    total=sum(normalized.values())
    if rng is not None:
        offset = float(rng.random())
        cumulative = 0.0
        previous = 0
        result = []
        for index, fraction in enumerate(fractions):
            cumulative += count * normalized[fraction] / total
            boundary = count if index == len(fractions) - 1 else math.floor(cumulative + offset)
            result.extend([fraction] * (boundary - previous))
            previous = boundary
        return result
    raw=[count*normalized[f]/total for f in fractions]
    quotas=[math.floor(v) for v in raw]
    order=sorted(range(len(fractions)),key=lambda i:raw[i]-quotas[i],reverse=True)
    for i in order[:count-sum(quotas)]: quotas[i]+=1
    if count>=len(fractions):
        for i in range(len(fractions)):
            if quotas[i]==0:
                donor=max(range(len(fractions)),key=lambda j:quotas[j])
                quotas[donor]-=1; quotas[i]+=1
    return [f for f,n in zip(fractions,quotas) for _ in range(n)]


def speed_fraction_weights(round_index: int, settings: Mapping[str, Any]) -> dict[str, float]:
    """Interpolate command quotas from contrast-rich to frontier-heavy data."""
    base = settings.get('dagger_teacher_speed_fraction_weights', {})
    schedule = settings.get('dagger_teacher_speed_weight_schedule')
    if not schedule:
        return {str(k): float(v) for k, v in base.items()}
    canonical = lambda value: str(float(value)).rstrip('0').rstrip('.')
    start = {canonical(k): float(v) for k, v in schedule['start_weights'].items()}
    end = {canonical(k): float(v) for k, v in schedule['end_weights'].items()}
    fractions = {canonical(v) for v in settings['dagger_teacher_speed_fractions']}
    if set(start) != fractions or set(end) != fractions:
        raise ValueError('speed weight schedule must cover every configured fraction')
    end_round = int(schedule.get('end_round', settings.get('rounds', 1)))
    values = (*start.values(), *end.values())
    if round_index < 1 or end_round < 2 or any(v <= 0 or not math.isfinite(v) for v in values):
        raise ValueError('invalid speed weight schedule')
    progress = min((round_index - 1) / (end_round - 1), 1.)
    kind = str(schedule.get('type', 'cosine'))
    if kind == 'cosine':
        progress = .5 * (1. - math.cos(math.pi * progress))
    elif kind != 'linear':
        raise ValueError('speed weight schedule type must be cosine or linear')
    return {key: start[key] + (end[key] - start[key]) * progress for key in start}


def recovery_anneal(round_index: int, settings: Mapping[str, Any]) -> dict[str, float]:
    """Anneal learner occupancy toward a configurable late teacher floor.

    Pure-phase updates use only that round's expert trajectories, so old learner
    states cannot leak through replay or be resurrected by checkpoint resume.
    """
    schedule = settings.get('dagger_recovery_anneal')
    if not schedule or not schedule.get('enabled', True):
        return {}
    end = int(schedule['pure_expert_round'])
    start_beta = float(schedule.get('initial_teacher_beta', .35))
    late_beta = float(schedule.get('late_teacher_beta', 1.0))
    recovery_floor = float(schedule.get('recovery_fraction_floor', 0.0))
    pure_expert_phase = bool(schedule.get('pure_expert_phase', True))
    warmup = int(schedule.get('teacher_warmup_rounds', 0))
    if (round_index < 1 or end < 2 or not 0 <= start_beta <= 1
            or not 0 <= late_beta <= 1 or not 0 <= recovery_floor <= .25):
        raise ValueError('invalid recovery annealing schedule')
    if warmup < 0 or warmup == 1 or warmup >= end:
        raise ValueError('teacher warmup must be zero or at least two rounds before the pure phase')
    if (float(settings.get('online_fraction', 0)) != 1.0
            or float(settings.get('dagger_permanent_expert_fraction', 0)) != 0.0
            or int(settings.get('dagger_permanent_expert_rounds', 0)) != 0
            or int(settings.get('dagger_dart_expert_rounds', 0)) != 0):
        raise ValueError('recovery anneal requires fresh online-only replay and no DART/permanent pool')
    if (settings.get('collector_backend','process') != 'process'
            or not settings.get('dagger_async_collection',True)
            or not settings.get('dagger_lazy_teacher_actions',True)):
        raise ValueError('recovery anneal requires asynchronous lazy process collection')
    anneal_start = max(warmup, 1)
    progress = min(max((round_index - anneal_start)/(end - anneal_start), 0.), 1.)
    remaining = .5 * (1. + math.cos(math.pi * progress))
    recovery = recovery_floor + (.25 - recovery_floor) * remaining
    teacher_beta = late_beta - (late_beta - start_beta) * remaining
    if warmup and round_index <= warmup:
        teacher_beta = 1. + (start_beta - 1.) * (round_index - 1)/(warmup - 1)
    return dict(teacher_beta=teacher_beta,
                dagger_replay_nominal_fraction=.65-recovery,
                dagger_replay_critical_fraction=.35,
                dagger_replay_recovery_fraction=recovery,
                recovery_anneal_remaining=remaining,
                pure_expert=float(pure_expert_phase and round_index >= end))


def round_learning_rate_factor(round_index: int, rounds: int, schedule: Mapping[str, Any] | None) -> float:
    """Multiplier for round ``round_index`` (1-based) of ``rounds`` under ``schedule``."""
    if not schedule or not bool(schedule.get('enabled', True)) or str(schedule.get('type', 'constant')) == 'constant':
        return 1.0
    kind = str(schedule['type'])
    # A continuation may extend the run without stretching a completed schedule.
    rounds = int(schedule.get('horizon_rounds', rounds))
    warmup = int(schedule.get('warmup_rounds', 0))
    final = float(schedule.get('final_fraction', 0.1))
    start = float(schedule.get('warmup_start_fraction', final))
    if rounds < 1 or round_index < 1 or warmup < 0 or not 0.0 < final <= 1.0 or not 0.0 < start <= 1.0:
        raise ValueError('invalid learning-rate schedule settings')
    if warmup and round_index <= warmup:
        return start + (1.0 - start) * (round_index / warmup)
    if kind == 'cosine':
        span = max(rounds - warmup, 1)
        progress = min(max((round_index - warmup) / span, 0.0), 1.0)
        return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if kind == 'linear':
        span = max(rounds - warmup, 1)
        progress = min(max((round_index - warmup) / span, 0.0), 1.0)
        return 1.0 + (final - 1.0) * progress
    raise ValueError(f'unknown learning-rate schedule type {kind!r}')


def apply_round_learning_rate(optimizer, round_index: int, rounds: int, schedule: Mapping[str, Any] | None) -> float:
    """Scale every param group's initial lr by the round factor; returns the factor."""
    factor = round_learning_rate_factor(round_index, rounds, schedule)
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', float(group['lr']))
        group['lr'] = float(group['initial_lr']) * factor
    return factor
