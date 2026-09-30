"""Undiscounted lap-time PPO with empirical completion constraints.

Successful return is -T/Tref. Failed return is -(timeout/Tref+1+dual),
independent of failure time. This prevents profitable early termination.
Duals are frozen for an entire episode, including across rollout windows.
Validation guards are empirical, not probabilistic safety guarantees.
"""
from pathlib import Path
import math
import numpy as np


def succeeded(row):
    return not row['crashed'] and row['gates'] >= row['target_gates']


def transition_reward(reference, hz, max_steps, *, done=False, success=False,
                      steps=0, dual=0.):
    if reference <= 0 or hz <= 0 or max_steps <= 0 or dual < 0:
        raise ValueError('invalid lap-time reward parameters')
    reward = -1. / (hz * reference)
    if done and not success:
        reward += steps / (hz * reference) - (max_steps / (hz * reference) + 1. + dual)
    return reward


class LapTimeConstraints:
    def __init__(self, settings):
        self.config = settings['ppo_lap_time']
        if (float(settings['discount']) != 1. or not settings['ppo_timeout_is_terminal']
                or settings.get('ppo_exploration_correlation', 0) != 0
                or settings.get('ppo_saturation_mask_threshold') is not None):
            raise ValueError('lap-time objective requires gamma=1, terminal timeouts, IID/full likelihood')
        self.hz = float(settings['control_hz'])
        self.courses = self.config['training_courses']
        if not self.courses or any(not math.isfinite(row['reference_seconds']) or row['reference_seconds'] <= 0
                or not 0 <= row['success_rate'] <= 1 for row in self.courses.values()):
            raise ValueError('invalid frozen source calibration')
        if (settings.get('ppo_adaptive_level_replay', {}).get('enabled', False)
                or settings.get('ppo_online_course_bank', {}).get('enabled', False)):
            raise ValueError('lap-time constraints require a frozen course/lane roster')
        self.duals = {name: float(self.config.get('initial_dual', 1.)) for name in self.courses}
        self.ema = {name: float(row['success_rate']) for name, row in self.courses.items()}
        self.pending = {name: [] for name in self.courses}

    def state_dict(self):
        return dict(duals=self.duals.copy(), ema=self.ema.copy(), pending=self.pending.copy())

    def load_state_dict(self, state):
        if state is not None:
            if set(state['duals']) != set(self.duals):
                raise ValueError('lap-time constraint roster changed')
            self.duals.update(state['duals']); self.ema.update(state['ema'])
            self.pending.update(state['pending'])

    def reward(self, slot, result, done, max_steps):
        name = Path(slot.track).stem
        if not hasattr(slot, 'lap_time_dual'):
            slot.lap_time_dual = self.duals[name]
        return transition_reward(self.courses[name]['reference_seconds'], self.hz, max_steps,
            done=done, success=succeeded(result) if done else False,
            steps=int(result['steps']) if done else 0, dual=slot.lap_time_dual)

    def update(self, rows):
        for row in rows:
            self.pending[Path(row['track']).stem].append(float(succeeded(row)))
        observed = 0
        for name, samples in self.pending.items():
            if len(samples) < int(self.config.get('dual_minimum_episodes', 16)):
                continue
            observed += 1
            self.ema[name] = .75 * self.ema[name] + .25 * float(np.mean(samples))
            floor = max(0., self.courses[name]['success_rate'] - self.config.get('training_success_tolerance', .05))
            self.duals[name] = float(np.clip(self.duals[name] + self.config.get('dual_learning_rate', 2.) *
                                           (floor - self.ema[name]), 0., self.config.get('maximum_dual', 20.)))
            samples.clear()
        return dict(lap_constraint_observed_courses=observed,
                    lap_constraint_dual_mean=float(np.mean(list(self.duals.values()))),
                    lap_constraint_dual_max=max(self.duals.values()))


def selection_metrics(metrics, config):
    """Fixed course pool: no dropping newly failed courses from the ranking."""
    reference = config['validation']
    margins = [float(metrics.get('full_course_success', -1.)) - reference['success_floor']]
    ratios = []
    for name, row in reference['courses'].items():
        prefix = f'track/{name}/'
        sr = float(metrics.get(prefix + 'full_course_success', -1.))
        margins.append(sr - row['success_floor'])
        if row['pace_eligible']:
            time = float(metrics.get(prefix + 'successful_lap_time_seconds', float('nan')))
            if not math.isfinite(time):
                margins.append(-1.)
            else:
                ratios.append(time / row['reference_seconds'])
    eligible_count = sum(row['pace_eligible'] for row in reference['courses'].values())
    qualified = min(margins) >= -1e-9 and len(ratios) == eligible_count and eligible_count > 0
    score = 1. - float(np.mean(ratios)) if len(ratios) == eligible_count and eligible_count else -1e6
    return dict(lap_time_score=score, lap_completion_qualified=float(qualified),
                lap_completion_minimum_margin=min(margins),
                lap_completion_violation_count=sum(m < -1e-9 for m in margins),
                lap_pace_course_count=eligible_count, selection_score=score)
