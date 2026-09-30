"""Versioned, fixed-unit plant conditioning for the privileged racing teacher.

49 episode constants and 15 causal context values. No fitted statistics, seeds,
future simulator states, or teacher actions enter this descriptor.
"""
from __future__ import annotations

import numpy as np

PLANT_OBSERVATION_CONTRACT = 'starscream_route_plant_v1'
PLANT_SETTINGS_DIM = 64
PLANT_STATIC_DIM = 49
AXES = 'xyz'
PLANT_SETTINGS_NAMES = (
    'mass', 'arm_length', 'inertia_xx', 'inertia_yy', 'inertia_zz',
    'motor_omega_min', 'motor_omega_max', 'motor_tau',
    'thrust_map_quadratic', 'thrust_map_linear', 'thrust_map_constant', 'kappa',
    *(f'body_rate_max_{a}' for a in AXES),
    *(f'{kind}_{a}' for kind in ('linear_drag', 'quadratic_drag', 'rotor_drag',
      'angular_drag', 'wind_mean', 'wind_gust_amplitude', 'wind_gust_frequency') for a in AXES),
    *(f'initial_gust_phase_sin_{a}' for a in AXES),
    *(f'initial_gust_phase_cos_{a}' for a in AXES),
    *(f'center_of_mass_{a}' for a in AXES),
    'action_delay', 'control_dt', 'motor_thrust_min', 'motor_thrust_max',
    *(f'current_gust_phase_sin_{a}' for a in AXES),
    *(f'current_gust_phase_cos_{a}' for a in AXES),
    *(f'wind_body_{a}' for a in AXES),
    *(f'world_from_body_{row}{col}' for row in range(3) for col in range(2)),
)
# Characteristic physical units: preserve nominal/zero meaning and magnitude.
# Thrust polynomial terms use their contribution near 3000 rad/s, not a divide
# by their nominal coefficient (which can be zero).
PLANT_STATIC_SCALE = np.asarray([
    .73, .17, .73/12*.17**2*4.5, .73/12*.17**2*4.5, .73/12*.17**2*7,
    150, 3000, .0001, 25/3000**2, 25/3000, 25, .016, 6, 6, 6,
    .16, .16, .22, .03, .03, .04, 12e-6, 12e-6, 2e-6,
    .008, .008, .008, 1.2, 1.2, .25, .8, .8, .2, .8, .8, .8,
    1, 1, 1, 1, 1, 1, .006, .006, .003, .02, .01, 25, 25,
], np.float32)
assert len(PLANT_SETTINGS_NAMES) == PLANT_SETTINGS_DIM
assert len(PLANT_STATIC_SCALE) == PLANT_STATIC_DIM


def plant_static_settings(dynamics, aerodynamics, actuator, action_delay, control_dt):
    dynamics = np.asarray(dynamics, np.float32)
    aero = np.asarray(aerodynamics, np.float32)
    if dynamics.shape != (15,) or aero.shape != (27,):
        raise ValueError('Plant settings require applied dynamics[15] and aerodynamics[27]')
    values = np.concatenate([dynamics, aero[:21], np.sin(aero[21:24]),
        np.cos(aero[21:24]), aero[24:27],
        np.asarray([action_delay, control_dt, *np.asarray(actuator)[-2:]], np.float32)])
    if not np.isfinite(values).all():
        raise ValueError('Non-finite applied plant settings')
    return (values / PLANT_STATIC_SCALE).astype(np.float32)


def plant_settings(static, aerodynamics, time, world_from_body):
    """Add observable gust phase and world/body frame context at command time."""
    aero = np.asarray(aerodynamics, np.float32)
    phase = np.float32(2*np.pi) * aero[18:21] * np.float32(time) + aero[21:24]
    sine, cosine = np.sin(phase), np.cos(phase)
    rotation = np.asarray(world_from_body, np.float32)
    wind_body = (aero[12:15] + aero[15:18]*sine) @ rotation
    return np.concatenate([static, sine, cosine, wind_body/2.,
                           rotation[:, :2].reshape(-1)]).astype(np.float32)
