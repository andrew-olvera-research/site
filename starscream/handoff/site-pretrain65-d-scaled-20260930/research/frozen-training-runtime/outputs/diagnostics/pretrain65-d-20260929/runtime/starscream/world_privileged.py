"""Versioned fixed-world teacher features; raw SI units, never per-track resize.

The normalizer owns fixed physical scaling. World origin stays the simulator's
origin throughout the episode. Body rates/actions stay in their native frame.
This is NOT the production/raw or Green observation contract.
"""
import numpy as np
from .env.tracks import quaternion_matrix

WORLD_OBSERVATION_CONTRACT = 'starscream_world_route_v2'
WORLD_DISPLACEMENT_CONTRACT = 'starscream_world_displacement_v3'
WORLD_OBSERVATION_CONTRACTS = {WORLD_OBSERVATION_CONTRACT, WORLD_DISPLACEMENT_CONTRACT}

def world_observation_features(observation, route_gates, action_encoder, *, displacement=False):
    privileged = observation['privileged']
    state = np.asarray(privileged['state'], np.float32)
    rotation = quaternion_matrix(state[3:7])
    task = np.concatenate([state[:3], state[7:10],
                           rotation[:, :2].T.reshape(-1), state[10:13],
                           np.asarray(observation['motor_omega'], np.float32)])
    route = np.asarray(privileged['world_route_records'], np.float32)
    if task.shape != (19,) or route.shape != (route_gates, 13):
        raise ValueError('world teacher requires state19 and route[N,13]')
    if displacement:
        route = route.copy()
        route[:, :3] -= state[:3]
    previous = action_encoder(np.asarray(observation['previous_action'], np.float32))
    result = np.concatenate([task, route.reshape(-1), previous,
        [float(observation['age']['previous_action']),
         float(observation['valid']['previous_action'])]]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite world teacher observation')
    return result

def world_feature_scales(route_gates=6):
    # ±30m is a numerical scale, NOT a hard bound. No clipping of any positions.
    task = [30.]*3 + [20.]*3 + [1.]*6 + [6.]*3 + [2000.]*4
    route = [30.]*3 + [1.]*6 + [2.]*2 + [1.]*2
    return np.asarray(task + route*route_gates + [1.]*4 + [.02,1.], np.float32)

def calibrated_world_statistics(features, route_gates=6, *, displacement=False):
    """Frozen physical-group calibration, shared position units across token types.

    Input rows are absolute-world teacher features, never validation or student
    estimates. No per-slot near-zero standard deviations for fixed route flags.
    Displacement and absolute arms share every scale and state statistic; only
    route position centering changes, since subtracting two positions cancels
    the common origin. Dynamics target units are deliberately separate.
    """
    x = np.asarray(features, np.float64)
    width = 19 + 13 * route_gates + 6
    if x.ndim != 2 or x.shape[1] != width or len(x) < 100 or not np.isfinite(x).all():
        raise ValueError('calibration requires at least 100 finite world rows')
    mean = np.zeros(width, np.float64)
    std = world_feature_scales(route_gates).astype(np.float64)
    route = x[:, 19:19+13*route_gates].reshape(-1, route_gates, 13)
    # Equal weight for the vehicle and the set of upcoming positions.
    center = .5 * (x[:, :3].mean(0) + route[:, :, :3].mean((0, 1)))
    position_scale = max(1., float(np.sqrt(.5 * (
        np.mean((x[:, :3]-center)**2) + np.mean((route[:, :, :3]-center)**2)))))
    mean[:3] = center; std[:3] = position_scale
    for i in range(route_gates):
        lo=19+13*i
        mean[lo:lo+3] = 0 if displacement else center
        std[lo:lo+3] = position_scale
    for lo,hi,floor in [(3,6,1.),(12,15,.5),(15,19,100.)]:
        # Velocity/body rate origins are physically meaningful; use RMS about
        # zero. Motors share one offset/scale to avoid axis-dependent artifacts.
        center_group = float(x[:,lo:hi].mean()) if lo==15 else 0.
        mean[lo:hi] = center_group
        std[lo:hi] = max(floor, float(np.sqrt(np.mean((x[:,lo:hi]-center_group)**2))))
    return mean.astype(np.float32), std.astype(np.float32)
