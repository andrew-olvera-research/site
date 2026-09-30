"""Native OpenMP integration of independent existing rich Flightmare plants.

Reset/randomization and Python reward/observation oracles remain authoritative.
No Unity, auto-reset, hidden substeps, altered action delays or physics surrogate.
"""
import numpy as np


def require_native_batch():
    import flightgym
    if getattr(flightgym, 'rich_batch_schema_version', None) != 1:
        raise RuntimeError('rebuild project Flightgym: native rich batch schema v1 required')
    return flightgym


def telemetry_row(row):
    return dict(state=row[:25], time=float(row[25]), motor_thrusts=row[26:30],
                motor_omega=row[30:34], ctbr_command=row[34:38], collision=bool(row[38]))


def advance_prepared(envs, *, threads=1):
    """Native integration; all lanes must have prepared their delayed actions."""
    native = require_native_batch()
    if int(threads) > 1 and not getattr(native, 'rich_batch_openmp', False):
        raise RuntimeError('native multi-thread stepping requires an OpenMP build')
    if len({id(e) for e in envs}) != len(envs):
        raise ValueError('duplicate environment in native batch')
    if not envs:
        return []
    if any(not getattr(e, '_step_pending', False) or e.render_observations or e._connected for e in envs):
        raise ValueError('native batch requires prepared, non-rendering lanes')
    actions = np.stack([e._step_pending[1].as_array() for e in envs])
    times = np.asarray([e.control_dt for e in envs], np.float32)
    rows = native.step_ctbr_batch([e._native for e in envs], actions, times, int(threads))
    return [telemetry_row(row) for row in rows]


def step_batch(envs, actions, *, threads=1):
    actions = np.asarray(actions, np.float32)
    if actions.shape != (len(envs), 4) or not np.isfinite(actions).all():
        raise ValueError('batch actions must be finite[N,4]')
    if len({id(e) for e in envs}) != len(envs):
        raise ValueError('duplicate environment in native batch')
    require_native_batch()
    if any(getattr(e, '_step_pending', False) for e in envs):
        raise RuntimeError('cannot overwrite a pending step')
    if not 1 <= int(threads) <= 256 or any(e.render_observations or e._connected for e in envs):
        raise ValueError('native batch requires non-rendering lanes and threads1..256')
    for env, action in zip(envs, actions):
        env.prepare_step(action)
    return [env.finish_step(raw) for env, raw in zip(envs, advance_prepared(envs, threads=threads))]
