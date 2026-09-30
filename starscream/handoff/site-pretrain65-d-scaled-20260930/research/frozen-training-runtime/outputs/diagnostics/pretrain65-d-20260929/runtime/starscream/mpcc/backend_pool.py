"""Process-local ACADOS reuse keyed by the complete immutable MPCC config.

Vehicle parameters and world bounds are refreshed by MPCCController. Controller
weights, horizon, slew and authority limits must instead match the compiled
backend. A worker can keep its original backend as a stable pool handle.
"""
from typing import Any
from .config import MPCCConfig


def select_backend(handle: Any | None, config: MPCCConfig):
    if handle is None:
        return None, {}
    pool=getattr(handle,'_starscream_config_backend_pool',None)
    if pool is None:
        pool={handle.config.fingerprint:handle}
        handle._starscream_config_backend_pool=pool
    return pool.get(config.fingerprint),pool


def register_backend(pool: dict, backend: Any | None):
    if backend is not None:
        pool[backend.config.fingerprint]=backend
        backend._starscream_config_backend_pool=pool
