"""Vector-environment constructors kept separate from native FlightGym."""

from __future__ import annotations

import gymnasium as gym

from .flightgym import FlightmareEnv


def make_vector_env(num_envs: int, **environment_kwargs) -> gym.vector.VectorEnv:
    """Create isolated native simulators; rendering is intended for one env only."""

    if environment_kwargs.get("render_observations") and num_envs != 1:
        raise ValueError("the Unity bridge singleton supports one rich rendered env per process")
    return gym.vector.SyncVectorEnv(
        [lambda kwargs=environment_kwargs: FlightmareEnv(**kwargs) for _ in range(num_envs)]
    )
