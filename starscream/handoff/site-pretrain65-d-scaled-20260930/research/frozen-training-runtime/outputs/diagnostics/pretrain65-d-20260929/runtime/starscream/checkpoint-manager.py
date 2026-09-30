"""Compatibility entrypoint for the importable checkpoint_manager module.

Python module names cannot contain hyphens. Training code imports
``starscream.checkpoint_manager``; this file remains as the requested visible
entrypoint and can also be executed directly for import validation.
"""

from starscream.checkpoint_manager import (  # noqa: F401
    CheckpointManager,
    RankedCheckpoint,
    capture_rng_state,
    restore_rng_state,
)

__all__ = [
    "CheckpointManager",
    "RankedCheckpoint",
    "capture_rng_state",
    "restore_rng_state",
]
