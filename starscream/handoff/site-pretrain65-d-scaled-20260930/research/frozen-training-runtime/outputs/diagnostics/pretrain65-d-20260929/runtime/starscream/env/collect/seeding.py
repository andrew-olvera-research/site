"""Deterministic, collision-resistant seeds for resumable collection."""

from __future__ import annotations

import hashlib


def collection_episode_seed(
    track_name: str, domain: str, episode_index: int, attempt: int,
) -> int:
    """Return a stable seed without cross-track arithmetic aliasing."""

    payload = (
        f"starscream-perfect-expert-v2\0{track_name}\0{domain}\0"
        f"{int(episode_index)}\0{int(attempt)}"
    ).encode("utf-8")
    # Flightmare consumes a nonnegative 31-bit seed. Using identity fields
    # avoids the exact 104729-stride collisions of the legacy linear formula.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)
