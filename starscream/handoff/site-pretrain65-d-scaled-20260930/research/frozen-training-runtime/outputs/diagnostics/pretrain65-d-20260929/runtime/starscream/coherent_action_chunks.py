"""Contiguous re-solved expert commands, with explicit unavailable-label encoding.

Normalized CTBR commands lie in [-1,1]. A finite 2 sentinel represents a missing
chunk (not an action), preserving the existing FP16 replay schema. Consumers
MUST decode the validity mask before computing losses. Single-step labels remain
separate and valid even when a coherent future continuation was not observed.
"""
import numpy as np
import torch

MISSING_CHUNK = 2.0


def build_coherent_chunks(actions, events, horizon):
    """events: one (local/global row index or None, executed clean expert) per tick.

    A query at i+k is on the expert continuation only when all k preceding
    actions followed that expert, without DART noise or invalid solver steps.
    No crossing invalid gaps, episode boundaries, or repeating terminal labels.
    """
    if horizon < 2:
        raise ValueError('coherent chunks require at least two actions')
    result = {}
    for start, (row, _) in enumerate(events):
        if row is None:
            continue
        chunk = np.full((horizon, 4), MISSING_CHUNK, np.float32)
        window = events[start:start+horizon]
        if (len(window) == horizon and all(i is not None for i, _ in window)
                and all(follows for _, follows in window[:-1])):
            chunk = np.stack([actions[i] for i, _ in window]).astype(np.float32)
            if not np.isfinite(chunk).all() or np.abs(chunk).max() > 1.0001:
                raise ValueError('coherent command is not normalized finite CTBR')
        result[row] = chunk
    return result


def decode_coherent_chunk_targets(targets):
    valid = torch.isfinite(targets).all(dim=(-1, -2)) & (targets.abs() <= 1.0001).all(dim=(-1, -2))
    clean = torch.where(valid[:, None, None], targets, torch.zeros_like(targets))
    return clean, valid
