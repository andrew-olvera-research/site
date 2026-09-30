"""Quantized ordered transition cells: the manifold coordinates used by the v6.21
evaluation-set design, the corpus gap analysis and cell-directed course generation.

A cell is a behaviour class of one gate arrival (turn bin, altitude change, wrong-side
approach, incoming segment length) or an ordered pair/triple of consecutive arrivals.
Gate aperture and height are venue properties and are reported separately.
"""
from __future__ import annotations

import numpy as np

from .corpus_coverage import geometry_record

TURN_EDGES = (0, 30, 60, 90, 120, 150, 181)
TURN_NAMES = ('0-30', '30-60', '60-90', '90-120', '120-150', '150-180')
RISE_NAMES = {'-1': 'drop', '0': 'level', '1': 'climb'}
INCOMING_NAMES = ('<5', '5-12', '>=12')


def transition_symbol(t):
    """(turn bin, rise) of one arrival descriptor from ``transition_metrics``."""
    tb = next(i for i in range(6) if TURN_EDGES[i] <= t['turn_deg'] < TURN_EDGES[i + 1])
    rise = -1 if t['height_change_m'] < -0.75 else (1 if t['height_change_m'] > 0.75 else 0)
    return tb, rise


def arrival_cells(t, prev, nxt=None):
    """Cells of one arrival given its predecessor (and optional successor for trigrams)."""
    tb, rise = transition_symbol(t)
    inc = 0 if t['incoming_m'] < 5 else (1 if t['incoming_m'] < 12 else 2)
    cells = {f'u:{tb}:{rise}:{int(t["preceding_gate_on_exit_side"])}:{inc}'}
    pb, pr = transition_symbol(prev)
    cells.add(f'b:{pb}:{pr},{tb}:{rise}')
    if nxt is not None:
        nb, nr = transition_symbol(nxt)
        cells.add(f't:{pb}:{pr},{tb}:{rise},{nb}:{nr}')
    if t.get('reverse_entry'):
        cells.add('rev')
    return cells


def cells_from_transitions(transitions, trigrams=True):
    n = len(transitions)
    out = set()
    for k in range(n):
        out |= arrival_cells(transitions[k], transitions[k - 1], transitions[(k + 1) % n] if trigrams else None)
    return out


def transition_cells(track, trigrams=True):
    """All cells present in a track (its geometry_record transitions)."""
    return cells_from_transitions(geometry_record(track)['transitions'], trigrams)


def describe_cell(cell):
    if cell == 'rev':
        return 'reverse (wrong-side) entry'
    if cell.startswith('u:'):
        tb, r, ex, inc = cell[2:].split(':')
        return (f'turn {TURN_NAMES[int(tb)]}deg, {RISE_NAMES[r]}, '
                f'{"wrong-side approach, " if ex == "1" else ""}incoming {INCOMING_NAMES[int(inc)]} m')
    if cell.startswith(('b:', 't:')):
        parts = cell[2:].split(',')
        return ' -> '.join(f'{TURN_NAMES[int(p.split(":")[0])]}deg {RISE_NAMES[p.split(":")[1]]}' for p in parts)
    return cell


def cell_weight_from_gap(cell, gap_cells, train_support):
    """Weight for directed generation: episodes lost per unit of training support."""
    lost = float(gap_cells.get(cell, 0.0))
    return lost / (1.0 + float(train_support.get(cell, 0)))
