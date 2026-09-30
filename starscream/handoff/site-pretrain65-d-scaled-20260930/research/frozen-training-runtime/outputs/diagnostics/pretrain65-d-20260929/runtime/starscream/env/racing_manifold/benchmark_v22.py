"""Versioned benchmark coordinates and independent course proposals.

Cells are geometric requirements, never a claim about executed speed/braking.
Use MPCC traces for those measurements. No policy scores enter selection.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from starscream.course_model.schema import static_reasons
from starscream.env.tracks import Gate, Track, forward_up_quaternion
from .corpus_coverage import geometry_record
from .transition_corpus_v21 import generate_course as legacy_generate, _bezier_return

VERSION = 'v622-requirements-1'
FAMILIES = ('flow', 'slalom', 'hairpin_chain', 'diving_hairpin',
            'long_low', 'long_braking', 'ordered_3d', 'stacked_reversal', 'go_around')


class CloneIndex:
    """Vectorized equivalent of the existing cyclic/reflection shape metric."""

    def __init__(self, tracks=()):
        self.groups = {}
        for track in tracks:
            self.add(track)

    @staticmethod
    def coordinates(track):
        p = np.asarray([g.position for g in track.gates], float)
        p -= p.mean(0)
        return p / max(np.sqrt((p*p).sum(1).mean()), 1e-8)

    def add(self, track):
        p = self.coordinates(track)
        shifted = np.stack([np.roll(p, i, axis=0) for i in range(len(p))])
        self.groups.setdefault(len(p), []).append(shifted)

    def distance(self, track):
        x = self.coordinates(track)
        if len(x) not in self.groups:
            return 9.
        ys = np.concatenate(self.groups[len(x)])
        best = 0.
        for mirror in (1, -1):
            xx = x * [1, mirror, 1]
            dot = (ys[:, :, :2]*xx[None, :, :2]).sum((1, 2))
            cross = (xx[None, :, 0]*ys[:, :, 1]-xx[None, :, 1]*ys[:, :, 0]).sum(1)
            vertical = (ys[:, :, 2]*xx[None, :, 2]).sum(1)
            best = max(best, float((np.hypot(dot, cross)+vertical).max())/len(x))
        return float(np.sqrt(max(0., 2.-2.*best)))


def requirement_cells(track):
    """Factorized joint cells avoid an almost-empty full Cartesian product.

    Course witnesses are sets: repeating a motif does not inflate support.
    Ordered chains retain handedness, vertical magnitude and directed entry.
    Closure cells are geometric only until multi-lap qualification is available.
    """
    return set().union(*requirement_cells_by_gate(track))


def requirement_cells_by_gate(track, *, finite_route=False):
    """Same requirement coordinates, retaining each training gate's identity."""
    rows = geometry_record(track)['transitions']
    gates, symbols = [], []
    for t in rows:
        cells = set()
        turn = min(5, int(t['turn_deg'] // 30))
        length = int(np.searchsorted([5, 12, 20, 30, 40], t['incoming_m'], side='right'))
        dz = t['height_change_m']
        rise = int(np.searchsorted([.75, 2, 3], abs(dz), side='right')) * int(np.sign(dz))
        low = int(np.searchsorted([.80001, 1.20001, 2.50001], t['gate_center_height_m']))
        aperture = int(np.searchsorted([1.20001, 1.65001, 2.20001], min(t['width_m'], t['height_m'])))
        side = int(t['preceding_gate_on_exit_side'])
        incidence = int(np.searchsorted([0, .35, .7], t['incoming_alignment']))
        hand = int(np.sign(t['signed_horizontal_turn_deg'])) if turn else 0
        cells.update((f'approach:{length}:{turn}:{low}:{aperture}',
                      f'vertical:{rise}:{turn}:{low}',
                      f'entry:{side}:{int(t["reverse_entry"])}:{incidence}:{turn}'))
        symbols.append(f'{hand * (turn + 1)}:{rise}:{side}')
        gates.append(cells)
    for horizon in (2, 3, 4):
        for i in range(len(rows)):
            if not finite_route or i + horizon <= len(rows):
                gates[i].add(f'chain{horizon}:' + '/'.join(symbols[(i+j) % len(rows)] for j in range(horizon)))
    return gates


def validate_geometry(track):
    reasons = static_reasons(track)
    p = np.asarray([g.position for g in track.gates])
    legs = np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1)
    if not 5 <= len(p) <= 19:
        reasons.append('benchmark_gate_count')
    if legs.max() > 55 or legs.min() < 1.5:
        reasons.append('benchmark_leg_envelope')
    for g in track.gates:
        if np.any(g.position < track.bounds[:, 0]) or np.any(g.position > track.bounds[:, 1]):
            reasons.append('bounds')
        if not np.allclose(g.rotation.T @ g.rotation, np.eye(3), atol=1e-5):
            reasons.append('frame')
        if min(g.size) < 1.2 or max(g.size) > 3.1:
            reasons.append('aperture_envelope')
    return sorted(set(reasons))


def generate(family, count, seed, name, hard=False):
    """Independent seeds and explicit motifs; never transform a real course."""
    rng = np.random.default_rng(seed)
    if family not in ('long_low', 'long_braking', 'ordered_3d', 'flow'):
        track = legacy_generate(family, min(count, 10), seed, name,
                                low_gates=True, narrow_gates=hard)
        # Keep existing grammar semantics; vary physical realization jointly.
        gates = []
        scale = rng.uniform(.7, 1.5, 2)
        protected = set((track.metadata or {}).get('transition_gate_indices', []))
        for i, g in enumerate(track.gates):
            position = g.position.copy(); position[:2] *= scale
            if i not in protected:
                position[:2] += rng.uniform(-2., 2., 2)
            height = min(float(g.size[1]), 2 * float(position[2]))
            # Normals are transformed by the same planar map; explicit directed
            # crossings in stacked/go-around motifs retain their signs.
            normal = g.physical_normal.copy(); normal[:2] /= scale
            gates.append(replace(g, position=position, quaternion_wxyz=forward_up_quaternion(normal),
                                 size=np.array([g.size[0], height], np.float32)))
        return _track(name, gates, family, seed, hard)
    if family == 'flow':
        angles = np.cumsum(rng.uniform(.65, 1.4, count)); angles *= 2*np.pi/angles[-1]
        radial = rng.uniform(12, 25) * rng.uniform(.85, 1.15, count)
        p = np.column_stack((radial*np.cos(angles), radial*rng.uniform(.55, 1.1)*np.sin(angles),
                             rng.uniform(1.0, 2.5, count)))
    else:
        heading, sign = 0., float(rng.choice([-1, 1]))
        if family == 'ordered_3d':
            # Two descending hard arrivals, then a hard climbing arrival and
            # short descending leg. Four ordered transitions, with entry/exit.
            z = rng.uniform(5.7, 6.4)
            lengths = rng.uniform(4.8 if hard else 6., 8.5, 5)
            lengths[-1] = rng.uniform(3.5, 5.8)
            drops = rng.uniform(1.6, 2.3, 2)
            rises = [0., -drops[0], -drops[1], rng.uniform(2.5, 3.3), -rng.uniform(1., 2.)]
            turns = [rng.uniform(10, 30), *rng.uniform(100, 132 if hard else 122, 3), rng.uniform(100, 150)]
        else:
            z = rng.uniform(.77, 1.1) if family == 'long_low' else rng.uniform(1.3, 2.2)
            lengths = [rng.uniform(30 if hard else 20, 46 if hard else 35),
                       rng.uniform(6, 10), rng.uniform(20, 38 if hard else 30)]
            turns = [rng.uniform(145, 172) if family == 'long_braking' else rng.choice([20., 85., 150.]),
                     rng.uniform(-25, 25), rng.uniform(30, 65)]
            rises = [0., rng.uniform(0, 2.5) if hard and family == 'long_braking' else 0., 0.]
        pts = [np.array([0., 0., z])]
        for length, turn, rise in zip(lengths, turns, rises):
            pts.append(pts[-1] + [length*np.cos(heading), length*np.sin(heading), rise])
            heading += np.radians(turn)*sign
            if family == 'ordered_3d' and rng.random() < .45:
                sign *= -1
        motif = np.asarray(pts)
        ret = _bezier_return(motif[-1], heading, motif[0], 0., count-len(motif), rng)
        p = np.vstack((motif, ret))
    gates = []
    for i, point in enumerate(p):
        a = point-p[i-1]; b = p[(i+1) % len(p)]-point
        normal = a/max(np.linalg.norm(a), 1e-9)+b/max(np.linalg.norm(b), 1e-9)
        normal[2] = 0
        if np.linalg.norm(normal) < 1e-6:
            normal = np.array([b[0], b[1], 0.])
        width = rng.uniform(1.45, 1.65) if family == 'long_low' else rng.uniform(1.5 if hard else 1.8, 2.5)
        height = min(width, 2*float(point[2]))
        gates.append(Gate(point, forward_up_quaternion(normal), np.array([width, height]), name=f'gate_{i:02d}'))
    offset = int(rng.integers(len(gates)))
    gates = gates[offset:] + gates[:offset]
    return _track(name, gates, family, seed, hard)


def _track(name, gates, family, seed, hard):
    p = np.asarray([g.position for g in gates])
    bounds = np.stack((p.min(0)-5, p.max(0)+5), axis=1); bounds[2, 0] = 0
    return Track(name, tuple(gates), bounds, True,
                 dict(source=VERSION, seed=int(seed), intended_transition=family,
                      hard=bool(hard), behavior_certified=False))
