"""v6.21: motif grammars for the transition regimes the v6.20.1 corpus lacked, plus
cell-directed rejection sampling.

The three real references and the v6.21 held-out suite fail on (a) hairpins that follow
hairpins with 6-10 m legs (CDRA), (b) hairpins combined with 2-3 m altitude change into
low gates (Swift gate 2, A2RL gates 5-7) and (c) hairpins entered from long straights
(UTT). The v6.20.1 corpus contains 7 / 2 / 19 such arrivals in 245. Each grammar here
builds an explicit motif in a local frame and closes the loop with a smooth Bezier
return so joins are compatible; static admission and MPCC qualification still decide
whether a proposal is raceable. Nothing here copies real-track coordinates.
"""
from __future__ import annotations

import numpy as np

from ..tracks import Gate, Track, forward_up_quaternion
from .transition_cells import cells_from_transitions
from .transition_corpus import transition_metrics
from .transition_corpus_v201 import generate_course as generate_course_v201, screen

MOTIF_GRAMMARS = ('hairpin_chain', 'diving_hairpin', 'straight_hairpin')
BASE_GRAMMARS = ('banking', 'slalom', 'braking_reversal', 'stacked_reversal', 'go_around')
GRAMMARS = MOTIF_GRAMMARS + BASE_GRAMMARS
REVISION = 5


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _heading(angle):
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def _motif(grammar, rng, low_gates):
    """Return motif points (k,3) in a local frame heading +x, plus the exit heading angle."""
    z0 = rng.uniform(1.0, 1.4) if low_gates else rng.uniform(1.3, 4.0)
    p = [np.array([0.0, 0.0, z0])]
    heading = 0.0
    if grammar == 'hairpin_chain':
        hairpins = int(rng.integers(2, 4))          # two or three consecutive hairpins
        pattern = rng.choice(['alternate', 'same'])
        sign = rng.choice([-1.0, 1.0])
        z = z0
        for i in range(hairpins + 1):
            leg = rng.uniform(6.0, 10.0)
            if rng.random() < 0.5:
                dz = rng.uniform(-2.5, 2.5)
                z = float(np.clip(z + dz, 1.0 if low_gates else 1.2, 6.0))
            p.append(p[-1] + leg * _heading(heading) + np.array([0, 0, z - p[-1][2]]))
            if i < hairpins:
                turn = np.radians(rng.uniform(112.0, 150.0)) * sign
                heading += turn
                if pattern == 'alternate':
                    sign = -sign
    elif grammar == 'diving_hairpin':
        # Swift gates 1-3: a climbing hard turn onto a high gate, then a hairpin whose
        # arrival drops 2-3 m into a low gate, then a level exit. Turn and drop sit on
        # the same arrival descriptor (turn at the low gate, incoming segment descending).
        climb = rng.uniform(2.0, 3.2)
        leg = rng.uniform(8.0, 12.0)
        heading += np.radians(rng.uniform(-30.0, 30.0))
        p.append(p[-1] + leg * _heading(heading) + np.array([0, 0, climb]))      # high gate (climbing arrival)
        heading += np.radians(rng.uniform(100.0, 135.0)) * rng.choice([-1.0, 1.0])
        low = rng.uniform(1.0, 1.3)
        leg = rng.uniform(8.0, 12.0)
        p.append(p[-1] + leg * _heading(heading) + np.array([0, 0, low - p[-1][2]]))  # low gate (diving arrival)
        heading += np.radians(rng.uniform(112.0, 140.0)) * rng.choice([-1.0, 1.0])   # hairpin AT the low gate
        leg = rng.uniform(8.0, 12.0)
        p.append(p[-1] + leg * _heading(heading) + np.array([0, 0, rng.uniform(-0.1, 0.5)]))
    elif grammar == 'straight_hairpin':
        straight = rng.uniform(15.0, 24.0)
        p.append(p[-1] + straight * _heading(heading) + np.array([0, 0, rng.uniform(-0.4, 0.4)]))
        turn = np.radians(rng.uniform(140.0, 170.0)) * rng.choice([-1.0, 1.0])
        heading += turn
        leg = rng.uniform(6.0, 9.0)
        p.append(p[-1] + leg * _heading(heading) + np.array([0, 0, rng.uniform(-0.6, 0.6)]))
        heading += np.radians(rng.uniform(-20.0, 20.0))
        straight = rng.uniform(12.0, 20.0)
        p.append(p[-1] + straight * _heading(heading) + np.array([0, 0, rng.uniform(-0.4, 0.4)]))
    else:
        raise ValueError(grammar)
    return np.asarray(p, float), heading


def _bezier_return(exit_point, exit_heading, entry_point, entry_heading, m, rng):
    """m gates on a cubic Bezier from the motif exit back to its entry with compatible tangents."""
    chord = float(np.linalg.norm(entry_point[:2] - exit_point[:2]))
    reach = max(0.9 * chord, 10.0 + 3.5 * m)
    c1 = exit_point + reach * _heading(exit_heading)
    c2 = entry_point - reach * _heading(entry_heading)
    # A lateral bulge away from the motif keeps the return leg clear of it.
    side = np.cross([0, 0, 1.0], _heading(exit_heading))
    bulge = rng.uniform(4.0, 9.0) * rng.choice([-1.0, 1.0])
    c1 = c1 + bulge * side
    c2 = c2 + bulge * side
    ts = np.linspace(0.0, 1.0, m + 2)[1:-1]
    pts = []
    for t in ts:
        b = ((1 - t) ** 3) * exit_point + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * (t ** 2) * c2 + (t ** 3) * entry_point
        b[2] = float(np.clip(b[2] + rng.uniform(-0.6, 0.6), 1.0, 6.0))
        pts.append(b)
    return np.asarray(pts, float)


def generate_course(grammar, count, seed, name, *, low_gates=False, narrow_gates=False):
    """One v6.21 course. Base grammars delegate to the v6.20.1 revision-4 generator."""
    if grammar in BASE_GRAMMARS:
        return generate_course_v201(grammar, count, seed, name, revision=4)
    if grammar not in MOTIF_GRAMMARS:
        raise ValueError(grammar)
    if not 5 <= count <= 10:
        raise ValueError('v6.21 motif courses need 5..10 gates')
    rng = np.random.default_rng(seed)
    motif, exit_heading = _motif(grammar, rng, low_gates)
    if len(motif) > count - 1:
        raise ValueError(f'{grammar} motif needs at least {len(motif) + 1} gates')
    entry_heading = 0.0
    ret = _bezier_return(motif[-1], exit_heading, motif[0], entry_heading, count - len(motif), rng)
    p = np.vstack([motif, ret])
    n = len(p)
    normals = []
    for i in range(n):
        a = _unit(p[i] - p[(i - 1) % n]); b = _unit(p[(i + 1) % n] - p[i])
        nrm = a + b; nrm[2] = 0.0
        if np.linalg.norm(nrm) < 1e-7:
            nrm = b.copy(); nrm[2] = 0.0
        nrm = _unit(nrm)
        yaw = np.radians(rng.uniform(-25.0, 25.0)); c, s = np.cos(yaw), np.sin(yaw)
        normals.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ nrm)
    mirror = -1 if seed % 2 else 1
    p[:, 1] *= mirror
    normals = [nrm * [1, mirror, 1] for nrm in normals]
    yaw = rng.uniform(-np.pi, np.pi); c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    p = p @ R.T
    gates = []
    for i, (point, nrm) in enumerate(zip(p, normals)):
        width = rng.uniform(1.45, 1.65) if narrow_gates else rng.uniform(1.45, 2.5)
        height = min(rng.uniform(1.45, 2.3), 2.0 * (float(point[2]) - 0.28))   # keep >=0.25 m floor clearance
        gates.append(Gate(point.astype(np.float32), forward_up_quaternion(R @ nrm),
                          np.array([width, max(height, 1.2)], np.float32), name=f'gate_{i:02}'))
    offset = int(rng.integers(0, n))                 # motif not always at the cold spawn
    gates = list(np.roll(np.asarray(gates, dtype=object), offset))
    motif_ids = [(i + offset) % n for i in range(len(motif))]
    bounds = np.stack((p.min(0) - 4, p.max(0) + 4), axis=1); bounds[2, 0] = 0
    return Track(name, tuple(gates), bounds.astype(np.float32), True, dict(
        source='v621_directed_motif', revision=REVISION, seed=int(seed), mirror=mirror,
        intended_transition=grammar, transition_gate_indices=motif_ids,
        low_gates=bool(low_gates), narrow_gates=bool(narrow_gates), behavior_certified=False))


def course_cells(track, trigrams=True):
    return cells_from_transitions(transition_metrics_with_frames(track), trigrams)


def transition_metrics_with_frames(track):
    """transition_metrics plus the gate-frame fields geometry_record adds (width/height/centre)."""
    from .corpus_coverage import geometry_record
    return geometry_record(track)['transitions']


def generate_for_cells(target_weights, seed, *, counts=(5, 6, 7, 8, 9, 10), grammars=GRAMMARS,
                       attempts=200, name='directed', low_gate_probability=0.35, narrow_probability=0.25,
                       reject=None):
    """Rejection-sample a screened course maximizing weighted coverage of ``target_weights``.

    ``target_weights`` maps cell -> weight (e.g. episodes lost per unit training support).
    ``reject`` is an optional predicate on tracks (e.g. anti-clone against a pool).
    Returns (track, covered_cells, score) or (None, set(), 0.0) if nothing screened.
    """
    rng = np.random.default_rng(seed)
    best = (None, set(), 0.0)
    for attempt in range(attempts):
        grammar = str(rng.choice(grammars))
        count = int(rng.choice(counts))
        if grammar in BASE_GRAMMARS:
            count = int(np.clip(count, 4, 10))
        s = int(seed * 1000 + attempt)
        try:
            track = generate_course(grammar, count, s, f'{name}_{grammar}_{count}_{s}',
                                    low_gates=bool(rng.random() < low_gate_probability),
                                    narrow_gates=bool(rng.random() < narrow_probability))
        except ValueError:
            continue
        if screen(track):
            continue
        if reject is not None and reject(track):
            continue
        cells = course_cells(track)
        covered = {c for c in cells if c in target_weights}
        score = float(sum(target_weights[c] for c in covered))
        if score > best[2]:
            best = (track, covered, score)
    return best
