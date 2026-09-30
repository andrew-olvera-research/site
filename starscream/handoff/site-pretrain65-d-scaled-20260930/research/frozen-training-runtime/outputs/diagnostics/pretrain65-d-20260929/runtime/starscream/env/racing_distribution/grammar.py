"""Compositional maneuver grammar and deterministic coverage scheduling."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from typing import Iterable, Sequence

import numpy as np

from .schema import (
    CourseProgram, GeneratorBackend, HARD_MANEUVERS, ManeuverKind, ManeuverSpec,
    RacingDistributionConfig, VERTICAL_MANEUVERS,
)


K = ManeuverKind

# These are semantic coverage obligations, not a probability table.  The list
# captures acceleration-to-braking, planar direction changes, vertical-mode
# transitions, and difficult exits.  It stays intentionally finite so coverage
# can be achieved and falsified with a bounded corpus.
REQUIRED_BIGRAMS: frozenset[tuple[K, K]] = frozenset({
    (K.ACCELERATION, K.BRAKING), (K.ACCELERATION, K.TURN_LEFT),
    (K.ACCELERATION, K.TURN_RIGHT), (K.BRAKING, K.HAIRPIN_LEFT),
    (K.BRAKING, K.HAIRPIN_RIGHT), (K.BRAKING, K.SPLIT_S_LEFT),
    (K.BRAKING, K.SPLIT_S_RIGHT), (K.STRAIGHT, K.CLIMB),
    (K.STRAIGHT, K.DIVE), (K.CLIMB, K.TURN_LEFT),
    (K.CLIMB, K.TURN_RIGHT), (K.DIVE, K.TURN_LEFT),
    (K.DIVE, K.TURN_RIGHT), (K.TURN_LEFT, K.TURN_RIGHT),
    (K.TURN_RIGHT, K.TURN_LEFT), (K.TURN_LEFT, K.SLALOM_RIGHT),
    (K.TURN_RIGHT, K.SLALOM_LEFT), (K.SLALOM_LEFT, K.BRAKING),
    (K.SLALOM_RIGHT, K.BRAKING), (K.HAIRPIN_LEFT, K.ACCELERATION),
    (K.HAIRPIN_RIGHT, K.ACCELERATION), (K.SPLIT_S_LEFT, K.ACCELERATION),
    (K.SPLIT_S_RIGHT, K.ACCELERATION), (K.CORKSCREW_LEFT, K.DIVE),
    (K.CORKSCREW_RIGHT, K.DIVE), (K.STACKED_REVERSAL, K.ACCELERATION),
    (K.CLIMB, K.CORKSCREW_LEFT), (K.CLIMB, K.CORKSCREW_RIGHT),
    (K.DIVE, K.STACKED_REVERSAL), (K.BRAKING, K.STACKED_REVERSAL),
})

REQUIRED_TRIGRAMS: frozenset[tuple[K, K, K]] = frozenset({
    (K.ACCELERATION, K.BRAKING, K.HAIRPIN_LEFT),
    (K.ACCELERATION, K.BRAKING, K.HAIRPIN_RIGHT),
    (K.ACCELERATION, K.BRAKING, K.SPLIT_S_LEFT),
    (K.ACCELERATION, K.BRAKING, K.SPLIT_S_RIGHT),
    (K.STRAIGHT, K.CLIMB, K.DIVE),
    (K.STRAIGHT, K.DIVE, K.CLIMB),
    (K.CLIMB, K.TURN_LEFT, K.DIVE),
    (K.CLIMB, K.TURN_RIGHT, K.DIVE),
    (K.TURN_LEFT, K.SLALOM_RIGHT, K.TURN_LEFT),
    (K.TURN_RIGHT, K.SLALOM_LEFT, K.TURN_RIGHT),
    (K.BRAKING, K.STACKED_REVERSAL, K.ACCELERATION),
    (K.BRAKING, K.SPLIT_S_LEFT, K.ACCELERATION),
    (K.BRAKING, K.SPLIT_S_RIGHT, K.ACCELERATION),
    (K.CLIMB, K.CORKSCREW_LEFT, K.DIVE),
    (K.CLIMB, K.CORKSCREW_RIGHT, K.DIVE),
    (K.HAIRPIN_LEFT, K.ACCELERATION, K.TURN_RIGHT),
    (K.HAIRPIN_RIGHT, K.ACCELERATION, K.TURN_LEFT),
    (K.DIVE, K.TURN_LEFT, K.ACCELERATION),
    (K.DIVE, K.TURN_RIGHT, K.ACCELERATION),
    (K.ACCELERATION, K.TURN_LEFT, K.SLALOM_RIGHT),
    (K.ACCELERATION, K.TURN_RIGHT, K.SLALOM_LEFT),
    (K.STACKED_REVERSAL, K.ACCELERATION, K.BRAKING),
})

# Procedural composition probes can force these combinations while excluding
# them from the ordinary train/validation scheduler.  They are not a substitute
# for sealed real courses, but make compositional extrapolation measurable.
COMPOSITION_HOLDOUT_TRIGRAMS: tuple[tuple[K, K, K], ...] = (
    (K.SPLIT_S_LEFT, K.CLIMB, K.HAIRPIN_RIGHT),
    (K.SPLIT_S_RIGHT, K.CLIMB, K.HAIRPIN_LEFT),
    (K.STRAIGHT, K.CLIMB, K.CORKSCREW_LEFT),
    (K.STRAIGHT, K.CLIMB, K.CORKSCREW_RIGHT),
)


def cyclic_ngrams(values: Sequence[K | str], order: int) -> tuple[tuple[str, ...], ...]:
    """Return pose-independent cyclic primitive n-grams."""

    if order < 1 or len(values) < order:
        return ()
    normalized = tuple(item.value if isinstance(item, K) else str(item) for item in values)
    return tuple(
        tuple(normalized[(index + offset) % len(normalized)] for offset in range(order))
        for index in range(len(normalized))
    )


def required_ngrams(order: int) -> frozenset[tuple[str, ...]]:
    raw: Iterable[tuple[K, ...]]
    if order == 2:
        raw = REQUIRED_BIGRAMS
    elif order == 3:
        raw = REQUIRED_TRIGRAMS
    else:
        raise ValueError("coverage order must be two or three")
    return frozenset(tuple(item.value for item in gram) for gram in raw)


class ManeuverGrammar:
    """Sample valid programs and greedily maximize finite n-gram coverage."""

    _easy = (K.STRAIGHT, K.ACCELERATION, K.BRAKING)
    _planar = (
        K.TURN_LEFT, K.TURN_RIGHT, K.HAIRPIN_LEFT, K.HAIRPIN_RIGHT,
        K.SLALOM_LEFT, K.SLALOM_RIGHT,
    )
    _vertical = tuple(sorted(VERTICAL_MANEUVERS, key=lambda item: item.value))
    _hard = tuple(sorted(HARD_MANEUVERS, key=lambda item: item.value))

    def __init__(self, config: RacingDistributionConfig | None = None) -> None:
        self.config = config or RacingDistributionConfig()

    @staticmethod
    def _program_id(program: Sequence[ManeuverSpec], seed: int, split: str) -> str:
        payload = json.dumps({
            "kinds": [item.kind.value for item in program],
            "parameters": [item.to_mapping() for item in program],
            "seed": int(seed), "split": split,
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @staticmethod
    def _valid_adjacency(left: K, right: K) -> bool:
        if left == right and left in HARD_MANEUVERS:
            return False
        if left in {K.CLIMB, K.DIVE} and right == left:
            return False
        if left in {K.ACCELERATION, K.STRAIGHT} and right == left:
            return False
        return True

    def _spec(self, kind: K, rng: np.random.Generator) -> ManeuverSpec:
        return ManeuverSpec(
            kind=kind,
            length_scale=float(rng.uniform(0.82, 1.22)),
            vertical_scale=float(rng.uniform(0.78, 1.25)),
            severity=float(rng.uniform(0.25, 0.95)),
        )

    def sample_program(
        self, *, seed: int, split: str,
        forced_ngram: Sequence[K] | None = None,
    ) -> CourseProgram:
        rng = np.random.default_rng(int(seed))
        count = int(rng.integers(self.config.minimum_macros, self.config.maximum_macros + 1))
        forced = tuple(forced_ngram or ())
        if len(forced) > count:
            raise ValueError("forced n-gram is longer than the sampled program")
        values: list[K] = list(forced)
        if not values:
            values.extend([
                self._easy[int(rng.integers(len(self._easy)))],
                self._planar[int(rng.integers(len(self._planar)))],
                self._vertical[int(rng.integers(len(self._vertical)))],
                self._hard[int(rng.integers(len(self._hard)))],
            ])
        pool = self._easy + self._planar + self._vertical + self._hard
        while len(values) < count:
            weights = np.asarray([
                1.4 if candidate in self._easy else
                1.2 if candidate in self._planar else 0.8
                for candidate in pool
            ], np.float64)
            weights /= weights.sum()
            for _ in range(100):
                candidate = pool[int(rng.choice(len(pool), p=weights))]
                if self._valid_adjacency(values[-1], candidate):
                    values.append(candidate)
                    break
            else:
                raise RuntimeError("maneuver grammar could not find a valid continuation")
        if not any(item in HARD_MANEUVERS for item in values):
            values[-1] = self._hard[int(rng.integers(len(self._hard)))]
        if not any(item in VERTICAL_MANEUVERS for item in values):
            values[-2] = self._vertical[int(rng.integers(len(self._vertical)))]
        if not any(item in {K.STRAIGHT, K.ACCELERATION} for item in values):
            # Preserve a forced n-gram at the front of the pre-rotation list.
            # There is always at least one sampled continuation because the
            # shortest program is six macros and obligations are length three.
            values[-1] = K.ACCELERATION
        # Rotate rather than shuffle so a forced composition remains intact.
        shift = int(rng.integers(len(values)))
        values = values[shift:] + values[:shift]
        if split != "composition_holdout":
            holdouts = {
                tuple(item.value for item in gram)
                for gram in COMPOSITION_HOLDOUT_TRIGRAMS
            }
            # Keep the compositional probe scientifically disjoint even when
            # a random program happens to sample one of its defining triples.
            # Replacing the middle with a dive retains a vertical challenge and
            # leaves the hard primitive at either edge of every holdout triple.
            for _ in range(len(values)):
                grams = cyclic_ngrams(values, 3)
                collision = next((i for i, gram in enumerate(grams) if gram in holdouts), None)
                if collision is None:
                    break
                values[(collision + 1) % len(values)] = K.DIVE
            else:
                raise RuntimeError("could not remove composition-holdout leakage")
        specs = tuple(self._spec(item, rng) for item in values)
        return CourseProgram(
            maneuvers=specs, seed=int(seed), split=split,
            backend=GeneratorBackend.MANEUVER_GRAMMAR,
            program_id=self._program_id(specs, seed, split),
        )

    def coverage_programs(
        self, *, count: int, seed: int, split: str,
    ) -> tuple[CourseProgram, ...]:
        """Greedy max-coverage schedule over explicitly required n-grams."""

        if count < 1:
            raise ValueError("program count must be positive")
        order = self.config.required_ngram_order
        coverage_order = order
        obligations = set(required_ngrams(order))
        candidates: list[CourseProgram] = []
        forced_pool: Sequence[Sequence[K]] = tuple(sorted(
            REQUIRED_TRIGRAMS if order == 3 else REQUIRED_BIGRAMS,
            key=lambda gram: tuple(item.value for item in gram),
        ))
        if split == "composition_holdout":
            forced_pool = COMPOSITION_HOLDOUT_TRIGRAMS
            coverage_order = 3
            obligations = {
                tuple(item.value for item in gram)
                for gram in COMPOSITION_HOLDOUT_TRIGRAMS
            }
        candidate_count = max(count * 24, len(forced_pool) * 3)
        for index in range(candidate_count):
            forced = forced_pool[index % len(forced_pool)] if forced_pool else None
            candidates.append(self.sample_program(
                seed=int(seed + 104729 * index), split=split, forced_ngram=forced,
            ))

        selected: list[CourseProgram] = []
        covered: set[tuple[str, ...]] = set()
        kind_counts: Counter[str] = Counter()
        while candidates and len(selected) < count:
            def score(item: CourseProgram) -> tuple[float, float, str]:
                grams = set(cyclic_ngrams(item.kinds, coverage_order))
                gain = len((grams & obligations) - covered)
                rare = sum(1.0 / (1.0 + kind_counts[kind.value]) for kind in item.kinds)
                return float(gain), float(rare), item.program_id
            best = max(candidates, key=score)
            candidates.remove(best)
            selected.append(best)
            covered.update(set(cyclic_ngrams(best.kinds, coverage_order)) & obligations)
            kind_counts.update(item.value for item in best.kinds)
        # Give every selected program a stable split-local seed lineage.  The
        # realization seed remains the original candidate seed.
        return tuple(replace(item, split=split) for item in selected)


def coverage_fraction(programs: Sequence[CourseProgram], order: int) -> float:
    obligations = required_ngrams(order)
    observed = {
        gram for program in programs for gram in cyclic_ngrams(program.kinds, order)
    }
    return float(len(observed & obligations) / max(len(obligations), 1))
