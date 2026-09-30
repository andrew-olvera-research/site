"""Cell-directed online course bank for fixed-window PPO (v6.21).

The bank keeps PPO in the regime that produced the rl.5 gain: courses that are neither
solved nor hopeless. Every course carries its transition cells; each window the bank
turns per-course success EMAs into per-cell competence, retires courses whose cells are
all solved (they keep a floor lane, never zero), and generates new screened courses for
frontier cells that too few active courses cover. Anchor courses (the frozen training
corpus) are never retired and act as the drift guard: if their mean competence falls
by more than a tolerance from its running maximum, generation pauses and anchors are
rehearsed at a higher multiplier until they recover. Generated geometry is never a copy
of a real track and is anti-clone checked against the bank and the held-out suite.

Pure logic lives here; the trainer supplies competence arrays and applies lane weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .env.procedural_tracks import geometry_fingerprint, save_track_yaml
from .env.racing_manifold.corpus_coverage import clone_distance
from .env.racing_manifold.transition_cells import describe_cell, transition_cells
from .env.racing_manifold.transition_corpus_v21 import GRAMMARS, MOTIF_GRAMMARS, generate_for_cells
from .env.tracks import load_track

DEFAULTS = dict(
    enabled=False,
    frontier_low=0.25,            # cell competence band that counts as learnable
    frontier_high=0.85,
    min_active_support=3,         # active courses per frontier cell before generating
    max_new_per_window=2,
    max_bank_size=160,
    continual_replacement=False,  # when true, retired courses release active capacity
    max_total_bank_size=None,     # optional hard bound on retained active + retired history
    generation_start_window=6,    # let the competence EMAs settle first
    generation_attempts=200,
    grammars=list(GRAMMARS),
    counts=[5, 6, 7, 8, 9, 10],
    retire_competence=0.95,
    retire_patience=6,            # consecutive windows above retire_competence
    retired_multiplier=0.05,      # relative lane weight of a retired course (never zero)
    anchor_drop_tolerance=0.10,   # pause generation if anchor mean EMA falls this far from its max
    anchor_rehearsal_multiplier=2.0,
    anchor_recovery_margin=0.03,
    stage_warmup_windows=5,       # establish a new command-regime baseline after collector reset
    clone_distance=0.12,
    raceability='static',         # 'static' | 'nominal_mpcc' (async subprocess screen)
    screen_timeout_windows=8,
)


class OnlineCourseBank:
    def __init__(self, config: Mapping[str, Any], *, anchors: Sequence[str], bank_dir: str | Path,
                 protected_tracks: Sequence[str] = (), seed: int = 0,
                 screen_launcher: Callable[[str, Path], Any] | None = None) -> None:
        self.config = {**DEFAULTS, **dict(config)}
        c = self.config
        if not 0.0 <= c['frontier_low'] < c['frontier_high'] <= 1.0:
            raise ValueError('frontier band must satisfy 0 <= low < high <= 1')
        if c['max_bank_size'] < len(anchors) or c['max_new_per_window'] < 0 or c['retire_patience'] < 1:
            raise ValueError('invalid online course bank settings')
        total_limit = c.get('max_total_bank_size')
        if total_limit is not None and int(total_limit) < int(c['max_bank_size']):
            raise ValueError('max_total_bank_size must be at least max_bank_size')
        if not 0.0 < c['retired_multiplier'] <= 1.0 or c['anchor_rehearsal_multiplier'] < 1.0:
            raise ValueError('retired multiplier must be in (0,1] and rehearsal multiplier >= 1')
        if c['raceability'] not in ('static', 'nominal_mpcc'):
            raise ValueError("raceability must be 'static' or 'nominal_mpcc'")
        if int(c['stage_warmup_windows']) < 1:
            raise ValueError('stage_warmup_windows must be positive')
        self.bank_dir = Path(bank_dir)
        self.rng = np.random.default_rng(int(seed) ^ 0x42414E4B)
        self.courses: list[dict[str, Any]] = []
        self._tracks: dict[str, Any] = {}
        self._protected = [load_track(p) for p in protected_tracks]
        for path in anchors:
            self._register(path, origin='anchor')
        self.pending: list[dict[str, Any]] = []      # courses awaiting the async raceability screen
        self.screen_launcher = screen_launcher
        self.anchor_max = 0.0
        self.paused = False
        self.generated_total = 0
        self.retired_total = 0
        self.windows = 0
        self.stage_context: str | None = None
        self.stage_windows = 0

    def begin_stage(self, context: str, *, reset_competence: bool = False) -> None:
        """Keep geometry, but never compare fresh EMA priors to another stage's peak.

        Same-stage checkpoint restoration preserves the guard and its warmup.
        Legacy checkpoints can adopt a context without erasing a valid guard;
        the trainer explicitly requests reset when sampling state is missing.
        """
        context = str(context)
        changed = self.stage_context is not None and context != self.stage_context
        if changed or reset_competence:
            self.anchor_max = 0.0
            self.paused = False
            self.stage_windows = 0
            # Solved under an easier command does not imply solved here.
            for rec in self.courses:
                rec['above'] = 0
                rec['retired'] = False
        elif self.stage_context is None:
            self.stage_windows = self.windows
        self.stage_context = context

    # ----- registry -----------------------------------------------------------------
    def _register(self, path: str, *, origin: str, cells: set[str] | None = None, cycle: int = -1) -> dict[str, Any]:
        track = load_track(path)
        rec = dict(name=track.name, path=str(path), origin=origin, added_cycle=int(cycle), retired=False,
                   above=0, cells=sorted(cells if cells is not None else transition_cells(track)),
                   fingerprint=geometry_fingerprint(track))
        self.courses.append(rec)
        self._tracks[rec['name']] = track
        return rec

    @property
    def paths(self) -> list[str]:
        return [c['path'] for c in self.courses]

    def anchor_mask(self) -> np.ndarray:
        return np.asarray([c['origin'] == 'anchor' for c in self.courses], bool)

    def retired_mask(self) -> np.ndarray:
        return np.asarray([c['retired'] for c in self.courses], bool)

    # ----- competence bookkeeping --------------------------------------------------
    def cell_competence(self, competence: np.ndarray) -> dict[str, tuple[float, int]]:
        """cell -> (mean success EMA over active courses containing it, active support)."""
        sums: dict[str, list[float]] = {}
        for rec, value in zip(self.courses, competence):
            if rec['retired'] or not np.isfinite(value):
                continue
            for cell in rec['cells']:
                sums.setdefault(cell, []).append(float(value))
        return {cell: (float(np.mean(v)), len(v)) for cell, v in sums.items()}

    def frontier_targets(self, cell_comp: Mapping[str, tuple[float, int]]) -> dict[str, float]:
        """Weight = frontier closeness x support deficit, for cells with too few active courses."""
        c = self.config
        targets: dict[str, float] = {}
        mid = 0.5 * (c['frontier_low'] + c['frontier_high'])
        half = 0.5 * (c['frontier_high'] - c['frontier_low'])
        for cell, (value, support) in cell_comp.items():
            if not c['frontier_low'] <= value <= c['frontier_high'] or support >= c['min_active_support']:
                continue
            closeness = 1.0 - abs(value - mid) / half
            targets[cell] = (0.25 + 0.75 * closeness) * (c['min_active_support'] - support)
        return targets

    def _is_clone(self, track) -> bool:
        limit = float(self.config['clone_distance'])
        for other in list(self._tracks.values()) + self._protected:
            if len(other.gates) != len(track.gates):
                continue
            d = clone_distance(track, other)
            if d is not None and d < limit:
                return True
        return False

    # ----- per-window step -----------------------------------------------------------
    def step(self, competence: np.ndarray, cycle: int) -> tuple[list[str], np.ndarray, dict[str, float]]:
        """Return (new track paths to add, lane multiplier per existing course, metrics)."""
        c = self.config
        competence = np.asarray(competence, np.float64)
        if competence.shape != (len(self.courses),):
            raise ValueError('competence must align with the bank')
        self.windows += 1
        self.stage_windows += 1
        calibrating = (self.stage_context is not None and
                       self.stage_windows <= int(c['stage_warmup_windows']))
        # retirement: every cell of the course already solved by the course itself
        for rec, value in zip(self.courses, competence):
            if calibrating or rec['origin'] == 'anchor' or not np.isfinite(value):
                continue
            rec['above'] = rec['above'] + 1 if value >= c['retire_competence'] else 0
            if not rec['retired'] and rec['above'] > c['retire_patience']:
                rec['retired'] = True
                self.retired_total += 1
        # drift guard on anchors
        anchor = self.anchor_mask()
        anchor_values = competence[anchor & np.isfinite(competence)]
        anchor_mean = float(anchor_values.mean()) if len(anchor_values) else float('nan')
        if np.isfinite(anchor_mean):
            if calibrating:
                self.anchor_max = anchor_mean
                self.paused = False
            else:
                self.anchor_max = max(self.anchor_max, anchor_mean)
            if not calibrating and self.paused:
                self.paused = anchor_mean < self.anchor_max - c['anchor_drop_tolerance'] + c['anchor_recovery_margin']
            elif not calibrating:
                self.paused = anchor_mean < self.anchor_max - c['anchor_drop_tolerance']
        multipliers = np.ones(len(self.courses), np.float64)
        multipliers[self.retired_mask()] = c['retired_multiplier']
        if self.paused:
            multipliers[anchor] *= c['anchor_rehearsal_multiplier']
        # promote screened courses, then generate for frontier cells
        added = self._collect_screened(cycle)
        cell_comp = self.cell_competence(competence)
        targets = self.frontier_targets(cell_comp)
        generated = 0
        active_count = int((~self.retired_mask()).sum())
        if bool(c.get('continual_replacement', False)):
            # Retired courses remain in the collector at their rehearsal floor,
            # but no longer permanently occupy the generation budget.  Keeping
            # them registered avoids unsafe collector/index surgery mid-run.
            capacity = int(c['max_bank_size']) - active_count - len(self.pending)
            total_limit = c.get('max_total_bank_size')
            if total_limit is not None:
                capacity = min(
                    capacity,
                    int(total_limit) - len(self.courses) - len(self.pending),
                )
        else:
            capacity = int(c['max_bank_size']) - len(self.courses) - len(self.pending)
        capacity = max(capacity, 0)
        if (not calibrating and not self.paused and self.windows >= c['generation_start_window'] and targets
                and c['max_new_per_window'] > 0 and capacity > 0):
            for _ in range(min(c['max_new_per_window'], capacity)):
                track, covered, score = generate_for_cells(
                    targets, int(self.rng.integers(0, 2**31 - 1)), counts=tuple(c['counts']),
                    grammars=tuple(c['grammars']), attempts=int(c['generation_attempts']),
                    name=f'bank_c{cycle}', reject=self._is_clone)
                if track is None:
                    break
                self.bank_dir.mkdir(parents=True, exist_ok=True)
                path = save_track_yaml(track, self.bank_dir / f'{track.name}.yaml')
                self.generated_total += 1
                generated += 1
                for cell in covered:
                    targets[cell] = targets[cell] * 0.5    # diminish, so a second course targets other cells
                if c['raceability'] == 'nominal_mpcc' and self.screen_launcher is not None:
                    handle = self.screen_launcher(str(path), self.bank_dir / f'{track.name}.screen.json')
                    self.pending.append(dict(path=str(path), handle=handle, cycle=cycle, covered=sorted(covered)))
                else:
                    self._register(str(path), origin='generated', cycle=cycle)
                    added.append(str(path))
        frontier_count = sum(1 for v, _ in cell_comp.values() if c['frontier_low'] <= v <= c['frontier_high'])
        metrics = dict(
            bank_size=float(len(self.courses)), bank_active=float((~self.retired_mask()).sum()),
            bank_generated=float(self.generated_total), bank_generated_this_window=float(generated),
            bank_pending_screen=float(len(self.pending)), bank_retired=float(self.retired_total),
            bank_frontier_cells=float(frontier_count), bank_target_cells=float(len(targets)),
            bank_anchor_competence=anchor_mean, bank_anchor_max=float(self.anchor_max),
            bank_paused=float(self.paused), bank_cells_tracked=float(len(cell_comp)),
            bank_stage_windows=float(self.stage_windows), bank_stage_calibrating=float(calibrating),
            bank_generation_capacity=float(capacity),
            bank_continual_replacement=float(bool(c.get('continual_replacement', False))),
        )
        return added, multipliers, metrics

    def _collect_screened(self, cycle: int) -> list[str]:
        added: list[str] = []
        keep: list[dict[str, Any]] = []
        for item in self.pending:
            handle = item['handle']
            result = handle.poll() if hasattr(handle, 'poll') else handle
            if result is None and cycle - item['cycle'] < self.config['screen_timeout_windows']:
                keep.append(item)
                continue
            report_path = Path(item['path']).with_suffix('.screen.json')
            passed = False
            if report_path.exists():
                try:
                    passed = bool(json.loads(report_path.read_text()).get('qualified', False))
                except json.JSONDecodeError:
                    passed = False
            if passed:
                self._register(item['path'], origin='generated', cycle=cycle)
                added.append(item['path'])
        self.pending = keep
        return added

    # ----- checkpointing -----------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return dict(
            courses=[{k: v for k, v in rec.items()} for rec in self.courses],
            anchor_max=self.anchor_max, paused=self.paused, generated_total=self.generated_total,
            retired_total=self.retired_total, windows=self.windows,
            stage_context=self.stage_context, stage_windows=self.stage_windows,
            rng_state=self.rng.bit_generator.state,
            pending=[dict(path=p['path'], cycle=p['cycle'], covered=p['covered']) for p in self.pending],
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> list[str]:
        """Restore; returns generated track paths that must be re-added to the collector."""
        anchors = [rec['path'] for rec in self.courses if rec['origin'] == 'anchor']
        restored_anchors = [rec['path'] for rec in state['courses'] if rec['origin'] == 'anchor']
        if anchors != restored_anchors:
            raise ValueError('online course bank anchors differ from the checkpoint')
        self.courses = [dict(rec) for rec in state['courses']]
        self._tracks = {}
        for rec in self.courses:
            if not Path(rec['path']).is_file():
                raise FileNotFoundError(rec['path'])
            track = load_track(rec['path'])
            if geometry_fingerprint(track) != rec['fingerprint']:
                raise ValueError(f'bank geometry drift for {rec["name"]}')
            self._tracks[rec['name']] = track
        self.anchor_max = float(state['anchor_max']); self.paused = bool(state['paused'])
        self.stage_context = state.get('stage_context')
        self.stage_windows = int(state.get('stage_windows', state['windows']))
        self.generated_total = int(state['generated_total']); self.retired_total = int(state['retired_total'])
        self.windows = int(state['windows'])
        self.rng.bit_generator.state = state['rng_state']
        self.pending = []   # a pending screen does not survive a restart; its course is simply not admitted
        return [rec['path'] for rec in self.courses if rec['origin'] != 'anchor']

    def describe_targets(self, targets: Mapping[str, float], limit: int = 8) -> list[str]:
        return [f'{describe_cell(cell)} ({weight:.2f})' for cell, weight in sorted(targets.items(), key=lambda kv: -kv[1])[:limit]]


def subprocess_screen_launcher(script: str = 'scripts/audits/screen_course_nominal.py',
                               python: str | None = None) -> Callable[[str, Path], Any]:
    """Launch the nominal-MPCC raceability screen out of process; returns a Popen handle."""
    import os
    import subprocess
    import sys

    def launch(track_path: str, report_path: Path):
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        log = Path(report_path).with_suffix('.log')
        return subprocess.Popen(
            [python or sys.executable, '-u', script, '--track', str(track_path), '--out', str(report_path)],
            stdout=log.open('w'), stderr=subprocess.STDOUT, env=env, cwd=os.getcwd(),
        )
    return launch
