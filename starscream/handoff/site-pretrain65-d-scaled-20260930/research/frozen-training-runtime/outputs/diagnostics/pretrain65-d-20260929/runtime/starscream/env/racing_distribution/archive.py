"""Quality-diversity archive for racing-task selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .descriptors import CourseDescriptor
from .schema import RacingDistributionConfig


STATIC_ARCHIVE_FIELDS: tuple[str, ...] = (
    "length_m", "vertical_excursion_m", "p95_turn_degrees",
    "gate_density_per_100m", "p99_curvature_m_inv",
)
DYNAMIC_ARCHIVE_FIELDS: tuple[str, ...] = (
    "qualified_speed_mps", "lateral_demand_ratio",
)


class DescriptorGrid:
    """Discretize course descriptors without depending on benchmark data."""

    def __init__(
        self,
        config: RacingDistributionConfig | None = None,
        *,
        fields: Sequence[str] | None = None,
    ) -> None:
        cfg = config or RacingDistributionConfig()
        self.edges = {
            str(name): tuple(float(value) for value in values)
            for name, values in cfg.qd_edges.items()
        }
        self.fields = tuple(fields or STATIC_ARCHIVE_FIELDS)
        unknown = set(self.fields) - set(self.edges)
        if unknown:
            raise ValueError(f"descriptor grid has unknown fields: {sorted(unknown)}")

    def key(self, descriptor: CourseDescriptor | Mapping[str, float]) -> tuple[int, ...]:
        values = descriptor.values if isinstance(descriptor, CourseDescriptor) else descriptor
        return tuple(
            int(np.searchsorted(self.edges[name], float(values.get(name, 0.0)), side="right"))
            for name in self.fields
        )

    @property
    def cell_count(self) -> int:
        return int(np.prod([len(self.edges[name]) + 1 for name in self.fields]))

    def all_keys(self) -> Iterable[tuple[int, ...]]:
        return product(*(range(len(self.edges[name]) + 1) for name in self.fields))

    def key_mapping(self, key: Sequence[int]) -> dict[str, int]:
        if len(key) != len(self.fields):
            raise ValueError("descriptor-grid key length mismatch")
        return {name: int(value) for name, value in zip(self.fields, key)}


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    task_id: str
    descriptor: CourseDescriptor
    quality: float
    payload: Any = None


class QualityDiversityArchive:
    """One deterministic highest-quality task per occupied descriptor cell."""

    def __init__(self, grid: DescriptorGrid | None = None) -> None:
        self.grid = grid or DescriptorGrid()
        self._cells: dict[tuple[int, ...], ArchiveEntry] = {}

    def admit(self, entry: ArchiveEntry) -> bool:
        if not np.isfinite(entry.quality):
            raise ValueError("archive quality must be finite")
        key = self.grid.key(entry.descriptor)
        current = self._cells.get(key)
        rank = (float(entry.quality), entry.task_id)
        if current is not None and rank <= (float(current.quality), current.task_id):
            return False
        self._cells[key] = entry
        return True

    def extend(self, entries: Iterable[ArchiveEntry]) -> int:
        return sum(self.admit(entry) for entry in entries)

    @property
    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(
            self._cells[key] for key in sorted(self._cells)
        )

    @property
    def occupied_cells(self) -> int:
        return len(self._cells)

    @property
    def occupancy_fraction(self) -> float:
        return float(self.occupied_cells / max(self.grid.cell_count, 1))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "fields": list(self.grid.fields),
            "cell_count": self.grid.cell_count,
            "occupied_cells": self.occupied_cells,
            "occupancy_fraction": self.occupancy_fraction,
            "entries": [
                {
                    "task_id": entry.task_id,
                    "quality": float(entry.quality),
                    "cell": self.grid.key_mapping(self.grid.key(entry.descriptor)),
                }
                for entry in self.entries
            ],
        }
