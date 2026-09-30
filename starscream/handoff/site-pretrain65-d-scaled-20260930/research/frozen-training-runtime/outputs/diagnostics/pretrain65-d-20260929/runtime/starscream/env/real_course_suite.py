"""Immutable registry loader for reportable real-course evaluations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .procedural_tracks import geometry_fingerprint
from .tracks import load_track


def load_active_real_course_suite(path: str | Path) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Resolve active exact courses and verify their declared geometry hashes."""

    suite_path = Path(path).resolve()
    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    if payload.get("schema") != "starscream-real-course-suite-v1":
        raise ValueError(f"unsupported real-course suite schema in {suite_path}")
    active = list(payload.get("active", ()))
    if not active:
        raise ValueError(f"real-course suite has no active tracks: {suite_path}")
    resolved: list[str] = []
    names: set[str] = set()
    for entry in active:
        name = str(entry.get("name", ""))
        if not name or name in names:
            raise ValueError(f"invalid or duplicate active course name {name!r}")
        names.add(name)
        raw = Path(str(entry.get("track", "")))
        candidates = [raw] if raw.is_absolute() else [
            Path.cwd() / raw,
            suite_path.parents[2] / raw if len(suite_path.parents) >= 3
            else suite_path.parent / raw,
        ]
        track_path = next(
            (candidate.resolve() for candidate in candidates if candidate.is_file()), None
        )
        if track_path is None:
            raise FileNotFoundError(f"active real-course geometry not found for {name}: {raw}")
        expected = str(entry.get("geometry_fingerprint", ""))
        if not expected:
            raise ValueError(f"active real course {name} has no immutable geometry fingerprint")
        actual = geometry_fingerprint(load_track(track_path))
        if actual != expected:
            raise ValueError(
                f"real-course geometry drift for {name}: expected {expected}, got {actual}"
            )
        resolved.append(str(track_path))
    report = {
        "suite": str(suite_path),
        "active_names": sorted(names),
        "active_count": len(resolved),
        "pending_count": len(payload.get("pending_geometry", ())),
    }
    return tuple(resolved), report
