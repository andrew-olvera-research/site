#!/usr/bin/env python3
"""Fail fast when the mounted Agilicious checkout is missing or docs-only."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED = (
    "agilib/CMakeLists.txt",
    "agiros/CMakeLists.txt",
    "agiros/package.xml",
)


def verify(root: Path) -> list[str]:
    return [relative for relative in REQUIRED if not (root / relative).is_file()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=Path("/catkin_ws/src/agilicious"))
    args = parser.parse_args()
    root = args.path.resolve()
    missing = verify(root)
    if missing:
        details = "\n  - ".join(missing)
        raise SystemExit(
            f"Agilicious controller sources are unavailable at {root}.\n"
            f"Missing:\n  - {details}\n"
            "The public uzh-rpg/agilicious checkout is documentation-only. "
            "Mount the complete UZH-licensed checkout (including initialized "
            "submodules) at third_party/agilicious."
        )
    print(f"Agilicious source checkout verified: {root}")


if __name__ == "__main__":
    main()
