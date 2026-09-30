"""Compare the matched augmented two-parent DAgger baseline arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("success", "timely_success", "clean_timely_success")


def read_events(kind: str, *, full: bool = False) -> list[dict]:
    suffix = ".full.events.jsonl" if full else ".events.jsonl"
    path = ROOT / "outputs/logs" / f"starscream-v6.21.1.1-mini-slalom-aug2-{kind}-r24{suffix}"
    if not path.exists():
        return []
    rows = [json.loads(line)["metrics"] for line in path.read_text().splitlines()]
    return [row for row in rows if "eval/selection_suite_success" in row]


def track_rows(row: dict) -> list[dict]:
    names = sorted(
        key.split("/")[2]
        for key in row
        if key.startswith("eval/track/") and key.endswith("/full_course_success") and key.count("/") == 3
    )
    result = []
    for name in names:
        base = f"eval/track/{name}/"
        median_steps = row.get(base + "successful_median_steps")
        if isinstance(median_steps, float) and not math.isfinite(median_steps):
            median_steps = None
        result.append(
            {
                "track": name,
                "success": row.get(base + "full_course_success"),
                "timely_success": row.get(base + "timely_success"),
                "clean_timely_success": row.get(base + "clean_timely_success"),
                "recovered_success": row.get(base + "recovered_success"),
                "successful_median_steps": median_steps,
            }
        )
    return result


def report(kind: str, window: int) -> dict:
    rows = read_events(kind)
    full_rows = read_events(kind, full=True)
    if not rows:
        return {"kind": kind, "status": "no evaluation yet"}
    initial = rows[0]
    latest = rows[-1]
    recent = rows[max(1, len(rows) - window) :]
    return {
        "kind": kind,
        "rounds_evaluated": len(rows) - 1,
        "steps": int(latest["eval/step"]),
        "initial": {metric: initial.get(f"eval/selection_suite_{metric}") for metric in METRICS},
        "latest": {metric: latest.get(f"eval/selection_suite_{metric}") for metric in METRICS},
        "recent_mean": {
            metric: sum(row[f"eval/selection_suite_{metric}"] for row in recent) / len(recent)
            for metric in METRICS
        }
        if recent
        else None,
        "latest_tracks": track_rows(full_rows[-1]) if full_rows else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    print(json.dumps([report(kind, args.window) for kind in ("broad", "bounded")], indent=2))


if __name__ == "__main__":
    main()
