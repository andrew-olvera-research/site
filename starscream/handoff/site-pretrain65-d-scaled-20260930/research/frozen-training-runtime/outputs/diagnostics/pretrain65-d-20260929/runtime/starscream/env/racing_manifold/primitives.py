"""Gate-frame and maneuver primitives orthogonal to traversal geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence

import numpy as np

from ..tracks import Gate, Track, matrix_quaternion


PrimitiveKind = Literal["inverted_gate", "split_s", "slalom", "corkscrew"]


@dataclass(frozen=True, slots=True)
class PrimitiveProfile:
    name: str
    labels: tuple[str, ...]
    maximum_roll_degrees: float
    inverted_gate_fraction: float
    high_roll_gate_fraction: float
    vertical_reversal_count: int
    labeled_counts: dict[str, int]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "labels": list(self.labels),
            "maximum_roll_degrees": self.maximum_roll_degrees,
            "inverted_gate_fraction": self.inverted_gate_fraction,
            "high_roll_gate_fraction": self.high_roll_gate_fraction,
            "vertical_reversal_count": self.vertical_reversal_count,
            "labeled_counts": dict(self.labeled_counts),
        }


def _signed_gate_roll(gate: Gate) -> float:
    normal = np.asarray(gate.normal, np.float64)
    up = np.asarray(gate.up, np.float64)
    world_up = np.asarray([0.0, 0.0, 1.0])
    reference = world_up - normal * float(world_up @ normal)
    if float(np.linalg.norm(reference)) < 1.0e-6:
        reference = np.asarray([0.0, 1.0, 0.0])
        reference -= normal * float(reference @ normal)
    reference /= np.linalg.norm(reference)
    up -= normal * float(up @ normal)
    up /= max(float(np.linalg.norm(up)), 1.0e-12)
    sine = float(normal @ np.cross(reference, up))
    cosine = float(np.clip(reference @ up, -1.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))


def analyze_primitives(track: Track) -> PrimitiveProfile:
    metadata = track.metadata or {}
    raw_labels = metadata.get("primitive_labels", ())
    labels = tuple(str(item) for item in raw_labels)
    rolls = np.abs(np.asarray([_signed_gate_roll(gate) for gate in track.gates]))
    points = np.stack([gate.position for gate in track.gates]).astype(np.float64)
    vertical = np.roll(points[:, 2], -1) - points[:, 2]
    active = np.sign(vertical[np.abs(vertical) >= 0.35])
    reversal_count = int(np.sum(active[1:] != active[:-1])) if len(active) > 1 else 0
    counts: dict[str, int] = {}
    for label in labels:
        root = label.split("_")[0] if label else "unlabeled"
        counts[root] = counts.get(root, 0) + 1
    return PrimitiveProfile(
        name=track.name,
        labels=labels,
        maximum_roll_degrees=float(rolls.max(initial=0.0)),
        inverted_gate_fraction=float(np.mean(rolls >= 145.0)),
        high_roll_gate_fraction=float(np.mean(rolls >= 80.0)),
        vertical_reversal_count=reversal_count,
        labeled_counts=counts,
    )


def _roll_gate(gate: Gate, degrees: float) -> Gate:
    angle = np.radians(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    normal = np.asarray(gate.physical_normal, np.float64)
    skew = np.asarray([
        [0.0, -normal[2], normal[1]],
        [normal[2], 0.0, -normal[0]],
        [-normal[1], normal[0], 0.0],
    ])
    rotation = np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)
    return replace(gate, quaternion_wxyz=matrix_quaternion(rotation @ gate.rotation))


def graft_primitive(
    track: Track,
    kind: PrimitiveKind,
    *,
    anchor: int,
    severity: float = 0.75,
    name: str | None = None,
) -> Track:
    """Graft one local primitive while preserving the rest of a course.

    ``inverted_gate`` changes only the gate frame. Split-S and corkscrew also
    move a four-gate window. They are curriculum candidates and still require
    MPCC/closed-loop qualification before admission.
    """

    if not 0.0 < severity <= 1.0:
        raise ValueError("severity must be in (0, 1]")
    count = len(track.gates)
    if count < 4:
        raise ValueError("primitive grafting needs at least four gates")
    anchor %= count
    gates = list(track.gates)
    labels = list((track.metadata or {}).get("primitive_labels", ["base"] * count))
    if len(labels) != count:
        labels = ["base"] * count
    if kind == "inverted_gate":
        gates[anchor] = _roll_gate(gates[anchor], 180.0 * severity)
        labels[anchor] = "inverted_gate"
    else:
        indices = [(anchor + offset) % count for offset in range(4)]
        positions = np.stack([gates[index].position for index in indices]).astype(np.float64)
        tangent = positions[-1] - positions[0]
        tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
        up = np.asarray([0.0, 0.0, 1.0])
        lateral = np.cross(up, tangent)
        if float(np.linalg.norm(lateral)) < 1.0e-6:
            lateral = np.asarray([0.0, 1.0, 0.0])
        lateral /= np.linalg.norm(lateral)
        local_up = np.cross(tangent, lateral)
        local_up /= np.linalg.norm(local_up)
        if kind == "split_s":
            lateral_shape = np.asarray([0.0, 0.65, 0.75, 0.0])
            vertical_shape = np.asarray([0.0, 1.20, -1.35, 0.0])
            rolls = (30.0, 92.0, 172.0, 180.0)
            motif_labels = ("split_s_entry", "split_s_apex", "split_s_descent", "split_s_exit")
        elif kind == "corkscrew":
            lateral_shape = np.asarray([0.0, 0.85, -0.85, 0.0])
            vertical_shape = np.asarray([0.0, 0.85, 0.85, 0.0])
            rolls = (20.0, 70.0, 135.0, 195.0)
            motif_labels = ("corkscrew_entry", "corkscrew_quarter", "corkscrew_half", "corkscrew_exit")
        elif kind == "slalom":
            # A local left-right chain.  Unlike the split-S this is primarily
            # a traversal primitive: gate roll and vertical motion remain
            # small so low-severity grafts stay inside the source manifold.
            lateral_shape = np.asarray([0.0, 0.90, -0.90, 0.0])
            vertical_shape = np.asarray([0.0, 0.08, -0.08, 0.0])
            rolls = (0.0, 12.0, -12.0, 0.0)
            motif_labels = (
                "slalom_entry", "slalom_left", "slalom_right", "slalom_exit",
            )
        else:
            raise ValueError(f"unsupported primitive {kind!r}")
        scale = severity * max(1.5, 0.14 * float(np.linalg.norm(positions[-1] - positions[0])))
        for offset, index in enumerate(indices):
            displacement = scale * (
                lateral_shape[offset] * lateral + vertical_shape[offset] * local_up
            )
            gates[index] = _roll_gate(
                replace(gates[index], position=(positions[offset] + displacement).astype(np.float32)),
                severity * rolls[offset],
            )
            labels[index] = motif_labels[offset]
    points = np.stack([gate.position for gate in gates]).astype(np.float64)
    bounds = np.stack([points.min(axis=0) - 4.0, points.max(axis=0) + 4.0], axis=1).astype(np.float32)
    bounds[2, 0] = min(float(bounds[2, 0]), 0.0)
    metadata = dict(track.metadata or {})
    metadata.update({
        "primitive_graft": {"kind": kind, "anchor": anchor, "severity": severity},
        "primitive_labels": labels,
        "primitive_source_track": track.name,
        "primitive_source_fingerprint": track.fingerprint,
    })
    return Track(
        name=name or f"{track.name}_{kind}_{anchor:02d}", gates=tuple(gates),
        bounds=bounds, loop=track.loop, metadata=metadata,
    )


def primitive_coverage(profiles: Sequence[PrimitiveProfile]) -> dict[str, Any]:
    labels = sorted({label for profile in profiles for label in profile.labels if label != "base"})
    return {
        "track_count": len(profiles),
        "labels": labels,
        "label_count": len(labels),
        "maximum_roll_degrees": max((profile.maximum_roll_degrees for profile in profiles), default=0.0),
        "maximum_vertical_reversals": max((profile.vertical_reversal_count for profile in profiles), default=0),
        "has_inverted_gate": any(
            profile.inverted_gate_fraction > 0.0
            or "inverted_gate" in profile.labels
            for profile in profiles
        ),
        "has_split_s": any(any(label.startswith("split_s") for label in profile.labels) for profile in profiles),
        "has_slalom": any(any(label.startswith("slalom") for label in profile.labels) for profile in profiles),
        "has_corkscrew": any(any(label.startswith("corkscrew") for label in profile.labels) for profile in profiles),
    }
