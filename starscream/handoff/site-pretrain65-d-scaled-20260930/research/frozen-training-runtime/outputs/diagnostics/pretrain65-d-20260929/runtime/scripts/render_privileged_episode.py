#!/usr/bin/env python3
"""Roll out a privileged racing policy and render an auditable episode video."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_privileged_racing import (
    CausalHistory, episode_target_speed, load_config, make_env, make_reward,
    parse_stage, ppo_normalized_to_ctbr, ppo_observation_features, reset_env,
)
from starscream.env.tracks import quaternion_matrix
from starscream.privileged_racing import load_policy_checkpoint


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("dagger", "ppo"), default="dagger",
        help="Configuration section whose evaluation curriculum and environment settings are rendered.",
    )
    parser.add_argument("--track-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026089900)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--curriculum", choices=("evaluation", "reporting"), default="evaluation",
        help="Render the selection evaluation or held-out reporting curriculum.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def _equal_axes(ax: object, points: np.ndarray) -> None:
    low, high = points.min(0), points.max(0)
    center = (low + high) * 0.5
    radius = max(float((high - low).max()) * 0.58, 3.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius * 0.55), center[2] + radius * 0.55)
    ax.set_box_aspect((1, 1, 0.55))


def _gate_corners(gate: object) -> np.ndarray:
    half_y, half_z = np.asarray(gate.size) * 0.5
    return gate.position + np.stack([
        -half_y * gate.lateral - half_z * gate.up,
        +half_y * gate.lateral - half_z * gate.up,
        +half_y * gate.lateral + half_z * gate.up,
        -half_y * gate.lateral + half_z * gate.up,
    ])


def audit_crossings(track: object, frames: list[dict]) -> list[dict]:
    """Independently intersect trajectory segments with each advanced gate plane."""
    crossings: list[dict] = []
    for frame_index, (before, after) in enumerate(zip(frames, frames[1:])):
        if int(after["passed"]) <= int(before["passed"]):
            continue
        gate_index = int(before["gate_index"])
        gate = track.gates[gate_index]
        p0, p1 = before["state"][:3], after["state"][:3]
        d0 = float((p0 - gate.position) @ gate.normal)
        d1 = float((p1 - gate.position) @ gate.normal)
        alpha = float(np.clip(-d0 / (d1 - d0), 0.0, 1.0)) if abs(d1 - d0) > 1e-9 else 0.5
        point = p0 + alpha * (p1 - p0)
        local_y = float((point - gate.position) @ gate.lateral)
        local_z = float((point - gate.position) @ gate.up)
        margin_y = float(gate.size[0] * 0.5 - abs(local_y))
        margin_z = float(gate.size[1] * 0.5 - abs(local_z))
        crossings.append({
            "ordinal": int(after["passed"]), "gate_index": gate_index,
            "frame_index": frame_index + 1,
            "time_seconds": float(after["time"]), "directional_plane_before_m": d0,
            "directional_plane_after_m": d1, "lateral_offset_m": local_y,
            "vertical_offset_m": local_z, "minimum_aperture_margin_m": min(margin_y, margin_z),
            "direction_correct": bool(d0 <= 1e-4 and d1 >= -1e-4),
            "inside_aperture": bool(margin_y >= -1e-4 and margin_z >= -1e-4),
        })
    return crossings


@torch.no_grad()
def rollout(args: argparse.Namespace) -> tuple[object, list[dict], dict]:
    config = load_config(args.config)
    settings = config[args.stage]
    curriculum_key = (
        "reporting_evaluation_curriculum"
        if args.curriculum == "reporting" else "evaluation_curriculum"
    )
    stage = parse_stage(settings[curriculum_key])
    track_path = stage.tracks[args.track_index % len(stage.tracks)]
    stage = replace(stage, tracks=(track_path,))
    policy, normalizer, _, checkpoint = load_policy_checkpoint(args.checkpoint, args.device)
    policy.eval()
    env = make_env(settings, track=track_path, reward=make_reward(settings, stage.target_speed))
    frames: list[dict] = []
    last_info: dict = {}
    try:
        observation, start_passed = reset_env(
            env, stage, seed=args.seed, episode_index=args.episode_index
        )
        start_gate = int(env.tracker.index)
        history = CausalHistory(policy.context_steps)
        history.reset_feature(ppo_observation_features(observation, settings))
        target = min(stage.target_gates, len(env.track.gates))
        target_speed = episode_target_speed(stage, args.seed)
        previous_passed = start_passed
        for step in range(stage.max_steps + 1):
            state = np.asarray(observation["state"], np.float32)
            passed = env.tracker.passed_count - start_passed
            frames.append({
                "time": float(observation["time"]), "state": state.copy(),
                "gate_index": int(env.tracker.index), "passed": int(passed),
                "gate_event": bool(env.tracker.passed_count > previous_passed),
                "action": None, "normalized_action": None,
                "maximum_motor_utilization": 0.0,
            })
            previous_passed = env.tracker.passed_count
            if passed >= target or step >= stage.max_steps:
                break
            batch = torch.from_numpy(normalizer.numpy(history.array()[None])).to(args.device)
            speed_command = torch.as_tensor(
                [target_speed], device=args.device, dtype=batch.dtype
            )
            normalized_action = policy(batch, speed_command).float().cpu().numpy()[0]
            action = ppo_normalized_to_ctbr(normalized_action, settings)
            frames[-1]["action"] = np.asarray(action, np.float32)
            frames[-1]["normalized_action"] = np.asarray(normalized_action, np.float32)
            observation, _, terminated, _, info = env.step(action)
            last_info = dict(info)
            frames[-1]["maximum_motor_utilization"] = float(
                np.max(info.get("applied_motor_normalized", 0.0))
            )
            history.append_feature(ppo_observation_features(observation, settings))
            if terminated:
                state = np.asarray(observation["state"], np.float32)
                frames.append({
                    "time": float(observation["time"]), "state": state.copy(),
                    "gate_index": int(env.tracker.index),
                    "passed": int(env.tracker.passed_count - start_passed),
                    "gate_event": bool(info.get("gate_passed", False)),
                    "action": None, "normalized_action": None,
                    "maximum_motor_utilization": float(
                        np.max(info.get("applied_motor_normalized", 0.0))
                    ),
                })
                break
        actions = np.stack([f["action"] for f in frames if f["action"] is not None])
        normalized_actions = np.stack([
            f["normalized_action"] for f in frames
            if f["normalized_action"] is not None
        ])
        velocities = np.stack([f["state"][7:10] for f in frames])
        altitudes = np.asarray([f["state"][2] for f in frames], np.float32)
        tilt_degrees = np.asarray([
            np.degrees(np.arccos(np.clip(
                quaternion_matrix(f["state"][3:7])[2, 2], -1.0, 1.0
            )))
            for f in frames
        ], np.float32)
        accelerations = np.diff(velocities, axis=0) * float(settings.get("control_hz", 90.0))
        summary = {
            "checkpoint": str(checkpoint), "track": env.track.name,
            "stage": args.stage,
            "track_fingerprint": env.track.fingerprint, "track_index": args.track_index,
            "seed": args.seed, "start_gate": start_gate, "target_gates": target,
            "episode_index": args.episode_index,
            "target_speed_mps": target_speed,
            "passed_gates": int(env.tracker.passed_count - start_passed),
            "completed": bool(env.tracker.passed_count - start_passed >= target),
            "steps": len(frames) - 1, "duration_seconds": float(frames[-1]["time"]),
            "maximum_speed_mps": float(max(np.linalg.norm(f["state"][7:10]) for f in frames)),
            "mean_speed_mps": float(np.mean(np.linalg.norm(velocities, axis=1))),
            "mean_collective_command_mps2": float(np.mean(actions[:, 0])),
            "maximum_collective_command_mps2": float(np.max(actions[:, 0])),
            "collective_saturation_fraction": float(
                np.mean(np.abs(normalized_actions[:, 0]) >= 0.95)
            ),
            "maximum_absolute_body_rate_command_rps": float(
                np.max(np.abs(actions[:, 1:]))
            ),
            "maximum_horizontal_acceleration_mps2": float(
                np.max(np.linalg.norm(accelerations[:, :2], axis=1))
                if len(accelerations) else 0.0
            ),
            "maximum_motor_utilization": float(max(
                f["maximum_motor_utilization"] for f in frames
            )),
            "initial_altitude_m": float(altitudes[0]),
            "minimum_altitude_m": float(altitudes.min()),
            "final_altitude_m": float(altitudes[-1]),
            "minimum_vertical_velocity_mps": float(velocities[:, 2].min()),
            "maximum_tilt_degrees": float(tilt_degrees.max()),
            "ground_contact": bool(last_info.get("ground_contact", False)),
            "unity_collision": bool(last_info.get("unity_collision", False)),
        }
        summary["crossings"] = audit_crossings(env.track, frames)
        post_crossing_windows = []
        for crossing in summary["crossings"]:
            start = int(crossing["frame_index"])
            stop = min(len(frames), start + int(round(float(settings.get(
                "control_hz", 90.0
            )))))
            local_altitude = altitudes[start:stop]
            local_velocity = velocities[start:stop, 2]
            local_actions = [
                frame["action"] for frame in frames[start:stop]
                if frame["action"] is not None
            ]
            post_crossing_windows.append({
                "ordinal": crossing["ordinal"],
                "altitude_drop_next_1s_m": float(
                    altitudes[start] - local_altitude.min()
                ),
                "minimum_vertical_velocity_next_1s_mps": float(
                    local_velocity.min()
                ),
                "mean_collective_next_1s_mps2": float(
                    np.mean(np.stack(local_actions)[:, 0])
                    if local_actions else 0.0
                ),
            })
        summary["post_crossing_windows"] = post_crossing_windows
        summary["geometry_audit_passed"] = bool(
            len(summary["crossings"]) == summary["passed_gates"]
            and all(c["direction_correct"] and c["inside_aperture"] for c in summary["crossings"])
        )
        return env.track, frames, summary
    finally:
        env.close()


def render(track: object, frames: list[dict], summary: dict, args: argparse.Namespace) -> None:
    stride = max(1, round(90 / args.fps))
    indices = list(range(0, len(frames), stride))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)
    positions = np.stack([f["state"][:3] for f in frames])
    gates = np.stack([gate.position for gate in track.gates])
    points = np.concatenate([positions, gates])
    fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi, facecolor="#071019")
    ax = fig.add_subplot(111, projection="3d", facecolor="#071019")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.91)
    writer = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgba", "-s", f"{int(fig.get_figwidth()*args.dpi)}x{int(fig.get_figheight()*args.dpi)}",
        "-r", str(args.fps), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ], stdin=subprocess.PIPE)
    try:
        for frame_index in indices:
            ax.cla()
            ax.set_facecolor("#071019")
            ax.view_init(elev=25, azim=-55)
            _equal_axes(ax, points)
            ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
            ax.tick_params(colors="#9fb3c8")
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.label.set_color("#b9cad8")
            frame = frames[frame_index]
            active = int(frame["gate_index"])
            for index, gate in enumerate(track.gates):
                corners = _gate_corners(gate)
                loop = np.vstack([corners, corners[0]])
                color = "#ffca3a" if index == active else "#4cc9f0"
                ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color,
                        linewidth=3.2 if index == active else 1.5, alpha=1.0 if index == active else 0.48)
                normal = gate.normal * 1.2
                ax.plot([gate.position[0], gate.position[0] + normal[0]],
                        [gate.position[1], gate.position[1] + normal[1]],
                        [gate.position[2], gate.position[2] + normal[2]], color=color, linewidth=1.2)
                ax.text(*gate.position, str(index), color="#e8f1f8", fontsize=8)
            ax.plot(positions[:frame_index + 1, 0], positions[:frame_index + 1, 1],
                    positions[:frame_index + 1, 2], color="#ff4d8d", linewidth=2.5)
            pos = frame["state"][:3]
            rotation = quaternion_matrix(frame["state"][3:7])
            for local, color in ((np.array([1.,0,0]), "#ff595e"),
                                 (np.array([0,1.,0]), "#8ac926"),
                                 (np.array([0,0,1.]), "#1982c4")):
                endpoint = pos + rotation @ local * 0.9
                ax.plot([pos[0], endpoint[0]], [pos[1], endpoint[1]], [pos[2], endpoint[2]],
                        color=color, linewidth=4)
            speed = float(np.linalg.norm(frame["state"][7:10]))
            thrust = (
                float(frame["action"][0]) if frame["action"] is not None else 0.0
            )
            ax.set_title(
                f"{track.name}  |  t={frame['time']:.2f}s  |  speed={speed:.1f} m/s  |  "
                f"collective={thrust:.1f} m/s²  |  "
                f"gates={frame['passed']}/{summary['target_gates']}  |  active={active}",
                color="white", fontsize=14,
            )
            fig.canvas.draw()
            writer.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    finally:
        writer.stdin.close()
        code = writer.wait()
        plt.close(fig)
    if code:
        raise RuntimeError(f"ffmpeg exited with status {code}")


def main() -> None:
    args = arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    track, frames, summary = rollout(args)
    render(track, frames, summary, args)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
