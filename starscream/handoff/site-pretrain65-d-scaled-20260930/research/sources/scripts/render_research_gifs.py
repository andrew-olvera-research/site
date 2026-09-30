#!/usr/bin/env python3
"""Select diverse, recovery-free base-policy laps and render academic GIFs.

This is deliberately independent of the W&B video publisher.  It reuses the
frozen evaluator's reset, environment, observation, action, and seed contracts,
ranks successful episodes by simulated step count, then replays the selected
state sequence through a clean fixed world-space course overview.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_privileged_racing import (  # noqa: E402
    CausalHistory,
    ProcessRaceCollector,
    completion_gate_count,
    episode_target_speed,
    load_config,
    make_env,
    make_reward,
    parse_stage,
    ppo_normalized_to_ctbr,
    ppo_observation_features,
    reset_env,
)
from starscream.env.tracks import load_track, quaternion_matrix  # noqa: E402
from starscream.inference_graph import HostPolicyInferenceGraphs  # noqa: E402
from starscream.privileged_racing import load_policy_checkpoint  # noqa: E402
from scripts.export_site_trajectories import demo_payload  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/exp/v6.21.1/update_fix_dagger.yaml"
DEFAULT_CHECKPOINT = (
    ROOT / "outputs/checkpoints/starscream-v6.21.1.update-fix-pretrain65-dagger/"
    "best-step-121343545-full_course_success-0.63181818.pt"
)
DEFAULT_SUITE = ROOT / "configs/eval/v6_22_real100_hard_v2.yaml"
DEFAULT_BENCHMARK = (
    ROOT / "outputs/evals/v6211-update-fix-final/update-fix-real100-hard-v2-e8.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/research-gifs-v6211-update-fix-minimal"
SELECTION_PROTOCOL = (
    "32-seed cohort per course; fastest reproducible zero-missed-gate completion "
    "when available, otherwise fastest reproducible completed recovery; separate "
    "recovery examples are exported for the flagged courses when available"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--track", action="append", default=[],
        help="explicit suite track name; repeat to override automatic diverse selection",
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument(
        "--use-all-suite", action="store_true",
        help="evaluate every active course in suite order (ignores --count)",
    )
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2034091462)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=405)
    parser.add_argument("--fov", type=float, default=72.0)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument(
        "--recovery-only", action="store_true",
        help="append distinct recovery episodes recorded in selection.json",
    )
    parser.add_argument(
        "--reuse-selection", action="store_true",
        help="reuse output/selection.json instead of rerunning episode ranking",
    )
    parser.add_argument(
        "--reuse-capture", action="store_true",
        help="replay capture_episode_index from selection.json instead of reranking",
    )
    args = parser.parse_args()
    if args.reuse_capture and not args.reuse_selection:
        parser.error("--reuse-capture requires --reuse-selection")
    if args.recovery_only and not args.reuse_selection:
        parser.error("--recovery-only requires --reuse-selection")
    if args.count < 1 or args.episodes < 1 or args.workers < 1:
        parser.error("count, episodes, and workers must be positive")
    if args.fps < 1 or min(args.width, args.height) < 64:
        parser.error("fps must be positive and image dimensions must be at least 64")
    return args


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def suite_records(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    records = [dict(item) for item in payload.get("active", [])]
    if not records:
        raise ValueError(f"suite has no active records: {path}")
    for record in records:
        track = Path(str(record["track"]))
        record["path"] = track if track.is_absolute() else ROOT / track
    return records


def benchmark_success(metrics: Mapping[str, Any], name: str) -> float:
    return float(metrics.get(f"track/{name}/full_course_success", 0.0))


def benchmark_fastest(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(f"track/{name}/successful_minimum_steps", float("inf"))
    value = float(value)
    return value if math.isfinite(value) else float("inf")


def choose_diverse_tracks(
    records: list[dict[str, Any]], metrics: Mapping[str, Any], count: int,
    explicit: list[str],
) -> list[dict[str, Any]]:
    """Balance geometry novelty, distinct families, and benchmark completion speed."""
    by_name = {str(item["name"]): item for item in records}
    if explicit:
        missing = [name for name in explicit if name not in by_name]
        if missing:
            raise ValueError(f"tracks are not active in suite: {missing}")
        return [by_name[name] for name in explicit[:count]]

    successful = [
        item for item in records
        if benchmark_success(metrics, str(item["name"])) > 0.0
    ]
    successful.sort(key=lambda item: (
        -benchmark_success(metrics, str(item["name"])),
        benchmark_fastest(metrics, str(item["name"])),
        str(item["name"]),
    ))
    # Farthest-point geometry sampling, with a distinct family at each step.
    features = []
    for item in successful:
        track = load_track(item["path"])
        xyz = np.stack([g.position for g in track.gates])
        delta = np.roll(xyz, -1, axis=0) - xyz
        lengths = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(lengths[:, None], 1e-6)
        turns = np.arccos(np.clip(np.sum(direction * np.roll(direction, 1, axis=0), axis=1), -1, 1))
        span = np.ptp(xyz, axis=0)
        features.append([len(xyz), lengths.sum(), span[2],
                         min(span[:2]) / max(max(span[:2]), 1e-6),
                         turns.mean(), turns.max(), lengths.std()])
    values = np.asarray(features)
    if not len(values):
        raise RuntimeError("No successful benchmark courses")
    values = (values - values.mean(0)) / np.maximum(values.std(0), 1e-6)
    selected = []
    indices = []
    used_families = set()
    while len(selected) < min(count, len(successful)):
        candidates = [i for i, r in enumerate(successful)
                      if i not in indices and r.get("family") not in used_families]
        if not candidates:
            candidates = [i for i in range(len(successful)) if i not in indices]
        def score(i):
            novelty = min(np.linalg.norm(values[i] - values[j]) for j in indices) if indices else np.linalg.norm(values[i])
            return novelty + 0.35 * benchmark_success(metrics, str(successful[i]["name"])) - benchmark_fastest(metrics, str(successful[i]["name"])) / 1500.0
        index = max(candidates, key=score)
        indices.append(index)
        selected.append(successful[index])
        used_families.add(successful[index].get("family"))
    if len(selected) < count:
        raise RuntimeError(
            f"benchmark contains only {len(selected)} suitable successful tracks; "
            f"requested {count}"
        )
    return selected[:count]


def select_fastest_episodes(
    policy: torch.nn.Module,
    normalizer: Any,
    settings: dict[str, Any],
    base_stage: Any,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records, 1):
        track_path = str(Path(record["path"]).resolve())
        stage = replace(base_stage, tracks=(track_path,), max_steps=args.max_steps)
        print(
            f"[{ordinal}/{len(records)}] evaluating {record['name']} "
            f"({args.episodes} episodes)", flush=True,
        )
        collector = ProcessRaceCollector(
            policy, normalizer, settings, stage, args.device,
            workers=min(args.workers, args.episodes),
        )
        try:
            rows = collector.evaluate_rows(
                episodes=args.episodes, seed_base=args.seed
            )
        finally:
            collector.close()
        successes = [
            dict(row) for row in rows
            if int(row["gates"]) >= int(row["target_gates"])
            and not bool(row["crashed"])
        ]
        successes.sort(key=lambda row: (int(row["steps"]), int(row["episode_index"])))
        if not successes:
            raise RuntimeError(
                f"{record['name']} had no clean completion in {args.episodes} episodes"
            )
        best = successes[0]
        selection = {
            **{key: value for key, value in record.items() if key != "path"},
            "track_path": track_path,
            "episodes_evaluated": len(rows),
            "successful_episodes": len(successes),
            "selected_episode_index": int(best["episode_index"]),
            "selected_steps": int(best["steps"]),
            "selected_seed": int(args.seed + 1009 * int(best["episode_index"])),
            "target_gates": int(best["target_gates"]),
            "selection_row": best,
            "episode_rows": [dict(row) for row in rows],
            "successful_episode_rows": successes,
        }
        selections.append(selection)
        print(
            f"  selected episode {selection['selected_episode_index']}: "
            f"{selection['selected_steps']} steps; {len(successes)}/{len(rows)} successes",
            flush=True,
        )
    return selections


@torch.no_grad()
def capture_rollout(
    policy: torch.nn.Module,
    normalizer: Any,
    inference: HostPolicyInferenceGraphs,
    settings: Mapping[str, Any],
    base_stage: Any,
    selection: Mapping[str, Any],
    device: str,
    max_steps: int,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    track_path = str(selection["track_path"])
    stage = replace(base_stage, tracks=(track_path,), max_steps=max_steps)
    target_speed = episode_target_speed(stage, int(selection["selected_seed"]))
    env = make_env(
        settings, track=track_path, reward=make_reward(settings, target_speed)
    )
    frames: list[dict[str, Any]] = []
    try:
        observation, start_passed = reset_env(
            env, stage, seed=int(selection["selected_seed"]),
            episode_index=int(selection["selected_episode_index"]),
        )
        history = CausalHistory(policy.context_steps)
        history.reset_feature(ppo_observation_features(observation, settings))
        target = completion_gate_count(
            env.track, stage.target_gates,
            enabled=bool(settings.get("allow_curriculum_completion_override", False)),
        ) * stage.rollout_laps
        crashed = False
        missed_gate_crossings = 0
        for step in range(stage.max_steps + 1):
            passed = int(env.tracker.passed_count - start_passed)
            state = np.asarray(observation["state"], np.float32).copy()
            frames.append({
                "time": float(observation["time"]),
                "state": state,
                "passed": passed,
                "speed": float(np.linalg.norm(state[7:10])),
            })
            if passed >= target or step >= stage.max_steps:
                break
            normalized = normalizer.numpy(history.array()[None])
            normalized_action = inference.predict_numpy(
                normalized, np.asarray([target_speed], np.float32)
            )[0]
            action = ppo_normalized_to_ctbr(normalized_action, settings)
            gate = env.track.gates[env.tracker.index]
            previous_side = float((state[:3] - gate.position) @ gate.normal)
            previous_passed = env.tracker.passed_count
            observation, _, terminated, _, info = env.step(action)
            current_side = float((np.asarray(observation["state"])[:3] - gate.position) @ gate.normal)
            if previous_side < 0 <= current_side and env.tracker.passed_count == previous_passed:
                missed_gate_crossings += 1
            history.append_feature(ppo_observation_features(observation, settings))
            crashed = bool(info.get("ground_contact") or info.get("unity_collision"))
            if terminated:
                state = np.asarray(observation["state"], np.float32).copy()
                frames.append({
                    "time": float(observation["time"]),
                    "state": state,
                    "passed": int(env.tracker.passed_count - start_passed),
                    "speed": float(np.linalg.norm(state[7:10])),
                })
                break
        summary = {
            "track": env.track.name,
            "track_fingerprint": env.track.fingerprint,
            "episode_index": int(selection["selected_episode_index"]),
            "seed": int(selection["selected_seed"]),
            "steps": len(frames) - 1,
            "target_gates": int(target),
            "passed_gates": int(frames[-1]["passed"]),
            "completed": bool(frames[-1]["passed"] >= target and not crashed),
            "crashed": crashed,
            "missed_gate_crossings": missed_gate_crossings,
            "target_speed_mps": float(target_speed),
            "duration_seconds": float(frames[-1]["time"] - frames[0]["time"]),
            "maximum_speed_mps": float(max(frame["speed"] for frame in frames)),
        }
        return env.track, frames, summary
    finally:
        env.close()


def find_fastest_reproducible_rollout(
    policy: torch.nn.Module,
    normalizer: Any,
    settings: Mapping[str, Any],
    base_stage: Any,
    selection: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], tuple[Any, list[dict[str, Any]], dict[str, Any]] | None]:
    """Rank the full cohort using the exact one-lane graph used for capture."""
    inference = HostPolicyInferenceGraphs(policy, 1)
    audit = []
    best: tuple[Any, list[dict[str, Any]], dict[str, Any]] | None = None
    best_recovery: tuple[Any, list[dict[str, Any]], dict[str, Any]] | None = None
    successful_rows = list(selection.get("successful_episode_rows", []))
    successful_rows.sort(key=lambda row: (int(row["steps"]), int(row["episode_index"])))
    if not successful_rows:
        successful_rows = [{"episode_index": i, "steps": 0} for i in range(args.episodes)]
    replay_recovery_cohort = selection.get("role") == "recovery_search"
    for row in successful_rows:
        episode_index = int(row["episode_index"])
        trial = dict(selection)
        trial["selected_episode_index"] = episode_index
        trial["selected_seed"] = int(args.seed + 1009 * episode_index)
        result = capture_rollout(
            policy, normalizer, inference, settings, base_stage, trial,
            args.device, args.max_steps,
        )
        audit.append(result[2])
        if result[2]["completed"] and result[2]["missed_gate_crossings"] == 0:
            if best is None or result[2]["steps"] < best[2]["steps"]:
                best = result
        elif result[2]["completed"] and result[2]["missed_gate_crossings"] > 0:
            if best_recovery is None or result[2]["steps"] < best_recovery[2]["steps"]:
                best_recovery = result
        if best is not None and not replay_recovery_cohort:
            # Candidates are fastest-first; the first clean replay is the
            # fastest clean lap in the parallel evaluation cohort.
            break
    if best is None:
        if best_recovery is None:
            raise RuntimeError(f"{selection['name']} had no reproducible completion")
        best = best_recovery
        selection["capture_kind"] = "recovery_fallback"
    selection["capture_episode_audit"] = audit
    selection["capture_episodes_replayed"] = len(audit)
    selection["capture_episodes_evaluated"] = int(selection.get("episodes_evaluated", args.episodes))
    selection["capture_successful_episodes"] = int(selection.get("successful_episodes", 0))
    selection["capture_clean_successful_episodes"] = sum(
        bool(row["completed"]) and int(row["missed_gate_crossings"]) == 0 for row in audit
    )
    selection["capture_episode_index"] = int(best[2]["episode_index"])
    selection["capture_seed"] = int(best[2]["seed"])
    selection["capture_steps"] = int(best[2]["steps"])
    return best[0],best[1],best[2],best_recovery


def _gate_corners(track: Any) -> np.ndarray:
    corners = []
    for gate in track.gates:
        half_y, half_z = np.asarray(gate.size, np.float32) * 0.5
        for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append(
                gate.position + sy * half_y * gate.lateral + sz * half_z * gate.up
            )
    return np.asarray(corners, np.float32)


def overview_camera(
    track: Any, frames: list[dict[str, Any]], width: int, height: int, fov: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trajectory = np.stack([frame["state"][:3] for frame in frames])
    points = np.concatenate([_gate_corners(track), trajectory[::max(1, len(trajectory)//300)]])
    target = 0.5 * (points.min(0) + points.max(0))
    target[2] = max(1.2, float(np.percentile(points[:, 2], 45)))
    centered_xy = points[:, :2] - points[:, :2].mean(0)
    covariance = centered_xy.T @ centered_xy
    major = np.linalg.eigh(covariance)[1][:, -1]
    horizontal = np.asarray([-major[1], major[0], 0.0], np.float32)
    horizontal_tan = math.tan(math.radians(fov) * 0.5)
    vertical_tan = horizontal_tan * height / width
    span = float(np.linalg.norm(points.max(0) - points.min(0)))
    gate_points = _gate_corners(track)
    candidates = []
    # Search fixed overview views to separate gate silhouettes and avoid edge-on apertures.
    base_angle = math.atan2(horizontal[1], horizontal[0])
    for elevation_degrees in (35, 45, 57, 65):
        elevation = math.radians(elevation_degrees)
        for offset in range(0, 360, 10):
            azimuth = base_angle + math.radians(offset)
            view = np.asarray([math.cos(azimuth)*math.cos(elevation),
                               math.sin(azimuth)*math.cos(elevation), math.sin(elevation)])
            forward = -view
            right = np.cross([0., 0., 1.], forward)
            right /= np.linalg.norm(right)
            rotation = np.stack([right, np.cross(forward, right), forward], axis=1)
            distance = max(8., span * .7)
            for _ in range(100):
                position = target + distance * view
                local = (points - position) @ rotation
                depth = local[:, 2]
                if (np.all(depth > .2) and
                    np.max(np.abs(local[:, 0]/depth)) <= horizontal_tan*.90 and
                    np.max(np.abs(local[:, 1]/depth)) <= vertical_tan*.84):
                    break
                distance *= 1.04
            pixels = project_points(gate_points, position, rotation, width, height, fov).reshape(-1,4,2)
            low, high = pixels.min(1)-3, pixels.max(1)+3
            overlap = 0.
            for i in range(len(pixels)):
                for j in range(i):
                    intersection = np.maximum(0., np.minimum(high[i],high[j])-np.maximum(low[i],low[j]))
                    overlap += float(np.prod(intersection))
            # Relative polygon area penalizes gates rendered as a nearly invisible line.
            area = .5*np.abs(np.sum(pixels[:,:,0]*np.roll(pixels[:,:,1],-1,axis=1)-
                                   pixels[:,:,1]*np.roll(pixels[:,:,0],-1,axis=1),axis=1))
            box_area = np.prod(pixels.max(1)-pixels.min(1),axis=1)
            flat = np.sum(np.maximum(0., .30-area/np.maximum(box_area,1.)))
            coverage = np.ptp(project_points(points, position, rotation, width, height, fov)[:,0]) / width
            cost = (overlap + 20*flat + .3*min(offset,360-offset)
                    + .5*abs(elevation_degrees-57) + 80*max(0.,.65-coverage))
            candidates.append((cost, position, rotation))
    _, position, rotation = min(candidates, key=lambda item: item[0])
    return position.astype(np.float32), rotation.astype(np.float32), target


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def display_name(name: str) -> str:
    name = name.removesuffix("_showcase")
    replacements = {
        "v622_hard_v2b_vertical_chain_008_06": "Vertical chain",
        "v622_hard_v2b_compound_reversal_021_03": "Compound reversal",
        "v622_hard_diving_hairpin_039_011_arena1": "Diving hairpin",
        "v622_hard_slalom_028_029": "Slalom",
        "v622_hard_long_braking_023_005_arena1_mirror1": "Long braking",
        "v622_hard_v2b_radius_switch_029_21": "Changing-radius turns",
        "v622_hard_ordered_3d_042_003_mirror1": "Ordered 3D",
        "v622_hard_v2b_wrong_side_incidence_015_15": "Oblique gate approaches",
        "swift_champion_2022_exact": "Swift Champion 2022",
        "multigp_cdra_2026_reconstructed_arena1": "MultiGP CDRA 2026",
        "multigp_utt03_bessel_run": "MultiGP UTT03 · Bessel Run",
        "multigp_utt05_nautilus": "MultiGP UTT05 · Nautilus",
    }
    if name in replacements:
        return replacements[name]
    family_titles = {
        "go_around": "Go-Around · Held-Out Course",
        "ordered_3d": "Ordered 3D · Held-Out Course",
        "stacked_reversal": "Stacked Reversal · Held-Out Course",
        "slalom": "Slalom · Held-Out Course",
    }
    for token, title in family_titles.items():
        if token in name:
            return title
    text = re.sub(r"_(arena\d+|mirror\d+)$", "", name)
    return " ".join(word.upper() if word in {"3d"} else word.capitalize()
                    for word in text.split("_"))


def project_points(
    points: np.ndarray, camera_position: np.ndarray, world_from_camera: np.ndarray,
    width: int, height: int, fov: float,
) -> np.ndarray:
    local = (np.asarray(points) - camera_position) @ world_from_camera
    focal = 0.5 * width / math.tan(math.radians(fov) * 0.5)
    depth = np.maximum(local[:, 2], 1.0e-3)
    return np.stack([
        width * 0.5 + focal * local[:, 0] / depth,
        height * 0.5 - focal * local[:, 1] / depth,
    ], axis=1)


def render_academic_frame(track, frames, source_index, camera_position,
                          camera_rotation, args, background):
    image = background.copy()
    draw = ImageDraw.Draw(image)
    def project(points):
        return project_points(np.asarray(points), camera_position, camera_rotation,
                              args.width, args.height, args.fov)
    frame = frames[source_index]
    for gate in track.gates:
        corners = _gate_corners(type("OneGate", (), {"gates": [gate]})())
        pixels = project(np.vstack([corners, corners[0]]))
        draw.line([tuple(p) for p in pixels], fill=(115, 120, 126), width=2)
    trajectory = project([f["state"][:3] for f in frames[:source_index + 1]])
    if len(trajectory) > 1:
        draw.line([tuple(p) for p in trajectory], fill=(42, 105, 160), width=2)
    # Compact quadrotor: physical arm proportions, projected rotor discs.
    state = frame["state"]
    rotation = quaternion_matrix(state[3:7])
    position = state[:3]
    rotors = np.asarray([[.20,.20,0],[.20,-.20,0],[-.20,-.20,0],[-.20,.20,0]])
    center = project([position])[0]
    for rotor in rotors:
        endpoint = project([position + rotation @ rotor])[0]
        draw.line([tuple(center), tuple(endpoint)], fill=(30,30,30), width=2)
        theta = np.linspace(0, 2*np.pi, 17)
        disc = rotor + np.stack([.10*np.cos(theta), .10*np.sin(theta), np.zeros_like(theta)], axis=1)
        pixels = project(position + disc @ rotation.T)
        draw.line([tuple(p) for p in pixels], fill=(50,50,50), width=1)
    draw.ellipse((center[0]-2, center[1]-2, center[0]+2, center[1]+2), fill=(25,25,25))
    draw.text((14, 12), display_name(track.name), font=_font(13), fill=(60,60,60))
    draw.text((14, args.height-25), f"{frame['time']:.2f} s", font=_font(12), fill=(60,60,60))
    return image


def encode_assets(frame_dir: Path, stem: Path, fps: int) -> tuple[Path, Path]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to encode GIF and MP4 assets")
    pattern = str(frame_dir / "frame-%05d.png")
    mp4 = stem.with_suffix(".mp4")
    gif = stem.with_suffix(".gif")
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", pattern, "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "slow",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(mp4),
    ], check=True)
    palette = (
        "split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=full[p];"
        "[s1][p]paletteuse=dither=none"
    )
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", pattern, "-filter_complex", palette, "-loop", "0", str(gif),
    ], check=True)
    return gif, mp4


def render_selection(
    track: Any,
    frames: list[dict[str, Any]],
    summary: dict[str, Any],
    selection: Mapping[str, Any],
    ordinal: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    slug = re.sub(r"[^a-z0-9]+", "-", str(selection["name"]).lower()).strip("-")
    stem = args.output / f"{ordinal:02d}-{slug}"
    np.savez_compressed(stem.with_suffix(".npz"),
                        states=np.stack([f["state"] for f in frames]),
                        times=np.asarray([f["time"] for f in frames]),
                        passed=np.asarray([f["passed"] for f in frames]))
    dt=1.0/float(summary.get("control_hz",130.0))
    demo=demo_payload(track,np.stack([f["state"] for f in frames]),
        np.asarray([f["time"] for f in frames],np.float64),
        np.asarray([f["passed"] for f in frames],np.int64),
        name=f"{selection['name']} - {selection.get('capture_kind','fastest')}",
        control_hz=1.0/dt)
    demo_path=stem.with_name(stem.name+".demo.json")
    with demo_path.open('x',encoding='utf-8') as stream:
        json.dump(demo,stream,separators=(',',':'))
    with (args.output/'trajectories.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(demo,separators=(',',':'))+'\n')
    frame_dir = args.output / "frames" / stem.name
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    camera_position, camera_rotation, target = overview_camera(
        track, frames, args.width, args.height, args.fov
    )
    start_time = float(frames[0]["time"])
    end_time = float(frames[-1]["time"])
    source_times = np.asarray([float(frame["time"]) for frame in frames])
    output_times = np.arange(start_time, end_time + 0.5 / args.fps, 1.0 / args.fps)
    indices = np.searchsorted(source_times, output_times, side="left")
    indices = np.clip(indices, 0, len(frames) - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = (
        np.abs(source_times[previous] - output_times)
        < np.abs(source_times[indices] - output_times)
    )
    indices[use_previous] = previous[use_previous]
    background = Image.new("RGB", (args.width, args.height), "white")
    for output_index, source_index in enumerate(indices):
        rendered = render_academic_frame(
            track, frames, int(source_index), camera_position,
            camera_rotation, args, background,
        )
        rendered.save(frame_dir / f"frame-{output_index:05d}.png", optimize=True)
    gif, mp4 = encode_assets(frame_dir, stem, args.fps)
    gif_image = Image.open(gif)
    gif_frames = int(getattr(gif_image, "n_frames", len(indices)))
    gif_duration = gif_frames / args.fps
    gif_image.seek(0)
    poster = stem.with_name(stem.name + "-poster.jpg")
    gif_image.convert("RGB").save(poster, quality=90, optimize=True)
    if not args.keep_frames:
        shutil.rmtree(frame_dir)
    result = {
        **summary,
        "name": str(selection["name"]),
        "family": selection.get("family"),
        "exposure": selection.get("exposure"),
        "geometry_edit": selection.get("geometry_edit"),
        "track_path": str(selection["track_path"]),
        "episodes_evaluated": int(selection.get(
            "capture_episodes_evaluated", selection["episodes_evaluated"]
        )),
        "successful_episodes_in_cohort": int(selection.get("successful_episodes", 0)),
        "capture_episodes_replayed": int(selection.get("capture_episodes_replayed", 0)),
        "clean_completions_in_replays": int(selection.get("capture_clean_successful_episodes", 0)),
        "benchmark_success_rate": selection.get("benchmark_success_rate"),
        "benchmark_fastest_steps": selection.get("benchmark_fastest_steps"),
        "capture_kind": selection.get("capture_kind", "fastest"),
        "role": selection.get("role"),
        "gif": gif.name,
        "mp4": mp4.name,
        "trajectory_json": demo_path.name,
        "poster": poster.name,
        "gif_frames": gif_frames,
        "fps": args.fps,
        "gif_duration_seconds": gif_duration,
        "playback_rate_vs_simulation": gif_duration / summary["duration_seconds"],
        "renderer": "starscream-academic-v3",
        "camera_position": camera_position.tolist(),
        "camera_target": target.tolist(),
        "gif_bytes": gif.stat().st_size,
        "mp4_bytes": mp4.stat().st_size,
    }
    stem.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    settings = dict(config["dagger"])
    settings["evaluation_workers"] = min(args.workers, args.episodes)
    settings["evaluation_backend"] = "process"
    base_stage = replace(
        parse_stage(settings["evaluation_curriculum"]), max_steps=args.max_steps
    )
    benchmark = _load_json(args.benchmark)
    suite = suite_records(args.suite)
    records = (suite if args.use_all_suite else choose_diverse_tracks(
        suite, benchmark["metrics"], args.count, args.track
    ))
    policy, normalizer, _, resolved = load_policy_checkpoint(
        args.checkpoint, args.device
    )
    policy.eval()
    selection_path = args.output / "selection.json"
    if args.reuse_selection:
        if not selection_path.exists():
            raise FileNotFoundError(f"selection manifest does not exist: {selection_path}")
        selections = list(_load_json(selection_path)["selections"])
    else:
        selections = [{**{k: v for k, v in record.items() if k != "path"},
                       "track_path": str(record["path"]),
                       "episodes_evaluated": 0, "successful_episodes": 0}
                      for record in records]
    selection_payload = {
        "schema": "starscream-research-gif-selection-v1",
        "checkpoint": str(resolved),
        "config": str(args.config.resolve()),
        "suite": str(args.suite.resolve()),
        "benchmark": str(args.benchmark.resolve()),
        "seed_base": args.seed,
        "episodes_per_track": args.episodes,
        "control_hz": float(settings.get("control_hz", 90.0)),
        "selections": selections,
    }
    selection_path.write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.selection_only:
        selections = select_fastest_episodes(policy, normalizer, settings, base_stage, records, args)
        selection_payload["selections"] = selections
        selection_path.write_text(json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(selection_payload, indent=2, sort_keys=True), flush=True)
        return

    for selection in selections:
        selection["benchmark_success_rate"] = benchmark_success(benchmark["metrics"], selection["name"])
        selection["benchmark_fastest_steps"] = benchmark_fastest(benchmark["metrics"], selection["name"])
    if args.recovery_only:
        inference = HostPolicyInferenceGraphs(policy, 1)
        extra_assets = []
        next_ordinal = len(list(args.output.glob("*.demo.json"))) + 1
        for selection in selections:
            if selection.get("role") != "recovery_search":
                continue
            fastest_index = int(selection.get("capture_episode_index", -1))
            recovered = [row for row in selection.get("capture_episode_audit", [])
                         if row.get("completed") and int(row.get("missed_gate_crossings", 0)) > 0
                         and int(row.get("episode_index", -1)) != fastest_index]
            if not recovered:
                selection["additional_recovery_result"] = (
                    "No distinct recovery episode in this cohort; the fastest lap is the captured recovery."
                )
                print(f"{selection['name']}: fastest lap is the only distinct recovery", flush=True)
                continue
            candidate = min(recovered, key=lambda row: (int(row["steps"]), int(row["episode_index"])))
            trial = dict(selection)
            trial["selected_episode_index"] = int(candidate["episode_index"])
            trial["selected_seed"] = int(args.seed + 1009 * int(candidate["episode_index"]))
            track, frames, summary = capture_rollout(
                policy, normalizer, inference, settings, base_stage, trial,
                args.device, args.max_steps,
            )
            if not summary["completed"] or int(summary["missed_gate_crossings"]) <= 0:
                raise RuntimeError(f"recovery episode no longer reproduces: {summary}")
            summary["control_hz"] = float(settings.get("control_hz", 130.0))
            recovery_selection = dict(selection, capture_kind="recovery")
            prior_asset = None
            for sidecar in args.output.glob("*.json"):
                if sidecar.name in {"manifest.json", "selection.json"}:
                    continue
                item = _load_json(sidecar)
                if (item.get("name") == selection["name"]
                        and item.get("capture_kind") == "recovery"
                        and int(item.get("episode_index", -1)) == int(summary["episode_index"])):
                    prior_asset = item
                    break
            if prior_asset is not None:
                extra_assets.append(prior_asset)
            else:
                extra_assets.append(render_selection(
                    track, frames, summary, recovery_selection, next_ordinal, args,
                ))
                next_ordinal += 1
            selection["additional_recovery_result"] = {
                "episode_index": int(summary["episode_index"]),
                "seed": int(summary["seed"]),
                "steps": int(summary["steps"]),
                "missed_gate_crossings": int(summary["missed_gate_crossings"]),
                "completed": True,
            }
            print(
                f"{selection['name']}: exported distinct recovery episode "
                f"{summary['episode_index']} ({summary['missed_gate_crossings']} missed crossings)",
                flush=True,
            )
        manifest_path = args.output / "manifest.json"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            manifest["selection_protocol"] = SELECTION_PROTOCOL
            seen = {item.get("trajectory_json") for item in manifest.get("assets", [])}
            manifest.setdefault("assets", []).extend(
                item for item in extra_assets if item.get("trajectory_json") not in seen
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            gallery_path = args.output / "index.html"
            gallery = gallery_path.read_text(encoding="utf-8") if gallery_path.exists() else ""
            gallery = gallery.replace(
                "<p>Curated full successful episodes, selected from 32 seeds per course. "
                "Zero detected missed active-gate crossings. Real-time playback.</p>",
                "<p>Fastest reproducible course completions selected from 32 seeds per "
                "course; clean laps are preferred, and recovery fallbacks are identified "
                "in metadata. Separate recovery examples are shown when available. "
                "Real-time playback.</p>",
            )
            for item in extra_assets:
                if item["gif"] in gallery:
                    continue
                caption = (
                    f'{html.escape(display_name(item["name"]))} · recovery · '
                    f'{item["duration_seconds"]:.2f} s · episode {item["episode_index"]} · '
                    f'seed {item["seed"]}'
                )
                gallery += (
                    f'\n<figure><img src="{html.escape(item["gif"])}" width="720" '
                    f'alt="{html.escape(display_name(item["name"]))} recovery"><figcaption>'
                    f'{caption}</figcaption></figure>'
                )
            gallery_path.write_text(gallery + "\n", encoding="utf-8")
        selection_payload["additional_recovery_assets"] = extra_assets
        selection_path.write_text(
            json.dumps(selection_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    assets = []
    for ordinal, selection in enumerate(selections, 1):
        print(
            f"[{ordinal}/{len(selections)}] ranking reproducible laps for "
            f"{selection['name']}", flush=True,
        )
        if args.reuse_capture:
            if "capture_episode_index" not in selection:
                raise ValueError(
                    f"selection for {selection['name']} has no captured episode to reuse"
                )
            trial = dict(selection)
            trial["selected_episode_index"] = int(selection["capture_episode_index"])
            trial["selected_seed"] = int(selection["capture_seed"])
            track, frames, summary = capture_rollout(
                policy, normalizer, HostPolicyInferenceGraphs(policy, 1),
                settings, base_stage, trial, args.device, args.max_steps,
            )
            if not summary["completed"] or summary["missed_gate_crossings"]:
                raise RuntimeError(f"captured episode no longer reproduces: {summary}")
        else:
            track, frames, summary, recovery = find_fastest_reproducible_rollout(
                policy, normalizer, settings, base_stage, selection, args,
            )
        print(
            f"  selected episode {summary['episode_index']}: {summary['steps']} steps; "
            f"{selection['capture_successful_episodes']}/{args.episodes} successes",
            flush=True,
        )
        print(
            f"  course rendering {summary['duration_seconds']:.2f} s "
            f"at {args.fps} fps", flush=True,
        )
        assets.append(render_selection(
            track, frames, summary, selection, ordinal, args
        ))
        if selection.get("role")=="recovery_search":
            if recovery is None:
                selection["recovery_search_result"]="No completed miss-and-return episode appeared in the evaluated cohort."
                print(f"  recovery example not found in {args.episodes} episodes",flush=True)
            elif int(recovery[2]["episode_index"]) == int(summary["episode_index"]):
                selection["recovery_search_result"]="The selected fastest lap is also the only captured completed recovery example."
                print("  fastest lap is also the only completed recovery example",flush=True)
            else:
                recovery_summary=dict(recovery[2],control_hz=float(settings.get("control_hz",90.0)))
                recovery_selection=dict(selection,capture_kind="recovery")
                print(f"  recovery example: {recovery_summary['missed_gate_crossings']} missed active-gate crossings, then completed",flush=True)
                assets.append(render_selection(track,recovery[1],recovery_summary,
                    recovery_selection,len(assets)+1,args))
        selection_path.write_text(
            json.dumps(selection_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema": "starscream-research-gifs-v1",
        "checkpoint": str(resolved),
        "config": str(args.config.resolve()),
        "suite": str(args.suite.resolve()),
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_full_course_success": benchmark["metrics"].get("full_course_success"),
        "selection_protocol": SELECTION_PROTOCOL,
        "rendering": {
            "engine": "Starscream academic renderer v3",
            "camera": "fixed world-space course overview",
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "timing": "one output second per simulator second; nearest 130 Hz state",
            "trajectory_schema": "name/gates/trajectory{dt,pos,quat,passed}; m,s,rad; z up; quaternion wxyz",
            "trajectory_control_hz": float(settings.get("control_hz",90.0)),
        },
        "assets": assets,
    }
    gallery = ["<!doctype html><meta charset='utf-8'><title>Starscream research GIFs</title>",
               "<h1>v6.21.1 update-fix — real100-v2</h1>",
               "<p>Fastest reproducible course completions selected from 32 seeds per "
               "course; clean laps are preferred, and recovery fallbacks are identified "
               "in metadata. Separate recovery examples are shown when available. "
               "Real-time playback.</p>"]
    for item in assets:
        gallery.append(f'<figure><img src="{html.escape(item["gif"])}" width="720" '
                       f'alt="{html.escape(display_name(item["name"]))}"><figcaption>'
                       f'{html.escape(display_name(item["name"]))}{" (edited showcase course)" if item.get("geometry_edit") else ""} · '
                       f'{item["duration_seconds"]:.2f} s · episode {item["episode_index"]}'
                       f' · seed {item["seed"]}</figcaption></figure>')
    (args.output / "index.html").write_text("\n".join(gallery), encoding="utf-8")
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output), "manifest": str(manifest_path),
        "gifs": [item["gif"] for item in assets],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
