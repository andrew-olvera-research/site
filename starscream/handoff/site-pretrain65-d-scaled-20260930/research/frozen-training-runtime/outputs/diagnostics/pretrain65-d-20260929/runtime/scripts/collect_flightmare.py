#!/usr/bin/env python3
"""Collect self-describing Flightmare episodes for tokenizer/dynamics debugging."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np

from starscream.env import FlightmareEnv
from starscream.env.collect import (
    CollectionMixtureConfig,
    CollectionRegime,
    DiverseScenarioSampler,
    EpisodeCollector,
    GateChasePolicy,
    GeometricExpertPolicy,
    HoverPolicy,
    PerturbedControllerPolicy,
    SmoothExcitationPolicy,
    TrajectorySpawnSampler,
    allocate_regimes,
    episode_outcome_metadata,
    evaluate_champion_episode,
    save_episode_hdf5,
)
from starscream.env.tracks import matrix_quaternion
from starscream.gatenet import CameraCalibration, CameraRectifier, GateNetAdapter, checkpoint_sha256, import_factory
from starscream.mpcc import MPCCConfig, MPCCController, RacingLinePlanner, RacingLinePlannerConfig


# Accuracy-qualified all-start timing references.  With the default 1.10 ratio,
# these admit the slowest verified 6 m/s lap while still rejecting stalls and
# long recovery loops.  Speed promotion is qualified separately and must not
# cause perfect, safe reference laps to disappear from the learning shard.
CHAMPION_LAP_REFERENCE_SECONDS = {
    "big_s": 10.04,
    "figure8": 7.95,
    "kidney": 10.79,
    "split_s": 7.15,
    "swift_eval_inspired": 11.72,
    "vertical_3d": 8.10,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/data/debug"))
    parser.add_argument("--track", default="figure8")
    parser.add_argument("--tracks", default=None, help="comma-separated track names")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episodes-per-track", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--next-gates", type=int, default=3)
    parser.add_argument("--control-hz", type=float, default=90.0)
    parser.add_argument(
        "--policy", choices=("hover", "excitation", "gate-chase", "geometric-expert", "mpcc", "agilicious"), default="excitation"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-renderer", action="store_true")
    parser.add_argument("--no-render", action="store_true", help="deprecated alias for --mask-source none")
    parser.add_argument("--mask-source", choices=("geometry", "unity", "none"), default="geometry")
    parser.add_argument("--retain-render-images", action="store_true", help="debug only: save RGB/depth/full segmentation/flow")
    parser.add_argument("--mask-width", type=int, default=160)
    parser.add_argument("--mask-height", type=int, default=128)
    parser.add_argument("--image-delay", type=float, default=0.033, help="seconds from mask capture to observation availability")
    parser.add_argument("--action-delay", type=float, default=0.011, help="seconds from command to simulated application")
    parser.add_argument("--gatenet-factory")
    parser.add_argument("--gatenet-checkpoint", type=Path)
    parser.add_argument("--gatenet-device", default="cuda")
    parser.add_argument("--gatenet-base-channels", type=int)
    parser.add_argument("--gatenet-input-width", type=int)
    parser.add_argument("--gatenet-input-height", type=int)
    parser.add_argument("--camera-calibration", type=Path)
    parser.add_argument("--controller-factory", help="package.module:function returning an Agilicious policy")
    parser.add_argument("--mpcc-backend", choices=("auto", "acados", "predictive"), default="acados")
    parser.add_argument("--mpcc-line-offset-iterations", type=int, default=30)
    parser.add_argument("--random-spawns", action="store_true")
    parser.add_argument("--spawn-min-distance", type=float, default=2.5)
    parser.add_argument("--spawn-max-distance", type=float, default=5.0)
    parser.add_argument("--spawn-max-speed", type=float, default=2.0)
    parser.add_argument("--terminate-on-collision", action="store_true")
    parser.add_argument("--allow-infeasible-track", action="store_true")
    parser.add_argument(
        "--quality-filter",
        choices=("none", "champion"),
        default="none",
        help="drop failed, infeasible, recovery-heavy, or slow MPCC laps before writing",
    )
    parser.add_argument("--champion-lap-time-ratio", type=float, default=1.10)
    parser.add_argument("--champion-max-recovery-fraction", type=float, default=0.10)
    parser.add_argument(
        "--distribution-profile",
        choices=("clean", "generalization"),
        default="clean",
        help="generalization uses an exact expert/recovery/near-crash/crash episode mixture",
    )
    parser.add_argument("--mix-champion", type=float, default=0.40)
    parser.add_argument("--mix-recovery", type=float, default=0.35)
    parser.add_argument("--mix-near-crash", type=float, default=0.15)
    parser.add_argument("--mix-crash", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.no_render:
        args.mask_source = "none"
        args.retain_render_images = False
    needs_renderer = args.mask_source == "unity" or args.retain_render_images
    requested_tracks = [item.strip() for item in (args.tracks or args.track).split(",") if item.strip()]
    if len(requested_tracks) > 1 and needs_renderer:
        # Flightmare Unity 0.0.5 is single-client for the lifetime of the
        # standalone process.  Give every rendered track an isolated renderer
        # rather than attempting an unsupported hot reconnect.
        forwarded: list[str] = []
        skip_next = False
        for argument in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if argument == "--tracks":
                skip_next = True
                continue
            if argument.startswith("--tracks="):
                continue
            if argument == "--track":
                skip_next = True
                continue
            if argument.startswith("--track="):
                continue
            forwarded.append(argument)
        for track in requested_tracks:
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *forwarded, "--track", track],
                check=True,
            )
        return
    renderer = None
    if args.start_renderer and needs_renderer:
        renderer = subprocess.Popen(
            ["/workspace/scripts/run_flightmare_renderer.sh"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    if args.gatenet_factory and not args.gatenet_checkpoint:
        raise SystemExit("--gatenet-factory requires --gatenet-checkpoint")
    segmenter = None
    if args.gatenet_checkpoint:
        gatenet_input_size = (
            (args.gatenet_input_height, args.gatenet_input_width)
            if args.gatenet_input_height and args.gatenet_input_width
            else None
        )
        if args.gatenet_factory:
            segmenter = GateNetAdapter.from_factory(
                args.gatenet_factory,
                args.gatenet_checkpoint,
                output_size=(args.mask_width, args.mask_height),
                input_size=gatenet_input_size or (384, 384),
                device=args.gatenet_device,
            )
        else:
            segmenter = GateNetAdapter.from_checkpoint(
                args.gatenet_checkpoint,
                output_size=(args.mask_width, args.mask_height),
                device=args.gatenet_device,
                base_channels=args.gatenet_base_channels,
                input_size=gatenet_input_size,
            )
        if args.camera_calibration:
            segmenter.preprocessor = CameraRectifier(
                CameraCalibration.from_yaml(args.camera_calibration),
                (args.width, args.height),
            )
        print(f"GateNet checkpoint sha256={checkpoint_sha256(args.gatenet_checkpoint)}")
    if args.policy == "agilicious" and not args.controller_factory:
        raise SystemExit("--policy agilicious requires --controller-factory package.module:function")
    if args.quality_filter == "champion" and args.policy != "mpcc":
        raise SystemExit("--quality-filter champion requires --policy mpcc")
    if args.distribution_profile == "generalization" and args.policy != "mpcc":
        raise SystemExit("--distribution-profile generalization requires --policy mpcc")
    mixture = CollectionMixtureConfig(
        champion=args.mix_champion,
        recovery=args.mix_recovery,
        near_crash=args.mix_near_crash,
        crash=args.mix_crash,
    )
    tracks = requested_tracks
    episodes_per_track = args.episodes_per_track or args.episodes
    env = None
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        for track in tracks:
            if env is not None:
                env.close()
            env = FlightmareEnv(
                track=track.strip(),
                next_gates=args.next_gates,
                image_size=(args.width, args.height),
                control_dt=1.0 / args.control_hz,
                render_observations=needs_renderer,
                mask_source=args.mask_source,
                mask_size=(args.mask_width, args.mask_height),
                retain_render_images=args.retain_render_images,
                image_delay=args.image_delay,
                action_delay=args.action_delay,
                terminate_on_collision=(
                    args.terminate_on_collision or args.distribution_profile == "generalization"
                ),
            )
            geometry = env.track.geometry_report()
            if not geometry["feasible"] and not args.allow_infeasible_track:
                raise SystemExit(
                    f"track {env.track.name!r} failed geometry audit: " + "; ".join(geometry["issues"])
                )
            spawn_sampler = TrajectorySpawnSampler(
                env.track,
                seed=args.seed,
                approach_distance=(args.spawn_min_distance, args.spawn_max_distance),
                forward_speed=(0.0, args.spawn_max_speed),
            )
            controller_factory = import_factory(args.controller_factory) if args.controller_factory else None
            mpcc_policy = None
            mpcc_line = None
            if args.policy == "mpcc":
                mpcc_line = RacingLinePlanner(
                    RacingLinePlannerConfig(
                        offset_iterations=args.mpcc_line_offset_iterations,
                        cache_directory=str(args.output / "racing-lines"),
                    )
                ).plan(env.track)
                mpcc_policy = MPCCController(
                    env.track,
                    mpcc_line,
                    config=MPCCConfig(
                        backend=args.mpcc_backend,
                        actuation_delay=args.action_delay,
                    ),
                )
            diverse_sampler = (
                DiverseScenarioSampler(env.track, mpcc_line, seed=args.seed)
                if args.distribution_profile == "generalization" and mpcc_line is not None
                else None
            )
            regime_schedule = (
                allocate_regimes(episodes_per_track, mixture, seed=args.seed)
                if diverse_sampler is not None
                else tuple(CollectionRegime.CHAMPION for _ in range(episodes_per_track))
            )
            accepted_episodes = 0
            rejected_episodes = 0
            attempt_index = 0
            maximum_attempts = max(episodes_per_track * 10, episodes_per_track + 10)
            while accepted_episodes < episodes_per_track:
                if attempt_index >= maximum_attempts:
                    raise RuntimeError(
                        f"could not collect {episodes_per_track} qualified episodes for "
                        f"{env.track.name} in {maximum_attempts} attempts"
                    )
                episode_index = attempt_index
                regime = regime_schedule[accepted_episodes]
                if args.policy == "hover":
                    policy = HoverPolicy()
                elif args.policy == "gate-chase":
                    policy = GateChasePolicy()
                elif args.policy == "geometric-expert":
                    policy = GeometricExpertPolicy(env.track)
                elif args.policy == "mpcc":
                    policy = (
                        PerturbedControllerPolicy(
                            mpcc_policy,
                            regime,
                            seed=args.seed + 104729 * episode_index,
                        )
                        if diverse_sampler is not None
                        else mpcc_policy
                    )
                elif args.policy == "agilicious":
                    policy = controller_factory(env=env, arguments=args)
                else:
                    policy = SmoothExcitationPolicy(
                        env.control_dt, args.seed + episode_index
                    )
                gate_index = episode_index % len(env.track.gates)
                reset_options = {"gate_index": gate_index}
                if diverse_sampler is not None:
                    state, spawn = diverse_sampler.sample(regime, episode_index)
                    gate_index = int(spawn["gate_index"])
                    reset_options.update(gate_index=gate_index, state=state, spawn=spawn)
                elif args.random_spawns:
                    state, spawn = spawn_sampler.sample(episode_index, gate_index)
                    reset_options.update(state=state, spawn=spawn)
                elif args.quality_filter == "champion" and mpcc_line is not None:
                    rng = np.random.default_rng(args.seed + episode_index)
                    progress = float(mpcc_line.gate_progress[gate_index] - 3.0)
                    frame = mpcc_line.evaluate(progress)
                    state = np.zeros(25, dtype=np.float32)
                    state[0:3] = (
                        frame["position"]
                        + rng.uniform(-0.1, 0.1) * frame["lateral"]
                    )
                    state[3:7] = matrix_quaternion(
                        np.stack([frame["tangent"], frame["lateral"], frame["up"]], axis=1)
                    )
                    state[7:10] = 3.0 * frame["tangent"]
                    reset_options["state"] = state
                if hasattr(policy, "reset"):
                    policy.reset()
                collection_metadata = {
                    "collector_policy": args.policy,
                    "random_spawns": int(args.random_spawns),
                    "gatenet_checkpoint_sha256": (
                        checkpoint_sha256(args.gatenet_checkpoint)
                        if args.gatenet_checkpoint else "none"
                    ),
                    "clock_anchor": "simulator",
                    "control_hz": args.control_hz,
                    "distribution_profile": args.distribution_profile,
                    "distribution_regime": regime.value,
                    "distribution_target_champion": mixture.champion,
                    "distribution_target_recovery": mixture.recovery,
                    "distribution_target_near_crash": mixture.near_crash,
                    "distribution_target_crash": mixture.crash,
                }
                episode = EpisodeCollector(
                    env, policy, segmentation_model=segmenter, metadata=collection_metadata
                ).collect(
                    args.steps,
                    reset_options=reset_options,
                    seed=args.seed + episode_index,
                    stop_after_gates=(
                        len(env.track.gates)
                        if args.quality_filter == "champion" or (
                            diverse_sampler is not None and regime == CollectionRegime.CHAMPION
                        )
                        else None
                    ),
                )
                episode.update(episode_outcome_metadata(episode))
                if not args.retain_render_images:
                    forbidden = {"observation/rgb", "observation/depth", "observation/segmentation", "observation/optical_flow"}
                    leaked = sorted(forbidden.intersection(episode))
                    if leaked:
                        raise RuntimeError(f"mask-only collection leaked direct images: {leaked}")
                if args.quality_filter == "champion" or (
                    diverse_sampler is not None and regime == CollectionRegime.CHAMPION
                ):
                    reference = CHAMPION_LAP_REFERENCE_SECONDS.get(env.track.name)
                    quality = evaluate_champion_episode(
                        episode,
                        expected_gates=len(env.track.gates),
                        maximum_lap_time=(
                            args.champion_lap_time_ratio * reference
                            if reference is not None else None
                        ),
                        maximum_recovery_fraction=args.champion_max_recovery_fraction,
                        track=env.track,
                        racing_line_length=(mpcc_line.length if mpcc_line is not None else None),
                    )
                    episode["metadata/quality_accepted"] = np.asarray(quality.accepted)
                    episode["metadata/quality_elapsed_time"] = np.asarray(
                        quality.elapsed_time, np.float32
                    )
                    episode["metadata/quality_recovery_fraction"] = np.asarray(
                        quality.recovery_fraction, np.float32
                    )
                    if quality.geometry_audit is not None:
                        episode["metadata/quality_minimum_center_clearance"] = np.asarray(
                            quality.geometry_audit.minimum_center_clearance, np.float32
                        )
                        episode["metadata/quality_minimum_body_clearance"] = np.asarray(
                            quality.geometry_audit.minimum_body_clearance, np.float32
                        )
                    episode["metadata/quality_maximum_gate_transition_progress_jump"] = np.asarray(
                        quality.maximum_gate_transition_progress_jump, np.float32
                    )
                    if not quality.accepted:
                        rejected_episodes += 1
                        print(
                            f"dropped {env.track.name} attempt={episode_index} "
                            f"reasons={','.join(quality.reasons)} transitions={quality.transitions}"
                        )
                        attempt_index += 1
                        continue
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                path = args.output / f"{env.track.name}_{stamp}_{episode_index:04d}.h5"
                save_episode_hdf5(path, episode)
                accepted_episodes += 1
                attempt_index += 1
                frames = episode["observation/gate_mask"].shape[0] if "observation/gate_mask" in episode else 0
                size_mb = path.stat().st_size / (1024 * 1024)
                print(
                    f"saved {path} transitions={episode['action/ctbr'].shape[0]} "
                    f"frames={frames} size={size_mb:.1f} MiB "
                    f"regime={regime.value} terminal={bool(np.any(episode['is_terminal']))} "
                    f"reward_sum={float(np.sum(episode['reward/total'])):.3f}"
                )
            print(
                f"track={env.track.name} accepted={accepted_episodes} "
                f"rejected={rejected_episodes} attempts={attempt_index}"
            )
    finally:
        if env is not None:
            env.close()
        if renderer is not None:
            renderer.terminate()
            try:
                renderer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                renderer.kill()
                renderer.wait(timeout=5)


if __name__ == "__main__":
    main()
