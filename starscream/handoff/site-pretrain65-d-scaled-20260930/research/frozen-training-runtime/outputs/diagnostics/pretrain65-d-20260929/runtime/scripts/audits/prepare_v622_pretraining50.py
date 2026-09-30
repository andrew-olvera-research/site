"""Prepare the v6.22 scratch DAgger setup for the admitted 50-course panel."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from starscream.course_model.training import atomic_json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/exp/v6.21/scratch_5m_dagger_r218.yaml"
MANIFEST = ROOT / "outputs/v622-pretraining50/manifest.json"
TUNED_MANIFEST = ROOT / "outputs/v622-pretraining50/mpcc-frontier-v3/manifest.json"
TEACHER_PROFILES = ROOT / "outputs/v622-pretraining50/mpcc-frontier-v3/teacher-profiles.json"
OUT = ROOT / "configs/exp/v6.22/pretraining_behavior50_dagger.yaml"


def main():
    if not TUNED_MANIFEST.exists() or not TEACHER_PROFILES.exists():
        raise FileNotFoundError("complete MPCC frontier qualification before preparing v6.22")
    manifest = json.loads(TUNED_MANIFEST.read_text())
    profiles = json.loads(TEACHER_PROFILES.read_text())
    rows = manifest["records"]
    if len(rows) != 50 or not all(r.get("qualified_speed_mps") for r in rows):
        raise ValueError("pretraining50 must be exactly 50 MPCC-admitted courses before preparing training")
    cfg = yaml.safe_load(BASE.read_text())
    s = cfg["dagger"]
    train_paths = [str(r["path"]) if str(r["path"]).startswith("/") else "/workspace/" + str(r["path"]) for r in rows]
    families = [str(r["family"]) for r in rows]
    weights = {family: 1.0 for family in families}
    aliases = {family: str(r["label"]) for r in rows for family in [str(r["family"])]}
    n = len(rows)
    episodes_per_course = 8
    episodes = n * episodes_per_course
    # Calibrated from stopped-run rounds 8-12, the window that actually ran at
    # the restored beta ~= .35: 4,990,719 transitions across 2,000 episodes.
    # Round to the nearest whole round; 75 rounds projects to 1.497M/course.
    mean_steps = 4990719 / 2000
    target_steps_per_course = 1.5e6
    rounds = int(round(target_steps_per_course * n / (episodes * mean_steps)))
    name = "starscream-v6.22-base-pretrain50"
    # Keep the winning v6.21 scratch-5M DAgger recipe. Change only the frozen
    # corpus, course-count-scaled budgets, course-specific teacher profiles,
    # held-out panels, timeouts, and measured 1.5M/course round count.
    s.update(
        run_name=name, tags=["v6.22", "pretraining", "behavior-cells", "50-courses", "a2rl-technical", "v6.21-optimal-dagger"],
        initial_checkpoint=None, resume_checkpoint=None, dagger_anchor_checkpoint=None,
        rounds=rounds, episodes_per_round=episodes,
        updates_per_round=int(math.ceil(2026 * n / 45)),
        online_fraction=.65,
        online_replay_capacity=int(math.ceil(1575001 * n / 45)),
        dagger_online_replay_rows_per_round=int(math.ceil(27001 * n / 45)),
        dagger_permanent_expert_rounds=4, dagger_permanent_expert_fraction=.35,
        dagger_permanent_expert_capacity=int(math.ceil(675000 * n / 45)),
        dagger_dart_expert_rounds=4, dagger_dart_expert_episode_fraction=.50,
        dagger_successful_coverage_rounds=4,
        dagger_dart_exempt_tracks=[
            (str(r["path"]) if str(r["path"]).startswith("/") else "/workspace/" + str(r["path"]))
            for r in rows if not r.get("dart_eligible", False)
        ],
        track_manifest="/workspace/outputs/v622-pretraining50/mpcc-frontier-v3/manifest.json", track_split="train", qualified_tracks_only=True,
        mpcc_use_manifest_speed=True, mpcc_manifest_speed_scale=1.0,
        # Admission stores the selected profile under the protocol evidence,
        # but this panel intentionally uses one reproducible controller recipe
        # during data collection; the per-course frontier still comes from
        # qualified_speed_mps.  Profile-specific tuning remains in the
        # qualification artifacts for a later high-speed collection pass.
        mpcc_use_manifest_teacher_profile=True, mpcc_use_manifest_planner_profile=True,
        dagger_condition_on_teacher_speed=True, evaluation_condition_on_manifest_speed=True,
        ppo_manifest_speed_field="qualified_speed_mps",
        track_sampling_family_weights=weights, dagger_replay_family_weights=weights,
        reliability_family_aliases=aliases, dagger_permanent_expert_required_families=sorted(weights),
        dagger_initialization_stats_checkpoint="/workspace/outputs/v622-pretraining50/normalization.pt",
        racing_line_cache="/workspace/outputs/v622-pretraining50/racing-lines",
        mpcc_build_root="/tmp/starscream-v622-pretraining50",
        curriculum=dict(s["curriculum"], name="v622_pretraining50", tracks=train_paths, max_steps=10000),
    )
    s["mpcc_manifest_teacher_profile_controller_configs"].update(profiles["controller_profiles"])
    s["mpcc_manifest_teacher_profile_planner_configs"].update(profiles["planner_profiles"])
    # real60 selects checkpoints; the disjoint real100-hard suite reports only.
    real60 = yaml.safe_load((ROOT / "configs/eval/v6_22_real60.yaml").read_text())
    real100 = yaml.safe_load((ROOT / "configs/eval/v6_22_real100_hard.yaml").read_text())
    suite_paths = lambda suite: ["/workspace/" + str(r["track"]).lstrip("/") for r in suite["active"]]
    val_paths, report_paths = suite_paths(real60), suite_paths(real100)
    evaluation_speeds = {}
    for filename in ('v6_22_real60.manifest.json', 'v6_22_real100_hard.manifest.json'):
        for row in json.loads((ROOT / 'configs/eval' / filename).read_text())['records']:
            path = str(row['path'])
            path = path if path.startswith('/') else '/workspace/' + path
            q = row['qualification']
            evaluation_speeds[path] = float(q.get('speed_command', q.get('candidate', {}).get('speed')))
    assert set(evaluation_speeds) == set(val_paths + report_paths)
    s['evaluation_track_speed_commands'] = evaluation_speeds
    # This frozen panel has no canonical source-family correspondence to real60.
    # The inherited adaptive sampler would fail at startup or invent gate-1
    # frontiers from absent competence observations. Keep balanced coverage.
    s['dagger_dynamic_sampling']['enabled'] = False
    s["evaluation_curriculum"] = dict(s["evaluation_curriculum"], name="v622_real60_validation", tracks=val_paths, max_steps=6000)
    s["reporting_evaluation_curriculum"] = dict(s["reporting_evaluation_curriculum"], name="v622_real100_periodic", tracks=report_paths, max_steps=6000)
    s.update(evaluation_episodes=len(val_paths) * 4, reporting_evaluation_episodes=len(report_paths) * 2,
             evaluation_workers=12, evaluation_envs_per_worker=2,
             reporting_evaluation_interval=10, monitor="full_course_success", monitor_mode="max", top_k=5,
             checkpoint_interval=1)
    cfg["checkpoint"].update(run_name=name, monitor="full_course_success", mode="max", top_k=5)
    cfg["wandb"].update(run_name=name, group="starscream-v6.22-pretraining", tags=s["tags"], local_event_path=f"/workspace/outputs/logs/{name}.events.jsonl")
    cfg["experiment_notes"] = {
        "status": "PREPARED; run statistics bootstrap before training",
        "dataset": "outputs/v622-pretraining50/mpcc-frontier-v3/manifest.json",
        "courses": 50, "technical_courses": 16, "target_steps_per_course": target_steps_per_course,
        "budget": f"{rounds} rounds x {episodes} episodes; original 690-step estimate is not reliable for this panel; measure actual collected steps",
        "cell_plan": "outputs/v622-pretraining50/cell-plan.json",
        "admission": manifest.get("admission"),
        "sampling": "balanced family/course coverage and hierarchical replay; adaptive sampler disabled without canonical source-family competence correspondence; real60 selects, real100 reports only",
        "speed_conditioning": "v6.21-optimal behavior: one qualified course-local teacher command; speed is observed by the actor, while downstream RL owns the time/speed objective",
        "dagger_recipe": "v6.21 scratch-5M: beta 1->.35 over 20 rounds, .65 online/.35 permanent expert, replay .40/.35/.25 nominal/critical/recovery, four-round DART and successful-coverage bootstrap, expert-prefix targeted starts",
        "normalization_prerequisite": "scripts/audits/collect_dagger_initialization_statistics.py --config this-config --output outputs/v622-pretraining50/normalization.pt --episodes 100",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cfg, indent=2) + "\n")
    atomic_json(OUT.with_suffix(".preflight.json"), {
        "config_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(TUNED_MANIFEST.read_bytes()).hexdigest(),
        "train_courses": n, "episodes_per_round": episodes, "rounds": rounds,
        "updates_per_round": s["updates_per_round"], "technical_courses": 16,
        "normalization_path": s["dagger_initialization_stats_checkpoint"],
        "validation_courses": len(val_paths), "real100_reporting_courses": len(report_paths),
        "evaluation_episodes": s["evaluation_episodes"],
        "reporting_evaluation_episodes": s["reporting_evaluation_episodes"],
        "hardware_profile": "configs/hardware/dagger_local_16c_v625.yaml",
        "hardware_profile_sha256": hashlib.sha256((ROOT / "configs/hardware/dagger_local_16c_v625.yaml").read_bytes()).hexdigest(),
        "normalization_sha256": (
            hashlib.sha256(Path(s["dagger_initialization_stats_checkpoint"]).read_bytes()).hexdigest()
            if Path(s["dagger_initialization_stats_checkpoint"]).exists() else None
        ),
    })
    print(json.dumps({"config":str(OUT),"courses":n,"rounds":rounds,
        "episodes_per_round":episodes,"updates_per_round":s["updates_per_round"],
        "validation_courses":len(val_paths),"reporting_courses":len(report_paths)},indent=2))


if __name__ == "__main__": main()
