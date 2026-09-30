"""Freeze the one-train/two-validation slalom recovery mini-benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train_privileged_racing import configured_tracks, load_config, stage_config
from starscream.dagger_quality import quality_config
from starscream.evaluation_suite import configure_selection_suite


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/exp/v6.21.1.1"
TRAIN = "/workspace/outputs/course-pools/v6201-transitions-r6/frozen/train/v6201_train_10_slalom_0.yaml"
SLOTS = ("real100-hard-010", "real100-hard-019")
SUITE_REL = "configs/eval/v62111_mini_slalom_selection2.json"
PROTOCOL = ROOT / "configs/eval/real100_v2_timed_protocol_v1.json"
SOURCE_SUITE = ROOT / "configs/eval/v6211_vision_selection25.json"
QUALITY = CONFIG_DIR / "plant_quality_v1.yaml"
SOURCE_TRAIN_MANIFEST = ROOT / "outputs/diagnostics/v62111-train65-teacher-v4/frozen/approved/manifest.json"
MINI_TRAIN_MANIFEST = CONFIG_DIR / "mini_slalom_train_manifest.json"
COMMON = CONFIG_DIR / "mini_slalom_common.yaml"
BROAD = CONFIG_DIR / "mini_slalom_broad.yaml"
BOUNDED = CONFIG_DIR / "mini_slalom_bounded.yaml"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare() -> None:
    source = json.loads(SOURCE_SUITE.read_text())
    records = [r for r in source["records"] if r["slot"] in SLOTS]
    assert len(records) == 2 and {r["slot"] for r in records} == set(SLOTS)
    protocol = json.loads(PROTOCOL.read_text())
    deadlines = {r["name"]: r["deadline_steps"] for r in protocol["records"] if r["slot"] in SLOTS}
    assert set(deadlines) == {r["name"] for r in records}
    suite = dict(schema="starscream-vision-selection-v1", source=source["source"],
                 source_sha256=source["source_sha256"], method="fixed two-course slalom subset of selection25",
                 family_quotas={"slalom": 2}, family_weights={"slalom": 1.0},
                 records=records)
    suite_path = ROOT / SUITE_REL
    write_json(suite_path, suite)
    suite_sha = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    train_manifest = json.loads(SOURCE_TRAIN_MANIFEST.read_text())
    train_records = [r for r in train_manifest["records"] if r["path"] == TRAIN]
    assert len(train_records) == 1 and train_records[0]["qualified"]
    train_manifest["records"] = train_records
    train_manifest["family_weights"] = {"10_slalom": 1.0}
    write_json(MINI_TRAIN_MANIFEST, train_manifest)
    quality = json.loads(QUALITY.read_text())["dagger"]["dagger_trajectory_quality"]
    assert TRAIN in quality["track_bounds"]

    # Original update density is 5081/520 = 9.77 updates per collected episode,
    # 67714/520 = 130 retained online rows per episode. The mini run keeps both
    # ratios while preserving the full 1536-row optimizer batch and online mix.
    # 32 rounds x 32 episodes is ~1M environment steps at ~1000 steps/episode.
    common = {
        "inherits": "plant_dagger.yaml",
        "dagger": {
            "seed": 2026092112, "rounds": 32, "episodes_per_round": 32,
            "rollout_envs": 16, "updates_per_round": 313, "batch_size": 1536,
            "dagger_online_replay_rows_per_round": 4167,
            "online_replay_capacity": 130000,
            "dagger_permanent_expert_capacity": 26043,
            "dagger_permanent_expert_rounds": 2,
            "dagger_dart_expert_rounds": 2,
            "dagger_successful_coverage_rounds": 0,
            "dagger_require_successful_episodes": False,
            "teacher_beta_schedule_rounds": 5,
            "dagger_learning_rate_schedule": {"warmup_rounds": 2},
            "curriculum": {"tracks": [TRAIN]},
            "track_sampling_family_weights": {"__replace__": True, "10_slalom": 1.0},
            "dagger_replay_family_weights": {"__replace__": True, "10_slalom": 1.0},
            "dagger_permanent_expert_required_families": ["10_slalom"],
            "dagger_dynamic_sampling": {"enabled": False},
            "track_manifest": "/workspace/" + str(MINI_TRAIN_MANIFEST.relative_to(ROOT)),
            "evaluation_suite_manifest": SUITE_REL,
            "evaluation_suite_sha256": suite_sha,
            "evaluation_suite_episodes_per_track": 8,
            "evaluation_curriculum": {"tracks": ["/workspace/" + r["path"] for r in records], "max_steps": 6000},
            "evaluation_clean_deadlines": deadlines,
            "evaluation_workers": 2,
            "evaluation_envs_per_worker": 8,
            "ppo_reference_aware_gate_events": True,
            "reporting_evaluation_curriculum": None,
            "top_k": 3,
        },
        "experiment_notes": {
            "mini_recovery_baseline": {
                "train": TRAIN, "validation_slots": list(SLOTS),
                "budget": "32 rounds x 32 episodes; expected ~1M environment steps, measured steps are authoritative",
                "update_scaling": "313 updates and 4167 retained rows per round preserve plant 520-episode ratios",
                "warmup": "2 initial teacher/DART rounds; beta reaches 0.35 by round 5",
                "comparison": "Only trajectory quality collection/replay package differs between broad and bounded",
            }
        },
    }
    write_json(COMMON, common)

    for name, path in (("broad", BROAD), ("bounded", BOUNDED)):
        run = f"starscream-v6.21.1.1-mini-slalom-{name}-r32"
        dagger = {"run_name": run, "tags": ["mini-slalom", "one-train-two-val", name,
                  "async-dagger", "scratch"]}
        if name == "bounded":
            dagger["dagger_trajectory_quality"] = quality
        write_json(path, {
            "inherits": COMMON.name,
            "dagger": dagger,
            "wandb": {"group": "v62111-mini-slalom-recovery-baselines",
                      "name": run,
                      "local_event_path": f"/workspace/outputs/logs/{run}.events.jsonl",
                      "local_full_event_path": f"/workspace/outputs/logs/{run}.full.events.jsonl",
                      "eval_metric_allowlist": ["selection_suite_success", "selection_suite_timely_success",
                                                "selection_suite_clean_success", "selection_suite_clean_timely_success",
                                                "full_course_success", "crash_rate", "mean_gates", "mean_steps",
                                                "dagger_policy_version"]},
        })

    settings = []
    for path in (BROAD, BOUNDED):
        config = stage_config(load_config(path), "dagger")
        d = config["dagger"]
        configure_selection_suite(d)
        assert d["dagger_async_pipeline"] and d["dagger_capture_updates"]
        assert d["dagger_packed_transport"] and d["mpcc_native_model_step"]
        assert d["dagger_overlap_teacher_inference"] and d["dagger_lazy_teacher_actions"]
        assert d["evaluation_episodes"] == 16 and len(d["curriculum"]["tracks"]) == 1
        assert configured_tracks(d) == (TRAIN,)
        assert d["evaluation_suite_resolved_sha256"] == suite_sha
        assert (quality_config(d) is not None) == (path == BOUNDED)
        settings.append(d)
    different = {k for k in settings[0].keys() | settings[1].keys()
                 if settings[0].get(k) != settings[1].get(k)}
    assert different == {"run_name", "tags", "dagger_trajectory_quality"}, different
    print(json.dumps({"configs": [str(BROAD.relative_to(ROOT)), str(BOUNDED.relative_to(ROOT))],
                      "suite_sha256": suite_sha, "train": TRAIN,
                      "validation": {r["slot"]: r["path"] for r in records},
                      "deadlines": deadlines, "only_differences": sorted(different)}, indent=2))


if __name__ == "__main__":
    prepare()
