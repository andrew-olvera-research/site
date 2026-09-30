"""Runtime contract for policy-gated online racing-manifold expansion.

The geometry generator and MPCC admission sidecar deliberately publish frozen
manifest generations.  This module owns the other half of the contract: PPO
may inspect a newly published bank only between updates, probe candidates with
a frozen policy, and then publish a separate immutable *active* manifest for
the next rollout.  Candidate-bank feasibility is never equivalent to policy
readiness.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .env.procedural_tracks import read_manifest
from .env.racing_manifold import (
    OnlineCandidateEvidence,
    OnlineCurriculumConfig,
    OnlineCurriculumDecision,
    OnlineManifoldCurriculum,
    OnlinePolicyProbe,
    StratifiedCourseProposal,
    stratified_position,
)
from .env.tracks import Track, load_track
from .env.racing_manifold.task_selector import PolicyDependentTaskSelector, completion_gate_count


@dataclass(frozen=True, slots=True)
class OnlineManifestTask:
    name: str
    path: str
    family: str
    operation: str
    qualified_speed_mps: float
    mpcc_eligible: bool
    source_record: Mapping[str, Any]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _record_path(manifest: Path, record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def _mpcc_eligible(record: Mapping[str, Any]) -> bool:
    admission = dict(record.get("mpcc_admission") or {})
    if admission:
        return bool(admission.get("rl_eligible", False))
    qualification = dict(record.get("qualification_suite") or {})
    if qualification:
        return bool(qualification.get("passed", False))
    # Base manifests were sometimes frozen before admission-v2 existed.  They
    # are allowed only when the manifest itself records a successful dynamic
    # qualification; arbitrary unqualified records never receive this default.
    dynamic = record.get("dynamic_qualification")
    return dynamic is not None


class OnlineManifestCurriculum:
    """Checkpointable active-pool controller over frozen manifest banks."""

    def __init__(
        self,
        raw: Mapping[str, Any],
        *,
        restored_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.raw = dict(raw)
        selector_raw = dict(self.raw.get("policy_dependent_selector") or {})
        self.task_selector = PolicyDependentTaskSelector(selector_raw) if selector_raw.get("enabled",False) else None
        self.source_name = str(self.raw["source_name"])
        self.target_name = str(self.raw["target_name"])
        self.active_manifest_path = Path(
            str(self.raw["active_manifest_path"])
        ).resolve()
        self.required_passes = int(self.raw.get("required_consecutive_passes", 2))
        self.lookahead = int(self.raw.get("candidate_lookahead", 1))
        self.max_active_tracks = int(self.raw.get("maximum_active_tracks", 16))
        self.prune_preceding_candidates = bool(
            self.raw.get("prune_preceding_candidates_on_expand", False)
        )
        if min(self.required_passes, self.lookahead, self.max_active_tracks) < 1:
            raise ValueError("online manifold counts must be positive")

        thresholds = OnlineCurriculumConfig(**dict(self.raw.get("thresholds", {})))
        self.controller = OnlineManifoldCurriculum(thresholds)
        self.tasks: dict[str, OnlineManifestTask] = {}
        for value in self.raw.get("manifests", ()):  # earlier manifests win
            self._load_manifest(Path(str(value)).resolve())
        if self.source_name not in self.tasks:
            raise ValueError("online manifold source is absent from its manifests")
        if self.target_name not in self.tasks:
            # The released target may remain outside the admitted bank.  It is
            # used for coordinates, never silently made an active PPO task.
            target_path = Path(str(self.raw["target_track"])).resolve()
            self.target_track = load_track(target_path)
        else:
            self.target_track = load_track(self.tasks[self.target_name].path)
        self.source_track = load_track(self.tasks[self.source_name].path)

        initial = tuple(str(item) for item in self.raw["initial_active_names"])
        missing = [name for name in initial if name not in self.tasks]
        if missing:
            raise ValueError(f"online manifold initial tasks are missing: {missing}")
        candidates = tuple(str(item) for item in self.raw.get("candidate_names", ()))
        if self.raw.get("candidate_names_from_manifest",False):
            candidates=tuple(name for name in self.tasks if name!=self.source_name)
        missing = [name for name in candidates if name not in self.tasks]
        if missing:
            raise ValueError(f"online manifold candidates are missing: {missing}")
        if any(not self.tasks[name].mpcc_eligible for name in initial + candidates):
            raise ValueError("online manifold pool contains a non-MPCC-admitted task")

        self.active_names = list(initial)
        self.candidate_names = list(candidates)
        self.frontier_name = str(self.raw.get("initial_frontier_name", initial[-1]))
        if self.frontier_name not in self.active_names:
            raise ValueError("initial online frontier must be active")
        self.pending_name: str | None = None
        self.pending_passes = 0
        self.generation = 0
        self.trace: list[dict[str, Any]] = []
        self.sample_weights = {
            "anchor_rehearsal": thresholds.anchor_weight,
            "interior_replay": thresholds.interior_weight,
            "frontier": thresholds.frontier_weight,
        }
        if restored_state:
            self.load_state_dict(restored_state)
        self.publish_active_manifest()

    def _load_manifest(self, path: Path) -> None:
        payload = read_manifest(path)
        for record in payload["records"]:
            name = str(record["name"])
            resolved = str(_record_path(path, record))
            task = OnlineManifestTask(
                name=name,
                path=resolved,
                family=str(record.get("family", "online_manifold")),
                operation=str(record.get("operation", "prequalified")),
                qualified_speed_mps=float(record.get("qualified_speed_mps", 16.5)),
                mpcc_eligible=_mpcc_eligible(record),
                source_record=dict(record),
            )
            existing = self.tasks.get(name)
            if existing is not None:
                if load_track(existing.path).fingerprint != load_track(resolved).fingerprint:
                    raise ValueError(f"online task name aliases two geometries: {name}")
                continue
            self.tasks[name] = task

    @property
    def active_paths(self) -> tuple[str, ...]:
        return tuple(self.tasks[name].path for name in self.active_names)

    @property
    def rollout_paths(self) -> tuple[str, ...]:
        """Positive sampling support; keep dormant tasks in the evaluation bank.

        Rehearsal can intentionally set interior replay to zero. Do not give
        those tasks a minimum persistent slot or remove their retained state.
        """
        weights = self.track_sampling_weights()
        if any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("online sampling weights must be finite and nonnegative")
        paths = tuple(self.tasks[n].path for n in self.active_names if weights[n] > 0)
        if not paths:
            raise ValueError("online rollout support must not be empty")
        return paths

    def candidate_probe_names(self) -> tuple[str, ...]:
        if self.task_selector is not None:
            return self.task_selector.probes(self.candidate_names,self.frontier_name,self.lookahead)
        remaining = [name for name in self.candidate_names if name not in self.active_names]
        return tuple(remaining[: self.lookahead])

    def candidate_probe_paths(self) -> tuple[str, ...]:
        return tuple(self.tasks[name].path for name in self.candidate_probe_names())

    def track_sampling_weights(self) -> dict[str, float]:
        weights = dict(self.sample_weights)
        anchor = self.source_name
        frontier = self.frontier_name
        interior = [name for name in self.active_names if name not in {anchor, frontier}]
        result = {name: 1.0e-6 for name in self.active_names}
        if anchor == frontier:
            result[anchor] = weights["anchor_rehearsal"] + weights["frontier"]
        else:
            result[anchor] = weights["anchor_rehearsal"]
            result[frontier] = weights["frontier"]
        if interior:
            share = weights["interior_replay"] / len(interior)
            result.update({name: share for name in interior})
        else:
            result[anchor] += weights["interior_replay"]
        return result

    def publish_active_manifest(self) -> str:
        records: list[dict[str, Any]] = []
        sampling = self.track_sampling_weights()
        for name in self.active_names:
            task = self.tasks[name]
            record = dict(task.source_record)
            record.update({
                "name": name,
                "path": task.path,
                "family": task.family,
                "operation": task.operation,
                "qualified_speed_mps": task.qualified_speed_mps,
                "online_role": (
                    "anchor" if name == self.source_name
                    else "frontier" if name == self.frontier_name
                    else "interior"
                ),
                "online_sampling_weight": sampling[name],
            })
            records.append(record)
        payload = {
            "schema": ("starscream-goal-conditioned-task-manifest-v1" if self.task_selector is not None or self.raw.get('goal_conditioned_manifest',False)
                       else "starscream-qualified-reference-training-manifest-v1"),
            "generator": "policy-gated-online-manifold-curriculum-v1",
            "generation": self.generation,
            "records": records,
            "freeze": {
                "immutable_within_ppo_rollout_update": True,
                "source_name": self.source_name,
                "frontier_name": self.frontier_name,
            },
        }
        digest = _atomic_json(self.active_manifest_path, payload)
        _atomic_json(self.active_manifest_path.with_name("current.json"), {
            "schema": "starscream-online-active-pool-pointer-v1",
            "generation": self.generation,
            "manifest": str(self.active_manifest_path),
            "manifest_sha256": digest,
        })
        return digest

    @staticmethod
    def _metric(metrics: Mapping[str, float], name: str, key: str, default: float) -> float:
        return float(metrics.get(f"track/{name}/{key}", default))

    def decide(
        self,
        probe_metrics: Mapping[str, float],
        active_metrics: Mapping[str, float],
        *,
        cycle: int,
        reporting_metrics: Mapping[str, float] | None = None,
        environment_steps: int = 0,
    ) -> tuple[OnlineCurriculumDecision, bool]:
        candidates: list[OnlineCandidateEvidence] = []
        for name in self.candidate_probe_names():
            task = self.tasks[name]
            track = load_track(task.path)
            success = self._metric(probe_metrics, name, "full_course_success", 0.0)
            gates = self._metric(probe_metrics, name, "mean_gates", 0.0)
            episodes = max(int(self._metric(probe_metrics, name, "episodes", 1.0)), 1)
            gate_fraction = float(np.clip(gates / max(len(track.gates), 1), 0.0, 1.0))
            probe = OnlinePolicyProbe(
                episodes=episodes,
                success_rate=float(np.clip(success, 0.0, 1.0)),
                mean_gate_fraction=gate_fraction,
                robust_score=float(np.clip(0.75 * success + 0.25 * gate_fraction, 0.0, 1.0)),
            )
            proposal = StratifiedCourseProposal(
                track=track,
                operation=task.operation,
                position=stratified_position(self.source_track, self.target_track, track),
                local_route_proposal=None,
                valid=True,
                reasons=(),
            )
            candidates.append(OnlineCandidateEvidence(proposal, task.mpcc_eligible, probe))

        source_retention = self._metric(
            probe_metrics, self.source_name, "full_course_success",
            self._metric(active_metrics, self.source_name, "full_course_success", 0.0),
        )
        frontier_success = self._metric(
            active_metrics, self.frontier_name, "full_course_success", 0.0,
        )
        decision = self.controller.decide(
            candidates,
            source_retention=source_retention,
            current_frontier_success=frontier_success,
            target_reached=not self.candidate_probe_names(),
        )
        if bool(self.raw.get('ordered_geometry_ladder', False)):
            from .course_model.traversal import ordered_geometry_choice
            observations=[(name,
                self._metric(probe_metrics,name,'full_course_success',0.),
                self._metric(probe_metrics,name,'mean_gates',0.)/len(load_track(self.tasks[name].path).gates),
                self._metric(probe_metrics,name,'episodes',0.)) for name in self.candidate_probe_names()]
            cfg=self.controller.config
            action,selected=ordered_geometry_choice(source_retention,frontier_success,observations,
                source_floor=cfg.source_retention_floor,frontier_floor=cfg.frontier_success_floor,
                mastery=cfg.mastery_success,minimum_gate_fraction=cfg.minimum_gate_fraction,
                minimum_episodes=int(self.raw.get('probe_episodes_per_track',16)))
            decision=OnlineCurriculumDecision(action,selected,'ordered physical-displacement competence ladder',
                self.controller._weights(source_retention,rehearse=action in {'rehearse','retreat'}))
        if bool(self.raw.get('classical_goal_ladder', False)):
            from .env.racing_manifold.classical_tasks import classical_ladder_choice
            names=self.candidate_probe_names()
            selected=names[0] if names else None
            goal=(completion_gate_count(load_track(self.tasks[selected].path),100,enabled=True)
                  if selected else 1)
            action=classical_ladder_choice(source_success=source_retention,
                frontier_success=frontier_success,
                candidate_success=self._metric(probe_metrics,selected,'full_course_success',0),
                candidate_episodes=self._metric(probe_metrics,selected,'episodes',0),
                candidate_fraction=self._metric(probe_metrics,selected,'mean_gates',0)/goal,
                source_floor=self.controller.config.source_retention_floor,
                mastery=self.controller.config.mastery_success,
                candidate_floor=self.controller.config.frontier_success_floor)
            decision=OnlineCurriculumDecision(action, selected if action=='expand' else None,
                'ordered classical completion-goal prerequisite',
                self.controller._weights(source_retention,rehearse=action=='rehearse'))
        if self.task_selector is not None:
            selector=self.task_selector
            selector.observe(probe_metrics,{name:completion_gate_count(load_track(self.tasks[name].path),100,enabled=True)
                             for name in self.candidate_names},cycle)
            selector.record_response(reporting_metrics,target=self.target_name,source=self.source_name,
                frontier=self.frontier_name,record=self.tasks[self.frontier_name].source_record,
                steps=environment_steps)
            selected=selector.select({name:self.tasks[name].source_record for name in self.candidate_names
                                     if self.tasks[name].mpcc_eligible},
                                     active=self.active_names,frontier=self.frontier_name,cycle=cycle)
            # Keep the existing source/frontier safety guards ahead of scoring.
            if source_retention < self.controller.config.source_retention_floor:
                selected=None
                action="rehearse"
            elif frontier_success < self.controller.config.frontier_success_floor:
                selected=None
                action="retreat"
            else:
                action="expand" if selected else "hold"
            decision=OnlineCurriculumDecision(action,selected,"policy-dependent competence/target-response selector",
                self.controller._weights(source_retention,rehearse=action in {"retreat","rehearse"}))
        changed = False
        if decision.action == "expand" and decision.selected_name is not None:
            if self.pending_name == decision.selected_name:
                self.pending_passes += 1
            else:
                self.pending_name = decision.selected_name
                self.pending_passes = 1
            if self.pending_passes < self.required_passes:
                decision = OnlineCurriculumDecision(
                    "hold", None,
                    f"candidate passed {self.pending_passes}/{self.required_passes} frozen-policy probes",
                    decision.sample_weights,
                )
            else:
                selected = self.pending_name
                assert selected is not None
                if self.prune_preceding_candidates:
                    selected_index = self.candidate_names.index(selected)
                    # A farther ordered rung passing the same frozen-policy
                    # probe dominates easier lookahead siblings.  Retaining
                    # those siblings would waste every later probe slot and
                    # turn a smaller geometric step into a slower cadence.
                    self.candidate_names = self.candidate_names[selected_index:]
                if selected not in self.active_names:
                    self.active_names.append(selected)
                self.frontier_name = selected
                if self.task_selector is not None:
                    self.task_selector.last_switch=cycle
                    if self.task_selector.previous:
                        self.task_selector.previous["frontier"]=selected
                        self.task_selector.previous["features"]=self.task_selector.features(
                            self.tasks[selected].source_record,self.task_selector.observations.get(selected,{})).tolist()
                self.pending_name = None
                self.pending_passes = 0
                if len(self.active_names) > self.max_active_tracks:
                    removable = [
                        name for name in self.active_names
                        if name not in {self.source_name, self.frontier_name}
                    ]
                    self.active_names.remove(removable[0])
                self.generation += 1
                changed = True
        else:
            self.pending_name = None
            self.pending_passes = 0
            if decision.action == "retreat" and self.frontier_name not in set(
                self.raw["initial_active_names"]
            ):
                self.active_names.remove(self.frontier_name)
                self.frontier_name = self.active_names[-1]
                if self.task_selector is not None:
                    self.task_selector.previous = None
                    self.task_selector.last_switch = cycle
                self.generation += 1
                changed = True
        weight_changed = (
            set(decision.sample_weights) != set(self.sample_weights)
            or any(
                not np.isclose(value, self.sample_weights[key], atol=1.0e-12)
                for key, value in decision.sample_weights.items()
            )
        )
        if weight_changed:
            self.sample_weights = dict(decision.sample_weights)
            changed = True
        if changed:
            self.publish_active_manifest()
        self.trace.append({
            "cycle": int(cycle),
            "source_retention": source_retention,
            "frontier_success": frontier_success,
            "probe_names": [item.proposal.track.name for item in candidates],
            "decision": decision.to_mapping(),
            "active_names": list(self.active_names),
            "frontier_name": self.frontier_name,
            "selector_scores": [] if self.task_selector is None else self.task_selector.last_scores,
        })
        return decision, changed

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "starscream-online-manifold-curriculum-state-v1",
            "task_selector": None if self.task_selector is None else self.task_selector.state_dict(),
            "active_names": list(self.active_names),
            "candidate_names": list(self.candidate_names),
            "frontier_name": self.frontier_name,
            "pending_name": self.pending_name,
            "pending_passes": self.pending_passes,
            "generation": self.generation,
            "sample_weights": dict(self.sample_weights),
            "trace": list(self.trace[-128:]),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "starscream-online-manifold-curriculum-state-v1":
            raise ValueError("unsupported online manifold curriculum state")
        active = [str(item) for item in state["active_names"]]
        if any(name not in self.tasks for name in active):
            raise ValueError("restored online active pool is absent from configured banks")
        self.active_names = active
        candidates=[str(x) for x in state.get("candidate_names",self.candidate_names)]
        if any(name not in self.tasks for name in candidates):
            raise ValueError("restored candidates absent from bank")
        self.candidate_names=candidates
        if self.task_selector is not None and state.get("task_selector"):
            self.task_selector.load_state_dict(state["task_selector"])
        self.frontier_name = str(state["frontier_name"])
        self.pending_name = (
            None if state.get("pending_name") is None else str(state["pending_name"])
        )
        self.pending_passes = int(state.get("pending_passes", 0))
        self.generation = int(state.get("generation", 0))
        self.sample_weights = {
            str(key): float(value)
            for key, value in dict(state.get("sample_weights", self.sample_weights)).items()
        }
        self.trace = [dict(item) for item in state.get("trace", ())]
