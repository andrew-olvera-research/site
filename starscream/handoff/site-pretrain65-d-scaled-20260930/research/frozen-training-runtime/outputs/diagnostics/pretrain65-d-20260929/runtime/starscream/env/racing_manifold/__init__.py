"""Geometry-manifold analysis and controlled course expansion.

This package deliberately depends on the lower-level racing-distribution
descriptors and track contracts.  It does not generate training manifests or
admit tracks into a curriculum; qualification remains a separate step.
"""

from .atlas import ManifoldAtlas, SupportDiagnostic
from .admission import (
    BehavioralAdmissionConfig,
    CandidateAdmissionDecision,
    CandidateAdmissionEvidence,
    evaluate_candidate_admission,
)
from .behavior import (
    BehaviorManifoldReport,
    analyze_evaluation_payloads,
    analyze_phase_probe,
    extract_track_metrics,
)
from .descriptors import (
    FEATURE_NAMES,
    TrackGeometryProfile,
    analyze_track_geometry,
    ordered_transition_signature,
)
from .generator import (
    ManifoldExtensionProposal,
    RouteExtensionProposal,
    RouteSplineManifoldGenerator,
    SplineManifoldGenerator,
)
from .morph import MorphAlignment, PhaseAlignedTrackMorpher
from .occupancy import (
    DEFAULT_BLOCK_WEIGHTS,
    BehavioralEmbeddingModel,
    BehavioralPosition,
    PhaseAlignment,
    TrajectoryBehaviorProfile,
    TrajectoryDistribution,
    align_phase_signatures,
    behavioral_position,
    corrected_target_embedding,
    replay_trajectory_distributions,
)
from .primitives import (
    PrimitiveProfile,
    analyze_primitives,
    graft_primitive,
    primitive_coverage,
)
from .curriculum import CoverageSnapshot, ExpansionStep, ManifoldCurriculumHarness
from .frontier import FrontierCalibration, calibrate_frontier, wilson_interval
from .expert_qualification import (
    MPCCAdmissionDecision,
    MPCCAdmissionThresholds,
    MPCCRegimeEvidence,
    evaluate_mpcc_admission,
    evaluate_record_dynamic_qualification,
)
from .transition import DirectedPosition, DirectedTransitionModel, FeatureShift
from .online import (
    ManifoldStratum,
    OnlineCandidateEvidence,
    OnlineCurriculumConfig,
    OnlineCurriculumDecision,
    OnlineManifoldCurriculum,
    OnlinePolicyProbe,
    StratifiedCourseProposal,
    StratifiedPosition,
    StratifiedRouteGenerator,
    manifold_stratum,
    stratified_position,
)
from .tube import CourseTubeCandidate, CourseTubeConfig, DirectionalCourseTube
from .validation import (
    CandidateTransferValidation,
    GeneratorTransferScorecard,
    PolicyCohortObservation,
    RobustPolicyOutcome,
    aggregate_policy_cohorts,
    score_generator_transfer,
)
from .route_grammar import (
    RouteDirectedPosition,
    RouteGrammarProfile,
    route_directed_position,
    route_grammar_alignment,
    route_grammar_profile,
)

__all__ = [
    "BehaviorManifoldReport",
    "BehavioralAdmissionConfig",
    "FEATURE_NAMES",
    "CoverageSnapshot",
    "ExpansionStep",
    "FrontierCalibration",
    "MPCCAdmissionDecision",
    "MPCCAdmissionThresholds",
    "MPCCRegimeEvidence",
    "DirectedPosition",
    "DirectedTransitionModel",
    "DirectionalCourseTube",
    "CourseTubeCandidate",
    "CourseTubeConfig",
    "CandidateAdmissionDecision",
    "CandidateAdmissionEvidence",
    "CandidateTransferValidation",
    "FeatureShift",
    "ManifoldAtlas",
    "ManifoldCurriculumHarness",
    "ManifoldExtensionProposal",
    "ManifoldStratum",
    "RouteExtensionProposal",
    "RouteSplineManifoldGenerator",
    "MorphAlignment",
    "PhaseAlignedTrackMorpher",
    "PrimitiveProfile",
    "OnlineCandidateEvidence",
    "OnlineCurriculumConfig",
    "OnlineCurriculumDecision",
    "OnlineManifoldCurriculum",
    "OnlinePolicyProbe",
    "PolicyCohortObservation",
    "RobustPolicyOutcome",
    "GeneratorTransferScorecard",
    "SplineManifoldGenerator",
    "SupportDiagnostic",
    "StratifiedCourseProposal",
    "StratifiedPosition",
    "StratifiedRouteGenerator",
    "TrackGeometryProfile",
    "TrajectoryBehaviorProfile",
    "TrajectoryDistribution",
    "BehavioralEmbeddingModel",
    "BehavioralPosition",
    "PhaseAlignment",
    "DEFAULT_BLOCK_WEIGHTS",
    "analyze_evaluation_payloads",
    "analyze_phase_probe",
    "analyze_primitives",
    "aggregate_policy_cohorts",
    "RouteGrammarProfile",
    "RouteDirectedPosition",
    "route_directed_position",
    "route_grammar_alignment",
    "route_grammar_profile",
    "score_generator_transfer",
    "analyze_track_geometry",
    "align_phase_signatures",
    "behavioral_position",
    "calibrate_frontier",
    "corrected_target_embedding",
    "extract_track_metrics",
    "evaluate_candidate_admission",
    "evaluate_mpcc_admission",
    "evaluate_record_dynamic_qualification",
    "graft_primitive",
    "manifold_stratum",
    "ordered_transition_signature",
    "primitive_coverage",
    "replay_trajectory_distributions",
    "stratified_position",
    "wilson_interval",
]
