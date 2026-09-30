"""Zero-shot racing-task distribution engine.

The public surface is intentionally small.  Generation, validation, and task
selection remain separate so a candidate cannot enter training merely because
it was generated successfully.
"""

from .archive import ArchiveEntry, DescriptorGrid, QualityDiversityArchive
from .curriculum import TaskLearningState, adaptive_task_probabilities
from .comparison import compare_distribution_manifests, write_distribution_gallery
from .descriptors import (
    CourseDescriptor, course_descriptor, descriptor_distance,
    transition_descriptors,
)
from .generator import RacingTaskGenerator
from .flight_mimic import (
    FlightMimicComposer, FlightSegment, FlightSegmentLibrary,
    build_segment_library, compose_valid_candidates, compose_valid_programs,
    extract_track_segments,
)
from .grammar import ManeuverGrammar, cyclic_ngrams
from .named_course_augmentation import (
    NamedCourseAugmentationConfig, audit_named_course_augmentation,
    augment_named_course, generate_named_course_augmentations,
)
from .manifest import (
    freeze_distribution_manifest, generate_distribution_manifest,
    load_distribution_manifest, merge_qualification_evidence,
    select_qualified_distribution_manifest,
)
from .schema import (
    CourseProgram, GeneratorBackend, ManeuverKind, ManeuverSpec,
    RacingDistributionConfig,
)
from .validator import (
    DistributionValidationReport, RacingDistributionValidator,
    TrackValidationReport,
)

__all__ = [
    "ArchiveEntry", "CourseDescriptor", "CourseProgram", "DescriptorGrid",
    "DistributionValidationReport", "FlightMimicComposer", "FlightSegment",
    "FlightSegmentLibrary", "GeneratorBackend", "ManeuverGrammar",
    "ManeuverKind", "ManeuverSpec", "QualityDiversityArchive",
    "NamedCourseAugmentationConfig",
    "RacingDistributionConfig", "RacingDistributionValidator",
    "RacingTaskGenerator", "TaskLearningState", "TrackValidationReport",
    "adaptive_task_probabilities", "build_segment_library",
    "audit_named_course_augmentation", "augment_named_course",
    "compose_valid_candidates", "compose_valid_programs", "course_descriptor",
    "cyclic_ngrams",
    "compare_distribution_manifests",
    "descriptor_distance", "generate_distribution_manifest",
    "generate_named_course_augmentations",
    "freeze_distribution_manifest",
    "extract_track_segments", "load_distribution_manifest", "merge_qualification_evidence",
    "transition_descriptors",
    "select_qualified_distribution_manifest",
    "write_distribution_gallery",
]
