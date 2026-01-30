"""Schema definitions for Karenina.

This module contains Pydantic models for data validation and serialization.

All models are organized into submodules:
- entities/: Core business entities (Question, Answer, Rubric)
- verification/: Verification configuration and result models
- results/: Result containers and aggregation
- config/: LLM and workflow configuration
- outputs/: Structured LLM output models
- dataframes/: DataFrame builder utilities
- shared: Shared utilities (SearchResultItem)
- checkpoint: JSON-LD checkpoint models

Deprecated paths (re-export for backward compatibility):
- domain/: → entities/
- workflow/: → verification/, results/, config/, outputs/, dataframes/
"""

# ============================================================================
# Checkpoint models
# ============================================================================
from .checkpoint import (
    DatasetMetadata,
    JsonLdCheckpoint,
    SchemaOrgAnswer,
    SchemaOrgCreativeWork,
    SchemaOrgDataFeed,
    SchemaOrgDataFeedItem,
    SchemaOrgPerson,
    SchemaOrgPropertyValue,
    SchemaOrgQuestion,
    SchemaOrgRating,
    SchemaOrgSoftwareSourceCode,
)

# ============================================================================
# Configuration models (from config/)
# ============================================================================
from .config import (
    INTERFACE_LANGCHAIN,
    INTERFACE_MANUAL,
    INTERFACE_OPENROUTER,
    INTERFACES_NO_PROVIDER_REQUIRED,
    FewShotConfig,
    ModelConfig,
    QuestionFewShotConfig,
)

# ============================================================================
# DataFrame builders (from dataframes/)
# ============================================================================
from .dataframes import (
    JudgmentDataFrameBuilder,
    RubricDataFrameBuilder,
    TemplateDataFrameBuilder,
)

# ============================================================================
# Core entity models (from entities/)
# ============================================================================
from .entities import (
    BaseAnswer,
    CallableTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    Question,
    RegexTrait,
    Rubric,
    RubricEvaluation,
    TraitKind,
    capture_answer_source,
    merge_rubrics,
)

# ============================================================================
# Structured output models (from outputs/)
# ============================================================================
from .outputs import (
    BatchLiteralClassifications,
    BatchRubricScores,
    ConfusionMatrixOutput,
    HallucinationRiskOutput,
    SingleBooleanScore,
    SingleLiteralClassification,
    SingleNumericScore,
    TraitExcerpt,
    TraitExcerptsOutput,
)

# ============================================================================
# Result containers (from results/)
# ============================================================================
from .results import (
    AggregatorRegistry,
    JudgmentResults,
    ResultAggregator,
    RubricJudgmentResults,
    RubricResults,
    TemplateResults,
    VerificationResultSet,
    create_default_registry,
)

# ============================================================================
# Shared models
# ============================================================================
from .shared import SearchResultItem

# ============================================================================
# Verification models (from verification/)
# ============================================================================
from .verification import (
    DEFAULT_ANSWERING_SYSTEM_PROMPT,
    DEFAULT_PARSING_SYSTEM_PROMPT,
    FinishedTemplate,
    VerificationConfig,
    VerificationJob,
    VerificationRequest,
    VerificationResult,
    VerificationResultDeepJudgment,
    VerificationResultDeepJudgmentRubric,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
    VerificationStartResponse,
    VerificationStatusResponse,
)

# ============================================================================
# Rebuild models to resolve forward references
# Order matters: result containers before Job
# ============================================================================
RubricResults.model_rebuild()
RubricJudgmentResults.model_rebuild()
TemplateResults.model_rebuild()
JudgmentResults.model_rebuild()
VerificationResultSet.model_rebuild()
VerificationJob.model_rebuild()  # Rebuild after VerificationResultSet to resolve forward reference

__all__ = [
    # Entity models
    "BaseAnswer",
    "capture_answer_source",
    "Question",
    "Rubric",
    "LLMRubricTrait",
    "RegexTrait",
    "CallableTrait",
    "MetricRubricTrait",
    "RubricEvaluation",
    "TraitKind",
    "merge_rubrics",
    # Configuration models
    "ModelConfig",
    "FewShotConfig",
    "QuestionFewShotConfig",
    "INTERFACE_OPENROUTER",
    "INTERFACE_MANUAL",
    "INTERFACE_LANGCHAIN",
    "INTERFACES_NO_PROVIDER_REQUIRED",
    # Verification configuration
    "DEFAULT_ANSWERING_SYSTEM_PROMPT",
    "DEFAULT_PARSING_SYSTEM_PROMPT",
    "VerificationConfig",
    # Verification results
    "VerificationResult",
    "VerificationResultMetadata",
    "VerificationResultTemplate",
    "VerificationResultRubric",
    "VerificationResultDeepJudgment",
    "VerificationResultDeepJudgmentRubric",
    # Verification job and API
    "VerificationJob",
    "FinishedTemplate",
    "VerificationRequest",
    "VerificationStatusResponse",
    "VerificationStartResponse",
    # Result containers
    "VerificationResultSet",
    "RubricResults",
    "RubricJudgmentResults",
    "TemplateResults",
    "JudgmentResults",
    # DataFrame builders
    "RubricDataFrameBuilder",
    "TemplateDataFrameBuilder",
    "JudgmentDataFrameBuilder",
    # Aggregation framework
    "ResultAggregator",
    "AggregatorRegistry",
    "create_default_registry",
    # Structured output models
    "BatchRubricScores",
    "SingleBooleanScore",
    "SingleNumericScore",
    "ConfusionMatrixOutput",
    "TraitExcerpt",
    "TraitExcerptsOutput",
    "HallucinationRiskOutput",
    "SingleLiteralClassification",
    "BatchLiteralClassifications",
    # Shared models
    "SearchResultItem",
    # Checkpoint models
    "JsonLdCheckpoint",
    "SchemaOrgDataFeed",
    "SchemaOrgDataFeedItem",
    "SchemaOrgQuestion",
    "SchemaOrgAnswer",
    "SchemaOrgSoftwareSourceCode",
    "SchemaOrgRating",
    "SchemaOrgPropertyValue",
    "SchemaOrgPerson",
    "SchemaOrgCreativeWork",
    "DatasetMetadata",
]
