"""Schema definitions for Karenina.

This module contains Pydantic models for data validation and serialization.

All models are organized into submodules:
- domain/: Core business entities (Question, Answer, Rubric)
- workflow/: Verification execution models (Config, Result, Job)
- shared/: Shared utilities (SearchResultItem)
- checkpoint/: JSON-LD checkpoint models
"""

# Domain models
# Checkpoint models
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
from .domain import (
    BaseAnswer,
    ManualRubricTrait,
    MetricRubricTrait,
    Question,
    Rubric,
    RubricEvaluation,
    RubricTrait,
    TraitKind,
    capture_answer_source,
    merge_rubrics,
)

# Shared models
from .shared import SearchResultItem

# Workflow models
from .workflow import (
    DEFAULT_ANSWERING_SYSTEM_PROMPT,
    DEFAULT_PARSING_SYSTEM_PROMPT,
    INTERFACE_LANGCHAIN,
    INTERFACE_MANUAL,
    INTERFACE_OPENROUTER,
    INTERFACES_NO_PROVIDER_REQUIRED,
    FewShotConfig,
    FinishedTemplate,
    ModelConfig,
    QuestionFewShotConfig,
    VerificationConfig,
    VerificationJob,
    VerificationRequest,
    VerificationResult,
    VerificationStartResponse,
    VerificationStatusResponse,
)

__all__ = [
    # Domain models
    "BaseAnswer",
    "capture_answer_source",
    "Question",
    "Rubric",
    "RubricTrait",
    "ManualRubricTrait",
    "MetricRubricTrait",
    "RubricEvaluation",
    "TraitKind",
    "merge_rubrics",
    # Workflow models
    "ModelConfig",
    "FewShotConfig",
    "QuestionFewShotConfig",
    "INTERFACE_OPENROUTER",
    "INTERFACE_MANUAL",
    "INTERFACE_LANGCHAIN",
    "INTERFACES_NO_PROVIDER_REQUIRED",
    "DEFAULT_ANSWERING_SYSTEM_PROMPT",
    "DEFAULT_PARSING_SYSTEM_PROMPT",
    "VerificationConfig",
    "VerificationResult",
    "VerificationJob",
    "FinishedTemplate",
    "VerificationRequest",
    "VerificationStatusResponse",
    "VerificationStartResponse",
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
