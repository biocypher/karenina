"""Workflow models for verification execution."""

from .aggregation import AggregatorRegistry, ResultAggregator, create_default_registry
from .judgment_results import JudgmentResults
from .models import (
    INTERFACE_LANGCHAIN,
    INTERFACE_MANUAL,
    INTERFACE_OPENROUTER,
    INTERFACES_NO_PROVIDER_REQUIRED,
    FewShotConfig,
    ModelConfig,
    QuestionFewShotConfig,
)
from .rubric_results import RubricResults
from .template_results import TemplateResults
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
from .verification_result_set import VerificationResultSet

# Rebuild models to resolve forward references
RubricResults.model_rebuild()
TemplateResults.model_rebuild()
JudgmentResults.model_rebuild()
VerificationResultSet.model_rebuild()
VerificationJob.model_rebuild()  # Rebuild after VerificationResultSet to resolve forward reference

__all__ = [
    # Model configuration
    "ModelConfig",
    "FewShotConfig",
    "QuestionFewShotConfig",
    # Interface constants
    "INTERFACE_OPENROUTER",
    "INTERFACE_MANUAL",
    "INTERFACE_LANGCHAIN",
    "INTERFACES_NO_PROVIDER_REQUIRED",
    # System prompt defaults
    "DEFAULT_ANSWERING_SYSTEM_PROMPT",
    "DEFAULT_PARSING_SYSTEM_PROMPT",
    # Verification
    "VerificationConfig",
    "VerificationResult",
    "VerificationResultMetadata",
    "VerificationResultTemplate",
    "VerificationResultRubric",
    "VerificationResultDeepJudgment",
    "VerificationResultDeepJudgmentRubric",
    "VerificationJob",
    "FinishedTemplate",
    "VerificationRequest",
    "VerificationStatusResponse",
    "VerificationStartResponse",
    # Result set and specialized results
    "VerificationResultSet",
    "RubricResults",
    "TemplateResults",
    "JudgmentResults",
    # Aggregation framework
    "ResultAggregator",
    "AggregatorRegistry",
    "create_default_registry",
]
