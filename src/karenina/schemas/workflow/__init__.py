"""Workflow models for verification execution.

DEPRECATED: This module is deprecated. Import from the following modules instead:
- schemas.verification: VerificationConfig, VerificationResult, VerificationJob
- schemas.results: VerificationResultSet, RubricResults, TemplateResults, JudgmentResults
- schemas.config: ModelConfig, FewShotConfig
- schemas.outputs: BatchRubricScores, ConfusionMatrixOutput, etc.
- schemas.dataframes: RubricDataFrameBuilder, TemplateDataFrameBuilder, JudgmentDataFrameBuilder

This module re-exports from new locations for backward compatibility.
"""

import warnings

# Re-export from new locations for backward compatibility
from ..config import (
    INTERFACE_LANGCHAIN,
    INTERFACE_MANUAL,
    INTERFACE_OPENROUTER,
    INTERFACES_NO_PROVIDER_REQUIRED,
    FewShotConfig,
    ModelConfig,
    QuestionFewShotConfig,
)
from ..dataframes import (
    JudgmentDataFrameBuilder,
    RubricDataFrameBuilder,
    TemplateDataFrameBuilder,
)
from ..outputs import (
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
from ..results import (
    AggregatorRegistry,
    JudgmentResults,
    ResultAggregator,
    RubricJudgmentResults,
    RubricResults,
    TemplateResults,
    VerificationResultSet,
    create_default_registry,
)
from ..verification import (
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
    "RubricDataFrameBuilder",
    "RubricJudgmentResults",
    "TemplateResults",
    "TemplateDataFrameBuilder",
    "JudgmentResults",
    "JudgmentDataFrameBuilder",
    # Aggregation framework
    "ResultAggregator",
    "AggregatorRegistry",
    "create_default_registry",
    # Structured output models for rubric evaluation
    "BatchRubricScores",
    "SingleBooleanScore",
    "SingleNumericScore",
    "ConfusionMatrixOutput",
    # Deep Judgment Rubric structured output models
    "TraitExcerpt",
    "TraitExcerptsOutput",
    "HallucinationRiskOutput",
    # Literal trait classification models
    "SingleLiteralClassification",
    "BatchLiteralClassifications",
]


def __getattr__(name: str) -> object:
    """Emit deprecation warning when accessing this module."""
    if name in __all__:
        warnings.warn(
            f"Importing {name} from 'karenina.schemas.workflow' is deprecated. "
            f"Use 'karenina.schemas' or the specific submodule instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
