"""Verification pipeline stages.

This package contains individual stage implementations for the modular
verification pipeline. Each stage is a self-contained unit that performs
a specific verification task.

Package Structure:
- core/: Infrastructure (base classes, context, orchestrator)
- pipeline/: Stage implementations (13 verification stages)
- helpers/: Utility functions (deep judgment, export)

Core Types (from core/base.py):
- ArtifactKeys: Type-safe constants for artifact and result field keys
- VerificationContext: Shared state across stages
- VerificationStage: Protocol defining stage interface
- BaseVerificationStage: Abstract base class for stages
- StageRegistry: Manages stage instances and dependencies
- StageList: Type alias for list of stages

Base Classes (for extending):
- BaseCheckStage: Base class for check stages (abstention, sufficiency)
- BaseAutoFailStage: Base class for auto-fail stages (recursion_limit, deep_judgment, etc.)

Available Stages:
- ValidateTemplateStage: Template syntax validation
- GenerateAnswerStage: LLM answer generation
- RecursionLimitAutoFailStage: Auto-fail on recursion limit
- TraceValidationAutoFailStage: Auto-fail if trace doesn't end with AI message
- ParseTemplateStage: Response parsing to Pydantic
- VerifyTemplateStage: Field and regex validation
- EmbeddingCheckStage: Semantic similarity fallback
- AbstentionCheckStage: Refusal detection
- SufficiencyCheckStage: Trace sufficiency detection
- DeepJudgmentAutoFailStage: Excerpt validation (templates)
- DeepJudgmentRubricAutoFailStage: Excerpt validation (rubric traits)
- RubricEvaluationStage: Qualitative evaluation
- FinalizeResultStage: Result object construction
"""

# Core infrastructure
from .core import (
    ArtifactKeys,
    BaseAutoFailStage,
    BaseCheckStage,
    BaseVerificationStage,
    StageList,
    StageOrchestrator,
    StageRegistry,
    VerificationContext,
    VerificationStage,
)

# Helper functions
from .helpers import (
    HasTraitNames,
    apply_deep_judgment_config_to_traits,
    create_export_filename,
    export_verification_results_csv,
    export_verification_results_json,
    resolve_deep_judgment_config_for_trait,
)

# Pipeline stages
from .pipeline import (
    AbstentionCheckStage,
    DeepJudgmentAutoFailStage,
    DeepJudgmentRubricAutoFailStage,
    EmbeddingCheckStage,
    FinalizeResultStage,
    GenerateAnswerStage,
    ParseTemplateStage,
    RecursionLimitAutoFailStage,
    RubricEvaluationStage,
    SufficiencyCheckStage,
    TraceValidationAutoFailStage,
    ValidateTemplateStage,
    VerifyTemplateStage,
)

__all__ = [
    # Core types
    "ArtifactKeys",
    "VerificationContext",
    "VerificationStage",
    "BaseVerificationStage",
    "BaseAutoFailStage",
    "BaseCheckStage",
    "StageRegistry",
    "StageList",
    "StageOrchestrator",
    # Stage implementations
    "ValidateTemplateStage",
    "GenerateAnswerStage",
    "RecursionLimitAutoFailStage",
    "TraceValidationAutoFailStage",
    "ParseTemplateStage",
    "VerifyTemplateStage",
    "EmbeddingCheckStage",
    "AbstentionCheckStage",
    "SufficiencyCheckStage",
    "DeepJudgmentAutoFailStage",
    "DeepJudgmentRubricAutoFailStage",
    "RubricEvaluationStage",
    "FinalizeResultStage",
    # Helper functions
    "resolve_deep_judgment_config_for_trait",
    "apply_deep_judgment_config_to_traits",
    # Export functions
    "export_verification_results_csv",
    "export_verification_results_json",
    "create_export_filename",
    "HasTraitNames",
]
