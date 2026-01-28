"""Verification pipeline stages.

This package contains individual stage implementations for the modular
verification pipeline. Each stage is a self-contained unit that performs
a specific verification task.

Core Types (from base.py):
- VerificationContext: Shared state across stages
- VerificationStage: Protocol defining stage interface
- BaseVerificationStage: Abstract base class for stages
- StageRegistry: Manages stage instances and dependencies
- StageList: Type alias for list of stages

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

# Core types from base module
# Stage implementations
from .abstention_check import AbstentionCheckStage
from .base import (
    BaseVerificationStage,
    StageList,
    StageRegistry,
    VerificationContext,
    VerificationStage,
)
from .deep_judgment_autofail import DeepJudgmentAutoFailStage
from .deep_judgment_rubric_auto_fail import DeepJudgmentRubricAutoFailStage
from .embedding_check import EmbeddingCheckStage
from .finalize_result import FinalizeResultStage
from .generate_answer import GenerateAnswerStage
from .parse_template import ParseTemplateStage
from .recursion_limit_autofail import RecursionLimitAutoFailStage
from .rubric_evaluation import RubricEvaluationStage
from .sufficiency_check import SufficiencyCheckStage
from .trace_validation_autofail import TraceValidationAutoFailStage
from .validate_template import ValidateTemplateStage
from .verify_template import VerifyTemplateStage

__all__ = [
    # Core types
    "VerificationContext",
    "VerificationStage",
    "BaseVerificationStage",
    "StageRegistry",
    "StageList",
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
]
