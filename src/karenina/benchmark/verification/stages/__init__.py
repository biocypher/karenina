"""Verification pipeline stages.

This package contains individual stage implementations for the modular
verification pipeline. Each stage is a self-contained unit that performs
a specific verification task.

Available Stages:
- ValidateTemplateStage: Template syntax validation
- GenerateAnswerStage: LLM answer generation
- ParseTemplateStage: Response parsing to Pydantic
- VerifyTemplateStage: Field and regex validation
- EmbeddingCheckStage: Semantic similarity fallback
- AbstentionCheckStage: Refusal detection
- DeepJudgmentAutoFailStage: Excerpt validation
- RubricEvaluationStage: Qualitative evaluation
- FinalizeResultStage: Result object construction
"""

from .abstention_check import AbstentionCheckStage
from .deep_judgment_autofail import DeepJudgmentAutoFailStage
from .embedding_check import EmbeddingCheckStage
from .finalize_result import FinalizeResultStage
from .generate_answer import GenerateAnswerStage
from .parse_template import ParseTemplateStage
from .rubric_evaluation import RubricEvaluationStage
from .validate_template import ValidateTemplateStage
from .verify_template import VerifyTemplateStage

__all__ = [
    "ValidateTemplateStage",
    "GenerateAnswerStage",
    "ParseTemplateStage",
    "VerifyTemplateStage",
    "EmbeddingCheckStage",
    "AbstentionCheckStage",
    "DeepJudgmentAutoFailStage",
    "RubricEvaluationStage",
    "FinalizeResultStage",
]
