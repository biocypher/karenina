"""Pipeline stage implementations.

This package contains all 13 verification pipeline stages:

| # | Stage | Description |
|---|-------|-------------|
| 1 | ValidateTemplateStage | Validates template syntax/attributes |
| 2 | GenerateAnswerStage | LLM generates response |
| 3 | RecursionLimitAutoFailStage | Auto-fail if recursion limit hit |
| 4 | TraceValidationAutoFailStage | Auto-fail if trace doesn't end with AI |
| 5 | AbstentionCheckStage | Detect model refusals |
| 6 | SufficiencyCheckStage | Detect insufficient responses |
| 7 | ParseTemplateStage | Parse response into Pydantic schema |
| 8 | VerifyTemplateStage | Run verify() method |
| 9 | EmbeddingCheckStage | Semantic similarity fallback |
| 10 | DeepJudgmentAutoFailStage | Excerpt validation for templates |
| 11 | RubricEvaluationStage | Evaluate rubric traits |
| 12 | DeepJudgmentRubricAutoFailStage | Excerpt validation for rubrics |
| 13 | FinalizeResultStage | Build VerificationResult |
"""

from .abstention_check import AbstentionCheckStage
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
    "ValidateTemplateStage",
    "GenerateAnswerStage",
    "RecursionLimitAutoFailStage",
    "TraceValidationAutoFailStage",
    "AbstentionCheckStage",
    "SufficiencyCheckStage",
    "ParseTemplateStage",
    "VerifyTemplateStage",
    "EmbeddingCheckStage",
    "DeepJudgmentAutoFailStage",
    "RubricEvaluationStage",
    "DeepJudgmentRubricAutoFailStage",
    "FinalizeResultStage",
]
