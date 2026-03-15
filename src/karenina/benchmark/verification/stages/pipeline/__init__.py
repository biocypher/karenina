"""Pipeline stage implementations.

This package contains all verification pipeline stages:

| # | Stage | Description |
|---|-------|-------------|
| 1 | ValidateTemplateStage | Validates template syntax/attributes |
| 2 | GenerateAnswerStage | LLM generates response |
| 3 | RecursionLimitAutoFailStage | Auto-fail if recursion limit hit |
| 4 | TraceValidationAutoFailStage | Auto-fail if trace doesn't end with AI |
| 5 | AbstentionCheckStage | Detect model refusals |
| 6 | SufficiencyCheckStage | Detect insufficient responses |
| 7a | ParseTemplateStage | Parse response into Pydantic schema |
| 7b | AgenticParseTemplateStage | Agentic investigation + extraction |
| 8 | VerifyTemplateStage | Run verify() method |
| 9 | EmbeddingCheckStage | Semantic similarity fallback |
| 10 | DeepJudgmentAutoFailStage | Excerpt validation for templates |
| 11a | RubricEvaluationStage | Evaluate rubric traits |
| 11b | AgenticRubricEvaluationStage | Agentic rubric investigation + scoring |
| 12 | DeepJudgmentRubricAutoFailStage | Excerpt validation for rubrics |
| 13 | FinalizeResultStage | Build VerificationResult |
"""

from .abstention_check import AbstentionCheckStage
from .agentic_parse_template import AgenticParseTemplateStage
from .agentic_rubric_evaluation import AgenticRubricEvaluationStage
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
    "AgenticParseTemplateStage",
    "AgenticRubricEvaluationStage",
    "VerifyTemplateStage",
    "EmbeddingCheckStage",
    "DeepJudgmentAutoFailStage",
    "RubricEvaluationStage",
    "DeepJudgmentRubricAutoFailStage",
    "FinalizeResultStage",
]
