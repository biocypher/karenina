"""Prompt construction for rubric evaluation.

This package provides prompt builder classes for constructing prompts used
in rubric trait evaluation operations. Prompts are organized by evaluation type:

- LLMTraitPromptBuilder: Boolean and score trait evaluation prompts
- LiteralTraitPromptBuilder: Categorical classification prompts
- DeepJudgmentPromptBuilder: Multi-stage deep judgment prompts
- MetricTraitPromptBuilder: Confusion matrix analysis prompts

NOTE: LLMTraitPromptBuilder, LiteralTraitPromptBuilder, and MetricTraitPromptBuilder
are re-exported from their canonical location at
karenina.benchmark.verification.prompts.rubric for backwards compatibility.
"""

from .deep_judgment import DeepJudgmentPromptBuilder
from .literal_trait import LiteralTraitPromptBuilder
from .llm_trait import LLMTraitPromptBuilder
from .metric_trait import MetricTraitPromptBuilder

__all__ = [
    "LLMTraitPromptBuilder",
    "LiteralTraitPromptBuilder",
    "DeepJudgmentPromptBuilder",
    "MetricTraitPromptBuilder",
]
