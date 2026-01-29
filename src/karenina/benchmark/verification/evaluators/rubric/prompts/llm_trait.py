"""Prompt construction for LLM-based trait evaluation (boolean and score kinds).

Backwards-compatibility re-export. Canonical location:
karenina.benchmark.verification.prompts.rubric.llm_trait

This module provides the LLMTraitPromptBuilder class for constructing prompts
used in standard LLM trait evaluation, supporting both batch (all traits at once)
and sequential (one at a time) evaluation modes.
"""

from karenina.benchmark.verification.prompts.rubric.llm_trait import LLMTraitPromptBuilder

__all__ = ["LLMTraitPromptBuilder"]
