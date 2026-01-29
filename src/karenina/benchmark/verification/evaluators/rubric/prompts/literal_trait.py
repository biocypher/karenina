"""Prompt construction for literal (categorical) trait evaluation.

Backwards-compatibility re-export. Canonical location:
karenina.benchmark.verification.prompts.rubric.literal_trait

This module provides the LiteralTraitPromptBuilder class for constructing prompts
used in literal trait evaluation, where responses are classified into predefined
categories.
"""

from karenina.benchmark.verification.prompts.rubric.literal_trait import LiteralTraitPromptBuilder

__all__ = ["LiteralTraitPromptBuilder"]
