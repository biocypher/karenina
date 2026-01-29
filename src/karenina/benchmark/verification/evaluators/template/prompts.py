"""Prompt construction for template evaluation.

Backwards-compatible re-export. The canonical location is now:
    karenina.benchmark.verification.prompts.parsing.parsing_instructions

All imports from this module continue to work unchanged.
"""

from karenina.benchmark.verification.prompts.parsing.parsing_instructions import (
    TemplatePromptBuilder,
)

__all__ = [
    "TemplatePromptBuilder",
]
