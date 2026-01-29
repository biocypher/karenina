"""Prompt construction for template evaluation.

Backwards-compatible re-export. The canonical location is now:
    karenina.benchmark.verification.prompts.parsing.instructions

All imports from this module continue to work unchanged.
"""

from karenina.benchmark.verification.prompts.parsing.instructions import (
    TemplatePromptBuilder,
)

__all__ = [
    "TemplatePromptBuilder",
]
