"""Template evaluation components for parsing and verifying LLM responses.

This package provides:
- TemplateEvaluator: Main class for template parsing and verification
- ParseResult, FieldVerificationResult, RegexVerificationResult: Result dataclasses
- TemplatePromptBuilder: Constructs prompts for template parsing
- deep_judgment_parse: Multi-stage parsing with excerpt extraction
"""

from .deep_judgment import deep_judgment_parse
from .evaluator import TemplateEvaluator
from .prompts import TemplatePromptBuilder
from .results import FieldVerificationResult, ParseResult, RegexVerificationResult

__all__ = [
    "TemplateEvaluator",
    "TemplatePromptBuilder",
    "ParseResult",
    "FieldVerificationResult",
    "RegexVerificationResult",
    "deep_judgment_parse",
]
