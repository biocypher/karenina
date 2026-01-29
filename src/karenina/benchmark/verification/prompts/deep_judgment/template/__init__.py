"""Centralized prompt builders for template deep judgment evaluation.

Canonical location for prompts used in the multi-stage deep judgment parsing
flow (evaluators/template/deep_judgment.py). Three stages:

- excerpt: Stage 1 excerpt extraction prompts
- hallucination: Stage 1.5 per-excerpt hallucination assessment prompts
- reasoning: Stage 2 reasoning generation prompts

Stage 3 (parameter extraction) uses ParserPort directly and has no
assembler-managed prompts.
"""

from .excerpt import build_excerpt_system_prompt, build_excerpt_user_prompt
from .hallucination import build_assessment_system_prompt, build_assessment_user_prompt
from .reasoning import build_reasoning_system_prompt, build_reasoning_user_prompt

__all__ = [
    "build_excerpt_system_prompt",
    "build_excerpt_user_prompt",
    "build_assessment_system_prompt",
    "build_assessment_user_prompt",
    "build_reasoning_system_prompt",
    "build_reasoning_user_prompt",
]
