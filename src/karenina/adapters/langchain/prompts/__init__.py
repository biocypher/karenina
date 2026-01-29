"""Prompts for LangChain adapter operations.

Each prompt is in its own file for easy review and editing.
Import from this module for convenient access to all prompts.

Note: Parser prompt construction is now centralized in
benchmark/verification/prompts/parsing/instructions.py (TemplatePromptBuilder).
Adapter-specific modifications are applied via AdapterInstructionRegistry.
"""

from .feedback_format import PROMPT as FEEDBACK_FORMAT
from .feedback_null import PROMPT as FEEDBACK_NULL
from .format_instructions import PROMPT as FORMAT_INSTRUCTIONS
from .summarization import PROMPT as SUMMARIZATION
from .summarization import build_question_context

__all__ = [
    "FEEDBACK_FORMAT",
    "FEEDBACK_NULL",
    "FORMAT_INSTRUCTIONS",
    "SUMMARIZATION",
    "build_question_context",
]
