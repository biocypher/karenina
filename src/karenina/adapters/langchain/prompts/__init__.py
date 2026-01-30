"""Prompts for LangChain adapter operations.

Each prompt is in its own file for easy review and editing.
Import from this module for convenient access to all prompts.
"""

from .feedback_format import PROMPT as FEEDBACK_FORMAT
from .feedback_null import PROMPT as FEEDBACK_NULL
from .format_instructions import PROMPT as FORMAT_INSTRUCTIONS
from .parser_system import PROMPT as PARSER_SYSTEM
from .parser_user import PROMPT as PARSER_USER
from .summarization import PROMPT as SUMMARIZATION
from .summarization import build_question_context

__all__ = [
    "FEEDBACK_FORMAT",
    "FEEDBACK_NULL",
    "FORMAT_INSTRUCTIONS",
    "PARSER_SYSTEM",
    "PARSER_USER",
    "SUMMARIZATION",
    "build_question_context",
]
