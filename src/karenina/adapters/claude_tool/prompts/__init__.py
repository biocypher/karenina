"""Prompts for Claude Tool adapter operations.

Each prompt is in its own file for easy review and editing.
Import from this module for convenient access to all prompts.
"""

from .parser_system import PROMPT as PARSER_SYSTEM
from .parser_user import PROMPT as PARSER_USER

__all__ = [
    "PARSER_SYSTEM",
    "PARSER_USER",
]
