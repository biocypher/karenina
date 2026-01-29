"""Prompts for Claude Agent SDK adapter operations.

Each prompt is in its own file for easy review and editing.
Import from this module for convenient access to all prompts.
"""

from .parser import SYSTEM_PROMPT as PARSER_SYSTEM
from .parser import USER_PROMPT as PARSER_USER

__all__ = [
    "PARSER_SYSTEM",
    "PARSER_USER",
]
