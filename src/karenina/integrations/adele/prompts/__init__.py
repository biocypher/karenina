"""
ADeLe classification prompts.

This module contains all prompts used by the QuestionClassifier
for ADeLe question classification.
"""

from .system import SYSTEM_PROMPT_BATCH, SYSTEM_PROMPT_SINGLE_TRAIT
from .user import USER_PROMPT_BATCH_TEMPLATE, USER_PROMPT_SINGLE_TRAIT_TEMPLATE

__all__ = [
    "SYSTEM_PROMPT_SINGLE_TRAIT",
    "SYSTEM_PROMPT_BATCH",
    "USER_PROMPT_SINGLE_TRAIT_TEMPLATE",
    "USER_PROMPT_BATCH_TEMPLATE",
]
