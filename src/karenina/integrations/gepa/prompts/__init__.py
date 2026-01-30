"""GEPA prompts module.

This module centralizes all prompts used by the GEPA integration
for feedback generation and default configurations.
"""

from karenina.integrations.gepa.prompts.defaults import (
    DEFAULT_ANSWERING_SYSTEM_PROMPT,
    DEFAULT_PARSING_INSTRUCTIONS,
)
from karenina.integrations.gepa.prompts.feedback import (
    DIFFERENTIAL_FEEDBACK_SYSTEM_PROMPT,
    RUBRIC_FEEDBACK_SYSTEM_PROMPT,
    SINGLE_FEEDBACK_SYSTEM_PROMPT,
)

__all__ = [
    "SINGLE_FEEDBACK_SYSTEM_PROMPT",
    "DIFFERENTIAL_FEEDBACK_SYSTEM_PROMPT",
    "RUBRIC_FEEDBACK_SYSTEM_PROMPT",
    "DEFAULT_ANSWERING_SYSTEM_PROMPT",
    "DEFAULT_PARSING_INSTRUCTIONS",
]
