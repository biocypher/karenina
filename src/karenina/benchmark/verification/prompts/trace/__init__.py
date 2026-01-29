"""Centralized trace analysis prompts for the verification pipeline.

Contains prompt constants for abstention detection, sufficiency detection,
and answer evaluation (orphaned - no current consumers).
"""

from karenina.benchmark.verification.prompts.trace.abstention import (
    ABSTENTION_DETECTION_SYS,
    ABSTENTION_DETECTION_USER,
)
from karenina.benchmark.verification.prompts.trace.answer_evaluation import (
    ANSWER_EVALUATION_SYS,
    ANSWER_EVALUATION_USER,
)
from karenina.benchmark.verification.prompts.trace.sufficiency import (
    SUFFICIENCY_DETECTION_SYS,
    SUFFICIENCY_DETECTION_USER,
)

__all__ = [
    "ABSTENTION_DETECTION_SYS",
    "ABSTENTION_DETECTION_USER",
    "ANSWER_EVALUATION_SYS",
    "ANSWER_EVALUATION_USER",
    "SUFFICIENCY_DETECTION_SYS",
    "SUFFICIENCY_DETECTION_USER",
]
