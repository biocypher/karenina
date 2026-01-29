"""Prompts for verification operations.

Canonical location: karenina.benchmark.verification.prompts.trace
This module re-exports from the centralized location for backwards compatibility.
"""

from karenina.benchmark.verification.prompts.trace.abstention import (
    ABSTENTION_DETECTION_SYS,
    ABSTENTION_DETECTION_USER,
)
from karenina.benchmark.verification.prompts.trace.sufficiency import (
    SUFFICIENCY_DETECTION_SYS,
    SUFFICIENCY_DETECTION_USER,
)

__all__ = [
    "ABSTENTION_DETECTION_SYS",
    "ABSTENTION_DETECTION_USER",
    "SUFFICIENCY_DETECTION_SYS",
    "SUFFICIENCY_DETECTION_USER",
]
