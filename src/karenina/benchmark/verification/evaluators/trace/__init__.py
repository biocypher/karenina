"""Trace analysis components for detecting abstention and sufficiency.

This package provides:
- detect_abstention: Detect when models refuse to answer questions
- detect_sufficiency: Detect if responses have sufficient information for templates
"""

from .abstention import detect_abstention
from .sufficiency import detect_sufficiency

__all__ = [
    "detect_abstention",
    "detect_sufficiency",
]
