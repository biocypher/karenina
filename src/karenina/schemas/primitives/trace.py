"""Trace verification primitives for raw LLM response evaluation.

Trace primitives operate on the raw LLM response text rather than
judge-extracted values. Fields using TracePrimitive are excluded from
the judge's parsing schema.
"""

from __future__ import annotations

import re
from typing import Any

from karenina.schemas.primitives.comparisons import VerificationPrimitive
from karenina.schemas.primitives.registry import _register_primitive


class TracePrimitive(VerificationPrimitive):
    """Base class for primitives that operate on raw LLM response text.

    Fields using TracePrimitive are excluded from the judge's parsing
    schema. The pipeline evaluates them directly against the raw response.
    """

    def check(self, extracted: Any, expected: Any) -> bool:
        """Not used for trace primitives. Use check_trace() instead."""
        raise NotImplementedError("TracePrimitive uses check_trace(), not check()")

    def check_trace(self, raw_trace: str) -> bool:
        """Evaluate the raw LLM response.

        Args:
            raw_trace: The raw text response from the answering LLM.

        Returns:
            True if the pattern is found/condition is met.
        """
        raise NotImplementedError


@_register_primitive
class TraceRegex(TracePrimitive):
    """Check for regex pattern in raw LLM response.

    Returns True if the pattern is found (or count >= count_min).
    """

    pattern: str
    count_min: int | None = None

    def check_trace(self, raw_trace: str) -> bool:
        matches = re.findall(self.pattern, raw_trace)
        if self.count_min is not None:
            return len(matches) >= self.count_min
        return len(matches) > 0


@_register_primitive
class TraceContains(TracePrimitive):
    """Check for substring in raw LLM response."""

    substring: str

    def check_trace(self, raw_trace: str) -> bool:
        return self.substring in raw_trace


@_register_primitive
class TraceLength(TracePrimitive):
    """Check length of raw LLM response."""

    min: int | None = None
    max: int | None = None
    unit: str = "chars"

    def check_trace(self, raw_trace: str) -> bool:
        length = len(raw_trace.split()) if self.unit == "words" else len(raw_trace)
        if self.min is not None and length < self.min:
            return False
        return not (self.max is not None and length > self.max)
