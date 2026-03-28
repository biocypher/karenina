"""Parsed verification primitives for VerifiedField-based answer templates.

Parsed primitives operate on judge-extracted values: the field is included
in the parsing schema, and check() compares the extracted value against
ground truth.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Literal

from dateutil import parser as dateutil_parser  # type: ignore[import-untyped]
from pydantic import BaseModel

from karenina.schemas.primitives.normalizers import (
    Normalizer,
    apply_normalizers,
)
from karenina.schemas.primitives.registry import _register_primitive

logger = logging.getLogger(__name__)


# --- Base Class ---


class VerificationPrimitive(BaseModel):
    """Base class for all verification primitives.

    Subclasses implement check() to compare an extracted value against
    an expected value. The primitive type determines whether the field
    is included in the judge's parsing schema.
    """

    def check(self, extracted: Any, expected: Any) -> bool:
        """Compare extracted value against expected value.

        Args:
            extracted: Value extracted by the judge LLM.
            expected: Ground truth value from VerifiedField.

        Returns:
            True if the values match according to this primitive's rules.
        """
        raise NotImplementedError


# --- Boolean Primitives ---


@_register_primitive
class BooleanMatch(VerificationPrimitive):
    """Compare extracted bool to ground truth bool.

    Both values are coerced to bool before comparison.
    None is treated as distinct from both True and False:
    None only matches None via identity comparison.
    """

    def check(self, extracted: Any, expected: Any) -> bool:
        if extracted is None or expected is None:
            return extracted is expected
        return bool(extracted) == bool(expected)


# --- String Primitives ---


@_register_primitive
class ExactMatch(VerificationPrimitive):
    """Normalize then compare strings for equality.

    Default normalization: lowercase + strip whitespace.
    """

    normalize: list[Normalizer] = ["lowercase", "strip"]

    def check(self, extracted: Any, expected: Any) -> bool:
        e, x = str(extracted), str(expected)
        e = apply_normalizers(self.normalize, e)
        x = apply_normalizers(self.normalize, x)
        return e == x


@_register_primitive
class ContainsAny(VerificationPrimitive):
    """Check that extracted text contains at least one of the given substrings."""

    substrings: list[str]
    normalize: list[Normalizer] = []

    def check(self, extracted: Any, _expected: Any) -> bool:
        text = apply_normalizers(self.normalize, str(extracted))
        normalized_subs = [apply_normalizers(self.normalize, s) for s in self.substrings]
        return any(sub in text for sub in normalized_subs)


@_register_primitive
class ContainsAll(VerificationPrimitive):
    """Check that extracted text contains all of the given substrings."""

    substrings: list[str]
    normalize: list[Normalizer] = []

    def check(self, extracted: Any, _expected: Any) -> bool:
        text = apply_normalizers(self.normalize, str(extracted))
        normalized_subs = [apply_normalizers(self.normalize, s) for s in self.substrings]
        return all(sub in text for sub in normalized_subs)


@_register_primitive
class RegexMatch(VerificationPrimitive):
    """Check that extracted text matches a regex pattern."""

    pattern: str
    flags: list[str] = []

    def check(self, extracted: Any, _expected: Any) -> bool:
        flag_value = 0
        for f in self.flags:
            resolved = getattr(re, f, None)
            if resolved is None:
                logger.warning("Unknown regex flag %r, ignoring", f)
            else:
                flag_value |= resolved
        return bool(re.search(self.pattern, str(extracted), flag_value))


@_register_primitive
class SemanticMatch(VerificationPrimitive):
    """Check embedding similarity between extracted and expected text.

    Requires an embedding model to be configured at runtime.
    """

    threshold: float = 0.85

    def check(self, extracted: Any, expected: Any) -> bool:
        raise NotImplementedError(
            "SemanticMatch.check() requires embedding infrastructure. Use the embedding_check pipeline stage instead."
        )


# --- Numeric Primitives ---


@_register_primitive
class NumericExact(VerificationPrimitive):
    """Exact numeric equality after float coercion."""

    def check(self, extracted: Any, expected: Any) -> bool:
        return float(extracted) == float(expected)


@_register_primitive
class NumericTolerance(VerificationPrimitive):
    """Check that extracted number is within tolerance of expected.

    Modes:
    - "relative": |extracted - expected| / |expected| <= tolerance
    - "absolute": |extracted - expected| <= tolerance
    """

    tolerance: float
    mode: Literal["relative", "absolute"] = "relative"

    def check(self, extracted: Any, expected: Any) -> bool:
        diff = abs(float(extracted) - float(expected))
        if self.mode == "relative":
            if float(expected) == 0:
                return diff == 0
            return diff / abs(float(expected)) <= self.tolerance
        return diff <= self.tolerance


@_register_primitive
class NumericRange(VerificationPrimitive):
    """Check that extracted number falls within a range.

    Either min or max can be None for open-ended ranges.
    Boundaries are inclusive by default. Set ``exclusive_min`` or
    ``exclusive_max`` to True for strict inequality on that side.
    """

    min: float | None = None
    max: float | None = None
    exclusive_min: bool = False
    exclusive_max: bool = False

    def check(self, extracted: Any, _expected: Any) -> bool:
        val = float(extracted)
        if self.min is not None:
            if self.exclusive_min and val <= self.min:
                return False
            if not self.exclusive_min and val < self.min:
                return False
        if self.max is not None:
            if self.exclusive_max and val >= self.max:
                return False
            if not self.exclusive_max and val > self.max:
                return False
        return True


# --- List Primitives ---


@_register_primitive
class SetContainment(VerificationPrimitive):
    """Compare lists as sets with configurable containment mode.

    Modes:
    - "exact": extracted and expected contain the same elements
    - "subset": extracted is a subset of expected
    - "superset": extracted is a superset of expected
    - "overlap": at least min_overlap elements in common
    """

    mode: str = "exact"
    min_overlap: int | None = None

    def check(self, extracted: Any, expected: Any) -> bool:
        e_set = set(extracted)
        x_set = set(expected)
        if self.mode == "exact":
            return e_set == x_set
        if self.mode == "subset":
            return e_set <= x_set
        if self.mode == "superset":
            return e_set >= x_set
        if self.mode == "overlap":
            overlap = len(e_set & x_set)
            return overlap >= (self.min_overlap or 1)
        raise ValueError(f"Unknown mode: {self.mode!r}")


@_register_primitive
class OrderedMatch(VerificationPrimitive):
    """Compare lists element-by-element after normalization."""

    normalize: list[Normalizer] = ["lowercase", "strip"]

    def check(self, extracted: Any, expected: Any) -> bool:
        if len(extracted) != len(expected):
            return False
        for e, x in zip(extracted, expected, strict=False):
            e_norm = apply_normalizers(self.normalize, str(e))
            x_norm = apply_normalizers(self.normalize, str(x))
            if e_norm != x_norm:
                return False
        return True


# --- Categorical Primitives ---


@_register_primitive
class LiteralMatch(VerificationPrimitive):
    """Exact equality for Literal-typed fields."""

    def check(self, extracted: Any, expected: Any) -> bool:
        return bool(extracted == expected)


# --- Date/Time Primitives ---


def _parse_date(value: str, fmt: str | None = None) -> datetime:
    """Parse a date string, optionally with a specific format.

    Args:
        value: Date string to parse.
        fmt: Optional strftime format string.

    Returns:
        Parsed datetime object.
    """
    if fmt:
        return datetime.strptime(value, fmt)
    result: datetime = dateutil_parser.parse(str(value))
    return result


@_register_primitive
class DateMatch(VerificationPrimitive):
    """Parse and compare dates (format-flexible).

    Uses python-dateutil for flexible parsing when no format is specified.
    """

    format: str | None = None

    def check(self, extracted: Any, expected: Any) -> bool:
        try:
            e_date = _parse_date(str(extracted), self.format)
            x_date = _parse_date(str(expected), self.format)
            return e_date.date() == x_date.date()
        except (ValueError, TypeError):
            logger.warning("Date parsing failed for %r vs %r", extracted, expected)
            return False


@_register_primitive
class DateTolerance(VerificationPrimitive):
    """Check that extracted date is within tolerance of expected date."""

    tolerance: int
    unit: Literal["days", "hours", "minutes"] = "days"

    def check(self, extracted: Any, expected: Any) -> bool:
        try:
            e_date = _parse_date(str(extracted))
            x_date = _parse_date(str(expected))
            if self.unit == "days":
                delta = timedelta(days=self.tolerance)
            elif self.unit == "hours":
                delta = timedelta(hours=self.tolerance)
            elif self.unit == "minutes":
                delta = timedelta(minutes=self.tolerance)
            else:
                raise ValueError(f"Unknown unit: {self.unit!r}")
            return abs(e_date - x_date) <= delta
        except (ValueError, TypeError):
            logger.warning("Date parsing failed for %r vs %r", extracted, expected)
            return False


@_register_primitive
class DateRange(VerificationPrimitive):
    """Check that extracted date falls within a range."""

    min: str | None = None
    max: str | None = None

    def check(self, extracted: Any, _expected: Any) -> bool:
        try:
            e_date = _parse_date(str(extracted))
            if self.min is not None and e_date < _parse_date(self.min):
                return False
            return not (self.max is not None and e_date > _parse_date(self.max))
        except (ValueError, TypeError):
            logger.warning("Date parsing failed for %r", extracted)
            return False
