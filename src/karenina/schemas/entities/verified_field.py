"""VerifiedField factory function and VerificationMeta model.

VerifiedField wraps Pydantic's Field() to carry verification metadata
(ground truth, primitive, weight, extraction hint) in json_schema_extra.
The metadata is stripped before the schema reaches the judge LLM.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from karenina.schemas.entities.conditional import ConditionalGroundTruth

logger = logging.getLogger(__name__)


class VerificationMeta(BaseModel):
    """Metadata stored on each VerifiedField, not visible to the judge.

    Serialized into json_schema_extra["__verification__"] on the Pydantic
    FieldInfo. The prompt builder strips this key before sending the
    schema to the judge LLM.
    """

    ground_truth: Any
    verify_with: dict[str, Any]  # Serialized primitive (type + params)
    weight: float = 1.0
    extraction_hint: str | None = None


def _warn_ground_truth_mismatch(ground_truth: Any, verify_with: Any) -> None:
    """Log a warning if ground_truth obviously mismatches the primitive's expected type.

    This catches common authoring mistakes early. Only flags obvious
    mismatches; ambiguous cases are left to runtime verification.

    Args:
        ground_truth: The expected correct value.
        verify_with: The verification primitive instance.
    """
    # Lazy import to avoid circular dependency at module level
    from karenina.schemas.primitives.comparisons import (
        BooleanMatch,
        NumericExact,
        NumericGraded,
        NumericMaximum,
        NumericMinimum,
        NumericRange,
        NumericRangeGraded,
        NumericThresholdGraded,
        NumericTolerance,
    )

    primitive_name = type(verify_with).__name__

    # Check numeric primitives
    if isinstance(
        verify_with,
        NumericTolerance
        | NumericExact
        | NumericRange
        | NumericMinimum
        | NumericMaximum
        | NumericGraded
        | NumericRangeGraded
        | NumericThresholdGraded,
    ):
        if isinstance(ground_truth, int | float) and not isinstance(ground_truth, bool):
            return  # Already numeric, no mismatch
        if isinstance(ground_truth, str):
            try:
                float(ground_truth)
                return  # String is coercible to float
            except (ValueError, TypeError):
                pass
        logger.warning(
            "ground_truth %r may not match %s: expected a numeric value or a string coercible to float.",
            ground_truth,
            primitive_name,
        )
        return

    # Check boolean primitive
    if isinstance(verify_with, BooleanMatch):
        if isinstance(ground_truth, bool):
            return  # Already bool
        if isinstance(ground_truth, int | float) and ground_truth in (0, 1, 0.0, 1.0):
            return  # Common bool-like values
        if isinstance(ground_truth, str) and ground_truth.lower() in (
            "true",
            "false",
            "yes",
            "no",
            "1",
            "0",
        ):
            return  # Common bool-like strings
        logger.warning(
            "ground_truth %r may not match %s: expected a boolean or bool-like value (True/False, 0/1, 'yes'/'no').",
            ground_truth,
            primitive_name,
        )


def VerifiedField(
    description: str,
    ground_truth: Any,
    verify_with: Any,
    weight: float = 1.0,
    extraction_hint: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create a Pydantic Field with verification metadata attached.

    Unlike plain Field(), description is mandatory because the judge LLM
    relies on it to know what to extract.

    Args:
        description: What to extract (goes into the JSON schema description).
        ground_truth: Expected correct value.
        verify_with: Verification primitive instance (ExactMatch, BooleanMatch, etc.).
        weight: Weight for verify_granular() scoring. Default: 1.0.
        extraction_hint: Optional formatting guidance for the judge.
        **kwargs: Additional Pydantic Field arguments.

    Returns:
        Pydantic FieldInfo with verification metadata in json_schema_extra.

    Raises:
        ValueError: If verify_with is None or description is empty/whitespace.
    """
    # Issue 056: reject empty or whitespace-only description
    if not description or not description.strip():
        raise ValueError(
            "description is required for VerifiedField: the judge LLM relies on it to know what to extract."
        )

    # Issue 053: reject None verify_with with a clear message
    if verify_with is None:
        raise ValueError(
            "verify_with is required: pass a verification primitive instance (e.g., ExactMatch(), BooleanMatch())."
        )

    # Serialize the primitive for storage
    primitive_data = verify_with.model_dump(mode="json")
    primitive_data["type"] = type(verify_with).__name__

    # Serialize conditional ground truth, or warn on type mismatches
    if isinstance(ground_truth, ConditionalGroundTruth):
        ground_truth_serialized = ground_truth.serialize()
    else:
        ground_truth_serialized = ground_truth
        # Issue 010: warn on obvious ground_truth type mismatches
        _warn_ground_truth_mismatch(ground_truth, verify_with)

    meta = VerificationMeta(
        ground_truth=ground_truth_serialized,
        verify_with=primitive_data,
        weight=weight,
        extraction_hint=extraction_hint,
    )

    # Merge with any user-provided json_schema_extra
    extra = kwargs.pop("json_schema_extra", None) or {}
    extra["__verification__"] = meta.model_dump(mode="json")

    return Field(description=description, json_schema_extra=extra, **kwargs)
