"""VerifiedField factory function and VerificationMeta model.

VerifiedField wraps Pydantic's Field() to carry verification metadata
(ground truth, primitive, weight, extraction hint) in json_schema_extra.
The metadata is stripped before the schema reaches the judge LLM.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

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
    """
    # Serialize the primitive for storage
    primitive_data = verify_with.model_dump(mode="json")
    primitive_data["type"] = type(verify_with).__name__

    meta = VerificationMeta(
        ground_truth=ground_truth,
        verify_with=primitive_data,
        weight=weight,
        extraction_hint=extraction_hint,
    )

    # Merge with any user-provided json_schema_extra
    extra = kwargs.pop("json_schema_extra", None) or {}
    extra["__verification__"] = meta.model_dump(mode="json")

    return Field(description=description, json_schema_extra=extra, **kwargs)
