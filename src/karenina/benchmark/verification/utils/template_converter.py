"""Bidirectional converter between Python template code and TemplateSpec JSON.

This module provides the bridge between the visual template builder GUI
(which works with TemplateSpec JSON) and the verification pipeline
(which works with Python source code strings).

Functions:
    python_to_spec: Parse Python code into TemplateSpec JSON
    spec_to_python: Generate Python code from TemplateSpec JSON
    detect_template_mode: Classify template as 'verified', 'classic', or 'mixed'
"""

import logging
import typing
from typing import Any, Literal

from karenina.benchmark.verification.utils.class_discovery import find_answer_class
from karenina.benchmark.verification.utils.template_validation import _build_exec_namespace
from karenina.schemas.entities import BaseAnswer
from karenina.schemas.entities.primitives import TracePrimitive, _reconstruct_primitive
from karenina.schemas.entities.template_spec import TemplateFieldSpec, TemplateSpec
from karenina.schemas.entities.verified_field import VerificationMeta

logger = logging.getLogger(__name__)

# Note: _build_exec_namespace and _reconstruct_primitive are private but
# used intentionally within the karenina ecosystem. The converter needs
# the same exec namespace and primitive reconstruction logic as the pipeline.


def detect_template_mode(code: str) -> Literal["verified", "classic", "mixed"]:
    """Detect whether template code uses VerifiedField, classic Field, or both.

    Args:
        code: Python source code string for a template class.

    Returns:
        'verified' if all fields use VerifiedField,
        'classic' if no fields use VerifiedField,
        'mixed' if some fields use VerifiedField and some use plain Field.

    Raises:
        ValueError: If the code cannot be compiled or no Answer class is found.
    """
    ns = _build_exec_namespace()
    try:
        exec(code, ns)  # noqa: S102
    except Exception as e:
        raise ValueError(f"Failed to compile template code: {e}") from e

    try:
        answer_cls: type[BaseAnswer] = find_answer_class(ns)
    except ValueError as e:
        raise ValueError(f"No Answer class found in template code: {e}") from e

    verified_fields = answer_cls._get_verified_fields()
    total_fields = {name for name in answer_cls.model_fields if name not in ("id", "correct")}

    if not total_fields:
        return "classic"

    verified_names = set(verified_fields)
    if verified_names == total_fields:
        return "verified"
    elif not verified_names:
        return "classic"
    else:
        return "mixed"


def python_to_spec(code: str) -> TemplateSpec:
    """Convert Python template code to a TemplateSpec JSON structure.

    Only works with VerifiedField-based templates (mode 'verified' or 'mixed').
    Classic templates raise ValueError since they cannot be represented
    losslessly in the TemplateSpec format.

    Args:
        code: Python source code string for a VerifiedField template class.

    Returns:
        TemplateSpec instance with all field definitions and metadata.

    Raises:
        ValueError: If the template is classic (no VerifiedField fields)
            or if the code cannot be compiled.
    """
    # Compile and find the answer class (single exec, reused below)
    ns = _build_exec_namespace()
    try:
        exec(code, ns)  # noqa: S102
    except Exception as e:
        raise ValueError(f"Failed to compile template code: {e}") from e

    try:
        answer_cls: type[BaseAnswer] = find_answer_class(ns)
    except ValueError as e:
        raise ValueError(f"No Answer class found in template code: {e}") from e

    # Detect mode from the already-compiled class
    verified_fields: dict[str, VerificationMeta] = answer_cls._get_verified_fields()
    if not verified_fields:
        raise ValueError(
            "Cannot convert classic template to TemplateSpec. "
            "Classic templates use manual verify() methods that cannot "
            "be represented in the structured format."
        )

    # Note: VerificationStrategy inner class extraction is not implemented
    # in this version. Custom strategies set in Python code will not round-trip
    # through the GUI builder. This is a known v1 limitation.

    fields: list[TemplateFieldSpec] = []

    for field_name, meta in verified_fields.items():
        field_info = answer_cls.model_fields[field_name]

        # Detect if this is a trace primitive
        primitive = _reconstruct_primitive(meta.verify_with)
        is_trace = isinstance(primitive, TracePrimitive)

        # Determine the field type string
        field_type = _python_type_to_spec_type(field_info.annotation)

        # Extract literal values if applicable
        literal_values = _extract_literal_values(field_info.annotation)

        fields.append(
            TemplateFieldSpec(
                name=field_name,
                type=field_type,
                description=field_info.description or "",
                extraction_hint=meta.extraction_hint,
                ground_truth=meta.ground_truth,
                literal_values=literal_values,
                verify_with=meta.verify_with,
                weight=meta.weight,
                is_trace=is_trace,
            )
        )

    return TemplateSpec(
        fields=fields,
        verify_strategy=None,
        class_name=answer_cls.__name__,
    )


def _python_type_to_spec_type(annotation: Any) -> str:
    """Map a Python type annotation to the spec type string."""
    origin = getattr(annotation, "__origin__", None)

    if annotation is bool:
        return "bool"
    elif annotation is int:
        return "int"
    elif annotation is float:
        return "float"
    elif annotation is str:
        return "str"
    elif origin is list:
        return "list_str"
    elif origin is typing.Literal:
        return "literal"
    else:
        return "str"  # Default fallback


def _extract_literal_values(annotation: Any) -> list[str] | None:
    """Extract Literal values from a type annotation."""
    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Literal:
        args = getattr(annotation, "__args__", ())
        return [str(a) for a in args]
    return None
