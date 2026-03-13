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


def validate_spec(spec: TemplateSpec) -> list[str]:
    """Validate a TemplateSpec against the primitive registry.

    Checks that every field has a valid type, a recognized primitive with
    correct parameters, and that the primitive is compatible with the field
    type. Returns a list of error messages; an empty list means the spec
    is valid.

    Args:
        spec: The TemplateSpec to validate.

    Returns:
        List of human-readable error strings. Empty if valid.
    """
    from pydantic import ValidationError

    from karenina.schemas.entities.primitives import _PRIMITIVE_REGISTRY

    valid_field_types = {"bool", "str", "int", "float", "list_str", "literal", "date"}

    # Primitive-to-compatible-field-types mapping
    type_compatibility: dict[str, set[str]] = {
        "BooleanMatch": {"bool"},
        "ExactMatch": {"str", "int", "float"},
        "ContainsAny": {"str"},
        "ContainsAll": {"str"},
        "RegexMatch": {"str"},
        "SemanticMatch": {"str"},
        "NumericExact": {"int", "float"},
        "NumericTolerance": {"float"},
        "NumericRange": {"int", "float"},
        "SetContainment": {"list_str"},
        "OrderedMatch": {"list_str"},
        "LiteralMatch": {"literal", "str"},
        "DateMatch": {"date", "str"},
        "DateTolerance": {"date", "str"},
        "DateRange": {"date", "str"},
        "TraceRegex": {"bool"},
        "TraceContains": {"bool"},
        "TraceLength": {"bool"},
    }

    errors: list[str] = []

    for field in spec.fields:
        prefix = f"Field '{field.name}'"

        # 1. Check field type is valid
        if field.type not in valid_field_types:
            errors.append(f"{prefix}: invalid field type '{field.type}'. Must be one of: {sorted(valid_field_types)}")

        # 2. Check literal fields have literal_values
        if field.type == "literal" and not field.literal_values:
            errors.append(f"{prefix}: literal type requires 'literal_values'")

        # 3. Check verify_with has a 'type' key
        prim_type = field.verify_with.get("type")
        if not prim_type:
            errors.append(f"{prefix}: 'verify_with' must include a 'type' key")
            continue

        # 4. Check primitive name exists in registry
        if prim_type not in _PRIMITIVE_REGISTRY:
            available = sorted(_PRIMITIVE_REGISTRY.keys())
            errors.append(f"{prefix}: unknown primitive '{prim_type}'. Available: {available}")
            continue

        # 5. Check primitive parameters are valid (try to instantiate)
        try:
            _reconstruct_primitive(field.verify_with)
        except (ValueError, ValidationError) as e:
            errors.append(f"{prefix}: invalid parameters for '{prim_type}': {e}")
            continue

        # 6. Check primitive/field-type compatibility
        compatible_types = type_compatibility.get(prim_type, set())
        if compatible_types and field.type not in compatible_types:
            errors.append(
                f"{prefix}: primitive '{prim_type}' is not compatible with "
                f"field type '{field.type}'. Compatible types: {sorted(compatible_types)}"
            )

        # 7. Check trace primitives are only on bool fields
        if field.is_trace and field.type != "bool":
            errors.append(f"{prefix}: trace primitives require field type 'bool', got '{field.type}'")

    return errors


def spec_to_python(spec: TemplateSpec) -> str:
    """Generate Python template code from a TemplateSpec.

    Produces a complete, self-contained Python source string that defines
    a BaseAnswer subclass with VerifiedField declarations.

    Args:
        spec: TemplateSpec with field definitions and optional strategy.

    Returns:
        Python source code string, ready for exec() or file storage.

    Raises:
        ValueError: If the spec fails validation.
    """
    errors = validate_spec(spec)
    if errors:
        raise ValueError("Invalid TemplateSpec:\n" + "\n".join(f"  - {e}" for e in errors))

    lines: list[str] = []
    used_primitives: set[str] = set()

    # Collect used primitives
    for field in spec.fields:
        prim_type = field.verify_with.get("type", "")
        if prim_type:
            used_primitives.add(prim_type)

    # Build imports
    imports = ["BaseAnswer", "VerifiedField"]
    imports.extend(sorted(used_primitives))

    # Check if we need typing imports
    needs_literal = any(f.type == "literal" for f in spec.fields)

    if needs_literal:
        lines.append("from typing import Literal")
        lines.append("")

    lines.append(f"from karenina.schemas.entities import {', '.join(imports)}")
    lines.append("")
    lines.append("")
    lines.append(f"class {spec.class_name}(BaseAnswer):")

    # Field definitions
    for field in spec.fields:
        type_annotation = _spec_type_to_python_annotation(field)
        primitive_code = _primitive_dict_to_code(field.verify_with)
        ground_truth_repr = repr(field.ground_truth)

        lines.append(f"    {field.name}: {type_annotation} = VerifiedField(")
        escaped_desc = field.description.replace('"', '\\"')
        lines.append(f'        description="{escaped_desc}",')

        if field.extraction_hint:
            escaped_hint = field.extraction_hint.replace('"', '\\"')
            lines.append(f'        extraction_hint="{escaped_hint}",')

        lines.append(f"        ground_truth={ground_truth_repr},")
        lines.append(f"        verify_with={primitive_code},")

        if field.weight != 1.0:
            lines.append(f"        weight={field.weight},")

        lines.append("    )")

    # Add trailing newline
    lines.append("")

    return "\n".join(lines)


def _spec_type_to_python_annotation(field: TemplateFieldSpec) -> str:
    """Convert a spec type string to a Python type annotation."""
    type_map = {
        "bool": "bool",
        "str": "str",
        "int": "int",
        "float": "float",
        "list_str": "list[str]",
        "date": "str",
    }

    if field.type == "literal" and field.literal_values:
        values = ", ".join(repr(v) for v in field.literal_values)
        return f"Literal[{values}]"

    return type_map.get(field.type, "str")


def _primitive_dict_to_code(primitive: dict[str, Any]) -> str:
    """Convert a serialized primitive dict to Python constructor code.

    Args:
        primitive: Dict with 'type' key and optional parameters.

    Returns:
        Python expression string, e.g. 'ExactMatch(normalize=["lowercase"])'.
    """
    prim_type = primitive.get("type", "ExactMatch")
    params = {k: v for k, v in primitive.items() if k != "type"}

    if not params:
        return f"{prim_type}()"

    param_strs = [f"{k}={v!r}" for k, v in params.items()]
    return f"{prim_type}({', '.join(param_strs)})"
