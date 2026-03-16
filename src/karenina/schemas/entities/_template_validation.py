"""Validation helpers for template-kind Pydantic classes."""

import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

_ALLOWED_PRIMITIVE_TYPES = (int, float, str, bool)
_UNION_TYPES = (Union, types.UnionType)  # Union for typing.Optional, UnionType for PEP 604


def _validate_template_fields(cls: type[BaseModel]) -> None:
    """Ensure template class only uses primitives + list[primitive].

    Optional fields must have explicit defaults to survive JSON Schema
    roundtrip (Pydantic v2 emits default values in the schema).

    Raises:
        ValueError: If a field uses a disallowed type.
    """
    for name, field_info in cls.model_fields.items():
        annotation: type[Any] | None = field_info.annotation
        if annotation is None:
            raise ValueError(f"Field '{name}': annotation is None, cannot validate")
        _check_annotation(name, annotation, field_info)


def _check_annotation(name: str, annotation: type[Any], field_info: FieldInfo) -> None:
    """Validate a single field annotation."""
    origin = get_origin(annotation)

    # Handle Optional[T] (Union[T, None] or T | None via PEP 604)
    if origin in _UNION_TYPES:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) != 1:
            raise ValueError(f"Field '{name}': only Optional[T] unions allowed, got {annotation}")
        # Optional fields must have defaults for schema roundtrip
        if field_info.is_required():
            raise ValueError(
                f"Field '{name}': Optional fields must have an explicit "
                f"default value for JSON Schema roundtrip fidelity"
            )
        inner = args[0]
        inner_origin = get_origin(inner)
        if inner_origin is list:
            _check_list_inner(name, inner)
        elif inner not in _ALLOWED_PRIMITIVE_TYPES:
            raise ValueError(
                f"Field '{name}': type {inner} not allowed in template kind. "
                f"Allowed: {_ALLOWED_PRIMITIVE_TYPES} and list[primitive]."
            )
        return

    # Handle list[T]
    if origin is list:
        _check_list_inner(name, annotation)
        return

    # Handle bare primitives
    if annotation in _ALLOWED_PRIMITIVE_TYPES:
        return

    raise ValueError(
        f"Field '{name}': type {annotation} not allowed in template kind. "
        f"Allowed: {_ALLOWED_PRIMITIVE_TYPES} and list[primitive]."
    )


def _check_list_inner(name: str, annotation: type) -> None:
    """Validate list element type."""
    args = get_args(annotation)
    if not args or args[0] not in _ALLOWED_PRIMITIVE_TYPES:
        raise ValueError(f"Field '{name}': list must contain primitive type ({_ALLOWED_PRIMITIVE_TYPES}), got {args}")
