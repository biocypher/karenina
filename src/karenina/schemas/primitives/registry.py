"""Primitive type registry for serialization and deserialization.

Provides a global registry mapping primitive class names to their types,
enabling reconstruction of primitive instances from serialized dicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from karenina.schemas.primitives.comparisons import VerificationPrimitive

_PRIMITIVE_REGISTRY: dict[str, type[VerificationPrimitive]] = {}


def _register_primitive(cls: type[VerificationPrimitive]) -> type[VerificationPrimitive]:
    """Register a primitive class for deserialization."""
    _PRIMITIVE_REGISTRY[cls.__name__] = cls
    return cls


def _reconstruct_primitive(data: dict[str, Any]) -> VerificationPrimitive:
    """Reconstruct a primitive instance from serialized dict.

    Args:
        data: Dict with 'type' key and primitive parameters.

    Returns:
        Instantiated primitive.

    Raises:
        ValueError: If the primitive type is not recognized.
    """
    data = dict(data)  # Copy to avoid mutating
    type_name = data.pop("type", None)
    if type_name is None:
        raise ValueError("Primitive data missing 'type' key")
    cls = _PRIMITIVE_REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown primitive type: {type_name!r}")
    return cls.model_validate(data)
