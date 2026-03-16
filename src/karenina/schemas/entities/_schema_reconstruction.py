"""Reconstruct Pydantic BaseModel subclasses from JSON Schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, create_model

_JSON_SCHEMA_TO_PYTHON: dict[str, type[Any]] = {
    "integer": int,
    "number": float,
    "string": str,
    "boolean": bool,
}


def _reconstruct_model_from_schema(schema: dict[str, Any]) -> type[BaseModel]:
    """Rebuild a BaseModel subclass from a JSON Schema dict.

    Used during checkpoint deserialization to recover template-kind
    Pydantic classes from their stored JSON Schema representation.

    Args:
        schema: JSON Schema dict (from model_json_schema()).

    Returns:
        A dynamically created BaseModel subclass.
    """
    fields: dict[str, Any] = {}
    required = set(schema.get("required", []))

    for name, prop in schema.get("properties", {}).items():
        py_type = _resolve_type(prop)
        description = prop.get("description")
        default = prop.get("default", ... if name in required else None)
        fields[name] = (py_type, Field(default=default, description=description))

    title = schema.get("title", "ReconstructedTemplate")
    return create_model(title, **fields)


def _resolve_type(prop: dict[str, Any]) -> Any:
    """Map a JSON Schema property to a Python type.

    Returns a Python type or a union type (e.g. ``int | None``).
    The return is typed as ``Any`` because PEP 604 unions produce
    ``types.UnionType``, which is not a subclass of ``type``.
    """
    # Handle Optional (anyOf with null)
    if "anyOf" in prop:
        non_null = [s for s in prop["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            return _resolve_type(non_null[0]) | None
        return str  # Fallback

    # Handle array
    if prop.get("type") == "array":
        items = prop.get("items", {})
        item_type: str = items.get("type", "string")
        inner = _JSON_SCHEMA_TO_PYTHON.get(item_type, str)
        return list[inner]  # type: ignore[valid-type]

    # Handle primitive
    schema_type: str = prop.get("type", "string")
    return _JSON_SCHEMA_TO_PYTHON.get(schema_type, str)
