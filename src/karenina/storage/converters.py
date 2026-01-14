"""Bidirectional converters between Pydantic models and SQLAlchemy ORM objects.

This module provides utilities for converting between nested Pydantic BaseModel
instances and flat SQLAlchemy ORM objects, supporting the auto-generated schema
approach where nested structures are flattened into prefixed columns.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import BaseModel

from .utils import is_pydantic_model as _is_pydantic_model
from .utils import unwrap_optional as _unwrap_optional

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase


def pydantic_to_flat_dict(
    obj: BaseModel,
    flatten_config: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Convert a nested Pydantic model to a flat dictionary.

    Args:
        obj: The Pydantic model instance to flatten
        flatten_config: Configuration for how to flatten each field, e.g.:
            {
                "metadata": {"prefix": "metadata_", "optional": False},
                "template": {"prefix": "template_", "optional": True},
            }

    Returns:
        Flat dictionary with prefixed keys
    """
    result: dict[str, Any] = {}

    # Get type hints for the model
    try:
        hints = get_type_hints(type(obj))
    except Exception:
        hints = {name: field.annotation for name, field in type(obj).model_fields.items() if field.annotation}

    for field_name, field_type in hints.items():
        value = getattr(obj, field_name, None)
        config = flatten_config.get(field_name, {})
        prefix = config.get("prefix", f"{field_name}_")

        inner_type, _ = _unwrap_optional(field_type)

        if value is None:
            # For optional nested models, we need to set all nested fields to None
            if _is_pydantic_model(inner_type):
                try:
                    nested_hints = get_type_hints(inner_type)
                except Exception:
                    # Type checker doesn't know inner_type is BaseModel, but we checked with _is_pydantic_model
                    nested_hints = {
                        name: field.annotation
                        for name, field in inner_type.model_fields.items()  # type: ignore[attr-defined]
                        if field.annotation
                    }
                for nested_name in nested_hints:
                    result[f"{prefix}{nested_name}"] = None
            else:
                result[field_name] = None
        elif _is_pydantic_model(inner_type) and isinstance(value, BaseModel):
            # Flatten nested Pydantic model
            nested_data = _flatten_nested_model(value, prefix)
            result.update(nested_data)
        else:
            # Root-level field (not in flatten_config or not a nested model)
            result[field_name] = value

    return result


def _flatten_nested_model(obj: BaseModel, prefix: str) -> dict[str, Any]:
    """Recursively flatten a nested Pydantic model.

    Args:
        obj: The nested Pydantic model instance
        prefix: Prefix to add to all field names

    Returns:
        Flat dictionary with prefixed keys
    """
    result: dict[str, Any] = {}

    try:
        hints = get_type_hints(type(obj))
    except Exception:
        hints = {name: field.annotation for name, field in type(obj).model_fields.items() if field.annotation}

    for field_name, field_type in hints.items():
        value = getattr(obj, field_name, None)
        column_name = f"{prefix}{field_name}"

        inner_type, _ = _unwrap_optional(field_type)

        if _is_pydantic_model(inner_type) and isinstance(value, BaseModel):
            # Recursively flatten deeper nested models
            nested_data = _flatten_nested_model(value, f"{column_name}_")
            result.update(nested_data)
        else:
            result[column_name] = value

    return result


def pydantic_to_orm(
    obj: BaseModel,
    orm_class: type[DeclarativeBase],
    flatten_config: dict[str, dict[str, Any]],
    extra_values: dict[str, Any] | None = None,
) -> DeclarativeBase:
    """Convert a nested Pydantic model to a flat SQLAlchemy ORM instance.

    Args:
        obj: The Pydantic model instance to convert
        orm_class: The SQLAlchemy ORM class to instantiate
        flatten_config: Configuration for how to flatten each field
        extra_values: Additional values to set on the ORM instance (e.g., run_id)

    Returns:
        SQLAlchemy ORM instance with flattened data
    """
    # Get flat dictionary
    flat_data = pydantic_to_flat_dict(obj, flatten_config)

    # Add extra values
    if extra_values:
        flat_data.update(extra_values)

    # Filter to only include columns that exist on the ORM class
    # This prevents errors from extra fields in the Pydantic model
    valid_columns = set()
    if hasattr(orm_class, "__table__"):
        valid_columns = {c.name for c in orm_class.__table__.columns}

    filtered_data = {k: v for k, v in flat_data.items() if not valid_columns or k in valid_columns}

    return orm_class(**filtered_data)


def flat_dict_to_pydantic(
    data: dict[str, Any],
    pydantic_class: type[BaseModel],
    flatten_config: dict[str, dict[str, Any]],
) -> BaseModel:
    """Convert a flat dictionary to a nested Pydantic model.

    Args:
        data: Flat dictionary with prefixed keys
        pydantic_class: The Pydantic model class to instantiate
        flatten_config: Configuration that was used to flatten

    Returns:
        Nested Pydantic model instance
    """
    # Build nested structure
    nested_data: dict[str, Any] = {}

    try:
        hints = get_type_hints(pydantic_class)
    except Exception:
        hints = {name: field.annotation for name, field in pydantic_class.model_fields.items() if field.annotation}

    for field_name, field_type in hints.items():
        config = flatten_config.get(field_name, {})
        prefix = config.get("prefix", f"{field_name}_")

        inner_type, is_optional = _unwrap_optional(field_type)

        if _is_pydantic_model(inner_type):
            # Reconstruct nested model from prefixed columns
            nested_model_data = _extract_nested_data(data, prefix, inner_type)

            # Check if all values are None (the nested object wasn't set)
            if nested_model_data and all(v is None for v in nested_model_data.values()):
                nested_data[field_name] = None
            elif nested_model_data:
                try:
                    nested_data[field_name] = inner_type(**nested_model_data)
                except Exception:
                    # If construction fails, set to None for optional fields
                    if is_optional:
                        nested_data[field_name] = None
                    else:
                        raise
            else:
                nested_data[field_name] = None
        else:
            # Root-level field
            if field_name in data:
                nested_data[field_name] = data[field_name]

    return pydantic_class(**nested_data)


def _extract_nested_data(
    data: dict[str, Any],
    prefix: str,
    model_class: type[BaseModel],
) -> dict[str, Any]:
    """Extract data for a nested model from flat dictionary.

    Args:
        data: Flat dictionary with prefixed keys
        prefix: Prefix used for this nested model's fields
        model_class: The nested Pydantic model class

    Returns:
        Dictionary with unprefixed keys for the nested model
    """
    result: dict[str, Any] = {}

    try:
        hints = get_type_hints(model_class)
    except Exception:
        hints = {name: field.annotation for name, field in model_class.model_fields.items() if field.annotation}

    for field_name, field_type in hints.items():
        column_name = f"{prefix}{field_name}"
        inner_type, _ = _unwrap_optional(field_type)

        if _is_pydantic_model(inner_type):
            # Recursively extract deeper nested models
            nested_data = _extract_nested_data(data, f"{column_name}_", inner_type)
            if nested_data:
                try:
                    result[field_name] = inner_type(**nested_data)
                except Exception:
                    result[field_name] = None
        elif column_name in data:
            result[field_name] = data[column_name]

    return result


def orm_to_pydantic(
    orm_obj: DeclarativeBase,
    pydantic_class: type[BaseModel],
    flatten_config: dict[str, dict[str, Any]],
) -> BaseModel:
    """Convert a SQLAlchemy ORM instance to a nested Pydantic model.

    Args:
        orm_obj: The SQLAlchemy ORM instance
        pydantic_class: The Pydantic model class to instantiate
        flatten_config: Configuration that was used to flatten

    Returns:
        Nested Pydantic model instance
    """
    # Extract all column values as a flat dictionary
    flat_data: dict[str, Any] = {}

    if hasattr(orm_obj, "__table__"):
        for column in orm_obj.__table__.columns:
            flat_data[column.name] = getattr(orm_obj, column.name, None)
    else:
        # Fallback: try to get all attributes
        for key in dir(orm_obj):
            if not key.startswith("_"):
                with contextlib.suppress(Exception):
                    flat_data[key] = getattr(orm_obj, key)

    return flat_dict_to_pydantic(flat_data, pydantic_class, flatten_config)


def update_orm_from_pydantic(
    orm_obj: DeclarativeBase,
    pydantic_obj: BaseModel,
    flatten_config: dict[str, dict[str, Any]],
    exclude_fields: set[str] | None = None,
) -> None:
    """Update an existing ORM instance from a Pydantic model.

    Args:
        orm_obj: The SQLAlchemy ORM instance to update
        pydantic_obj: The Pydantic model with new values
        flatten_config: Configuration for how to flatten
        exclude_fields: Column names to exclude from update
    """
    flat_data = pydantic_to_flat_dict(pydantic_obj, flatten_config)
    exclude = exclude_fields or set()

    # Get valid columns
    valid_columns = set()
    if hasattr(orm_obj, "__table__"):
        valid_columns = {c.name for c in orm_obj.__table__.columns}

    for key, value in flat_data.items():
        if key not in exclude and (not valid_columns or key in valid_columns):
            setattr(orm_obj, key, value)
