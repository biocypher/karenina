"""Shared utilities for the storage module.

This module consolidates type introspection utilities that were previously
duplicated across auto_mapper.py and converters.py in the storage subsystem.

Functions:
    is_pydantic_model: Check if a type is a Pydantic BaseModel subclass
    unwrap_optional: Unwrap Optional[T] or T | None to get the inner type
"""

from __future__ import annotations

import types
from typing import Union, get_args, get_origin

from pydantic import BaseModel

__all__ = ["is_pydantic_model", "unwrap_optional"]


def is_pydantic_model(field_type: type) -> bool:
    """Check if a type is a Pydantic BaseModel subclass.

    This function safely checks whether the given type is a subclass of Pydantic's
    BaseModel, handling edge cases where issubclass() might raise TypeError
    (e.g., for generic type aliases like list[str]).

    Args:
        field_type: The type to check

    Returns:
        True if field_type is a Pydantic BaseModel subclass, False otherwise

    Example:
        >>> from pydantic import BaseModel
        >>> class MyModel(BaseModel):
        ...     name: str
        >>> is_pydantic_model(MyModel)
        True
        >>> is_pydantic_model(str)
        False
        >>> is_pydantic_model(list[str])  # Generic alias - handled safely
        False
    """
    try:
        return isinstance(field_type, type) and issubclass(field_type, BaseModel)
    except TypeError:
        return False


def unwrap_optional(field_type: type) -> tuple[type, bool]:
    """Unwrap Optional[T] or T | None to get the inner type.

    Handles both typing.Union (for Optional[T]) and types.UnionType (for T | None
    syntax introduced in Python 3.10+).

    Args:
        field_type: The type annotation to unwrap

    Returns:
        Tuple of (inner_type, is_optional) where:
        - inner_type: The unwrapped type (or dict for complex unions)
        - is_optional: True if the original type was Optional/nullable

    Example:
        >>> from typing import Optional
        >>> unwrap_optional(Optional[str])
        (str, True)
        >>> unwrap_optional(str | None)  # Python 3.10+ syntax
        (str, True)
        >>> unwrap_optional(str)  # Non-optional type
        (str, False)
        >>> unwrap_optional(str | int | None)  # Complex union -> dict fallback
        (dict, True)
    """
    origin = get_origin(field_type)

    # Handle Union types - both typing.Union (Optional[T]) and types.UnionType (T | None)
    # Python 3.10+ uses types.UnionType for the T | None syntax
    if origin is Union or origin is types.UnionType:
        args = get_args(field_type)
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0], True
        elif len(non_none_args) > 1:
            # Complex union - treat as JSON
            return dict, True
        else:
            # Only None - shouldn't happen but handle gracefully
            return type(None), True

    return field_type, False
