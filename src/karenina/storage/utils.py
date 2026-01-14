"""Shared utilities for the storage module.

This module provides common helper functions used across the storage subsystem,
including type introspection utilities for working with Pydantic models.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["is_pydantic_model"]


def is_pydantic_model(field_type: type) -> bool:
    """Check if a type is a Pydantic BaseModel subclass.

    This function safely checks whether the given type is a subclass of Pydantic's
    BaseModel, handling edge cases where issubclass() might raise TypeError.

    Args:
        field_type: The type to check

    Returns:
        True if field_type is a Pydantic BaseModel subclass, False otherwise
    """
    try:
        return isinstance(field_type, type) and issubclass(field_type, BaseModel)
    except TypeError:
        return False
