"""Unit tests for storage utility functions.

Tests the shared utilities in storage/utils.py:
- is_pydantic_model(): Check if a type is a Pydantic BaseModel subclass
- unwrap_optional(): Unwrap Optional[T] or T | None to get the inner type
"""

from typing import Optional

import pytest
from pydantic import BaseModel

from karenina.storage.utils import is_pydantic_model, unwrap_optional

# =============================================================================
# Test models
# =============================================================================


class SimpleModel(BaseModel):
    """Simple Pydantic model for testing."""

    name: str
    value: int


class NestedModel(BaseModel):
    """Model with nested Pydantic model."""

    data: str
    inner: SimpleModel


class NonPydanticClass:
    """Regular Python class (not a Pydantic model)."""

    def __init__(self, value: str) -> None:
        self.value = value


# =============================================================================
# is_pydantic_model tests
# =============================================================================


@pytest.mark.unit
def test_is_pydantic_model_with_pydantic_model() -> None:
    """Test is_pydantic_model returns True for Pydantic model classes."""
    assert is_pydantic_model(SimpleModel) is True
    assert is_pydantic_model(NestedModel) is True


@pytest.mark.unit
def test_is_pydantic_model_with_base_model() -> None:
    """Test is_pydantic_model returns True for BaseModel itself."""
    assert is_pydantic_model(BaseModel) is True


@pytest.mark.unit
def test_is_pydantic_model_with_builtin_types() -> None:
    """Test is_pydantic_model returns False for built-in types."""
    assert is_pydantic_model(str) is False
    assert is_pydantic_model(int) is False
    assert is_pydantic_model(float) is False
    assert is_pydantic_model(bool) is False
    assert is_pydantic_model(list) is False
    assert is_pydantic_model(dict) is False
    assert is_pydantic_model(tuple) is False
    assert is_pydantic_model(set) is False


@pytest.mark.unit
def test_is_pydantic_model_with_regular_class() -> None:
    """Test is_pydantic_model returns False for regular Python classes."""
    assert is_pydantic_model(NonPydanticClass) is False


@pytest.mark.unit
def test_is_pydantic_model_with_instance() -> None:
    """Test is_pydantic_model returns False for model instances (not classes)."""
    instance = SimpleModel(name="test", value=42)
    assert is_pydantic_model(instance) is False  # type: ignore[arg-type]


@pytest.mark.unit
def test_is_pydantic_model_with_none() -> None:
    """Test is_pydantic_model returns False for None."""
    assert is_pydantic_model(None) is False  # type: ignore[arg-type]


@pytest.mark.unit
def test_is_pydantic_model_with_type_none() -> None:
    """Test is_pydantic_model returns False for type(None)."""
    assert is_pydantic_model(type(None)) is False


@pytest.mark.unit
def test_is_pydantic_model_with_generic_alias() -> None:
    """Test is_pydantic_model returns False for generic type aliases."""
    assert is_pydantic_model(list[str]) is False
    assert is_pydantic_model(dict[str, int]) is False


# =============================================================================
# unwrap_optional tests
# =============================================================================


@pytest.mark.unit
def test_unwrap_optional_with_optional_type() -> None:
    """Test unwrapping Optional[T] (typing.Union[T, None])."""
    inner_type, is_optional = unwrap_optional(Optional[str])  # noqa: UP045
    assert inner_type is str
    assert is_optional is True


@pytest.mark.unit
def test_unwrap_optional_with_union_none_syntax() -> None:
    """Test unwrapping T | None (Python 3.10+ UnionType)."""
    inner_type, is_optional = unwrap_optional(str | None)
    assert inner_type is str
    assert is_optional is True


@pytest.mark.unit
def test_unwrap_optional_with_non_optional() -> None:
    """Test unwrapping a non-optional type returns the same type."""
    inner_type, is_optional = unwrap_optional(str)
    assert inner_type is str
    assert is_optional is False


@pytest.mark.unit
def test_unwrap_optional_with_int() -> None:
    """Test unwrapping int | None."""
    inner_type, is_optional = unwrap_optional(int | None)
    assert inner_type is int
    assert is_optional is True


@pytest.mark.unit
def test_unwrap_optional_with_pydantic_model() -> None:
    """Test unwrapping Optional[PydanticModel]."""
    inner_type, is_optional = unwrap_optional(Optional[SimpleModel])  # noqa: UP045
    assert inner_type is SimpleModel
    assert is_optional is True


@pytest.mark.unit
def test_unwrap_optional_with_complex_union() -> None:
    """Test unwrapping Union[str, int, None] returns dict (complex union fallback)."""
    # Test using Optional[str | int] which is Union[str, int, None]
    inner_type, is_optional = unwrap_optional(Optional[str | int])  # noqa: UP045
    assert inner_type is dict
    assert is_optional is True


@pytest.mark.unit
def test_unwrap_optional_with_complex_union_pipe_syntax() -> None:
    """Test unwrapping str | int | None returns dict."""
    inner_type, is_optional = unwrap_optional(str | int | None)
    assert inner_type is dict
    assert is_optional is True


@pytest.mark.unit
def test_unwrap_optional_with_none_type() -> None:
    """Test unwrapping type(None) returns itself (edge case)."""
    inner_type, is_optional = unwrap_optional(type(None))
    assert inner_type is type(None)
    # type(None) alone is not a Union, so not considered optional
    assert is_optional is False


@pytest.mark.unit
def test_unwrap_optional_with_list() -> None:
    """Test unwrapping list[str] | None."""
    inner_type, is_optional = unwrap_optional(list[str] | None)
    assert inner_type == list[str]
    assert is_optional is True


@pytest.mark.unit
def test_unwrap_optional_with_dict() -> None:
    """Test unwrapping dict[str, int] | None."""
    inner_type, is_optional = unwrap_optional(dict[str, int] | None)
    assert inner_type == dict[str, int]
    assert is_optional is True


@pytest.mark.unit
def test_unwrap_optional_non_union_returns_same() -> None:
    """Test unwrapping a non-union type returns the exact same type."""
    for type_to_test in [str, int, float, bool, list, dict, SimpleModel]:
        inner_type, is_optional = unwrap_optional(type_to_test)
        assert inner_type is type_to_test
        assert is_optional is False
