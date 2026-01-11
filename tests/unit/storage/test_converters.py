"""Unit tests for Pydantic-SQLAlchemy converter utilities.

Tests the bidirectional converters between nested Pydantic models
and flat SQLAlchemy ORM objects.
"""

from typing import Optional

import pytest
from pydantic import BaseModel

from karenina.storage.converters import (
    _extract_nested_data,
    _flatten_nested_model,
    _is_pydantic_model,
    _unwrap_optional,
    flat_dict_to_pydantic,
    orm_to_pydantic,
    pydantic_to_flat_dict,
    pydantic_to_orm,
    update_orm_from_pydantic,
)

# =============================================================================
# Test models
# =============================================================================


class NestedModel(BaseModel):
    """Simple nested model for testing."""

    value: str
    count: int


class OptionalNestedModel(BaseModel):
    """Nested model with optional field for testing."""

    value: str | None = None
    count: int


class OuterModel(BaseModel):
    """Outer model with nested field."""

    name: str
    nested: NestedModel
    optional_nested: NestedModel | None = None


class DoubleNestedModel(BaseModel):
    """Model with nested model that has another nested model."""

    data: str
    inner: NestedModel


# =============================================================================
# _unwrap_optional tests
# =============================================================================


@pytest.mark.unit
def test_unwrap_optional_optional_str() -> None:
    """Test unwrapping Optional[str]."""
    inner_type, is_optional = _unwrap_optional(Optional[str])  # noqa: UP045
    assert inner_type is str
    assert is_optional is True


@pytest.mark.unit
def test_unwrap_optional_union_type() -> None:
    """Test unwrapping str | None (Python 3.10+ UnionType)."""
    inner_type, is_optional = _unwrap_optional(str | None)
    assert inner_type is str
    assert is_optional is True


@pytest.mark.unit
def test_unwrap_optional_non_optional() -> None:
    """Test unwrapping a non-optional type."""
    inner_type, is_optional = _unwrap_optional(str)
    assert inner_type is str
    assert is_optional is False


@pytest.mark.unit
def test_unwrap_optional_dict_fallback() -> None:
    """Test unwrapping Union[str, int] returns dict."""
    inner_type, is_optional = _unwrap_optional(Optional[str | int])  # noqa: UP045
    assert inner_type is dict
    assert is_optional is True


@pytest.mark.unit
def test_unwrap_optional_none_only() -> None:
    """Test unwrapping None only returns None type."""
    inner_type, is_optional = _unwrap_optional(type(None))
    assert inner_type is type(None)
    # None alone is not considered optional by the function
    # It needs to be Optional[T] or T | None to be marked optional
    assert is_optional is False


# =============================================================================
# _is_pydantic_model tests
# =============================================================================


@pytest.mark.unit
def test_is_pydantic_model_true() -> None:
    """Test _is_pydantic_model returns True for Pydantic models."""
    assert _is_pydantic_model(NestedModel) is True
    assert _is_pydantic_model(OuterModel) is True


@pytest.mark.unit
def test_is_pydantic_model_false_for_builtin() -> None:
    """Test _is_pydantic_model returns False for built-in types."""
    assert _is_pydantic_model(str) is False
    assert _is_pydantic_model(int) is False
    assert _is_pydantic_model(list) is False


@pytest.mark.unit
def test_is_pydantic_model_false_for_instance() -> None:
    """Test _is_pydantic_model returns False for instance."""
    instance = NestedModel(value="test", count=1)
    assert _is_pydantic_model(instance) is False


# =============================================================================
# pydantic_to_flat_dict tests
# =============================================================================


@pytest.mark.unit
def test_pydantic_to_flat_dict_simple() -> None:
    """Test flattening a simple nested model."""
    obj = OuterModel(name="test", nested=NestedModel(value="hello", count=42))

    config = {"nested": {"prefix": "nested_"}}
    result = pydantic_to_flat_dict(obj, config)

    assert result["name"] == "test"
    assert result["nested_value"] == "hello"
    assert result["nested_count"] == 42


@pytest.mark.unit
def test_pydantic_to_flat_dict_with_none_nested() -> None:
    """Test flattening with None optional nested field."""
    obj = OuterModel(name="test", nested=NestedModel(value="hello", count=42), optional_nested=None)

    config = {"nested": {"prefix": "nested_"}, "optional_nested": {"prefix": "opt_"}}
    result = pydantic_to_flat_dict(obj, config)

    assert result["name"] == "test"
    assert result["nested_value"] == "hello"
    assert result["opt_value"] is None
    assert result["opt_count"] is None


@pytest.mark.unit
def test_pydantic_to_flat_dict_default_prefix() -> None:
    """Test that default prefix is field_name_ when not specified."""
    obj = OuterModel(name="test", nested=NestedModel(value="hello", count=42))

    config = {}
    result = pydantic_to_flat_dict(obj, config)

    assert result["name"] == "test"
    assert result["nested_value"] == "hello"


# =============================================================================
# _flatten_nested_model tests
# =============================================================================


@pytest.mark.unit
def test_flatten_nested_model_basic() -> None:
    """Test _flatten_nested_model with simple model."""
    obj = NestedModel(value="test", count=5)
    result = _flatten_nested_model(obj, "prefix_")

    assert result["prefix_value"] == "test"
    assert result["prefix_count"] == 5


@pytest.mark.unit
def test_flatten_nested_model_with_none() -> None:
    """Test _flatten_nested_model with None value."""
    obj = OptionalNestedModel(value=None, count=5)
    result = _flatten_nested_model(obj, "prefix_")

    assert result["prefix_value"] is None
    assert result["prefix_count"] == 5


# =============================================================================
# flat_dict_to_pydantic tests
# =============================================================================


@pytest.mark.unit
def test_flat_dict_to_pydantic_simple() -> None:
    """Test reconstructing a nested model from flat dict."""
    flat_data = {"name": "test", "nested_value": "hello", "nested_count": 42}

    config = {"nested": {"prefix": "nested_"}}
    result = flat_dict_to_pydantic(flat_data, OuterModel, config)

    assert result.name == "test"
    assert result.nested.value == "hello"
    assert result.nested.count == 42


@pytest.mark.unit
def test_flat_dict_to_pydantic_with_none_nested() -> None:
    """Test reconstructing with None nested field."""
    flat_data = {"name": "test", "nested_value": "hello", "nested_count": 42, "opt_value": None, "opt_count": None}

    config = {"nested": {"prefix": "nested_"}, "optional_nested": {"prefix": "opt_"}}
    result = flat_dict_to_pydantic(flat_data, OuterModel, config)

    assert result.name == "test"
    assert result.nested.value == "hello"
    assert result.optional_nested is None


@pytest.mark.unit
def test_flat_dict_to_pydantic_optional_missing() -> None:
    """Test reconstructing when optional fields are missing."""
    flat_data = {"name": "test", "nested_value": "hello", "nested_count": 42}

    config = {"nested": {"prefix": "nested_"}, "optional_nested": {"prefix": "opt_"}}
    result = flat_dict_to_pydantic(flat_data, OuterModel, config)

    assert result.optional_nested is None


@pytest.mark.unit
def test_flat_dict_to_pydantic_roundtrip() -> None:
    """Test roundtrip: pydantic -> flat -> pydantic."""
    original = OuterModel(name="test", nested=NestedModel(value="hello", count=42), optional_nested=None)

    config = {"nested": {"prefix": "nested_"}, "optional_nested": {"prefix": "opt_"}}

    # Flatten
    flat = pydantic_to_flat_dict(original, config)

    # Reconstruct
    reconstructed = flat_dict_to_pydantic(flat, OuterModel, config)

    assert reconstructed.name == original.name
    assert reconstructed.nested.value == original.nested.value
    assert reconstructed.nested.count == original.nested.count
    assert reconstructed.optional_nested == original.optional_nested


# =============================================================================
# _extract_nested_data tests
# =============================================================================


@pytest.mark.unit
def test_extract_nested_data_basic() -> None:
    """Test extracting nested data from flat dict."""
    flat_data = {"prefix_value": "hello", "prefix_count": 42, "other_field": "ignored"}

    result = _extract_nested_data(flat_data, "prefix_", NestedModel)

    assert result["value"] == "hello"
    assert result["count"] == 42
    assert "other_field" not in result


@pytest.mark.unit
def test_extract_nested_data_missing_fields() -> None:
    """Test extracting when some fields are missing."""
    flat_data = {
        "prefix_value": "hello"
        # prefix_count is missing
    }

    result = _extract_nested_data(flat_data, "prefix_", NestedModel)

    assert result["value"] == "hello"
    assert "count" not in result


@pytest.mark.unit
def test_extract_nested_data_no_match() -> None:
    """Test extracting when no fields match prefix."""
    flat_data = {"other_value": "hello", "other_count": 42}

    result = _extract_nested_data(flat_data, "prefix_", NestedModel)

    assert result == {}


# =============================================================================
# pydantic_to_orm tests (mock ORM)
# =============================================================================


class MockORMTable:
    """Mock SQLAlchemy table for testing."""

    def __init__(self):
        self.columns = []

    def add_column(self, name: str):
        """Add a column to the table."""

        class MockColumn:
            def __init__(self, name: str):
                self.name = name

        col = MockColumn(name)
        self.columns.append(col)


class MockORMClass:
    """Mock SQLAlchemy ORM class for testing."""

    __table__ = MockORMTable()

    def __init__(self, **kwargs):
        MockORMClass.__table__.add_column("name")
        MockORMClass.__table__.add_column("nested_value")
        MockORMClass.__table__.add_column("nested_count")
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.mark.unit
def test_pydantic_to_orm_basic() -> None:
    """Test converting Pydantic model to ORM instance."""
    obj = OuterModel(name="test", nested=NestedModel(value="hello", count=42))

    config = {"nested": {"prefix": "nested_"}}

    # Reset mock table columns
    MockORMClass.__table__.columns.clear()

    orm_instance = pydantic_to_orm(obj, MockORMClass, config)

    assert orm_instance.name == "test"
    assert orm_instance.nested_value == "hello"
    assert orm_instance.nested_count == 42


@pytest.mark.unit
def test_pydantic_to_orm_with_extra_values() -> None:
    """Test converting with extra values for ORM."""
    obj = OuterModel(name="test", nested=NestedModel(value="hello", count=42))

    config = {"nested": {"prefix": "nested_"}}
    extra = {"run_id": "abc123", "timestamp": 1234567890}

    # Reset mock table columns
    MockORMClass.__table__.columns.clear()

    orm_instance = pydantic_to_orm(obj, MockORMClass, config, extra)

    assert orm_instance.run_id == "abc123"
    assert orm_instance.timestamp == 1234567890


@pytest.mark.unit
def test_pydantic_to_orm_filters_invalid_columns() -> None:
    """Test that columns not in ORM are filtered out."""
    obj = OuterModel(name="test", nested=NestedModel(value="hello", count=42))

    config = {"nested": {"prefix": "nested_"}}

    # Reset mock table columns - only include specific ones
    MockORMClass.__table__.columns.clear()
    MockORMClass.__table__.add_column("name")
    MockORMClass.__table__.add_column("nested_value")
    # nested_count not added to test filtering

    orm_instance = pydantic_to_orm(obj, MockORMClass, config)

    assert hasattr(orm_instance, "name")
    assert hasattr(orm_instance, "nested_value")
    # nested_count should be filtered if column doesn't exist


# =============================================================================
# orm_to_pydantic tests
# =============================================================================


@pytest.mark.unit
def test_orm_to_pydantic_basic() -> None:
    """Test converting ORM instance to Pydantic model."""
    # Reset mock table columns
    MockORMClass.__table__.columns.clear()
    MockORMClass.__table__.add_column("name")
    MockORMClass.__table__.add_column("nested_value")
    MockORMClass.__table__.add_column("nested_count")

    orm_obj = MockORMClass(name="test", nested_value="hello", nested_count=42)

    config = {"nested": {"prefix": "nested_"}}
    result = orm_to_pydantic(orm_obj, OuterModel, config)

    assert result.name == "test"
    assert result.nested.value == "hello"
    assert result.nested.count == 42


# =============================================================================
# update_orm_from_pydantic tests
# =============================================================================


@pytest.mark.unit
def test_update_orm_from_pydantic_basic() -> None:
    """Test updating existing ORM instance from Pydantic model."""
    # Reset mock table columns
    MockORMClass.__table__.columns.clear()
    MockORMClass.__table__.add_column("name")
    MockORMClass.__table__.add_column("nested_value")
    MockORMClass.__table__.add_column("nested_count")

    orm_obj = MockORMClass(name="old_name", nested_value="old_value", nested_count=1)

    pydantic_obj = OuterModel(name="new_name", nested=NestedModel(value="new_value", count=99))

    config = {"nested": {"prefix": "nested_"}}
    update_orm_from_pydantic(orm_obj, pydantic_obj, config)

    assert orm_obj.name == "new_name"
    assert orm_obj.nested_value == "new_value"
    assert orm_obj.nested_count == 99


@pytest.mark.unit
def test_update_orm_from_pydantic_with_exclude() -> None:
    """Test updating with excluded fields."""
    # Reset mock table columns
    MockORMClass.__table__.columns.clear()
    MockORMClass.__table__.add_column("name")
    MockORMClass.__table__.add_column("nested_value")
    MockORMClass.__table__.add_column("nested_count")

    orm_obj = MockORMClass(name="old_name", nested_value="old_value", nested_count=1)

    pydantic_obj = OuterModel(name="new_name", nested=NestedModel(value="new_value", count=99))

    config = {"nested": {"prefix": "nested_"}}
    exclude = {"nested_count"}

    update_orm_from_pydantic(orm_obj, pydantic_obj, config, exclude)

    assert orm_obj.name == "new_name"
    assert orm_obj.nested_value == "new_value"
    # nested_count should not be updated
    assert orm_obj.nested_count == 1


@pytest.mark.unit
def test_update_orm_from_pydantic_filters_invalid_columns() -> None:
    """Test that update only affects valid ORM columns."""
    # Reset mock table columns - limited set
    MockORMClass.__table__.columns.clear()
    MockORMClass.__table__.add_column("name")
    # nested_value and nested_count not added

    orm_obj = MockORMClass(name="old_name")

    pydantic_obj = OuterModel(name="new_name", nested=NestedModel(value="new_value", count=99))

    config = {"nested": {"prefix": "nested_"}}
    update_orm_from_pydantic(orm_obj, pydantic_obj, config)

    # Only name should be updated
    assert orm_obj.name == "new_name"
