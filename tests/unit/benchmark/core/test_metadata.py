"""Unit tests for MetadataManager class.

Tests cover:
- Getting and setting custom properties
- Removing properties
- Batch operations
"""

import pytest

from karenina.benchmark.core.base import BenchmarkBase
from karenina.benchmark.core.metadata import MetadataManager


@pytest.mark.unit
class TestMetadataManagerInit:
    """Tests for MetadataManager initialization."""

    def test_init_with_benchmark_base(self) -> None:
        """Test MetadataManager initialization with BenchmarkBase."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        assert manager.base is base
        # BenchmarkBase adds a default benchmark_format_version property
        assert manager.base._checkpoint.additionalProperty is not None
        assert len(manager.base._checkpoint.additionalProperty) > 0


@pytest.mark.unit
class TestGetCustomProperty:
    """Tests for get_custom_property method."""

    def test_get_property_from_empty_metadata(self) -> None:
        """Test getting property when none exist."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        result = manager.get_custom_property("nonexistent")

        assert result is None

    def test_get_existing_property(self) -> None:
        """Test getting an existing property."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("test_prop", "test_value")
        result = manager.get_custom_property("test_prop")

        assert result == "test_value"

    def test_get_nonexistent_property(self) -> None:
        """Test getting a property that doesn't exist."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop1", "value1")
        result = manager.get_custom_property("prop2")

        assert result is None

    def test_get_property_with_none_value(self) -> None:
        """Test getting a property with None value."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("none_prop", None)
        result = manager.get_custom_property("none_prop")

        assert result is None

    @pytest.mark.parametrize(
        "prop_name,prop_value",
        [
            ("string_prop", "text"),
            ("int_prop", 42),
            ("bool_prop", True),
            ("list_prop", [1, 2, 3]),
            ("dict_prop", {"key": "value"}),
        ],
        ids=["string", "int", "bool", "list", "dict"],
    )
    def test_get_property_with_various_types(self, prop_name: str, prop_value: object) -> None:
        """Test getting properties with different value types."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property(prop_name, prop_value)

        assert manager.get_custom_property(prop_name) == prop_value


@pytest.mark.unit
class TestSetCustomProperty:
    """Tests for set_custom_property method."""

    def test_set_new_property(self) -> None:
        """Test setting a new property."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("new_prop", "new_value")

        assert manager.get_custom_property("new_prop") == "new_value"

    def test_set_property_updates_existing(self) -> None:
        """Test that setting an existing property updates its value."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop", "value1")
        manager.set_custom_property("prop", "value2")

        assert manager.get_custom_property("prop") == "value2"

    def test_set_property_updates_date_modified(self) -> None:
        """Test that setting a property updates dateModified."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        initial_modified = base._checkpoint.dateModified
        manager.set_custom_property("prop", "value")
        updated_modified = base._checkpoint.dateModified

        assert updated_modified != initial_modified

    def test_set_multiple_properties(self) -> None:
        """Test setting multiple properties."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop1", "value1")
        manager.set_custom_property("prop2", "value2")
        manager.set_custom_property("prop3", "value3")

        assert manager.get_custom_property("prop1") == "value1"
        assert manager.get_custom_property("prop2") == "value2"
        assert manager.get_custom_property("prop3") == "value3"


@pytest.mark.unit
class TestRemoveCustomProperty:
    """Tests for remove_custom_property method."""

    def test_remove_existing_property(self) -> None:
        """Test removing an existing property."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop", "value")
        result = manager.remove_custom_property("prop")

        assert result is True
        assert manager.get_custom_property("prop") is None

    def test_remove_nonexistent_property(self) -> None:
        """Test removing a property that doesn't exist."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        result = manager.remove_custom_property("nonexistent")

        assert result is False

    def test_remove_from_empty_metadata(self) -> None:
        """Test removing from empty metadata."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        result = manager.remove_custom_property("any")

        assert result is False

    def test_remove_updates_date_modified(self) -> None:
        """Test that removing a property updates dateModified."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop", "value")
        initial_modified = base._checkpoint.dateModified
        manager.remove_custom_property("prop")
        updated_modified = base._checkpoint.dateModified

        assert updated_modified != initial_modified

    def test_remove_one_of_many_properties(self) -> None:
        """Test removing one property when multiple exist."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop1", "value1")
        manager.set_custom_property("prop2", "value2")
        manager.set_custom_property("prop3", "value3")

        manager.remove_custom_property("prop2")

        assert manager.get_custom_property("prop1") == "value1"
        assert manager.get_custom_property("prop2") is None
        assert manager.get_custom_property("prop3") == "value3"


@pytest.mark.unit
class TestGetAllCustomProperties:
    """Tests for get_all_custom_properties method."""

    def test_get_all_from_empty_metadata(self) -> None:
        """Test getting all properties with only default properties."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        result = manager.get_all_custom_properties()

        # BenchmarkBase has a default benchmark_format_version property
        assert "benchmark_format_version" in result
        assert result["benchmark_format_version"] == "3.0.0-jsonld"

    def test_get_all_returns_all_properties(self) -> None:
        """Test getting all properties returns complete dictionary."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop1", "value1")
        manager.set_custom_property("prop2", "value2")
        manager.set_custom_property("prop3", "value3")

        result = manager.get_all_custom_properties()

        # Includes default benchmark_format_version property
        assert "prop1" in result
        assert "prop2" in result
        assert "prop3" in result
        assert result["prop1"] == "value1"
        assert result["prop2"] == "value2"
        assert result["prop3"] == "value3"

    def test_get_all_preserves_types(self) -> None:
        """Test that get_all preserves value types."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("str_prop", "text")
        manager.set_custom_property("int_prop", 42)
        manager.set_custom_property("bool_prop", True)

        result = manager.get_all_custom_properties()

        assert result["str_prop"] == "text"
        assert result["int_prop"] == 42
        assert result["bool_prop"] is True


@pytest.mark.unit
class TestSetMultipleCustomProperties:
    """Tests for set_multiple_custom_properties method."""

    def test_set_multiple_empty_dict(self) -> None:
        """Test setting multiple properties with empty dict."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        # BenchmarkBase has a default benchmark_format_version property
        initial_count = len(manager.get_all_custom_properties())

        manager.set_multiple_custom_properties({})

        # Count should be unchanged
        assert len(manager.get_all_custom_properties()) == initial_count

    def test_set_multiple_new_properties(self) -> None:
        """Test setting multiple new properties."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_multiple_custom_properties(
            {
                "prop1": "value1",
                "prop2": "value2",
                "prop3": "value3",
            }
        )

        assert manager.get_custom_property("prop1") == "value1"
        assert manager.get_custom_property("prop2") == "value2"
        assert manager.get_custom_property("prop3") == "value3"

    def test_set_multiple_updates_existing(self) -> None:
        """Test that set_multiple updates existing properties."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop1", "old_value")
        manager.set_multiple_custom_properties(
            {
                "prop1": "new_value",
                "prop2": "value2",
            }
        )

        assert manager.get_custom_property("prop1") == "new_value"
        assert manager.get_custom_property("prop2") == "value2"
