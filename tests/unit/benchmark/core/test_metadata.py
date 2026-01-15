"""Unit tests for MetadataManager class.

Tests cover:
- Getting and setting custom properties
- Removing properties
- Batch operations
- Prefix-based filtering
- Metadata statistics and export/import
- Timestamp handling
- Backup and restore
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


@pytest.mark.unit
class TestClearAllCustomProperties:
    """Tests for clear_all_custom_properties method."""

    def test_clear_empty_properties(self) -> None:
        """Test clearing when only default properties exist."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        # BenchmarkBase has a default benchmark_format_version property
        count = manager.clear_all_custom_properties()

        # Should clear the default property
        assert count == 1
        assert manager.get_all_custom_properties() == {}

    def test_clear_all_properties(self) -> None:
        """Test clearing all properties."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop1", "value1")
        manager.set_custom_property("prop2", "value2")
        manager.set_custom_property("prop3", "value3")

        count = manager.clear_all_custom_properties()

        # 3 custom + 1 default = 4 total
        assert count == 4
        assert manager.get_all_custom_properties() == {}

    def test_clear_updates_date_modified(self) -> None:
        """Test that clearing updates dateModified."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop", "value")
        initial_modified = base._checkpoint.dateModified
        manager.clear_all_custom_properties()
        updated_modified = base._checkpoint.dateModified

        assert updated_modified != initial_modified


@pytest.mark.unit
class TestHasCustomProperty:
    """Tests for has_custom_property method."""

    @pytest.mark.parametrize(
        "prop_name,prop_value,expected",
        [
            ("prop", "value", True),
            ("nonexistent", None, False),
            ("none_prop", None, False),  # None is a valid value but has_property returns False
        ],
        ids=["existing_property", "nonexistent_property", "none_value_property"],
    )
    def test_has_property(self, prop_name: str, prop_value: object, expected: bool) -> None:
        """Test has_property returns correct boolean for various scenarios."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        # Only set property if value is not None (for the nonexistent case)
        if prop_value is not None or prop_name == "none_prop":
            manager.set_custom_property(prop_name, prop_value)

        assert manager.has_custom_property(prop_name) is expected


@pytest.mark.unit
class TestGetCustomPropertiesByPrefix:
    """Tests for get_custom_properties_by_prefix method."""

    def test_get_by_prefix_empty(self) -> None:
        """Test getting by prefix when no properties match."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("other_prop", "value")

        result = manager.get_custom_properties_by_prefix("test_")

        assert result == {}

    def test_get_by_prefix_multiple_matches(self) -> None:
        """Test getting multiple properties by prefix."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("test_prop1", "value1")
        manager.set_custom_property("test_prop2", "value2")
        manager.set_custom_property("other_prop", "value3")

        result = manager.get_custom_properties_by_prefix("test_")

        assert result == {
            "test_prop1": "value1",
            "test_prop2": "value2",
        }

    def test_get_by_prefix_case_sensitive(self) -> None:
        """Test that prefix matching is case-sensitive."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("Test_prop", "value")
        manager.set_custom_property("test_prop", "value")

        result = manager.get_custom_properties_by_prefix("test_")

        assert result == {"test_prop": "value"}

    def test_get_by_prefix_empty_string(self) -> None:
        """Test getting by empty prefix returns all."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop1", "value1")
        manager.set_custom_property("prop2", "value2")

        result = manager.get_custom_properties_by_prefix("")

        # Includes default benchmark_format_version property
        assert "prop1" in result
        assert "prop2" in result
        assert result["prop1"] == "value1"
        assert result["prop2"] == "value2"


@pytest.mark.unit
class TestRemoveCustomPropertiesByPrefix:
    """Tests for remove_custom_properties_by_prefix method."""

    def test_remove_by_prefix_empty(self) -> None:
        """Test removing by prefix when no properties match."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("other_prop", "value")

        count = manager.remove_custom_properties_by_prefix("test_")

        assert count == 0
        assert manager.get_custom_property("other_prop") == "value"

    def test_remove_by_prefix_multiple(self) -> None:
        """Test removing multiple properties by prefix."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("test_prop1", "value1")
        manager.set_custom_property("test_prop2", "value2")
        manager.set_custom_property("other_prop", "value3")

        count = manager.remove_custom_properties_by_prefix("test_")

        assert count == 2
        assert manager.get_custom_property("test_prop1") is None
        assert manager.get_custom_property("test_prop2") is None
        assert manager.get_custom_property("other_prop") == "value3"

    def test_remove_by_prefix_updates_date_modified(self) -> None:
        """Test that removing by prefix updates dateModified when properties removed."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("test_prop", "value")
        initial_modified = base._checkpoint.dateModified
        manager.remove_custom_properties_by_prefix("test_")
        updated_modified = base._checkpoint.dateModified

        assert updated_modified != initial_modified


@pytest.mark.unit
class TestUpdateCustomProperty:
    """Tests for update_custom_property method."""

    def test_update_existing_property(self) -> None:
        """Test updating an existing property with a function."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("count", 5)

        result = manager.update_custom_property("count", lambda x: x * 2)

        assert result is True
        assert manager.get_custom_property("count") == 10

    def test_update_nonexistent_property(self) -> None:
        """Test updating a property that doesn't exist."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        result = manager.update_custom_property("nonexistent", lambda x: x * 2)

        assert result is False

    def test_update_with_complex_function(self) -> None:
        """Test updating with a more complex function."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("items", [1, 2, 3])

        result = manager.update_custom_property("items", lambda x: x + [4])

        assert result is True
        assert manager.get_custom_property("items") == [1, 2, 3, 4]


@pytest.mark.unit
class TestGetMetadataStatistics:
    """Tests for get_metadata_statistics method."""

    def test_statistics_empty(self) -> None:
        """Test statistics with only default properties."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        stats = manager.get_metadata_statistics()

        # BenchmarkBase has a default benchmark_format_version property
        assert stats["total_custom_properties"] == 1
        assert stats["property_types"]["str"] == 1
        assert stats["verification_results_stored"] == 0
        assert stats["has_custom_metadata"] is True

    def test_statistics_with_properties(self) -> None:
        """Test statistics with various property types."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("str_prop", "text")
        manager.set_custom_property("int_prop", 42)
        manager.set_custom_property("another_int", 100)
        manager.set_custom_property("bool_prop", True)

        stats = manager.get_metadata_statistics()

        # 4 custom + 1 default benchmark_format_version = 5 total
        assert stats["total_custom_properties"] == 5
        assert stats["property_types"]["str"] == 2  # text + default
        assert stats["property_types"]["int"] == 2
        assert stats["property_types"]["bool"] == 1
        assert stats["has_custom_metadata"] is True

    def test_statistics_with_verification_results(self) -> None:
        """Test statistics with verification result properties."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("verification_results_q1", "passed")
        manager.set_custom_property("verification_results_q2", "failed")
        manager.set_custom_property("other_prop", "value")

        stats = manager.get_metadata_statistics()

        # 3 custom + 1 default = 4 total
        assert stats["total_custom_properties"] == 4
        assert stats["verification_results_stored"] == 2


@pytest.mark.unit
class TestExportMetadata:
    """Tests for export_metadata method."""

    def test_export_empty_metadata(self) -> None:
        """Test exporting with only default properties."""
        base = BenchmarkBase(name="test_benchmark", description="Test description")
        manager = MetadataManager(base)

        exported = manager.export_metadata()

        assert "benchmark_metadata" in exported
        assert exported["benchmark_metadata"]["name"] == "test_benchmark"
        assert exported["benchmark_metadata"]["description"] == "Test description"
        # BenchmarkBase has a default benchmark_format_version property
        assert "benchmark_format_version" in exported["custom_properties"]
        assert exported["verification_results"] == {}
        assert exported["statistics"]["total_custom_properties"] == 1

    def test_export_with_properties(self) -> None:
        """Test exporting with custom properties."""
        base = BenchmarkBase(name="test", description="Test", version="1.0")
        manager = MetadataManager(base)

        manager.set_custom_property("custom1", "value1")
        manager.set_custom_property("custom2", 42)
        manager.set_custom_property("verification_results_q1", "passed")

        exported = manager.export_metadata()

        # Note: includes default benchmark_format_version property
        assert "custom1" in exported["custom_properties"]
        assert "custom2" in exported["custom_properties"]
        assert exported["custom_properties"]["custom1"] == "value1"
        assert exported["custom_properties"]["custom2"] == 42
        assert exported["verification_results"] == {
            "verification_results_q1": "passed",
        }

    def test_export_includes_statistics(self) -> None:
        """Test that export includes metadata statistics."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop", "value")

        exported = manager.export_metadata()

        assert "statistics" in exported
        # 1 custom + 1 default = 2 total
        assert exported["statistics"]["total_custom_properties"] == 2


@pytest.mark.unit
class TestImportMetadata:
    """Tests for import_metadata method."""

    def test_import_benchmark_metadata(self) -> None:
        """Test importing benchmark metadata."""
        base = BenchmarkBase(name="old_name", description="old_desc")
        manager = MetadataManager(base)

        metadata = {
            "benchmark_metadata": {
                "name": "new_name",
                "description": "new_description",
                "version": "2.0",
            }
        }

        manager.import_metadata(metadata)

        assert base.name == "new_name"
        assert base.description == "new_description"
        assert base.version == "2.0"

    def test_import_custom_properties(self) -> None:
        """Test importing custom properties."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        metadata = {
            "custom_properties": {
                "imported1": "value1",
                "imported2": 42,
            }
        }

        manager.import_metadata(metadata)

        assert manager.get_custom_property("imported1") == "value1"
        assert manager.get_custom_property("imported2") == 42

    def test_import_verification_results(self) -> None:
        """Test importing verification results."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        metadata = {
            "verification_results": {
                "verification_results_q1": "passed",
                "verification_results_q2": "failed",
            }
        }

        manager.import_metadata(metadata)

        assert manager.get_custom_property("verification_results_q1") == "passed"
        assert manager.get_custom_property("verification_results_q2") == "failed"


@pytest.mark.unit
class TestBackupAndRestore:
    """Tests for backup_metadata and restore_metadata methods."""

    def test_backup_creates_snapshot(self) -> None:
        """Test that backup creates a snapshot of metadata."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop1", "value1")
        manager.set_custom_property("prop2", "value2")

        backup = manager.backup_metadata()

        assert "timestamp" in backup
        assert "metadata" in backup
        assert backup["metadata"]["custom_properties"]["prop1"] == "value1"

    def test_restore_from_backup(self) -> None:
        """Test restoring from a backup."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        # Set initial properties
        manager.set_custom_property("prop1", "value1")
        manager.set_custom_property("prop2", "value2")

        # Create backup
        backup = manager.backup_metadata()

        # Clear and modify
        manager.clear_all_custom_properties()
        manager.set_custom_property("prop3", "value3")

        # Restore - import_metadata adds properties but doesn't clear existing ones
        # So we need to clear first for proper restore behavior
        manager.clear_all_custom_properties()
        manager.restore_metadata(backup)

        assert manager.get_custom_property("prop1") == "value1"
        assert manager.get_custom_property("prop2") == "value2"
        # prop3 was cleared before restore
        assert manager.get_custom_property("prop3") is None

    def test_restore_empty_backup(self) -> None:
        """Test restoring from an empty backup."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_custom_property("prop", "value")

        manager.restore_metadata({})

        # Properties should remain unchanged
        assert manager.get_custom_property("prop") == "value"


@pytest.mark.unit
class TestSetPropertyWithTimestamp:
    """Tests for set_property_with_timestamp method."""

    def test_set_with_timestamp(self) -> None:
        """Test setting a property with timestamp."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_property_with_timestamp("timestamped_prop", "value")

        result = manager.get_custom_property("timestamped_prop")

        assert isinstance(result, dict)
        assert result["value"] == "value"
        assert "timestamp" in result

    def test_set_with_timestamp_is_iso_format(self) -> None:
        """Test that timestamp is in ISO format."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_property_with_timestamp("prop", "value")

        stored = manager.get_custom_property("prop")
        timestamp = stored["timestamp"]

        # ISO format should contain 'T' and end with Z or timezone
        assert "T" in timestamp


@pytest.mark.unit
class TestGetPropertyWithTimestamp:
    """Tests for get_property_with_timestamp method."""

    def test_get_timestamped_property(self) -> None:
        """Test getting a timestamped property."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        manager.set_property_with_timestamp("prop", "value")

        value, timestamp = manager.get_property_with_timestamp("prop")

        assert value == "value"
        assert timestamp is not None
        assert "T" in timestamp

    @pytest.mark.parametrize(
        "setup_type,expected_value,has_timestamp",
        [
            ("regular", "value", False),
            ("nonexistent", None, False),
        ],
        ids=["nontimestamped_property", "nonexistent_property"],
    )
    def test_get_property_without_timestamp(
        self, setup_type: str, expected_value: str | None, has_timestamp: bool
    ) -> None:
        """Test getting properties without timestamps."""
        base = BenchmarkBase(name="test")
        manager = MetadataManager(base)

        if setup_type == "regular":
            manager.set_custom_property("prop", "value")

        value, timestamp = manager.get_property_with_timestamp("prop" if setup_type == "regular" else "nonexistent")

        assert value == expected_value
        assert (timestamp is not None) == has_timestamp
