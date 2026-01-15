"""Metadata management functionality for benchmarks."""

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BenchmarkBase


class MetadataManager:
    """Manager for custom properties and metadata operations."""

    def __init__(self, base: "BenchmarkBase") -> None:
        """Initialize with reference to benchmark base."""
        self.base = base

    def get_custom_property(self, name: str) -> Any:
        """
        Get a custom property from benchmark metadata.

        Args:
            name: Property name

        Returns:
            Property value or None if not found
        """
        if not self.base._checkpoint.additionalProperty:
            return None

        for prop in self.base._checkpoint.additionalProperty:
            if prop.name == name:
                return prop.value
        return None

    def set_custom_property(self, name: str, value: Any) -> None:
        """
        Set a custom property in benchmark metadata.

        Args:
            name: Property name
            value: Property value
        """
        from ...schemas.checkpoint import SchemaOrgPropertyValue

        if not self.base._checkpoint.additionalProperty:
            self.base._checkpoint.additionalProperty = []

        # Update existing property or create new one
        for prop in self.base._checkpoint.additionalProperty:
            if prop.name == name:
                prop.value = value
                self.base._checkpoint.dateModified = datetime.now().isoformat()
                return

        # Create new property
        new_prop = SchemaOrgPropertyValue(name=name, value=value)
        self.base._checkpoint.additionalProperty.append(new_prop)
        self.base._checkpoint.dateModified = datetime.now().isoformat()

    def remove_custom_property(self, name: str) -> bool:
        """
        Remove a custom property from benchmark metadata.

        Args:
            name: Property name

        Returns:
            True if property was found and removed, False otherwise
        """
        if not self.base._checkpoint.additionalProperty:
            return False

        for i, prop in enumerate(self.base._checkpoint.additionalProperty):
            if prop.name == name:
                del self.base._checkpoint.additionalProperty[i]
                self.base._checkpoint.dateModified = datetime.now().isoformat()
                return True
        return False

    def get_all_custom_properties(self) -> dict[str, Any]:
        """
        Get all custom properties as a dictionary.

        Returns:
            Dictionary of property name -> value pairs
        """
        if not self.base._checkpoint.additionalProperty:
            return {}

        return {prop.name: prop.value for prop in self.base._checkpoint.additionalProperty}

    def set_multiple_custom_properties(self, properties: dict[str, Any]) -> None:
        """
        Set multiple custom properties at once.

        Args:
            properties: Dictionary of property name -> value pairs
        """
        for name, value in properties.items():
            self.set_custom_property(name, value)

    def clear_all_custom_properties(self) -> int:
        """
        Remove all custom properties.

        Returns:
            Number of properties that were removed
        """
        if not self.base._checkpoint.additionalProperty:
            return 0

        count = len(self.base._checkpoint.additionalProperty)
        self.base._checkpoint.additionalProperty.clear()
        self.base._checkpoint.dateModified = datetime.now().isoformat()
        return count

    def has_custom_property(self, name: str) -> bool:
        """
        Check if a custom property exists.

        Args:
            name: Property name

        Returns:
            True if property exists, False otherwise
        """
        return self.get_custom_property(name) is not None

    def get_custom_properties_by_prefix(self, prefix: str) -> dict[str, Any]:
        """
        Get all custom properties that start with a given prefix.

        Args:
            prefix: Property name prefix

        Returns:
            Dictionary of matching property name -> value pairs
        """
        all_props = self.get_all_custom_properties()
        return {name: value for name, value in all_props.items() if name.startswith(prefix)}

    def remove_custom_properties_by_prefix(self, prefix: str) -> int:
        """
        Remove all custom properties that start with a given prefix.

        Args:
            prefix: Property name prefix

        Returns:
            Number of properties that were removed
        """
        if not self.base._checkpoint.additionalProperty:
            return 0

        # Find properties to remove
        props_to_remove = []
        for i, prop in enumerate(self.base._checkpoint.additionalProperty):
            if prop.name.startswith(prefix):
                props_to_remove.append(i)

        # Remove in reverse order to maintain indices
        for i in reversed(props_to_remove):
            del self.base._checkpoint.additionalProperty[i]

        if props_to_remove:
            self.base._checkpoint.dateModified = datetime.now().isoformat()

        return len(props_to_remove)

    def update_custom_property(self, name: str, updater: Callable[[Any], Any]) -> bool:
        """
        Update a custom property using a function.

        Args:
            name: Property name
            updater: Function that takes current value and returns new value

        Returns:
            True if property was found and updated, False otherwise
        """
        current_value = self.get_custom_property(name)
        if current_value is None:
            return False

        new_value = updater(current_value)
        self.set_custom_property(name, new_value)
        return True

    def get_metadata_statistics(self) -> dict[str, Any]:
        """
        Get statistics about metadata usage.

        Returns:
            Dictionary with metadata statistics
        """
        all_props = self.get_all_custom_properties()

        # Count properties by type
        type_counts: dict[str, int] = {}
        for value in all_props.values():
            value_type = type(value).__name__
            type_counts[value_type] = type_counts.get(value_type, 0) + 1

        # Count verification results
        verification_props = self.get_custom_properties_by_prefix("verification_results_")

        return {
            "total_custom_properties": len(all_props),
            "property_types": type_counts,
            "verification_results_stored": len(verification_props),
            "has_custom_metadata": len(all_props) > 0,
        }

    def export_metadata(self) -> dict[str, Any]:
        """
        Export all metadata in a structured format.

        Returns:
            Dictionary with organized metadata
        """
        all_props = self.get_all_custom_properties()

        # Organize properties by category
        verification_results = self.get_custom_properties_by_prefix("verification_results_")
        custom_properties = {
            name: value for name, value in all_props.items() if not name.startswith("verification_results_")
        }

        return {
            "benchmark_metadata": {
                "name": self.base.name,
                "description": self.base.description,
                "version": self.base.version,
                "creator": self.base.creator,
                "created_at": self.base.created_at,
                "modified_at": self.base.modified_at,
                "id": self.base.id,
            },
            "custom_properties": custom_properties,
            "verification_results": verification_results,
            "statistics": self.get_metadata_statistics(),
        }

    def import_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Import metadata from a structured format.

        Args:
            metadata: Dictionary with metadata to import
        """
        # Import benchmark metadata
        if "benchmark_metadata" in metadata:
            benchmark_meta = metadata["benchmark_metadata"]
            for key, value in benchmark_meta.items():
                if hasattr(self.base, key) and value is not None:
                    setattr(self.base, key, value)

        # Import custom properties
        if "custom_properties" in metadata:
            self.set_multiple_custom_properties(metadata["custom_properties"])

        # Import verification results
        if "verification_results" in metadata:
            self.set_multiple_custom_properties(metadata["verification_results"])

    def backup_metadata(self) -> dict[str, Any]:
        """
        Create a backup of all metadata.

        Returns:
            Dictionary with backup data
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "metadata": self.export_metadata(),
        }

    def restore_metadata(self, backup: dict[str, Any]) -> None:
        """
        Restore metadata from a backup.

        Args:
            backup: Backup data created by backup_metadata
        """
        if "metadata" in backup:
            self.import_metadata(backup["metadata"])

    def set_property_with_timestamp(self, name: str, value: Any) -> None:
        """
        Set a property with an associated timestamp.

        Args:
            name: Property name
            value: Property value
        """
        timestamp = datetime.now().isoformat()
        timestamped_value = {"value": value, "timestamp": timestamp}
        self.set_custom_property(name, timestamped_value)

    def get_property_with_timestamp(self, name: str) -> tuple[Any, str | None]:
        """
        Get a property value and its timestamp if available.

        Args:
            name: Property name

        Returns:
            Tuple of (value, timestamp). Timestamp is None if not available.
        """
        prop_value = self.get_custom_property(name)

        if isinstance(prop_value, dict) and "value" in prop_value and "timestamp" in prop_value:
            return prop_value["value"], prop_value["timestamp"]
        else:
            return prop_value, None
