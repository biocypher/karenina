"""Metadata management functionality for benchmarks."""

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
