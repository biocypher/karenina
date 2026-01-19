"""Tests for adapter registry module.

This module tests the AdapterRegistry and verifies consistency between:
- Registry-registered interfaces
- ModelConfig Literal type definition
- Interface constants in factory module
"""

from __future__ import annotations

from typing import get_args

import pytest


class TestAdapterRegistry:
    """Tests for AdapterRegistry class."""

    def test_registry_has_expected_interfaces(self) -> None:
        """Test registry contains all expected interfaces."""
        from karenina.adapters.registry import AdapterRegistry

        interfaces = AdapterRegistry.get_interfaces()

        assert "langchain" in interfaces
        assert "openrouter" in interfaces
        assert "openai_endpoint" in interfaces
        assert "claude_agent_sdk" in interfaces
        assert "manual" in interfaces

    def test_get_spec_returns_spec_for_registered_interface(self) -> None:
        """Test get_spec returns AdapterSpec for registered interface."""
        from karenina.adapters.registry import AdapterRegistry, AdapterSpec

        spec = AdapterRegistry.get_spec("langchain")

        assert spec is not None
        assert isinstance(spec, AdapterSpec)
        assert spec.interface == "langchain"
        assert spec.agent_factory is not None
        assert spec.llm_factory is not None
        assert spec.parser_factory is not None

    def test_get_spec_returns_none_for_unknown_interface(self) -> None:
        """Test get_spec returns None for unregistered interface."""
        from karenina.adapters.registry import AdapterRegistry

        spec = AdapterRegistry.get_spec("nonexistent_interface")

        assert spec is None

    def test_check_availability_returns_availability(self) -> None:
        """Test check_availability returns AdapterAvailability."""
        from karenina.adapters.registry import AdapterAvailability, AdapterRegistry

        result = AdapterRegistry.check_availability("langchain")

        assert isinstance(result, AdapterAvailability)
        assert result.available is True

    def test_check_availability_unknown_interface(self) -> None:
        """Test check_availability for unknown interface."""
        from karenina.adapters.registry import AdapterRegistry

        result = AdapterRegistry.check_availability("nonexistent_interface")

        assert result.available is False
        assert "Unknown interface" in result.reason

    def test_format_model_string_langchain(self) -> None:
        """Test format_model_string for langchain interface."""
        from karenina.adapters.registry import AdapterRegistry
        from karenina.schemas.workflow.models import ModelConfig

        config = ModelConfig(
            id="test",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="langchain",
        )

        result = AdapterRegistry.format_model_string(config)

        assert result == "anthropic/claude-sonnet-4-20250514"

    def test_format_model_string_openrouter(self) -> None:
        """Test format_model_string for openrouter interface."""
        from karenina.adapters.registry import AdapterRegistry
        from karenina.schemas.workflow.models import ModelConfig

        config = ModelConfig(
            id="test",
            model_name="anthropic/claude-3-sonnet",
            model_provider="openrouter",
            interface="openrouter",
        )

        result = AdapterRegistry.format_model_string(config)

        assert result == "anthropic/claude-3-sonnet"

    def test_format_model_string_openai_endpoint(self) -> None:
        """Test format_model_string for openai_endpoint interface."""
        from karenina.adapters.registry import AdapterRegistry
        from karenina.schemas.workflow.models import ModelConfig

        config = ModelConfig(
            id="test",
            model_name="gpt-4",
            model_provider="openai",
            interface="openai_endpoint",
        )

        result = AdapterRegistry.format_model_string(config)

        assert result == "endpoint/gpt-4"

    def test_resolve_interface_no_routing(self) -> None:
        """Test resolve_interface for interface without routing."""
        from karenina.adapters.registry import AdapterRegistry

        result = AdapterRegistry.resolve_interface("langchain")

        assert result == "langchain"

    def test_resolve_interface_with_routing(self) -> None:
        """Test resolve_interface for interface that routes to another."""
        from karenina.adapters.registry import AdapterRegistry

        # openrouter routes to langchain
        result = AdapterRegistry.resolve_interface("openrouter")

        assert result == "langchain"


class TestRegistryConsistency:
    """Tests to verify consistency between registry and other interface definitions."""

    def test_model_config_literal_matches_registry(self) -> None:
        """Verify ModelConfig Literal type includes all registered interfaces.

        This test ensures the hardcoded Literal type in ModelConfig stays in sync
        with the dynamically registered adapters in the registry.
        """
        from karenina.adapters.registry import AdapterRegistry
        from karenina.schemas.workflow.models import ModelConfig

        # Get interfaces from registry
        registry_interfaces = AdapterRegistry.get_interfaces()

        # Get interfaces from ModelConfig Literal type
        interface_field = ModelConfig.model_fields["interface"]
        literal_type = interface_field.annotation
        literal_interfaces = set(get_args(literal_type))

        # Verify they match
        assert literal_interfaces == registry_interfaces, (
            f"ModelConfig Literal type is out of sync with registry.\\n"
            f"Literal has: {sorted(literal_interfaces)}\\n"
            f"Registry has: {sorted(registry_interfaces)}\\n"
            f"Missing from Literal: {sorted(registry_interfaces - literal_interfaces)}\\n"
            f"Extra in Literal: {sorted(literal_interfaces - registry_interfaces)}"
        )

    def test_factory_constants_match_registry(self) -> None:
        """Verify factory constants are consistent with registry.

        This test ensures the interface constants in factory.py are valid
        registered interfaces.
        """
        from karenina.adapters.factory import (
            INTERFACE_CLAUDE_AGENT_SDK,
            INTERFACE_MANUAL,
            LANGCHAIN_ROUTED_INTERFACES,
        )
        from karenina.adapters.registry import AdapterRegistry

        registry_interfaces = AdapterRegistry.get_interfaces()

        # Check constants are in registry
        assert INTERFACE_CLAUDE_AGENT_SDK in registry_interfaces
        assert INTERFACE_MANUAL in registry_interfaces

        # Check all routed interfaces are in registry
        for interface in LANGCHAIN_ROUTED_INTERFACES:
            assert interface in registry_interfaces, f"Routed interface '{interface}' not in registry"


class TestRegistrationBehavior:
    """Tests for adapter registration behavior."""

    def test_duplicate_registration_raises_error(self) -> None:
        """Test that registering the same interface twice raises ValueError."""
        from karenina.adapters.registry import AdapterRegistry, AdapterSpec

        # Use a unique interface name to avoid conflicts with real registrations
        spec = AdapterSpec(
            interface="test_unique_interface_duplicate",
            description="Test adapter",
        )

        # First registration should succeed
        AdapterRegistry.register(spec)

        # Second registration should raise
        with pytest.raises(ValueError) as exc_info:
            AdapterRegistry.register(spec)

        assert "already registered" in str(exc_info.value)
        assert "test_unique_interface_duplicate" in str(exc_info.value)

    def test_duplicate_registration_with_force_overwrites(self) -> None:
        """Test that force=True allows overwriting an existing registration."""
        from karenina.adapters.registry import AdapterRegistry, AdapterSpec

        # Use a unique interface name to avoid conflicts
        spec1 = AdapterSpec(
            interface="test_unique_interface_force",
            description="First adapter",
        )
        spec2 = AdapterSpec(
            interface="test_unique_interface_force",
            description="Second adapter",
        )

        AdapterRegistry.register(spec1)
        AdapterRegistry.register(spec2, force=True)

        # Should have the second spec
        result = AdapterRegistry.get_spec("test_unique_interface_force")
        assert result is not None
        assert result.description == "Second adapter"


class TestManualAdapterSpec:
    """Tests specifically for manual adapter registration."""

    def test_manual_adapter_registered(self) -> None:
        """Test manual adapter is properly registered."""
        from karenina.adapters.registry import AdapterRegistry

        spec = AdapterRegistry.get_spec("manual")

        assert spec is not None
        assert spec.interface == "manual"
        assert spec.description is not None
        assert "pre-recorded" in spec.description.lower()

    def test_manual_adapter_factories_return_manual_adapters(self) -> None:
        """Test manual adapter factories return ManualAdapter instances."""
        from unittest.mock import MagicMock

        from karenina.adapters.manual import ManualAgentAdapter, ManualLLMAdapter, ManualParserAdapter
        from karenina.adapters.registry import AdapterRegistry

        spec = AdapterRegistry.get_spec("manual")
        mock_config = MagicMock()

        assert spec is not None
        assert spec.agent_factory is not None
        assert spec.llm_factory is not None
        assert spec.parser_factory is not None

        # Create adapters
        agent = spec.agent_factory(mock_config)
        llm = spec.llm_factory(mock_config)
        parser = spec.parser_factory(mock_config)

        assert isinstance(agent, ManualAgentAdapter)
        assert isinstance(llm, ManualLLMAdapter)
        assert isinstance(parser, ManualParserAdapter)
