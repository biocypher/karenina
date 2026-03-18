"""Conformance tests for adapter registration.

Validates that all registered adapters have properly configured specs
in the AdapterRegistry.
"""

from __future__ import annotations

import pytest

from karenina.adapters.registry import AdapterRegistry


@pytest.mark.unit
class TestAdapterRegistrationConformance:
    """Verify all registered adapters have valid specs."""

    def test_all_interfaces_have_specs(self, all_registered_interfaces):
        """Every registered interface must have a non-None spec."""
        for interface in all_registered_interfaces:
            spec = AdapterRegistry.get_spec(interface)
            assert spec is not None, f"No spec for interface: {interface}"

    def test_all_specs_have_at_least_one_factory(self, all_registered_interfaces):
        """Each spec must provide at least one factory (agent, llm, or parser)."""
        for interface in all_registered_interfaces:
            spec = AdapterRegistry.get_spec(interface)
            has_factory = (
                spec.agent_factory is not None or spec.llm_factory is not None or spec.parser_factory is not None
            )
            assert has_factory, f"Interface {interface} has no factories"

    def test_supports_flags_are_booleans(self, all_registered_interfaces):
        """Feature flags must be boolean values."""
        for interface in all_registered_interfaces:
            spec = AdapterRegistry.get_spec(interface)
            assert isinstance(spec.supports_mcp, bool), f"{interface}: supports_mcp not bool"
            assert isinstance(spec.supports_tools, bool), f"{interface}: supports_tools not bool"
            assert isinstance(spec.natively_agentic, bool), f"{interface}: natively_agentic not bool"

    def test_natively_agentic_adapters_support_tools(self, all_registered_interfaces):
        """Natively agentic adapters should support tools (they handle tool loops internally)."""
        for interface in all_registered_interfaces:
            spec = AdapterRegistry.get_spec(interface)
            if spec.natively_agentic:
                assert spec.supports_tools, f"{interface} is natively_agentic but doesn't support tools"

    def test_fallback_interface_is_registered_or_none(self, all_registered_interfaces):
        """If a fallback is specified, it must be a registered interface."""
        registered = set(all_registered_interfaces)
        for interface in all_registered_interfaces:
            spec = AdapterRegistry.get_spec(interface)
            if spec.fallback_interface is not None:
                assert spec.fallback_interface in registered, (
                    f"{interface} fallback '{spec.fallback_interface}' not registered"
                )

    def test_langchain_deep_agents_spec(self):
        """Specific validation for the langchain_deep_agents adapter."""
        spec = AdapterRegistry.get_spec("langchain_deep_agents")
        if spec is None:
            pytest.skip("langchain_deep_agents not registered")
        assert spec.natively_agentic is True
        assert spec.supports_mcp is True
        assert spec.supports_tools is True
        assert spec.fallback_interface is None
        assert spec.agent_factory is not None
        assert spec.llm_factory is not None
        assert spec.parser_factory is not None
