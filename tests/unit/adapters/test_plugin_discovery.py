"""Tests for adapter plugin discovery and registration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from karenina.adapters.registry import AdapterRegistry, AdapterSpec


@pytest.fixture()
def _isolate_registry():
    """Save and restore registry state around a test.

    Use this for tests that need an empty registry (e.g., conflict detection).
    Do NOT use for tests that rely on built-in adapters being registered.
    """
    saved_specs = dict(AdapterRegistry._specs)
    saved_initialized = AdapterRegistry._initialized
    AdapterRegistry._specs = {}
    AdapterRegistry._initialized = False
    yield
    AdapterRegistry._specs = saved_specs
    AdapterRegistry._initialized = saved_initialized


class TestAdapterSpecRequiresProvider:
    """Tests for AdapterSpec.requires_provider field."""

    def test_requires_provider_defaults_to_true(self):
        spec = AdapterSpec(interface="test", description="test")
        assert spec.requires_provider is True

    def test_requires_provider_can_be_set_false(self):
        spec = AdapterSpec(interface="test", description="test", requires_provider=False)
        assert spec.requires_provider is False


class TestLoadBuiltins:
    """Tests for built-in adapter registration.

    These tests use the real initialized registry (no isolation)
    because built-in registration modules are cached in sys.modules
    and cannot be re-executed by re-calling _load_builtins().
    """

    def test_builtins_register_known_interfaces(self):
        """After initialization, all built-in interfaces are registered."""
        interfaces = AdapterRegistry.get_interfaces()
        assert "langchain" in interfaces
        assert "manual" in interfaces
        assert "claude_tool" in interfaces
        assert "claude_agent_sdk" in interfaces
        assert "langchain_deep_agents" in interfaces
        assert "openrouter" in interfaces
        assert "openai_endpoint" in interfaces

    def test_builtins_set_requires_provider_correctly(self):
        """Built-in specs have correct requires_provider values."""
        lc_spec = AdapterRegistry.get_spec("langchain")
        assert lc_spec is not None
        assert lc_spec.requires_provider is True

        manual_spec = AdapterRegistry.get_spec("manual")
        assert manual_spec is not None
        assert manual_spec.requires_provider is False

        claude_tool_spec = AdapterRegistry.get_spec("claude_tool")
        assert claude_tool_spec is not None
        assert claude_tool_spec.requires_provider is False


class TestDiscoverEntryPoints:
    """Tests for _discover_entry_points() method."""

    @pytest.mark.usefixtures("_isolate_registry")
    def test_discover_loads_external_adapter(self):
        """Mock an entry point that registers a fake adapter."""
        fake_spec = AdapterSpec(
            interface="fake_external",
            description="Fake external adapter",
        )

        def fake_load():
            AdapterRegistry.register(fake_spec)

        mock_ep = MagicMock()
        mock_ep.name = "fake_external"
        mock_ep.load = fake_load

        with patch(
            "karenina.adapters.registry.entry_points",
            return_value=[mock_ep],
        ):
            AdapterRegistry._discover_entry_points()

        assert "fake_external" in AdapterRegistry._specs

    @pytest.mark.usefixtures("_isolate_registry")
    def test_discover_skips_builtin_conflict(self):
        """Entry point conflicting with a registered adapter is skipped."""
        builtin_spec = AdapterSpec(interface="langchain", description="Built-in")
        AdapterRegistry.register(builtin_spec)

        mock_ep = MagicMock()
        mock_ep.name = "langchain"

        with patch(
            "karenina.adapters.registry.entry_points",
            return_value=[mock_ep],
        ):
            AdapterRegistry._discover_entry_points()

        mock_ep.load.assert_not_called()

    @pytest.mark.usefixtures("_isolate_registry")
    def test_discover_handles_external_vs_external_conflict(self):
        """Two entry points registering the same interface: second is skipped."""
        spec = AdapterSpec(interface="dupe", description="First")

        call_count = 0

        def load_first():
            nonlocal call_count
            call_count += 1
            AdapterRegistry.register(spec)

        def load_second():
            nonlocal call_count
            call_count += 1
            AdapterRegistry.register(AdapterSpec(interface="dupe", description="Second"))

        ep1 = MagicMock()
        ep1.name = "dupe"
        ep1.load = load_first

        ep2 = MagicMock()
        ep2.name = "dupe2"
        ep2.load = load_second

        with patch(
            "karenina.adapters.registry.entry_points",
            return_value=[ep1, ep2],
        ):
            AdapterRegistry._discover_entry_points()

        assert call_count == 2
        assert AdapterRegistry._specs["dupe"].description == "First"

    @pytest.mark.usefixtures("_isolate_registry")
    def test_discover_handles_load_failure(self):
        """Entry point that raises on load is skipped gracefully."""
        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = ImportError("SDK not found")

        with patch(
            "karenina.adapters.registry.entry_points",
            return_value=[mock_ep],
        ):
            AdapterRegistry._discover_entry_points()

        assert "broken" not in AdapterRegistry._specs

    @pytest.mark.usefixtures("_isolate_registry")
    def test_discover_with_no_entry_points(self):
        """No entry points installed: no error, no new specs."""
        initial_count = len(AdapterRegistry._specs)

        with patch(
            "karenina.adapters.registry.entry_points",
            return_value=[],
        ):
            AdapterRegistry._discover_entry_points()

        assert len(AdapterRegistry._specs) == initial_count


class TestManualRegistration:
    """Tests for manual (runtime) adapter registration."""

    def test_manual_register_and_lookup(self):
        spec = AdapterSpec(
            interface="custom_test_adapter",
            description="Custom adapter",
            requires_provider=False,
        )
        AdapterRegistry.register(spec, force=True)

        result = AdapterRegistry.get_spec("custom_test_adapter")
        assert result is not None
        assert result.interface == "custom_test_adapter"
        assert result.requires_provider is False

    @pytest.mark.usefixtures("_isolate_registry")
    def test_manual_register_blocks_duplicate(self):
        spec = AdapterSpec(interface="custom", description="First")
        AdapterRegistry.register(spec)

        with pytest.raises(ValueError, match="already registered"):
            AdapterRegistry.register(AdapterSpec(interface="custom", description="Second"))

    @pytest.mark.usefixtures("_isolate_registry")
    def test_manual_register_with_force_overwrites(self):
        spec1 = AdapterSpec(interface="custom", description="First")
        AdapterRegistry.register(spec1)

        spec2 = AdapterSpec(interface="custom", description="Second")
        AdapterRegistry.register(spec2, force=True)

        assert AdapterRegistry._specs["custom"].description == "Second"
