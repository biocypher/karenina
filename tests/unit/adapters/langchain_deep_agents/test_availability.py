"""Tests for Deep Agents availability checking."""

from __future__ import annotations

import pytest

from karenina.adapters.langchain_deep_agents.availability import check_deep_agents_available


@pytest.mark.unit
class TestDeepAgentsAvailability:
    def test_returns_availability_result(self):
        result = check_deep_agents_available()
        assert hasattr(result, "available")
        assert hasattr(result, "reason")
        assert isinstance(result.available, bool)
        assert isinstance(result.reason, str)

    def test_unavailable_when_import_fails(self, monkeypatch):
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "deepagents":
                raise ImportError("No module named 'deepagents'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = check_deep_agents_available()
        assert result.available is False
        assert "deepagents" in result.reason.lower()
        assert result.fallback_interface is None
