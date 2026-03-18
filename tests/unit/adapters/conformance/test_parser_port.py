"""Conformance tests for ParserPort implementations.

Validates that ParserPort adapters correctly satisfy the protocol
and declare capabilities properly.
"""

from __future__ import annotations

import inspect

import pytest

from karenina.ports import ParserPort
from karenina.ports.capabilities import PortCapabilities


@pytest.mark.unit
class TestParserPortConformance:
    """Verify ParserPort protocol compliance for Deep Agents adapter."""

    def test_isinstance_check(self, deep_agents_parser_adapter):
        """Adapter must pass runtime_checkable isinstance."""
        assert isinstance(deep_agents_parser_adapter, ParserPort)

    def test_aparse_to_pydantic_signature(self, deep_agents_parser_adapter):
        """aparse_to_pydantic() must accept messages and schema parameters."""
        sig = inspect.signature(deep_agents_parser_adapter.aparse_to_pydantic)
        params = list(sig.parameters.keys())
        assert "messages" in params
        assert "schema" in params

    def test_parse_to_pydantic_signature(self, deep_agents_parser_adapter):
        """parse_to_pydantic() must accept messages and schema parameters."""
        sig = inspect.signature(deep_agents_parser_adapter.parse_to_pydantic)
        params = list(sig.parameters.keys())
        assert "messages" in params
        assert "schema" in params

    def test_capabilities_returns_port_capabilities(self, deep_agents_parser_adapter):
        """capabilities property must return PortCapabilities."""
        caps = deep_agents_parser_adapter.capabilities
        assert isinstance(caps, PortCapabilities)
        assert isinstance(caps.supports_system_prompt, bool)
        assert isinstance(caps.supports_structured_output, bool)
