"""Conformance tests for LLMPort implementations.

Validates that LLMPort adapters correctly satisfy the protocol,
support structured output, and declare capabilities properly.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel, Field

from karenina.ports import LLMPort
from karenina.ports.capabilities import PortCapabilities


@pytest.mark.unit
class TestLLMPortConformance:
    """Verify LLMPort protocol compliance for Deep Agents adapter."""

    def test_isinstance_check(self, deep_agents_llm_adapter):
        """Adapter must pass runtime_checkable isinstance."""
        assert isinstance(deep_agents_llm_adapter, LLMPort)

    def test_ainvoke_signature(self, deep_agents_llm_adapter):
        """ainvoke() must accept messages parameter."""
        sig = inspect.signature(deep_agents_llm_adapter.ainvoke)
        params = list(sig.parameters.keys())
        assert "messages" in params

    def test_invoke_signature(self, deep_agents_llm_adapter):
        """invoke() must accept messages parameter."""
        sig = inspect.signature(deep_agents_llm_adapter.invoke)
        params = list(sig.parameters.keys())
        assert "messages" in params

    def test_capabilities_returns_port_capabilities(self, deep_agents_llm_adapter):
        """capabilities property must return PortCapabilities."""
        caps = deep_agents_llm_adapter.capabilities
        assert isinstance(caps, PortCapabilities)
        assert isinstance(caps.supports_system_prompt, bool)
        assert isinstance(caps.supports_structured_output, bool)

    def test_with_structured_output_returns_llm_port(self, deep_agents_llm_adapter):
        """with_structured_output() must return a new LLMPort instance."""

        class TestSchema(BaseModel):
            value: str = Field(description="test value")

        structured = deep_agents_llm_adapter.with_structured_output(TestSchema)
        assert isinstance(structured, LLMPort)

    def test_with_structured_output_preserves_original(self, deep_agents_llm_adapter):
        """with_structured_output() must not mutate the original adapter."""

        class TestSchema(BaseModel):
            value: str = Field(description="test value")

        original_schema = getattr(deep_agents_llm_adapter, "_structured_schema", None)
        deep_agents_llm_adapter.with_structured_output(TestSchema)
        assert getattr(deep_agents_llm_adapter, "_structured_schema", None) == original_schema
