"""Tests for abstention/sufficiency adapter instruction registration."""

from __future__ import annotations

import pytest

# Import registration modules to trigger side-effect registration
import karenina.adapters.claude_agent_sdk.prompts.abstention  # noqa: F401
import karenina.adapters.claude_agent_sdk.prompts.sufficiency  # noqa: F401
import karenina.adapters.claude_tool.prompts.abstention  # noqa: F401
import karenina.adapters.claude_tool.prompts.sufficiency  # noqa: F401
import karenina.adapters.langchain.prompts.abstention  # noqa: F401
import karenina.adapters.langchain.prompts.sufficiency  # noqa: F401
import karenina.adapters.langchain_deep_agents.prompts.abstention  # noqa: F401
import karenina.adapters.langchain_deep_agents.prompts.sufficiency  # noqa: F401
from karenina.ports.adapter_instruction import AdapterInstructionRegistry

INTERFACES = [
    "langchain",
    "openrouter",
    "openai_endpoint",
    "claude_tool",
    "claude_agent_sdk",
    "langchain_deep_agents",
]


@pytest.mark.unit
class TestAbstentionAdapterRegistration:
    """All adapter interfaces should register instructions for abstention_detection."""

    @pytest.mark.parametrize("interface", INTERFACES)
    def test_registered(self, interface: str) -> None:
        factories = AdapterInstructionRegistry.get_instructions(interface, "abstention_detection")
        assert len(factories) > 0, f"No abstention_detection instruction for {interface}"

    @pytest.mark.parametrize("interface", INTERFACES)
    def test_factory_produces_valid_instruction(self, interface: str) -> None:
        factories = AdapterInstructionRegistry.get_instructions(interface, "abstention_detection")
        instruction = factories[0]()
        assert isinstance(instruction.system_addition, str)
        assert isinstance(instruction.user_addition, str)


@pytest.mark.unit
class TestSufficiencyAdapterRegistration:
    """All adapter interfaces should register instructions for sufficiency_detection."""

    @pytest.mark.parametrize("interface", INTERFACES)
    def test_registered(self, interface: str) -> None:
        factories = AdapterInstructionRegistry.get_instructions(interface, "sufficiency_detection")
        assert len(factories) > 0, f"No sufficiency_detection instruction for {interface}"

    @pytest.mark.parametrize("interface", INTERFACES)
    def test_factory_produces_valid_instruction(self, interface: str) -> None:
        factories = AdapterInstructionRegistry.get_instructions(interface, "sufficiency_detection")
        instruction = factories[0]()
        assert isinstance(instruction.system_addition, str)
        assert isinstance(instruction.user_addition, str)
