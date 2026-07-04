"""Tests for rubric_dynamic_presence_check adapter instruction registration."""

from __future__ import annotations

import pytest

import karenina.adapters.claude_agent_sdk.prompts.rubric  # noqa: F401
import karenina.adapters.claude_tool.prompts.rubric  # noqa: F401

# Import registration modules to trigger side-effect adapter instruction registration
import karenina.adapters.langchain.prompts.rubric  # noqa: F401
from karenina.ports.adapter_instruction import AdapterInstructionRegistry


@pytest.mark.unit
class TestPresenceCheckAdapterRegistration:
    """Verify that all adapter interfaces register instructions for the presence check task."""

    def test_langchain_registered(self) -> None:
        factories = AdapterInstructionRegistry.get_instructions("langchain", "rubric_dynamic_presence_check")
        assert len(factories) > 0

    def test_openrouter_registered(self) -> None:
        factories = AdapterInstructionRegistry.get_instructions("openrouter", "rubric_dynamic_presence_check")
        assert len(factories) > 0

    def test_openai_endpoint_registered(self) -> None:
        factories = AdapterInstructionRegistry.get_instructions("openai_endpoint", "rubric_dynamic_presence_check")
        assert len(factories) > 0

    def test_claude_tool_registered(self) -> None:
        factories = AdapterInstructionRegistry.get_instructions("claude_tool", "rubric_dynamic_presence_check")
        assert len(factories) > 0

    def test_claude_agent_sdk_registered(self) -> None:
        factories = AdapterInstructionRegistry.get_instructions("claude_agent_sdk", "rubric_dynamic_presence_check")
        assert len(factories) > 0


@pytest.mark.unit
class TestPresenceCheckAdapterInstructionContent:
    """Verify that factories produce valid AdapterInstruction objects."""

    def test_langchain_factory_produces_instruction(self) -> None:
        factories = AdapterInstructionRegistry.get_instructions("langchain", "rubric_dynamic_presence_check")
        instruction = factories[0](json_schema=None, example_json="", output_format_hint="")
        assert isinstance(instruction.system_addition, str)
        assert isinstance(instruction.user_addition, str)

    def test_claude_tool_factory_produces_instruction(self) -> None:
        factories = AdapterInstructionRegistry.get_instructions("claude_tool", "rubric_dynamic_presence_check")
        instruction = factories[0]()
        assert isinstance(instruction.system_addition, str)
        assert isinstance(instruction.user_addition, str)

    def test_claude_agent_sdk_factory_produces_instruction(self) -> None:
        factories = AdapterInstructionRegistry.get_instructions("claude_agent_sdk", "rubric_dynamic_presence_check")
        instruction = factories[0]()
        assert isinstance(instruction.system_addition, str)
        assert isinstance(instruction.user_addition, str)
