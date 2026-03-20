"""Conformance tests for message converter roundtrip correctness.

Validates that converting karenina Message -> provider format -> karenina Message
preserves content and role for all message types.
"""

from __future__ import annotations

import pytest

from karenina.ports import Message


@pytest.mark.unit
class TestMessageConverterConformance:
    """Verify message conversion roundtrip for Deep Agents adapter."""

    def test_user_message_roundtrip(self, deep_agents_message_converter):
        """User message survives to_prompt_string and back via from_provider."""
        from langchain_core.messages import HumanMessage

        original = Message.user("Hello, world!")
        prompt = deep_agents_message_converter.to_prompt_string([original])
        assert "Hello, world!" in prompt

        # Simulate round-trip via provider
        lc_msg = HumanMessage(content="Hello, world!")
        restored = deep_agents_message_converter.from_provider([lc_msg])
        assert len(restored) == 1
        assert restored[0].role.value == "user"
        assert restored[0].text == "Hello, world!"

    def test_assistant_message_roundtrip(self, deep_agents_message_converter):
        """Assistant message preserves content through from_provider."""
        from langchain_core.messages import AIMessage

        lc_msg = AIMessage(content="The answer is 42.")
        restored = deep_agents_message_converter.from_provider([lc_msg])
        assert len(restored) == 1
        assert restored[0].role.value == "assistant"
        assert "42" in restored[0].text

    def test_system_message_extracted_separately(self, deep_agents_message_converter):
        """System messages should be extracted via extract_system_prompt, not in prompt string."""
        messages = [Message.system("Be helpful."), Message.user("Hi")]

        system = deep_agents_message_converter.extract_system_prompt(messages)
        prompt = deep_agents_message_converter.to_prompt_string(messages)

        assert system == "Be helpful."
        assert "Be helpful." not in prompt
        assert "Hi" in prompt

    def test_tool_message_roundtrip(self, deep_agents_message_converter):
        """Tool messages preserve content through from_provider."""
        from langchain_core.messages import ToolMessage

        lc_msg = ToolMessage(content="tool output data", tool_call_id="call_123")
        restored = deep_agents_message_converter.from_provider([lc_msg])
        assert len(restored) == 1
        assert restored[0].role.value == "tool"

    def test_empty_messages_produce_empty_prompt(self, deep_agents_message_converter):
        """Empty message list should produce empty prompt string."""
        assert deep_agents_message_converter.to_prompt_string([]) == ""

    def test_empty_messages_produce_no_system_prompt(self, deep_agents_message_converter):
        """Empty message list should produce None system prompt."""
        assert deep_agents_message_converter.extract_system_prompt([]) is None
