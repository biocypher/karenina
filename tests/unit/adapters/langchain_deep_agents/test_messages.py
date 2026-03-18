"""Tests for DeepAgentsMessageConverter."""

from __future__ import annotations

import pytest

from karenina.adapters.langchain_deep_agents.messages import DeepAgentsMessageConverter
from karenina.ports.messages import Message


@pytest.mark.unit
class TestDeepAgentsMessageConverter:
    def setup_method(self):
        self.converter = DeepAgentsMessageConverter()

    def test_user_message_to_prompt_string(self):
        messages = [Message.user("What is 2 + 2?")]
        result = self.converter.to_prompt_string(messages)
        assert "What is 2 + 2?" in result

    def test_system_message_extracted_separately(self):
        messages = [Message.system("You are helpful."), Message.user("Hello")]
        system = self.converter.extract_system_prompt(messages)
        prompt = self.converter.to_prompt_string(messages)
        assert system == "You are helpful."
        assert "You are helpful." not in prompt
        assert "Hello" in prompt

    def test_multiple_user_messages_concatenated(self):
        messages = [Message.user("First"), Message.user("Second")]
        result = self.converter.to_prompt_string(messages)
        assert "First" in result
        assert "Second" in result

    def test_no_system_message_returns_none(self):
        messages = [Message.user("Hello")]
        assert self.converter.extract_system_prompt(messages) is None

    def test_from_provider_human_message(self):
        from langchain_core.messages import HumanMessage

        lc_messages = [HumanMessage(content="Hello")]
        result = self.converter.from_provider(lc_messages)
        assert len(result) == 1
        assert result[0].role.value == "user"

    def test_from_provider_ai_message(self):
        from langchain_core.messages import AIMessage

        lc_messages = [AIMessage(content="The answer is 42.")]
        result = self.converter.from_provider(lc_messages)
        assert len(result) == 1
        assert result[0].role.value == "assistant"

    def test_from_provider_tool_message(self):
        from langchain_core.messages import ToolMessage

        lc_messages = [ToolMessage(content="result data", tool_call_id="call_1")]
        result = self.converter.from_provider(lc_messages)
        assert len(result) == 1
        assert result[0].role.value == "tool"

    def test_from_provider_ai_with_tool_calls(self):
        from langchain_core.messages import AIMessage

        lc_messages = [
            AIMessage(
                content="Let me search.",
                tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "call_1"}],
            )
        ]
        result = self.converter.from_provider(lc_messages)
        assert len(result) == 1
        assert result[0].role.value == "assistant"
