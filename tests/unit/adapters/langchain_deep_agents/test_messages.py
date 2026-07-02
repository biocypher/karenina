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

    def test_from_provider_ai_with_reasoning_content_emits_thinking_block(self):
        """Parity with CSDK: reasoning surfaces as a ``ThinkingContent`` block
        in the unified Message content list, not just in the raw_trace string."""
        from langchain_core.messages import AIMessage

        from karenina.ports import TextContent, ThinkingContent, ToolUseContent

        lc_messages = [
            AIMessage(
                content="The answer is 42.",
                additional_kwargs={"reasoning_content": "Computed via 2+40."},
                tool_calls=[{"name": "submit", "args": {"a": 1}, "id": "call_x"}],
            )
        ]
        result = self.converter.from_provider(lc_messages)
        assert len(result) == 1
        msg = result[0]
        # Block order matches CSDK: thinking, text, tool_use.
        kinds = [type(b).__name__ for b in msg.content]
        assert kinds == ["ThinkingContent", "TextContent", "ToolUseContent"]
        assert isinstance(msg.content[0], ThinkingContent)
        assert msg.content[0].thinking == "Computed via 2+40."
        assert isinstance(msg.content[1], TextContent)
        assert isinstance(msg.content[2], ToolUseContent)

    def test_from_provider_ai_anthropic_style_thinking_block(self):
        """Content-list ``type='thinking'`` should also produce a ThinkingContent."""
        from langchain_core.messages import AIMessage

        from karenina.ports import ThinkingContent

        lc_messages = [
            AIMessage(
                content=[
                    {"type": "thinking", "thinking": "Reasoning step.", "signature": "sig"},
                    {"type": "text", "text": "Answer."},
                ],
            )
        ]
        result = self.converter.from_provider(lc_messages)
        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg.content[0], ThinkingContent)
        assert msg.content[0].thinking == "Reasoning step."
        assert msg.content[0].signature == "sig"

    def test_from_provider_ai_blank_reasoning_is_ignored(self):
        """Whitespace-only reasoning must NOT produce an empty ThinkingContent."""
        from langchain_core.messages import AIMessage

        from karenina.ports import ThinkingContent

        lc_messages = [
            AIMessage(
                content="Answer.",
                additional_kwargs={"reasoning_content": "   "},
            )
        ]
        result = self.converter.from_provider(lc_messages)
        assert len(result) == 1
        kinds = [type(b).__name__ for b in result[0].content]
        assert "ThinkingContent" not in kinds
