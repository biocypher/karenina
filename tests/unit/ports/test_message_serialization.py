"""Tests for Message.to_dict() and Message.from_dict() serialization round-trip."""

from karenina.ports.messages import (
    Message,
    Role,
    TextContent,
    ThinkingContent,
    ToolResultContent,
    ToolUseContent,
)


class TestMessageToDict:
    """Tests for Message.to_dict()."""

    def test_simple_user_message(self):
        msg = Message.user("Hello, world!")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello, world!"
        assert d["block_index"] == 0
        assert "tool_calls" not in d
        assert "tool_result" not in d
        assert "thinking" not in d

    def test_simple_assistant_message(self):
        msg = Message.assistant("The answer is 42.")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "The answer is 42."

    def test_system_message(self):
        msg = Message.system("You are a helpful assistant.")
        d = msg.to_dict()
        assert d["role"] == "system"
        assert d["content"] == "You are a helpful assistant."

    def test_assistant_with_tool_calls(self):
        tool_call = ToolUseContent(id="tc_1", name="search", input={"q": "test"})
        msg = Message.assistant("Let me search.", tool_calls=[tool_call])
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Let me search."
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["id"] == "tc_1"
        assert d["tool_calls"][0]["name"] == "search"
        assert d["tool_calls"][0]["input"] == {"q": "test"}

    def test_tool_result_message(self):
        msg = Message.tool_result("tc_1", "Results found", is_error=False)
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert d["content"] == "Results found"
        assert d["tool_result"]["tool_use_id"] == "tc_1"
        assert d["tool_result"]["is_error"] is False

    def test_tool_result_error(self):
        msg = Message.tool_result("tc_1", "Connection failed", is_error=True)
        d = msg.to_dict()
        assert d["tool_result"]["is_error"] is True

    def test_message_with_thinking(self):
        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ThinkingContent(thinking="Let me reason...", signature="sig123"),
                TextContent(text="The answer is 42."),
            ],
        )
        d = msg.to_dict()
        assert d["thinking"]["thinking"] == "Let me reason..."
        assert d["thinking"]["signature"] == "sig123"
        assert d["content"] == "The answer is 42."


class TestMessageFromDict:
    """Tests for Message.from_dict()."""

    def test_simple_user_message(self):
        d = {"role": "user", "content": "Hello!", "block_index": 0}
        msg = Message.from_dict(d)
        assert msg.role == Role.USER
        assert msg.text == "Hello!"

    def test_simple_assistant_message(self):
        d = {"role": "assistant", "content": "Hi there!", "block_index": 0}
        msg = Message.from_dict(d)
        assert msg.role == Role.ASSISTANT
        assert msg.text == "Hi there!"

    def test_assistant_with_tool_calls(self):
        d = {
            "role": "assistant",
            "content": "Searching...",
            "block_index": 0,
            "tool_calls": [{"id": "tc_1", "name": "search", "input": {"q": "test"}}],
        }
        msg = Message.from_dict(d)
        assert msg.role == Role.ASSISTANT
        assert msg.text == "Searching..."
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_tool_result_from_dict(self):
        d = {
            "role": "tool",
            "content": "Found results",
            "block_index": 1,
            "tool_result": {"tool_use_id": "tc_1", "is_error": False},
        }
        msg = Message.from_dict(d)
        assert msg.role == Role.TOOL
        tool_results = [c for c in msg.content if isinstance(c, ToolResultContent)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_use_id == "tc_1"

    def test_thinking_from_dict(self):
        d = {
            "role": "assistant",
            "content": "Answer is 42.",
            "block_index": 0,
            "thinking": {"thinking": "Reasoning...", "signature": "sig"},
        }
        msg = Message.from_dict(d)
        thinking = [c for c in msg.content if isinstance(c, ThinkingContent)]
        assert len(thinking) == 1
        assert thinking[0].thinking == "Reasoning..."
        assert thinking[0].signature == "sig"


class TestRoundTrip:
    """Test to_dict/from_dict round-trip preserves semantics."""

    def test_user_round_trip(self):
        original = Message.user("Test message")
        restored = Message.from_dict(original.to_dict())
        assert restored.role == original.role
        assert restored.text == original.text

    def test_assistant_with_tools_round_trip(self):
        tool_call = ToolUseContent(id="tc_1", name="search", input={"q": "x"})
        original = Message.assistant("Searching...", tool_calls=[tool_call])
        restored = Message.from_dict(original.to_dict())
        assert restored.role == original.role
        assert restored.text == original.text
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].name == "search"
        assert restored.tool_calls[0].input == {"q": "x"}

    def test_tool_result_round_trip(self):
        original = Message.tool_result("tc_1", "results", is_error=True)
        restored = Message.from_dict(original.to_dict())
        assert restored.role == original.role
        tool_results = [c for c in restored.content if isinstance(c, ToolResultContent)]
        assert tool_results[0].is_error is True
