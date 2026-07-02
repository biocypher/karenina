"""Tests for CodexMessageConverter conversions per thread item type."""

from __future__ import annotations

from types import SimpleNamespace

from karenina.adapters.codex_sdk.messages import CodexMessageConverter
from karenina.ports import Message, Role, ThinkingContent, ToolResultContent

from .conftest import make_agent_message, make_command_execution, make_item, make_reasoning


class TestInputConversion:
    def test_to_prompt_string_joins_user_messages(self) -> None:
        converter = CodexMessageConverter()
        messages = [
            Message.system("Be helpful"),
            Message.user("Question 1"),
            Message.user("Question 2"),
        ]
        assert converter.to_prompt_string(messages) == "Question 1\n\nQuestion 2"

    def test_extract_system_prompt(self) -> None:
        converter = CodexMessageConverter()
        messages = [Message.system("Be helpful"), Message.system("Be concise"), Message.user("Hi")]
        assert converter.extract_system_prompt(messages) == "Be helpful\n\nBe concise"

    def test_extract_system_prompt_none_when_absent(self) -> None:
        converter = CodexMessageConverter()
        assert converter.extract_system_prompt([Message.user("Hi")]) is None


class TestHistoryTranscript:
    """Multi-turn history serialization for scenario re-invocation."""

    def setup_method(self) -> None:
        self.converter = CodexMessageConverter()

    def test_single_turn_behavior_unchanged(self) -> None:
        messages = [Message.system("Be helpful"), Message.user("Q1"), Message.user("Q2")]
        assert self.converter.to_prompt_string(messages) == "Q1\n\nQ2"

    def test_history_renders_labeled_transcript(self) -> None:
        messages = [
            Message.system("Be helpful"),
            Message.user("Pick a fruit"),
            Message.assistant("I pick mango."),
            Message.user("Which fruit did you pick?"),
        ]
        prompt = self.converter.to_prompt_string(messages)
        assert prompt.startswith("Conversation so far:")
        assert "User: Pick a fruit" in prompt
        assert "Assistant: I pick mango." in prompt
        assert prompt.endswith("Current user message:\n\nWhich fruit did you pick?")
        # System prompt stays out of the transcript.
        assert "Be helpful" not in prompt

    def test_history_order_preserved(self) -> None:
        messages = [
            Message.user("Q1"),
            Message.assistant("A1"),
            Message.user("Q2"),
            Message.assistant("A2"),
            Message.user("Q3"),
        ]
        prompt = self.converter.to_prompt_string(messages)
        assert prompt.index("User: Q1") < prompt.index("Assistant: A1")
        assert prompt.index("Assistant: A1") < prompt.index("User: Q2")
        assert prompt.index("Assistant: A2") < prompt.index("Current user message:")
        assert prompt.rstrip().endswith("Q3")

    def test_tool_activity_summarized(self) -> None:
        from karenina.ports import ToolUseContent

        messages = [
            Message.user("List the files"),
            Message.assistant(
                "",
                tool_calls=[ToolUseContent(id="call_1", name="shell", input={"command": "ls"})],
            ),
            Message.tool_result(tool_use_id="call_1", content="hello.txt"),
            Message.user("What did you see?"),
        ]
        prompt = self.converter.to_prompt_string(messages)
        assert "Assistant ran tool shell with arguments:" in prompt
        assert '"command": "ls"' in prompt
        assert "Tool result (call_1): hello.txt" in prompt

    def test_error_tool_result_labeled(self) -> None:
        messages = [
            Message.user("Run it"),
            Message.tool_result(tool_use_id="call_2", content="boom", is_error=True),
            Message.user("And now?"),
        ]
        prompt = self.converter.to_prompt_string(messages)
        assert "Tool result (error) (call_2): boom" in prompt

    def test_thinking_content_skipped(self) -> None:
        messages = [
            Message.user("Q1"),
            Message(role=Role.ASSISTANT, content=[ThinkingContent(thinking="secret reasoning")]),
            Message.assistant("A1"),
            Message.user("Q2"),
        ]
        prompt = self.converter.to_prompt_string(messages)
        assert "secret reasoning" not in prompt
        assert "Assistant: A1" in prompt

    def test_long_tool_result_truncated(self) -> None:
        long_output = "x" * 5000
        messages = [
            Message.user("Run it"),
            Message.tool_result(tool_use_id="call_3", content=long_output),
            Message.user("Summarize"),
        ]
        prompt = self.converter.to_prompt_string(messages)
        assert "... [truncated, 5000 chars total]" in prompt
        assert "x" * 2001 not in prompt

    def test_history_without_trailing_user_message(self) -> None:
        messages = [Message.user("Q1"), Message.assistant("A1")]
        prompt = self.converter.to_prompt_string(messages)
        assert "Assistant: A1" in prompt
        assert "Current user message:" not in prompt


class TestFromProvider:
    def setup_method(self) -> None:
        self.converter = CodexMessageConverter()

    def test_agent_message_becomes_assistant_text(self) -> None:
        result = self.converter.from_provider([make_agent_message("The answer is 42.")])
        assert len(result) == 1
        assert result[0].role == Role.ASSISTANT
        assert result[0].text == "The answer is 42."

    def test_empty_agent_message_skipped(self) -> None:
        assert self.converter.from_provider([make_agent_message("")]) == []

    def test_reasoning_becomes_thinking_content(self) -> None:
        result = self.converter.from_provider([make_reasoning(content=["step one", "step two"])])
        assert len(result) == 1
        assert result[0].role == Role.ASSISTANT
        thinking = [c for c in result[0].content if isinstance(c, ThinkingContent)]
        assert thinking[0].thinking == "step one\n\nstep two"

    def test_reasoning_falls_back_to_summary(self) -> None:
        item = make_item("reasoning", id="r1", content=[], summary=["summarized thought"])
        result = self.converter.from_provider([item])
        assert isinstance(result[0].content[0], ThinkingContent)
        assert result[0].content[0].thinking == "summarized thought"

    def test_command_execution_becomes_shell_pair(self) -> None:
        item = make_command_execution(item_id="cmd_1", command="echo hi", output="hi\n", exit_code=0)
        result = self.converter.from_provider([item])
        assert len(result) == 2
        use, tool = result
        assert use.role == Role.ASSISTANT
        assert use.tool_calls[0].name == "shell"
        assert use.tool_calls[0].id == "cmd_1"
        assert use.tool_calls[0].input == {"command": "echo hi"}
        assert tool.role == Role.TOOL
        block = tool.content[0]
        assert isinstance(block, ToolResultContent)
        assert block.tool_use_id == "cmd_1"
        assert block.content == "hi\n"
        assert block.is_error is False

    def test_command_execution_nonzero_exit_is_error(self) -> None:
        item = make_command_execution(exit_code=1, output="boom")
        _, tool = self.converter.from_provider([item])
        assert tool.content[0].is_error is True

    def test_command_execution_failed_status_is_error(self) -> None:
        item = make_command_execution(exit_code=None, status="failed")
        _, tool = self.converter.from_provider([item])
        assert tool.content[0].is_error is True

    def test_mcp_tool_call_pair(self) -> None:
        item = make_item(
            "mcpToolCall",
            id="mcp_1",
            server="fetch",
            tool="get_url",
            arguments={"url": "http://x"},
            result=SimpleNamespace(content=["ok"]),
            error=None,
            status="completed",
        )
        use, tool = self.converter.from_provider([item])
        assert use.tool_calls[0].name == "mcp__fetch__get_url"
        assert use.tool_calls[0].input == {"url": "http://x"}
        assert tool.content[0].tool_use_id == "mcp_1"
        assert tool.content[0].is_error is False

    def test_mcp_tool_call_error(self) -> None:
        item = make_item(
            "mcpToolCall",
            id="mcp_2",
            server="fetch",
            tool="get_url",
            arguments=None,
            result=None,
            error=SimpleNamespace(message="connection refused"),
            status="failed",
        )
        use, tool = self.converter.from_provider([item])
        assert tool.content[0].is_error is True
        assert tool.content[0].content == "connection refused"
        assert use.tool_calls[0].input == {"arguments": ""}

    def test_file_change_pair(self) -> None:
        item = make_item(
            "fileChange",
            id="fc_1",
            changes=[SimpleNamespace(path="src/main.py", kind="update", diff="...")],
            status="completed",
        )
        use, tool = self.converter.from_provider([item])
        assert use.tool_calls[0].name == "apply_patch"
        assert use.tool_calls[0].input == {"changes": [{"path": "src/main.py", "kind": "update"}]}
        assert "src/main.py" in tool.content[0].content
        assert tool.content[0].is_error is False

    def test_web_search_pair(self) -> None:
        item = make_item("webSearch", id="ws_1", query="codex sdk docs", action=None)
        use, tool = self.converter.from_provider([item])
        assert use.tool_calls[0].name == "web_search"
        assert use.tool_calls[0].input == {"query": "codex sdk docs"}
        assert tool.content[0].tool_use_id == "ws_1"

    def test_user_message_converted(self) -> None:
        item = make_item(
            "userMessage",
            id="u1",
            content=[SimpleNamespace(text="hello there")],
        )
        result = self.converter.from_provider([item])
        assert result[0].role == Role.USER
        assert result[0].text == "hello there"

    def test_unknown_item_type_skipped(self) -> None:
        assert self.converter.from_provider([make_item("plan", id="p1")]) == []

    def test_root_model_wrapper_unwrapped(self) -> None:
        wrapped = SimpleNamespace(root=make_agent_message("wrapped"))
        result = self.converter.from_provider([wrapped])
        assert result[0].text == "wrapped"
