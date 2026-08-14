from __future__ import annotations

from karenina.adapters.omp.messages import build_omp_prompt, extract_final_response, omp_messages_to_raw_trace
from karenina.ports import Message, ToolUseContent


def test_prompt_preserves_system_history_and_tool_context() -> None:
    prompt = build_omp_prompt(
        [
            Message.system("Be exact."),
            Message.user("Inspect the file."),
            Message.assistant(tool_calls=[ToolUseContent(id="call-1", name="read", input={"path": "x"})]),
            Message.tool_result("call-1", "contents"),
            Message.user("Summarize it."),
        ],
        system_prompt="Return one sentence.",
    )

    assert "System instructions:\nBe exact.\n\nReturn one sentence." in prompt
    assert "[Tool call read (call-1)" in prompt
    assert "Tool result (call-1):\ncontents" in prompt
    assert prompt.endswith("User:\nSummarize it.")


def test_trace_and_final_response_use_unified_messages() -> None:
    messages = [Message.assistant("first"), Message.assistant("final")]

    assert extract_final_response(messages) == "final"
    assert "--- AI Message ---\nfinal" in omp_messages_to_raw_trace(messages)
