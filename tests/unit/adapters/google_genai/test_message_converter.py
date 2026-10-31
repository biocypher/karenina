"""Tests for the google_genai message converter."""

from __future__ import annotations

import base64

import pytest
from google.genai import types

from karenina.adapters.google_genai.messages import (
    decode_signature,
    encode_signature,
    extract_system_instruction,
    from_contents,
    model_parts_to_message,
    to_contents,
)
from karenina.ports import Message, Role, TextContent, ThinkingContent, ToolResultContent, ToolUseContent
from tests.unit.adapters.google_genai._fakes import call_part, text_part, thought_part

SIGNATURE = b"\x00\x01opaque-signature\xff"
SIGNATURE_B64 = base64.b64encode(SIGNATURE).decode("ascii")


@pytest.mark.unit
class TestSystemInstruction:
    def test_system_messages_are_joined(self) -> None:
        messages = [Message.system("One."), Message.user("Hi"), Message.system("Two.")]
        assert extract_system_instruction(messages) == "One.\n\nTwo."

    def test_no_system_message(self) -> None:
        assert extract_system_instruction([Message.user("Hi")]) is None


@pytest.mark.unit
class TestToContents:
    def test_user_assistant_and_tool_mapping(self) -> None:
        messages = [
            Message.system("sys"),
            Message.user("Question?"),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ThinkingContent(thinking="Let me check.", signature=SIGNATURE_B64),
                    TextContent(text="Checking."),
                    ToolUseContent(id="c1", name="search docs", input={"q": "x"}),
                    ToolUseContent(id="c2", name="fetch", input={}),
                ],
            ),
            Message.tool_result("c1", "found"),
            Message.tool_result("c2", "boom", is_error=True),
            Message.user("Thanks"),
        ]
        contents = to_contents(messages)

        assert [c.role for c in contents] == ["user", "model", "user", "user"]
        thought, text, call1, call2 = contents[1].parts
        assert thought.thought is True and thought.text == "Let me check."
        assert thought.thought_signature == SIGNATURE
        assert text.text == "Checking."
        assert call1.function_call.name == "search_docs"  # sanitised
        assert call1.function_call.id == "c1"
        assert call1.function_call.args == {"q": "x"}
        assert call2.function_call.name == "fetch"

        # Both tool results travel in one user Content, named after their call.
        resp1, resp2 = contents[2].parts
        assert resp1.function_response.name == "search_docs"
        assert resp1.function_response.id == "c1"
        assert resp1.function_response.response == {"result": "found"}
        assert resp2.function_response.response == {"error": "boom"}
        assert contents[3].parts[0].text == "Thanks"

    def test_synthetic_server_side_pairs_are_dropped(self) -> None:
        messages = [
            Message.user("Q"),
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseContent(id="google_search_1", name="google_search", input={"queries": ["a"]})],
            ),
            Message.tool_result("google_search_1", '{"server_side": true}'),
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseContent(id="url_context_1", name="url_context", input={"urls": ["u"]})],
            ),
            Message.tool_result("url_context_1", '{"server_side": true}'),
            Message.assistant("Answer."),
        ]
        contents = to_contents(messages)
        assert [c.role for c in contents] == ["user", "model"]
        assert contents[1].parts[0].text == "Answer."

    def test_invalid_signature_is_dropped(self) -> None:
        assert decode_signature("not base64!!") is None
        assert decode_signature(None) is None
        assert encode_signature(None) is None


@pytest.mark.unit
class TestRoundTrip:
    def test_messages_survive_a_round_trip(self) -> None:
        original = [
            Message.user("Question?"),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ThinkingContent(thinking="Plan.", signature=SIGNATURE_B64),
                    TextContent(text="Looking."),
                    ToolUseContent(id="c1", name="lookup", input={"id": 1}),
                ],
            ),
            Message.tool_result("c1", "row 1"),
            Message.assistant("Done."),
        ]
        restored = from_contents(to_contents(original))

        assert [m.role for m in restored] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
        assert restored[0].text == "Question?"
        thinking = restored[1].content[0]
        assert isinstance(thinking, ThinkingContent)
        assert thinking.thinking == "Plan."
        assert thinking.signature == SIGNATURE_B64  # base64 signature round-trips exactly
        assert restored[1].text == "Looking."
        assert restored[1].tool_calls == [ToolUseContent(id="c1", name="lookup", input={"id": 1})]
        result = restored[2].content[0]
        assert isinstance(result, ToolResultContent)
        assert (result.tool_use_id, result.content, result.is_error) == ("c1", "row 1", False)
        assert restored[3].text == "Done."

    def test_response_without_id_matches_the_open_call_by_name(self) -> None:
        contents = [
            types.Content(role="model", parts=[call_part("lookup", {"id": 1})]),
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(name="lookup", response={"result": "ok"})],
            ),
        ]
        restored = from_contents(contents)
        assert restored[0].tool_calls[0].id == "call_1_0"
        assert restored[1].content[0].tool_use_id == "call_1_0"


@pytest.mark.unit
class TestModelPartsToMessage:
    def test_fragments_are_joined_without_separators(self) -> None:
        parts = [
            thought_part("Thinking "),
            thought_part("more.", signature=SIGNATURE),
            text_part("The ans"),
            text_part("wer."),
            text_part("", signature=b"trailing"),
        ]
        message, calls = model_parts_to_message(parts, turn=1, usage_metadata={"input_tokens": 3})
        assert message is not None
        assert calls == []
        thinking, text = message.content
        assert isinstance(thinking, ThinkingContent)
        assert thinking.thinking == "Thinking more."
        assert thinking.signature == SIGNATURE_B64
        assert text == TextContent(text="The answer.")
        assert message.usage_metadata == {"input_tokens": 3}

    def test_function_calls_get_ids_and_original_names(self) -> None:
        parts = [call_part("search_docs", {"q": "a"}), call_part("fetch", {"u": 1}, call_id="api-7")]
        original_names = {"search_docs": "search docs"}
        message, calls = model_parts_to_message(parts, turn=3, name_for=lambda name: original_names.get(name, name))
        assert [c.tool_use_id for c in calls] == ["call_3_0", "api-7"]
        assert [c.api_id for c in calls] == [None, "api-7"]
        assert [c.name for c in calls] == ["search_docs", "fetch"]
        assert message is not None
        # The trace shows the tool's real name.
        assert [t.name for t in message.tool_calls] == ["search docs", "fetch"]

    def test_empty_turn_has_no_message(self) -> None:
        message, calls = model_parts_to_message([text_part("", signature=b"s")], turn=1)
        assert message is None
        assert calls == []
