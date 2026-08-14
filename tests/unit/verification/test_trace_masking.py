"""Tests for judge-side masking of tool-result bodies in traces."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.utils.trace_masking import (
    MaskStats,
    is_graphql_schema_payload,
    mask_graphql_schema_messages,
    mask_tool_messages,
)

SCHEMA_BODY = (
    "type Query {\n  primaryRecord: A\n  secondaryRecord: A\n}\n"
    "type A {\n  identifier: String\n  relatedRecord: B\n}\n"
    "type B {\n  identifier: String\n  displayName: String\n  description: String\n}"
)
TRACE = (
    "--- AI Message ---\ncalling the tool\n"
    "--- Tool Message (call_id: abc123) ---\n" + SCHEMA_BODY + "\n"
    '--- Tool Message (call_id: def456) ---\n{"status": "success"}\n'
    "--- AI Message ---\nfinal answer\n"
)


@pytest.mark.unit
class TestGraphqlPredicate:
    def test_three_type_declarations_match(self) -> None:
        assert is_graphql_schema_payload(SCHEMA_BODY) is True

    def test_dunder_schema_matches(self) -> None:
        assert is_graphql_schema_payload('{"__schema": {}}') is True

    def test_plain_payload_does_not_match(self) -> None:
        assert is_graphql_schema_payload('{"status": "success"}') is False


@pytest.mark.unit
class TestMaskGraphqlSchemaMessages:
    def test_masks_only_schema_tool_bodies(self) -> None:
        stats = mask_graphql_schema_messages(TRACE)
        assert stats.messages_masked == 1
        assert stats.chars_removed > 0
        assert "type Query" not in stats.text
        assert '{"status": "success"}' in stats.text
        assert "calling the tool" in stats.text
        assert "final answer" in stats.text
        assert "[masked: GraphQL schema introspection response" in stats.text

    def test_no_schema_means_no_change(self) -> None:
        clean = "--- AI Message ---\nhello\n--- Tool Message (call_id: x) ---\nok\n"
        stats = mask_graphql_schema_messages(clean)
        assert stats == MaskStats(text=clean, messages_masked=0, chars_removed=0)

    def test_ai_message_with_schema_text_is_untouched(self) -> None:
        trace = "--- AI Message ---\n" + SCHEMA_BODY + "\n"
        stats = mask_graphql_schema_messages(trace)
        assert stats.messages_masked == 0
        assert stats.text == trace

    def test_preserves_extended_markers_and_unmasked_content(self) -> None:
        trace = (
            "preamble\n--- Human Message (name: user) ---\nquestion\n"
            '--- Tool Message (call_id: keep, name: query) ---\n{"ok": true}\n'
        )
        assert mask_graphql_schema_messages(trace).text == trace


@pytest.mark.unit
class TestMaskToolMessages:
    def test_custom_predicate_and_description(self) -> None:
        stats = mask_tool_messages(
            TRACE,
            should_mask=lambda body: "success" in body,
            describe=lambda _body: "[hidden]",
        )
        assert stats.messages_masked == 1
        assert "[hidden]" in stats.text
        assert "type Query" in stats.text
