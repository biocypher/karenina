"""Tests for the trace-conversion subsection of case_renderer."""

from __future__ import annotations

import pytest

from karenina.benchmark.error_analysis.case_renderer import render_qa_case

from .fixtures import make_pass, make_template


def _minimal_trace_message(role: str, text: str) -> dict:
    # Matches the format produced by Message.to_dict() and consumed by
    # Message.from_dict(): a flat dict whose ``content`` is a plain
    # string. Anthropic-style content-block lists are intentionally not
    # supported here (see ports/messages.py).
    return {
        "role": role,
        "content": text,
    }


@pytest.mark.unit
class TestCaseRendererTrace:
    def test_no_trace_messages_emits_stub(self):
        result = make_pass()
        body = render_qa_case(result, template_source=None, artifacts_dir=None)
        assert "# Trace" in body
        assert "_No trace captured for this case._" in body

    def test_trace_with_messages_emits_xml_turns(self, tmp_path):
        result = make_pass()
        result.template.trace_messages = [
            _minimal_trace_message("user", "please answer 2+2"),
            _minimal_trace_message("assistant", "4"),
        ]
        body = render_qa_case(
            result,
            template_source=None,
            artifacts_dir=tmp_path / "artifacts",
        )
        assert "<turn" in body
        assert "4" in body
        assert "_No trace captured" not in body

    def test_trace_offloads_long_messages(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KARENINA_TRACE_TRUNCATION_THRESHOLD", "50")
        long_text = "X" * 500
        result = make_pass()
        result.template.trace_messages = [
            _minimal_trace_message("user", "short prompt"),
            _minimal_trace_message("assistant", long_text),
        ]
        body = render_qa_case(
            result,
            template_source=None,
            artifacts_dir=tmp_path / "artifacts",
        )
        assert "[Content offloaded:" in body
        assert 'offloaded="true"' in body
        # Pointer names the offloaded file.
        assert "text_" in body
        # Artifact file actually exists.
        artifacts = list((tmp_path / "artifacts").glob("text_*.txt"))
        assert artifacts, "expected at least one offloaded artifact"
        assert artifacts[0].read_text() == long_text

    def test_qa_trace_without_user_role_renders_xml_turn(self, tmp_path):
        """QA trace_messages with only assistant/tool roles still materialize a turn."""
        result = make_pass()
        result.template.trace_messages = [
            _minimal_trace_message("assistant", "I will search."),
            _minimal_trace_message("assistant", "lookup(q='slc5a1')"),
            _minimal_trace_message("tool", "no hits"),
            _minimal_trace_message("assistant", "final answer"),
        ]
        body = render_qa_case(
            result,
            template_source=None,
            artifacts_dir=tmp_path / "artifacts",
        )
        assert "# Trace" in body
        assert '<turn number="1">' in body
        assert "<assistant" in body
        assert "I will search." in body
        assert "final answer" in body
        assert "_No trace captured" not in body


@pytest.mark.unit
class TestLLMResponseSectionPolicy:
    """When trace_messages is populated, # LLM Response must be dropped."""

    def test_drops_llm_response_when_trace_messages_present(self, tmp_path):
        result = make_pass()
        # Attach a populated trace and a separate raw_llm_response.
        result.template.trace_messages = [
            _minimal_trace_message("assistant", "structured trace body"),
        ]
        result.template.raw_llm_response = "RAW DUMP THAT SHOULD NOT APPEAR"
        body = render_qa_case(
            result,
            template_source=None,
            artifacts_dir=tmp_path / "artifacts",
        )
        assert "# Trace" in body
        assert "# LLM Response" not in body
        assert "RAW DUMP THAT SHOULD NOT APPEAR" not in body

    def test_keeps_llm_response_when_trace_messages_empty(self, tmp_path):
        result = make_pass()
        result.template = make_template(
            raw_response="raw-only response",
            trace_messages=[],
            verify_result=True,
        )
        body = render_qa_case(
            result,
            template_source=None,
            artifacts_dir=tmp_path / "artifacts",
        )
        assert "# LLM Response" in body
        assert "raw-only response" in body
