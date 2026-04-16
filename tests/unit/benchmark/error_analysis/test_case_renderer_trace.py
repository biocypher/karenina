"""Tests for the trace-conversion subsection of case_renderer."""

from __future__ import annotations

import pytest

from karenina.benchmark.error_analysis.case_renderer import render_qa_case

from .fixtures import make_pass


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
