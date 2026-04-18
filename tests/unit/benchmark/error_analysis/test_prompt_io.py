"""Tests for prompt resolution and placeholder substitution."""

from __future__ import annotations

import pytest

from karenina.benchmark.error_analysis.prompt_io import (
    PromptContext,
    resolve_and_write_prompt,
)


@pytest.mark.unit
class TestPromptIO:
    def test_default_prompt_written_with_substitutions(self, tmp_path):
        context = PromptContext(
            benchmark_name="legal_qa",
            answering_model="anthropic/claude-opus-4-6",
            total=10,
            passed=7,
            failed=3,
            failure_categories=["content", "parsing"],
            run_timestamp="2026-04-18T12:34:56Z",
        )
        path = resolve_and_write_prompt(prompt_path=None, out_dir=tmp_path, context=context)
        assert path == tmp_path / "PROMPT.md"
        text = path.read_text()
        assert "legal_qa" in text
        assert "anthropic/claude-opus-4-6" in text
        assert "10" in text
        assert "content, parsing" in text
        assert "2026-04-18T12:34:56Z" in text

    def test_user_prompt_copied_verbatim_when_no_placeholders(self, tmp_path):
        source = tmp_path / "custom.md"
        source.write_text("# Custom\nJust do the thing.\n")
        context = PromptContext(
            benchmark_name="x",
            answering_model="m",
            total=0,
            passed=0,
            failed=0,
            failure_categories=[],
        )
        path = resolve_and_write_prompt(prompt_path=source, out_dir=tmp_path, context=context)
        assert path.read_text() == "# Custom\nJust do the thing.\n"

    def test_user_prompt_with_placeholders_substituted(self, tmp_path):
        source = tmp_path / "custom.md"
        source.write_text("Audit $BENCHMARK_NAME on $ANSWERING_MODEL: $FAILED failures.")
        context = PromptContext(
            benchmark_name="bench",
            answering_model="mod",
            total=10,
            passed=7,
            failed=3,
            failure_categories=[],
        )
        path = resolve_and_write_prompt(prompt_path=source, out_dir=tmp_path, context=context)
        assert path.read_text() == "Audit bench on mod: 3 failures."
