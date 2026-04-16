"""Tests for INDEX.md generation."""

from __future__ import annotations

import pytest

from karenina.benchmark.error_analysis.indexer import IndexEntry, build_index_markdown


def _entry(
    *,
    case_id: str,
    bucket_path: str,
    category: str,
    group: str,
    stage: str,
    reason: str,
    outcome: str = "failure",
) -> IndexEntry:
    return IndexEntry(
        case_id=case_id,
        bucket_path=bucket_path,
        outcome=outcome,
        category=category,
        group=group,
        stage=stage,
        reason=reason,
    )


@pytest.mark.unit
class TestIndexBuilder:
    def test_header_has_totals_and_percentage(self):
        entries = [
            _entry(
                case_id="q_001",
                bucket_path="passes/q_001.md",
                category="",
                group="",
                stage="",
                reason="",
                outcome="pass",
            ),
            _entry(
                case_id="q_002",
                bucket_path="failures/content/q_002.md",
                category="content",
                group="content",
                stage="verify_template",
                reason="mismatch on price",
            ),
        ]
        md = build_index_markdown(
            benchmark_name="legal_qa",
            answering_model="anthropic/claude-opus-4-6",
            run_timestamp="2026-04-16T12:00:00Z",
            entries=entries,
        )
        assert "# Error Analysis: legal_qa" in md
        assert "anthropic/claude-opus-4-6" in md
        assert "Total: 2" in md
        assert "Passed: 1" in md
        assert "Failed: 1" in md
        assert "50" in md  # pass rate

    def test_failure_breakdown_table_rows(self):
        entries = [
            _entry(
                case_id="q_a",
                bucket_path="failures/content/q_a.md",
                category="content",
                group="content",
                stage="verify_template",
                reason="r",
            ),
            _entry(
                case_id="q_b",
                bucket_path="failures/parsing/q_b.md",
                category="parsing",
                group="system",
                stage="parse_template",
                reason="r",
            ),
        ]
        md = build_index_markdown(
            benchmark_name="x",
            answering_model="m",
            run_timestamp="t",
            entries=entries,
        )
        assert "| content | 1 |" in md or "| content  | 1 |" in md
        assert "| parsing | 1 |" in md or "| parsing  | 1 |" in md

    def test_failures_by_category_links(self):
        entries = [
            _entry(
                case_id="q_a",
                bucket_path="failures/content/q_a.md",
                category="content",
                group="content",
                stage="verify_template",
                reason="a" * 200,
            ),
        ]
        md = build_index_markdown(
            benchmark_name="x",
            answering_model="m",
            run_timestamp="t",
            entries=entries,
        )
        assert "### content (1)" in md
        assert "[q_a](failures/content/q_a.md)" in md
        # One-line reason capped at 100 chars.
        assert "a" * 100 in md
        assert "a" * 101 not in md

    def test_pass_table_caps_at_50(self):
        entries = [
            _entry(
                case_id=f"q_{i}",
                bucket_path=f"passes/q_{i}.md",
                category="",
                group="",
                stage="",
                reason="",
                outcome="pass",
            )
            for i in range(60)
        ]
        md = build_index_markdown(
            benchmark_name="x",
            answering_model="m",
            run_timestamp="t",
            entries=entries,
        )
        lines = md.splitlines()
        pass_rows = [line for line in lines if line.startswith("| q_")]
        assert 50 <= len(pass_rows) <= 51  # 50 data rows + optional header variations
        assert "10 more" in md
