"""Offline tests for the typed QA result loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paper.common.qa_results import (
    GROUP_TO_OUTCOME,
    QAResultRow,
    iter_rows,
    load_benchmark_text,
    load_reference_answers,
)


def _row_dict(question_id: str = "q1", failure_group: str | None = None) -> dict[str, Any]:
    failure = None
    if failure_group is not None:
        category = {
            "content": "content",
            "abstained": "abstention",
        }.get(failure_group, failure_group)
        failure = {
            "category": category,
            "stage": "verify_template",
            "reason": "verification failed",
            "details": None,
            "group": failure_group,
        }
    return {
        "metadata": {
            "question_id": question_id,
            "template_id": "template-hash",
            "question_text": "What is X?",
            "answering": {
                "interface": "langchain",
                "model_name": "answerer-a",
                "tools": ["otp"],
                "config_id": None,
            },
            "parsing": {
                "interface": "langchain",
                "model_name": "claude-opus-4-6",
                "tools": [],
                "config_id": None,
            },
            "replicate": 1,
            "run_name": "qa-evaluation",
            "result_id": "abc123",
            "execution_time": "1.0",
            "timestamp": "2026-08-12T00:00:00+00:00",
            "failure": failure,
        },
        "template": {
            "raw_llm_response": "--- AI Message ---\nanswer",
            "trace_messages": [{"role": "assistant", "content": "answer"}],
            "verify_result": True,
            "usage_metadata": {"answer_generation": {"total_tokens": 1234}},
        },
    }


@pytest.mark.unit
class TestIterRows:
    def test_yields_typed_rows(self, tmp_path: Path) -> None:
        source = tmp_path / "mini.json"
        source.write_text(json.dumps({"format_version": "2.2", "results": [_row_dict()]}))
        rows = list(iter_rows(source, regime="mcp"))
        assert rows == [
            QAResultRow(
                question_id="q1",
                answerer="answerer-a",
                answering_tools=("otp",),
                parser="claude-opus-4-6",
                regime="mcp",
                replicate=1,
                run_name="qa-evaluation",
                result_id="abc123",
                outcome="pass",
                raw_trace="--- AI Message ---\nanswer",
                trace_message_count=1,
                tokens_answerer=1234,
                verify_result=True,
            )
        ]

    def test_failure_group_maps_to_outcome(self, tmp_path: Path) -> None:
        source = tmp_path / "mini.json"
        source.write_text(json.dumps({"format_version": "2.2", "results": [_row_dict(failure_group="abstained")]}))
        assert next(iter_rows(source, regime="mcp")).outcome == "abstain"
        assert GROUP_TO_OUTCOME["content"] == "fail-content"

    def test_unknown_failure_group_raises(self, tmp_path: Path) -> None:
        source = tmp_path / "mini.json"
        source.write_text(json.dumps({"format_version": "2.2", "results": [_row_dict(failure_group="mystery")]}))
        with pytest.raises(ValueError, match="mystery"):
            list(iter_rows(source, regime="mcp"))


@pytest.mark.unit
class TestBenchmarkLookup:
    def test_question_and_reference_lookup_by_id(self, tmp_path: Path) -> None:
        jsonld = {
            "dataFeedElement": [
                {
                    "@id": "q1",
                    "item": {
                        "text": "What is X?",
                        "acceptedAnswer": {"text": "the reference"},
                    },
                }
            ]
        }
        path = tmp_path / "bench.jsonld"
        path.write_text(json.dumps(jsonld))
        assert load_reference_answers(path) == {"q1": "the reference"}
        assert load_benchmark_text(path) == {"q1": ("What is X?", "the reference")}
