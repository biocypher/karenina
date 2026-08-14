"""Tests for streaming rows from results JSON files."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from karenina.benchmark.core.results_stream import iter_results_from_json
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity


def _result() -> VerificationResult:
    answering = ModelIdentity(interface="openai_endpoint", model_name="answerer")
    parsing = ModelIdentity(interface="openai_endpoint", model_name="judge")
    timestamp = datetime.now(UTC).isoformat()
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q1",
            template_id="template_hash",
            question_text="What is 2+2?",
            answering=answering,
            parsing=parsing,
            execution_time=0.5,
            timestamp=timestamp,
            result_id=VerificationResultMetadata.compute_result_id(
                question_id="q1",
                answering=answering,
                parsing=parsing,
                timestamp=timestamp,
            ),
            run_name="test_run",
        ),
        template=VerificationResultTemplate(raw_llm_response="four"),
    )


def _write_v22(path: Path, rows: list[dict[str, object]]) -> None:
    payload = {
        "format_version": "2.2",
        "metadata": {"job_id": "test"},
        "shared_data": {},
        "results": rows,
    }
    path.write_text(json.dumps(payload))


@pytest.mark.unit
class TestIterResultsFromJson:
    def test_streams_v22_rows_raw(self, tmp_path: Path) -> None:
        rows = [_result().model_dump(mode="json") for _ in range(3)]
        source = tmp_path / "results.json"
        _write_v22(source, rows)
        streamed = list(iter_results_from_json(source, raw=True))
        assert len(streamed) == 3
        assert streamed[0]["metadata"]["question_id"] == rows[0]["metadata"]["question_id"]

    def test_streams_v22_rows_validated(self, tmp_path: Path) -> None:
        rows = [_result().model_dump(mode="json")]
        source = tmp_path / "results.json"
        _write_v22(source, rows)
        streamed = list(iter_results_from_json(source))
        assert streamed[0].metadata.question_id == rows[0]["metadata"]["question_id"]

    def test_streams_legacy_array_and_drops_row_index(self, tmp_path: Path) -> None:
        row = _result().model_dump(mode="json")
        row["row_index"] = 1
        source = tmp_path / "legacy.json"
        source.write_text(json.dumps([row]))
        streamed = list(iter_results_from_json(source))
        assert streamed[0].metadata.question_id == row["metadata"]["question_id"]

    def test_ignores_leading_whitespace(self, tmp_path: Path) -> None:
        row = _result().model_dump(mode="json")
        source = tmp_path / "whitespace.json"
        source.write_text("\n  " + json.dumps([row]))
        streamed = list(iter_results_from_json(source))
        assert len(streamed) == 1
