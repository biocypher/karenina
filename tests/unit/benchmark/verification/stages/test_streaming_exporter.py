"""Tests for export_verification_results_json_stream.

Covers structural correctness (format_version, metadata shape, is_complete),
error semantics (iterator raises, disk write fails), and golden byte-
equality against checked-in fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.export_format.fixture_builders import (  # noqa: F401
    FIXED_EXPORT_TIMESTAMP,
    FIXED_KARENINA_VERSION,
    build_empty_job,
    build_empty_results,
    build_full_job,
    build_full_results,
    build_full_rubric,
)


@pytest.fixture
def deterministic_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin export_timestamp and karenina_version so header bytes are stable."""
    import time

    import karenina.benchmark.verification.stages.helpers.results_exporter as exporter_mod

    def fake_strftime(fmt: str, t: object = None) -> str:
        return FIXED_EXPORT_TIMESTAMP

    monkeypatch.setattr(time, "strftime", fake_strftime)
    monkeypatch.setattr(exporter_mod, "get_karenina_version", lambda: FIXED_KARENINA_VERSION)


@pytest.mark.unit
class TestStreamingExporterEmpty:
    def test_empty_iter_produces_valid_json_with_empty_results(
        self,
        tmp_path: Path,
        deterministic_header: None,  # noqa: ARG002
    ) -> None:
        from karenina.benchmark.verification.stages.helpers.results_exporter import (
            export_verification_results_json_stream,
        )

        out_path = tmp_path / "out.json"
        job = build_empty_job()

        export_verification_results_json_stream(
            job,
            iter(build_empty_results().results),
            out_path=out_path,
        )

        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["format_version"] == "2.2"
        assert data["results"] == []
        assert data["metadata"]["job_id"] == job.job_id
        assert data["metadata"]["job_summary"]["is_complete"] is False
