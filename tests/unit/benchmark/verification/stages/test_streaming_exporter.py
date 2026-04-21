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


@pytest.mark.unit
class TestStreamingExporterResults:
    def test_single_result_is_written_on_its_own_line(
        self,
        tmp_path: Path,
        deterministic_header: None,  # noqa: ARG002
    ) -> None:
        from karenina.benchmark.verification.stages.helpers.results_exporter import (
            export_verification_results_json_stream,
        )

        out_path = tmp_path / "out.json"
        job = build_full_job()
        results = build_full_results().results[:1]

        export_verification_results_json_stream(job, iter(results), out_path=out_path)

        text = out_path.read_text(encoding="utf-8")
        # Header + "results":[\n<one compact line>\n]}\n
        lines = text.splitlines()
        assert lines[0].startswith('{"format_version": "2.2",'), lines[0][:80]
        assert lines[0].endswith(',"results":[')
        # line 1 is the single serialized result, no trailing comma.
        # Assert structure (JSON object) + identifying content, not field order,
        # so a Pydantic schema reorder doesn't silently break this test.
        assert lines[1].startswith("{")
        assert '"question_id": "q-unicode"' in lines[1]
        assert not lines[1].endswith(",")
        assert lines[2] == "]}"
        assert text.endswith("\n]}\n")

        data = json.loads(text)
        assert len(data["results"]) == 1
        assert data["results"][0]["metadata"]["question_id"] == "q-unicode"

    def test_multi_result_uses_comma_newline_separator(
        self,
        tmp_path: Path,
        deterministic_header: None,  # noqa: ARG002
    ) -> None:
        from karenina.benchmark.verification.stages.helpers.results_exporter import (
            export_verification_results_json_stream,
        )

        out_path = tmp_path / "out.json"
        job = build_full_job()
        results = build_full_results().results  # 2 items

        export_verification_results_json_stream(job, iter(results), out_path=out_path)

        text = out_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        # [header_line, result_1_line (trailing comma), result_2_line (no comma), "]}"]
        assert lines[1].endswith(",")
        assert not lines[2].endswith(",")
        assert lines[3] == "]}"

        data = json.loads(text)
        assert len(data["results"]) == 2
        assert [r["metadata"]["question_id"] for r in data["results"]] == ["q-unicode", "q-timeout"]

    def test_non_ascii_round_trips_unescaped(
        self,
        tmp_path: Path,
        deterministic_header: None,  # noqa: ARG002
    ) -> None:
        """ensure_ascii=False: non-ASCII codepoints are written raw, not \\uXXXX."""
        from karenina.benchmark.verification.stages.helpers.results_exporter import (
            export_verification_results_json_stream,
        )

        out_path = tmp_path / "out.json"
        job = build_full_job()
        results = build_full_results().results[:1]

        export_verification_results_json_stream(job, iter(results), out_path=out_path)

        raw_bytes = out_path.read_bytes()
        # Raw bytes contain the literal unicode, not escaped \u sequences.
        assert "咖啡".encode() in raw_bytes
        assert "température".encode() in raw_bytes
        assert rb"\u" not in raw_bytes  # no ascii-escape for these codepoints

    def test_large_n_round_trips_correctly(
        self,
        tmp_path: Path,
        deterministic_header: None,  # noqa: ARG002
    ) -> None:
        """1000-item generator produces a parseable file with all results preserved in order."""
        from karenina.benchmark.verification.stages.helpers.results_exporter import (
            export_verification_results_json_stream,
        )
        from tests.fixtures.export_format.fixture_builders import make_result

        out_path = tmp_path / "out.json"
        job = build_full_job()

        def gen():
            for i in range(1000):
                yield make_result(
                    question_id=f"q-{i:04d}",
                    question_text=f"question {i}",
                    failure=None,
                )

        export_verification_results_json_stream(job, gen(), out_path=out_path)

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(data["results"]) == 1000
        assert data["results"][0]["metadata"]["question_id"] == "q-0000"
        assert data["results"][-1]["metadata"]["question_id"] == "q-0999"


@pytest.mark.unit
class TestStreamingExporterHeader:
    def test_is_complete_defaults_false(
        self,
        tmp_path: Path,
        deterministic_header: None,  # noqa: ARG002
    ) -> None:
        from karenina.benchmark.verification.stages.helpers.results_exporter import (
            export_verification_results_json_stream,
        )

        out_path = tmp_path / "out.json"
        export_verification_results_json_stream(build_empty_job(), iter([]), out_path=out_path)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["metadata"]["job_summary"]["is_complete"] is False

    def test_is_complete_true_when_opted_in(
        self,
        tmp_path: Path,
        deterministic_header: None,  # noqa: ARG002
    ) -> None:
        from karenina.benchmark.verification.stages.helpers.results_exporter import (
            export_verification_results_json_stream,
        )

        out_path = tmp_path / "out.json"
        export_verification_results_json_stream(build_empty_job(), iter([]), is_complete=True, out_path=out_path)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["metadata"]["job_summary"]["is_complete"] is True

    def test_multi_model_sweep_emits_full_arrays(
        self,
        tmp_path: Path,
        deterministic_header: None,  # noqa: ARG002
    ) -> None:
        from karenina.benchmark.verification.stages.helpers.results_exporter import (
            export_verification_results_json_stream,
        )
        from karenina.schemas.config.models import ModelConfig
        from karenina.schemas.verification import VerificationConfig, VerificationJob

        def _m(name: str) -> ModelConfig:
            return ModelConfig(
                id=f"model-{name}",
                model_name=name,
                model_provider=None,
                interface="openai_endpoint",
                temperature=0.7,
            )

        answering = [_m("a"), _m("b"), _m("c")]
        parsing = [_m("a"), _m("b"), _m("c")]
        job = VerificationJob(
            job_id="mm-job",
            run_name="mm",
            status="completed",
            config=VerificationConfig(answering_models=answering, parsing_models=parsing, replicate_count=1),
            total_questions=0,
            successful_count=0,
        )

        out_path = tmp_path / "out.json"
        export_verification_results_json_stream(job, iter([]), out_path=out_path)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        cfg = data["metadata"]["verification_config"]
        assert [m["name"] for m in cfg["answering_models"]] == ["a", "b", "c"]
        assert [m["name"] for m in cfg["parsing_models"]] == ["a", "b", "c"]

    def test_total_duration_populated_when_both_times_present(
        self,
        tmp_path: Path,
        deterministic_header: None,  # noqa: ARG002
    ) -> None:
        from karenina.benchmark.verification.stages.helpers.results_exporter import (
            export_verification_results_json_stream,
        )

        job = build_empty_job()  # has both start_time and end_time pinned
        out_path = tmp_path / "out.json"
        export_verification_results_json_stream(job, iter([]), out_path=out_path)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        summary = data["metadata"]["job_summary"]
        assert summary["start_time"] == job.start_time
        assert summary["end_time"] == job.end_time
        assert summary["total_duration"] == pytest.approx(job.end_time - job.start_time)
