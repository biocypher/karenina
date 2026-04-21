"""Tests that export_verification_results_json_stream emits arrays for multi-model sweeps.

The v2.1 format hardcoded answering_models[0] / parsing_models[0], which
misrepresented multi-model runs (e.g., a 3x3 answering x parsing sweep looks
like a single-model run in the exported header). The exporter must emit
arrays of model identities so the header matches the actual sweep.
"""

import json
from pathlib import Path

import pytest

from karenina.benchmark.verification.stages.helpers.results_exporter import (
    export_verification_results_json_stream,
)
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification import VerificationConfig, VerificationJob


def _model(name: str, interface: str = "openai_endpoint") -> ModelConfig:
    return ModelConfig(
        id=f"model-{name}",
        model_name=name,
        model_provider=None,
        interface=interface,
        temperature=0.7,
    )


def _job_with(
    answering: list[ModelConfig],
    parsing: list[ModelConfig],
    *,
    start: float | None = None,
    end: float | None = None,
) -> VerificationJob:
    config = VerificationConfig(
        answering_models=answering,
        parsing_models=parsing,
        replicate_count=1,
    )
    return VerificationJob(
        job_id="test-job",
        run_name="test",
        status="completed",
        config=config,
        total_questions=0,
        successful_count=0,
        start_time=start,
        end_time=end,
    )


@pytest.mark.unit
class TestExportHeaderMultiModel:
    def test_single_model_run_emits_single_element_arrays(self, tmp_path: Path) -> None:
        job = _job_with([_model("qwen-a")], [_model("qwen-a")])
        dst = tmp_path / "out.json"
        export_verification_results_json_stream(job, iter([]), out_path=dst)
        out = json.loads(dst.read_text(encoding="utf-8"))

        cfg = out["metadata"]["verification_config"]
        assert cfg["answering_models"] == [
            {"provider": None, "name": "qwen-a", "temperature": 0.7, "interface": "openai_endpoint"}
        ]
        assert cfg["parsing_models"] == [
            {"provider": None, "name": "qwen-a", "temperature": 0.7, "interface": "openai_endpoint"}
        ]

    def test_multi_model_sweep_emits_full_arrays(self, tmp_path: Path) -> None:
        answering = [_model("qwen-a"), _model("qwen-b"), _model("qwen-c")]
        parsing = [_model("qwen-a"), _model("qwen-b"), _model("qwen-c")]
        job = _job_with(answering, parsing)

        dst = tmp_path / "out.json"
        export_verification_results_json_stream(job, iter([]), out_path=dst)
        out = json.loads(dst.read_text(encoding="utf-8"))

        cfg = out["metadata"]["verification_config"]
        assert [m["name"] for m in cfg["answering_models"]] == ["qwen-a", "qwen-b", "qwen-c"]
        assert [m["name"] for m in cfg["parsing_models"]] == ["qwen-a", "qwen-b", "qwen-c"]

    def test_job_duration_is_emitted_when_both_times_present(self, tmp_path: Path) -> None:
        job = _job_with([_model("qwen-a")], [_model("qwen-a")], start=1000.0, end=1250.5)
        dst = tmp_path / "out.json"
        export_verification_results_json_stream(job, iter([]), out_path=dst)
        out = json.loads(dst.read_text(encoding="utf-8"))

        summary = out["metadata"]["job_summary"]
        assert summary["start_time"] == 1000.0
        assert summary["end_time"] == 1250.5
        assert summary["total_duration"] == pytest.approx(250.5)

    def test_format_version_reflects_schema_bump(self, tmp_path: Path) -> None:
        """format_version bumped to 2.2 because answering_model/parsing_model
        keys became plural arrays (breaking change for v2.1 readers)."""
        job = _job_with([_model("qwen-a")], [_model("qwen-a")])
        dst = tmp_path / "out.json"
        export_verification_results_json_stream(job, iter([]), out_path=dst)
        out = json.loads(dst.read_text(encoding="utf-8"))

        assert out["format_version"] == "2.2"

    def test_is_complete_defaults_false(self, tmp_path: Path) -> None:
        """Callers must opt in to is_complete=True. Default is False so that
        a future intermediate-snapshot caller is not silently mislabeled."""
        job = _job_with([_model("qwen-a")], [_model("qwen-a")])
        dst = tmp_path / "out.json"
        export_verification_results_json_stream(job, iter([]), out_path=dst)
        out = json.loads(dst.read_text(encoding="utf-8"))

        assert out["metadata"]["job_summary"]["is_complete"] is False
