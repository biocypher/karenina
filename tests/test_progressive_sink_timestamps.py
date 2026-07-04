"""Tests that ProgressiveFileSink.write_final_export sets job timestamps.

Without end_time, export_verification_results_json_stream emits
total_duration=None. The sink must populate start_time (from its state)
and end_time (now) before calling the exporter.
"""

import json
import time
from pathlib import Path

import pytest

from karenina.benchmark.verification.sinks import ProgressiveFileSink
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification import VerificationConfig


def _make_config() -> VerificationConfig:
    model = ModelConfig(
        id="qwen-test",
        model_name="qwen-test",
        model_provider=None,
        interface="openai_endpoint",
        temperature=0.7,
    )
    return VerificationConfig(
        answering_models=[model],
        parsing_models=[model],
        replicate_count=1,
    )


@pytest.mark.unit
class TestProgressiveSinkTimestamps:
    def test_final_export_populates_start_end_duration(self, tmp_path: Path) -> None:
        config = _make_config()
        sink = ProgressiveFileSink(
            output_path=tmp_path / "out.json",
            config=config,
            benchmark_path="benchmark.jsonld",
        )
        sink.on_start(manifest=[], config=config)

        # Simulate at least 10ms of elapsed time
        time.sleep(0.01)
        export_path = sink.write_final_export()

        data = json.loads(export_path.read_text())
        summary = data["metadata"]["job_summary"]

        assert summary["start_time"] is not None
        assert summary["end_time"] is not None
        assert summary["total_duration"] is not None
        assert summary["total_duration"] >= 0.01

    def test_is_complete_true_on_final_export(self, tmp_path: Path) -> None:
        config = _make_config()
        sink = ProgressiveFileSink(
            output_path=tmp_path / "out.json",
            config=config,
            benchmark_path="benchmark.jsonld",
        )
        sink.on_start(manifest=[], config=config)

        export_path = sink.write_final_export()
        data = json.loads(export_path.read_text())

        assert data["metadata"]["job_summary"]["is_complete"] is True
