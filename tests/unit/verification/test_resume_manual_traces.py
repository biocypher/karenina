"""Unit tests for resuming manual-interface runs.

The progressive state file excludes ``manual_traces`` (not serializable),
so a manual-interface resume must re-supply them. These tests cover the
injection path in ``ProgressiveFileSink.load_for_resume`` and the CLI
resume handler that combines ``--resume`` with ``--manual-traces``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typer

from karenina.adapters.manual import (
    clear_manual_traces,
    get_manual_trace_count,
    load_manual_traces_from_file,
)
from karenina.benchmark.verification.sinks import ProgressiveFileSink
from karenina.cli.verify import _handle_resume_mode
from karenina.schemas import VerificationConfig
from karenina.schemas.config import ModelConfig

TRACE_DATA = {
    "d41d8cd98f00b204e9800998ecf8427e": "Answer 1",
    "936dbc8755f623c951d96ea2b03e13bc": "Answer 2",
}


def _manual_config() -> VerificationConfig:
    """Config with a manual answering model (sentinel satisfies the validator)."""
    return VerificationConfig(
        answering_models=[ModelConfig(interface="manual", manual_traces=object())],
        parsing_models=[
            ModelConfig(id="p1", model_name="gpt-4.1-mini", model_provider="openai", interface="langchain")
        ],
    )


def _write_state(tmp_path: Path) -> Path:
    """Persist a manual-interface progressive state via ``_save_state``."""
    sink = ProgressiveFileSink(
        output_path=tmp_path / "results.json",
        config=_manual_config(),
        benchmark_path=str(tmp_path / "bench.jsonld"),
    )
    sink.on_start(["q1|manual|langchain:gpt-4.1-mini|0"], sink.config)
    return sink.state_path


def _traces_file(tmp_path: Path) -> Path:
    path = tmp_path / "traces.json"
    path.write_text(json.dumps(TRACE_DATA))
    return path


@pytest.fixture(autouse=True)
def _clean_trace_manager():
    """Keep the global manual trace manager isolated between tests."""
    clear_manual_traces()
    yield
    clear_manual_traces()


@pytest.mark.unit
class TestLoadForResumeManualTraces:
    def test_state_excludes_manual_traces(self, tmp_path: Path) -> None:
        """The saved config cannot carry manual_traces (root cause of the bug)."""
        state = json.loads(_write_state(tmp_path).read_text())
        assert "manual_traces" not in state["config"]["answering_models"][0]

    def test_resume_without_traces_fails_with_validation_error(self, tmp_path: Path) -> None:
        """Without re-supplied traces the config revalidation still fails clearly."""
        state_path = _write_state(tmp_path)
        with pytest.raises(ValueError, match="manual_traces is required when interface='manual'"):
            ProgressiveFileSink.load_for_resume(state_path)

    def test_resume_with_traces_attaches_before_validation(self, tmp_path: Path) -> None:
        """Traces passed to load_for_resume land on the manual answering model."""
        state_path = _write_state(tmp_path)
        traces = load_manual_traces_from_file(_traces_file(tmp_path), MagicMock())

        sink = ProgressiveFileSink.load_for_resume(state_path, manual_traces=traces)

        assert sink.config.answering_models[0].manual_traces is traces
        assert get_manual_trace_count() == len(TRACE_DATA)
        assert sink.completed_count == 0
        assert sink.total_tasks == 1

    def test_resume_with_traces_leaves_non_manual_models_alone(self, tmp_path: Path) -> None:
        """A config whose answering models are not manual resumes unchanged."""
        config = VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="a1",
                    model_name="gpt-4.1-mini",
                    model_provider="openai",
                    interface="langchain",
                )
            ],
            parsing_models=[
                ModelConfig(id="p1", model_name="gpt-4.1-mini", model_provider="openai", interface="langchain")
            ],
        )
        sink = ProgressiveFileSink(
            output_path=tmp_path / "results.json", config=config, benchmark_path=str(tmp_path / "bench.jsonld")
        )
        sink.on_start(["q1|langchain:gpt-4.1-mini|langchain:gpt-4.1-mini|0"], config)

        resumed = ProgressiveFileSink.load_for_resume(sink.state_path, manual_traces=MagicMock())
        assert resumed.config.answering_models[0].manual_traces is None


@pytest.mark.unit
class TestCliResumeManualTraces:
    def test_resume_with_manual_traces_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--resume combined with --manual-traces loads the state with traces attached."""
        import importlib

        state_path = _write_state(tmp_path)
        # karenina.cli shadows its verify submodule with the command function,
        # so patch through the real module object. The benchmark file behind
        # the state's benchmark_path does not exist on disk.
        verify_module = importlib.import_module("karenina.cli.verify")
        monkeypatch.setattr(verify_module, "_load_benchmark", lambda _path: MagicMock())

        sink, config, benchmark_path, output, output_format, benchmark = _handle_resume_mode(
            state_path, manual_traces=_traces_file(tmp_path)
        )

        assert config.answering_models[0].manual_traces is not None
        assert get_manual_trace_count() == len(TRACE_DATA)
        assert benchmark is not None
        assert sink.completed_count == 0

    def test_resume_without_manual_traces_flag_exits_with_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Resuming a manual state without --manual-traces keeps the clear error."""
        state_path = _write_state(tmp_path)
        with pytest.raises(typer.Exit) as excinfo:
            _handle_resume_mode(state_path)

        assert excinfo.value.exit_code == 1
        assert "manual_traces is required" in capsys.readouterr().out
