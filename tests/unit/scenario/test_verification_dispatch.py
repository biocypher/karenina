"""Tests for Benchmark.run_verification() scenario dispatch logic."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from karenina.benchmark.benchmark import Benchmark
from karenina.scenario.builder import Scenario
from karenina.schemas.config import ModelConfig
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.scenario.types import END
from karenina.schemas.verification import VerificationConfig

if TYPE_CHECKING:
    from karenina.schemas.entities import Question


def _make_question(text: str = "What is X?") -> Question:
    from karenina.schemas.entities import Question

    return Question(question=text, raw_answer="Y", answer_template="class Answer: pass")


def _make_model(name: str = "claude", provider: str = "anthropic") -> ModelConfig:
    return ModelConfig(id=name, model_name=name, model_provider=provider)


def _build_scenario(name: str = "test") -> Scenario:
    s = Scenario(name)
    s.add_node("ask", question=_make_question())
    s.add_edge("ask", END)
    s.set_entry("ask")
    return s


def _make_config() -> VerificationConfig:
    return VerificationConfig(
        answering_models=[_make_model("claude")],
        parsing_models=[_make_model("haiku", "anthropic")],
    )


@pytest.mark.unit
class TestVerificationDispatch:
    """Verify that run_verification dispatches to the scenario path."""

    def test_scenario_benchmark_dispatches_to_scenario_verification(self, monkeypatch):
        """When is_scenario_benchmark is True, _run_scenario_verification is called."""
        bm = Benchmark("scenario_bm")
        bm.add_scenario(_build_scenario())

        sentinel = VerificationResultSet(results=[])
        mock_scenario_verif = MagicMock(return_value=sentinel)
        monkeypatch.setattr(bm, "_run_scenario_verification", mock_scenario_verif)

        config = _make_config()
        result = bm.run_verification(config, run_name="test_run")

        assert result is sentinel
        mock_scenario_verif.assert_called_once()
        call_kwargs = mock_scenario_verif.call_args
        assert call_kwargs.kwargs["config"] is config
        assert call_kwargs.kwargs["run_name"] == "test_run"

    def test_standalone_benchmark_dispatches_to_verification_manager(self, monkeypatch):
        """When is_scenario_benchmark is False, VerificationManager is used."""
        bm = Benchmark("question_bm")
        bm.add_question(
            "What?",
            raw_answer="Y",
            answer_template="class Answer: pass",
        )

        sentinel = VerificationResultSet(results=[])
        mock_manager = MagicMock(return_value=sentinel)
        monkeypatch.setattr(bm._verification_manager, "run_verification", mock_manager)

        config = _make_config()
        result = bm.run_verification(config)

        assert result is sentinel
        mock_manager.assert_called_once()

    def test_dispatch_passes_async_enabled(self, monkeypatch):
        """async_enabled parameter is forwarded to _run_scenario_verification."""
        bm = Benchmark("scenario_bm")
        bm.add_scenario(_build_scenario())

        sentinel = VerificationResultSet(results=[])
        mock_scenario_verif = MagicMock(return_value=sentinel)
        monkeypatch.setattr(bm, "_run_scenario_verification", mock_scenario_verif)

        config = _make_config()
        bm.run_verification(config, async_enabled=True)

        call_kwargs = mock_scenario_verif.call_args.kwargs
        assert call_kwargs["async_enabled"] is True

    def test_dispatch_passes_progress_callback(self, monkeypatch):
        """progress_callback is forwarded to _run_scenario_verification."""
        bm = Benchmark("scenario_bm")
        bm.add_scenario(_build_scenario())

        sentinel = VerificationResultSet(results=[])
        mock_scenario_verif = MagicMock(return_value=sentinel)
        monkeypatch.setattr(bm, "_run_scenario_verification", mock_scenario_verif)

        config = _make_config()
        cb = MagicMock()
        bm.run_verification(config, progress_callback=cb)

        call_kwargs = mock_scenario_verif.call_args.kwargs
        assert call_kwargs["progress_callback"] is cb
