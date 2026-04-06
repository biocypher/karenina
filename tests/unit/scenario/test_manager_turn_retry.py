"""Tests for ScenarioManager turn failure behavior (no retry).

The scenario turn retry loop has been removed. The adapter handles retries
internally. When a turn fails, the scenario terminates immediately with
status 'error'.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from karenina.scenario.manager import ScenarioManager
from karenina.schemas.config import ModelConfig
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultDeepJudgment,
    VerificationResultDeepJudgmentRubric,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


def _make_model(name: str = "test-model") -> ModelConfig:
    return ModelConfig(id=name, model_name=name, model_provider="openai")


def _make_vr(
    *,
    completed: bool = True,
    error_category: str | None = None,
    error: str | None = None,
    verify_result: bool | None = True,
) -> VerificationResult:
    """Create a minimal VerificationResult with controllable error state."""
    identity = ModelIdentity(model_name="test", interface="openai")
    metadata = VerificationResultMetadata(
        question_id="q1",
        template_id="tpl1",
        completed_without_errors=completed,
        error=error,
        error_category=error_category,
        question_text="What?",
        answering=identity,
        parsing=identity,
        execution_time=1.0,
        timestamp="2026-01-01T00:00:00Z",
        result_id="abcdef1234567890",
    )
    template = VerificationResultTemplate(verify_result=verify_result)
    return VerificationResult(
        metadata=metadata,
        template=template,
        rubric=VerificationResultRubric(),
        deep_judgment=VerificationResultDeepJudgment(),
        deep_judgment_rubric=VerificationResultDeepJudgmentRubric(),
    )


def _make_scenario() -> ScenarioDefinition:
    """Single-node scenario: ask -> END."""
    from karenina.schemas.entities import Question

    q = Question(question="What?", raw_answer="Y", answer_template="class Answer: pass")
    node = ScenarioNode(node_id="ask", question=q)
    return ScenarioDefinition(
        name="test_scenario",
        description="test",
        nodes={"ask": node},
        edges=[ScenarioEdge(source="ask", target=END)],
        entry_node="ask",
        outcome_criteria=[],
    )


def _make_config(**overrides) -> VerificationConfig:
    defaults = {
        "answering_models": [_make_model()],
        "parsing_models": [_make_model()],
    }
    defaults.update(overrides)
    with patch("karenina.schemas.verification.config.os.getenv", return_value=None):
        return VerificationConfig(**defaults)


@pytest.mark.unit
class TestScenarioManagerNoTurnRetry:
    """Turn failures terminate the scenario immediately; no retry loop."""

    @patch.object(ScenarioManager, "_run_turn")
    def test_successful_turn_calls_run_turn_once(self, mock_run_turn: MagicMock) -> None:
        """Successful turn: _run_turn called once, scenario completes."""
        vr_ok = _make_vr(completed=True)
        mock_run_turn.return_value = (vr_ok, None, None, "answer")

        manager = ScenarioManager()
        result = manager.run(
            scenario=_make_scenario(),
            config=_make_config(),
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
        )
        assert result.status == "completed"
        assert mock_run_turn.call_count == 1

    @patch.object(ScenarioManager, "_run_turn")
    def test_transient_error_terminates_immediately(self, mock_run_turn: MagicMock) -> None:
        """Transient error terminates the scenario; no retry attempted."""
        vr_fail = _make_vr(completed=False, error_category="connection", error="Connection error")
        mock_run_turn.return_value = (vr_fail, None, None, "")

        manager = ScenarioManager()
        result = manager.run(
            scenario=_make_scenario(),
            config=_make_config(),
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
        )
        assert result.status == "error"
        assert mock_run_turn.call_count == 1

    @patch.object(ScenarioManager, "_run_turn")
    def test_permanent_error_terminates_immediately(self, mock_run_turn: MagicMock) -> None:
        """Permanent error terminates the scenario; no retry attempted."""
        vr_fail = _make_vr(completed=False, error_category="permanent", error="ValueError: bad")
        mock_run_turn.return_value = (vr_fail, None, None, "")

        manager = ScenarioManager()
        result = manager.run(
            scenario=_make_scenario(),
            config=_make_config(),
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
        )
        assert result.status == "error"
        assert mock_run_turn.call_count == 1

    @patch.object(ScenarioManager, "_run_turn")
    def test_error_with_no_category_terminates_immediately(self, mock_run_turn: MagicMock) -> None:
        """Error with no category terminates the scenario; no retry attempted."""
        vr_fail = _make_vr(completed=False, error_category=None, error="Unknown error")
        mock_run_turn.return_value = (vr_fail, None, None, "")

        manager = ScenarioManager()
        result = manager.run(
            scenario=_make_scenario(),
            config=_make_config(),
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
        )
        assert result.status == "error"
        assert mock_run_turn.call_count == 1

    @patch.object(ScenarioManager, "_run_turn")
    def test_failure_logs_error_with_details(self, mock_run_turn: MagicMock, caplog: pytest.LogCaptureFixture) -> None:
        """Turn failure logs error with scenario name, node, error, and category."""
        vr_fail = _make_vr(completed=False, error_category="connection", error="Connection refused")
        mock_run_turn.return_value = (vr_fail, None, None, "")

        manager = ScenarioManager()
        with caplog.at_level(logging.ERROR, logger="karenina.scenario.manager"):
            manager.run(
                scenario=_make_scenario(),
                config=_make_config(),
                base_answering_model=_make_model(),
                base_parsing_model=_make_model(),
            )

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) == 1
        msg = error_records[0].message
        assert "test_scenario" in msg
        assert "ask" in msg
        assert "Connection refused" in msg
        assert "connection" in msg

    def test_time_module_not_imported_in_manager(self) -> None:
        """The time module is no longer imported; retry sleep logic has been removed."""
        import karenina.scenario.manager as mgr_module

        assert not hasattr(mgr_module, "time"), "time should not be imported in scenario manager"
