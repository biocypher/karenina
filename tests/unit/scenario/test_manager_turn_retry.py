"""Tests for ScenarioManager per-turn retry on transient errors."""

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
    transient: bool = False,
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
        is_transient_error=transient,
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
class TestScenarioManagerTurnRetry:
    """Per-turn retry when _run_turn returns a transient error VR."""

    @patch.object(ScenarioManager, "_run_turn")
    def test_no_retry_on_success(self, mock_run_turn: MagicMock) -> None:
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

    @patch("karenina.scenario.manager.time.sleep")
    @patch.object(ScenarioManager, "_run_turn")
    def test_retry_on_transient_error_then_success(self, mock_run_turn: MagicMock, _mock_sleep: MagicMock) -> None:
        """Transient error on first attempt, success on second."""
        vr_fail = _make_vr(completed=False, transient=True, error="Connection error")
        vr_ok = _make_vr(completed=True)
        mock_run_turn.side_effect = [
            (vr_fail, None, None, ""),
            (vr_ok, None, None, "answer"),
        ]

        manager = ScenarioManager()
        config = _make_config(max_scenario_turn_retries=2)
        result = manager.run(
            scenario=_make_scenario(),
            config=config,
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
        )
        assert result.status == "completed"
        assert mock_run_turn.call_count == 2

    @patch.object(ScenarioManager, "_run_turn")
    def test_no_retry_on_permanent_error(self, mock_run_turn: MagicMock) -> None:
        """Non-transient error: no retry, status is 'error'."""
        vr_fail = _make_vr(completed=False, transient=False, error="ValueError: bad")
        mock_run_turn.return_value = (vr_fail, None, None, "")

        manager = ScenarioManager()
        config = _make_config(max_scenario_turn_retries=3)
        result = manager.run(
            scenario=_make_scenario(),
            config=config,
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
        )
        assert result.status == "error"
        assert mock_run_turn.call_count == 1

    @patch("karenina.scenario.manager.time.sleep")
    @patch.object(ScenarioManager, "_run_turn")
    def test_retries_exhaust_then_error(self, mock_run_turn: MagicMock, _mock_sleep: MagicMock) -> None:
        """All retry attempts fail with transient errors."""
        vr_fail = _make_vr(completed=False, transient=True, error="Connection error")
        mock_run_turn.return_value = (vr_fail, None, None, "")

        manager = ScenarioManager()
        config = _make_config(max_scenario_turn_retries=3)
        result = manager.run(
            scenario=_make_scenario(),
            config=config,
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
        )
        assert result.status == "error"
        assert mock_run_turn.call_count == 3

    @patch("karenina.scenario.manager.time.sleep")
    @patch.object(ScenarioManager, "_run_turn")
    def test_retry_count_from_config(self, mock_run_turn: MagicMock, _mock_sleep: MagicMock) -> None:
        """max_scenario_turn_retries controls total attempts."""
        vr_fail = _make_vr(completed=False, transient=True, error="timeout")
        mock_run_turn.return_value = (vr_fail, None, None, "")

        manager = ScenarioManager()
        config = _make_config(max_scenario_turn_retries=5)
        manager.run(
            scenario=_make_scenario(),
            config=config,
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
        )
        assert mock_run_turn.call_count == 5

    @patch("karenina.scenario.manager.time.sleep")
    @patch.object(ScenarioManager, "_run_turn")
    def test_retry_sleeps_between_attempts(self, mock_run_turn: MagicMock, mock_sleep: MagicMock) -> None:
        """time.sleep(1) called between retries but not after the last attempt."""
        vr_fail = _make_vr(completed=False, transient=True, error="Connection error")
        mock_run_turn.return_value = (vr_fail, None, None, "")

        manager = ScenarioManager()
        config = _make_config(max_scenario_turn_retries=3)
        manager.run(
            scenario=_make_scenario(),
            config=config,
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
        )
        # 3 attempts = 2 sleeps (between attempt 1-2 and 2-3, not after 3)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1)

    @patch("karenina.scenario.manager.time.sleep")
    @patch.object(ScenarioManager, "_run_turn")
    def test_retry_logs_warning(
        self, mock_run_turn: MagicMock, _mock_sleep: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning logged for each retry with scenario name, node, attempt number."""
        vr_fail = _make_vr(completed=False, transient=True, error="Connection error")
        vr_ok = _make_vr(completed=True)
        mock_run_turn.side_effect = [
            (vr_fail, None, None, ""),
            (vr_ok, None, None, "answer"),
        ]

        manager = ScenarioManager()
        config = _make_config(max_scenario_turn_retries=2)
        with caplog.at_level(logging.WARNING, logger="karenina.scenario.manager"):
            manager.run(
                scenario=_make_scenario(),
                config=config,
                base_answering_model=_make_model(),
                base_parsing_model=_make_model(),
            )

        retry_warnings = [r for r in caplog.records if "transient" in r.message.lower()]
        assert len(retry_warnings) == 1
        assert "test_scenario" in retry_warnings[0].message
        assert "ask" in retry_warnings[0].message
        assert "1/2" in retry_warnings[0].message

    @patch("karenina.scenario.manager.time.sleep")
    @patch.object(ScenarioManager, "_run_turn")
    def test_retry_preserves_conversation_history(self, mock_run_turn: MagicMock, _mock_sleep: MagicMock) -> None:
        """Same conversation_history passed on each retry attempt."""
        vr_fail = _make_vr(completed=False, transient=True, error="timeout")
        vr_ok = _make_vr(completed=True)
        mock_run_turn.side_effect = [
            (vr_fail, None, None, ""),
            (vr_ok, None, None, "answer"),
        ]

        manager = ScenarioManager()
        config = _make_config(max_scenario_turn_retries=2)
        manager.run(
            scenario=_make_scenario(),
            config=config,
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
        )

        # Both calls should receive the same conversation_history kwarg
        call1_kwargs = mock_run_turn.call_args_list[0].kwargs
        call2_kwargs = mock_run_turn.call_args_list[1].kwargs
        assert call1_kwargs["conversation_history"] == call2_kwargs["conversation_history"]

    @patch("karenina.scenario.manager.time.sleep")
    @patch.object(ScenarioManager, "_run_turn")
    def test_cache_completed_only_after_final_attempt(self, mock_run_turn: MagicMock, _mock_sleep: MagicMock) -> None:
        """Cache complete() called once after the retry loop, not between attempts."""
        vr_fail = _make_vr(completed=False, transient=True, error="timeout")
        vr_ok = _make_vr(completed=True)
        mock_run_turn.side_effect = [
            (vr_fail, None, None, ""),
            (vr_ok, None, None, "answer"),
        ]

        mock_cache = MagicMock()
        mock_cache.get_or_reserve.return_value = ("MISS", None)
        mock_cache.wait_for_completion.return_value = True

        manager = ScenarioManager()
        config = _make_config(max_scenario_turn_retries=2)
        manager.run(
            scenario=_make_scenario(),
            config=config,
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
            answer_cache=mock_cache,
        )

        # complete() called once (after retry loop), not twice
        assert mock_cache.complete.call_count == 1

    @patch.object(ScenarioManager, "_run_turn")
    def test_cache_not_completed_on_hit(self, mock_run_turn: MagicMock) -> None:
        """When cache returns HIT, complete() should not be called."""
        vr_ok = _make_vr(completed=True)
        mock_run_turn.return_value = (vr_ok, None, None, "cached answer")

        mock_cache = MagicMock()
        mock_cache.get_or_reserve.return_value = ("HIT", {"raw_response": "cached"})

        manager = ScenarioManager()
        config = _make_config()
        manager.run(
            scenario=_make_scenario(),
            config=config,
            base_answering_model=_make_model(),
            base_parsing_model=_make_model(),
            answer_cache=mock_cache,
        )

        mock_cache.complete.assert_not_called()
