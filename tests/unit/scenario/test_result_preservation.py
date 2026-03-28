"""Tests for issues 137 and 143: ScenarioExecutionResult preservation and error collection."""

from __future__ import annotations

from typing import Any

import pytest

from karenina.benchmark.benchmark import Benchmark
from karenina.scenario.builder import Scenario
from karenina.schemas.config import ModelConfig
from karenina.schemas.scenario.state import ScenarioExecutionResult, ScenarioState
from karenina.schemas.scenario.types import END
from karenina.schemas.verification import VerificationConfig


def _make_question(text: str = "What is X?"):
    from karenina.schemas.entities import Question

    return Question(question=text, raw_answer="Y", answer_template="class Answer: pass")


def _make_model(name: str = "claude", provider: str = "anthropic") -> ModelConfig:
    return ModelConfig(id=name, model_name=name, model_provider=provider)


def _make_exec_result(scenario_id: str = "test") -> ScenarioExecutionResult:
    state = ScenarioState(
        turn=1,
        current_node="ask",
        verify_result=True,
        parsed={},
        node_visits={"ask": 1},
        history=[],
        accumulated={},
        node_results={},
    )
    return ScenarioExecutionResult(
        scenario_id=scenario_id,
        status="completed",
        path=["ask"],
        turn_count=1,
        history=[],
        turn_results=[],
        final_state=state,
        outcome_results={"passed": True},
    )


def _make_config() -> VerificationConfig:
    return VerificationConfig(
        answering_models=[_make_model()],
        parsing_models=[_make_model("haiku")],
    )


@pytest.mark.unit
class TestScenarioResultPreservation:
    """Issue 137: VerificationResultSet should preserve ScenarioExecutionResult."""

    def test_sync_path_preserves_scenario_results(self, monkeypatch):
        """Sync scenario verification populates result_set.scenario_results."""
        bm = Benchmark("scenario_bm")
        s = Scenario("alpha")
        s.add_node("ask", question=_make_question())
        s.add_edge("ask", END)
        s.set_entry("ask")
        bm.add_scenario(s)

        exec_result = _make_exec_result("alpha")

        monkeypatch.setattr(
            "karenina.scenario.manager.ScenarioManager.run",
            lambda _self, **_kw: exec_result,
        )

        config = _make_config()
        result_set = bm._run_scenario_verification(config=config)

        assert result_set.scenario_results is not None
        assert len(result_set.scenario_results) == 1
        assert result_set.scenario_results[0].scenario_id == "alpha"
        assert result_set.scenario_results[0].status == "completed"
        assert result_set.scenario_results[0].outcome_results == {"passed": True}


@pytest.mark.unit
class TestParallelErrorCollection:
    """Issue 143: parallel execution should collect errors, not drop them."""

    def test_parallel_errors_collected_on_result_set(self, monkeypatch):
        """Failed parallel executions are collected in result_set.errors."""
        bm = Benchmark("scenario_bm")
        s = Scenario("alpha")
        s.add_node("ask", question=_make_question())
        s.add_edge("ask", END)
        s.set_entry("ask")
        bm.add_scenario(s)

        config = VerificationConfig(
            answering_models=[_make_model("claude"), _make_model("gpt4")],
            parsing_models=[_make_model("haiku")],
        )

        exec_ok = _make_exec_result("alpha")
        err = RuntimeError("LLM provider down")

        call_count = {"n": 0}

        async def fake_arun(_self, **kwargs: Any) -> ScenarioExecutionResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return exec_ok
            raise err

        monkeypatch.setattr(
            "karenina.scenario.manager.ScenarioManager.arun",
            fake_arun,
        )

        result_set = bm._run_scenario_verification(
            config=config,
            async_enabled=True,
        )

        # Successful results still present
        assert len(result_set.results) == 0  # exec_ok has empty turn_results

        # Errors should be collected
        assert result_set.errors is not None
        assert len(result_set.errors) == 1
        desc, exc = result_set.errors[0]
        assert "alpha" in desc
        assert isinstance(exc, RuntimeError)
