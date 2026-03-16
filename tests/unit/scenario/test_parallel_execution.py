"""Tests for parallel scenario execution via asyncio.gather."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from karenina.benchmark.benchmark import Benchmark
from karenina.scenario.builder import Scenario
from karenina.schemas.config import ModelConfig
from karenina.schemas.scenario.state import ScenarioExecutionResult, ScenarioState
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


def _make_config(
    answering: list[ModelConfig] | None = None,
    parsing: list[ModelConfig] | None = None,
) -> VerificationConfig:
    return VerificationConfig(
        answering_models=answering or [_make_model("claude")],
        parsing_models=parsing or [_make_model("haiku", "anthropic")],
    )


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
        outcome_results={},
    )


@pytest.mark.unit
class TestScenarioRunSync:
    """Tests for synchronous (non-parallel) scenario verification."""

    def test_single_scenario_single_model_invokes_manager_run(self, monkeypatch):
        """With one scenario and one model pair, manager.run is called once."""
        bm = Benchmark("scenario_bm")
        bm.add_scenario(_build_scenario("alpha"))
        config = _make_config()

        call_count = {"n": 0}
        exec_result = _make_exec_result("alpha")

        def fake_run(**kwargs: Any) -> ScenarioExecutionResult:
            call_count["n"] += 1
            return exec_result

        monkeypatch.setattr(
            "karenina.scenario.manager.ScenarioManager.run",
            lambda _self, **kw: fake_run(**kw),
        )

        result = bm._run_scenario_verification(config=config)
        assert call_count["n"] == 1
        assert len(result.results) == 0  # exec_result has empty turn_results

    def test_cross_product_2_scenarios_2_models(self, monkeypatch):
        """2 scenarios x 2 answering models x 1 parsing model = 4 runs."""
        bm = Benchmark("scenario_bm")
        bm.add_scenario(_build_scenario("alpha"))
        bm.add_scenario(_build_scenario("beta"))

        config = _make_config(
            answering=[_make_model("claude"), _make_model("gpt-4o", "openai")],
            parsing=[_make_model("haiku")],
        )

        call_count = {"n": 0}

        def fake_run(**kwargs: Any) -> ScenarioExecutionResult:
            call_count["n"] += 1
            return _make_exec_result(kwargs.get("scenario", MagicMock()).name)

        monkeypatch.setattr(
            "karenina.scenario.manager.ScenarioManager.run",
            lambda _self, **kw: fake_run(**kw),
        )

        bm._run_scenario_verification(config=config)
        assert call_count["n"] == 4

    def test_sync_mode_when_async_false(self, monkeypatch):
        """When async_enabled is False, runs synchronously even with multiple combos."""
        bm = Benchmark("scenario_bm")
        bm.add_scenario(_build_scenario("alpha"))
        bm.add_scenario(_build_scenario("beta"))

        config = _make_config()

        call_count = {"n": 0}

        def fake_run(**kwargs: Any) -> ScenarioExecutionResult:
            call_count["n"] += 1
            return _make_exec_result()

        monkeypatch.setattr(
            "karenina.scenario.manager.ScenarioManager.run",
            lambda _self, **kw: fake_run(**kw),
        )

        bm._run_scenario_verification(config=config, async_enabled=False)
        assert call_count["n"] == 2


@pytest.mark.unit
class TestScenarioRunParallel:
    """Tests for parallel scenario execution via asyncio.gather."""

    def test_async_enabled_calls_arun(self, monkeypatch):
        """When async_enabled=True and multiple combos, uses _run_scenario_parallel."""
        bm = Benchmark("scenario_bm")
        bm.add_scenario(_build_scenario("alpha"))
        bm.add_scenario(_build_scenario("beta"))

        config = _make_config()

        parallel_called = {"called": False}

        def fake_parallel(_self, **kwargs: Any) -> list:
            parallel_called["called"] = True
            return []

        monkeypatch.setattr(
            Benchmark,
            "_run_scenario_parallel",
            fake_parallel,
        )

        bm._run_scenario_verification(config=config, async_enabled=True)
        assert parallel_called["called"] is True

    def test_async_single_combo_stays_sync(self, monkeypatch):
        """With only one combo, async_enabled=True still runs synchronously."""
        bm = Benchmark("scenario_bm")
        bm.add_scenario(_build_scenario("alpha"))

        config = _make_config()

        sync_called = {"n": 0}

        def fake_run(**kwargs: Any) -> ScenarioExecutionResult:
            sync_called["n"] += 1
            return _make_exec_result()

        monkeypatch.setattr(
            "karenina.scenario.manager.ScenarioManager.run",
            lambda _self, **kw: fake_run(**kw),
        )

        # Should NOT call _run_scenario_parallel because len(combos) == 1
        bm._run_scenario_verification(config=config, async_enabled=True)
        assert sync_called["n"] == 1

    def test_parallel_gathers_results_from_arun(self, monkeypatch):
        """_run_scenario_parallel invokes arun for each combo and collects results."""
        from karenina.scenario.manager import ScenarioManager

        bm = Benchmark("scenario_bm")
        bm.add_scenario(_build_scenario("alpha"))
        bm.add_scenario(_build_scenario("beta"))

        config = _make_config()

        arun_count = {"n": 0}

        async def fake_arun(_self, **kwargs: Any) -> ScenarioExecutionResult:
            arun_count["n"] += 1
            return _make_exec_result()

        monkeypatch.setattr(ScenarioManager, "arun", fake_arun)

        manager = ScenarioManager()
        combos = [
            (bm.get_scenario("alpha"), _make_model("claude"), _make_model("haiku")),
            (bm.get_scenario("beta"), _make_model("claude"), _make_model("haiku")),
        ]

        results = bm._run_scenario_parallel(
            manager=manager,
            combos=combos,
            config=config,
            run_name=None,
            global_rubric=None,
            progress_callback=None,
        )

        assert arun_count["n"] == 2
        # Both exec_results have empty turn_results, so total is 0
        assert len(results) == 0

    def test_parallel_exception_in_one_combo_does_not_block_others(self, monkeypatch):
        """If one arun raises, the others still complete successfully."""
        from karenina.scenario.manager import ScenarioManager

        bm = Benchmark("scenario_bm")
        bm.add_scenario(_build_scenario("alpha"))
        bm.add_scenario(_build_scenario("beta"))

        config = _make_config()

        call_index = {"n": 0}

        async def fake_arun(_self, **kwargs: Any) -> ScenarioExecutionResult:
            idx = call_index["n"]
            call_index["n"] += 1
            if idx == 0:
                raise RuntimeError("boom")
            return _make_exec_result()

        monkeypatch.setattr(ScenarioManager, "arun", fake_arun)

        manager = ScenarioManager()
        combos = [
            (bm.get_scenario("alpha"), _make_model("claude"), _make_model("haiku")),
            (bm.get_scenario("beta"), _make_model("claude"), _make_model("haiku")),
        ]

        # Should not raise; the exception is logged and skipped
        results = bm._run_scenario_parallel(
            manager=manager,
            combos=combos,
            config=config,
            run_name=None,
            global_rubric=None,
            progress_callback=None,
        )
        # Only the second combo succeeded (with empty turn_results)
        assert isinstance(results, list)
