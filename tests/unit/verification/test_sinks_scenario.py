"""Unit tests for sink support on scenario benchmarks.

Covers the combo-atomic resume contract:

- ``TaskIdentifier.from_result`` keys on ``metadata.scenario_id`` when set,
  so every turn in the same combo shares a task key.
- ``ProgressiveFileSink.on_result`` appends every turn to the JSONL but
  only marks the combo-level task as complete once.
- ``completed_triples()`` returns combo-level keys (scenario_id, ans, parse,
  replicate).
- ``Benchmark._run_scenario_verification(sink=...)`` feeds the sink via
  ``on_start`` / ``on_result`` per turn / ``on_finalize``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.sinks import ProgressiveFileSink
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.utils.progressive_save import TaskIdentifier


def _turn_result(
    scenario_id: str,
    question_id: str,
    *,
    scenario_turn: int = 1,
    scenario_node: str = "n0",
    replicate: int | None = None,
    answering: str = "qwen3.5-a3b",
    parsing: str = "qwen3.5-a3b",
    failure_category: FailureCategory | None = None,
    verify_result: bool | None = None,
) -> VerificationResult:
    """Build a VerificationResult that looks like one scenario turn."""
    ans = ModelIdentity(interface="openai_endpoint", model_name=answering)
    parse = ModelIdentity(interface="openai_endpoint", model_name=parsing)
    ts = datetime.utcnow().isoformat()
    rid = VerificationResultMetadata.compute_result_id(question_id, ans, parse, ts, replicate)
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="tmpl",
            failure=Failure(category=failure_category, stage="verify_template", reason="failed")
            if failure_category is not None
            else None,
            question_text="turn text",
            answering=ans,
            parsing=parse,
            execution_time=0.1,
            timestamp=ts,
            replicate=replicate,
            result_id=rid,
            run_name="scenario-run",
            scenario_id=scenario_id,
            scenario_node=scenario_node,
            scenario_turn=scenario_turn,
        ),
        template=VerificationResultTemplate(verify_result=verify_result),
    )


def _config() -> VerificationConfig:
    # id == model_name so ModelIdentity.config_id is None and canonical_key
    # matches what _turn_result produces via ModelIdentity(interface=..., model_name=...).
    return VerificationConfig(
        answering_models=[
            ModelConfig(
                id="qwen3.5-a3b",
                model_name="qwen3.5-a3b",
                interface="openai_endpoint",
                endpoint_base_url="http://codon-gpu-001:8002",
                endpoint_api_key="EMPTY",
            )
        ],
        parsing_models=[
            ModelConfig(
                id="qwen3.5-a3b",
                model_name="qwen3.5-a3b",
                interface="openai_endpoint",
                endpoint_base_url="http://codon-gpu-001:8002",
                endpoint_api_key="EMPTY",
            )
        ],
    )


@pytest.mark.unit
class TestTaskIdentifierScenario:
    def test_from_result_prefers_scenario_id(self):
        r = _turn_result(scenario_id="scenario-A", question_id="q_node_1")
        tid = TaskIdentifier.from_result(r)
        assert tid.question_id == "scenario-A"

    def test_from_result_qa_unchanged(self):
        """For QA results (scenario_id is None) we still key on question_id."""
        ans = ModelIdentity(interface="openai_endpoint", model_name="qwen3.5-a3b")
        parse = ModelIdentity(interface="openai_endpoint", model_name="qwen3.5-a3b")
        ts = datetime.utcnow().isoformat()
        rid = VerificationResultMetadata.compute_result_id("qid", ans, parse, ts, None)
        r = VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="qid",
                template_id="t",
                question_text="q",
                answering=ans,
                parsing=parse,
                execution_time=0.0,
                timestamp=ts,
                result_id=rid,
            )
        )
        assert TaskIdentifier.from_result(r).question_id == "qid"

    def test_turns_in_same_combo_share_task_key(self):
        t1 = _turn_result("scenario-A", "q_turn1", scenario_turn=1)
        t2 = _turn_result("scenario-A", "q_turn2", scenario_turn=2)
        t3 = _turn_result("scenario-A", "q_turn3", scenario_turn=3)
        keys = {TaskIdentifier.from_result(t).to_key() for t in (t1, t2, t3)}
        assert len(keys) == 1


@pytest.mark.unit
class TestProgressiveFileSinkScenario:
    def _fresh(self, tmp_path: Path) -> ProgressiveFileSink:
        return ProgressiveFileSink(
            output_path=tmp_path / "results.json",
            config=_config(),
            benchmark_path=str(tmp_path / "bench.jsonld"),
        )

    def test_multiple_turns_one_combo_key(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        combo_key = TaskIdentifier.from_result(_turn_result("S1", "q_t1")).to_key()
        sink.on_start([combo_key], sink.config)

        for turn in range(1, 4):
            sink.on_result(_turn_result("S1", f"q_t{turn}", scenario_turn=turn))

        assert sink.completed_count == 1
        assert len(sink.jsonl_path.read_text().splitlines()) == 3

    def test_completed_triples_returns_combo_keys(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        combo_key = TaskIdentifier.from_result(_turn_result("S1", "q_t1", replicate=2)).to_key()
        sink.on_start([combo_key], sink.config)
        sink.on_result(_turn_result("S1", "q_t1", scenario_turn=1, replicate=2))
        sink.on_result(_turn_result("S1", "q_t2", scenario_turn=2, replicate=2))

        triples = sink.completed_triples()
        assert len(triples) == 1
        scen_id, ans_key, parse_key, rep = next(iter(triples))
        assert scen_id == "S1"
        assert rep == 2
        assert ans_key.startswith("openai_endpoint:")
        assert parse_key.startswith("openai_endpoint:")

    def test_load_for_resume_reconstructs_combo_state(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        combo_key = TaskIdentifier.from_result(_turn_result("S1", "q_t1")).to_key()
        sink.on_start([combo_key], sink.config)
        sink.on_result(_turn_result("S1", "q_t1", scenario_turn=1, failure_category=FailureCategory.CONTENT))
        sink.on_result(_turn_result("S1", "q_t2", scenario_turn=2))

        reloaded = ProgressiveFileSink.load_for_resume(sink.state_path)
        assert reloaded.completed_count == 1
        assert len(reloaded.get_all_results()) == 2
        assert reloaded.completed_triples() == sink.completed_triples()

    def test_final_export_reconstructs_scenario_results(self, tmp_path: Path):
        sink = self._fresh(tmp_path)
        combo_key = TaskIdentifier.from_result(_turn_result("S1", "q_t1", replicate=1)).to_key()
        sink.on_start([combo_key], sink.config)
        sink.on_result(_turn_result("S1", "q_t1", scenario_turn=1, scenario_node="ask", replicate=1))
        sink.on_result(
            _turn_result(
                "S1",
                "q_t2",
                scenario_turn=2,
                scenario_node="adversarial",
                replicate=1,
                answering="sonnet-4.6",
                failure_category=FailureCategory.TRACE_VALIDATION,
            )
        )
        assert sink.completed_count == 1

        sink.on_finalize(all_complete=True)

        payload = json.loads(sink.output_path.read_text())
        assert len(payload["results"]) == 2
        assert len(payload["scenario_results"]) == 1
        scenario = payload["scenario_results"][0]
        assert scenario["scenario_id"] == "S1"
        assert scenario["status"] == "error"
        assert scenario["path"] == ["ask", "adversarial"]
        assert scenario["replicate"] == 1
        assert scenario["terminal_failure"] == {
            "node_id": "adversarial",
            "category": "trace_validation",
            "stage": "verify_template",
            "reason": "failed",
        }


@pytest.mark.unit
class TestRunScenarioVerificationWithSink:
    """End-to-end wiring test: sink lifecycle is invoked correctly."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_on_start_on_result_on_finalize_called(self, MockExecutor, tmp_path: Path):
        from karenina.benchmark.benchmark import Benchmark

        # Two turn results that both belong to combo ("S1", ans, parse, None).
        exec_result = MagicMock()
        exec_result.turn_results = [
            _turn_result("S1", "q_t1", scenario_turn=1),
            _turn_result("S1", "q_t2", scenario_turn=2),
        ]

        # The mock executor must fire the progress_callback with the exec
        # result, exactly like the real ScenarioExecutor, so the sink
        # adapter sees the completion event.
        def _fake_run_batch(**kwargs):
            cb = kwargs.get("progress_callback")
            if cb is not None:
                cb(1, 1, exec_result)
            return ([exec_result], [])

        MockExecutor.return_value.run_batch.side_effect = _fake_run_batch

        benchmark = Benchmark(name="scenario-test")
        scenario = MagicMock()
        scenario.name = "S1"
        scenario.nodes = {}
        benchmark._scenarios = {"S1": scenario}

        config = _config()
        sink = ProgressiveFileSink(
            output_path=tmp_path / "out.json",
            config=config,
            benchmark_path=str(tmp_path / "bench.jsonld"),
        )

        result_set = benchmark._run_scenario_verification(config, async_enabled=False, sink=sink)

        # Sink sees manifest of one combo, both turns appended, combo marked done.
        assert sink.total_tasks == 1
        assert sink.completed_count == 1
        # On clean completion, sidecars are deleted and final export exists.
        assert not sink.state_path.exists()
        assert not sink.jsonl_path.exists()
        assert sink.output_path.exists()
        # Returned set contains both turns.
        assert len(result_set) == 2

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_resume_skips_completed_combo(self, MockExecutor, tmp_path: Path):
        from karenina.benchmark.benchmark import Benchmark

        benchmark = Benchmark(name="scenario-test")
        s1 = MagicMock()
        s1.name = "S1"
        s1.nodes = {}
        s1.outcome_criteria = []
        s2 = MagicMock()
        s2.name = "S2"
        s2.nodes = {}
        s2.outcome_criteria = []
        benchmark._scenarios = {"S1": s1, "S2": s2}

        # Pre-populate a sink with S1 already done.
        config = _config()
        sink = ProgressiveFileSink(
            output_path=tmp_path / "out.json",
            config=config,
            benchmark_path=str(tmp_path / "bench.jsonld"),
        )
        first_key = TaskIdentifier.from_result(_turn_result("S1", "q_t1")).to_key()
        second_key = TaskIdentifier.from_result(_turn_result("S2", "q_t1")).to_key()
        sink.on_start([first_key, second_key], config)
        sink.on_result(_turn_result("S1", "q_t1", scenario_turn=1))

        # Executor should only see S2 this pass.
        exec_result_s2 = MagicMock()
        exec_result_s2.scenario_id = "S2"
        exec_result_s2.turn_results = [_turn_result("S2", "q_t1", scenario_turn=1)]
        MockExecutor.return_value.run_batch.return_value = ([exec_result_s2], [])

        result_set = benchmark._run_scenario_verification(config, async_enabled=False, sink=sink)

        combos_passed = MockExecutor.return_value.run_batch.call_args.kwargs["combos"]
        scen_names = {c[0].name for c in combos_passed}
        assert scen_names == {"S2"}
        assert len(result_set.results) == 2
        assert {r.metadata.scenario_id for r in result_set.results} == {"S1", "S2"}
        assert result_set.scenario_results is not None
        assert len(result_set.scenario_results) == 2
        assert {r.scenario_id for r in result_set.scenario_results} == {"S1", "S2"}
        prior = next(r for r in result_set.scenario_results if r.scenario_id == "S1")
        assert prior.status == "completed"
