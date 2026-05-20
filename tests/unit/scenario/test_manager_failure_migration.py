"""Tests covering ScenarioManager error-path reads of metadata.failure.

The scenario manager's turn loop checks ``vr.metadata.failure`` to decide when
to log and break out. This test stubs ``_run_turn`` to return a result with a
``Failure`` populated and asserts the manager surfaces ``status='error'`` plus
the failure reason/category in the log record.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from karenina.schemas.results.failure import Failure, FailureCategory


def _build_single_node_scenario(name: str = "failing_scenario"):
    """Minimal one-node scenario that terminates after a single turn."""
    from karenina.schemas.entities import Question
    from karenina.schemas.scenario.definition import ScenarioDefinition
    from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode

    question = Question(
        question="what?",
        raw_answer="y",
        answer_template="class Answer: pass",
    )
    return ScenarioDefinition(
        name=name,
        nodes={"n1": ScenarioNode(node_id="n1", question=question)},
        edges=[ScenarioEdge(source="n1", target=END)],
        entry_node="n1",
    )


def _build_false_branch_scenario(name: str = "false_branch_scenario"):
    from karenina.schemas.entities import Question
    from karenina.schemas.primitives import BooleanMatch
    from karenina.schemas.scenario.definition import ScenarioDefinition
    from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode, StateCheck

    question = Question(
        question="what?",
        raw_answer="y",
        answer_template="class Answer: pass",
    )
    return ScenarioDefinition(
        name=name,
        nodes={
            "n1": ScenarioNode(node_id="n1", question=question),
            "n2": ScenarioNode(node_id="n2", question=question),
        },
        edges=[
            ScenarioEdge(
                source="n1",
                target="n2",
                condition=StateCheck(field="verify_result", expected=False, verify_with=BooleanMatch()),
            ),
            ScenarioEdge(source="n2", target=END),
        ],
        entry_node="n1",
    )


def _make_config() -> MagicMock:
    config = MagicMock()
    config.replay_store = None
    config.replicate_count = 1
    config.request_timeout = None
    config.evaluation_mode = "template_only"
    config.scenario_turn_limit = 5
    config.custom_error_patterns = None
    config.use_full_trace_for_template = False
    config.use_full_trace_for_rubric = False
    return config


@pytest.mark.unit
class TestScenarioManagerFailureMigration:
    """ScenarioManager.run reads metadata.failure to decide error/break."""

    def test_turn_with_failure_sets_error_status_and_logs_failure_fields(self, monkeypatch, caplog) -> None:
        from karenina.scenario import manager as mgr_mod

        failure = Failure(
            category=FailureCategory.TIMEOUT,
            stage="generate_answer",
            reason="timed out after 120s",
        )

        def fake_run_turn(self, **kwargs):
            vr = MagicMock()
            vr.metadata.failure = failure
            vr.metadata.replicate = kwargs.get("replicate")
            vr.metadata.result_id = "rid_err"
            vr.template.verify_result = False
            vr.rubric = None
            return (vr, [], None, None)

        monkeypatch.setattr(mgr_mod.ScenarioManager, "_run_turn", fake_run_turn, raising=True)

        manager = mgr_mod.ScenarioManager()
        scenario = _build_single_node_scenario()
        caplog.set_level(logging.ERROR, logger="karenina.scenario.manager")
        result = manager.run(
            scenario=scenario,
            config=_make_config(),
            base_answering_model=MagicMock(id="m", model_name="m", system_prompt="", request_timeout=None),
            base_parsing_model=MagicMock(id="p", model_name="p", system_prompt="", request_timeout=None),
        )

        assert result.status == "error"
        assert result.terminal_failure is not None
        assert result.terminal_failure.node_id == "n1"
        assert result.terminal_failure.category == "timeout"
        assert result.terminal_failure.stage == "generate_answer"
        assert result.terminal_failure.reason == "timed out after 120s"
        # Manager logs the failure reason and category (not the legacy fields).
        log_messages = [record.getMessage() for record in caplog.records]
        assert any("timed out after 120s" in msg for msg in log_messages)
        assert any("timeout" in msg for msg in log_messages)

    def test_turn_with_failure_none_completes_without_error_branch(self, monkeypatch) -> None:
        from karenina.scenario import manager as mgr_mod

        def fake_run_turn(self, **kwargs):
            vr = MagicMock()
            vr.metadata.failure = None
            vr.metadata.replicate = kwargs.get("replicate")
            vr.metadata.result_id = "rid_ok"
            vr.template.verify_result = True
            vr.rubric = None
            return (vr, [], None, None)

        monkeypatch.setattr(mgr_mod.ScenarioManager, "_run_turn", fake_run_turn, raising=True)

        manager = mgr_mod.ScenarioManager()
        scenario = _build_single_node_scenario()
        result = manager.run(
            scenario=scenario,
            config=_make_config(),
            base_answering_model=MagicMock(id="m", model_name="m", system_prompt="", request_timeout=None),
            base_parsing_model=MagicMock(id="p", model_name="p", system_prompt="", request_timeout=None),
        )

        assert result.status == "completed"

    def test_content_failure_routes_on_false_verify_result(self, monkeypatch) -> None:
        from karenina.scenario import manager as mgr_mod

        content_failure = Failure(
            category=FailureCategory.CONTENT,
            stage="verify_template",
            reason="verify_template returned False",
        )
        calls = iter([(False, content_failure, "rid_false"), (True, None, "rid_true")])

        def fake_run_turn(self, **kwargs):
            verify_result, failure, result_id = next(calls)
            vr = MagicMock()
            vr.metadata.failure = failure
            vr.metadata.replicate = kwargs.get("replicate")
            vr.metadata.result_id = result_id
            vr.template.verify_result = verify_result
            vr.rubric = None
            return (vr, [], None, None)

        monkeypatch.setattr(mgr_mod.ScenarioManager, "_run_turn", fake_run_turn, raising=True)

        manager = mgr_mod.ScenarioManager()
        result = manager.run(
            scenario=_build_false_branch_scenario(),
            config=_make_config(),
            base_answering_model=MagicMock(id="m", model_name="m", system_prompt="", request_timeout=None),
            base_parsing_model=MagicMock(id="p", model_name="p", system_prompt="", request_timeout=None),
        )

        assert result.status == "completed"
        assert result.terminal_failure is None
        assert result.path == ["n1", "n2"]
        assert [turn.verify_result for turn in result.history] == [False, True]
