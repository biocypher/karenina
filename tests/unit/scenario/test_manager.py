"""Tests for ScenarioManager core turn loop and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from karenina.scenario.manager import (
    ScenarioManager,
    _evaluate_outcome_criteria,
    _resolve_models,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.state import ScenarioExecutionResult, ScenarioState
from karenina.schemas.scenario.types import (
    ModelOverride,
    ScenarioNode,
    ScenarioOutcomeCriterion,
)

if TYPE_CHECKING:
    from karenina.schemas.entities import Question


def _make_question(text: str = "What?") -> Question:
    from karenina.schemas.entities import Question

    return Question(question=text, raw_answer="Y", answer_template="class Answer: pass")


def _make_model(
    name: str = "claude",
    provider: str = "anthropic",
    model_id: str | None = None,
) -> ModelConfig:
    return ModelConfig(id=model_id or name, model_name=name, model_provider=provider)


# ---------------------------------------------------------------------------
# ScenarioManager instantiation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScenarioManagerInit:
    def test_creates_manager(self):
        manager = ScenarioManager()
        assert manager is not None


# ---------------------------------------------------------------------------
# _resolve_models
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelResolution:
    def test_no_override_uses_base(self):
        q = _make_question()
        node = ScenarioNode(node_id="ask", question=q)
        base_ans = _make_model("claude")
        base_parse = _make_model("claude")
        ans, parse = _resolve_models(node, base_ans, base_parse)
        assert ans.model_name == "claude"
        assert parse.model_name == "claude"

    def test_override_answering(self):
        q = _make_question()
        override = ModelOverride(
            answering_model=_make_model("gpt-4o", "openai"),
        )
        node = ScenarioNode(node_id="ask", question=q, model_override=override)
        base_ans = _make_model("claude")
        base_parse = _make_model("claude")
        ans, parse = _resolve_models(node, base_ans, base_parse)
        assert ans.model_name == "gpt-4o"
        assert parse.model_name == "claude"

    def test_override_parsing(self):
        q = _make_question()
        override = ModelOverride(
            parsing_model=_make_model("gpt-4o-mini", "openai"),
        )
        node = ScenarioNode(node_id="ask", question=q, model_override=override)
        base_ans = _make_model("claude")
        base_parse = _make_model("claude")
        ans, parse = _resolve_models(node, base_ans, base_parse)
        assert ans.model_name == "claude"
        assert parse.model_name == "gpt-4o-mini"

    def test_override_both(self):
        q = _make_question()
        override = ModelOverride(
            answering_model=_make_model("gpt-4o", "openai"),
            parsing_model=_make_model("gpt-4o-mini", "openai"),
        )
        node = ScenarioNode(node_id="ask", question=q, model_override=override)
        base_ans = _make_model("claude")
        base_parse = _make_model("claude")
        ans, parse = _resolve_models(node, base_ans, base_parse)
        assert ans.model_name == "gpt-4o"
        assert parse.model_name == "gpt-4o-mini"

    def test_empty_override_object_uses_base(self):
        q = _make_question()
        override = ModelOverride()  # neither field set
        node = ScenarioNode(node_id="ask", question=q, model_override=override)
        base_ans = _make_model("claude")
        base_parse = _make_model("claude")
        ans, parse = _resolve_models(node, base_ans, base_parse)
        assert ans.model_name == "claude"
        assert parse.model_name == "claude"


# ---------------------------------------------------------------------------
# _evaluate_outcome_criteria
# ---------------------------------------------------------------------------


def _make_execution_result() -> ScenarioExecutionResult:
    state = ScenarioState(
        turn=2,
        current_node="done",
        verify_result=True,
        parsed={},
        node_visits={"ask": 1, "done": 1},
        history=[],
        accumulated={},
        node_results={},
    )
    return ScenarioExecutionResult(
        scenario_id="test",
        status="completed",
        path=["ask", "done"],
        turn_count=2,
        history=[],
        turn_results=[],
        final_state=state,
        outcome_results={},
    )


def _make_scenario_with_criteria(
    criteria: list[ScenarioOutcomeCriterion],
) -> ScenarioDefinition:
    q = _make_question()
    return ScenarioDefinition(
        name="test",
        nodes={"ask": ScenarioNode(node_id="ask", question=q)},
        edges=[],
        entry_node="ask",
        outcome_criteria=criteria,
    )


@pytest.mark.unit
class TestEvaluateOutcomeCriteria:
    def test_callable_criterion_true(self):
        criterion = ScenarioOutcomeCriterion(
            name="fast",
            description="Done quickly?",
            evaluate=lambda r: r.turn_count <= 3,
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["fast"] is True

    def test_callable_criterion_false(self):
        criterion = ScenarioOutcomeCriterion(
            name="slow",
            description="Took more than 5 turns?",
            evaluate=lambda r: r.turn_count > 5,
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["slow"] is False

    def test_callable_exception_returns_false(self):
        criterion = ScenarioOutcomeCriterion(
            name="broken",
            description="Always fails",
            evaluate=lambda _r: 1 / 0,
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["broken"] is False

    def test_declarative_check_dispatches_to_evaluate_outcome(self):
        """Declarative check (criterion.check is not None) dispatches to evaluate_outcome."""
        from karenina.schemas.primitives import ExactMatch
        from karenina.schemas.scenario.checks import ResultCheck

        check_node = ResultCheck(
            field="status",
            expected="completed",
            verify_with=ExactMatch(),
        )
        criterion = ScenarioOutcomeCriterion(
            name="completed_status",
            description="Scenario completed successfully",
            check=check_node,
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["completed_status"] is True

    def test_declarative_check_false_result(self):
        """Declarative check returns False when condition is not met."""
        from karenina.schemas.primitives import ExactMatch
        from karenina.schemas.scenario.checks import ResultCheck

        check_node = ResultCheck(
            field="status",
            expected="error",
            verify_with=ExactMatch(),
        )
        criterion = ScenarioOutcomeCriterion(
            name="errored",
            description="Scenario errored",
            check=check_node,
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["errored"] is False

    def test_multiple_criteria(self):
        criteria = [
            ScenarioOutcomeCriterion(
                name="fast",
                description="Done quickly?",
                evaluate=lambda r: r.turn_count <= 3,
            ),
            ScenarioOutcomeCriterion(
                name="correct_path",
                description="Correct path?",
                evaluate=lambda r: "ask" in r.path,
            ),
        ]
        scenario = _make_scenario_with_criteria(criteria)
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes["fast"] is True
        assert outcomes["correct_path"] is True

    def test_no_criteria_returns_empty(self):
        scenario = _make_scenario_with_criteria([])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert outcomes == {}

    def test_criterion_with_neither_check_nor_evaluate(self):
        """Criterion with neither check nor evaluate logs a warning."""
        criterion = ScenarioOutcomeCriterion(
            name="empty",
            description="No evaluation defined",
        )
        scenario = _make_scenario_with_criteria([criterion])
        result = _make_execution_result()
        outcomes = _evaluate_outcome_criteria(scenario, result)
        assert "empty" not in outcomes


# ---------------------------------------------------------------------------
# ScenarioManager._report_progress
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReportProgress:
    def test_callback_invoked_with_kwargs(self):
        captured = {}

        def cb(**kwargs):
            captured.update(kwargs)

        ScenarioManager._report_progress(
            cb,
            "my-scenario",
            0,
            "ask",
            True,
            "done",
        )
        assert captured["scenario_id"] == "my-scenario"
        assert captured["scenario_turn"] == 0
        assert captured["scenario_node"] == "ask"
        assert captured["verify_result"] is True
        assert captured["next_node"] == "done"

    def test_none_callback_is_noop(self):
        # Should not raise
        ScenarioManager._report_progress(
            None,
            "my-scenario",
            0,
            "ask",
            True,
            "done",
        )

    def test_callback_exception_is_swallowed(self):
        def bad_cb(**kwargs):
            raise RuntimeError("boom")

        # Should not raise
        ScenarioManager._report_progress(
            bad_cb,
            "my-scenario",
            0,
            "ask",
            True,
            None,
        )


# ---------------------------------------------------------------------------
# ScenarioManager._run_turn: per-question rubric stamping
# ---------------------------------------------------------------------------


def _make_pipeline_config_with_retry(
    request_timeout: float = 600.0,
    timeout_max_attempts: int = 5,
):
    """Build a VerificationConfig with non-default timeout and retry policy."""
    from karenina.schemas.verification.config import VerificationConfig
    from karenina.utils.retry_policy import (
        CategoryRetryConfig,
        RetryPolicy,
        TimeoutEscalationConfig,
    )

    retry_policy = RetryPolicy(
        timeout=CategoryRetryConfig(
            max_attempts=timeout_max_attempts,
            backoff_min=2.0,
            backoff_max=20.0,
        ),
        timeout_escalation=TimeoutEscalationConfig(
            strategy="additive",
            increment=120.0,
            max_timeout=1200.0,
        ),
    )
    return VerificationConfig(
        answering_models=[_make_model("base_ans")],
        parsing_models=[_make_model("base_parse")],
        request_timeout=request_timeout,
        retry_policy=retry_policy,
        evaluation_mode="template_and_rubric",
    )


def _make_deep_agent_override(model_id: str = "guard_agent"):
    """Build a ModelConfig with claude_agent_sdk interface for AgenticRubricTrait.model_override."""
    return ModelConfig(
        id=model_id,
        model_name=model_id,
        model_provider="anthropic",
        interface="claude_agent_sdk",
    )


@pytest.mark.unit
class TestPerQuestionRubricAgenticStamping:
    """Verify that per-question rubrics on scenario nodes get agentic trait stamping.

    The benchmark facade stamps the global rubric before passing it to the
    scenario executor, but per-question rubrics live on each Question and are
    materialized by ScenarioManager._run_turn. They must also receive the same
    pipeline-level retry_policy and request_timeout stamping for any
    AgenticRubricTrait.model_override they carry.
    """

    def _build_node_with_question_rubric(self, override):
        from karenina.schemas.entities import Question
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric

        agentic = AgenticRubricTrait(
            name="investigate",
            description="Investigate the trace",
            kind="boolean",
            higher_is_better=True,
            model_override=override,
        )
        rubric = Rubric(agentic_traits=[agentic])
        question = Question(
            question="Is it safe?",
            raw_answer="yes",
            answer_template="class Answer: pass",
            question_rubric=rubric.model_dump(),
        )
        return ScenarioNode(node_id="ask", question=question), rubric

    def test_per_question_rubric_agentic_override_is_stamped(self, monkeypatch):
        """A scenario node carrying a per-question rubric with an agentic
        trait override receives stamping during _run_turn."""
        from karenina.benchmark.verification.stages.core.base import VerificationContext
        from karenina.benchmark.verification.stages.core.orchestrator import StageOrchestrator

        config = _make_pipeline_config_with_retry(request_timeout=600.0, timeout_max_attempts=5)
        override = _make_deep_agent_override()
        node, _original_rubric = self._build_node_with_question_rubric(override)

        captured: dict[str, object] = {}

        def fake_execute(self, context: VerificationContext):
            captured["context_rubric"] = context.rubric
            from karenina.schemas.verification import VerificationResult, VerificationResultMetadata
            from karenina.schemas.verification.model_identity import ModelIdentity

            ans_id = ModelIdentity.from_model_config(context.answering_model, role="answering")
            parse_id = ModelIdentity.from_model_config(context.parsing_model, role="parsing")
            return VerificationResult(
                metadata=VerificationResultMetadata(
                    question_id=context.question_id,
                    template_id=context.template_id,
                    completed_without_errors=True,
                    question_text=context.question_text,
                    answering=ans_id,
                    parsing=parse_id,
                    execution_time=0.0,
                    timestamp="2026-01-01T00:00:00",
                    result_id="test",
                )
            )

        monkeypatch.setattr(StageOrchestrator, "execute", fake_execute)

        manager = ScenarioManager()
        manager._run_turn(
            node=node,
            conversation_history=[],
            answering_model=_make_model("base_ans"),
            parsing_model=_make_model("base_parse"),
            config=config,
            run_name=None,
            global_rubric=None,
        )

        passed_rubric = captured["context_rubric"]
        assert passed_rubric is not None
        assert len(passed_rubric.agentic_traits) == 1
        passed_override = passed_rubric.agentic_traits[0].model_override
        assert passed_override is not None
        # Stamping populated the unset fields from the pipeline config
        assert passed_override.request_timeout == 600.0
        assert passed_override.retry_policy is not None
        assert passed_override.retry_policy.timeout.max_attempts == 5
        assert passed_override.retry_policy.timeout_escalation is not None
        assert passed_override.retry_policy.timeout_escalation.strategy == "additive"

        # The original override on the rubric we built locally is unchanged
        # (it was deep-copied through model_dump round-trip on the question,
        # so we instead assert on the question_rubric dict identity).
        assert override.request_timeout is None
        assert override.retry_policy is None

    def test_per_question_rubric_explicit_fields_preserved(self, monkeypatch):
        """Explicit request_timeout / retry_policy on a per-question rubric
        override are preserved (not overwritten by the pipeline config)."""
        from karenina.benchmark.verification.stages.core.base import VerificationContext
        from karenina.benchmark.verification.stages.core.orchestrator import StageOrchestrator
        from karenina.schemas.entities import Question
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric
        from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy

        explicit_policy = RetryPolicy(
            timeout=CategoryRetryConfig(max_attempts=9, backoff_min=1.0, backoff_max=2.0),
        )
        explicit_override = ModelConfig(
            id="guard_agent",
            model_name="guard_agent",
            model_provider="anthropic",
            interface="claude_agent_sdk",
            request_timeout=42.0,
            retry_policy=explicit_policy,
        )
        agentic = AgenticRubricTrait(
            name="investigate",
            description="Investigate the trace",
            kind="boolean",
            higher_is_better=True,
            model_override=explicit_override,
        )
        rubric = Rubric(agentic_traits=[agentic])
        question = Question(
            question="Is it safe?",
            raw_answer="yes",
            answer_template="class Answer: pass",
            question_rubric=rubric.model_dump(),
        )
        node = ScenarioNode(node_id="ask", question=question)

        config = _make_pipeline_config_with_retry(request_timeout=600.0, timeout_max_attempts=5)

        captured: dict[str, object] = {}

        def fake_execute(self, context: VerificationContext):
            captured["context_rubric"] = context.rubric
            from karenina.schemas.verification import VerificationResult, VerificationResultMetadata
            from karenina.schemas.verification.model_identity import ModelIdentity

            ans_id = ModelIdentity.from_model_config(context.answering_model, role="answering")
            parse_id = ModelIdentity.from_model_config(context.parsing_model, role="parsing")
            return VerificationResult(
                metadata=VerificationResultMetadata(
                    question_id=context.question_id,
                    template_id=context.template_id,
                    completed_without_errors=True,
                    question_text=context.question_text,
                    answering=ans_id,
                    parsing=parse_id,
                    execution_time=0.0,
                    timestamp="2026-01-01T00:00:00",
                    result_id="test",
                )
            )

        monkeypatch.setattr(StageOrchestrator, "execute", fake_execute)

        manager = ScenarioManager()
        manager._run_turn(
            node=node,
            conversation_history=[],
            answering_model=_make_model("base_ans"),
            parsing_model=_make_model("base_parse"),
            config=config,
            run_name=None,
            global_rubric=None,
        )

        passed_rubric = captured["context_rubric"]
        assert passed_rubric is not None
        passed_override = passed_rubric.agentic_traits[0].model_override
        assert passed_override is not None
        # Explicit values preserved
        assert passed_override.request_timeout == 42.0
        assert passed_override.retry_policy is not None
        assert passed_override.retry_policy.timeout.max_attempts == 9
        # Confirm not replaced by pipeline default
        assert passed_override.retry_policy.timeout.max_attempts != 5

    def test_global_rubric_used_when_no_per_question_rubric(self, monkeypatch):
        """When the node has no per-question rubric, global_rubric is used as-is."""
        from karenina.benchmark.verification.stages.core.base import VerificationContext
        from karenina.benchmark.verification.stages.core.orchestrator import StageOrchestrator
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        config = _make_pipeline_config_with_retry()
        global_rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="quality",
                    description="A quality trait",
                    kind="boolean",
                    higher_is_better=True,
                )
            ]
        )

        node = ScenarioNode(node_id="ask", question=_make_question())

        captured: dict[str, object] = {}

        def fake_execute(self, context: VerificationContext):
            captured["context_rubric"] = context.rubric
            from karenina.schemas.verification import VerificationResult, VerificationResultMetadata
            from karenina.schemas.verification.model_identity import ModelIdentity

            ans_id = ModelIdentity.from_model_config(context.answering_model, role="answering")
            parse_id = ModelIdentity.from_model_config(context.parsing_model, role="parsing")
            return VerificationResult(
                metadata=VerificationResultMetadata(
                    question_id=context.question_id,
                    template_id=context.template_id,
                    completed_without_errors=True,
                    question_text=context.question_text,
                    answering=ans_id,
                    parsing=parse_id,
                    execution_time=0.0,
                    timestamp="2026-01-01T00:00:00",
                    result_id="test",
                )
            )

        monkeypatch.setattr(StageOrchestrator, "execute", fake_execute)

        manager = ScenarioManager()
        manager._run_turn(
            node=node,
            conversation_history=[],
            answering_model=_make_model("base_ans"),
            parsing_model=_make_model("base_parse"),
            config=config,
            run_name=None,
            global_rubric=global_rubric,
        )

        # global_rubric is passed through unchanged (identity)
        assert captured["context_rubric"] is global_rubric
