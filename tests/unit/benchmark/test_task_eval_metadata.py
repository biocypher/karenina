"""Tests for TaskEval metadata coherence fixes (issues 024, 114, 115, 117, 160, 165, 166, 168, 179)."""

from typing import Any

import pytest

from karenina.benchmark.task_eval import StepEval, TaskEval
from karenina.exceptions import KareninaError
from karenina.ports.messages import Message
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


@pytest.mark.unit
class TestTaskEvalInterface:
    """Issue 166: the taskeval interface must be registered in AdapterRegistry.

    The previous version of this test only round-tripped ``interface='taskeval'``
    through ``ModelConfig`` and asserted the value came back out — a tautology
    since Pydantic always echoes its inputs. That test would have stayed green
    even if the registration module were deleted and TaskEval evaluations
    started failing at adapter construction. These tests instead verify the
    behavior that actually matters downstream: the ``taskeval`` interface is
    registered in ``AdapterRegistry`` (via the import side-effect in
    ``karenina.adapters.taskeval.registration``) and is reported as available.
    """

    def test_taskeval_interface_is_registered(self):
        """AdapterRegistry must expose the taskeval interface spec.

        If the registration module is deleted, the spec goes unregistered, or
        the entry-point discovery chain stops importing it, this assertion
        fails — surfacing the breakage before any TaskEval evaluation runs.
        """
        from karenina.adapters.registry import AdapterRegistry

        spec = AdapterRegistry.get_spec("taskeval")
        assert spec is not None, "taskeval interface is not registered with AdapterRegistry"
        assert spec.interface == "taskeval"

    def test_taskeval_interface_is_available(self):
        """check_availability must mark taskeval as available (no provider required).

        TaskEval always uses pre-collected outputs, so the interface must be
        available regardless of environment / credentials. A regression that
        adds an availability gate (e.g. requiring a provider) would break
        every TaskEval evaluation; this test catches that.
        """
        from karenina.adapters.registry import AdapterRegistry

        availability = AdapterRegistry.check_availability("taskeval")
        assert availability.available is True, (
            f"taskeval must always be available; got available=False ({availability.reason!r})"
        )

    def test_taskeval_interface_does_not_route_or_fallback(self):
        """taskeval is a terminal interface (no fallback / no routing).

        If someone adds ``fallback_interface='langchain'`` to the spec, the
        framework would silently substitute langchain for taskeval when an
        adapter is built — corrupting TaskEval's no-LLM-invocation contract.
        """
        from karenina.adapters.registry import AdapterRegistry

        spec = AdapterRegistry.get_spec("taskeval")
        assert spec is not None
        assert spec.fallback_interface is None, f"taskeval must not declare a fallback; got {spec.fallback_interface!r}"
        assert spec.routes_to is None, f"taskeval must not route to another interface; got {spec.routes_to!r}"
        assert spec.requires_provider is False, "taskeval must not require a provider (outputs are pre-collected)"


@pytest.mark.unit
class TestStepEvalFailedQuestions:
    """Issue 117: StepEval tracks failed questions."""

    def test_failed_questions_defaults_empty(self):
        """failed_questions defaults to empty dict."""
        step = StepEval()
        assert step.failed_questions == {}

    def test_failed_questions_records_errors(self):
        """failed_questions stores error messages by question_id."""
        step = StepEval()
        step.failed_questions["q1"] = ["Some error"]
        step.failed_questions["q1"].append("Another error")
        assert len(step.failed_questions["q1"]) == 2
        assert step.failed_questions["q1"][0] == "Some error"


def _make_metadata(question_id: str = "test_q") -> VerificationResultMetadata:
    """Helper to create test metadata."""
    return VerificationResultMetadata(
        question_id=question_id,
        template_id="test_t",
        failure=None,
        question_text="test question",
        answering=ModelIdentity(interface="mock", model_name="mock"),
        parsing=ModelIdentity(interface="mock", model_name="mock"),
        execution_time=0.1,
        timestamp="2026-01-01T00:00:00",
        result_id="abcd1234abcd1234",
    )


@pytest.mark.unit
class TestSuccessRateRubricOnly:
    """Issue 024: success_rate should not be zero in rubric_only mode when traits pass."""

    def test_rubric_only_all_pass(self):
        """success_rate > 0 when all rubric traits pass and no template verification."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"trait_a": True, "trait_b": True},
        )
        template = VerificationResultTemplate(
            template_verification_performed=False,
            verify_result=None,
        )
        result = VerificationResult(
            metadata=_make_metadata(),
            template=template,
            rubric=rubric,
        )
        step = StepEval(verification_results={"q1": [result]})
        stats = step.get_summary_stats()
        assert stats["success_rate"] == 100.0
        assert stats["traces_passed"] == 1

    def test_rubric_only_some_fail(self):
        """success_rate == 0 when rubric traits fail."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"trait_a": True, "trait_b": False},
        )
        template = VerificationResultTemplate(
            template_verification_performed=False,
            verify_result=None,
        )
        result = VerificationResult(
            metadata=_make_metadata(),
            template=template,
            rubric=rubric,
        )
        step = StepEval(verification_results={"q1": [result]})
        stats = step.get_summary_stats()
        assert stats["success_rate"] == 0.0
        assert stats["traces_passed"] == 0

    def test_template_mode_unchanged(self):
        """Template-verified traces still use verify_result for pass counting."""
        template = VerificationResultTemplate(
            template_verification_performed=True,
            verify_result=True,
        )
        result = VerificationResult(
            metadata=_make_metadata(),
            template=template,
        )
        step = StepEval(verification_results={"q1": [result]})
        stats = step.get_summary_stats()
        assert stats["success_rate"] == 100.0


@pytest.mark.unit
class TestAvailableStepIds:
    """Issue 115: _get_available_step_ids() includes dynamic rubric steps."""

    def test_dynamic_rubric_only_step_discovered(self):
        """A step with only a dynamic rubric should appear in available step IDs."""
        from karenina.schemas.entities.rubric import DynamicRubric, LLMRubricTrait

        task = TaskEval()
        task.add_dynamic_rubric(
            DynamicRubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="dynamic_trait",
                        summary="check",
                        description="Dynamic check",
                        kind="boolean",
                    )
                ]
            ),
            step_id="dynamic_step",
        )
        step_ids = task._get_available_step_ids()
        assert "dynamic_step" in step_ids


# ---------------------------------------------------------------------------
# Helpers for TestParameterThreading
# ---------------------------------------------------------------------------


def _make_mock_result():
    """Create a minimal mock VerificationResult."""
    return VerificationResult(metadata=_make_metadata())


def _make_config(**overrides) -> VerificationConfig:
    """Create a test VerificationConfig."""
    defaults = {
        "parsing_models": [
            ModelConfig(
                id="mock",
                model_provider="mock",
                model_name="mock",
                interface="langchain",
            )
        ],
        "parsing_only": True,
    }
    defaults.update(overrides)
    return VerificationConfig(**defaults)


@pytest.mark.unit
class TestParameterThreading:
    """Issues 114, 160, 165, 166, 168: parameters reach run_single_model_verification."""

    def _capture_calls(self, monkeypatch) -> list[dict[str, Any]]:
        """Monkeypatch runner and return captured kwargs list."""
        captured: list[dict[str, Any]] = []

        def mock_run(*args, **kwargs):
            captured.append(kwargs)
            return _make_mock_result()

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )
        return captured

    def _setup_rubric_task(self):
        """Create a TaskEval with a rubric and trace."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )
        return task

    def test_sentinel_model_used_by_default(self, monkeypatch):
        """Issue 166: sentinel answering_model with interface='taskeval' when none provided."""
        captured = self._capture_calls(monkeypatch)
        task = self._setup_rubric_task()
        task.evaluate(_make_config())

        assert len(captured) > 0
        model = captured[0]["answering_model"]
        assert model.interface == "taskeval"
        assert model.model_name == "user-provided"
        assert model.model_provider == "user-provided"

    def test_custom_answering_model_passed_through(self, monkeypatch):
        """Issue 166: user-provided answering_model reaches the runner."""
        captured = self._capture_calls(monkeypatch)
        task = self._setup_rubric_task()
        custom_model = ModelConfig(
            id="gpt-4",
            model_provider="openai",
            model_name="gpt-4",
            interface="langchain",
        )
        task.evaluate(_make_config(), answering_model=custom_model)

        assert captured[0]["answering_model"].model_name == "gpt-4"

    def test_run_name_auto_generated(self, monkeypatch):
        """Issue 168: auto-generated run_name reaches the runner."""
        captured = self._capture_calls(monkeypatch)
        task = self._setup_rubric_task()
        task.evaluate(_make_config())

        assert captured[0]["run_name"] is not None
        assert captured[0]["run_name"].startswith("taskeval_")

    def test_run_name_explicit(self, monkeypatch):
        """Issue 168: explicit run_name reaches the runner."""
        captured = self._capture_calls(monkeypatch)
        task = self._setup_rubric_task()
        task.evaluate(_make_config(), run_name="my_run")

        assert captured[0]["run_name"] == "my_run"

    def test_replicate_index_threaded(self, monkeypatch):
        """Issue 114: replicate index reaches the runner when replicate_count > 1."""
        captured = self._capture_calls(monkeypatch)
        task = self._setup_rubric_task()
        config = _make_config(replicate_count=3)
        task.evaluate(config)

        assert len(captured) == 3
        assert captured[0]["replicate"] == 1
        assert captured[1]["replicate"] == 2
        assert captured[2]["replicate"] == 3

    def test_replicate_none_when_single(self, monkeypatch):
        """Issue 114: replicate is None when replicate_count is 1."""
        captured = self._capture_calls(monkeypatch)
        task = self._setup_rubric_task()
        task.evaluate(_make_config())

        assert captured[0]["replicate"] is None

    def test_guard_flags_from_config(self, monkeypatch):
        """Issue 160: abstention_enabled and sufficiency_enabled from config."""
        captured = self._capture_calls(monkeypatch)
        task = self._setup_rubric_task()
        config = _make_config(abstention_enabled=True, sufficiency_enabled=True)
        task.evaluate(config)

        assert captured[0]["abstention_enabled"] is True
        assert captured[0]["sufficiency_enabled"] is True

    def test_guard_flags_default_false(self, monkeypatch):
        """Issue 160: guard flags default to False."""
        captured = self._capture_calls(monkeypatch)
        task = self._setup_rubric_task()
        task.evaluate(_make_config())

        assert captured[0]["abstention_enabled"] is False
        assert captured[0]["sufficiency_enabled"] is False

    def test_question_id_not_force_hashed(self, monkeypatch):
        """Issue 165: non-MD5 question IDs pass through without hashing."""
        captured = self._capture_calls(monkeypatch)
        task = TaskEval(task_id="test")
        task.log("Some output")
        # Question with human-readable ID and an answer template
        task.add_question(
            {
                "id": "my_readable_id",
                "question": "What?",
                "answer_template": """
from karenina.schemas.entities import BaseAnswer
class Answer(BaseAnswer):
    value: str = ""
    def verify(self) -> bool:
        return True
""",
            }
        )
        task.evaluate(_make_config())

        assert captured[0]["question_id"] == "my_readable_id"


@pytest.mark.unit
class TestExceptionNarrowing:
    """Issue 117: _evaluate_and_store() narrows exception scope."""

    def _capture_calls(self, monkeypatch):
        """Monkeypatch runner to raise a specific exception."""

        def _set_exception(exc):
            def mock_run(*args, **kwargs):
                raise exc

            monkeypatch.setattr(
                "karenina.benchmark.verification.runner.run_single_model_verification",
                mock_run,
            )

        return _set_exception

    def _setup_rubric_task(self):
        """Create a TaskEval with a rubric and trace."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )
        return task

    def test_programming_error_propagates(self, monkeypatch):
        """TypeError should NOT be caught."""
        set_exc = self._capture_calls(monkeypatch)
        set_exc(TypeError("This is a programming bug"))
        task = self._setup_rubric_task()
        with pytest.raises(TypeError, match="programming bug"):
            task.evaluate(_make_config())

    def test_attribute_error_propagates(self, monkeypatch):
        """AttributeError should NOT be caught."""
        set_exc = self._capture_calls(monkeypatch)
        set_exc(AttributeError("missing attribute"))
        task = self._setup_rubric_task()
        with pytest.raises(AttributeError, match="missing attribute"):
            task.evaluate(_make_config())

    def test_domain_error_caught_and_recorded(self, monkeypatch):
        """KareninaError should be caught and recorded in failed_questions."""
        set_exc = self._capture_calls(monkeypatch)
        set_exc(KareninaError("Evaluation pipeline failed"))
        task = self._setup_rubric_task()
        result = task.evaluate(_make_config())
        step = result.global_eval
        assert step is not None
        assert len(step.failed_questions) > 0
        failed = list(step.failed_questions.values())[0]
        assert "Evaluation pipeline failed" in failed[0]

    def test_value_error_caught_and_recorded(self, monkeypatch):
        """ValueError should be caught and recorded in failed_questions."""
        set_exc = self._capture_calls(monkeypatch)
        set_exc(ValueError("Invalid template"))
        task = self._setup_rubric_task()
        result = task.evaluate(_make_config())
        step = result.global_eval
        assert step is not None
        assert len(step.failed_questions) > 0


@pytest.mark.unit
class TestFewShotConfigWarning:
    """Issue 179: FewShotConfig warning in TaskEval mode."""

    def test_few_shot_warning_emitted(self, monkeypatch, caplog):
        """Debug log emitted when FewShotConfig is enabled."""
        import logging

        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        def mock_run(*args, **kwargs):
            return _make_mock_result()

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )

        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("response")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(name="q", description="check", kind="boolean"),
                ]
            )
        )

        from karenina.schemas.config import FewShotConfig

        config = _make_config(few_shot_config=FewShotConfig(source="both"))

        with caplog.at_level(logging.DEBUG, logger="karenina.benchmark.task_eval.task_eval"):
            task.evaluate(config)

        assert any("FewShotConfig has no effect in TaskEval mode" in m for m in caplog.messages)


@pytest.mark.unit
class TestTemplateIdNormalization:
    """Issue 167: generate_template_id() produces consistent hashes regardless of indentation."""

    def test_same_hash_different_indentation(self):
        """Same template at different indentation levels produces the same hash."""
        from karenina.utils.checkpoint import generate_template_id

        # Module-level (no extra indentation)
        template_module = """from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    value: str = ""
    def verify(self) -> bool:
        return True
"""

        # Embedded in a string literal (extra indentation)
        template_embedded = """
            from karenina.schemas.entities import BaseAnswer

            class Answer(BaseAnswer):
                value: str = ""
                def verify(self) -> bool:
                    return True
            """

        assert generate_template_id(template_module) == generate_template_id(template_embedded)

    def test_different_templates_different_hash(self):
        """Semantically different templates still produce different hashes."""
        from karenina.utils.checkpoint import generate_template_id

        template_a = "class A:\n    def verify(self): return True"
        template_b = "class B:\n    def verify(self): return False"
        assert generate_template_id(template_a) != generate_template_id(template_b)

    def test_none_returns_no_template(self):
        """None input returns 'no_template' sentinel."""
        from karenina.utils.checkpoint import generate_template_id

        assert generate_template_id(None) == "no_template"

    def test_empty_returns_no_template(self):
        """Empty/whitespace-only returns 'no_template' sentinel."""
        from karenina.utils.checkpoint import generate_template_id

        assert generate_template_id("") == "no_template"
        assert generate_template_id("   \n  \n  ") == "no_template"

    def test_line_ending_normalization(self):
        """Windows vs Unix line endings produce the same hash."""
        from karenina.utils.checkpoint import generate_template_id

        unix = "class A:\n    pass\n"
        windows = "class A:\r\n    pass\r\n"
        assert generate_template_id(unix) == generate_template_id(windows)


def _make_metadata_with_failure(question_id: str = "test_q") -> VerificationResultMetadata:
    """Helper to create test metadata that carries a Failure object."""
    from karenina.schemas.results.failure import Failure, FailureCategory

    return VerificationResultMetadata(
        question_id=question_id,
        template_id="test_t",
        failure=Failure(category=FailureCategory.TIMEOUT, stage="generate_answer", reason="timeout"),
        question_text="test question",
        answering=ModelIdentity(interface="mock", model_name="mock"),
        parsing=ModelIdentity(interface="mock", model_name="mock"),
        execution_time=0.1,
        timestamp="2026-01-01T00:00:00",
        result_id="abcd1234abcd1234",
    )


@pytest.mark.unit
class TestStepEvalAggregationByFailure:
    """StepEval aggregation filters replicates via ``metadata.failure is None``."""

    def test_aggregate_excludes_failed_replicates(self):
        """A replicate with a non-None Failure is excluded from aggregation."""
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={"clarity": 4},
        )
        template = VerificationResultTemplate(
            template_verification_performed=True,
            verify_result=True,
        )
        passing = VerificationResult(metadata=_make_metadata("q1"), template=template, rubric=rubric)
        failing = VerificationResult(
            metadata=_make_metadata_with_failure("q1"),
            template=VerificationResultTemplate(template_verification_performed=False),
            rubric=None,
        )

        step = StepEval(verification_results={"q1": [passing, failing]})
        aggregated = step.aggregate_rubric_results()

        assert "q1" in aggregated
        assert aggregated["q1"].get("failed_replicate_count") == 1
        # clarity survived from the passing replicate
        assert aggregated["q1"]["llm"]["clarity"] == 4

    def test_aggregate_all_failed_returns_failed_count(self):
        """If all replicates fail, aggregation returns only the failed count."""
        failing_a = VerificationResult(
            metadata=_make_metadata_with_failure("q2"),
            template=VerificationResultTemplate(template_verification_performed=False),
            rubric=None,
        )
        failing_b = VerificationResult(
            metadata=_make_metadata_with_failure("q2"),
            template=VerificationResultTemplate(template_verification_performed=False),
            rubric=None,
        )

        step = StepEval(verification_results={"q2": [failing_a, failing_b]})
        aggregated = step.aggregate_rubric_results()

        assert aggregated["q2"] == {"failed_replicate_count": 2}
