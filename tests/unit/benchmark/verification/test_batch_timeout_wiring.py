"""Tests for the batch wall-clock timeout knob (T15).

``VerificationConfig.batch_timeout_seconds`` must reach the executor tuning
on both public paths and trigger the documented timed-out contract:

- QA path (``run_verification_batch``): raises ``VerificationBatchError``
  carrying ``partial_results`` plus a "timed out" message.
- Scenario path (``Benchmark._run_scenario_verification``): returns the
  partial result set with a TimeoutError entry in ``errors``.
- Default ``None`` keeps today's behavior (no batch-level timeout).
"""

import threading
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from karenina.benchmark.verification.batch_runner import run_verification_batch
from karenina.exceptions import VerificationBatchError
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import FinishedTemplate, VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import VerificationResultMetadata

BATCH_TIMEOUT_SECONDS = 0.6
SLOW_TASK_SECONDS = 0.8

# ============================================================================
# Helpers
# ============================================================================


def _make_identity(name: str = "test-model") -> ModelIdentity:
    """Create a minimal ModelIdentity for test results."""
    return ModelIdentity(model_name=name, interface="langchain")


def _make_result(question_id: str = "q1") -> VerificationResult:
    """Create a minimal VerificationResult for testing."""
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="test_template",
            failure=None,
            caveats=[],
            question_text="Test question",
            answering=_make_identity(),
            parsing=_make_identity(),
            execution_time=0.1,
            timestamp="2026-01-01T00:00:00",
            result_id="abcdef1234567890",
        )
    )


def _make_model(model_id: str) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        model_name=model_id,
        model_provider="openai",
        interface="langchain",
        system_prompt="test",
        temperature=0.0,
    )


def _make_template(question_id: str) -> FinishedTemplate:
    return FinishedTemplate(
        question_id=question_id,
        question_text=f"Question {question_id}",
        question_preview=f"Question {question_id}",
        template_code="class Answer(BaseAnswer): pass",
        last_modified="2026-01-01T00:00:00",
    )


def _make_config(**overrides) -> VerificationConfig:
    return VerificationConfig(
        answering_models=[_make_model("ans1")],
        parsing_models=[_make_model("parse1")],
        **overrides,
    )


# ============================================================================
# Field validation
# ============================================================================


@pytest.mark.unit
class TestBatchTimeoutFieldValidation:
    """batch_timeout_seconds accepts None and positive floats only."""

    @pytest.mark.parametrize("value", [None, 1.0, 10.5], ids=["none", "int_seconds", "fractional"])
    def test_accepts_none_or_positive_float(self, value: float | None) -> None:
        config = _make_config(batch_timeout_seconds=value)
        assert config.batch_timeout_seconds == value

    @pytest.mark.parametrize("value", [0, -1.0], ids=["zero", "negative"])
    def test_rejects_non_positive(self, value: float) -> None:
        with pytest.raises(ValidationError):
            _make_config(batch_timeout_seconds=value)


# ============================================================================
# Plumbing: config reaches both executor configs
# ============================================================================


@pytest.mark.unit
class TestBatchTimeoutPlumbing:
    """batch_timeout_seconds is forwarded to both executor config constructions."""

    @patch("karenina.benchmark.verification.executor.VerificationExecutor")
    def test_qa_executor_receives_timeout(self, MockExecutor: MagicMock) -> None:
        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = {}

        config = _make_config(batch_timeout_seconds=7.5)
        run_verification_batch(
            [_make_template("q1")],
            config,
            async_enabled=True,
            max_workers=2,
        )

        call_kwargs = MockExecutor.call_args
        executor_config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert executor_config.timeout_seconds == 7.5

    @patch("karenina.benchmark.verification.executor.VerificationExecutor")
    def test_qa_executor_default_none(self, MockExecutor: MagicMock) -> None:
        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = {}

        config = _make_config()
        run_verification_batch(
            [_make_template("q1")],
            config,
            async_enabled=True,
            max_workers=2,
        )

        call_kwargs = MockExecutor.call_args
        executor_config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert executor_config.timeout_seconds is None

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_scenario_executor_receives_timeout(self, MockExecutor: MagicMock) -> None:
        from karenina.benchmark.benchmark import Benchmark

        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = ([], [])

        benchmark = Benchmark(name="test")
        scenario = MagicMock()
        scenario.name = "s1"
        benchmark._scenarios = {"s1": scenario}

        config = _make_config(batch_timeout_seconds=42.0)
        benchmark._run_scenario_verification(config, async_enabled=True)

        call_kwargs = MockExecutor.call_args
        executor_config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert executor_config.timeout_seconds == 42.0

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_scenario_executor_default_none(self, MockExecutor: MagicMock) -> None:
        from karenina.benchmark.benchmark import Benchmark

        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = ([], [])

        benchmark = Benchmark(name="test")
        scenario = MagicMock()
        scenario.name = "s1"
        benchmark._scenarios = {"s1": scenario}

        config = _make_config()
        benchmark._run_scenario_verification(config, async_enabled=True)

        call_kwargs = MockExecutor.call_args
        executor_config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert executor_config.timeout_seconds is None


# ============================================================================
# QA path end-to-end through run_verification_batch
# ============================================================================


@pytest.mark.unit
class TestQABatchTimeoutEndToEnd:
    """A finite batch_timeout_seconds triggers the documented QA contract."""

    def test_timeout_raises_batch_error_with_partial_results(self, monkeypatch) -> None:
        """One fast task completes, one blocks past the batch timeout. The
        public entry point raises VerificationBatchError with the fast
        task's result in partial_results and a "timed out" message.
        """
        templates = [_make_template("q1"), _make_template("q2")]
        block = threading.Event()

        def mock_execute_task(task, answer_cache=None, cache_status=None, cached_answer_data=None):
            if task["question_id"] == "q2":
                block.wait(timeout=SLOW_TASK_SECONDS)
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        config = _make_config(batch_timeout_seconds=BATCH_TIMEOUT_SECONDS)

        with pytest.raises(VerificationBatchError) as exc_info:
            run_verification_batch(
                templates,
                config,
                async_enabled=True,
                max_workers=2,
            )

        err = exc_info.value
        assert "timed out" in str(err)
        assert "key_q1" in err.partial_results

    def test_none_timeout_completes_normally(self, monkeypatch) -> None:
        """With the default batch_timeout_seconds=None, a slow-ish batch
        completes without any timeout machinery engaging.
        """
        templates = [_make_template("q1"), _make_template("q2")]

        def mock_execute_task(task, answer_cache=None, cache_status=None, cached_answer_data=None):
            return (f"key_{task['question_id']}", _make_result(task["question_id"]))

        monkeypatch.setattr(
            "karenina.benchmark.verification.batch_runner.execute_task",
            mock_execute_task,
        )

        config = _make_config()
        result_set = run_verification_batch(
            templates,
            config,
            async_enabled=True,
            max_workers=2,
        )

        assert len(result_set) == 2


# ============================================================================
# Scenario path end-to-end through Benchmark._run_scenario_verification
# ============================================================================


def _make_scenario_mock(name: str) -> MagicMock:
    """A MagicMock scenario whose nodes dict iterates empty.

    _prepare_scenario walks scenario_def.nodes.items(); MagicMock's default
    iteration yields nothing, so the definition passes through unchanged.
    """
    scenario = MagicMock()
    scenario.name = name
    scenario.nodes = {}
    return scenario


@pytest.mark.unit
class TestScenarioBatchTimeoutEndToEnd:
    """A finite batch_timeout_seconds triggers the documented scenario contract."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_timeout_returns_partial_with_timeout_error(self, mock_manager_cls: MagicMock) -> None:
        """One fast combo completes, one blocks past the batch timeout. The
        facade returns the partial result set with a TimeoutError entry in
        errors instead of raising or wedging.
        """
        from karenina.benchmark.benchmark import Benchmark

        block = threading.Event()

        def mock_run(**kwargs):
            name = kwargs["scenario"].name
            result = MagicMock()
            result.scenario_id = name
            result.status = "completed"
            result.turn_results = []
            if name == "slow":
                block.wait(timeout=SLOW_TASK_SECONDS)
            return result

        mock_manager_cls.return_value.run.side_effect = mock_run

        benchmark = Benchmark(name="test")
        benchmark._scenarios = {
            "fast": _make_scenario_mock("fast"),
            "slow": _make_scenario_mock("slow"),
        }

        config = _make_config(batch_timeout_seconds=BATCH_TIMEOUT_SECONDS, async_max_workers=2)
        result_set = benchmark._run_scenario_verification(config, async_enabled=True)

        assert result_set.errors is not None
        timeout_errors = [(desc, exc) for desc, exc in result_set.errors if isinstance(exc, TimeoutError)]
        assert len(timeout_errors) >= 1
        assert "timed out" in str(timeout_errors[0][1])
        # The fast combo's result is kept.
        assert result_set.scenario_results is not None
        completed_ids = {r.scenario_id for r in result_set.scenario_results}
        assert "fast" in completed_ids

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioManager")
    def test_none_timeout_completes_normally(self, mock_manager_cls: MagicMock) -> None:
        """Default None: both combos finish, no timeout entries."""
        from karenina.benchmark.benchmark import Benchmark

        def mock_run(**kwargs):
            result = MagicMock()
            result.scenario_id = kwargs["scenario"].name
            result.status = "completed"
            result.turn_results = []
            return result

        mock_manager_cls.return_value.run.side_effect = mock_run

        benchmark = Benchmark(name="test")
        benchmark._scenarios = {
            "s1": _make_scenario_mock("s1"),
            "s2": _make_scenario_mock("s2"),
        }

        config = _make_config(async_max_workers=2)
        result_set = benchmark._run_scenario_verification(config, async_enabled=True)

        assert result_set.errors is None
        assert result_set.scenario_results is not None
        assert {r.scenario_id for r in result_set.scenario_results} == {"s1", "s2"}
