"""Tests for benchmark facade delegation to ScenarioExecutor."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestBenchmarkScenarioExecutorIntegration:
    """Verify that _run_scenario_verification delegates to ScenarioExecutor."""

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_delegates_to_scenario_executor(self, MockExecutor):
        """ScenarioExecutor is instantiated and run_batch is called."""
        from karenina.benchmark.benchmark import Benchmark
        from karenina.schemas.config.models import ModelConfig
        from karenina.schemas.verification.config import VerificationConfig

        mock_executor = MockExecutor.return_value
        mock_result = MagicMock()
        mock_result.turn_results = []
        mock_executor.run_batch.return_value = ([mock_result], [])

        benchmark = Benchmark(name="test")
        scenario = MagicMock()
        scenario.name = "test_scenario"
        benchmark._scenarios = {"test_scenario": scenario}

        config = VerificationConfig(
            answering_models=[ModelConfig(id="m1", model_name="m1", model_provider="openai")],
            parsing_models=[ModelConfig(id="m2", model_name="m2", model_provider="openai")],
        )

        result = benchmark._run_scenario_verification(config, async_enabled=True)

        MockExecutor.assert_called_once()
        mock_executor.run_batch.assert_called_once()
        assert result.results is not None

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_passes_max_concurrent_requests(self, MockExecutor):
        """max_concurrent_requests from config is forwarded to ScenarioExecutorConfig."""
        from karenina.benchmark.benchmark import Benchmark
        from karenina.schemas.config.models import ModelConfig
        from karenina.schemas.verification.config import VerificationConfig

        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = ([], [])

        benchmark = Benchmark(name="test")
        scenario = MagicMock()
        scenario.name = "s1"
        benchmark._scenarios = {"s1": scenario}

        config = VerificationConfig(
            answering_models=[ModelConfig(id="m1", model_name="m1", model_provider="openai")],
            parsing_models=[ModelConfig(id="m2", model_name="m2", model_provider="openai")],
            max_concurrent_requests=16,
        )

        benchmark._run_scenario_verification(config, async_enabled=True)

        # Verify max_concurrent_requests was passed to ScenarioExecutorConfig
        call_kwargs = MockExecutor.call_args
        executor_config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert executor_config.max_concurrent_requests == 16

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_passes_async_max_workers(self, MockExecutor):
        """async_max_workers from config is forwarded as max_workers to ScenarioExecutorConfig."""
        from karenina.benchmark.benchmark import Benchmark
        from karenina.schemas.config.models import ModelConfig
        from karenina.schemas.verification.config import VerificationConfig

        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = ([], [])

        benchmark = Benchmark(name="test")
        scenario = MagicMock()
        scenario.name = "s1"
        benchmark._scenarios = {"s1": scenario}

        config = VerificationConfig(
            answering_models=[ModelConfig(id="m1", model_name="m1", model_provider="openai")],
            parsing_models=[ModelConfig(id="m2", model_name="m2", model_provider="openai")],
            async_max_workers=8,
        )

        benchmark._run_scenario_verification(config, async_enabled=True)

        call_kwargs = MockExecutor.call_args
        executor_config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert executor_config.max_workers == 8

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_single_combo_creates_sequential_executor(self, MockExecutor):
        """With one combo, executor is created with parallel=False even when async_enabled=True."""
        from karenina.benchmark.benchmark import Benchmark
        from karenina.schemas.config.models import ModelConfig
        from karenina.schemas.verification.config import VerificationConfig

        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = ([], [])

        benchmark = Benchmark(name="test")
        scenario = MagicMock()
        scenario.name = "s1"
        benchmark._scenarios = {"s1": scenario}

        config = VerificationConfig(
            answering_models=[ModelConfig(id="m1", model_name="m1", model_provider="openai")],
            parsing_models=[ModelConfig(id="m2", model_name="m2", model_provider="openai")],
        )

        benchmark._run_scenario_verification(config, async_enabled=True)

        call_kwargs = MockExecutor.call_args
        assert call_kwargs.kwargs.get("parallel") is False or call_kwargs[1].get("parallel") is False

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_errors_forwarded_to_result_set(self, MockExecutor):
        """Errors from executor.run_batch are included in VerificationResultSet."""
        from karenina.benchmark.benchmark import Benchmark
        from karenina.schemas.config.models import ModelConfig
        from karenina.schemas.verification.config import VerificationConfig

        mock_executor = MockExecutor.return_value
        error = RuntimeError("provider down")
        mock_executor.run_batch.return_value = ([], [("combo description", error)])

        benchmark = Benchmark(name="test")
        scenario = MagicMock()
        scenario.name = "s1"
        benchmark._scenarios = {"s1": scenario}

        config = VerificationConfig(
            answering_models=[ModelConfig(id="m1", model_name="m1", model_provider="openai")],
            parsing_models=[ModelConfig(id="m2", model_name="m2", model_provider="openai")],
        )

        result = benchmark._run_scenario_verification(config)

        assert result.errors is not None
        assert len(result.errors) == 1
        assert result.errors[0][0] == "combo description"
        assert isinstance(result.errors[0][1], RuntimeError)


def _build_scenario_with_override(
    override_answering=None,
    override_parsing=None,
    node_id: str = "guardrail",
    scenario_name: str = "test_scenario",
):
    """Build a scenario containing one node with the given ModelOverride.

    Helper used by tests verifying that pipeline-level timeout and retry
    policy stamping reaches per-node ``ModelOverride`` instances.
    """
    from karenina.scenario.builder import Scenario
    from karenina.schemas.entities import Question
    from karenina.schemas.scenario.types import END, ModelOverride

    s = Scenario(scenario_name)
    s.add_node(
        node_id,
        question=Question(
            question="Is this safe?",
            raw_answer="yes",
            answer_template="class Answer: pass",
        ),
        model_override=ModelOverride(
            answering_model=override_answering,
            parsing_model=override_parsing,
        ),
    )
    s.add_edge(node_id, END)
    s.set_entry(node_id)
    return s.validate()


@pytest.mark.unit
class TestScenarioOverrideStamping:
    """Verify that per-node ModelOverride receives pipeline-level stamping.

    Without stamping, an overridden model runs with ``request_timeout=None``
    and ``retry_policy=None``, falling back to SDK defaults and the default
    ``RetryPolicy()``. The benchmark facade must propagate the configured
    request_timeout and retry_policy onto each ModelOverride field that is
    not already explicitly set, mirroring the behavior applied to the
    top-level answering and parsing models.
    """

    def _make_config(
        self,
        request_timeout: float = 600.0,
        timeout_max_attempts: int = 5,
    ):
        from karenina.schemas.config.models import ModelConfig
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
            answering_models=[
                ModelConfig(id="base_ans", model_name="base_ans", model_provider="openai"),
            ],
            parsing_models=[
                ModelConfig(id="base_parse", model_name="base_parse", model_provider="openai"),
            ],
            request_timeout=request_timeout,
            retry_policy=retry_policy,
            async_max_workers=4,
        )

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_override_models_receive_timeout_and_retry_stamping(self, MockExecutor):
        """ModelOverride.answering_model and parsing_model are stamped from config."""
        from karenina.benchmark.benchmark import Benchmark
        from karenina.schemas.config.models import ModelConfig

        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = ([], [])

        config = self._make_config(request_timeout=600.0, timeout_max_attempts=5)

        # Override models with no request_timeout / retry_policy set
        override_ans = ModelConfig(
            id="guard_ans",
            model_name="guard_ans",
            model_provider="openai",
        )
        override_parse = ModelConfig(
            id="guard_parse",
            model_name="guard_parse",
            model_provider="openai",
        )
        scenario_def = _build_scenario_with_override(
            override_answering=override_ans,
            override_parsing=override_parse,
        )

        benchmark = Benchmark(name="test_bm")
        benchmark.add_scenario(scenario_def)

        benchmark._run_scenario_verification(config, async_enabled=False)

        # Inspect the prepared scenario passed to the executor
        call_kwargs = MockExecutor.return_value.run_batch.call_args
        combos = call_kwargs.kwargs.get("combos") or call_kwargs[1].get("combos")
        assert len(combos) == 1
        prepared_scenario, _ans, _parse, _rep = combos[0]

        prepared_node = prepared_scenario.nodes["guardrail"]
        assert prepared_node.model_override is not None
        prepared_override_ans = prepared_node.model_override.answering_model
        prepared_override_parse = prepared_node.model_override.parsing_model
        assert prepared_override_ans is not None
        assert prepared_override_parse is not None

        # Both override fields must inherit the pipeline-level stamping
        assert prepared_override_ans.request_timeout == 600.0
        assert prepared_override_parse.request_timeout == 600.0
        assert prepared_override_ans.retry_policy is not None
        assert prepared_override_ans.retry_policy.timeout.max_attempts == 5
        assert prepared_override_ans.retry_policy.timeout_escalation is not None
        assert prepared_override_ans.retry_policy.timeout_escalation.strategy == "additive"
        assert prepared_override_parse.retry_policy is not None
        assert prepared_override_parse.retry_policy.timeout.max_attempts == 5

        # The original scenario stored on the benchmark must NOT be mutated
        original_node = benchmark.get_scenario("test_scenario").nodes["guardrail"]
        assert original_node.model_override is not None
        original_ans = original_node.model_override.answering_model
        original_parse = original_node.model_override.parsing_model
        assert original_ans is not None and original_ans.request_timeout is None
        assert original_ans.retry_policy is None
        assert original_parse is not None and original_parse.request_timeout is None
        assert original_parse.retry_policy is None

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_explicit_override_fields_are_preserved(self, MockExecutor):
        """Stamping must NOT overwrite explicit request_timeout or retry_policy on the override."""
        from karenina.benchmark.benchmark import Benchmark
        from karenina.schemas.config.models import ModelConfig
        from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy

        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = ([], [])

        config = self._make_config(request_timeout=600.0, timeout_max_attempts=5)

        explicit_policy = RetryPolicy(
            timeout=CategoryRetryConfig(max_attempts=9, backoff_min=1.0, backoff_max=2.0),
        )
        override_ans = ModelConfig(
            id="guard_ans",
            model_name="guard_ans",
            model_provider="openai",
            request_timeout=42.0,
            retry_policy=explicit_policy,
        )
        # parsing_model leaves both fields unset to verify the partial preserve case
        override_parse = ModelConfig(
            id="guard_parse",
            model_name="guard_parse",
            model_provider="openai",
        )
        scenario_def = _build_scenario_with_override(
            override_answering=override_ans,
            override_parsing=override_parse,
        )

        benchmark = Benchmark(name="test_bm")
        benchmark.add_scenario(scenario_def)

        benchmark._run_scenario_verification(config, async_enabled=False)

        call_kwargs = MockExecutor.return_value.run_batch.call_args
        combos = call_kwargs.kwargs.get("combos") or call_kwargs[1].get("combos")
        prepared_scenario, _ans, _parse, _rep = combos[0]
        prepared_override = prepared_scenario.nodes["guardrail"].model_override
        assert prepared_override is not None

        # Explicitly set fields are preserved
        prepared_ans = prepared_override.answering_model
        assert prepared_ans is not None
        assert prepared_ans.request_timeout == 42.0
        assert prepared_ans.retry_policy is explicit_policy
        assert prepared_ans.retry_policy.timeout.max_attempts == 9

        # Unset parsing model fields are still stamped from the config
        prepared_parse = prepared_override.parsing_model
        assert prepared_parse is not None
        assert prepared_parse.request_timeout == 600.0
        assert prepared_parse.retry_policy is not None
        assert prepared_parse.retry_policy.timeout.max_attempts == 5

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_no_override_leaves_scenario_definition_identity(self, MockExecutor):
        """Scenarios without per-node overrides reuse the original definition object."""
        from karenina.benchmark.benchmark import Benchmark
        from karenina.scenario.builder import Scenario
        from karenina.schemas.entities import Question
        from karenina.schemas.scenario.types import END

        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = ([], [])

        config = self._make_config()

        s = Scenario("plain")
        s.add_node(
            "ask",
            question=Question(
                question="Q?",
                raw_answer="A",
                answer_template="class Answer: pass",
            ),
        )
        s.add_edge("ask", END)
        s.set_entry("ask")
        scenario_def = s.validate()

        benchmark = Benchmark(name="test_bm")
        benchmark.add_scenario(scenario_def)
        original = benchmark.get_scenario("plain")

        benchmark._run_scenario_verification(config, async_enabled=False)

        call_kwargs = MockExecutor.return_value.run_batch.call_args
        combos = call_kwargs.kwargs.get("combos") or call_kwargs[1].get("combos")
        prepared_scenario, _ans, _parse, _rep = combos[0]
        # When no node has an override, _prepare_scenario returns the original
        assert prepared_scenario is original


def _make_deep_agent_override(model_id: str = "guard_agent"):
    """Build a ModelConfig with a deep_agent interface for AgenticRubricTrait.model_override."""
    from karenina.schemas.config.models import ModelConfig

    return ModelConfig(
        id=model_id,
        model_name=model_id,
        model_provider="anthropic",
        interface="claude_agent_sdk",
    )


def _make_pipeline_config_with_retry(
    request_timeout: float = 600.0,
    timeout_max_attempts: int = 5,
):
    """Build a VerificationConfig with non-default timeout and retry policy."""
    from karenina.schemas.config.models import ModelConfig
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
        answering_models=[
            ModelConfig(id="base_ans", model_name="base_ans", model_provider="openai"),
        ],
        parsing_models=[
            ModelConfig(id="base_parse", model_name="base_parse", model_provider="openai"),
        ],
        request_timeout=request_timeout,
        retry_policy=retry_policy,
        async_max_workers=4,
        evaluation_mode="template_and_rubric",
    )


@pytest.mark.unit
class TestAgenticRubricTraitOverrideStamping:
    """Verify that agentic rubric trait model_overrides receive pipeline-level stamping.

    Without stamping, an AgenticRubricTrait whose model_override has
    request_timeout=None and retry_policy=None will run with SDK defaults
    and the default RetryPolicy(), even when the user configured a different
    policy on VerificationConfig. This mirrors the per-node ModelOverride
    fix and applies to both the scenario path and the QA path.
    """

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_scenario_global_rubric_agentic_override_stamped(self, MockExecutor):
        """Scenario path: global_rubric agentic trait override receives stamping."""
        from karenina.benchmark.benchmark import Benchmark
        from karenina.scenario.builder import Scenario
        from karenina.schemas.entities import Question
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric
        from karenina.schemas.scenario.types import END

        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = ([], [])

        config = _make_pipeline_config_with_retry(request_timeout=600.0, timeout_max_attempts=5)

        # Build scenario with no per-node override (rubric carries the override instead)
        s = Scenario("plain")
        s.add_node(
            "ask",
            question=Question(
                question="Q?",
                raw_answer="A",
                answer_template="class Answer: pass",
            ),
        )
        s.add_edge("ask", END)
        s.set_entry("ask")
        scenario_def = s.validate()

        benchmark = Benchmark(name="test_bm")
        benchmark.add_scenario(scenario_def)

        # Attach a global rubric with an agentic trait that has a deep-agent override
        override = _make_deep_agent_override()
        agentic = AgenticRubricTrait(
            name="investigate",
            description="Investigate the trace",
            kind="boolean",
            higher_is_better=True,
            model_override=override,
        )
        benchmark.set_global_rubric(Rubric(agentic_traits=[agentic]))

        benchmark._run_scenario_verification(config, async_enabled=False)

        # Inspect the global_rubric handed to the executor
        call_kwargs = MockExecutor.return_value.run_batch.call_args
        passed_rubric = call_kwargs.kwargs.get("global_rubric") or call_kwargs[1].get("global_rubric")
        assert passed_rubric is not None
        assert len(passed_rubric.agentic_traits) == 1
        passed_override = passed_rubric.agentic_traits[0].model_override
        assert passed_override is not None
        assert passed_override.request_timeout == 600.0
        assert passed_override.retry_policy is not None
        assert passed_override.retry_policy.timeout.max_attempts == 5
        assert passed_override.retry_policy.timeout_escalation is not None
        assert passed_override.retry_policy.timeout_escalation.strategy == "additive"

        # The original rubric on the benchmark must NOT be mutated
        original_rubric = benchmark._rubric_manager.get_global_rubric()
        assert original_rubric is not None
        original_override = original_rubric.agentic_traits[0].model_override
        assert original_override is not None
        assert original_override.request_timeout is None
        assert original_override.retry_policy is None

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_scenario_explicit_override_fields_preserved(self, MockExecutor):
        """Scenario path: explicit override fields must NOT be overwritten."""
        from karenina.benchmark.benchmark import Benchmark
        from karenina.scenario.builder import Scenario
        from karenina.schemas.config.models import ModelConfig
        from karenina.schemas.entities import Question
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric
        from karenina.schemas.scenario.types import END
        from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy

        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = ([], [])

        config = _make_pipeline_config_with_retry(request_timeout=600.0, timeout_max_attempts=5)

        explicit_policy = RetryPolicy(
            timeout=CategoryRetryConfig(max_attempts=9, backoff_min=1.0, backoff_max=2.0),
        )
        override = ModelConfig(
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
            model_override=override,
        )

        s = Scenario("plain")
        s.add_node(
            "ask",
            question=Question(
                question="Q?",
                raw_answer="A",
                answer_template="class Answer: pass",
            ),
        )
        s.add_edge("ask", END)
        s.set_entry("ask")
        scenario_def = s.validate()

        benchmark = Benchmark(name="test_bm")
        benchmark.add_scenario(scenario_def)
        benchmark.set_global_rubric(Rubric(agentic_traits=[agentic]))

        benchmark._run_scenario_verification(config, async_enabled=False)

        call_kwargs = MockExecutor.return_value.run_batch.call_args
        passed_rubric = call_kwargs.kwargs.get("global_rubric") or call_kwargs[1].get("global_rubric")
        assert passed_rubric is not None
        passed_override = passed_rubric.agentic_traits[0].model_override
        assert passed_override is not None
        # Explicit values preserved (note: set_global_rubric round-trips through
        # the checkpoint serializer, so the retry_policy is structurally equal but
        # not the same instance)
        assert passed_override.request_timeout == 42.0
        assert passed_override.retry_policy is not None
        assert passed_override.retry_policy.timeout.max_attempts == 9
        # The retry policy must NOT have been replaced with the pipeline default
        assert passed_override.retry_policy.timeout.max_attempts != 5

    @patch("karenina.benchmark.verification.scenario_executor.ScenarioExecutor")
    def test_scenario_no_agentic_traits_short_circuits(self, MockExecutor):
        """Rubric without agentic traits is passed through unchanged (identity)."""
        from karenina.benchmark.benchmark import Benchmark
        from karenina.scenario.builder import Scenario
        from karenina.schemas.entities import Question
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric
        from karenina.schemas.scenario.types import END

        mock_executor = MockExecutor.return_value
        mock_executor.run_batch.return_value = ([], [])

        config = _make_pipeline_config_with_retry()

        s = Scenario("plain")
        s.add_node(
            "ask",
            question=Question(
                question="Q?",
                raw_answer="A",
                answer_template="class Answer: pass",
            ),
        )
        s.add_edge("ask", END)
        s.set_entry("ask")
        scenario_def = s.validate()

        benchmark = Benchmark(name="test_bm")
        benchmark.add_scenario(scenario_def)
        benchmark.set_global_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="quality",
                        description="A quality trait",
                        kind="boolean",
                        higher_is_better=True,
                    )
                ]
            )
        )

        benchmark._run_scenario_verification(config, async_enabled=False)

        call_kwargs = MockExecutor.return_value.run_batch.call_args
        passed_rubric = call_kwargs.kwargs.get("global_rubric") or call_kwargs[1].get("global_rubric")
        assert passed_rubric is not None
        # No agentic traits means stamp_agentic_trait_overrides returns the same instance
        assert passed_rubric.llm_traits[0].name == "quality"
        assert not passed_rubric.agentic_traits

    def test_qa_path_global_rubric_agentic_override_stamped(self):
        """QA path: merge_rubrics_for_task stamps global rubric agentic overrides.

        The QA verification path runs through batch_runner.generate_task_queue,
        which calls merge_rubrics_for_task per template. That helper already
        stamps via stamp_agentic_trait_overrides, so this test verifies the
        end-to-end behavior by inspecting the task queue produced for a
        global rubric carrying an agentic trait with an override.
        """
        from karenina.benchmark.verification.batch_runner import generate_task_queue
        from karenina.schemas.config.models import ModelConfig
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric
        from karenina.schemas.verification import FinishedTemplate

        config = _make_pipeline_config_with_retry(request_timeout=600.0, timeout_max_attempts=5)
        # Replace the answering/parsing models with stub configs that batch_runner accepts
        config = config.model_copy(
            update={
                "answering_models": [
                    ModelConfig(
                        id="ans1",
                        model_name="ans1",
                        model_provider="openai",
                        interface="langchain",
                    )
                ],
                "parsing_models": [
                    ModelConfig(
                        id="parse1",
                        model_name="parse1",
                        model_provider="openai",
                        interface="langchain",
                    )
                ],
            }
        )

        override = _make_deep_agent_override()
        agentic = AgenticRubricTrait(
            name="investigate",
            description="Investigate the trace",
            kind="boolean",
            higher_is_better=True,
            model_override=override,
        )
        global_rubric = Rubric(agentic_traits=[agentic])

        template = FinishedTemplate(
            question_id="q1",
            question_text="What is 2+2?",
            question_preview="What is 2+2?",
            template_code="class Answer(BaseAnswer): pass",
            last_modified="2026-01-01T00:00:00",
        )

        tasks = generate_task_queue([template], config, global_rubric=global_rubric)
        assert len(tasks) == 1
        task_rubric = tasks[0]["rubric"]
        assert task_rubric is not None
        assert len(task_rubric.agentic_traits) == 1
        task_override = task_rubric.agentic_traits[0].model_override
        assert task_override is not None
        assert task_override.request_timeout == 600.0
        assert task_override.retry_policy is not None
        assert task_override.retry_policy.timeout.max_attempts == 5
        assert task_override.retry_policy.timeout_escalation is not None
        assert task_override.retry_policy.timeout_escalation.strategy == "additive"

        # Original global rubric and override are not mutated
        assert global_rubric.agentic_traits[0].model_override is override
        assert override.request_timeout is None
        assert override.retry_policy is None

    def test_qa_path_explicit_fields_preserved(self):
        """QA path: explicit override fields are preserved through generate_task_queue."""
        from karenina.benchmark.verification.batch_runner import generate_task_queue
        from karenina.schemas.config.models import ModelConfig
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric
        from karenina.schemas.verification import FinishedTemplate
        from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy

        config = _make_pipeline_config_with_retry(request_timeout=600.0, timeout_max_attempts=5)
        config = config.model_copy(
            update={
                "answering_models": [
                    ModelConfig(
                        id="ans1",
                        model_name="ans1",
                        model_provider="openai",
                        interface="langchain",
                    )
                ],
                "parsing_models": [
                    ModelConfig(
                        id="parse1",
                        model_name="parse1",
                        model_provider="openai",
                        interface="langchain",
                    )
                ],
            }
        )

        explicit_policy = RetryPolicy(
            timeout=CategoryRetryConfig(max_attempts=9, backoff_min=1.0, backoff_max=2.0),
        )
        override = ModelConfig(
            id="agent-1",
            model_name="agent-1",
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
            model_override=override,
        )
        global_rubric = Rubric(agentic_traits=[agentic])

        template = FinishedTemplate(
            question_id="q1",
            question_text="What is 2+2?",
            question_preview="What is 2+2?",
            template_code="class Answer(BaseAnswer): pass",
            last_modified="2026-01-01T00:00:00",
        )

        tasks = generate_task_queue([template], config, global_rubric=global_rubric)
        task_override = tasks[0]["rubric"].agentic_traits[0].model_override
        assert task_override is not None
        assert task_override.request_timeout == 42.0
        assert task_override.retry_policy is explicit_policy
        assert task_override.retry_policy.timeout.max_attempts == 9
