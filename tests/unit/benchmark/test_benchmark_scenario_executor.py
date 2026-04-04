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
