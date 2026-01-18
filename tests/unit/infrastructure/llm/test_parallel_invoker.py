"""Unit tests for ParallelLLMInvoker utility.

Tests cover:
- Parallel execution vs sequential
- Result ordering preservation
- Error isolation (one failure doesn't block others)
- Usage metadata aggregation
- max_workers configuration
- Environment variable handling
- Edge cases (empty input, single task)
"""

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from karenina.infrastructure.llm.parallel_invoker import (
    ParallelLLMInvoker,
    read_async_config,
)

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class MockResponseModel(BaseModel):
    """Mock response model for testing."""

    result: str


class MockSlowLLM:
    """Mock LLM that takes time to respond (for testing parallelism)."""

    def __init__(self, delay: float = 0.1, responses: list[str] | None = None):
        self.delay = delay
        self.responses = responses or []
        self.call_count = 0
        self.call_times: list[float] = []

    def invoke(self, _messages: list[Any]) -> AIMessage:
        start = time.time()
        self.call_times.append(start)
        time.sleep(self.delay)
        response = self.responses[self.call_count] if self.call_count < len(self.responses) else "default"
        self.call_count += 1
        return AIMessage(content=f'{{"result": "{response}"}}')


class MockFailingLLM:
    """Mock LLM that fails on specific calls."""

    def __init__(self, fail_indices: set[int]):
        self.fail_indices = fail_indices
        self.call_count = 0

    def invoke(self, _messages: list[Any]) -> AIMessage:
        current = self.call_count
        self.call_count += 1
        if current in self.fail_indices:
            raise ValueError(f"Simulated failure at index {current}")
        return AIMessage(content='{"result": "success"}')


# =============================================================================
# ParallelLLMInvoker Tests
# =============================================================================


@pytest.mark.unit
def test_parallel_invoker_empty_tasks() -> None:
    """Test that empty task list returns empty results."""
    mock_llm = MagicMock()
    invoker = ParallelLLMInvoker(mock_llm, max_workers=2)

    results = invoker.invoke_batch([])

    assert results == []
    mock_llm.invoke.assert_not_called()


@pytest.mark.unit
def test_parallel_invoker_single_task() -> None:
    """Test invoker with a single task (degenerate case)."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content='{"result": "test"}')

    invoker = ParallelLLMInvoker(mock_llm, max_workers=2)

    messages = [SystemMessage(content="system"), HumanMessage(content="user")]
    tasks = [(messages, MockResponseModel)]

    with patch(
        "karenina.benchmark.verification.evaluators.rubric_parsing.invoke_with_structured_output"
    ) as mock_invoke:
        mock_invoke.return_value = (MockResponseModel(result="test"), {"total_tokens": 10})
        results = invoker.invoke_batch(tasks)

    assert len(results) == 1
    result, usage, error = results[0]
    assert error is None
    assert result.result == "test"
    assert usage == {"total_tokens": 10}


@pytest.mark.unit
def test_parallel_invoker_result_ordering_preserved() -> None:
    """Test that results are returned in input order regardless of completion order."""
    invoker = ParallelLLMInvoker(MagicMock(), max_workers=4)

    # Create tasks that would complete in different order if run in parallel
    messages = [HumanMessage(content="test")]
    tasks = [(messages, MockResponseModel) for _ in range(5)]

    responses = [MockResponseModel(result=f"result_{i}") for i in range(5)]

    with patch(
        "karenina.benchmark.verification.evaluators.rubric_parsing.invoke_with_structured_output"
    ) as mock_invoke:
        # Make invoke return different results based on call order
        mock_invoke.side_effect = [(responses[i], {"index": i}) for i in range(5)]

        results = invoker.invoke_batch(tasks)

    assert len(results) == 5
    # Results should be in input order (0, 1, 2, 3, 4)
    for i, (result, _usage, error) in enumerate(results):
        assert error is None
        assert result.result == f"result_{i}"


@pytest.mark.unit
def test_parallel_invoker_error_isolation() -> None:
    """Test that one failed task doesn't block others."""
    invoker = ParallelLLMInvoker(MagicMock(), max_workers=4)

    messages = [HumanMessage(content="test")]
    tasks = [(messages, MockResponseModel) for _ in range(5)]

    def mock_invoke_with_error(_llm, _msgs, _model_class):
        # Fail on index 2
        if mock_invoke_with_error.call_count == 2:
            mock_invoke_with_error.call_count += 1
            raise ValueError("Simulated failure")
        mock_invoke_with_error.call_count += 1
        return MockResponseModel(result="success"), {"tokens": 10}

    mock_invoke_with_error.call_count = 0

    with patch(
        "karenina.benchmark.verification.evaluators.rubric_parsing.invoke_with_structured_output",
        side_effect=mock_invoke_with_error,
    ):
        results = invoker.invoke_batch(tasks)

    assert len(results) == 5

    # Count successes and failures
    successes = sum(1 for r, u, e in results if e is None)
    failures = sum(1 for r, u, e in results if e is not None)

    assert successes == 4
    assert failures == 1

    # Verify the failed one has proper error
    failed_results = [(i, r, u, e) for i, (r, u, e) in enumerate(results) if e is not None]
    assert len(failed_results) == 1
    _, result, usage, error = failed_results[0]
    assert result is None
    assert usage is None
    assert "Simulated failure" in str(error)


@pytest.mark.unit
def test_parallel_invoker_partial_failures() -> None:
    """Test handling of multiple partial failures."""
    invoker = ParallelLLMInvoker(MagicMock(), max_workers=4)

    messages = [HumanMessage(content="test")]
    tasks = [(messages, MockResponseModel) for _ in range(6)]

    # Fail on indices 1, 3, 5 (every other starting from 1)
    call_count = [0]  # Use list to allow mutation in closure

    def mock_invoke_alternating(_llm, _msgs, _model_class):
        idx = call_count[0]
        call_count[0] += 1
        if idx % 2 == 1:  # Fail on odd indices
            raise ValueError(f"Failure at {idx}")
        return MockResponseModel(result=f"success_{idx}"), {"index": idx}

    with patch(
        "karenina.benchmark.verification.evaluators.rubric_parsing.invoke_with_structured_output",
        side_effect=mock_invoke_alternating,
    ):
        results = invoker.invoke_batch(tasks)

    assert len(results) == 6

    # Verify pattern: success, fail, success, fail, success, fail
    for i, (result, _usage, error) in enumerate(results):
        if i % 2 == 0:
            assert error is None
            assert result is not None
        else:
            assert error is not None
            assert result is None


@pytest.mark.unit
def test_parallel_invoker_max_workers_default() -> None:
    """Test default max_workers value when no env var is set."""
    # Clear the env var to test default
    with patch.dict("os.environ", {}, clear=True):
        invoker = ParallelLLMInvoker(MagicMock())
        # Should default to 2
        assert invoker.max_workers == 2


@pytest.mark.unit
def test_parallel_invoker_max_workers_explicit() -> None:
    """Test explicit max_workers value."""
    invoker = ParallelLLMInvoker(MagicMock(), max_workers=8)

    assert invoker.max_workers == 8


@pytest.mark.unit
def test_parallel_invoker_max_workers_from_env() -> None:
    """Test max_workers from KARENINA_ASYNC_MAX_WORKERS env var."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "6"}):
        invoker = ParallelLLMInvoker(MagicMock())
        assert invoker.max_workers == 6


@pytest.mark.unit
def test_parallel_invoker_explicit_overrides_env() -> None:
    """Test that explicit max_workers overrides env var."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "6"}):
        invoker = ParallelLLMInvoker(MagicMock(), max_workers=4)
        assert invoker.max_workers == 4


@pytest.mark.unit
def test_parallel_invoker_invalid_env_uses_default() -> None:
    """Test that invalid env var falls back to default."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "invalid"}):
        invoker = ParallelLLMInvoker(MagicMock())
        assert invoker.max_workers == 2


@pytest.mark.unit
def test_parallel_invoker_progress_callback() -> None:
    """Test that progress callback is called correctly."""
    invoker = ParallelLLMInvoker(MagicMock(), max_workers=2)

    messages = [HumanMessage(content="test")]
    tasks = [(messages, MockResponseModel) for _ in range(3)]

    progress_calls: list[tuple[int, int]] = []

    def progress_callback(completed: int, total: int) -> None:
        progress_calls.append((completed, total))

    with patch(
        "karenina.benchmark.verification.evaluators.rubric_parsing.invoke_with_structured_output"
    ) as mock_invoke:
        mock_invoke.return_value = (MockResponseModel(result="test"), {})
        invoker.invoke_batch(tasks, progress_callback=progress_callback)

    # Should have 3 progress calls (one per task completion)
    assert len(progress_calls) == 3
    # All should have total=3
    assert all(total == 3 for _, total in progress_calls)
    # Completed should be 1, 2, 3 (in some order due to parallel execution)
    completed_values = sorted(c for c, _ in progress_calls)
    assert completed_values == [1, 2, 3]


@pytest.mark.unit
def test_parallel_invoker_aggregated_usage() -> None:
    """Test invoke_batch_with_aggregated_usage method."""
    invoker = ParallelLLMInvoker(MagicMock(), max_workers=4)

    messages = [HumanMessage(content="test")]
    tasks = [(messages, MockResponseModel) for _ in range(3)]

    with patch(
        "karenina.benchmark.verification.evaluators.rubric_parsing.invoke_with_structured_output"
    ) as mock_invoke:
        mock_invoke.side_effect = [
            (MockResponseModel(result="r1"), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
            (MockResponseModel(result="r2"), {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18}),
            (MockResponseModel(result="r3"), {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}),
        ]

        results, aggregated = invoker.invoke_batch_with_aggregated_usage(tasks)

    assert len(results) == 3
    assert aggregated["calls"] == 3
    assert aggregated["input_tokens"] == 30  # 10 + 12 + 8
    assert aggregated["output_tokens"] == 15  # 5 + 6 + 4
    assert aggregated["total_tokens"] == 45  # 15 + 18 + 12


@pytest.mark.unit
def test_parallel_invoker_aggregated_usage_with_errors() -> None:
    """Test aggregated usage handles errors correctly."""
    invoker = ParallelLLMInvoker(MagicMock(), max_workers=4)

    messages = [HumanMessage(content="test")]
    tasks = [(messages, MockResponseModel) for _ in range(3)]

    call_count = [0]

    def mock_invoke_with_one_error(_llm, _msgs, _model_class):
        idx = call_count[0]
        call_count[0] += 1
        if idx == 1:
            raise ValueError("Error")
        return MockResponseModel(result=f"r{idx}"), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    with patch(
        "karenina.benchmark.verification.evaluators.rubric_parsing.invoke_with_structured_output",
        side_effect=mock_invoke_with_one_error,
    ):
        results, aggregated = invoker.invoke_batch_with_aggregated_usage(tasks)

    assert len(results) == 3
    # Only 2 successful calls
    assert aggregated["calls"] == 2
    assert aggregated["input_tokens"] == 20
    assert aggregated["output_tokens"] == 10
    assert aggregated["total_tokens"] == 30


# =============================================================================
# read_async_config Tests
# =============================================================================


@pytest.mark.unit
def test_read_async_config_defaults() -> None:
    """Test read_async_config with default values."""
    with patch.dict("os.environ", {}, clear=True):
        enabled, workers = read_async_config()

    assert enabled is True
    assert workers == 2


@pytest.mark.unit
def test_read_async_config_disabled() -> None:
    """Test read_async_config with KARENINA_ASYNC_ENABLED=false."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_ENABLED": "false"}):
        enabled, workers = read_async_config()

    assert enabled is False


@pytest.mark.unit
def test_read_async_config_enabled_variations() -> None:
    """Test various truthy values for KARENINA_ASYNC_ENABLED."""
    for value in ["true", "True", "TRUE", "1", "yes", "YES"]:
        with patch.dict("os.environ", {"KARENINA_ASYNC_ENABLED": value}):
            enabled, _ = read_async_config()
            assert enabled is True, f"Expected True for '{value}'"


@pytest.mark.unit
def test_read_async_config_disabled_variations() -> None:
    """Test various falsy values for KARENINA_ASYNC_ENABLED."""
    for value in ["false", "False", "FALSE", "0", "no", "NO", "anything_else"]:
        with patch.dict("os.environ", {"KARENINA_ASYNC_ENABLED": value}):
            enabled, _ = read_async_config()
            # Only "true", "1", "yes" are truthy
            if value.lower() in ("true", "1", "yes"):
                assert enabled is True
            else:
                assert enabled is False


@pytest.mark.unit
def test_read_async_config_max_workers() -> None:
    """Test read_async_config reads KARENINA_ASYNC_MAX_WORKERS."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "8"}):
        _, workers = read_async_config()

    assert workers == 8


@pytest.mark.unit
def test_read_async_config_invalid_max_workers() -> None:
    """Test read_async_config handles invalid max_workers gracefully."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "not_a_number"}):
        _, workers = read_async_config()

    # Should fall back to default
    assert workers == 2


@pytest.mark.unit
def test_read_async_config_combined() -> None:
    """Test read_async_config with both env vars set."""
    with patch.dict(
        "os.environ",
        {"KARENINA_ASYNC_ENABLED": "false", "KARENINA_ASYNC_MAX_WORKERS": "10"},
    ):
        enabled, workers = read_async_config()

    assert enabled is False
    assert workers == 10
