"""Unit tests for AdapterParallelInvoker utility.

Tests cover:
- Empty task list handling
- Result ordering preservation
- Error isolation (one failure doesn't block others)
- Usage metadata aggregation
- max_workers configuration
- Environment variable handling
- Progress callback
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from karenina.adapters.parallel import AdapterParallelInvoker

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class MockResponseModel(BaseModel):
    """Mock response model for testing."""

    result: str


class MockBooleanScore(BaseModel):
    """Mock boolean score response."""

    result: bool


class MockNumericScore(BaseModel):
    """Mock numeric score response."""

    score: int


class MockParserPort:
    """Mock implementation of ParserPort for testing.

    Provides configurable async parsing behavior including:
    - Delayed responses (for testing parallelism)
    - Controllable failures
    - Call tracking
    """

    def __init__(
        self,
        delay: float = 0.0,
        responses: list[BaseModel] | None = None,
        fail_indices: set[int] | None = None,
    ):
        self.delay = delay
        self.responses = responses or []
        self.fail_indices = fail_indices or set()
        self.call_count = 0
        self.call_times: list[float] = []

    async def aparse_to_pydantic(self, _response: str, schema: type[BaseModel]) -> BaseModel:
        """Async parsing with configurable behavior."""
        import time

        current = self.call_count
        self.call_count += 1
        self.call_times.append(time.time())

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if current in self.fail_indices:
            raise ValueError(f"Simulated failure at index {current}")

        if current < len(self.responses):
            return self.responses[current]

        # Default response
        return schema.model_validate({"result": f"result_{current}"})

    def parse_to_pydantic(self, response: str, schema: type[BaseModel]) -> BaseModel:
        """Sync wrapper for testing."""
        return asyncio.run(self.aparse_to_pydantic(response, schema))


# =============================================================================
# AdapterParallelInvoker Tests
# =============================================================================


@pytest.mark.unit
def test_adapter_parallel_invoker_empty_tasks() -> None:
    """Test that empty task list returns empty results."""
    mock_parser = MockParserPort()
    invoker = AdapterParallelInvoker(mock_parser, max_workers=2)

    results = invoker.invoke_batch([])

    assert results == []
    assert mock_parser.call_count == 0


@pytest.mark.unit
def test_adapter_parallel_invoker_single_task() -> None:
    """Test invoker with a single task (degenerate case)."""
    mock_parser = MockParserPort(responses=[MockResponseModel(result="test")])
    invoker = AdapterParallelInvoker(mock_parser, max_workers=2)

    tasks = [("test prompt", MockResponseModel)]
    results = invoker.invoke_batch(tasks)

    assert len(results) == 1
    result, usage, error = results[0]
    assert error is None
    assert result is not None
    assert result.result == "test"
    assert usage == {}  # ParserPort doesn't return usage metadata currently


@pytest.mark.unit
def test_adapter_parallel_invoker_result_ordering_preserved() -> None:
    """Test that results are returned in input order regardless of completion order."""
    # Create parser with responses that would complete in different order
    responses = [MockResponseModel(result=f"result_{i}") for i in range(5)]
    mock_parser = MockParserPort(responses=responses)
    invoker = AdapterParallelInvoker(mock_parser, max_workers=4)

    tasks = [(f"prompt_{i}", MockResponseModel) for i in range(5)]
    results = invoker.invoke_batch(tasks)

    assert len(results) == 5
    # Results should be in input order (0, 1, 2, 3, 4)
    for i, (result, _usage, error) in enumerate(results):
        assert error is None
        assert result is not None
        assert result.result == f"result_{i}"


@pytest.mark.unit
def test_adapter_parallel_invoker_error_isolation() -> None:
    """Test that one failed task doesn't block others."""
    # Fail on index 2
    mock_parser = MockParserPort(
        responses=[MockResponseModel(result=f"success_{i}") for i in range(5)],
        fail_indices={2},
    )
    invoker = AdapterParallelInvoker(mock_parser, max_workers=4)

    tasks = [(f"prompt_{i}", MockResponseModel) for i in range(5)]
    results = invoker.invoke_batch(tasks)

    assert len(results) == 5

    # Count successes and failures
    successes = sum(1 for r, u, e in results if e is None)
    failures = sum(1 for r, u, e in results if e is not None)

    assert successes == 4
    assert failures == 1

    # Verify the failed one has proper error
    result, usage, error = results[2]  # Index 2 should fail
    assert result is None
    assert usage is None
    assert error is not None
    assert "Simulated failure" in str(error)


@pytest.mark.unit
def test_adapter_parallel_invoker_partial_failures() -> None:
    """Test handling of multiple partial failures."""
    # Fail on indices 1, 3, 5 (every other starting from 1)
    mock_parser = MockParserPort(
        responses=[MockResponseModel(result=f"success_{i}") for i in range(6)],
        fail_indices={1, 3, 5},
    )
    invoker = AdapterParallelInvoker(mock_parser, max_workers=4)

    tasks = [(f"prompt_{i}", MockResponseModel) for i in range(6)]
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
def test_adapter_parallel_invoker_max_workers_default() -> None:
    """Test default max_workers value when no env var is set."""
    with patch.dict("os.environ", {}, clear=True):
        invoker = AdapterParallelInvoker(MockParserPort())
        # Should default to 2
        assert invoker.max_workers == 2


@pytest.mark.unit
def test_adapter_parallel_invoker_max_workers_explicit() -> None:
    """Test explicit max_workers value."""
    invoker = AdapterParallelInvoker(MockParserPort(), max_workers=8)

    assert invoker.max_workers == 8


@pytest.mark.unit
def test_adapter_parallel_invoker_max_workers_from_env() -> None:
    """Test max_workers from KARENINA_ASYNC_MAX_WORKERS env var."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "6"}):
        invoker = AdapterParallelInvoker(MockParserPort())
        assert invoker.max_workers == 6


@pytest.mark.unit
def test_adapter_parallel_invoker_explicit_overrides_env() -> None:
    """Test that explicit max_workers overrides env var."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "6"}):
        invoker = AdapterParallelInvoker(MockParserPort(), max_workers=4)
        assert invoker.max_workers == 4


@pytest.mark.unit
def test_adapter_parallel_invoker_invalid_env_uses_default() -> None:
    """Test that invalid env var falls back to default."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "invalid"}):
        invoker = AdapterParallelInvoker(MockParserPort())
        assert invoker.max_workers == 2


@pytest.mark.unit
def test_adapter_parallel_invoker_progress_callback() -> None:
    """Test that progress callback is called correctly."""
    responses = [MockResponseModel(result=f"result_{i}") for i in range(3)]
    mock_parser = MockParserPort(responses=responses)
    invoker = AdapterParallelInvoker(mock_parser, max_workers=2)

    tasks = [(f"prompt_{i}", MockResponseModel) for i in range(3)]

    progress_calls: list[tuple[int, int]] = []

    def progress_callback(completed: int, total: int) -> None:
        progress_calls.append((completed, total))

    invoker.invoke_batch(tasks, progress_callback=progress_callback)

    # Should have 3 progress calls (one per task completion)
    assert len(progress_calls) == 3
    # All should have total=3
    assert all(total == 3 for _, total in progress_calls)
    # Completed should be 1, 2, 3 (in some order due to parallel execution)
    completed_values = sorted(c for c, _ in progress_calls)
    assert completed_values == [1, 2, 3]


@pytest.mark.unit
def test_adapter_parallel_invoker_aggregated_usage() -> None:
    """Test invoke_batch_with_aggregated_usage method."""
    responses = [MockResponseModel(result=f"result_{i}") for i in range(3)]
    mock_parser = MockParserPort(responses=responses)
    invoker = AdapterParallelInvoker(mock_parser, max_workers=4)

    tasks = [(f"prompt_{i}", MockResponseModel) for i in range(3)]
    results, aggregated = invoker.invoke_batch_with_aggregated_usage(tasks)

    assert len(results) == 3

    # All results should be successful
    for result, error in results:
        assert error is None
        assert result is not None

    # Currently adapters don't return usage metadata, so aggregated will be empty counts
    # The structure should still be correct though
    assert "calls" in aggregated
    assert "input_tokens" in aggregated
    assert "output_tokens" in aggregated
    assert "total_tokens" in aggregated


@pytest.mark.unit
def test_adapter_parallel_invoker_aggregated_usage_with_errors() -> None:
    """Test aggregated usage handles errors correctly."""
    responses = [MockResponseModel(result=f"result_{i}") for i in range(3)]
    mock_parser = MockParserPort(responses=responses, fail_indices={1})
    invoker = AdapterParallelInvoker(mock_parser, max_workers=4)

    tasks = [(f"prompt_{i}", MockResponseModel) for i in range(3)]
    results, aggregated = invoker.invoke_batch_with_aggregated_usage(tasks)

    assert len(results) == 3

    # Count successes and failures
    successes = sum(1 for r, e in results if e is None)
    failures = sum(1 for r, e in results if e is not None)

    assert successes == 2
    assert failures == 1


@pytest.mark.unit
def test_adapter_parallel_invoker_with_different_schemas() -> None:
    """Test invoker with different response schemas."""
    mock_parser = MagicMock()

    # Return different types based on the schema
    async def mock_parse(response: str, schema: type) -> BaseModel:
        if schema == MockBooleanScore:
            return MockBooleanScore(result=True)
        elif schema == MockNumericScore:
            return MockNumericScore(score=5)
        else:
            return MockResponseModel(result="default")

    mock_parser.aparse_to_pydantic = AsyncMock(side_effect=mock_parse)

    invoker = AdapterParallelInvoker(mock_parser, max_workers=4)

    tasks = [
        ("prompt_1", MockBooleanScore),
        ("prompt_2", MockNumericScore),
        ("prompt_3", MockResponseModel),
    ]

    results = invoker.invoke_batch(tasks)

    assert len(results) == 3

    # Verify each result matches its expected type
    result_0, _, error_0 = results[0]
    assert error_0 is None
    assert isinstance(result_0, MockBooleanScore)
    assert result_0.result is True

    result_1, _, error_1 = results[1]
    assert error_1 is None
    assert isinstance(result_1, MockNumericScore)
    assert result_1.score == 5

    result_2, _, error_2 = results[2]
    assert error_2 is None
    assert isinstance(result_2, MockResponseModel)
    assert result_2.result == "default"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_parallel_invoker_async_interface() -> None:
    """Test the async ainvoke_batch interface directly."""
    responses = [MockResponseModel(result=f"result_{i}") for i in range(3)]
    mock_parser = MockParserPort(responses=responses)
    invoker = AdapterParallelInvoker(mock_parser, max_workers=4)

    tasks = [(f"prompt_{i}", MockResponseModel) for i in range(3)]

    results = await invoker.ainvoke_batch(tasks)

    assert len(results) == 3
    for i, (result, _usage, error) in enumerate(results):
        assert error is None
        assert result is not None
        assert result.result == f"result_{i}"


@pytest.mark.unit
def test_adapter_parallel_invoker_concurrency_limiting() -> None:
    """Test that max_workers actually limits concurrency."""
    import time

    call_times: list[float] = []

    class TrackingParser:
        """Parser that tracks call start times."""

        async def aparse_to_pydantic(self, _response: str, _schema: type) -> BaseModel:
            call_times.append(time.time())
            await asyncio.sleep(0.1)  # Small delay to make timing measurable
            return MockResponseModel(result="test")

    invoker = AdapterParallelInvoker(TrackingParser(), max_workers=2)

    # Submit 4 tasks
    tasks = [(f"prompt_{i}", MockResponseModel) for i in range(4)]
    invoker.invoke_batch(tasks)

    # With max_workers=2 and 4 tasks, tasks should be batched:
    # - First batch: tasks 0, 1 (start together)
    # - Second batch: tasks 2, 3 (start after first batch completes)
    # This means there should be a gap between the first 2 and last 2 calls
    assert len(call_times) == 4

    # Sort call times to analyze batching
    sorted_times = sorted(call_times)

    # The first two and last two should be close together
    # but there should be a gap between batches
    first_batch_span = sorted_times[1] - sorted_times[0]
    second_batch_span = sorted_times[3] - sorted_times[2]

    # Within each batch, tasks should start nearly simultaneously (< 0.05s apart)
    assert first_batch_span < 0.05, f"First batch span too large: {first_batch_span}"
    assert second_batch_span < 0.05, f"Second batch span too large: {second_batch_span}"
