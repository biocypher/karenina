"""Unit tests for LLMParallelInvoker utility.

Tests cover:
- Plain text mode (invoke_batch)
- Structured output mode (invoke_batch_structured)
- Empty task list handling
- Result ordering preservation
- Error isolation
- max_workers configuration
- Progress callback
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from karenina.adapters.llm_parallel import LLMParallelInvoker, read_async_config

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class MockResponseModel(BaseModel):
    """Mock response model for testing structured output."""

    result: str


class MockBooleanScore(BaseModel):
    """Mock boolean score response."""

    result: bool


@dataclass
class MockUsageMetadata:
    """Mock usage metadata."""

    input_tokens: int = 10
    output_tokens: int = 5
    total_tokens: int = 15


@dataclass
class MockLLMResponse:
    """Mock LLM response object."""

    content: str
    usage: MockUsageMetadata
    raw: BaseModel | None = None


class MockLLMPort:
    """Mock implementation of LLMPort for testing.

    Provides configurable async LLM behavior including:
    - Delayed responses (for testing parallelism)
    - Controllable failures
    - Call tracking
    """

    def __init__(
        self,
        delay: float = 0.0,
        responses: list[str] | None = None,
        fail_indices: set[int] | None = None,
    ):
        self.delay = delay
        self.responses = responses or []
        self.fail_indices = fail_indices or set()
        self.call_count = 0
        self.call_times: list[float] = []

    async def ainvoke(self, messages: list) -> MockLLMResponse:  # noqa: ARG002
        """Async LLM invocation with configurable behavior."""
        import time

        current = self.call_count
        self.call_count += 1
        self.call_times.append(time.time())

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if current in self.fail_indices:
            raise ValueError(f"Simulated failure at index {current}")

        content = self.responses[current] if current < len(self.responses) else f"response_{current}"

        return MockLLMResponse(
            content=content,
            usage=MockUsageMetadata(),
        )

    def invoke(self, messages: list) -> MockLLMResponse:
        """Sync wrapper for testing."""
        return asyncio.run(self.ainvoke(messages))

    def with_structured_output(self, schema: type[BaseModel]) -> MockStructuredLLMPort:
        """Return a structured output adapter."""
        return MockStructuredLLMPort(self, schema)


class MockStructuredLLMPort:
    """Mock structured output LLM port."""

    def __init__(self, base: MockLLMPort, schema: type[BaseModel]):
        self.base = base
        self.schema = schema

    async def ainvoke(self, messages: list) -> MockLLMResponse:  # noqa: ARG002
        """Return structured output."""
        response = await self.base.ainvoke(messages)
        # Create instance of schema
        if self.schema == MockBooleanScore:
            parsed = MockBooleanScore(result=True)
        else:
            parsed = self.schema.model_validate({"result": response.content})
        return MockLLMResponse(
            content=response.content,
            usage=response.usage,
            raw=parsed,
        )


# =============================================================================
# read_async_config Tests
# =============================================================================


@pytest.mark.unit
def test_read_async_config_defaults() -> None:
    """Test default values when no env vars are set."""
    with patch.dict("os.environ", {}, clear=True):
        enabled, workers = read_async_config()
        assert enabled is True
        assert workers == 2


@pytest.mark.unit
def test_read_async_config_from_env() -> None:
    """Test reading config from environment variables."""
    with patch.dict(
        "os.environ",
        {"KARENINA_ASYNC_ENABLED": "false", "KARENINA_ASYNC_MAX_WORKERS": "8"},
    ):
        enabled, workers = read_async_config()
        assert enabled is False
        assert workers == 8


@pytest.mark.unit
def test_read_async_config_invalid_workers() -> None:
    """Test that invalid max_workers falls back to default."""
    with patch.dict(
        "os.environ",
        {"KARENINA_ASYNC_MAX_WORKERS": "not_a_number"},
    ):
        enabled, workers = read_async_config()
        assert enabled is True
        assert workers == 2


# =============================================================================
# Plain Text Mode (invoke_batch) Tests
# =============================================================================


@pytest.mark.unit
def test_llm_parallel_invoker_plain_text_empty_tasks() -> None:
    """Test that empty task list returns empty results in plain text mode."""
    mock_llm = MockLLMPort()
    invoker = LLMParallelInvoker(mock_llm, max_workers=2)

    results = invoker.invoke_batch([])

    assert results == []
    assert mock_llm.call_count == 0


@pytest.mark.unit
def test_llm_parallel_invoker_plain_text_single_task() -> None:
    """Test plain text mode with a single task."""
    mock_llm = MockLLMPort(responses=["test response"])
    invoker = LLMParallelInvoker(mock_llm, max_workers=2)

    # Task is just a list of messages
    tasks = [[{"role": "user", "content": "Hello"}]]
    results = invoker.invoke_batch(tasks)

    assert len(results) == 1
    response, error = results[0]
    assert error is None
    assert response is not None
    assert response.content == "test response"


@pytest.mark.unit
def test_llm_parallel_invoker_plain_text_multiple_tasks() -> None:
    """Test plain text mode with multiple tasks."""
    mock_llm = MockLLMPort(responses=[f"response_{i}" for i in range(3)])
    invoker = LLMParallelInvoker(mock_llm, max_workers=4)

    tasks = [[{"role": "user", "content": f"Question {i}"}] for i in range(3)]
    results = invoker.invoke_batch(tasks)

    assert len(results) == 3
    for i, (response, error) in enumerate(results):
        assert error is None
        assert response is not None
        assert response.content == f"response_{i}"


@pytest.mark.unit
def test_llm_parallel_invoker_plain_text_error_isolation() -> None:
    """Test that errors are isolated in plain text mode."""
    mock_llm = MockLLMPort(
        responses=[f"response_{i}" for i in range(3)],
        fail_indices={1},
    )
    invoker = LLMParallelInvoker(mock_llm, max_workers=4)

    tasks = [[{"role": "user", "content": f"Question {i}"}] for i in range(3)]
    results = invoker.invoke_batch(tasks)

    assert len(results) == 3

    # First task should succeed
    response_0, error_0 = results[0]
    assert error_0 is None
    assert response_0 is not None

    # Second task should fail
    response_1, error_1 = results[1]
    assert response_1 is None
    assert error_1 is not None
    assert "Simulated failure" in str(error_1)

    # Third task should succeed
    response_2, error_2 = results[2]
    assert error_2 is None
    assert response_2 is not None


# =============================================================================
# Structured Output Mode (invoke_batch_structured) Tests
# =============================================================================


@pytest.mark.unit
def test_llm_parallel_invoker_structured_empty_tasks() -> None:
    """Test that empty task list returns empty results in structured mode."""
    mock_llm = MockLLMPort()
    invoker = LLMParallelInvoker(mock_llm, max_workers=2)

    results = invoker.invoke_batch_structured([])

    assert results == []
    assert mock_llm.call_count == 0


@pytest.mark.unit
def test_llm_parallel_invoker_structured_single_task() -> None:
    """Test structured output mode with a single task."""
    mock_llm = MockLLMPort(responses=["test"])
    invoker = LLMParallelInvoker(mock_llm, max_workers=2)

    # Task is (messages, schema)
    tasks = [([{"role": "user", "content": "Hello"}], MockResponseModel)]
    results = invoker.invoke_batch_structured(tasks)

    assert len(results) == 1
    result, usage, error = results[0]
    assert error is None
    assert result is not None
    assert isinstance(result, MockResponseModel)
    assert usage is not None
    assert "input_tokens" in usage


@pytest.mark.unit
def test_llm_parallel_invoker_structured_multiple_tasks() -> None:
    """Test structured output mode with multiple tasks."""
    mock_llm = MockLLMPort(responses=[f"response_{i}" for i in range(3)])
    invoker = LLMParallelInvoker(mock_llm, max_workers=4)

    tasks = [([{"role": "user", "content": f"Question {i}"}], MockResponseModel) for i in range(3)]
    results = invoker.invoke_batch_structured(tasks)

    assert len(results) == 3
    for _i, (result, _usage, error) in enumerate(results):
        assert error is None
        assert result is not None
        assert isinstance(result, MockResponseModel)


@pytest.mark.unit
def test_llm_parallel_invoker_structured_error_isolation() -> None:
    """Test that errors are isolated in structured mode."""
    mock_llm = MockLLMPort(
        responses=[f"response_{i}" for i in range(3)],
        fail_indices={1},
    )
    invoker = LLMParallelInvoker(mock_llm, max_workers=4)

    tasks = [([{"role": "user", "content": f"Question {i}"}], MockResponseModel) for i in range(3)]
    results = invoker.invoke_batch_structured(tasks)

    assert len(results) == 3

    # First task should succeed
    result_0, usage_0, error_0 = results[0]
    assert error_0 is None
    assert result_0 is not None

    # Second task should fail
    result_1, usage_1, error_1 = results[1]
    assert result_1 is None
    assert usage_1 is None
    assert error_1 is not None
    assert "Simulated failure" in str(error_1)

    # Third task should succeed
    result_2, usage_2, error_2 = results[2]
    assert error_2 is None
    assert result_2 is not None


# =============================================================================
# Common Configuration Tests
# =============================================================================


@pytest.mark.unit
def test_llm_parallel_invoker_max_workers_default() -> None:
    """Test default max_workers value when no env var is set."""
    with patch.dict("os.environ", {}, clear=True):
        invoker = LLMParallelInvoker(MockLLMPort())
        assert invoker.max_workers == 2


@pytest.mark.unit
def test_llm_parallel_invoker_max_workers_explicit() -> None:
    """Test explicit max_workers value."""
    invoker = LLMParallelInvoker(MockLLMPort(), max_workers=8)
    assert invoker.max_workers == 8


@pytest.mark.unit
def test_llm_parallel_invoker_max_workers_from_env() -> None:
    """Test max_workers from KARENINA_ASYNC_MAX_WORKERS env var."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "6"}):
        invoker = LLMParallelInvoker(MockLLMPort())
        assert invoker.max_workers == 6


@pytest.mark.unit
def test_llm_parallel_invoker_progress_callback_plain_text() -> None:
    """Test progress callback in plain text mode."""
    mock_llm = MockLLMPort(responses=[f"response_{i}" for i in range(3)])
    invoker = LLMParallelInvoker(mock_llm, max_workers=2)

    tasks = [[{"role": "user", "content": f"Question {i}"}] for i in range(3)]

    progress_calls: list[tuple[int, int]] = []

    def progress_callback(completed: int, total: int) -> None:
        progress_calls.append((completed, total))

    invoker.invoke_batch(tasks, progress_callback=progress_callback)

    assert len(progress_calls) == 3
    assert all(total == 3 for _, total in progress_calls)
    completed_values = sorted(c for c, _ in progress_calls)
    assert completed_values == [1, 2, 3]


@pytest.mark.unit
def test_llm_parallel_invoker_progress_callback_structured() -> None:
    """Test progress callback in structured mode."""
    mock_llm = MockLLMPort(responses=[f"response_{i}" for i in range(3)])
    invoker = LLMParallelInvoker(mock_llm, max_workers=2)

    tasks = [([{"role": "user", "content": f"Question {i}"}], MockResponseModel) for i in range(3)]

    progress_calls: list[tuple[int, int]] = []

    def progress_callback(completed: int, total: int) -> None:
        progress_calls.append((completed, total))

    invoker.invoke_batch_structured(tasks, progress_callback=progress_callback)

    assert len(progress_calls) == 3
    assert all(total == 3 for _, total in progress_calls)
    completed_values = sorted(c for c, _ in progress_calls)
    assert completed_values == [1, 2, 3]


# =============================================================================
# Async Interface Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_parallel_invoker_async_plain_text() -> None:
    """Test async interface for plain text mode."""
    mock_llm = MockLLMPort(responses=[f"response_{i}" for i in range(3)])
    invoker = LLMParallelInvoker(mock_llm, max_workers=4)

    tasks = [[{"role": "user", "content": f"Question {i}"}] for i in range(3)]

    results = await invoker.ainvoke_batch(tasks)

    assert len(results) == 3
    for i, (response, error) in enumerate(results):
        assert error is None
        assert response is not None
        assert response.content == f"response_{i}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_parallel_invoker_async_structured() -> None:
    """Test async interface for structured mode."""
    mock_llm = MockLLMPort(responses=[f"response_{i}" for i in range(3)])
    invoker = LLMParallelInvoker(mock_llm, max_workers=4)

    tasks = [([{"role": "user", "content": f"Question {i}"}], MockResponseModel) for i in range(3)]

    results = await invoker.ainvoke_batch_structured(tasks)

    assert len(results) == 3
    for _i, (result, _usage, error) in enumerate(results):
        assert error is None
        assert result is not None


# =============================================================================
# Concurrency Limiting Tests
# =============================================================================


@pytest.mark.unit
def test_llm_parallel_invoker_concurrency_limiting() -> None:
    """Test that max_workers actually limits concurrency."""
    import time

    call_times: list[float] = []

    class TrackingLLM:
        """LLM that tracks call start times."""

        async def ainvoke(self, messages: list) -> MockLLMResponse:  # noqa: ARG002
            call_times.append(time.time())
            await asyncio.sleep(0.1)
            return MockLLMResponse(content="test", usage=MockUsageMetadata())

        def with_structured_output(self, schema: type) -> TrackingLLM:  # noqa: ARG002
            return self

    invoker = LLMParallelInvoker(TrackingLLM(), max_workers=2)

    # Submit 4 tasks
    tasks = [[{"role": "user", "content": f"Question {i}"}] for i in range(4)]
    invoker.invoke_batch(tasks)

    assert len(call_times) == 4

    # Sort call times to analyze batching
    sorted_times = sorted(call_times)

    # The first two should start together, then next two after first batch
    first_batch_span = sorted_times[1] - sorted_times[0]
    second_batch_span = sorted_times[3] - sorted_times[2]

    # Within each batch, tasks should start nearly simultaneously
    assert first_batch_span < 0.05, f"First batch span too large: {first_batch_span}"
    assert second_batch_span < 0.05, f"Second batch span too large: {second_batch_span}"
