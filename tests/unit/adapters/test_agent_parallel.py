"""Unit tests for AgentParallelInvoker utility.

Tests cover:
- Empty task list handling
- Result ordering preservation
- Error isolation (one failure doesn't block others)
- Aggregated metrics
- max_workers configuration
- Environment variable handling
- Progress callback
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from karenina.adapters.agent_parallel import AgentParallelInvoker

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@dataclass
class MockUsageMetadata:
    """Mock usage metadata for agents."""

    input_tokens: int = 100
    output_tokens: int = 50
    total_tokens: int = 150
    cost_usd: float = 0.01


@dataclass
class MockAgentResult:
    """Mock AgentResult for testing."""

    final_response: str
    raw_trace: str
    trace_messages: list
    usage: MockUsageMetadata
    turns: int
    limit_reached: bool
    session_id: str | None = None
    actual_model: str | None = None


class MockAgentPort:
    """Mock implementation of AgentPort for testing.

    Provides configurable async agent behavior including:
    - Delayed responses (for testing parallelism)
    - Controllable failures
    - Call tracking
    """

    def __init__(
        self,
        delay: float = 0.0,
        responses: list[str] | None = None,
        fail_indices: set[int] | None = None,
        turns_per_response: int = 1,
        limit_reached_indices: set[int] | None = None,
    ):
        self.delay = delay
        self.responses = responses or []
        self.fail_indices = fail_indices or set()
        self.turns_per_response = turns_per_response
        self.limit_reached_indices = limit_reached_indices or set()
        self.call_count = 0
        self.call_times: list[float] = []

    async def run(
        self,
        messages: list,  # noqa: ARG002
        tools: list | None = None,  # noqa: ARG002
        mcp_servers: dict | None = None,  # noqa: ARG002
        config: object | None = None,  # noqa: ARG002
    ) -> MockAgentResult:
        """Async agent run with configurable behavior."""
        import time

        current = self.call_count
        self.call_count += 1
        self.call_times.append(time.time())

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if current in self.fail_indices:
            raise ValueError(f"Simulated failure at index {current}")

        response = self.responses[current] if current < len(self.responses) else f"response_{current}"

        return MockAgentResult(
            final_response=response,
            raw_trace=f"--- AI Message ---\n{response}",
            trace_messages=[],
            usage=MockUsageMetadata(),
            turns=self.turns_per_response,
            limit_reached=current in self.limit_reached_indices,
        )

    def run_sync(
        self,
        messages: list,
        tools: list | None = None,
        mcp_servers: dict | None = None,
        config: object | None = None,
    ) -> MockAgentResult:
        """Sync wrapper for testing."""
        return asyncio.run(self.run(messages, tools, mcp_servers, config))


# =============================================================================
# AgentParallelInvoker Tests
# =============================================================================


@pytest.mark.unit
def test_agent_parallel_invoker_empty_tasks() -> None:
    """Test that empty task list returns empty results."""
    mock_agent = MockAgentPort()
    invoker = AgentParallelInvoker(mock_agent, max_workers=2)

    results = invoker.invoke_batch([])

    assert results == []
    assert mock_agent.call_count == 0


@pytest.mark.unit
def test_agent_parallel_invoker_single_task() -> None:
    """Test invoker with a single task."""
    mock_agent = MockAgentPort(responses=["test response"])
    invoker = AgentParallelInvoker(mock_agent, max_workers=2)

    # Task is (messages, mcp_servers, config)
    tasks = [([{"role": "user", "content": "Hello"}], None, None)]
    results = invoker.invoke_batch(tasks)

    assert len(results) == 1
    result, error = results[0]
    assert error is None
    assert result is not None
    assert result.final_response == "test response"


@pytest.mark.unit
def test_agent_parallel_invoker_result_ordering_preserved() -> None:
    """Test that results are returned in input order regardless of completion order."""
    responses = [f"response_{i}" for i in range(5)]
    mock_agent = MockAgentPort(responses=responses)
    invoker = AgentParallelInvoker(mock_agent, max_workers=4)

    tasks = [([{"role": "user", "content": f"Question {i}"}], None, None) for i in range(5)]
    results = invoker.invoke_batch(tasks)

    assert len(results) == 5
    for i, (result, error) in enumerate(results):
        assert error is None
        assert result is not None
        assert result.final_response == f"response_{i}"


@pytest.mark.unit
def test_agent_parallel_invoker_error_isolation() -> None:
    """Test that one failed task doesn't block others."""
    mock_agent = MockAgentPort(
        responses=[f"success_{i}" for i in range(5)],
        fail_indices={2},
    )
    invoker = AgentParallelInvoker(mock_agent, max_workers=4)

    tasks = [([{"role": "user", "content": f"Question {i}"}], None, None) for i in range(5)]
    results = invoker.invoke_batch(tasks)

    assert len(results) == 5

    # Count successes and failures
    successes = sum(1 for r, e in results if e is None)
    failures = sum(1 for r, e in results if e is not None)

    assert successes == 4
    assert failures == 1

    # Verify the failed one has proper error
    result, error = results[2]
    assert result is None
    assert error is not None
    assert "Simulated failure" in str(error)


@pytest.mark.unit
def test_agent_parallel_invoker_partial_failures() -> None:
    """Test handling of multiple partial failures."""
    mock_agent = MockAgentPort(
        responses=[f"success_{i}" for i in range(6)],
        fail_indices={1, 3, 5},
    )
    invoker = AgentParallelInvoker(mock_agent, max_workers=4)

    tasks = [([{"role": "user", "content": f"Question {i}"}], None, None) for i in range(6)]
    results = invoker.invoke_batch(tasks)

    assert len(results) == 6

    # Verify pattern: success, fail, success, fail, success, fail
    for i, (result, error) in enumerate(results):
        if i % 2 == 0:
            assert error is None
            assert result is not None
        else:
            assert error is not None
            assert result is None


@pytest.mark.unit
def test_agent_parallel_invoker_max_workers_default() -> None:
    """Test default max_workers value when no env var is set."""
    with patch.dict("os.environ", {}, clear=True):
        invoker = AgentParallelInvoker(MockAgentPort())
        assert invoker.max_workers == 2


@pytest.mark.unit
def test_agent_parallel_invoker_max_workers_explicit() -> None:
    """Test explicit max_workers value."""
    invoker = AgentParallelInvoker(MockAgentPort(), max_workers=8)
    assert invoker.max_workers == 8


@pytest.mark.unit
def test_agent_parallel_invoker_max_workers_from_env() -> None:
    """Test max_workers from KARENINA_ASYNC_MAX_WORKERS env var."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "6"}):
        invoker = AgentParallelInvoker(MockAgentPort())
        assert invoker.max_workers == 6


@pytest.mark.unit
def test_agent_parallel_invoker_explicit_overrides_env() -> None:
    """Test that explicit max_workers overrides env var."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "6"}):
        invoker = AgentParallelInvoker(MockAgentPort(), max_workers=4)
        assert invoker.max_workers == 4


@pytest.mark.unit
def test_agent_parallel_invoker_invalid_env_uses_default() -> None:
    """Test that invalid env var falls back to default."""
    with patch.dict("os.environ", {"KARENINA_ASYNC_MAX_WORKERS": "invalid"}):
        invoker = AgentParallelInvoker(MockAgentPort())
        assert invoker.max_workers == 2


@pytest.mark.unit
def test_agent_parallel_invoker_progress_callback() -> None:
    """Test that progress callback is called correctly."""
    responses = [f"response_{i}" for i in range(3)]
    mock_agent = MockAgentPort(responses=responses)
    invoker = AgentParallelInvoker(mock_agent, max_workers=2)

    tasks = [([{"role": "user", "content": f"Question {i}"}], None, None) for i in range(3)]

    progress_calls: list[tuple[int, int]] = []

    def progress_callback(completed: int, total: int) -> None:
        progress_calls.append((completed, total))

    invoker.invoke_batch(tasks, progress_callback=progress_callback)

    assert len(progress_calls) == 3
    assert all(total == 3 for _, total in progress_calls)
    completed_values = sorted(c for c, _ in progress_calls)
    assert completed_values == [1, 2, 3]


@pytest.mark.unit
def test_agent_parallel_invoker_aggregated_metrics() -> None:
    """Test invoke_batch_with_aggregated_metrics method."""
    mock_agent = MockAgentPort(
        responses=[f"response_{i}" for i in range(3)],
        turns_per_response=2,
    )
    invoker = AgentParallelInvoker(mock_agent, max_workers=4)

    tasks = [([{"role": "user", "content": f"Question {i}"}], None, None) for i in range(3)]
    results, metrics = invoker.invoke_batch_with_aggregated_metrics(tasks)

    assert len(results) == 3

    # All results should be successful
    for result, error in results:
        assert error is None
        assert result is not None

    # Check aggregated metrics
    assert metrics["successful_runs"] == 3
    assert metrics["failed_runs"] == 0
    assert metrics["total_turns"] == 6  # 3 tasks * 2 turns each
    assert metrics["limits_reached"] == 0
    assert metrics["input_tokens"] == 300  # 3 tasks * 100 each
    assert metrics["output_tokens"] == 150  # 3 tasks * 50 each
    assert metrics["total_tokens"] == 450  # 3 tasks * 150 each
    assert metrics["cost_usd"] == pytest.approx(0.03)  # 3 tasks * 0.01 each


@pytest.mark.unit
def test_agent_parallel_invoker_aggregated_metrics_with_errors() -> None:
    """Test aggregated metrics handles errors correctly."""
    mock_agent = MockAgentPort(
        responses=[f"response_{i}" for i in range(3)],
        fail_indices={1},
        turns_per_response=2,
    )
    invoker = AgentParallelInvoker(mock_agent, max_workers=4)

    tasks = [([{"role": "user", "content": f"Question {i}"}], None, None) for i in range(3)]
    results, metrics = invoker.invoke_batch_with_aggregated_metrics(tasks)

    assert len(results) == 3

    assert metrics["successful_runs"] == 2
    assert metrics["failed_runs"] == 1
    assert metrics["total_turns"] == 4  # 2 successful * 2 turns each
    assert metrics["input_tokens"] == 200  # 2 successful * 100 each


@pytest.mark.unit
def test_agent_parallel_invoker_aggregated_metrics_with_limits() -> None:
    """Test aggregated metrics tracks limit_reached correctly."""
    mock_agent = MockAgentPort(
        responses=[f"response_{i}" for i in range(3)],
        limit_reached_indices={0, 2},
    )
    invoker = AgentParallelInvoker(mock_agent, max_workers=4)

    tasks = [([{"role": "user", "content": f"Question {i}"}], None, None) for i in range(3)]
    results, metrics = invoker.invoke_batch_with_aggregated_metrics(tasks)

    assert metrics["successful_runs"] == 3
    assert metrics["limits_reached"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_parallel_invoker_async_interface() -> None:
    """Test the async ainvoke_batch interface directly."""
    responses = [f"response_{i}" for i in range(3)]
    mock_agent = MockAgentPort(responses=responses)
    invoker = AgentParallelInvoker(mock_agent, max_workers=4)

    tasks = [([{"role": "user", "content": f"Question {i}"}], None, None) for i in range(3)]

    results = await invoker.ainvoke_batch(tasks)

    assert len(results) == 3
    for i, (result, error) in enumerate(results):
        assert error is None
        assert result is not None
        assert result.final_response == f"response_{i}"


@pytest.mark.unit
def test_agent_parallel_invoker_concurrency_limiting() -> None:
    """Test that max_workers actually limits concurrency."""
    import time

    call_times: list[float] = []

    class TrackingAgent:
        """Agent that tracks call start times."""

        async def run(
            self,
            messages: list,  # noqa: ARG002
            tools: list | None = None,  # noqa: ARG002
            mcp_servers: dict | None = None,  # noqa: ARG002
            config: object | None = None,  # noqa: ARG002
        ) -> MockAgentResult:
            call_times.append(time.time())
            await asyncio.sleep(0.1)
            return MockAgentResult(
                final_response="test",
                raw_trace="test",
                trace_messages=[],
                usage=MockUsageMetadata(),
                turns=1,
                limit_reached=False,
            )

    invoker = AgentParallelInvoker(TrackingAgent(), max_workers=2)

    # Submit 4 tasks
    tasks = [([{"role": "user", "content": f"Question {i}"}], None, None) for i in range(4)]
    invoker.invoke_batch(tasks)

    assert len(call_times) == 4

    # Sort call times to analyze batching
    sorted_times = sorted(call_times)

    # The first two and last two should be close together
    first_batch_span = sorted_times[1] - sorted_times[0]
    second_batch_span = sorted_times[3] - sorted_times[2]

    # Within each batch, tasks should start nearly simultaneously
    assert first_batch_span < 0.05, f"First batch span too large: {first_batch_span}"
    assert second_batch_span < 0.05, f"Second batch span too large: {second_batch_span}"


@pytest.mark.unit
def test_agent_parallel_invoker_with_mcp_servers() -> None:
    """Test that MCP servers are passed correctly to agent."""
    mock_agent = MockAgentPort(responses=["test"])
    invoker = AgentParallelInvoker(mock_agent, max_workers=2)

    mcp_servers = {"filesystem": {"command": "npx", "args": ["server"]}}
    tasks = [([{"role": "user", "content": "Hello"}], mcp_servers, None)]

    results = invoker.invoke_batch(tasks)

    assert len(results) == 1
    result, error = results[0]
    assert error is None
    assert result is not None
