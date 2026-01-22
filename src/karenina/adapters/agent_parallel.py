"""Parallel invocation utility for AgentPort adapters.

This module provides a reusable utility for executing multiple agent invocations
in parallel using asyncio.gather() with semaphore-based concurrency limiting.

Key characteristics:
- Uses asyncio.gather() for true async parallelism
- Preserves result ordering (critical requirement)
- Per-task error isolation (failed tasks don't block others)
- Integrates with existing BlockingPortal from batch_runner.py
- Same max_workers configuration pattern as LLMParallelInvoker

Environment Variables:
- KARENINA_ASYNC_ENABLED: Enable/disable parallel execution (default: true)
- KARENINA_ASYNC_MAX_WORKERS: Max concurrent workers (default: 2)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from ._parallel_base import get_max_workers, sync_invoke_via_portal

if TYPE_CHECKING:
    from ..ports import Message
    from ..ports.agent import AgentConfig, AgentPort, AgentResult, MCPServerConfig

logger = logging.getLogger(__name__)


class AgentParallelInvoker:
    """Execute multiple AgentPort invocations in parallel using asyncio.gather().

    This utility is designed for scenarios where multiple independent agent runs
    need to be made (e.g., generating multiple answers in parallel). By running
    these calls in parallel, we achieve significant speedup.

    Similar to LLMParallelInvoker but works with AgentPort:
    - asyncio.gather() with semaphore for concurrency control
    - Thread-safe result collection
    - Per-task error handling (failed tasks don't block others)
    - Aggregated metrics: total_turns, limits_reached, cost_usd

    Example usage:
        invoker = AgentParallelInvoker(agent, max_workers=4)
        tasks = [
            ([Message.user("Question 1")], mcp_servers, config),
            ([Message.user("Question 2")], mcp_servers, config),
        ]
        results = invoker.invoke_batch(tasks)
        for result, error in results:
            if error:
                print(f"Task failed: {error}")
            else:
                print(f"Response: {result.final_response}")
    """

    def __init__(
        self,
        agent: AgentPort,
        max_workers: int | None = None,
    ):
        """Initialize the agent parallel invoker.

        Args:
            agent: AgentPort implementation for agent operations.
            max_workers: Maximum number of concurrent agent runs. If None,
                        defaults to KARENINA_ASYNC_MAX_WORKERS env var or 2.
        """
        self.agent = agent
        self._max_workers = max_workers

    @property
    def max_workers(self) -> int:
        """Get the effective max_workers value."""
        return get_max_workers(self._max_workers)

    async def ainvoke_batch(
        self,
        tasks: Sequence[tuple[list[Message], dict[str, MCPServerConfig] | None, AgentConfig | None]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[AgentResult | None, Exception | None]]:
        """Invoke agent for multiple message/config pairs in parallel (async).

        Each task is a tuple of (messages, mcp_servers, config). The agent is
        invoked for each task, and results are collected in order.

        Args:
            tasks: Sequence of (messages, mcp_servers, config) tuples.
                   messages: List of Message objects to send.
                   mcp_servers: Optional MCP server configurations.
                   config: Optional AgentConfig for execution parameters.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            List of (result, error) tuples in same order as input.
            - result: AgentResult with final_response, trace, usage, etc., or None if error
            - error: Exception if the task failed, or None if success
        """
        if not tasks:
            return []

        total = len(tasks)
        progress_lock = asyncio.Lock()
        completed_count = 0
        semaphore = asyncio.Semaphore(self.max_workers)

        async def execute_task(
            index: int,
        ) -> tuple[int, AgentResult | None, Exception | None]:
            """Execute a single task and return (index, result, error)."""
            nonlocal completed_count

            messages, mcp_servers, config = tasks[index]

            async with semaphore:
                try:
                    result = await self.agent.run(
                        messages=messages,
                        mcp_servers=mcp_servers,
                        config=config,
                    )
                    return index, result, None
                except Exception as e:
                    logger.debug(f"AgentParallelInvoker: Task {index} failed: {e}")
                    return index, None, e
                finally:
                    if progress_callback:
                        async with progress_lock:
                            completed_count += 1
                            progress_callback(completed_count, total)

        task_coroutines = [execute_task(i) for i in range(total)]
        raw_results = await asyncio.gather(*task_coroutines, return_exceptions=False)

        # Build ordered results list
        results: list[tuple[AgentResult | None, Exception | None]] = [(None, None)] * total
        for index, result, error in raw_results:
            results[index] = (result, error)

        return results

    def invoke_batch(
        self,
        tasks: Sequence[tuple[list[Message], dict[str, MCPServerConfig] | None, AgentConfig | None]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[tuple[AgentResult | None, Exception | None]]:
        """Invoke agent for multiple message/config pairs in parallel (sync).

        This is a sync wrapper around ainvoke_batch(). Uses the shared BlockingPortal
        if available, otherwise falls back to asyncio.run().

        Args:
            tasks: Sequence of (messages, mcp_servers, config) tuples.
                   messages: List of Message objects to send.
                   mcp_servers: Optional MCP server configurations.
                   config: Optional AgentConfig for execution parameters.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            List of (result, error) tuples in same order as input.
            - result: AgentResult with final_response, trace, usage, etc., or None if error
            - error: Exception if the task failed, or None if success
        """
        return sync_invoke_via_portal(self.ainvoke_batch, tasks, progress_callback)

    def invoke_batch_with_aggregated_metrics(
        self,
        tasks: Sequence[tuple[list[Message], dict[str, MCPServerConfig] | None, AgentConfig | None]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[tuple[AgentResult | None, Exception | None]], dict[str, Any]]:
        """Invoke agent for multiple tasks and aggregate metrics.

        Similar to invoke_batch but aggregates metrics across all agent runs
        into a single dictionary with totals.

        Args:
            tasks: Sequence of (messages, mcp_servers, config) tuples.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            Tuple of (results_list, aggregated_metrics) where:
            - results_list: List of (result, error) tuples in input order
            - aggregated_metrics: Dict with aggregated metrics across all runs
        """
        results = self.invoke_batch(tasks, progress_callback)

        aggregated_metrics: dict[str, Any] = {
            "total_turns": 0,
            "limits_reached": 0,
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "successful_runs": 0,
            "failed_runs": 0,
        }

        for result, error in results:
            if error:
                aggregated_metrics["failed_runs"] += 1
            elif result is not None:
                aggregated_metrics["successful_runs"] += 1
                aggregated_metrics["total_turns"] += result.turns
                if result.limit_reached:
                    aggregated_metrics["limits_reached"] += 1

                # Aggregate usage from result.usage
                if result.usage:
                    aggregated_metrics["input_tokens"] += getattr(result.usage, "input_tokens", 0) or 0
                    aggregated_metrics["output_tokens"] += getattr(result.usage, "output_tokens", 0) or 0
                    aggregated_metrics["total_tokens"] += getattr(result.usage, "total_tokens", 0) or 0
                    aggregated_metrics["cost_usd"] += getattr(result.usage, "cost_usd", 0.0) or 0.0

        return results, aggregated_metrics
