"""Active-time budgets that can pause during tool calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


EventSink = Callable[[dict[str, Any]], Awaitable[None] | None]


class ActiveTimeTracker:
    """Track active (non-tool) time for a stage budget.

    Agents may emit events::

        {"kind": "tool_activity", "tool_id": "...", "status": "started"|"completed"|...}
        {"kind": "tool_activity", "name": "query_cypher", "status": "started"}

    While any matching tool is active, the budget clock pauses.
    """

    def __init__(
        self,
        *,
        exclude_tool_time: bool = True,
        tool_name_substrings: list[str] | None = None,
        downstream: EventSink | None = None,
    ) -> None:
        self.exclude_tool_time = exclude_tool_time
        self.tool_name_substrings = [s.lower() for s in (tool_name_substrings or ["tool", "query", "search"])]
        self.downstream = downstream
        self.active_tools: set[str] = set()
        self.known_tools: set[str] = set()
        self.tool_time_seconds: float = 0.0
        self._tool_started_at: dict[str, float] = {}
        self.changed = asyncio.Event()

    @property
    def paused(self) -> bool:
        return self.exclude_tool_time and bool(self.active_tools)

    def _matches_tool(self, name: str) -> bool:
        low = name.lower()
        if not self.tool_name_substrings:
            return True
        return any(sub in low for sub in self.tool_name_substrings)

    async def emit(self, event: dict[str, Any]) -> None:
        if event.get("kind") == "tool_activity":
            tool_id = str(event.get("tool_id") or event.get("name") or "")
            name = str(event.get("name") or tool_id)
            status = str(event.get("status") or "").lower()
            if tool_id and (self._matches_tool(name) or tool_id in self.known_tools):
                self.known_tools.add(tool_id)
                before = self.paused
                loop = asyncio.get_running_loop()
                if status in ("started", "start", "running", "pending", "in_progress"):
                    if tool_id not in self.active_tools:
                        self.active_tools.add(tool_id)
                        self._tool_started_at[tool_id] = loop.time()
                elif status in ("completed", "complete", "failed", "error", "cancelled", "canceled", "ended"):
                    if tool_id in self.active_tools:
                        started = self._tool_started_at.pop(tool_id, None)
                        if started is not None:
                            self.tool_time_seconds += max(0.0, loop.time() - started)
                        self.active_tools.discard(tool_id)
                if before != self.paused:
                    self.changed.set()
        if self.downstream is not None:
            result = self.downstream(event)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                await result  # type: ignore[arg-type]

    async def wait(self, task: asyncio.Task[Any], budget_seconds: float) -> bool:
        """Wait until task completes or active budget is exhausted.

        Returns True if the task finished within the active budget.
        """
        remaining = max(0.0, budget_seconds)
        loop = asyncio.get_running_loop()
        while not task.done() and remaining > 0:
            was_paused = self.paused
            started = loop.time()
            self.changed.clear()
            changed_task = asyncio.create_task(self.changed.wait())
            done, _ = await asyncio.wait(
                {task, changed_task},
                timeout=None if was_paused else remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not was_paused:
                remaining -= loop.time() - started
            if task in done:
                changed_task.cancel()
                await asyncio.gather(changed_task, return_exceptions=True)
                return True
            if changed_task not in done:
                changed_task.cancel()
                await asyncio.gather(changed_task, return_exceptions=True)
                return False
        return task.done()
