"""Tests for active-time tracking (tool-time exclusion)."""

from __future__ import annotations

import asyncio

import pytest

from karenina.question_qc.timing import ActiveTimeTracker


@pytest.mark.asyncio
async def test_active_time_pauses_during_tools() -> None:
    tracker = ActiveTimeTracker(exclude_tool_time=True, tool_name_substrings=["query"])

    async def work() -> str:
        await tracker.emit(
            {"kind": "tool_activity", "tool_id": "1", "name": "query_cypher", "status": "started"}
        )
        await asyncio.sleep(0.25)
        await tracker.emit(
            {"kind": "tool_activity", "tool_id": "1", "name": "query_cypher", "status": "completed"}
        )
        await asyncio.sleep(0.05)
        return "ok"

    task = asyncio.create_task(work())
    # Active budget 0.1s — would fail if tool 0.25s counted
    ok = await tracker.wait(task, 0.1)
    assert ok is True
    assert await task == "ok"
    assert tracker.tool_time_seconds >= 0.2


@pytest.mark.asyncio
async def test_active_time_exhausts_without_tools() -> None:
    tracker = ActiveTimeTracker(exclude_tool_time=True)

    async def work() -> str:
        await asyncio.sleep(0.3)
        return "late"

    task = asyncio.create_task(work())
    ok = await tracker.wait(task, 0.05)
    assert ok is False
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
