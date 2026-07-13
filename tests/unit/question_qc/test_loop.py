"""Unit tests for QcLoop with fake agents."""

from __future__ import annotations

import asyncio
import json

import pytest

from karenina.question_qc.loop import QcLoop
from karenina.question_qc.models import QcClassification, QcQuestion, QcTurn


class ScriptedAgent:
    """Returns canned responses in order; supports session protocol."""

    def __init__(self, responses: list[str], *, delay: float = 0.0) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.stages: list[str | None] = []
        self.delay = delay
        self._last: QcTurn | None = None

    def reset_session(self) -> None:
        self._last = None

    def snapshot_turn(self) -> QcTurn | None:
        return self._last

    async def cancel_turn(self) -> None:
        return None

    async def run_turn(
        self,
        prompt: str,
        *,
        active_budget_seconds: float | None = None,
        exclude_tool_time: bool = True,
        event_sink=None,
        stage: str | None = None,
    ) -> QcTurn:
        del exclude_tool_time, event_sink, active_budget_seconds
        self.prompts.append(prompt)
        self.stages.append(stage)
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.responses:
            turn = QcTurn(
                text="",
                error="no more scripted responses",
                prompt=prompt,
                stop_reason="error",
                stage=stage,
            )
            self._last = turn
            return turn
        text = self.responses.pop(0)
        turn = QcTurn(
            text=text,
            prompt=prompt,
            stop_reason="completed",
            raw_trace=f"--- AI ---\n{text}",
            trace_messages=[{"role": "assistant", "content": text, "block_index": 0}],
            turns=1,
            stage=stage,
            wall_time_seconds=0.01,
            active_time_seconds=0.01,
            tool_time_seconds=0.0,
        )
        self._last = turn
        return turn


class ToolAwareSlowAgent:
    """Spends time in a tool (excluded) then returns JSON within active budget."""

    def __init__(self, text: str, tool_delay: float, after_tool_delay: float = 0.0) -> None:
        self.text = text
        self.tool_delay = tool_delay
        self.after_tool_delay = after_tool_delay
        self._last: QcTurn | None = None
        self._task: asyncio.Task | None = None

    def reset_session(self) -> None:
        self._last = None

    def snapshot_turn(self) -> QcTurn | None:
        return self._last

    async def cancel_turn(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def run_turn(
        self,
        prompt: str,
        *,
        active_budget_seconds: float | None = None,
        exclude_tool_time: bool = True,
        event_sink=None,
        stage: str | None = None,
    ) -> QcTurn:
        from karenina.question_qc.timing import ActiveTimeTracker

        tracker = ActiveTimeTracker(exclude_tool_time=exclude_tool_time, downstream=event_sink)

        async def work() -> QcTurn:
            await tracker.emit(
                {"kind": "tool_activity", "tool_id": "t1", "name": "query_cypher", "status": "started"}
            )
            await asyncio.sleep(self.tool_delay)
            await tracker.emit(
                {"kind": "tool_activity", "tool_id": "t1", "name": "query_cypher", "status": "completed"}
            )
            if self.after_tool_delay:
                await asyncio.sleep(self.after_tool_delay)
            return QcTurn(
                text=self.text,
                prompt=prompt,
                stop_reason="completed",
                raw_trace=self.text,
                trace_messages=[
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "t1", "name": "query_cypher", "input": {}}],
                        "block_index": 0,
                    },
                    {
                        "role": "tool",
                        "content": "ok",
                        "tool_result": {"tool_use_id": "t1", "is_error": False},
                        "block_index": 1,
                    },
                    {"role": "assistant", "content": self.text, "block_index": 2},
                ],
                turns=2,
                stage=stage,
                tool_time_seconds=self.tool_delay,
            )

        task = asyncio.create_task(work())
        self._task = task
        if active_budget_seconds is not None:
            ok = await tracker.wait(task, active_budget_seconds)
            if not ok:
                await self.cancel_turn()
                await asyncio.gather(task, return_exceptions=True)
                turn = QcTurn(
                    text="",
                    error="agent did not conclude within active-time budget",
                    prompt=prompt,
                    stop_reason="timeout",
                    stage=stage,
                    tool_time_seconds=tracker.tool_time_seconds,
                )
                self._last = turn
                return turn
        turn = await task
        self._last = turn
        return turn


def _q() -> QcQuestion:
    return QcQuestion(
        question_id="q1",
        question="What is the target of venetoclax?",
        expected_answer="BCL2",
    )


@pytest.mark.asyncio
async def test_accept_on_first_attempt() -> None:
    proposer = ScriptedAgent(
        [json.dumps({"decision": "propose", "witness": "W1", "explanation": "finds BCL2"})]
    )
    validator = ScriptedAgent(
        [json.dumps({"supports_claim": True, "reasoning": "rows show BCL2", "quality_issues": ""})]
    )
    reviewer = ScriptedAgent(
        [json.dumps({"classification": "supported", "reasoning": "ok", "quality_issues": ""})]
    )
    loop = QcLoop(
        proposer,
        validator,
        reviewer,
        max_attempts=3,
        investigation_seconds=30,
        wrap_up_seconds=10,
        conclusion_seconds=5,
    )
    result = await loop.evaluate(_q())
    assert result.terminal_status == "supported"
    assert len(result.attempts) == 1
    assert result.witness == "W1"
    assert result.review is not None
    assert result.review.classification == QcClassification.SUPPORTED
    # traces preserved
    assert result.attempts[0].proposer_turn is not None
    assert result.attempts[0].proposer_turn.trace_messages
    assert result.attempts[0].proposer_turn.raw_trace
    assert result.reviewer_turn is not None
    assert result.reviewer_turn.trace_messages


@pytest.mark.asyncio
async def test_revise_then_accept() -> None:
    proposer = ScriptedAgent(
        [
            json.dumps({"decision": "propose", "witness": "W_bad", "explanation": "v1"}),
            json.dumps({"decision": "propose", "witness": "W_good", "explanation": "v2"}),
        ]
    )
    validator = ScriptedAgent(
        [
            json.dumps(
                {
                    "supports_claim": False,
                    "reasoning": "overbroad",
                    "quality_issues": "missing constraint",
                }
            ),
            json.dumps({"supports_claim": True, "reasoning": "ok", "quality_issues": ""}),
        ]
    )
    reviewer = ScriptedAgent(
        [json.dumps({"classification": "supported", "reasoning": "ok", "quality_issues": ""})]
    )
    loop = QcLoop(proposer, validator, reviewer, max_attempts=3, investigation_seconds=30)
    result = await loop.evaluate(_q())
    assert result.terminal_status == "supported"
    assert len(result.attempts) == 2
    assert result.witness == "W_good"
    assert "PREVIOUS_ATTEMPTS" in proposer.prompts[0] or any(
        "PREVIOUS_ATTEMPTS" in p for p in proposer.prompts
    )


@pytest.mark.asyncio
async def test_abandon_then_review() -> None:
    proposer = ScriptedAgent(
        [
            json.dumps(
                {
                    "decision": "abandon",
                    "witness": "",
                    "explanation": "",
                    "abandon_reason": "no support found",
                }
            )
        ]
    )
    validator = ScriptedAgent([])
    reviewer = ScriptedAgent(
        [
            json.dumps(
                {
                    "classification": "unsupported",
                    "reasoning": "cannot establish support",
                    "quality_issues": "",
                }
            )
        ]
    )
    loop = QcLoop(proposer, validator, reviewer, investigation_seconds=30)
    result = await loop.evaluate(_q())
    assert result.terminal_status == "unsupported"
    assert len(result.attempts) == 1
    assert result.attempts[0].proposal.abandoned
    assert result.attempts[0].proposer_turn is not None
    assert len(validator.prompts) == 0


@pytest.mark.asyncio
async def test_invalid_output_repair() -> None:
    proposer = ScriptedAgent(
        [
            "not json",
            json.dumps({"decision": "propose", "witness": "W", "explanation": "fixed"}),
        ]
    )
    validator = ScriptedAgent(
        [json.dumps({"supports_claim": True, "reasoning": "r", "quality_issues": ""})]
    )
    reviewer = ScriptedAgent(
        [json.dumps({"classification": "supported", "reasoning": "r", "quality_issues": ""})]
    )
    loop = QcLoop(
        proposer, validator, reviewer, invalid_output_retries=1, investigation_seconds=30
    )
    result = await loop.evaluate(_q())
    assert result.terminal_status == "supported"
    assert "repair" in proposer.stages or any(s == "repair" for s in proposer.stages)


@pytest.mark.asyncio
async def test_proposer_parse_error_after_retries() -> None:
    proposer = ScriptedAgent(["still not json", "also bad"])
    validator = ScriptedAgent([])
    reviewer = ScriptedAgent([])
    loop = QcLoop(
        proposer, validator, reviewer, invalid_output_retries=1, investigation_seconds=30
    )
    result = await loop.evaluate(_q())
    assert result.terminal_status == "error"
    assert result.error_stage == "proposer"
    # Failed parse must still preserve the agent turn / raw text for audit.
    assert len(result.attempts) == 1
    assert result.attempts[0].proposer_turn is not None
    assert "also bad" in (result.attempts[0].proposer_turn.text or "")
    assert "raw_response" in result.error_message


@pytest.mark.asyncio
async def test_steering_after_empty_investigation() -> None:
    """If investigation yields empty/timeout-like response, wrap-up steering runs."""
    proposer = ScriptedAgent(
        [
            "",  # investigation empty → not finished
            json.dumps({"decision": "propose", "witness": "W", "explanation": "after steer"}),
        ]
    )
    # Force first response to look incomplete
    original = proposer.run_turn

    async def run_turn_override(prompt, **kwargs):
        turn = await original(prompt, **kwargs)
        if turn.text == "":
            return turn.model_copy(
                update={"stop_reason": "timeout", "error": "budget", "text": ""}
            )
        return turn

    proposer.run_turn = run_turn_override  # type: ignore[method-assign]

    validator = ScriptedAgent(
        [json.dumps({"supports_claim": True, "reasoning": "r", "quality_issues": ""})]
    )
    reviewer = ScriptedAgent(
        [json.dumps({"classification": "supported", "reasoning": "r", "quality_issues": ""})]
    )
    loop = QcLoop(
        proposer,
        validator,
        reviewer,
        investigation_seconds=1,
        wrap_up_seconds=5,
        conclusion_seconds=5,
    )
    result = await loop.evaluate(_q())
    assert result.terminal_status == "supported"
    assert any(s == "wrap_up" for s in proposer.stages)
    assert result.attempts[0].proposer_turn is not None
    assert result.attempts[0].proposer_turn.steered


@pytest.mark.asyncio
async def test_tool_time_excluded_from_active_budget() -> None:
    """Long tool delay should not exhaust a short active budget when excluded."""
    body = json.dumps({"decision": "propose", "witness": "W", "explanation": "ok"})
    # tool 0.3s + after 0.05s; active budget 0.15s would fail if tool counted
    proposer = ToolAwareSlowAgent(body, tool_delay=0.3, after_tool_delay=0.02)
    validator = ScriptedAgent(
        [json.dumps({"supports_claim": True, "reasoning": "r", "quality_issues": ""})]
    )
    reviewer = ScriptedAgent(
        [json.dumps({"classification": "supported", "reasoning": "r", "quality_issues": ""})]
    )
    loop = QcLoop(
        proposer,
        validator,
        reviewer,
        investigation_seconds=0.15,
        wrap_up_seconds=0.05,
        conclusion_seconds=0.05,
        exclude_tool_time=True,
    )
    result = await loop.evaluate(_q())
    assert result.terminal_status == "supported"
    assert result.attempts[0].proposer_turn is not None
    assert result.attempts[0].proposer_turn.trace_messages
