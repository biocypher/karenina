"""Domain loop for multi-role question quality control.

This module must not import adapters, storage engines, or backend clients.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from .agent_adapter import merge_turns
from .contracts import (
    PROPOSER_CONTRACT,
    REVIEWER_CONTRACT,
    VALIDATOR_CONTRACT,
    parse_proposal,
    parse_review,
    parse_validation,
)
from .gates import derive_terminal_status, passes_evidence_gate
from .models import (
    Proposal,
    QcAttempt,
    QcQuestion,
    QcResult,
    QcRole,
    QcTurn,
    Review,
    Validation,
)
from .prompts import (
    proposer_prompt,
    repair_prompt,
    reviewer_prompt,
    steering_prompt,
    validator_prompt,
)

EventSink = Callable[[dict[str, Any]], Awaitable[None] | None]


class QcAgent(Protocol):
    """Agent protocol used by QcLoop (testable with fakes)."""

    async def run_turn(
        self,
        prompt: str,
        *,
        active_budget_seconds: float | None = None,
        exclude_tool_time: bool = True,
        event_sink: EventSink | None = None,
        stage: str | None = None,
    ) -> QcTurn: ...

    def reset_session(self) -> None: ...

    def snapshot_turn(self) -> QcTurn | None: ...

    async def cancel_turn(self) -> None: ...


class _ParseFailure(Exception):
    """Structured-output parse failed; carries the latest turn (incl. repairs/traces)."""

    def __init__(self, message: str, turn: QcTurn) -> None:
        super().__init__(message)
        self.turn = turn


class QcLoop:
    """Propose → validate → revise → review for a single question.

    Each role runs with staged active-time budgets:
    investigation → wrap-up steering → final conclusion.
    Tool time can be excluded from the active budget when agents emit
    tool-activity events (or via wall-clock buffer in adapters).
    Full agent traces are preserved on each role turn.
    """

    def __init__(
        self,
        proposer: QcAgent,
        validator: QcAgent,
        reviewer: QcAgent,
        *,
        max_attempts: int = 3,
        invalid_output_retries: int = 1,
        evidence_context: str = "",
        investigation_seconds: float = 180.0,
        wrap_up_seconds: float = 90.0,
        conclusion_seconds: float = 30.0,
        exclude_tool_time: bool = True,
        exclude_tool_name_substrings: list[str] | None = None,
    ) -> None:
        self.proposer = proposer
        self.validator = validator
        self.reviewer = reviewer
        self.max_attempts = max(1, max_attempts)
        self.invalid_output_retries = max(0, invalid_output_retries)
        self.evidence_context = evidence_context
        self.investigation_seconds = max(0.0, investigation_seconds)
        self.wrap_up_seconds = max(0.0, wrap_up_seconds)
        self.conclusion_seconds = max(0.0, conclusion_seconds)
        self.exclude_tool_time = exclude_tool_time
        self.exclude_tool_name_substrings = exclude_tool_name_substrings
        self.repair_timeout_seconds = max(1.0, conclusion_seconds or 30.0)

    def _reset(self, agent: QcAgent) -> None:
        reset = getattr(agent, "reset_session", None)
        if callable(reset):
            reset()

    async def _run_stage(
        self,
        agent: QcAgent,
        prompt: str,
        budget: float,
        *,
        stage: str | None,
    ) -> tuple[bool, QcTurn]:
        """Run one stage; return (finished_ok, turn)."""
        budget_arg = budget if budget > 0 else None
        turn = await agent.run_turn(
            prompt,
            active_budget_seconds=budget_arg,
            exclude_tool_time=self.exclude_tool_time,
            stage=stage,
        )
        if stage in ("wrap_up", "final_conclusion"):
            turn = turn.model_copy(update={"steered": True})
        ok = turn.stop_reason == "completed" and not turn.error and bool(turn.text.strip())
        return ok, turn

    async def _run_role(self, agent: QcAgent, role: QcRole, prompt: str) -> QcTurn:
        """Investigation → wrap-up steering → final conclusion."""
        self._reset(agent)
        stages: list[tuple[str, float, str | None]] = [
            (prompt, self.investigation_seconds, "investigation"),
            (steering_prompt(role, final=False), self.wrap_up_seconds, "wrap_up"),
            (steering_prompt(role, final=True), self.conclusion_seconds, "final_conclusion"),
        ]
        combined: QcTurn | None = None
        last: QcTurn | None = None
        for stage_prompt, budget, stage_name in stages:
            finished, turn = await self._run_stage(agent, stage_prompt, budget, stage=stage_name)
            combined = turn if combined is None else merge_turns(combined, turn)
            last = turn
            if finished:
                return combined
            if turn.stop_reason == "error" and turn.error and not turn.text.strip():
                return combined
        assert combined is not None and last is not None
        return combined.model_copy(
            update={
                "stop_reason": "timeout",
                "error": last.error or "agent did not conclude after final steering",
            }
        )

    @staticmethod
    def _format_parse_error(exc: Exception, turn: QcTurn) -> str:
        """Include a short raw-response snippet so failures are debuggable without digging."""
        snippet = (turn.text or "").strip().replace("\n", "\\n")
        if len(snippet) > 500:
            snippet = snippet[:500] + "…"
        base = str(exc)
        if snippet:
            return f"{base} | raw_response={snippet!r}"
        if turn.raw_trace:
            trace = turn.raw_trace.strip().replace("\n", "\\n")
            if len(trace) > 500:
                trace = trace[:500] + "…"
            return f"{base} | raw_trace={trace!r}"
        if turn.error:
            return f"{base} | turn_error={turn.error!r}"
        return (
            f"{base} | raw_response empty "
            f"(stop_reason={turn.stop_reason!r}, "
            f"trace_messages={len(turn.trace_messages)}, "
            f"turns={turn.turns})"
        )

    async def _parse_with_repair(
        self,
        agent: QcAgent,
        turn: QcTurn,
        parser: Callable[[str], Proposal | Validation | Review],
        contract: str,
    ) -> tuple[Proposal | Validation | Review, QcTurn]:
        current = turn
        if current.error and not current.text.strip():
            raise _ParseFailure(current.error, current)
        for retry in range(self.invalid_output_retries + 1):
            try:
                if not current.text.strip():
                    raise ValueError(current.error or "empty agent response")
                return parser(current.text), current
            except (ValueError, TypeError) as exc:
                if retry >= self.invalid_output_retries:
                    raise _ParseFailure(self._format_parse_error(exc, current), current) from exc
                correction = repair_prompt(str(exc), contract)
                repaired = await agent.run_turn(
                    correction,
                    active_budget_seconds=self.repair_timeout_seconds,
                    exclude_tool_time=self.exclude_tool_time,
                    stage="repair",
                )
                current = merge_turns(current, repaired.model_copy(update={"stage": "repair"}))
                if repaired.error and not repaired.text.strip():
                    raise _ParseFailure(
                        self._format_parse_error(RuntimeError(repaired.error), current),
                        current,
                    )
        raise _ParseFailure("invalid structured output", current)  # pragma: no cover

    async def evaluate(self, question: QcQuestion) -> QcResult:
        attempts: list[QcAttempt] = []
        try:
            for number in range(1, self.max_attempts + 1):
                p_prompt = proposer_prompt(
                    question,
                    attempts,
                    evidence_context=self.evidence_context,
                )
                p_turn = await self._run_role(self.proposer, QcRole.PROPOSER, p_prompt)
                try:
                    proposal_obj, p_turn = await self._parse_with_repair(
                        self.proposer, p_turn, parse_proposal, PROPOSER_CONTRACT
                    )
                except _ParseFailure as exc:
                    p_turn = exc.turn
                    # Preserve proposer turn + raw response for audit even when parse fails.
                    attempts.append(
                        QcAttempt(
                            number=number,
                            proposal=Proposal(
                                decision="propose",
                                raw_response=p_turn.text,
                                explanation="",
                            ),
                            validation=None,
                            proposer_turn=p_turn,
                        )
                    )
                    return QcResult(
                        question_id=question.question_id,
                        attempts=attempts,
                        terminal_status="timed_out" if p_turn.stop_reason == "timeout" else "error",
                        error_stage="proposer",
                        error_message=str(exc),
                    )
                assert isinstance(proposal_obj, Proposal)

                if proposal_obj.abandoned:
                    attempts.append(
                        QcAttempt(
                            number=number,
                            proposal=proposal_obj,
                            validation=None,
                            proposer_turn=p_turn,
                        )
                    )
                    break

                v_prompt = validator_prompt(
                    question,
                    proposal_obj.witness,
                    proposal_obj.explanation,
                )
                v_turn = await self._run_role(self.validator, QcRole.VALIDATOR, v_prompt)
                try:
                    validation_obj, v_turn = await self._parse_with_repair(
                        self.validator, v_turn, parse_validation, VALIDATOR_CONTRACT
                    )
                except _ParseFailure as exc:
                    v_turn = exc.turn
                    attempts.append(
                        QcAttempt(
                            number=number,
                            proposal=proposal_obj,
                            validation=None,
                            proposer_turn=p_turn,
                            validator_turn=v_turn,
                        )
                    )
                    return QcResult(
                        question_id=question.question_id,
                        attempts=attempts,
                        terminal_status="timed_out" if v_turn.stop_reason == "timeout" else "error",
                        error_stage="validator",
                        error_message=str(exc),
                    )
                assert isinstance(validation_obj, Validation)
                attempts.append(
                    QcAttempt(
                        number=number,
                        proposal=proposal_obj,
                        validation=validation_obj,
                        proposer_turn=p_turn,
                        validator_turn=v_turn,
                    )
                )
                if passes_evidence_gate(validation_obj):
                    break

            r_prompt = reviewer_prompt(question, attempts)
            r_turn = await self._run_role(self.reviewer, QcRole.REVIEWER, r_prompt)
            try:
                review_obj, r_turn = await self._parse_with_repair(
                    self.reviewer, r_turn, parse_review, REVIEWER_CONTRACT
                )
            except _ParseFailure as exc:
                r_turn = exc.turn
                return QcResult(
                    question_id=question.question_id,
                    attempts=attempts,
                    reviewer_turn=r_turn,
                    terminal_status="timed_out" if r_turn.stop_reason == "timeout" else "error",
                    error_stage="reviewer",
                    error_message=str(exc),
                )
            assert isinstance(review_obj, Review)
            return QcResult(
                question_id=question.question_id,
                attempts=attempts,
                review=review_obj,
                reviewer_turn=r_turn,
                terminal_status=derive_terminal_status(review_obj),
            )
        except TimeoutError as exc:
            return QcResult(
                question_id=question.question_id,
                attempts=attempts,
                terminal_status="timed_out",
                error_stage="timeout",
                error_message=str(exc) or "agent timed out",
            )
