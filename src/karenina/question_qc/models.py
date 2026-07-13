"""Domain models for question quality control (QC)."""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class QcRole(StrEnum):
    PROPOSER = "proposer"
    VALIDATOR = "validator"
    REVIEWER = "reviewer"


class QcClassification(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ILL_FORMED = "ill_formed"
    INCONCLUSIVE = "inconclusive"


class QcQuestion(BaseModel):
    """A benchmark question under QC (question text + expected answer)."""

    question_id: str
    question: str
    expected_answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Proposal(BaseModel):
    decision: str = "propose"  # propose | abandon
    witness: str = ""
    explanation: str = ""
    abandon_reason: str = ""
    progress_report: str = ""
    raw_response: str = ""

    @property
    def abandoned(self) -> bool:
        return self.decision == "abandon"


class Validation(BaseModel):
    supports_claim: bool | None = None
    reasoning: str = ""
    quality_issues: str = ""
    evidence_summary: dict[str, Any] | None = None
    progress_report: str = ""
    raw_response: str = ""

    @property
    def passes_evidence_gate(self) -> bool:
        return self.supports_claim is True and not (self.quality_issues or "").strip()


class Review(BaseModel):
    classification: QcClassification | None = None
    reasoning: str = ""
    quality_issues: str = ""
    remarks: str = ""
    progress_report: str = ""
    raw_response: str = ""


class QcUsage(BaseModel):
    """Token / cost usage for a turn or aggregated role run."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    model: str | None = None

    def merge(self, other: QcUsage | None) -> QcUsage:
        if other is None:
            return self
        cost: float | None
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return QcUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=cost,
            model=other.model or self.model,
        )


class QcTurn(BaseModel):
    """One agent stage/turn, including optional full agent trace."""

    text: str = ""
    error: str | None = None
    prompt: str = ""
    stop_reason: str = "completed"  # completed | timeout | error | cancelled
    # Agent traces (from AgentPort)
    raw_trace: str = ""
    trace_messages: list[dict[str, Any]] = Field(default_factory=list)
    usage: QcUsage | None = None
    turns: int = 0
    limit_reached: bool = False
    # Timing / steering metadata
    stage: str | None = None  # investigation | wrap_up | final_conclusion | repair
    wall_time_seconds: float | None = None
    active_time_seconds: float | None = None
    tool_time_seconds: float | None = None
    steered: bool = False


class QcAttempt(BaseModel):
    number: int
    proposal: Proposal
    validation: Validation | None = None
    proposer_turn: QcTurn | None = None
    validator_turn: QcTurn | None = None


class QcResult(BaseModel):
    question_id: str
    attempts: list[QcAttempt] = Field(default_factory=list)
    review: Review | None = None
    reviewer_turn: QcTurn | None = None
    terminal_status: str = ""
    error_stage: str = ""
    error_message: str = ""
    run_name: str | None = None

    @property
    def witness(self) -> str:
        """Last accepted witness, else last proposed witness."""
        for attempt in reversed(self.attempts):
            if attempt.validation and attempt.validation.passes_evidence_gate and attempt.proposal.witness:
                return attempt.proposal.witness
        for attempt in reversed(self.attempts):
            if attempt.proposal.witness:
                return attempt.proposal.witness
        return ""

    def all_turns(self) -> list[QcTurn]:
        """Flatten all preserved role turns (for audit export)."""
        turns: list[QcTurn] = []
        for attempt in self.attempts:
            if attempt.proposer_turn is not None:
                turns.append(attempt.proposer_turn)
            if attempt.validator_turn is not None:
                turns.append(attempt.validator_turn)
        if self.reviewer_turn is not None:
            turns.append(self.reviewer_turn)
        return turns


class QcResultSet(BaseModel):
    """Container for QC results from a ``run_qc`` call."""

    results: list[QcResult] = Field(default_factory=list)
    run_name: str | None = None

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[QcResult]:
        return iter(self.results)

    def filter(self, *, terminal: str | None = None) -> QcResultSet:
        """Return a subset of results, optionally filtered by terminal_status."""
        items = self.results
        if terminal is not None:
            items = [r for r in items if r.terminal_status == terminal]
        return QcResultSet(results=items, run_name=self.run_name)

    def by_question_id(self) -> dict[str, QcResult]:
        return {r.question_id: r for r in self.results}

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.model_dump() for r in self.results]
