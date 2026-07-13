"""Prompt builders for question QC roles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import (
    PROPOSER_ABANDON_CONTRACT,
    PROPOSER_CONTRACT,
    REVIEWER_CONTRACT,
    VALIDATOR_CONTRACT,
)
from .models import QcAttempt, QcQuestion, QcRole


def _bounded(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "… [truncated]"


def attempt_history(attempts: list[QcAttempt]) -> list[dict[str, Any]]:
    """Structured prior-attempt evidence for proposer revision (no raw dumps)."""
    history: list[dict[str, Any]] = []
    for attempt in attempts:
        entry: dict[str, Any] = {
            "attempt": attempt.number,
            "proposal": {
                "decision": attempt.proposal.decision,
                "witness": _bounded(attempt.proposal.witness, 2000),
                "explanation": _bounded(attempt.proposal.explanation),
                "abandon_reason": _bounded(attempt.proposal.abandon_reason),
            },
            "validation": None,
        }
        if attempt.validation is not None:
            v = attempt.validation
            entry["validation"] = {
                "supports_claim": v.supports_claim,
                "passes_evidence_gate": v.passes_evidence_gate,
                "reasoning": _bounded(v.reasoning),
                "quality_issues": _bounded(v.quality_issues),
                "evidence_summary": v.evidence_summary,
            }
        history.append(entry)
    return history


def load_evidence_context(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def proposer_prompt(
    question: QcQuestion,
    attempts: list[QcAttempt],
    *,
    evidence_context: str = "",
) -> str:
    history = attempt_history(attempts)
    parts = [
        "You are the PROPOSER in a multi-role question quality-control process.",
        "Goal: produce a MINIMAL executable QUERY that proves the EXPECTED ANSWER is supported",
        "under the constraints of the QUESTION, using only the available evidence tools.",
        "",
        "WHAT `witness` MEANS (critical):",
        "- `witness` MUST be a single minimal query string for the evidence system (the exact query text).",
        "- It must be executable via the evidence tools (e.g. a read-only graph/SQL/search query).",
        "- Prefer the smallest query that still returns rows/results demonstrating the expected answer.",
        "- You MAY bind or filter on the expected answer when that still constitutes graph/database proof.",
        "- Do NOT put natural-language essays, tool logs, or JSON results in `witness` — only the query.",
        "- Do NOT invent tool results. Explore with tools first, then return the final query as `witness`.",
        "",
        "Quality bar for the query:",
        "- Minimal / not over-broad (avoid returning huge unconstrained result sets).",
        "- Sufficiently constrained by the question conditions.",
        "- Returns evidence of the expected answer (not merely echoing a literal with no join/filter path).",
        "",
        "Return ONLY one JSON object matching this contract (no markdown, no prose outside JSON):",
        PROPOSER_CONTRACT,
        "If you cannot establish a supporting query, abandon with:",
        PROPOSER_ABANDON_CONTRACT,
        "",
        f"QUESTION: {question.question}",
        f"EXPECTED_ANSWER: {question.expected_answer}",
    ]
    if question.metadata:
        parts.append(f"METADATA: {json.dumps(question.metadata, ensure_ascii=False)}")
    if evidence_context.strip():
        parts.append("EVIDENCE_CONTEXT (catalog / schema approximation; verify with tools):")
        parts.append(_bounded(evidence_context, 12_000))
    if history:
        parts.append("PREVIOUS_ATTEMPTS (revise; do not repeat a failed witness unchanged):")
        parts.append(json.dumps(history, ensure_ascii=False, indent=2))
    return "\n".join(parts)


def validator_prompt(question: QcQuestion, proposal_witness: str, proposal_explanation: str) -> str:
    return "\n".join(
        [
            "You are the VALIDATOR in a multi-role question quality-control process.",
            "The proposer's `witness` is supposed to be a MINIMAL executable QUERY.",
            "Independently re-check it using evidence tools. Do not trust the proposer's authority.",
            "You SHOULD execute the proposed query as written (via the query/evidence tool) before deciding.",
            "supports_claim=true only if the executed results prove the system of record supports",
            "the EXPECTED ANSWER under the QUESTION's constraints.",
            "quality_issues must list material defects or be empty. Examples of defects:",
            "- not a query / not executable",
            "- over-broad or under-constrained",
            "- non-minimal",
            "- does not actually return/support the expected answer",
            "- hollow (echoes a literal without real evidence path)",
            "Return ONLY one JSON object matching this contract (no markdown, no prose outside JSON):",
            VALIDATOR_CONTRACT,
            "",
            f"QUESTION: {question.question}",
            f"EXPECTED_ANSWER: {question.expected_answer}",
            f"PROPOSED_WITNESS_QUERY: {proposal_witness}",
            f"PROPOSER_EXPLANATION: {proposal_explanation}",
        ]
    )


def reviewer_prompt(question: QcQuestion, attempts: list[QcAttempt]) -> str:
    history = attempt_history(attempts)
    return "\n".join(
        [
            "You are the REVIEWER in a multi-role question quality-control process.",
            "Independently classify the QUESTION (not a model answer) using the attempt history.",
            "Classifications:",
            "- supported: expected answer is backed by evidence under the question constraints",
            "- unsupported: question is clear but the system of record does not support the expected answer",
            "- ill_formed: question/answer pair is ambiguous, contradictory, or unusable as ground truth",
            "- inconclusive: insufficient process evidence / uncertainty; needs rerun or human review",
            "Return ONLY one JSON object matching this contract:",
            REVIEWER_CONTRACT,
            "",
            f"QUESTION: {question.question}",
            f"EXPECTED_ANSWER: {question.expected_answer}",
            "ATTEMPT_HISTORY:",
            json.dumps(history, ensure_ascii=False, indent=2),
        ]
    )


def repair_prompt(error: str, contract: str) -> str:
    return (
        f"Your previous response was invalid: {error}. "
        f"Do not repeat the invalid format. Return only one valid JSON object matching: {contract}"
    )


def steering_prompt(role: QcRole, *, final: bool) -> str:
    """Wrap-up / final-conclusion steering (reveals uncertainty options)."""
    force = (
        "You did not conclude after the wrap-up instruction. You must conclude now. "
        "Do not begin more exploration or tool calls. "
        if final
        else (
            "The investigation period is complete. Stop broad exploration and wrap up now. "
            "Perform at most one focused check if essential. "
        )
    )
    if role is QcRole.PROPOSER:
        return force + (
            "Return the required proposal JSON now. "
            "Remember: `witness` must be a MINIMAL executable QUERY string (not prose). "
            "Or disclose that you cannot produce a supporting query with: "
            + PROPOSER_ABANDON_CONTRACT
        )
    if role is QcRole.VALIDATOR:
        return force + (
            "Return the required validator JSON. If a reliable verdict is still impossible, "
            "you may now use the uncertainty option: "
            '{"supports_claim":null,"reasoning":"<why uncertain>","quality_issues":"",'
            '"progress_report":"<checks and evidence obtained>","evidence_summary":null}.'
        )
    return force + (
        "Return the required reviewer JSON. If a reliable classification is still impossible, "
        "you may now use the uncertainty option: "
        '{"classification":"inconclusive","reasoning":"<why uncertain>","quality_issues":"",'
        '"remarks":"","progress_report":"<checks and evidence obtained>"}.'
    )
