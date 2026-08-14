"""Candidate extraction and balanced sampling for citation audit traces."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from karenina.schemas.verification import VerificationResult
from paper.otp_citation_audit.config import CANONICAL_PARSER, CLAUDE_MODELS, MAX_CITATIONS_PER_TRACE

_EMPTY_TRAILING_AI = re.compile(r"--- AI Message ---\s*\Z")
_NO_AI_FINAL = re.compile(r"(?s)--- Tool Message \(call_id:[^)]*\) ---(?:(?!--- AI Message ---).)*\Z")
_BRACKETED_NOTE = re.compile(r"\[[^\[\]\n]+\]\s*\Z")


@dataclass(frozen=True)
class CitationCandidate:
    """One citation-bearing answer eligible for balanced sampling."""

    result: VerificationResult
    model: str
    regime: str
    outcome: str
    final_answer_text: str
    n_citations: int

    @property
    def result_id(self) -> str:
        """Return the source verification result id."""
        return str(self.result.metadata.result_id)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return "" if content is None else str(content)


def final_answer(result: VerificationResult) -> str:
    """Extract only the final assistant answer, excluding tool results."""
    template = result.template
    if template is None:
        return ""
    messages = template.trace_messages or []
    if messages:
        last = messages[-1]
        if last.get("role") != "assistant":
            return ""
        return _content_text(last.get("content")).strip()
    raw = template.raw_llm_response or ""
    marker = "--- AI Message ---"
    if marker not in raw:
        return raw.strip()
    tail = raw.rsplit(marker, 1)[1]
    if "--- Tool Message (call_id:" in tail:
        return ""
    return tail.split("\nTool Calls:\n", 1)[0].strip()


def structural_skip_reason(result: VerificationResult) -> str | None:
    """Return the pre-screen exclusion reason, if any."""
    template = result.template
    if template is None:
        return "empty_trace"
    messages = template.trace_messages or []
    raw = template.raw_llm_response or ""
    if not messages and not raw.strip():
        return "empty_trace"
    if messages and messages[-1].get("role") != "assistant":
        return "no_ai_final_message"
    if not messages and _NO_AI_FINAL.search(raw):
        return "no_ai_final_message"
    text = final_answer(result)
    if text and _BRACKETED_NOTE.fullmatch(text):
        return "bracketed_trace_note"
    if _BRACKETED_NOTE.search(raw):
        return "bracketed_trace_note"
    if not text.strip() or _EMPTY_TRAILING_AI.search(raw):
        return "empty_trailing_ai"
    return None


def outcome(result: VerificationResult) -> str:
    """Return pass, fail, abstain, or infra from structured failure metadata."""
    failure = result.metadata.failure
    if failure is None:
        return "pass"
    if failure.group.value == "content":
        return "fail"
    if failure.group.value == "abstained":
        return "abstain"
    return "infra"


def is_canonical_claude_row(result: VerificationResult) -> bool:
    """Return whether a row is one canonical parser view of a Claude answer."""
    return bool(
        result.metadata.answering.model_name in CLAUDE_MODELS
        and result.metadata.parsing.model_name == CANONICAL_PARSER
    )


def _pick_diverse(
    candidates: list[CitationCandidate],
    count: int,
    *,
    seen_question_ids: set[str] | None = None,
) -> list[CitationCandidate]:
    ranked = sorted(candidates, key=lambda item: (-item.n_citations, item.result_id))
    distinct: list[CitationCandidate] = []
    repeated: list[CitationCandidate] = []
    seen = set(seen_question_ids or ())
    for candidate in ranked:
        question_id = candidate.result.metadata.question_id
        target = distinct if question_id not in seen else repeated
        target.append(candidate)
        seen.add(question_id)
    return (distinct + repeated)[:count]


def balanced_sample(
    candidates: list[CitationCandidate],
    *,
    target_per_condition: int,
    max_citations: int | None = MAX_CITATIONS_PER_TRACE,
) -> list[CitationCandidate]:
    """Select balanced pass and fail traces within each model and regime."""
    buckets: dict[tuple[str, str, str], list[CitationCandidate]] = defaultdict(list)
    for candidate in candidates:
        within_cap = max_citations is None or candidate.n_citations <= max_citations
        if candidate.outcome in {"pass", "fail"} and within_cap:
            buckets[(candidate.model, candidate.regime, candidate.outcome)].append(candidate)
    selected: list[CitationCandidate] = []
    conditions = sorted({(model, regime) for model, regime, _outcome in buckets})
    half = target_per_condition // 2
    for model, regime in conditions:
        per_outcome = {
            status: _pick_diverse(buckets.get((model, regime, status), []), half) for status in ("pass", "fail")
        }
        if all(buckets.get((model, regime, status)) for status in ("pass", "fail")):
            deficit = target_per_condition - sum(len(rows) for rows in per_outcome.values())
            for status in ("pass", "fail"):
                if deficit <= 0:
                    break
                used = {row.result_id for row in per_outcome[status]}
                extra = _pick_diverse(
                    [row for row in buckets[(model, regime, status)] if row.result_id not in used],
                    deficit,
                    seen_question_ids={row.result.metadata.question_id for row in per_outcome[status]},
                )
                per_outcome[status].extend(extra)
                deficit -= len(extra)
        selected.extend(per_outcome["pass"])
        selected.extend(per_outcome["fail"])
    return sorted(
        selected,
        key=lambda item: (item.model, item.regime, item.outcome, -item.n_citations, item.result_id),
    )
