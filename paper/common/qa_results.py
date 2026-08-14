"""Typed access to the paper's archived QA result files.

The result files use the v2.2 export schema. ``ResultsIOManager`` streams and
validates each row, which is then projected into a small frozen dataclass
carrying the fields shared by the analyses.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from karenina.benchmark import ResultsIOManager
from karenina.schemas.verification import VerificationResult

REFERENCE_JUDGE = "claude-opus-4-6"

GROUP_TO_OUTCOME = {
    "content": "fail-content",
    "abstained": "abstain",
    "autofail": "infra",
    "retry": "infra",
    "system": "infra",
}


@dataclass(frozen=True, slots=True)
class QAResultRow:
    """One question, answerer, judge, and replicate row of a QA evaluation."""

    question_id: str
    answerer: str
    answering_tools: tuple[str, ...]
    parser: str
    regime: str
    replicate: int | None
    run_name: str | None
    result_id: str
    outcome: str
    raw_trace: str
    trace_message_count: int
    tokens_answerer: int | None
    verify_result: bool | None

    @classmethod
    def from_result(cls, result: VerificationResult, regime: str) -> QAResultRow:
        """Build a reduced analysis row from a validated Karenina result."""
        metadata = result.metadata
        template = result.template
        failure_group = metadata.failure.group.value if metadata.failure else None
        return cls(
            question_id=metadata.question_id,
            answerer=metadata.answering.model_name,
            answering_tools=tuple(metadata.answering.tools),
            parser=metadata.parsing.model_name,
            regime=regime,
            replicate=metadata.replicate,
            run_name=metadata.run_name,
            result_id=metadata.result_id,
            outcome=_outcome_from_failure_group(failure_group),
            raw_trace=template.raw_llm_response if template else "",
            trace_message_count=len(template.trace_messages or ()) if template else 0,
            tokens_answerer=_tokens_from_result(result),
            verify_result=template.verify_result if template else None,
        )


def _outcome_from_failure_group(group: str | None) -> str:
    """Map a stored failure group to the paper's outcome classes."""
    if group is None:
        return "pass"
    if group not in GROUP_TO_OUTCOME:
        raise ValueError(f"Unknown failure group in QA result row: {group!r}")
    return GROUP_TO_OUTCOME[group]


def _tokens_from_result(result: VerificationResult) -> int | None:
    """Return answer-generation tokens when stored on this parser sibling."""
    template = result.template
    usage = template.usage_metadata if template else None
    usage = usage or {}
    generation = usage.get("answer_generation") or {}
    tokens = generation.get("total_tokens")
    return int(tokens) if tokens is not None else None


def iter_rows(path: Path, regime: str) -> Iterator[QAResultRow]:
    """Stream an archived QA result file as typed rows.

    Args:
        path: An archived QA results JSON file.
        regime: Label recorded on every row, normally ``mcp`` or ``nomcp``.

    Yields:
        Reduced typed rows in source order.
    """
    for result in iter_results(path):
        yield QAResultRow.from_result(result, regime)


def iter_results(path: Path) -> Iterator[VerificationResult]:
    """Stream validated Karenina results through ``ResultsIOManager``."""
    for result in ResultsIOManager.iter_from_json(path, raw=False):
        if not isinstance(result, VerificationResult):
            raise TypeError("ResultsIOManager returned an unvalidated result row")
        yield result


def load_benchmark_text(jsonld_path: Path) -> dict[str, tuple[str, str]]:
    """Map each question id to its question text and reference answer."""
    payload = json.loads(jsonld_path.read_text())
    benchmark: dict[str, tuple[str, str]] = {}
    for element in payload["dataFeedElement"]:
        item = element["item"]
        benchmark[str(element["@id"])] = (
            str(item["text"]),
            str(item["acceptedAnswer"]["text"]),
        )
    return benchmark


def load_reference_answers(jsonld_path: Path) -> dict[str, str]:
    """Map each question id to its accepted reference answer."""
    return {
        question_id: reference_answer
        for question_id, (_question, reference_answer) in load_benchmark_text(jsonld_path).items()
    }
