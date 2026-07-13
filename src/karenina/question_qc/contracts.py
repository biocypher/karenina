"""Parse agent JSON responses into QC domain models."""

from __future__ import annotations

from typing import Any

from karenina.utils.json_extraction import extract_json_from_text, strip_markdown_fences

from .gates import normalize_classification
from .models import Proposal, Review, Validation


def _load_object(text: str) -> dict[str, Any]:
    import json

    cleaned = strip_markdown_fences(text) or text
    cleaned = cleaned.strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = extract_json_from_text(cleaned)
        if not extracted:
            raise ValueError("response does not contain a JSON object") from None
        value = json.loads(extracted)
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def parse_proposal(text: str) -> Proposal:
    data = _load_object(text)
    decision = str(data.get("decision", "propose")).strip().lower()
    if decision not in ("propose", "abandon", "give_up"):
        # Infer: empty witness with give_up_reason → abandon
        if data.get("give_up_reason") or data.get("abandon_reason"):
            decision = "abandon"
        else:
            decision = "propose"
    if decision == "give_up":
        decision = "abandon"
    witness = str(data.get("witness") or data.get("query") or data.get("cypher") or "")
    return Proposal(
        decision=decision,
        witness=witness,
        explanation=str(data.get("explanation", "")),
        abandon_reason=str(data.get("abandon_reason") or data.get("give_up_reason") or ""),
        progress_report=str(data.get("progress_report", "")),
        raw_response=text,
    )


def parse_validation(text: str) -> Validation:
    data = _load_object(text)
    supports = data.get("supports_claim", data.get("answers_question"))
    if supports is not None and not isinstance(supports, bool):
        if isinstance(supports, str):
            low = supports.strip().lower()
            if low in ("true", "yes"):
                supports = True
            elif low in ("false", "no"):
                supports = False
            elif low in ("null", "none", "uncertain"):
                supports = None
            else:
                raise ValueError("supports_claim must be boolean or null")
        else:
            raise ValueError("supports_claim must be boolean or null")
    summary = data.get("evidence_summary", data.get("query_result_summary"))
    if summary is not None and not isinstance(summary, dict):
        summary = {"value": summary}
    return Validation(
        supports_claim=supports,
        reasoning=str(data.get("reasoning", "")),
        quality_issues=str(data.get("quality_issues", "")),
        evidence_summary=summary,
        progress_report=str(data.get("progress_report", "")),
        raw_response=text,
    )


def parse_review(text: str) -> Review:
    data = _load_object(text)
    raw_class = data.get("classification")
    classification = normalize_classification(str(raw_class) if raw_class is not None else None)
    return Review(
        classification=classification,
        reasoning=str(data.get("reasoning", "")),
        quality_issues=str(data.get("quality_issues", "")),
        remarks=str(data.get("remarks", "")),
        progress_report=str(data.get("progress_report", "")),
        raw_response=text,
    )


PROPOSER_CONTRACT = (
    '{"decision":"propose",'
    '"witness":"<MINIMAL executable query string that returns evidence for the expected answer>",'
    '"explanation":"<one or two sentences: how executing this query supports the expected answer>",'
    '"abandon_reason":"","progress_report":""}'
)

PROPOSER_ABANDON_CONTRACT = (
    '{"decision":"abandon","witness":"","explanation":"",'
    '"abandon_reason":"<reason you cannot produce a supporting query>",'
    '"progress_report":"<checks performed>"}'
)

VALIDATOR_CONTRACT = (
    '{"supports_claim":true|false|null,"reasoning":"<evidence-based reason>",'
    '"quality_issues":"<issues or empty string>",'
    '"evidence_summary":<object or null>,"progress_report":""}'
)

REVIEWER_CONTRACT = (
    '{"classification":"supported|unsupported|ill_formed|inconclusive",'
    '"reasoning":"<question judgment>","quality_issues":"",'
    '"remarks":"","progress_report":""}'
)
