"""Inspect retrieved Maraviroc approval evidence in validated QA traces."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from karenina.schemas.verification import VerificationResult

EvidenceKey = tuple[str, str, str, int]


def result_key(result: VerificationResult) -> EvidenceKey | None:
    """Return the evidence-analysis key for a result with a replicate."""
    metadata = result.metadata
    if metadata.replicate is None:
        return None
    return (
        metadata.question_id,
        metadata.answering.model_name,
        metadata.parsing.model_name,
        metadata.replicate,
    )


def _message_text(content: Any) -> str | None:
    """Extract a serialized tool body from supported message shapes."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        value = block.get("content", block.get("text"))
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts) if parts else None


def _successful_tool_result(content: Any) -> Any | None:
    """Parse and return the payload of a successful tool result."""
    text = _message_text(content)
    if text is None:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and parsed.get("status") == "success":
        return parsed.get("result")
    return None


def _iter_leaf_values(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield dotted paths and scalar leaves from nested tool data."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_leaf_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_leaf_values(child, f"{path}[{index}]")
    else:
        yield path, value


def trace_support_flags(trace_messages: Iterable[dict[str, Any]]) -> tuple[bool, bool]:
    """Detect approval status and first-approval-date evidence in tool data."""
    has_approval_status = False
    has_first_approval_date = False
    for message in trace_messages:
        if message.get("role") != "tool":
            continue
        result = _successful_tool_result(message.get("content"))
        if result is None:
            continue
        for path, value in _iter_leaf_values(result):
            path_lower = path.lower()
            if ("stage" in path_lower or "clinicalstage" in path_lower) and str(value) == "APPROVAL":
                has_approval_status = True
            if (
                "firstapproval" in path_lower
                or "approvaldate" in path_lower
                or "approvalyear" in path_lower
            ) and value not in (None, "", []):
                has_first_approval_date = True
    return has_approval_status, has_first_approval_date


def collect_trace_support(
    results: Iterable[VerificationResult],
    target_keys: set[EvidenceKey],
) -> dict[EvidenceKey, tuple[bool, bool]]:
    """Collect support flags for selected rows from validated results."""
    support: dict[EvidenceKey, tuple[bool, bool]] = {}
    for result in results:
        key = result_key(result)
        if key is None or key not in target_keys:
            continue
        messages = result.template.trace_messages if result.template else []
        support[key] = trace_support_flags(messages)
        if len(support) == len(target_keys):
            break
    missing = target_keys.difference(support)
    if missing:
        raise ValueError(f"Maraviroc rows are absent from the validated MCP results: {sorted(missing)}")
    return support
