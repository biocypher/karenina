"""Capture helpers that turn verification results into a ReplayStore."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from karenina.replay.store import ReplayEntry, ReplayKey, ReplayStore

logger = logging.getLogger(__name__)


def capture_from_result_set(
    result_set: Any,
    *,
    include_parsed: bool = True,
    include_agent_traces: bool = True,
    only_successful: bool = True,
    answering_model_ids: set[str] | None = None,
    scenario_ids: set[str] | None = None,
) -> ReplayStore:
    """Walk a VerificationResultSet and build a ReplayStore.

    Iterates ``result_set.results`` (the flat per-turn VerificationResult
    list). Scenario turns are distinguished from QA turns by the presence
    of ``metadata.scenario_id``. A per-node visit counter keyed by
    (scenario_id, scenario_node) is used for ``ReplayKey.visit_index``.

    ``result_set.scenario_results`` is not required for the happy path
    because every scenario turn also produces a VerificationResult with
    populated scenario metadata.

    Args:
        result_set: A VerificationResultSet-like object with a ``results``
            attribute containing per-turn VerificationResult objects.
        include_parsed: Whether to copy ``template.parsed_llm_response``
            into the replay entry's ``parsed_answer_fields``.
        include_agent_traces: Whether to copy ``template.trace_messages``
            into the replay entry's ``trace_messages``.
        only_successful: If True, drop turns where
            ``metadata.completed_without_errors`` is False.
        answering_model_ids: Optional allow-list of answering model
            display strings; turns whose answering model is not in the
            set are skipped.
        scenario_ids: Optional allow-list of scenario ids; scenario
            turns outside the set are skipped (QA turns are unaffected).

    Returns:
        A ReplayStore with ``miss_policy="fall_through"``.
    """
    store = ReplayStore(miss_policy="fall_through")

    # Sort scenario turns into a predictable order before applying the
    # per-node visit counter. QA turns are emitted in iteration order.
    results = list(getattr(result_set, "results", None) or [])
    results.sort(
        key=lambda vr: (
            _none_last(getattr(vr.metadata, "scenario_id", None)),
            _none_last(getattr(vr.metadata, "scenario_node", None)),
            _none_last(getattr(vr.metadata, "scenario_turn", None)),
        )
    )

    visit_counts: dict[tuple[str, str], int] = {}

    for vr in results:
        md = vr.metadata

        if only_successful and not getattr(md, "completed_without_errors", True):
            continue

        answering_display = _answering_display(md)
        if answering_model_ids is not None and answering_display not in answering_model_ids:
            continue

        scenario_id = getattr(md, "scenario_id", None)
        scenario_node = getattr(md, "scenario_node", None)

        if scenario_ids is not None and scenario_id is not None and scenario_id not in scenario_ids:
            continue

        visit_index: int | None
        if scenario_id is not None:
            cell_key: tuple[str, str] = (scenario_id, scenario_node or "")
            visit_index = visit_counts.get(cell_key, 0)
            visit_counts[cell_key] = visit_index + 1
        else:
            visit_index = None

        replay_key = ReplayKey(
            question_id=md.question_id,
            scenario_id=scenario_id,
            scenario_node=scenario_node,
            answering_model_id=answering_display,
            visit_index=visit_index,
        )

        entry = _build_entry_from_vr(
            vr,
            include_parsed=include_parsed,
            include_agent_traces=include_agent_traces,
        )
        store.register(replay_key, entry)

    return store


def capture_from_scenario_result(
    scenario_result: Any,
    *,
    answering_model_id: str,
    scenario_id: str | None = None,
    nodes: set[str] | None = None,
    include_parsed: bool = True,
    include_agent_traces: bool = True,
) -> ReplayStore:
    """Build a ReplayStore from one ScenarioExecutionResult.

    ``answering_model_id`` is required because ScenarioExecutionResult
    does not carry per-turn model identity. The caller should pass
    ``ModelIdentity.from_model_config(m, role="answering").display_string``
    for consistency with ``capture_from_result_set``.

    Args:
        scenario_result: A ScenarioExecutionResult-like object with
            ``scenario_id`` and ``history`` (list of TurnRecord).
        answering_model_id: The canonical display string for the model
            that produced these turns.
        scenario_id: Optional override for the scenario id. Falls back
            to ``scenario_result.scenario_id`` if not supplied.
        nodes: Optional allow-list of node ids; turns on other nodes
            are skipped.
        include_parsed: Whether to copy ``record.parsed_fields`` into
            the replay entry's ``parsed_answer_fields``.
        include_agent_traces: Whether to copy ``record.trace_messages``
            into the replay entry's ``trace_messages``.

    Returns:
        A ReplayStore with ``miss_policy="fall_through"``.

    Raises:
        ValueError: If no scenario id can be determined.
    """
    from karenina.utils.checkpoint import generate_question_id

    store = ReplayStore(miss_policy="fall_through")

    effective_scenario_id = scenario_id or getattr(scenario_result, "scenario_id", None)
    if effective_scenario_id is None:
        raise ValueError("scenario_id is required and not available on the result")

    visit_counts: dict[str, int] = {}

    for record in getattr(scenario_result, "history", []):
        node_id = record.node_id
        if nodes is not None and node_id not in nodes:
            continue

        visit_index = visit_counts.get(node_id, 0)
        visit_counts[node_id] = visit_index + 1

        question_text = getattr(record, "question_text", "") or ""
        question_id = generate_question_id(question_text)

        parsed_fields = getattr(record, "parsed_fields", None) if include_parsed else None
        trace_messages = getattr(record, "trace_messages", None)
        trace_messages_dicts: list[dict[str, Any]] | None
        if include_agent_traces and trace_messages:
            trace_messages_dicts = [_message_to_dict(m) for m in trace_messages]
        else:
            trace_messages_dicts = None

        entry = ReplayEntry(
            raw_trace=getattr(record, "raw_response", "") or "",
            trace_messages=trace_messages_dicts,
            parsed_answer_fields=parsed_fields if parsed_fields else None,
            captured_model_id=answering_model_id,
            captured_at=datetime.now(UTC).isoformat(),
        )

        key = ReplayKey(
            question_id=question_id,
            scenario_id=effective_scenario_id,
            scenario_node=node_id,
            answering_model_id=answering_model_id,
            visit_index=visit_index,
        )
        store.register(key, entry)

    return store


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _answering_display(md: Any) -> str:
    """Pull the canonical answering model display string from a metadata object."""
    answering = getattr(md, "answering", None)
    if answering is None:
        return "unknown"
    return getattr(answering, "display_string", None) or "unknown"


def _none_last(value: Any) -> tuple[int, Any]:
    """Sort key helper: None sorts after real values."""
    return (1, 0) if value is None else (0, value)


def _build_entry_from_vr(
    vr: Any,
    *,
    include_parsed: bool,
    include_agent_traces: bool,
) -> ReplayEntry:
    template = vr.template
    parsed_fields = getattr(template, "parsed_llm_response", None)
    if not include_parsed:
        parsed_fields = None

    trace_messages = getattr(template, "trace_messages", None) or None
    if not include_agent_traces:
        trace_messages = None

    md = vr.metadata
    return ReplayEntry(
        raw_trace=getattr(template, "raw_llm_response", "") or "",
        trace_messages=trace_messages if trace_messages else None,
        parsed_answer_fields=parsed_fields if parsed_fields else None,
        captured_model_id=_answering_display(md),
        captured_at=getattr(md, "completed_at", None) or datetime.now(UTC).isoformat(),
    )


def _message_to_dict(m: Any) -> dict[str, Any]:
    """Convert a port Message (dataclass with to_dict) or dict to a plain dict."""
    if isinstance(m, dict):
        return m
    to_dict = getattr(m, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return result
    raise TypeError(f"Cannot convert {type(m).__name__} to dict for replay storage")
