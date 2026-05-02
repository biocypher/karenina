"""Capture helpers that turn verification results into a ReplayStore."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal

from karenina.replay.store import ReplayEntry, ReplayKey, ReplayStore
from karenina.schemas.results.failure import FailureCategory

logger = logging.getLogger(__name__)


def capture_from_result_set(
    result_set: Any,
    *,
    include_parsed: bool = True,
    include_agent_traces: bool = True,
    only_successful: bool = True,
    answering_model_ids: set[str] | None = None,
    scenario_ids: set[str] | None = None,
    replicate_selector: Literal["all", "first", "last"] = "all",
) -> ReplayStore:
    """Walk a VerificationResultSet and build a ReplayStore.

    Iterates ``result_set.results`` (the flat per-turn VerificationResult
    list). Scenario turns are distinguished from QA turns by the presence
    of ``metadata.scenario_id``. A per-replicate visit counter keyed by
    (scenario_id, scenario_node, replicate) is used for
    ``ReplayKey.visit_index``. ``ReplayKey.replicate`` is taken from
    ``metadata.replicate`` (which may be None for single-replicate runs
    or pre-R1 data).

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
        only_successful: If True, drop turns whose pipeline did not
            produce a verifiable output (retry exhaustion, autofail,
            abstention/sufficiency, parsing/system errors). Turns with
            a :class:`FailureCategory.CONTENT` failure (the pipeline
            ran cleanly and the answer simply did not pass
            ``verify()``) are retained, because the captured trace
            and parsed fields faithfully describe what the model
            produced. If False, every turn is captured regardless of
            failure state.
        answering_model_ids: Optional allow-list of answering model
            display strings; turns whose answering model is not in the
            set are skipped.
        scenario_ids: Optional allow-list of scenario ids; scenario
            turns outside the set are skipped (QA turns are unaffected).
        replicate_selector: Canonicalization knob across the replicate
            axis. ``"all"`` (default) preserves every replicate as a
            distinct entry with its integer replicate on the key.
            ``"first"`` keeps rows matching the minimum integer
            replicate (ignoring None rows if any integer replicate is
            present), and ``"last"`` keeps the maximum. For ``"first"``
            and ``"last"``, the emitted ``ReplayKey.replicate`` is set to
            None so the resulting entries act as wildcards on the
            replicate axis (model and visit axes stay concrete).

    Returns:
        A ReplayStore with ``miss_policy="fall_through"``.
    """
    store = ReplayStore(miss_policy="fall_through")

    if replicate_selector not in ("all", "first", "last"):
        raise ValueError(f"replicate_selector must be 'all', 'first', or 'last'; got {replicate_selector!r}")

    # Narrow the input per replicate_selector before sorting. 'first'
    # keeps rows matching min(metadata.replicate) (ignoring None rows if
    # any integer replicate is present), 'last' keeps max. Selected
    # rows later have their ReplayKey.replicate emitted as None so the
    # resulting entries act as wildcards on the replicate axis.
    results_iter = list(getattr(result_set, "results", None) or [])
    if replicate_selector != "all":
        replicate_values = [getattr(vr.metadata, "replicate", None) for vr in results_iter]
        non_none = [r for r in replicate_values if r is not None]
        if non_none:
            target = min(non_none) if replicate_selector == "first" else max(non_none)
            results_iter = [vr for vr in results_iter if getattr(vr.metadata, "replicate", None) == target]
        # If every row has replicate=None, keep them all; there is
        # nothing to select across.

    # Sort scenario turns into a predictable order before applying the
    # per-replicate visit counter. QA turns are emitted in iteration order.
    results_iter.sort(
        key=lambda vr: (
            _none_last(getattr(vr.metadata, "scenario_id", None)),
            _none_last(getattr(vr.metadata, "scenario_node", None)),
            _none_last(getattr(vr.metadata, "replicate", None)),
            _none_last(getattr(vr.metadata, "scenario_turn", None)),
        )
    )

    visit_counts: dict[tuple[str, str, int | None], int] = {}

    for vr in results_iter:
        md = vr.metadata

        if only_successful:
            failure = getattr(md, "failure", None)
            if failure is not None and getattr(failure, "category", None) != FailureCategory.CONTENT:
                continue

        answering_display = _answering_display(md)
        if answering_model_ids is not None and answering_display not in answering_model_ids:
            continue

        scenario_id = getattr(md, "scenario_id", None)
        scenario_node = getattr(md, "scenario_node", None)
        replicate_value = getattr(md, "replicate", None)

        if scenario_ids is not None and scenario_id is not None and scenario_id not in scenario_ids:
            continue

        emitted_replicate = None if replicate_selector != "all" else replicate_value

        visit_index: int | None
        if scenario_id is not None:
            cell_key: tuple[str, str, int | None] = (
                scenario_id,
                scenario_node or "",
                emitted_replicate,
            )
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
            replicate=emitted_replicate,
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
    replicate: int | None = None,
) -> ReplayStore:
    """Build a ReplayStore from one ScenarioExecutionResult.

    ``answering_model_id`` is required because ScenarioExecutionResult
    does not carry per-turn model identity. The caller should pass
    ``ModelIdentity.from_model_config(m, role="answering").display_string``
    for consistency with ``capture_from_result_set``.

    ``replicate`` is threaded into every emitted ReplayKey. This
    helper is single-scenario by construction and does not inspect
    any replicate metadata on the scenario result itself; the caller
    owns the value.

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
        replicate: Optional replicate index threaded into every emitted
            ReplayKey. Defaults to None, matching pre-R1 behavior.

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

    # Counter is keyed by node_id alone (not (scenario_id, node_id) like
    # capture_from_result_set) because this function is single-scenario
    # by construction; cross-scenario counting is the responsibility of
    # capture_from_result_set on a mixed flat list.
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
        if include_agent_traces and trace_messages is not None:
            trace_messages_dicts = [_message_to_dict(m) for m in trace_messages]
        else:
            trace_messages_dicts = None

        # Preserve empty dicts/lists. An empty parsed_fields {} is a valid
        # captured value (template with no extractable fields) and is
        # distinct from None (no parsed_fields captured at all).
        entry = ReplayEntry(
            raw_trace=getattr(record, "raw_response", "") or "",
            trace_messages=trace_messages_dicts,
            parsed_answer_fields=parsed_fields if parsed_fields is not None else None,
            verify_result=getattr(record, "verify_result", None),
            captured_model_id=answering_model_id,
            captured_at=datetime.now(UTC).isoformat(),
        )

        key = ReplayKey(
            question_id=question_id,
            scenario_id=effective_scenario_id,
            scenario_node=node_id,
            answering_model_id=answering_model_id,
            visit_index=visit_index,
            replicate=replicate,
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

    trace_messages = getattr(template, "trace_messages", None)
    if not include_agent_traces:
        trace_messages = None

    md = vr.metadata
    # Preserve empty dicts/lists. An empty parsed_fields {} is a valid
    # captured value (template with no extractable fields) and is
    # distinct from None (no parsed_fields captured at all). Same for
    # trace_messages: an empty list means an agent ran with no tool
    # calls; None means trace messages were never captured.
    return ReplayEntry(
        raw_trace=getattr(template, "raw_llm_response", "") or "",
        trace_messages=trace_messages if trace_messages is not None else None,
        parsed_answer_fields=parsed_fields if parsed_fields is not None else None,
        verify_result=getattr(template, "verify_result", None),
        usage_metadata=getattr(template, "usage_metadata", None),
        agent_metrics=getattr(template, "agent_metrics", None),
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
