"""ReplayStore and its keys.

This module defines the core value types for the replay layer:
ReplayKey (frozen identifier) and ReplayEntry (captured turn payload).
ReplayStore itself is added in a follow-up task; this file is
intentionally focused on the value types to keep each unit small.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ReplayKey(BaseModel):
    """Identifies a replay entry by its position in the evaluation graph.

    Two identity modes coexist in one store:
      - QA mode: scenario_id is None; lookup matches on question_id.
      - Scenario mode: scenario_id is set; lookup matches on
        (scenario_id, scenario_node). question_id is still stored for
        informational purposes but is not used for lookup.

    answering_model_id and visit_index are optional refinements. A
    registered entry with answering_model_id=None matches any answering
    model; an entry with visit_index=None matches any visit.

    Lookup specificity (see ReplayStore.lookup for details) falls back
    from most-specific to least-specific; first hit wins.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    scenario_id: str | None = None
    scenario_node: str | None = None
    answering_model_id: str | None = None
    visit_index: int | None = None


class ReplayEntry(BaseModel):
    """A single captured turn.

    raw_trace is a string (may be empty; empty traces are legitimate
    outputs of abstention, sufficiency skip, and streaming-timeout
    paths). trace_messages, when present, is a list of
    Message.to_dict() outputs for agent traces with tool calls.
    parsed_answer_fields, when present, is a plain dict suitable for
    Answer.model_validate().
    """

    model_config = ConfigDict(extra="forbid")

    raw_trace: str
    trace_messages: list[dict[str, Any]] | None = None
    parsed_answer_fields: dict[str, Any] | None = None
    agent_metrics: dict[str, Any] | None = None
    captured_model_id: str | None = None
    captured_at: str | None = None
