"""ReplayStore and its keys.

This module defines the core value types for the replay layer:
ReplayKey (frozen identifier), ReplayEntry (captured turn payload),
and ReplayStore (specificity-aware lookup).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from karenina.replay.exceptions import ReplayMissError

logger = logging.getLogger(__name__)


ReplayMissPolicy = Literal["fall_through", "strict"]


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


# Inner-index cell layout: dict[(model_id, visit_index)] -> ReplayEntry
_InnerCell = dict[tuple[str | None, int | None], ReplayEntry]


class ReplayStore(BaseModel):
    """Keyed store of ReplayEntry with specificity-based lookup.

    Source of truth: the ``entries`` list of (key, entry) pairs. The
    internal scenario and QA indexes are rebuilt from ``entries`` on
    every mutation via :py:meth:`_rebuild_indexes`. Lookups are O(1)
    against the indexes.

    Thread safety: reads are safe to run concurrently during pipeline
    execution. ``register`` must NOT be called while a verification run
    is in progress.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    miss_policy: ReplayMissPolicy = "fall_through"
    entries: list[tuple[ReplayKey, ReplayEntry]] = Field(default_factory=list)

    _scenario_index: dict[tuple[str, str], _InnerCell] = PrivateAttr(default_factory=dict)
    _qa_index: dict[str, _InnerCell] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self._rebuild_indexes()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, key: ReplayKey, entry: ReplayEntry) -> None:
        """Add or overwrite an entry for ``key``.

        Overwrites emit a WARNING log. Rebuilds the indexes after
        mutation.
        """
        inner_key = (key.answering_model_id, key.visit_index)
        existing = self._lookup_cell(key)
        if existing is not None and inner_key in existing:
            logger.warning(
                "Replay register: duplicate key overwritten (question_id=%s scenario=%s node=%s model=%s visit=%s)",
                key.question_id,
                key.scenario_id,
                key.scenario_node,
                key.answering_model_id,
                key.visit_index,
            )
            # Drop the prior entries-list row so there is exactly one
            # entry per key in the source of truth.
            self.entries = [
                (k, e)
                for (k, e) in self.entries
                if not (
                    k.question_id == key.question_id
                    and k.scenario_id == key.scenario_id
                    and k.scenario_node == key.scenario_node
                    and k.answering_model_id == key.answering_model_id
                    and k.visit_index == key.visit_index
                )
            ]
        self.entries.append((key, entry))
        self._rebuild_indexes()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        *,
        question_id: str,
        scenario_id: str | None = None,
        scenario_node: str | None = None,
        answering_model_id: str | None = None,
        visit_index: int | None = None,
    ) -> ReplayEntry | None:
        """Look up an entry by position.

        Returns the matching ReplayEntry or None on miss. In strict
        miss_policy, None is replaced with a raised ReplayMissError.
        """
        if scenario_id is not None:
            inner = self._scenario_index.get((scenario_id, scenario_node or ""))
        else:
            inner = self._qa_index.get(question_id)

        entry = self._walk_inner(inner, answering_model_id, visit_index) if inner else None

        if entry is None and self.miss_policy == "strict":
            key = ReplayKey(
                question_id=question_id,
                scenario_id=scenario_id,
                scenario_node=scenario_node,
                answering_model_id=answering_model_id,
                visit_index=visit_index,
            )
            raise ReplayMissError(
                f"No replay entry for {key}",
                key=key,
            )
        return entry

    def has_any_for(
        self,
        *,
        question_id: str,
        scenario_id: str | None = None,
        scenario_node: str | None = None,
    ) -> bool:
        """Return True if any entry is registered for this outer key.

        Does NOT walk the (model, visit) ladder; used by the scenario
        executor to decide whether to reserve an AnswerTraceCache slot.
        """
        if scenario_id is not None:
            return (scenario_id, scenario_node or "") in self._scenario_index
        return question_id in self._qa_index

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_indexes(self) -> None:
        scen: dict[tuple[str, str], _InnerCell] = {}
        qa: dict[str, _InnerCell] = {}
        for key, entry in self.entries:
            inner_key = (key.answering_model_id, key.visit_index)
            if key.scenario_id is not None:
                outer = (key.scenario_id, key.scenario_node or "")
                scen.setdefault(outer, {})[inner_key] = entry
            else:
                qa.setdefault(key.question_id, {})[inner_key] = entry
        self._scenario_index = scen
        self._qa_index = qa

    def _lookup_cell(self, key: ReplayKey) -> _InnerCell | None:
        if key.scenario_id is not None:
            return self._scenario_index.get((key.scenario_id, key.scenario_node or ""))
        return self._qa_index.get(key.question_id)

    @staticmethod
    def _walk_inner(
        inner: _InnerCell,
        model: str | None,
        visit: int | None,
    ) -> ReplayEntry | None:
        """Walk the (model, visit) specificity ladder. Most-specific wins.

        The ladder step is skipped when its cell would duplicate a prior
        step (if ``model`` is None, steps that widen the model do nothing
        new; same for ``visit``).
        """
        if model is not None and visit is not None:
            hit = inner.get((model, visit))
            if hit is not None:
                return hit
            hit = inner.get((model, None))
            if hit is not None:
                return hit
            hit = inner.get((None, visit))
            if hit is not None:
                return hit
        elif model is not None:
            hit = inner.get((model, None))
            if hit is not None:
                return hit
        elif visit is not None:
            hit = inner.get((None, visit))
            if hit is not None:
                return hit
        return inner.get((None, None))
