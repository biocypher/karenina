"""ReplayStore and its keys.

This module defines the core value types for the replay layer:
ReplayKey (frozen identifier), ReplayEntry (captured turn payload),
and ReplayStore (specificity-aware lookup).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, cast

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

    answering_model_id, visit_index, and replicate are optional
    refinements. A registered entry with answering_model_id=None matches
    any answering model; an entry with visit_index=None matches any
    visit; an entry with replicate=None matches any replicate.

    Lookup specificity (see ReplayStore.lookup for details) falls back
    from most-specific to least-specific; first hit wins.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    scenario_id: str | None = None
    scenario_node: str | None = None
    answering_model_id: str | None = None
    visit_index: int | None = None
    replicate: int | None = None


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


# Inner-index cell layout: dict[(model_id, visit_index, replicate)] -> ReplayEntry
_InnerCell = dict[tuple[str | None, int | None, int | None], ReplayEntry]


def _axis_options(axis_value: str | int | None) -> tuple[str | int | None, ...]:
    """Return the values to try for one axis in specificity order.

    If the request axis is set, try the concrete value then None. If
    the request axis is None, try only None (there is no more-specific
    value to widen from).
    """
    if axis_value is None:
        return (None,)
    return (axis_value, None)


def _ladder_rungs(
    model: str | None,
    visit: int | None,
    replicate: int | None,
) -> list[tuple[str | None, int | None, int | None]]:
    """Return the specificity-ordered list of rungs to probe.

    Rungs are generated in most-to-least specific order over the three
    axes (model, visit, replicate). A rung is emitted only when it is
    not already a duplicate of an earlier rung under the current
    request. Duplicates occur when an input axis is already None at a
    higher-specificity rung.

    The concrete sequence for a request with all three axes set:
        (M, V, R)
        (M, V, None)
        (M, None, R)
        (M, None, None)
        (None, V, R)
        (None, V, None)
        (None, None, R)
        (None, None, None)

    When ``replicate`` is None, the odd rungs collapse into their
    even neighbors and are omitted (so only four rungs remain,
    matching the pre-R1 2D ladder).
    """
    rungs: list[tuple[str | None, int | None, int | None]] = []
    for m in _axis_options(model):
        for v in _axis_options(visit):
            for r in _axis_options(replicate):
                rungs.append(cast(tuple[str | None, int | None, int | None], (m, v, r)))
    seen: set[tuple[str | None, int | None, int | None]] = set()
    deduped: list[tuple[str | None, int | None, int | None]] = []
    for rung in rungs:
        if rung not in seen:
            seen.add(rung)
            deduped.append(rung)
    return deduped


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
        inner_key = (key.answering_model_id, key.visit_index, key.replicate)
        existing = self._lookup_cell(key)
        if existing is not None and inner_key in existing:
            logger.warning(
                "Replay register: duplicate key overwritten (question_id=%s scenario=%s node=%s model=%s visit=%s replicate=%s)",
                key.question_id,
                key.scenario_id,
                key.scenario_node,
                key.answering_model_id,
                key.visit_index,
                key.replicate,
            )
            # Drop the prior entries-list row so there is exactly one
            # entry per key in the source of truth. ReplayKey is frozen
            # and Pydantic v2 gives it value equality across all fields,
            # so plain `!=` future-proofs this against new identity fields.
            self.entries = [(k, e) for (k, e) in self.entries if k != key]
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
        replicate: int | None = None,
    ) -> ReplayEntry | None:
        """Look up an entry by position.

        Returns the matching ReplayEntry or None on miss. In strict
        miss_policy, None is replaced with a raised ReplayMissError.
        """
        if scenario_id is not None:
            inner = self._scenario_index.get((scenario_id, scenario_node or ""))
        else:
            inner = self._qa_index.get(question_id)

        entry = self._walk_inner(inner, answering_model_id, visit_index, replicate) if inner else None

        if entry is None and self.miss_policy == "strict":
            key = ReplayKey(
                question_id=question_id,
                scenario_id=scenario_id,
                scenario_node=scenario_node,
                answering_model_id=answering_model_id,
                visit_index=visit_index,
                replicate=replicate,
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
    # Persistence convenience
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Convenience wrapper around karenina.replay.persistence.dump."""
        from karenina.replay.persistence import dump as _dump

        _dump(self, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        miss_policy: ReplayMissPolicy | None = None,
    ) -> ReplayStore:
        """Convenience wrapper around karenina.replay.persistence.load."""
        from karenina.replay.persistence import load as _load

        return _load(path, miss_policy=miss_policy)

    @classmethod
    def from_manual_traces(
        cls,
        manual_traces: Any,  # noqa: ARG003 - kept for API symmetry; lookup goes through the trace manager directly
        benchmark: Any,
        *,
        miss_policy: ReplayMissPolicy = "strict",
    ) -> ReplayStore:
        """Build a ReplayStore from a legacy ManualTraces instance.

        Walks ``benchmark._questions_cache`` forward (md5 is a one-way
        hash of the question text, so we cannot reconstruct text from
        the hashes stored by ManualTraceManager). For each question,
        computes ``md5(question_text)`` to look up the registered trace
        and ``generate_question_id(question_text)`` for the URN, then
        emits a wildcard entry keyed by ``ReplayKey(question_id=<URN>,
        answering_model_id=None, visit_index=None)``.
        """
        import hashlib

        from karenina.adapters.manual import get_manual_trace_with_metrics
        from karenina.utils.checkpoint import generate_question_id

        store = cls(miss_policy=miss_policy)

        questions_cache = getattr(benchmark, "_questions_cache", None)
        if questions_cache is None:
            base = getattr(benchmark, "_base", None)
            questions_cache = getattr(base, "_questions_cache", {}) if base is not None else {}

        for question_urn_id in questions_cache:
            question_data = benchmark.get_question(question_urn_id)
            question_text = question_data.get("question")
            if not question_text:
                continue
            md5 = hashlib.md5(question_text.encode("utf-8")).hexdigest()
            trace, metrics = get_manual_trace_with_metrics(md5)
            if trace is None:
                continue
            question_id = generate_question_id(question_text)
            store.register(
                ReplayKey(question_id=question_id),
                ReplayEntry(
                    raw_trace=trace,
                    agent_metrics=metrics,
                    captured_model_id="manual",
                ),
            )

        return store

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_indexes(self) -> None:
        scen: dict[tuple[str, str], _InnerCell] = {}
        qa: dict[str, _InnerCell] = {}
        for key, entry in self.entries:
            inner_key = (key.answering_model_id, key.visit_index, key.replicate)
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
        replicate: int | None,
    ) -> ReplayEntry | None:
        """Walk the (model, visit, replicate) specificity ladder. Most-specific wins.

        The ladder yields rungs in specificity order (model > visit >
        replicate). A rung is skipped when it would duplicate an earlier
        rung because one of the request axes is None. The duplicate-skip
        rule is a function of REQUEST axes only; store contents never
        influence which rungs are walked.
        """
        for rung in _ladder_rungs(model, visit, replicate):
            hit = inner.get(rung)
            if hit is not None:
                return hit
        return None
