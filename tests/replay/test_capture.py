"""Unit tests for replay capture helpers."""

from __future__ import annotations

import pytest

from karenina.replay.capture import (
    capture_from_result_set,
    capture_from_scenario_result,
)


def _build_fake_qa_result(
    *,
    question_id: str,
    scenario_id: str | None,
    scenario_node: str | None,
    scenario_turn: int | None,
    raw: str,
    parsed: dict | None,
    model_display: str,
    ok: bool = True,
    failure_category=None,
):
    """Build a minimal object that quacks like a VerificationResult for capture.

    When ``ok`` is False, ``failure_category`` is attached to the stubbed
    failure namespace so tests can exercise category-sensitive logic.
    Default is None (i.e. a generic non-CONTENT failure).
    """
    from types import SimpleNamespace

    metadata = SimpleNamespace(
        question_id=question_id,
        scenario_id=scenario_id,
        scenario_node=scenario_node,
        scenario_turn=scenario_turn,
        failure=None if ok else SimpleNamespace(reason="stub failure", category=failure_category),
        answering=SimpleNamespace(display_string=model_display),
        completed_at="2026-04-08T12:00:00Z",
    )
    template = SimpleNamespace(
        raw_llm_response=raw,
        parsed_llm_response=parsed,
        trace_messages=[],
    )
    return SimpleNamespace(metadata=metadata, template=template)


def _fake_result_set(results, scenario_results=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        results=results,
        scenario_results=scenario_results or [],
    )


@pytest.mark.unit
class TestCaptureFromResultSet:
    def test_qa_result_captured(self):
        rs = _fake_result_set(
            results=[
                _build_fake_qa_result(
                    question_id="urn:uuid:question-q1-abcd",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    raw="qa answer",
                    parsed={"value": 42},
                    model_display="gpt-5 (answering)",
                )
            ],
        )
        store = capture_from_result_set(rs)
        hit = store.lookup(
            question_id="urn:uuid:question-q1-abcd",
            answering_model_id="gpt-5 (answering)",
        )
        assert hit is not None
        assert hit.raw_trace == "qa answer"
        assert hit.parsed_answer_fields == {"value": 42}

    def test_scenario_results_use_per_node_visit_counter(self):
        rs = _fake_result_set(
            results=[
                _build_fake_qa_result(
                    question_id="q",
                    scenario_id="s",
                    scenario_node="retry",
                    scenario_turn=0,
                    raw="first",
                    parsed={"ok": True},
                    model_display="gpt-5 (answering)",
                ),
                _build_fake_qa_result(
                    question_id="q",
                    scenario_id="s",
                    scenario_node="retry",
                    scenario_turn=1,
                    raw="second",
                    parsed={"ok": True},
                    model_display="gpt-5 (answering)",
                ),
                _build_fake_qa_result(
                    question_id="q",
                    scenario_id="s",
                    scenario_node="retry",
                    scenario_turn=2,
                    raw="third",
                    parsed={"ok": True},
                    model_display="gpt-5 (answering)",
                ),
            ],
        )
        store = capture_from_result_set(rs)
        hit0 = store.lookup(
            question_id="q",
            scenario_id="s",
            scenario_node="retry",
            answering_model_id="gpt-5 (answering)",
            visit_index=0,
        )
        hit1 = store.lookup(
            question_id="q",
            scenario_id="s",
            scenario_node="retry",
            answering_model_id="gpt-5 (answering)",
            visit_index=1,
        )
        hit2 = store.lookup(
            question_id="q",
            scenario_id="s",
            scenario_node="retry",
            answering_model_id="gpt-5 (answering)",
            visit_index=2,
        )
        assert hit0.raw_trace == "first"
        assert hit1.raw_trace == "second"
        assert hit2.raw_trace == "third"

    def test_only_successful_filters_pipeline_failures(self):
        rs = _fake_result_set(
            results=[
                _build_fake_qa_result(
                    question_id="q1",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    raw="good",
                    parsed=None,
                    model_display="m",
                    ok=True,
                ),
                _build_fake_qa_result(
                    question_id="q2",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    raw="bad",
                    parsed=None,
                    model_display="m",
                    ok=False,
                ),
            ],
        )
        store = capture_from_result_set(rs, only_successful=True)
        assert store.lookup(question_id="q1", answering_model_id="m") is not None
        assert store.lookup(question_id="q2", answering_model_id="m") is None

    def test_only_successful_keeps_content_failures(self):
        from karenina.schemas.results.failure import FailureCategory

        rs = _fake_result_set(
            results=[
                _build_fake_qa_result(
                    question_id="q_content",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    raw="wrong but complete",
                    parsed={"answer": "wrong"},
                    model_display="m",
                    ok=False,
                    failure_category=FailureCategory.CONTENT,
                ),
                _build_fake_qa_result(
                    question_id="q_timeout",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    raw="",
                    parsed=None,
                    model_display="m",
                    ok=False,
                    failure_category=FailureCategory.TIMEOUT,
                ),
            ],
        )
        store = capture_from_result_set(rs, only_successful=True)
        content_hit = store.lookup(question_id="q_content", answering_model_id="m")
        timeout_hit = store.lookup(question_id="q_timeout", answering_model_id="m")
        assert content_hit is not None
        assert content_hit.raw_trace == "wrong but complete"
        assert content_hit.parsed_answer_fields == {"answer": "wrong"}
        assert timeout_hit is None

    def test_only_successful_false_keeps_all_failures(self):
        from karenina.schemas.results.failure import FailureCategory

        rs = _fake_result_set(
            results=[
                _build_fake_qa_result(
                    question_id="q_timeout",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    raw="partial",
                    parsed=None,
                    model_display="m",
                    ok=False,
                    failure_category=FailureCategory.TIMEOUT,
                ),
            ],
        )
        store = capture_from_result_set(rs, only_successful=False)
        assert store.lookup(question_id="q_timeout", answering_model_id="m") is not None

    def test_answering_model_filter(self):
        rs = _fake_result_set(
            results=[
                _build_fake_qa_result(
                    question_id="q",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    raw="A",
                    parsed=None,
                    model_display="gpt-5",
                ),
                _build_fake_qa_result(
                    question_id="q",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    raw="B",
                    parsed=None,
                    model_display="claude",
                ),
            ],
        )
        store = capture_from_result_set(rs, answering_model_ids={"gpt-5"})
        assert store.lookup(question_id="q", answering_model_id="gpt-5") is not None
        assert store.lookup(question_id="q", answering_model_id="claude") is None

    def test_include_parsed_false_drops_parsed_fields(self):
        rs = _fake_result_set(
            results=[
                _build_fake_qa_result(
                    question_id="q",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    raw="x",
                    parsed={"ok": True},
                    model_display="m",
                ),
            ],
        )
        store = capture_from_result_set(rs, include_parsed=False)
        hit = store.lookup(question_id="q", answering_model_id="m")
        assert hit is not None
        assert hit.parsed_answer_fields is None

    def test_returned_store_has_fall_through_policy(self):
        rs = _fake_result_set(results=[])
        store = capture_from_result_set(rs)
        assert store.miss_policy == "fall_through"

    def test_empty_parsed_dict_preserved_not_dropped_to_none(self):
        """An empty parsed_llm_response {} is a valid captured value
        (template with no extractable fields) and must not be coerced to None."""
        rs = _fake_result_set(
            results=[
                _build_fake_qa_result(
                    question_id="q",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    raw="x",
                    parsed={},
                    model_display="m",
                ),
            ],
        )
        store = capture_from_result_set(rs)
        hit = store.lookup(question_id="q", answering_model_id="m")
        assert hit is not None
        assert hit.parsed_answer_fields == {}


@pytest.mark.unit
class TestCaptureFromScenarioResult:
    def _turn(self, node_id: str, raw: str, parsed_fields: dict):
        from types import SimpleNamespace

        return SimpleNamespace(
            node_id=node_id,
            raw_response=raw,
            trace_messages=[],
            parsed_fields=parsed_fields,
            question_text=f"question for {node_id}",
        )

    def _scenario_result(self, scenario_id: str, history):
        from types import SimpleNamespace

        return SimpleNamespace(
            scenario_id=scenario_id,
            history=history,
            status="completed",
        )

    def test_simple_two_node_capture(self):
        result = self._scenario_result(
            scenario_id="s",
            history=[
                self._turn("setup", "setup answer", {"ok": True}),
                self._turn("followup", "followup answer", {"ok": True}),
            ],
        )
        store = capture_from_scenario_result(result, answering_model_id="gpt-5")
        assert (
            store.lookup(
                question_id=_question_id_for("question for setup"),
                scenario_id="s",
                scenario_node="setup",
                answering_model_id="gpt-5",
                visit_index=0,
            )
            is not None
        )
        assert (
            store.lookup(
                question_id=_question_id_for("question for followup"),
                scenario_id="s",
                scenario_node="followup",
                answering_model_id="gpt-5",
                visit_index=0,
            )
            is not None
        )

    def test_retry_loop_visit_counter(self):
        result = self._scenario_result(
            scenario_id="s",
            history=[
                self._turn("retry", "attempt 0", {"ok": False}),
                self._turn("retry", "attempt 1", {"ok": False}),
                self._turn("retry", "attempt 2", {"ok": True}),
            ],
        )
        store = capture_from_scenario_result(result, answering_model_id="gpt-5")
        qid = _question_id_for("question for retry")
        assert (
            store.lookup(
                question_id=qid, scenario_id="s", scenario_node="retry", answering_model_id="gpt-5", visit_index=0
            ).raw_trace
            == "attempt 0"
        )
        assert (
            store.lookup(
                question_id=qid, scenario_id="s", scenario_node="retry", answering_model_id="gpt-5", visit_index=2
            ).raw_trace
            == "attempt 2"
        )

    def test_nodes_filter(self):
        result = self._scenario_result(
            scenario_id="s",
            history=[
                self._turn("setup", "a", {"ok": True}),
                self._turn("followup", "b", {"ok": True}),
            ],
        )
        store = capture_from_scenario_result(result, answering_model_id="gpt-5", nodes={"setup"})
        assert (
            store.lookup(
                question_id=_question_id_for("question for setup"),
                scenario_id="s",
                scenario_node="setup",
                answering_model_id="gpt-5",
                visit_index=0,
            )
            is not None
        )
        assert (
            store.lookup(
                question_id=_question_id_for("question for followup"),
                scenario_id="s",
                scenario_node="followup",
                answering_model_id="gpt-5",
                visit_index=0,
            )
            is None
        )


def _question_id_for(text: str) -> str:
    from karenina.utils.checkpoint import generate_question_id

    return generate_question_id(text)


def _build_fake_qa_result_with_replicate(
    *,
    question_id: str,
    scenario_id: str | None,
    scenario_node: str | None,
    scenario_turn: int | None,
    replicate: int | None,
    raw: str,
    parsed: dict | None,
    model_display: str,
    ok: bool = True,
):
    """Same as _build_fake_qa_result but attaches metadata.replicate."""
    from types import SimpleNamespace

    metadata = SimpleNamespace(
        question_id=question_id,
        scenario_id=scenario_id,
        scenario_node=scenario_node,
        scenario_turn=scenario_turn,
        replicate=replicate,
        failure=None if ok else SimpleNamespace(reason="stub failure"),
        answering=SimpleNamespace(display_string=model_display),
        completed_at="2026-04-08T12:00:00Z",
    )
    template = SimpleNamespace(
        raw_llm_response=raw,
        parsed_llm_response=parsed,
        trace_messages=[],
    )
    return SimpleNamespace(metadata=metadata, template=template)


@pytest.mark.unit
class TestCaptureFromResultSetWithReplicate:
    def test_qa_three_replicates_produce_three_entries(self):
        rs = _fake_result_set(
            results=[
                _build_fake_qa_result_with_replicate(
                    question_id="urn:uuid:q-rep",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    replicate=rep,
                    raw=f"raw-{rep}",
                    parsed={"value": rep},
                    model_display="gpt-5",
                )
                for rep in (1, 2, 3)
            ]
        )
        store = capture_from_result_set(rs)
        assert len(store.entries) == 3
        replicates = {k.replicate for k, _ in store.entries}
        assert replicates == {1, 2, 3}
        for key, entry in store.entries:
            assert key.visit_index is None
            assert entry.raw_trace == f"raw-{key.replicate}"

    def test_scenario_two_visits_two_replicates_produce_four_entries(self):
        # Two replicates, each visits node "n1" twice (scenario_turn 0
        # and 1). The visit counter must reset per replicate.
        results = []
        for rep in (1, 2):
            for turn in (0, 1):
                results.append(
                    _build_fake_qa_result_with_replicate(
                        question_id="q1",
                        scenario_id="s1",
                        scenario_node="n1",
                        scenario_turn=turn,
                        replicate=rep,
                        raw=f"raw-r{rep}-t{turn}",
                        parsed=None,
                        model_display="gpt-5",
                    )
                )
        store = capture_from_result_set(_fake_result_set(results))
        assert len(store.entries) == 4
        seen = {(k.replicate, k.visit_index) for k, _ in store.entries}
        assert seen == {(1, 0), (1, 1), (2, 0), (2, 1)}

    def test_mixed_replicate_none_and_int_sort_stability(self):
        # Same (scenario_id, scenario_node) but one row has
        # replicate=None and the other has replicate=1. They form
        # distinct counter blocks; visit_index=0 appears in each block
        # independently.
        results = [
            _build_fake_qa_result_with_replicate(
                question_id="q1",
                scenario_id="s1",
                scenario_node="n1",
                scenario_turn=0,
                replicate=None,
                raw="raw-none",
                parsed=None,
                model_display="gpt-5",
            ),
            _build_fake_qa_result_with_replicate(
                question_id="q1",
                scenario_id="s1",
                scenario_node="n1",
                scenario_turn=0,
                replicate=1,
                raw="raw-rep1",
                parsed=None,
                model_display="gpt-5",
            ),
        ]
        store = capture_from_result_set(_fake_result_set(results))
        assert len(store.entries) == 2
        by_rep = {k.replicate: (k.visit_index, e.raw_trace) for k, e in store.entries}
        assert by_rep[None] == (0, "raw-none")
        assert by_rep[1] == (0, "raw-rep1")


@pytest.mark.unit
class TestCaptureFromResultSetReplicateSelector:
    def _three_replicate_rs(self):
        return _fake_result_set(
            results=[
                _build_fake_qa_result_with_replicate(
                    question_id="q1",
                    scenario_id=None,
                    scenario_node=None,
                    scenario_turn=None,
                    replicate=rep,
                    raw=f"raw-{rep}",
                    parsed=None,
                    model_display="gpt-5",
                )
                for rep in (1, 2, 3)
            ]
        )

    def test_all_is_default(self):
        store = capture_from_result_set(self._three_replicate_rs())
        reps = sorted(k.replicate for k, _ in store.entries)
        assert reps == [1, 2, 3]

    def test_first_keeps_min_with_replicate_none(self):
        store = capture_from_result_set(
            self._three_replicate_rs(),
            replicate_selector="first",
        )
        assert len(store.entries) == 1
        key, entry = store.entries[0]
        assert key.replicate is None
        # Model and visit axes remain concrete.
        assert key.answering_model_id == "gpt-5"
        assert key.visit_index is None
        assert entry.raw_trace == "raw-1"

    def test_last_keeps_max_with_replicate_none(self):
        store = capture_from_result_set(
            self._three_replicate_rs(),
            replicate_selector="last",
        )
        assert len(store.entries) == 1
        key, entry = store.entries[0]
        assert key.replicate is None
        assert entry.raw_trace == "raw-3"

    def test_invalid_selector_raises(self):
        with pytest.raises(ValueError, match="replicate_selector"):
            capture_from_result_set(
                self._three_replicate_rs(),
                replicate_selector="bogus",
            )


@pytest.mark.unit
class TestCaptureFromScenarioResultWithReplicate:
    def test_replicate_threaded_into_keys(self):
        from types import SimpleNamespace

        record_a = SimpleNamespace(
            node_id="n1",
            question_text="What?",
            raw_response="raw-a",
            parsed_fields=None,
            trace_messages=None,
        )
        record_b = SimpleNamespace(
            node_id="n1",
            question_text="What?",
            raw_response="raw-b",
            parsed_fields=None,
            trace_messages=None,
        )
        scenario_result = SimpleNamespace(
            scenario_id="s1",
            history=[record_a, record_b],
        )
        store = capture_from_scenario_result(
            scenario_result,
            answering_model_id="gpt-5",
            replicate=5,
        )
        assert len(store.entries) == 2
        for key, _ in store.entries:
            assert key.replicate == 5
            assert key.scenario_id == "s1"
            assert key.answering_model_id == "gpt-5"
        assert {k.visit_index for k, _ in store.entries} == {0, 1}

    def test_default_replicate_is_none(self):
        from types import SimpleNamespace

        record = SimpleNamespace(
            node_id="n1",
            question_text="What?",
            raw_response="raw",
            parsed_fields=None,
            trace_messages=None,
        )
        scenario_result = SimpleNamespace(
            scenario_id="s1",
            history=[record],
        )
        store = capture_from_scenario_result(
            scenario_result,
            answering_model_id="gpt-5",
        )
        assert len(store.entries) == 1
        key, _ = store.entries[0]
        assert key.replicate is None
