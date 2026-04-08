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
):
    """Build a minimal object that quacks like a VerificationResult for capture."""
    from types import SimpleNamespace

    metadata = SimpleNamespace(
        question_id=question_id,
        scenario_id=scenario_id,
        scenario_node=scenario_node,
        scenario_turn=scenario_turn,
        completed_without_errors=ok,
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

    def test_only_successful_filters_failures(self):
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
