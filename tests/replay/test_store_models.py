"""Unit tests for ReplayKey and ReplayEntry value types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from karenina.replay.store import ReplayEntry, ReplayKey


@pytest.mark.unit
class TestReplayKey:
    def test_qa_mode_defaults(self):
        key = ReplayKey(question_id="urn:uuid:question-q1-a1b2c3d4")
        assert key.question_id == "urn:uuid:question-q1-a1b2c3d4"
        assert key.scenario_id is None
        assert key.scenario_node is None
        assert key.answering_model_id is None
        assert key.visit_index is None

    def test_scenario_mode(self):
        key = ReplayKey(
            question_id="urn:uuid:question-q1-a1b2c3d4",
            scenario_id="syco",
            scenario_node="setup",
            answering_model_id="anthropic/claude-sonnet-4-6 (answering)",
            visit_index=0,
        )
        assert key.scenario_id == "syco"
        assert key.scenario_node == "setup"
        assert key.visit_index == 0

    def test_frozen(self):
        key = ReplayKey(question_id="q")
        with pytest.raises(ValidationError):
            key.scenario_id = "forbidden"

    def test_hashable_for_dict_keys(self):
        k1 = ReplayKey(question_id="q", scenario_id="s", scenario_node="n")
        k2 = ReplayKey(question_id="q", scenario_id="s", scenario_node="n")
        assert k1 == k2
        assert hash(k1) == hash(k2)
        d = {k1: "value"}
        assert d[k2] == "value"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ReplayKey(question_id="q", bogus="nope")


@pytest.mark.unit
class TestReplayEntry:
    def test_minimal(self):
        entry = ReplayEntry(raw_trace="hello")
        assert entry.raw_trace == "hello"
        assert entry.trace_messages is None
        assert entry.parsed_answer_fields is None
        assert entry.verify_result is None
        assert entry.agent_metrics is None
        assert entry.captured_model_id is None
        assert entry.captured_at is None

    def test_empty_raw_trace_is_allowed(self):
        """Empty traces are legitimate (abstention, streaming timeout, etc.)."""
        entry = ReplayEntry(raw_trace="")
        assert entry.raw_trace == ""

    def test_full_payload(self):
        entry = ReplayEntry(
            raw_trace="final answer",
            trace_messages=[{"role": "assistant", "content": "final answer"}],
            parsed_answer_fields={"mechanism": "COX inhibition"},
            verify_result=True,
            agent_metrics={"iterations": 2, "limit_reached": False},
            captured_model_id="anthropic/claude-sonnet-4-6 (answering)",
            captured_at="2026-04-08T12:34:56Z",
        )
        assert entry.parsed_answer_fields == {"mechanism": "COX inhibition"}
        assert entry.verify_result is True
        assert entry.agent_metrics == {"iterations": 2, "limit_reached": False}

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ReplayEntry(raw_trace="x", bogus="nope")


@pytest.mark.unit
class TestReplayKeyReplicateField:
    def test_replicate_defaults_to_none(self):
        key = ReplayKey(question_id="q1")
        assert key.replicate is None

    def test_replicate_accepts_int(self):
        key = ReplayKey(question_id="q1", replicate=3)
        assert key.replicate == 3

    def test_keys_differing_only_in_replicate_are_not_equal(self):
        a = ReplayKey(question_id="q1", replicate=1)
        b = ReplayKey(question_id="q1", replicate=2)
        assert a != b
        assert hash(a) != hash(b)

    def test_keys_identical_in_all_fields_including_replicate_are_equal(self):
        a = ReplayKey(question_id="q1", replicate=1)
        b = ReplayKey(question_id="q1", replicate=1)
        assert a == b
        assert hash(a) == hash(b)

    def test_replicate_round_trip(self):
        original = ReplayKey(
            question_id="q1",
            scenario_id="s1",
            scenario_node="n1",
            answering_model_id="m1",
            visit_index=2,
            replicate=4,
        )
        dumped = original.model_dump()
        assert dumped["replicate"] == 4
        restored = ReplayKey.model_validate(dumped)
        assert restored == original

    def test_legacy_payload_without_replicate_loads_as_none(self):
        legacy = {
            "question_id": "q1",
            "scenario_id": None,
            "scenario_node": None,
            "answering_model_id": None,
            "visit_index": None,
        }
        key = ReplayKey.model_validate(legacy)
        assert key.replicate is None
