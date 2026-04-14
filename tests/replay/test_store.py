"""Unit tests for ReplayStore.register / lookup / has_any_for."""

from __future__ import annotations

import pytest

from karenina.replay import ReplayEntry, ReplayKey, ReplayStore
from karenina.replay.exceptions import ReplayMissError


def _entry(tag: str) -> ReplayEntry:
    return ReplayEntry(raw_trace=f"trace-{tag}")


@pytest.mark.unit
class TestReplayStoreRegisterAndLookup:
    def test_qa_exact_match(self):
        store = ReplayStore()
        key = ReplayKey(question_id="q1", answering_model_id="gpt-5")
        store.register(key, _entry("a"))
        hit = store.lookup(question_id="q1", answering_model_id="gpt-5")
        assert hit is not None
        assert hit.raw_trace == "trace-a"

    def test_qa_miss_returns_none_by_default(self):
        store = ReplayStore()
        assert store.lookup(question_id="nope") is None

    def test_qa_strict_miss_raises(self):
        store = ReplayStore(miss_policy="strict")
        with pytest.raises(ReplayMissError) as exc_info:
            store.lookup(question_id="nope")
        assert exc_info.value.key is not None

    def test_scenario_exact_match(self):
        store = ReplayStore()
        key = ReplayKey(
            question_id="q1",
            scenario_id="syco",
            scenario_node="setup",
            answering_model_id="gpt-5",
            visit_index=0,
        )
        store.register(key, _entry("setup0"))
        hit = store.lookup(
            question_id="q1",
            scenario_id="syco",
            scenario_node="setup",
            answering_model_id="gpt-5",
            visit_index=0,
        )
        assert hit is not None
        assert hit.raw_trace == "trace-setup0"

    def test_specificity_model_wildcard(self):
        """A (model=None, visit=0) entry matches a lookup for a specific model."""
        store = ReplayStore()
        wildcard = ReplayKey(
            question_id="q1",
            scenario_id="syco",
            scenario_node="setup",
            answering_model_id=None,
            visit_index=0,
        )
        store.register(wildcard, _entry("wild"))
        hit = store.lookup(
            question_id="q1",
            scenario_id="syco",
            scenario_node="setup",
            answering_model_id="gpt-5",
            visit_index=0,
        )
        assert hit is not None
        assert hit.raw_trace == "trace-wild"

    def test_specificity_exact_beats_wildcard(self):
        store = ReplayStore()
        wildcard = ReplayKey(
            question_id="q1",
            scenario_id="syco",
            scenario_node="setup",
            answering_model_id=None,
            visit_index=None,
        )
        exact = ReplayKey(
            question_id="q1",
            scenario_id="syco",
            scenario_node="setup",
            answering_model_id="gpt-5",
            visit_index=0,
        )
        store.register(wildcard, _entry("wild"))
        store.register(exact, _entry("exact"))
        hit = store.lookup(
            question_id="q1",
            scenario_id="syco",
            scenario_node="setup",
            answering_model_id="gpt-5",
            visit_index=0,
        )
        assert hit is not None
        assert hit.raw_trace == "trace-exact"

    def test_visit_wildcard(self):
        """A (model=gpt-5, visit=None) entry matches every visit."""
        store = ReplayStore()
        key = ReplayKey(
            question_id="q1",
            scenario_id="syco",
            scenario_node="retry",
            answering_model_id="gpt-5",
            visit_index=None,
        )
        store.register(key, _entry("any-visit"))
        for v in (0, 1, 2, 7):
            hit = store.lookup(
                question_id="q1",
                scenario_id="syco",
                scenario_node="retry",
                answering_model_id="gpt-5",
                visit_index=v,
            )
            assert hit is not None
            assert hit.raw_trace == "trace-any-visit"

    def test_full_wildcard(self):
        store = ReplayStore()
        key = ReplayKey(
            question_id="q1",
            scenario_id="syco",
            scenario_node="setup",
        )
        store.register(key, _entry("wild"))
        hit = store.lookup(
            question_id="q1",
            scenario_id="syco",
            scenario_node="setup",
            answering_model_id="any-model",
            visit_index=42,
        )
        assert hit is not None

    def test_duplicate_register_warns_and_overwrites(self, caplog):
        store = ReplayStore()
        key = ReplayKey(question_id="q1")
        store.register(key, _entry("first"))
        with caplog.at_level("WARNING"):
            store.register(key, _entry("second"))
        assert "duplicate" in caplog.text.lower() or "overwrit" in caplog.text.lower()
        hit = store.lookup(question_id="q1")
        assert hit is not None
        assert hit.raw_trace == "trace-second"

    def test_has_any_for_qa(self):
        store = ReplayStore()
        store.register(ReplayKey(question_id="q1"), _entry("a"))
        assert store.has_any_for(question_id="q1") is True
        assert store.has_any_for(question_id="q2") is False

    def test_has_any_for_scenario(self):
        store = ReplayStore()
        store.register(
            ReplayKey(question_id="q1", scenario_id="s", scenario_node="n"),
            _entry("a"),
        )
        assert store.has_any_for(question_id="q1", scenario_id="s", scenario_node="n") is True
        assert store.has_any_for(question_id="q1", scenario_id="s", scenario_node="other") is False

    def test_none_input_short_circuits_inner_ladder(self):
        """When lookup is called with model=None,visit=None, only the wildcard
        cell is checked; specific entries should not match."""
        store = ReplayStore()
        specific = ReplayKey(
            question_id="q1",
            answering_model_id="gpt-5",
            visit_index=0,
        )
        store.register(specific, _entry("specific"))
        hit = store.lookup(question_id="q1", answering_model_id=None, visit_index=None)
        assert hit is None

    def test_parallel_safe_reads(self):
        """Concurrent reads across workers must not mutate or race."""
        import threading

        store = ReplayStore()
        for i in range(10):
            store.register(ReplayKey(question_id=f"q{i}"), _entry(str(i)))

        results: list[ReplayEntry | None] = []
        errors: list[BaseException] = []

        def worker(idx: int) -> None:
            try:
                for _ in range(100):
                    hit = store.lookup(question_id=f"q{idx}")
                    results.append(hit)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert all(r is not None for r in results)


@pytest.mark.unit
class TestReplayStoreRegisterWithReplicate:
    def test_register_preserves_distinct_replicates(self):
        """Registering three keys that differ only in replicate must
        keep three entries, not collapse to one."""
        store = ReplayStore()
        base = {
            "question_id": "q1",
            "scenario_id": "s1",
            "scenario_node": "n1",
            "answering_model_id": "m1",
            "visit_index": 0,
        }
        for rep in (1, 2, 3):
            store.register(
                ReplayKey(**base, replicate=rep),
                _entry(f"rep{rep}"),
            )
        assert len(store.entries) == 3
        raw_traces = {e.raw_trace for _, e in store.entries}
        assert raw_traces == {"trace-rep1", "trace-rep2", "trace-rep3"}

    def test_register_overwrite_with_replicate(self):
        """Duplicate (outer + model + visit + replicate) overwrites,
        with a warning, exactly like the 2D case."""
        store = ReplayStore()
        base = {
            "question_id": "q1",
            "answering_model_id": "m1",
            "visit_index": None,
            "replicate": 7,
        }
        store.register(ReplayKey(**base), _entry("first"))
        store.register(ReplayKey(**base), _entry("second"))
        assert len(store.entries) == 1
        assert store.entries[0][1].raw_trace == "trace-second"
