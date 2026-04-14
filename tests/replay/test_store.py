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


@pytest.mark.unit
class TestReplayStoreSpecificityLadder3D:
    """Parametrized coverage of every rung in the 8-rung 3D ladder.

    Each test registers one entry at one rung and asserts that a
    fully-specific request (model=M, visit=V, replicate=R) resolves to
    it when no higher-specificity rung is populated.
    """

    @pytest.mark.parametrize(
        "stored_inner, description",
        [
            (("M", 0, 1), "most-specific"),
            (("M", 0, None), "drop-replicate"),
            (("M", None, 1), "drop-visit"),
            (("M", None, None), "drop-visit-and-replicate"),
            ((None, 0, 1), "drop-model"),
            ((None, 0, None), "drop-model-and-replicate"),
            ((None, None, 1), "drop-model-and-visit"),
            ((None, None, None), "all-wildcard"),
        ],
    )
    def test_every_rung_resolvable(self, stored_inner, description):  # noqa: ARG002
        store = ReplayStore()
        model, visit, replicate = stored_inner
        key = ReplayKey(
            question_id="q1",
            answering_model_id=model,
            visit_index=visit,
            replicate=replicate,
        )
        store.register(key, _entry(description))
        hit = store.lookup(
            question_id="q1",
            answering_model_id="M",
            visit_index=0,
            replicate=1,
        )
        assert hit is not None
        assert hit.raw_trace == f"trace-{description}"

    def test_exact_beats_all_wildcards(self):
        """With every rung populated, the most-specific one wins."""
        store = ReplayStore()
        base = {"question_id": "q1"}
        rungs = [
            ("M", 0, 1, "exact"),
            ("M", 0, None, "drop-r"),
            ("M", None, 1, "drop-v"),
            ("M", None, None, "drop-vr"),
            (None, 0, 1, "drop-m"),
            (None, 0, None, "drop-mr"),
            (None, None, 1, "drop-mv"),
            (None, None, None, "all-wild"),
        ]
        for m, v, r, tag in rungs:
            store.register(
                ReplayKey(**base, answering_model_id=m, visit_index=v, replicate=r),
                _entry(tag),
            )
        hit = store.lookup(
            question_id="q1",
            answering_model_id="M",
            visit_index=0,
            replicate=1,
        )
        assert hit is not None
        assert hit.raw_trace == "trace-exact"

    def test_request_replicate_none_does_not_fall_through_to_replicate_bearing(self):
        """Mixed-store edge case: a store containing both (M, V, 1) and
        (M, V, None), queried with replicate=None, must resolve to the
        wildcard entry (M, V, None). The replicate-bearing entry is not
        returned. Locks in the rule that the walker filters rungs by
        request axes, never by store contents.
        """
        store = ReplayStore()
        base = {"question_id": "q1", "answering_model_id": "M", "visit_index": 0}
        store.register(ReplayKey(**base, replicate=1), _entry("rep1"))
        store.register(ReplayKey(**base, replicate=None), _entry("wild"))
        hit = store.lookup(
            question_id="q1",
            answering_model_id="M",
            visit_index=0,
            replicate=None,
        )
        assert hit is not None
        assert hit.raw_trace == "trace-wild"

    def test_legacy_request_no_replicate_matches_legacy_store(self):
        """Regression: when every stored key has replicate=None and the
        request has replicate=None, behavior is exactly the pre-R1 2D
        ladder.
        """
        store = ReplayStore()
        store.register(
            ReplayKey(question_id="q1", answering_model_id=None, visit_index=None),
            _entry("legacy"),
        )
        hit = store.lookup(
            question_id="q1",
            answering_model_id="gpt-5",
            visit_index=0,
        )
        assert hit is not None
        assert hit.raw_trace == "trace-legacy"
