"""Tests for standalone ResultsStore.

Tests cover:
- Adding result sets with explicit or auto-generated run names
- Querying results by run, question, and recency
- Clearing results (all or filtered)
- Summary and statistics generation
- Export to dict and round-trip file I/O
"""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from karenina.benchmark.core.results_store import ResultsStore
from karenina.schemas.results.verification_result_set import VerificationResultSet
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity


def _make_result(
    question_id: str = "q1",
    passed: bool = True,
    execution_time: float = 1.0,
) -> VerificationResult:
    """Create a minimal VerificationResult for testing."""
    timestamp = datetime.now().isoformat()
    answering = ModelIdentity(interface="langchain", model_name="test-model")
    parsing = ModelIdentity(interface="langchain", model_name="test-model")
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering=answering,
        parsing=parsing,
        timestamp=timestamp,
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="tpl_hash",
            completed_without_errors=passed,
            error=None if passed else "Test error",
            question_text="Test question",
            answering=answering,
            parsing=parsing,
            execution_time=execution_time,
            timestamp=timestamp,
            result_id=result_id,
        ),
        template=VerificationResultTemplate(
            raw_llm_response="response",
            parsed_llm_response={"value": "x"},
            parsed_gt_response={"value": "x"},
            verify_result=passed,
            verify_granular_result={"value": passed},
        ),
    )


def _make_result_set(results: list[VerificationResult]) -> VerificationResultSet:
    """Create a VerificationResultSet from results."""
    return VerificationResultSet(results=results)


@pytest.mark.unit
class TestResultsStoreAdd:
    """Tests for adding result sets to the store."""

    def test_add_with_run_name(self):
        """Adding a result set with an explicit run name stores it under that key."""
        store = ResultsStore()
        rs = _make_result_set([_make_result("q1")])
        store.add(rs, run_name="my_run")

        assert "my_run" in store.get_all_runs()

    def test_add_auto_generates_name(self):
        """Adding without a run name generates a timestamped name."""
        store = ResultsStore()
        rs = _make_result_set([_make_result("q1")])
        name = store.add(rs)

        assert name.startswith("run_")
        assert name in store.get_all_runs()

    def test_add_returns_run_name(self):
        """add() returns the run name used (explicit or generated)."""
        store = ResultsStore()
        rs = _make_result_set([_make_result("q1")])

        explicit = store.add(rs, run_name="explicit")
        assert explicit == "explicit"

        auto = store.add(rs)
        assert auto.startswith("run_")

    def test_add_duplicate_run_name_raises(self):
        """Adding with a run name that already exists raises ValueError."""
        store = ResultsStore()
        rs = _make_result_set([_make_result("q1")])
        store.add(rs, run_name="dup")

        with pytest.raises(ValueError, match="already exists"):
            store.add(rs, run_name="dup")


@pytest.mark.unit
class TestResultsStoreQuery:
    """Tests for querying results from the store."""

    def test_get_by_run(self):
        """get_by_run returns the VerificationResultSet for a given run."""
        store = ResultsStore()
        rs = _make_result_set([_make_result("q1"), _make_result("q2")])
        store.add(rs, run_name="run_a")

        fetched = store.get_by_run("run_a")
        assert len(fetched.results) == 2

    def test_get_by_run_missing_raises(self):
        """get_by_run raises KeyError for a nonexistent run name."""
        store = ResultsStore()

        with pytest.raises(KeyError):
            store.get_by_run("no_such_run")

    def test_get_by_question(self):
        """get_by_question collects results across runs for one question."""
        store = ResultsStore()
        store.add(
            _make_result_set([_make_result("q1"), _make_result("q2")]),
            run_name="run_a",
        )
        store.add(
            _make_result_set([_make_result("q1"), _make_result("q3")]),
            run_name="run_b",
        )

        by_q1 = store.get_by_question("q1")
        assert "run_a" in by_q1
        assert "run_b" in by_q1
        assert len(by_q1["run_a"]) == 1
        assert len(by_q1["run_b"]) == 1

    def test_get_by_question_no_matches(self):
        """get_by_question returns empty dict when question is not found."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="run_a")

        assert store.get_by_question("nonexistent") == {}

    def test_get_latest(self):
        """get_latest returns the most recent result per question."""
        store = ResultsStore()
        r1 = _make_result("q1")
        r2 = _make_result("q1")
        store.add(_make_result_set([r1]), run_name="run_old")
        store.add(_make_result_set([r2]), run_name="run_new")

        latest = store.get_latest()
        # Should contain q1 from the latest run (run_new)
        assert "q1" in latest
        assert latest["q1"] is r2

    def test_get_latest_with_question_filter(self):
        """get_latest with question_id filters to that question only."""
        store = ResultsStore()
        store.add(
            _make_result_set([_make_result("q1"), _make_result("q2")]),
            run_name="run_a",
        )

        latest = store.get_latest(question_id="q2")
        assert "q2" in latest
        assert "q1" not in latest

    def test_get_latest_empty_store(self):
        """get_latest on empty store returns empty dict."""
        store = ResultsStore()
        assert store.get_latest() == {}

    def test_has_results_empty(self):
        """has_results returns False on empty store."""
        store = ResultsStore()
        assert store.has_results() is False

    def test_has_results_with_data(self):
        """has_results returns True when results exist."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r")
        assert store.has_results() is True

    def test_has_results_by_question(self):
        """has_results filters by question_id."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r")
        assert store.has_results(question_id="q1") is True
        assert store.has_results(question_id="q99") is False

    def test_has_results_by_run(self):
        """has_results filters by run_name."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r")
        assert store.has_results(run_name="r") is True
        assert store.has_results(run_name="no_run") is False

    def test_get_all_runs_ordered(self):
        """get_all_runs returns run names in insertion order."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="alpha")
        store.add(_make_result_set([_make_result("q1")]), run_name="beta")
        store.add(_make_result_set([_make_result("q1")]), run_name="gamma")

        assert store.get_all_runs() == ["alpha", "beta", "gamma"]


@pytest.mark.unit
class TestResultsStoreClear:
    """Tests for clearing results from the store."""

    def test_clear_all(self):
        """clear() with no args removes all runs."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r1")
        store.add(_make_result_set([_make_result("q2")]), run_name="r2")

        cleared = store.clear()
        assert cleared == 2  # two result objects total
        assert store.get_all_runs() == []

    def test_clear_by_run(self):
        """clear(run_name=...) removes only that run."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="keep")
        store.add(_make_result_set([_make_result("q2")]), run_name="remove")

        cleared = store.clear(run_name="remove")
        assert cleared == 1
        assert store.get_all_runs() == ["keep"]

    def test_clear_by_question_ids(self):
        """clear(question_ids=...) removes matching results from all runs."""
        store = ResultsStore()
        store.add(
            _make_result_set([_make_result("q1"), _make_result("q2")]),
            run_name="r1",
        )

        cleared = store.clear(question_ids=["q1"])
        assert cleared == 1

        # q2 should remain in r1
        remaining = store.get_by_run("r1")
        assert len(remaining.results) == 1
        assert remaining.results[0].metadata.question_id == "q2"

    def test_clear_by_question_removes_empty_run(self):
        """Clearing all questions from a run removes the run entry entirely."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r1")

        store.clear(question_ids=["q1"])
        assert "r1" not in store.get_all_runs()

    def test_clear_nonexistent_run(self):
        """Clearing a nonexistent run returns 0."""
        store = ResultsStore()
        assert store.clear(run_name="nope") == 0


@pytest.mark.unit
class TestResultsStoreSummary:
    """Tests for summary and statistics methods."""

    def test_get_summary_empty(self):
        """get_summary on empty store returns zeros."""
        store = ResultsStore()
        summary = store.get_summary()
        assert summary["total_results"] == 0
        assert summary["total_runs"] == 0

    def test_get_summary_with_data(self):
        """get_summary returns correct aggregate counts."""
        store = ResultsStore()
        store.add(
            _make_result_set(
                [
                    _make_result("q1", passed=True),
                    _make_result("q2", passed=False),
                ]
            ),
            run_name="r1",
        )
        store.add(
            _make_result_set([_make_result("q1", passed=True)]),
            run_name="r2",
        )

        summary = store.get_summary()
        assert summary["total_results"] == 3
        assert summary["total_runs"] == 2
        assert summary["unique_questions"] == 2

    def test_get_summary_for_run(self):
        """get_summary(run_name=...) scopes to that run."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r1")
        store.add(
            _make_result_set([_make_result("q2"), _make_result("q3")]),
            run_name="r2",
        )

        summary = store.get_summary(run_name="r2")
        assert summary["total_results"] == 2
        assert summary["total_runs"] == 1

    def test_get_statistics_by_run(self):
        """get_statistics_by_run returns per-run summaries."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r1")
        store.add(
            _make_result_set([_make_result("q2"), _make_result("q3")]),
            run_name="r2",
        )

        stats = store.get_statistics_by_run()
        assert "r1" in stats
        assert "r2" in stats
        assert stats["r1"]["total_results"] == 1
        assert stats["r2"]["total_results"] == 2

    def test_export_returns_serializable(self):
        """export() returns a JSON-serializable dict."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r1")

        exported = store.export()
        # Must be serializable
        json_str = json.dumps(exported)
        assert isinstance(json.loads(json_str), dict)
        assert "r1" in exported["runs"]

    def test_export_filters_by_run(self):
        """export(run_name=...) includes only that run."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r1")
        store.add(_make_result_set([_make_result("q2")]), run_name="r2")

        exported = store.export(run_name="r1")
        assert "r1" in exported["runs"]
        assert "r2" not in exported["runs"]

    def test_export_filters_by_question(self):
        """export(question_ids=...) includes only matching results."""
        store = ResultsStore()
        store.add(
            _make_result_set([_make_result("q1"), _make_result("q2")]),
            run_name="r1",
        )

        exported = store.export(question_ids=["q1"])
        results_in_r1 = exported["runs"]["r1"]
        assert len(results_in_r1) == 1

    def test_export_to_file_and_from_file_roundtrip(self, tmp_path: Path):
        """export_to_file then from_file produces equivalent store."""
        store = ResultsStore()
        store.add(
            _make_result_set([_make_result("q1"), _make_result("q2")]),
            run_name="run_a",
        )

        file_path = tmp_path / "results.json"
        store.export_to_file(file_path)

        loaded = ResultsStore.from_file(file_path)
        assert loaded.get_all_runs() == ["run_a"]
        assert len(loaded.get_by_run("run_a").results) == 2

    def test_export_includes_scenario_outcomes(self):
        """export() surfaces per-scenario outcome_results when present."""
        store = ResultsStore()
        scenario_results = [
            SimpleNamespace(
                scenario_id="scn_001",
                outcome_results={"initial_correct": True, "resists_sycophancy": False},
            ),
            SimpleNamespace(
                scenario_id="scn_002",
                outcome_results={"initial_correct": False},
            ),
        ]
        rs = VerificationResultSet(results=[_make_result("q1")], scenario_results=scenario_results)
        store.add(rs, run_name="r1")

        exported = store.export()

        assert "scenario_outcomes" in exported
        assert exported["scenario_outcomes"] == {
            "r1": {
                "scn_001": {"initial_correct": True, "resists_sycophancy": False},
                "scn_002": {"initial_correct": False},
            }
        }

    def test_export_omits_scenario_outcomes_when_absent(self):
        """export() does not include scenario_outcomes key if no outcomes are present."""
        store = ResultsStore()
        store.add(_make_result_set([_make_result("q1")]), run_name="r1")

        exported = store.export()

        assert "scenario_outcomes" not in exported
        assert exported == {"runs": exported["runs"]}

    def test_export_omits_scenario_outcomes_when_outcomes_empty(self):
        """Scenario results with empty outcome_results are skipped entirely."""
        store = ResultsStore()
        scenario_results = [SimpleNamespace(scenario_id="scn_001", outcome_results={})]
        rs = VerificationResultSet(results=[_make_result("q1")], scenario_results=scenario_results)
        store.add(rs, run_name="r1")

        exported = store.export()

        assert "scenario_outcomes" not in exported
