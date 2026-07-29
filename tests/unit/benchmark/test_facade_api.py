"""Tests for facade API deprecations and their delegation contracts.

Each deprecated ``Benchmark`` results method must (a) emit a DeprecationWarning
pointing users at ``ResultsStore``, and (b) actually delegate to the new code
path rather than silently no-op'ing. The previous version of this file only
asserted (a) with empty inputs; a refactor that broke the delegation while
keeping the warning would have stayed green.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from datetime import datetime

import pytest

from karenina import Benchmark
from karenina.benchmark import Benchmark as BenchmarkAlias
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity


def _create_benchmark() -> Benchmark:
    return Benchmark.create(name="test_facade_api")


def _make_result(question_id: str = "q1", passed: bool = True) -> VerificationResult:
    """Build a minimal VerificationResult that round-trips through ResultsManager."""
    timestamp = datetime.now().isoformat()
    answering = ModelIdentity(interface="langchain", model_name="test-model")
    parsing = ModelIdentity(interface="langchain", model_name="test-model")
    result_id = VerificationResultMetadata.compute_result_id(
        question_id=question_id,
        answering=answering,
        parsing=parsing,
        timestamp=timestamp,
    )
    failure = (
        None if passed else Failure(category=FailureCategory.UNEXPECTED_ERROR, stage="generate_answer", reason="boom")
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="no_template",
            failure=failure,
            question_text=f"Question {question_id}",
            answering=answering,
            parsing=parsing,
            execution_time=0.5,
            timestamp=timestamp,
            result_id=result_id,
        ),
        template=VerificationResultTemplate(raw_llm_response="answer"),
    )


@pytest.mark.unit
class TestFacadeDeprecationWarnings:
    """Each deprecated method must emit a DeprecationWarning naming ResultsStore."""

    @pytest.mark.parametrize(
        "method_name,call",
        [
            ("store_verification_results", lambda b: b.store_verification_results({})),
            ("get_verification_results", lambda b: b.get_verification_results()),
            ("get_verification_history", lambda b: b.get_verification_history()),
            ("clear_verification_results", lambda b: b.clear_verification_results()),
            ("export_verification_results", lambda b: b.export_verification_results()),
            (
                "export_verification_results_to_file",
                lambda b: b.export_verification_results_to_file("/tmp/ignored.json"),
            ),
            (
                "load_verification_results_from_file",
                lambda b: b.load_verification_results_from_file(_write_empty_results_json()),
            ),
            ("get_verification_summary", lambda b: b.get_verification_summary()),
            ("get_all_run_names", lambda b: b.get_all_run_names()),
            ("get_results_statistics_by_run", lambda b: b.get_results_statistics_by_run()),
        ],
    )
    def test_method_warns_with_resultsstore_pointer(
        self, method_name: str, call: Callable[[Benchmark], object]
    ) -> None:
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(b)
        relevant = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(relevant) == 1, f"{method_name} should emit exactly one DeprecationWarning"
        assert "ResultsStore" in str(relevant[0].message), (
            f"{method_name} deprecation must name ResultsStore as the replacement; got {relevant[0].message!r}"
        )


def _write_empty_results_json() -> str:
    """Write an empty results JSON file the deprecated loader accepts; return path."""
    import json
    import tempfile
    from pathlib import Path

    fd, name = tempfile.mkstemp(suffix=".json")
    Path(name).write_text(json.dumps([]))
    return name


@pytest.mark.unit
class TestFacadeDelegation:
    """Deprecated methods must actually delegate to the new code path.

    Each test drives the deprecated method with real inputs and asserts that
    the side effect is observable through the new ``ResultsStore`` API. A
    refactor that drops the delegation (but keeps the warning) would fail
    these tests.
    """

    def test_store_verification_results_persists_through_results_store(self) -> None:
        """store_verification_results must write to the underlying ResultsManager."""
        b = _create_benchmark()
        result = _make_result("q1")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            b.store_verification_results({"q1": result}, run_name="deprec-run")

        runs = b._results_manager.get_all_run_names()
        assert "deprec-run" in runs
        stored = b._results_manager.get_verification_results(run_name="deprec-run")
        assert "q1" in stored
        assert stored["q1"].metadata.question_id == "q1"

    def test_get_verification_results_returns_just_stored(self) -> None:
        """get_verification_results must reflect prior store_verification_results."""
        b = _create_benchmark()
        result = _make_result("q1", passed=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            b.store_verification_results({"q1": result}, run_name="r")
            retrieved = b.get_verification_results(run_name="r")

        assert "q1" in retrieved
        # Failure must survive the delegation (not just metadata defaults).
        assert retrieved["q1"].metadata.failure is not None
        assert retrieved["q1"].metadata.failure.category is FailureCategory.UNEXPECTED_ERROR

    def test_clear_verification_results_actually_clears(self) -> None:
        """clear_verification_results must remove data from the underlying store."""
        b = _create_benchmark()
        result = _make_result("q1")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            b.store_verification_results({"q1": result}, run_name="r")
            assert b._results_manager.has_results(run_name="r")
            cleared = b.clear_verification_results(run_name="r")

        assert cleared == 1
        assert not b._results_manager.has_results(run_name="r")

    def test_get_all_run_names_lists_stored_run(self) -> None:
        """get_all_run_names must surface runs stored via the deprecated API."""
        b = _create_benchmark()
        result = _make_result("q1")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            b.store_verification_results({"q1": result}, run_name="run-x")
            names = b.get_all_run_names()

        assert "run-x" in names

    def test_benchmark_alias_is_stable(self) -> None:
        """The package-level Benchmark symbol must remain importable via the submodule.

        Both ``from karenina import Benchmark`` and ``from karenina.benchmark import
        Benchmark`` are documented entry points. A regression that removed the
        re-export would break user imports.
        """
        assert Benchmark is BenchmarkAlias
