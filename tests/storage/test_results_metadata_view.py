"""Tests for the results_metadata SQLAlchemy view column contract.

Guards the public column surface of ``results_metadata_view`` against
accidental regressions. The legacy ``completed_without_errors`` /
``error`` / ``error_category`` / ``failed_stage`` fields must no longer
appear; the unified failure taxonomy columns plus ``caveats`` must.

Also exercises the view's SQL CASE statement that maps
``metadata_failure_category`` to ``metadata_failure_group``. That CASE
statement duplicates ``karenina.schemas.results.failure.CATEGORY_TO_GROUP``
in SQL — without a behavioral test, adding a new ``FailureCategory`` and
updating the Python dict but forgetting the SQL CASE would silently
produce ``NULL`` groups in the view.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

from karenina.schemas.results.failure import CATEGORY_TO_GROUP, Failure, FailureCategory
from karenina.storage.base import Base
from karenina.storage.generated_models import VerificationResultModel  # noqa: F401  (registers table on Base)
from karenina.storage.models import (  # noqa: F401  (registers tables on Base)
    BenchmarkModel,
    BenchmarkQuestionModel,
    ImportMetadataModel,
    QuestionModel,
    VerificationRunModel,
)
from karenina.storage.views.results_metadata import create_results_metadata_view


def _reflect_view_columns() -> set[str]:
    """Create tables + the view in an in-memory SQLite and reflect column names.

    Returns:
        Set of column names present on ``results_metadata_view``.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    create_results_metadata_view(engine)
    return {col["name"] for col in inspect(engine).get_columns("results_metadata_view")}


def _seed_failure_rows(db_url: str) -> None:
    """Insert one verification_results row per FailureCategory via the public API.

    Uses ``save_benchmark`` + ``save_verification_results`` so the rows go
    through the same ORM/converter path the application uses — exercising the
    full serialization chain for ``metadata_failure_category``.
    """
    from datetime import datetime
    from uuid import uuid4

    from karenina import Benchmark
    from karenina.schemas.verification import (
        VerificationResult,
        VerificationResultMetadata,
        VerificationResultTemplate,
    )
    from karenina.schemas.verification.model_identity import ModelIdentity
    from karenina.storage.db_config import DBConfig
    from karenina.storage.engine import close_engine, init_database
    from karenina.storage.operations import save_benchmark, save_verification_results

    config = DBConfig(storage_url=db_url, auto_create=True, auto_commit=True)
    init_database(config)
    try:
        benchmark = Benchmark.create(name="view-test", description="d")
        for i, _cat in enumerate(FailureCategory, start=1):
            benchmark.add_question(question=f"Q{i}?", raw_answer="a", question_id=f"q{i}")
        save_benchmark(benchmark, config)

        answering = ModelIdentity(interface="langchain", model_name="m")
        parsing = ModelIdentity(interface="langchain", model_name="m")
        now = datetime.utcnow().isoformat()

        for i, cat in enumerate(FailureCategory, start=1):
            qid = f"q{i}"
            result_id = VerificationResultMetadata.compute_result_id(
                question_id=qid, answering=answering, parsing=parsing, timestamp=now
            )
            result = VerificationResult(
                metadata=VerificationResultMetadata(
                    question_id=qid,
                    template_id="no_template",
                    failure=Failure(category=cat, stage="generate_answer", reason="seed"),
                    question_text=f"Q{i}?",
                    answering=answering,
                    parsing=parsing,
                    execution_time=0.1,
                    timestamp=now,
                    result_id=result_id,
                ),
                template=VerificationResultTemplate(raw_llm_response=""),
            )
            run_id = uuid4().hex
            save_verification_results(
                results={qid: result},
                db_config=config,
                run_id=run_id,
                benchmark_name="view-test",
                run_name=f"run-{i}",
            )
    finally:
        close_engine(config)


@pytest.mark.unit
class TestResultsMetadataView:
    """Contract tests for the ``results_metadata_view`` column surface."""

    def test_legacy_columns_removed(self) -> None:
        """Legacy error-state columns must no longer appear on the view."""
        cols = _reflect_view_columns()
        for legacy in (
            "metadata_completed_without_errors",
            "metadata_error",
            "metadata_error_category",
            "metadata_failed_stage",
            "completed_without_errors",
            "error",
            "error_category",
            "failed_stage",
        ):
            assert legacy not in cols, f"Legacy column still present: {legacy}"

    def test_failure_columns_present(self) -> None:
        """New failure-taxonomy columns and caveats must be exposed."""
        cols = _reflect_view_columns()
        for added in (
            "metadata_failure_category",
            "metadata_failure_group",
            "metadata_failure_stage",
            "metadata_failure_reason",
            "metadata_caveats",
        ):
            assert added in cols, f"New column missing: {added}"


@pytest.mark.unit
class TestResultsMetadataViewFailureGroupMapping:
    """The view's SQL CASE must match Python's ``CATEGORY_TO_GROUP``.

    ``results_metadata_view`` re-derives ``metadata_failure_group`` from
    ``metadata_failure_category`` via an inline SQL CASE statement. That CASE
    duplicates ``CATEGORY_TO_GROUP``. If a new ``FailureCategory`` is added
    and the dict is updated but the SQL CASE is forgotten, the view returns
    ``NULL`` for that category while Python reports the real group — silent
    drift that breaks downstream aggregations and UI dashboards.

    These tests insert one row per category and assert the view's group column
    matches the Python mapping exactly.
    """

    @pytest.fixture
    def view_db_url(self, tmp_path):
        """Build a file-based SQLite DB with one failure row per category."""
        db_path = tmp_path / "view.db"
        db_url = f"sqlite:///{db_path}"
        _seed_failure_rows(db_url)
        # The view must be created after the tables exist.
        engine = create_engine(db_url)
        try:
            create_results_metadata_view(engine)
        finally:
            engine.dispose()
        return db_url

    def test_every_category_maps_to_expected_group(self, view_db_url) -> None:
        """For each FailureCategory, the view's group must equal CATEGORY_TO_GROUP."""
        engine = create_engine(view_db_url)
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text("SELECT metadata_failure_category, metadata_failure_group FROM results_metadata_view")
                ).fetchall()
        finally:
            engine.dispose()

        seen = {}
        for category_value, group_value in rows:
            seen[category_value] = group_value

        for cat in FailureCategory:
            assert cat.value in seen, (
                f"Category {cat.value!r} has no row in results_metadata_view; the seed fixture may need updating"
            )
            expected = CATEGORY_TO_GROUP[cat].value
            actual = seen[cat.value]
            assert actual == expected, (
                f"Category {cat.value!r}: view returned group={actual!r}, "
                f"but CATEGORY_TO_GROUP says {expected!r}. The SQL CASE in "
                "results_metadata_view is out of sync with the Python dict."
            )

    def test_no_unmapped_category_leaks_null_group(self, view_db_url) -> None:
        """No row in the view should have a NULL metadata_failure_group.

        If the SQL CASE is missing an arm for any category that exists in the
        seed rows, that row will fall through to ``ELSE NULL``. This is a
        stricter assertion than the per-category check above and surfaces any
        future category that is added without updating the CASE.
        """
        engine = create_engine(view_db_url)
        try:
            with engine.connect() as conn:
                nulls = conn.execute(
                    text(
                        "SELECT metadata_failure_category FROM results_metadata_view"
                        " WHERE metadata_failure_group IS NULL"
                    )
                ).fetchall()
        finally:
            engine.dispose()

        assert nulls == [], (
            f"View produced NULL metadata_failure_group for categories: "
            f"{[r[0] for r in nulls]!r}. Add the missing CASE arm to "
            "results_metadata_view."
        )
