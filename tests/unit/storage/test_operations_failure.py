"""Tests for storage.operations success/failure counting via Failure objects.

Verifies that ``save_verification_results`` counts runs as successful when
``metadata.failure is None`` and as failed when a ``Failure`` is attached,
with no dependency on the removed legacy ``completed_without_errors`` field.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

from karenina import Benchmark
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.storage.db_config import DBConfig
from karenina.storage.engine import close_engine, init_database
from karenina.storage.models import VerificationRunModel
from karenina.storage.operations import save_benchmark, save_verification_results


def _make_result(question_id: str, failure: Failure | None) -> VerificationResult:
    """Construct a VerificationResult carrying only a Failure (no legacy fields)."""
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
            template_id="no_template",
            failure=failure,
            question_text=f"Question {question_id}",
            answering=answering,
            parsing=parsing,
            execution_time=0.5,
            timestamp=timestamp,
            result_id=result_id,
        ),
        template=VerificationResultTemplate(
            raw_llm_response="",
        ),
    )


@pytest.fixture
def sqlite_db(tmp_path):
    """Yield a fresh DBConfig backed by a file-based SQLite database."""
    db_path = tmp_path / "operations_failure.db"
    config = DBConfig(storage_url=f"sqlite:///{db_path}", auto_create=True, auto_commit=True)
    init_database(config)
    try:
        yield config
    finally:
        close_engine(config)


@pytest.mark.unit
class TestSaveVerificationResultsFailureCounts:
    """save_verification_results must compute counts from ``metadata.failure``."""

    def test_pass_counted_as_successful_when_failure_none(self, sqlite_db: DBConfig) -> None:
        """Results with ``failure=None`` increment ``successful_count``."""
        benchmark = Benchmark.create(name="ops_pass_test", description="d")
        benchmark.add_question(question="q?", raw_answer="a", question_id="q1")
        save_benchmark(benchmark, sqlite_db)

        run_id = uuid4().hex
        results = {
            "q1": _make_result("q1", failure=None),
        }
        save_verification_results(
            results=results,
            db_config=sqlite_db,
            run_id=run_id,
            benchmark_name="ops_pass_test",
            run_name="pass_run",
        )

        from sqlalchemy import select

        from karenina.storage.engine import get_session

        with get_session(sqlite_db) as session:
            row = session.execute(select(VerificationRunModel).where(VerificationRunModel.id == run_id)).scalar_one()
            assert row.successful_count == 1
            assert row.failed_count == 0

    def test_failure_counted_as_failed(self, sqlite_db: DBConfig) -> None:
        """A timeout failure increments ``failed_count`` rather than ``successful_count``."""
        benchmark = Benchmark.create(name="ops_fail_test", description="d")
        benchmark.add_question(question="q?", raw_answer="a", question_id="q1")
        benchmark.add_question(question="q?2", raw_answer="a", question_id="q2")
        save_benchmark(benchmark, sqlite_db)

        run_id = uuid4().hex
        results = {
            "q1": _make_result("q1", failure=None),
            "q2": _make_result(
                "q2",
                failure=Failure(category=FailureCategory.TIMEOUT, stage="generate_answer", reason="timeout"),
            ),
        }
        save_verification_results(
            results=results,
            db_config=sqlite_db,
            run_id=run_id,
            benchmark_name="ops_fail_test",
            run_name="fail_run",
        )

        from sqlalchemy import select

        from karenina.storage.engine import get_session

        with get_session(sqlite_db) as session:
            row = session.execute(select(VerificationRunModel).where(VerificationRunModel.id == run_id)).scalar_one()
            assert row.successful_count == 1
            assert row.failed_count == 1
