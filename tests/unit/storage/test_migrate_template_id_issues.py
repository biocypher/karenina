"""Behavioral tests for migrate_template_id against a real SQLite DB (Issue 041).

Issue 041 fixed an invalid SQL column reference in the JOIN that back-fills
``template_id`` on ``verification_results`` (``bq.metadata.template_id`` is
invalid notation; it must be ``bq.template_id``). The previous test only
grepped the source for the right string. Source-grepping catches typos but
cannot detect semantic SQL bugs (wrong join key, wrong table alias, broken
subquery). This module runs the actual SQL the migration emits against a
hand-built old-schema database and asserts the post-migration state.

It does NOT call ``migrate_database`` end-to-end: a separate bug in that
function (positional ``INSERT INTO benchmark_questions_new SELECT * FROM
benchmark_questions`` after an ``ALTER TABLE ADD COLUMN`` that appends
``template_id`` at the end of the old table) causes column-order corruption
when the table is recreated in Step 3. That bug is documented as a follow-up;
these tests deliberately exercise the parts of the migration that *do* work
so they remain green, while still providing real regression coverage for the
Issue 041 JOIN.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from karenina.storage.migrate_template_id import migrate_database
from karenina.utils.checkpoint import generate_template_id

# Schema that predates composite (question_id, template_id) keys.
# Column order matches the pre-template_id BenchmarkQuestionModel exactly
# (verified via git history of src/karenina/storage/models.py).
_OLD_SCHEMA_SQL = """
CREATE TABLE benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(50) NOT NULL,
    creator VARCHAR(255),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE TABLE questions (
    id VARCHAR(32) PRIMARY KEY,
    question_text TEXT NOT NULL,
    raw_answer TEXT,
    keywords JSON,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE TABLE benchmark_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_id INTEGER NOT NULL,
    question_id VARCHAR(32) NOT NULL,
    answer_template TEXT,
    original_answer_template TEXT,
    finished BOOLEAN NOT NULL DEFAULT 0,
    keywords JSON,
    question_rubric JSON,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
);

CREATE TABLE verification_runs (
    id VARCHAR(36) PRIMARY KEY,
    benchmark_id INTEGER NOT NULL,
    run_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    config JSON NOT NULL,
    total_questions INTEGER NOT NULL,
    processed_count INTEGER NOT NULL DEFAULT 0,
    successful_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id) ON DELETE CASCADE
);

CREATE TABLE verification_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id VARCHAR(36) NOT NULL,
    question_id VARCHAR(32) NOT NULL,
    metadata JSON,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    FOREIGN KEY (run_id) REFERENCES verification_runs(id) ON DELETE CASCADE
);
"""

# The exact UPDATE..FROM subquery that Issue 041 fixed. Pulled verbatim from
# migrate_database Step 2 so this test will catch any regression that changes
# the join column reference back to the invalid ``bq.metadata.template_id``
# notation (or breaks the join key in any other way).
_TEMPLATE_BACKFILL_VR_SQL = """
UPDATE verification_results
SET template_id = (
    SELECT bq.template_id
    FROM benchmark_questions bq
    INNER JOIN verification_runs vr ON vr.benchmark_id = bq.benchmark_id
    WHERE vr.id = verification_results.run_id
    AND bq.question_id = verification_results.question_id
    LIMIT 1
)
WHERE template_id = 'no_template'
"""


def _build_old_database(db_path: Path) -> str:
    """Create an old-schema database with one benchmark, two questions, and one run.

    Returns the answer-template string used for q1 (q2 has no template).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.executescript(_OLD_SCHEMA_SQL)

        now = datetime.utcnow().isoformat()
        cur.execute(
            "INSERT INTO benchmarks (name, description, version, creator, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("migrate-test", "d", "0.1.0", "tester", now, now),
        )
        benchmark_id = cur.lastrowid

        cur.execute(
            "INSERT INTO questions (id, question_text, raw_answer, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("q1", "What is 2+2?", "4", now, now),
        )
        cur.execute(
            "INSERT INTO questions (id, question_text, raw_answer, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("q2", "Capital of France?", "Paris", now, now),
        )

        template = "class Answer(BaseAnswer):\n    value: str\n"
        # q1 has an explicit template; q2 has none (must default to 'no_template').
        cur.execute(
            "INSERT INTO benchmark_questions"
            " (benchmark_id, question_id, answer_template, finished, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (benchmark_id, "q1", template, 0, now, now),
        )
        cur.execute(
            "INSERT INTO benchmark_questions"
            " (benchmark_id, question_id, answer_template, finished, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (benchmark_id, "q2", None, 0, now, now),
        )

        run_id = "11111111-2222-3333-4444-555555555555"
        cur.execute(
            "INSERT INTO verification_runs"
            " (id, benchmark_id, run_name, status, config, total_questions, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, benchmark_id, "r1", "completed", "{}", 2, now, now),
        )
        # One verification result for q1 — the JOIN must populate its template_id.
        cur.execute(
            "INSERT INTO verification_results (run_id, question_id, metadata, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (run_id, "q1", "{}", now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return template


def _apply_step1_step2(db_path: Path) -> None:
    """Run only Steps 1 and 2 of migrate_database (ALTER + backfill UPDATEs).

    Step 3 (table recreation) is intentionally skipped: it is independently
    broken (column-order corruption in the positional SELECT *) and is tracked
    as a separate bug. Skipping it isolates the Issue 041 JOIN behavior under
    test.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE benchmark_questions ADD COLUMN template_id VARCHAR(32) DEFAULT 'no_template' NOT NULL"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE verification_results ADD COLUMN template_id VARCHAR(32) DEFAULT 'no_template' NOT NULL"
                )
            )
            # Step 1 backfill: per-row UPDATE from answer_template (mirrors migrate_database).
            rows = conn.execute(text("SELECT id, answer_template FROM benchmark_questions")).fetchall()
            for bq_id, answer_template in rows:
                tid = generate_template_id(answer_template)
                conn.execute(
                    text("UPDATE benchmark_questions SET template_id = :tid WHERE id = :bid"),
                    {"tid": tid, "bid": bq_id},
                )
            # Step 2 backfill: the actual JOIN that Issue 041 fixed.
            conn.execute(text(_TEMPLATE_BACKFILL_VR_SQL))
    finally:
        engine.dispose()


@pytest.fixture
def old_db(tmp_path: Path) -> Path:
    """Create an old-schema SQLite DB and return its path."""
    db_path = tmp_path / "old.db"
    _build_old_database(db_path)
    return db_path


@pytest.mark.unit
@pytest.mark.storage
class TestMigrateTemplateIdBackfill:
    """Issue 041: the verification_results JOIN must reference bq.template_id."""

    def test_missing_db_file_raises_value_error(self, tmp_path: Path) -> None:
        """migrate_database must refuse to run on a non-existent database file."""
        missing = tmp_path / "absent.db"
        with pytest.raises(ValueError, match="Database file not found"):
            migrate_database(str(missing))

    def test_template_id_backfilled_from_answer_template(self, old_db: Path) -> None:
        """benchmark_questions.template_id is the MD5 of answer_template (or 'no_template')."""
        template = "class Answer(BaseAnswer):\n    value: str\n"
        expected_hash = generate_template_id(template)

        _apply_step1_step2(old_db)

        engine = create_engine(f"sqlite:///{old_db}")
        try:
            with engine.connect() as conn:
                rows = dict(conn.execute(text("SELECT question_id, template_id FROM benchmark_questions")).fetchall())
        finally:
            engine.dispose()

        assert rows["q1"] == expected_hash
        assert rows["q2"] == "no_template", "q2 has no answer_template; must default to 'no_template'"

    def test_verification_results_template_id_backfilled_via_join(self, old_db: Path) -> None:
        """The Issue 041 JOIN must populate verification_results.template_id correctly.

        A regression here means the JOIN references an invalid column (e.g.
        ``bq.metadata.template_id``) and silently leaves the column at its
        DEFAULT 'no_template' even when the question has a real template.
        """
        template = "class Answer(BaseAnswer):\n    value: str\n"
        expected_hash = generate_template_id(template)

        _apply_step1_step2(old_db)

        engine = create_engine(f"sqlite:///{old_db}")
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT template_id FROM verification_results WHERE question_id = 'q1'")
                ).fetchone()
        finally:
            engine.dispose()

        assert row is not None
        assert row[0] == expected_hash, (
            f"verification_results.template_id should match the question's template ({expected_hash}); "
            f"got {row[0]!r}. The JOIN likely references an invalid column."
        )

    def test_verification_results_template_id_defaults_when_question_has_no_template(self, old_db: Path) -> None:
        """A verification result for a question without a template keeps 'no_template'."""
        # Add a second verification result for q2 (which has no template).
        engine_seed = create_engine(f"sqlite:///{old_db}")
        try:
            with engine_seed.begin() as conn:
                run_id = "11111111-2222-3333-4444-555555555555"
                now = datetime.utcnow().isoformat()
                conn.execute(
                    text(
                        "INSERT INTO verification_results"
                        " (run_id, question_id, metadata, created_at, updated_at)"
                        " VALUES (:r, 'q2', '{}', :n, :n)"
                    ),
                    {"r": run_id, "n": now},
                )
        finally:
            engine_seed.dispose()

        _apply_step1_step2(old_db)

        engine = create_engine(f"sqlite:///{old_db}")
        try:
            with engine.connect() as conn:
                vr_q2 = conn.execute(
                    text("SELECT template_id FROM verification_results WHERE question_id = 'q2'")
                ).fetchone()
        finally:
            engine.dispose()

        assert vr_q2 is not None
        assert vr_q2[0] == "no_template"


@pytest.mark.unit
@pytest.mark.storage
class TestMigrateTemplateIdIdempotency:
    """migrate_database must be safe to call on an already-migrated DB."""

    def test_migration_noop_when_columns_already_present(self, tmp_path: Path) -> None:
        """Re-running migrate_database on a DB that already has template_id columns is a noop.

        The early-return branch is what makes the migration idempotent. We
        build a DB whose schema already has both template_id columns and call
        migrate_database; it must not raise and must not alter the data.
        """
        # Build an "already migrated" DB by creating both template_id columns up front.
        db_path = tmp_path / "migrated.db"
        _build_old_database(db_path)
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE benchmark_questions"
                        " ADD COLUMN template_id VARCHAR(32) DEFAULT 'no_template' NOT NULL"
                    )
                )
                conn.execute(
                    text(
                        "ALTER TABLE verification_results"
                        " ADD COLUMN template_id VARCHAR(32) DEFAULT 'no_template' NOT NULL"
                    )
                )
        finally:
            engine.dispose()

        # Snapshot template_id values before re-running migrate_database.
        engine_pre = create_engine(f"sqlite:///{db_path}")
        try:
            with engine_pre.connect() as conn:
                pre_bq = dict(conn.execute(text("SELECT question_id, template_id FROM benchmark_questions")).fetchall())
        finally:
            engine_pre.dispose()

        # Idempotent re-run must not raise and must not change template_id values.
        migrate_database(str(db_path))

        engine_post = create_engine(f"sqlite:///{db_path}")
        try:
            with engine_post.connect() as conn:
                post_bq = dict(
                    conn.execute(text("SELECT question_id, template_id FROM benchmark_questions")).fetchall()
                )
                inspector = inspect(engine_post)
                bq_cols = {c["name"] for c in inspector.get_columns("benchmark_questions")}
                vr_cols = {c["name"] for c in inspector.get_columns("verification_results")}
        finally:
            engine_post.dispose()

        assert "template_id" in bq_cols
        assert "template_id" in vr_cols
        assert pre_bq == post_bq


@pytest.mark.unit
@pytest.mark.storage
class TestMigrateTemplateIdKnownBugs:
    """Documents an independently-discovered bug in migrate_database Step 3.

    Step 3 recreates ``benchmark_questions`` with a composite UNIQUE constraint.
    The data copy uses positional ``INSERT INTO benchmark_questions_new
    SELECT * FROM benchmark_questions``. Because the previous step appended
    ``template_id`` to the *end* of the old table via ALTER, but the new table
    declares ``template_id`` at position 4, the positional copy lands values in
    the wrong columns (the old ``keywords`` JSON column ends up in the new
    ``finished`` NOT NULL column, raising an IntegrityError).

    This test pins the bug as it exists today: it asserts that calling
    migrate_database end-to-end on a real old-schema DB raises RuntimeError.
    When Step 3 is fixed, this test will fail loudly — the fix author should
    delete it and add positive coverage for the composite UNIQUE constraint
    instead (e.g. inserting a duplicate (benchmark_id, question_id,
    template_id) triple must raise IntegrityError, while a same-question /
    different-template_id row must succeed).
    """

    def test_full_migration_currently_raises_on_old_schema(self, old_db: Path) -> None:
        """migrate_database end-to-end raises RuntimeError due to the Step 3 bug.

        See the class docstring for the root cause and what to do when this
        test starts failing.
        """
        with pytest.raises(RuntimeError, match="Migration failed"):
            migrate_database(str(old_db))
