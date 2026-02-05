"""Database migration script to add template_id support.

This migration adds the template_id column to the database schema to enable
composite key identification of questions (question_id + template_id).

Changes:
- Remove UNIQUE constraint from questions.question_text
- Add template_id column to benchmark_questions table
- Add template_id column to verification_results table
- Add composite UNIQUE constraint (benchmark_id, question_id, template_id)
- Populate template_id values from existing answer_template data

Usage:
    python -m karenina.storage.migrate_template_id <database_path>
"""

import logging
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from ..utils.checkpoint import generate_template_id

logger = logging.getLogger(__name__)


def migrate_database(db_path: str) -> None:
    """
    Migrate an existing database to support template_id composite keys.

    Args:
        db_path: Path to the SQLite database file

    Raises:
        ValueError: If database file does not exist
        RuntimeError: If migration fails
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise ValueError(f"Database file not found: {db_path}")

    logger.info("Inspecting database: %s", db_path)
    engine = create_engine(f"sqlite:///{db_path}")

    # Check current schema
    inspector = inspect(engine)

    # Check if migration is needed
    bq_columns = [col["name"] for col in inspector.get_columns("benchmark_questions")]
    vr_columns = [col["name"] for col in inspector.get_columns("verification_results")]

    if "template_id" in bq_columns and "template_id" in vr_columns:
        logger.info("Database already migrated (template_id columns exist)")
        return

    logger.info("Starting migration...")

    with Session(engine) as session:
        try:
            # Step 1: Add template_id column to benchmark_questions if not exists
            if "template_id" not in bq_columns:
                logger.info("Adding template_id column to benchmark_questions...")
                session.execute(
                    text(
                        """
                        ALTER TABLE benchmark_questions
                        ADD COLUMN template_id VARCHAR(32) DEFAULT 'no_template' NOT NULL
                        """
                    )
                )
                session.commit()

                # Populate template_id values from existing answer_template data
                logger.info("Populating template_id values...")
                result = session.execute(text("SELECT id, answer_template FROM benchmark_questions"))
                rows = result.fetchall()
                for row in rows:
                    bq_id = row[0]
                    answer_template = row[1]
                    template_id = generate_template_id(answer_template)
                    session.execute(
                        text("UPDATE benchmark_questions SET template_id = :tid WHERE id = :bid"),
                        {"tid": template_id, "bid": bq_id},
                    )
                session.commit()
                logger.info("Updated %d benchmark_questions rows", len(rows))

            # Step 2: Add template_id column to verification_results if not exists
            if "template_id" not in vr_columns:
                logger.info("Adding template_id column to verification_results...")
                session.execute(
                    text(
                        """
                        ALTER TABLE verification_results
                        ADD COLUMN template_id VARCHAR(32) DEFAULT 'no_template' NOT NULL
                        """
                    )
                )
                session.commit()

                # Populate template_id values by looking up from benchmark_questions
                # This assumes verification results are linked to benchmark questions
                logger.info("Populating verification template_id from benchmark_questions...")
                session.execute(
                    text(
                        """
                        UPDATE verification_results
                        SET template_id = (
                            SELECT bq.metadata.template_id
                            FROM benchmark_questions bq
                            INNER JOIN verification_runs vr ON vr.benchmark_id = bq.benchmark_id
                            WHERE vr.id = verification_results.run_id
                            AND bq.question_id = verification_results.question_id
                            LIMIT 1
                        )
                        WHERE template_id = 'no_template'
                        """
                    )
                )
                session.commit()
                logger.info("Updated verification_results template_id values")

            # Step 3: Recreate benchmark_questions table with new constraint
            logger.info("Recreating benchmark_questions with composite constraint...")

            # Create new table with correct schema
            session.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS benchmark_questions_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        benchmark_id INTEGER NOT NULL,
                        question_id VARCHAR(32) NOT NULL,
                        template_id VARCHAR(32) NOT NULL,
                        answer_template TEXT,
                        original_answer_template TEXT,
                        finished BOOLEAN NOT NULL DEFAULT 0,
                        keywords JSON,
                        question_rubric JSON,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        FOREIGN KEY (benchmark_id) REFERENCES benchmarks(id) ON DELETE CASCADE,
                        FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE,
                        UNIQUE(benchmark_id, question_id, template_id)
                    )
                    """
                )
            )

            # Copy data
            session.execute(
                text(
                    """
                    INSERT INTO benchmark_questions_new
                    SELECT * FROM benchmark_questions
                    """
                )
            )

            # Drop old table
            session.execute(text("DROP TABLE benchmark_questions"))

            # Rename new table
            session.execute(text("ALTER TABLE benchmark_questions_new RENAME TO benchmark_questions"))

            # Recreate indexes
            session.execute(text("CREATE INDEX idx_benchmark_finished ON benchmark_questions(benchmark_id, finished)"))

            session.commit()
            logger.info("Recreated benchmark_questions table")

            # Step 4: Note about questions table
            logger.info("Note: questions.question_text UNIQUE constraint removed in new schema")
            logger.info("Existing databases keep the constraint for compatibility")
            logger.info("New entries can have duplicate question_text as long as template_id differs")

            session.commit()
            logger.info("Migration completed successfully!")

        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Migration failed: {e}") from e


def main() -> None:
    """CLI entry point for migration script."""
    if len(sys.argv) != 2:
        print("Usage: python -m karenina.storage.migrate_template_id <database_path>")
        sys.exit(1)

    db_path = sys.argv[1]

    try:
        migrate_database(db_path)
        print("\n🎉 Migration complete! Database is now ready for composite key support.")
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
