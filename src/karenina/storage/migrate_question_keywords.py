"""Migration script to rename tags->keywords and add new columns on QuestionModel.

Usage:
    python -m karenina.storage.migrate_question_keywords <database_url>

This script:
1. Renames the ``tags`` column to ``keywords`` in the questions table
2. Adds ``author``, ``sources``, and ``custom_metadata`` JSON columns
"""

import logging
import sys

from sqlalchemy import create_engine, inspect, text

logger = logging.getLogger(__name__)


def migrate(database_url: str) -> None:
    """Run the tags->keywords migration on the given database."""
    engine = create_engine(database_url)
    inspector = inspect(engine)

    columns = [col["name"] for col in inspector.get_columns("questions")]

    with engine.begin() as conn:
        # Rename tags -> keywords (if tags column still exists)
        if "tags" in columns and "keywords" not in columns:
            conn.execute(text("ALTER TABLE questions RENAME COLUMN tags TO keywords"))
            logger.info("Renamed questions.tags -> questions.keywords")
        elif "keywords" in columns:
            logger.info("Column questions.keywords already exists, skipping rename")

        # Add new columns if they don't exist
        for col_name in ("author", "sources", "custom_metadata"):
            if col_name not in columns:
                conn.execute(text(f"ALTER TABLE questions ADD COLUMN {col_name} JSON"))
                logger.info("Added questions.%s column", col_name)
            else:
                logger.info("Column questions.%s already exists, skipping", col_name)

    logger.info("Migration complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python -m karenina.storage.migrate_question_keywords <database_url>")
        sys.exit(1)
    migrate(sys.argv[1])
