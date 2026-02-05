"""View helper utilities.

Provides shared functions for creating and dropping database views with
SQLite/PostgreSQL dialect handling, eliminating boilerplate across view modules.
"""

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

__all__ = ["create_view_safe", "drop_view_safe"]


def create_view_safe(
    engine: Engine,
    view_name: str,
    sqlite_sql: str,
    postgres_sql: str | None = None,
) -> None:
    """Create or replace a database view with dialect-specific SQL.

    Handles the differences between SQLite and PostgreSQL for view creation:
    - SQLite: DROP VIEW IF EXISTS + CREATE VIEW (no CREATE OR REPLACE)
    - PostgreSQL: DROP VIEW IF EXISTS + CREATE VIEW (for consistency)

    Args:
        engine: SQLAlchemy engine instance
        view_name: Name of the view to create
        sqlite_sql: SQL SELECT statement for SQLite (or universal SQL if postgres_sql is None)
        postgres_sql: Optional SQL SELECT statement for PostgreSQL. If None, sqlite_sql
            is used for both dialects.
    """
    sql = sqlite_sql if engine.dialect.name == "sqlite" else (postgres_sql or sqlite_sql)

    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {view_name}"))
        conn.execute(text(f"CREATE VIEW {view_name} AS {sql}"))
    logger.debug("Created view %s (%s dialect)", view_name, engine.dialect.name)


def drop_view_safe(engine: Engine, view_name: str) -> None:
    """Drop a database view if it exists.

    Works identically for SQLite and PostgreSQL.

    Args:
        engine: SQLAlchemy engine instance
        view_name: Name of the view to drop
    """
    with engine.begin() as conn:
        conn.execute(text(f"DROP VIEW IF EXISTS {view_name}"))
    logger.debug("Dropped view %s", view_name)
