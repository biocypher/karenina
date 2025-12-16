"""Database engine and session management for Karenina storage.

This module provides functions for creating and managing SQLAlchemy engines
and sessions, including automatic database and table creation.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .base import Base
from .db_config import DBConfig
from .views import create_all_views

# Global cache for engines (one per storage URL)
_engine_cache: dict[str, Engine] = {}


def get_engine(db_config: DBConfig) -> Engine:
    """Get or create a SQLAlchemy engine for the given configuration.

    Engines are cached per storage URL to avoid creating duplicate connections.

    Args:
        db_config: Database configuration

    Returns:
        SQLAlchemy engine instance
    """
    # Check cache first
    if db_config.storage_url in _engine_cache:
        return _engine_cache[db_config.storage_url]

    # Create engine with appropriate settings
    if db_config.is_sqlite:
        # SQLite-specific settings
        engine = create_engine(
            db_config.storage_url,
            echo=db_config.echo,
            # SQLite doesn't support server-side connections
            pool_pre_ping=False,
        )

        # Enable foreign keys for SQLite (disabled by default)
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn: Any, _connection_record: Any) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    else:
        # PostgreSQL, MySQL, etc.
        engine = create_engine(
            db_config.storage_url,
            echo=db_config.echo,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_recycle=db_config.pool_recycle,
            pool_pre_ping=db_config.pool_pre_ping,
        )

    # Cache the engine
    _engine_cache[db_config.storage_url] = engine

    return engine


def create_session_factory(db_config: DBConfig) -> sessionmaker[Session]:
    """Create a session factory for the given configuration.

    Args:
        db_config: Database configuration

    Returns:
        SQLAlchemy sessionmaker instance
    """
    engine = get_engine(db_config)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_session(db_config: DBConfig) -> Iterator[Session]:
    """Get a database session as a context manager.

    This is the recommended way to get a session for database operations.
    The session will be automatically committed (if auto_commit is True) and
    closed when the context exits.

    Args:
        db_config: Database configuration

    Yields:
        SQLAlchemy session instance

    Example:
        ```python
        with get_session(db_config) as session:
            result = session.query(BenchmarkModel).all()
        ```
    """
    session_factory = create_session_factory(db_config)
    session = session_factory()

    try:
        yield session
        if db_config.auto_commit:
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database(db_config: DBConfig) -> None:
    """Initialize the database by creating all tables and views.

    This should be called before any database operations if auto_create is True.
    It's safe to call multiple times (idempotent).

    Args:
        db_config: Database configuration
    """
    engine = get_engine(db_config)

    # Create all tables
    Base.metadata.create_all(engine)

    # Create all views
    create_all_views(engine)


def drop_database(db_config: DBConfig) -> None:
    """Drop all tables and views from the database.

    WARNING: This is a destructive operation! All data will be lost.

    Args:
        db_config: Database configuration
    """
    from .views import drop_all_views

    engine = get_engine(db_config)

    # Drop views first (they depend on tables)
    drop_all_views(engine)

    # Drop all tables
    Base.metadata.drop_all(engine)


def close_engine(db_config: DBConfig) -> None:
    """Close and dispose of the engine for the given configuration.

    This should be called when done with a database connection, especially
    in testing scenarios or when switching databases.

    Args:
        db_config: Database configuration
    """
    if db_config.storage_url in _engine_cache:
        engine = _engine_cache[db_config.storage_url]
        engine.dispose()
        del _engine_cache[db_config.storage_url]


def reset_database(db_config: DBConfig) -> None:
    """Reset the database by dropping and recreating all tables and views.

    WARNING: This is a destructive operation! All data will be lost.

    Args:
        db_config: Database configuration
    """
    drop_database(db_config)
    init_database(db_config)
