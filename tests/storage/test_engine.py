"""Unit tests for database engine and session management."""

import tempfile
from pathlib import Path

import pytest

from karenina.storage.db_config import DBConfig
from karenina.storage.engine import (
    close_engine,
    create_session_factory,
    drop_database,
    get_engine,
    get_session,
    init_database,
    reset_database,
)


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield f"sqlite:///{db_path}"

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_db_config(temp_sqlite_db):
    """Create a temporary DBConfig for testing."""
    return DBConfig(storage_url=temp_sqlite_db)


class TestGetEngine:
    """Test get_engine function."""

    def test_get_engine_sqlite(self, temp_db_config):
        """Test creating a SQLite engine."""
        engine = get_engine(temp_db_config)
        assert engine is not None
        assert str(engine.url).startswith("sqlite:///")

    def test_get_engine_caching(self, temp_db_config):
        """Test that engines are cached per URL."""
        engine1 = get_engine(temp_db_config)
        engine2 = get_engine(temp_db_config)

        # Should return the same cached engine
        assert engine1 is engine2

    def test_get_engine_different_urls(self):
        """Test that different URLs get different engines."""
        config1 = DBConfig(storage_url="sqlite:///:memory:")
        config2 = DBConfig(storage_url="sqlite:///other.db")

        engine1 = get_engine(config1)
        engine2 = get_engine(config2)

        # Should be different engines
        assert engine1 is not engine2

        # Cleanup
        close_engine(config1)
        close_engine(config2)

    def test_get_engine_sqlite_foreign_keys(self, temp_db_config):
        """Test that SQLite has foreign keys enabled."""
        from sqlalchemy import text

        engine = get_engine(temp_db_config)

        # Connect and check pragma
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA foreign_keys")).fetchone()
            assert result[0] == 1  # Foreign keys should be ON


class TestSessionFactory:
    """Test create_session_factory function."""

    def test_create_session_factory(self, temp_db_config):
        """Test creating a session factory."""
        factory = create_session_factory(temp_db_config)
        assert factory is not None

        # Create a session from the factory
        session = factory()
        assert session is not None
        session.close()

    def test_session_factory_creates_valid_session(self, temp_db_config):
        """Test that session factory creates a valid session."""
        factory = create_session_factory(temp_db_config)
        session = factory()

        # Session should be valid and bound to the engine
        assert session.bind is not None
        session.close()


class TestGetSession:
    """Test get_session context manager."""

    def test_get_session_context_manager(self, temp_db_config):
        """Test using get_session as context manager."""
        with get_session(temp_db_config) as session:
            assert session is not None
            # Session should be usable
            assert session.is_active

    def test_get_session_auto_close(self, temp_db_config):
        """Test that session is closed after context exits."""
        session_ref = None
        with get_session(temp_db_config) as session:
            session_ref = session
            assert session.is_active

        # Session should be closed now - trying to use it should fail
        from sqlalchemy.exc import InvalidRequestError

        with pytest.raises((InvalidRequestError, Exception)):
            # Attempting to execute on closed session should fail
            session_ref.execute(session_ref.connection())

    def test_get_session_auto_commit_enabled(self, temp_db_config):
        """Test auto-commit when enabled in config."""
        # Initialize database first
        init_database(temp_db_config)

        from karenina.storage.models import BenchmarkModel

        with get_session(temp_db_config) as session:
            # Add a benchmark
            benchmark = BenchmarkModel(
                name="Test",
                description="Test",
                version="1.0.0",
                metadata_json={},
            )
            session.add(benchmark)
            # Should auto-commit on exit

        # Verify it was committed
        with get_session(temp_db_config) as session:
            from sqlalchemy import select

            result = session.execute(select(BenchmarkModel).where(BenchmarkModel.name == "Test")).scalar_one_or_none()
            assert result is not None
            assert result.name == "Test"

    def test_get_session_auto_commit_disabled(self, temp_sqlite_db):
        """Test no auto-commit when disabled in config."""
        config = DBConfig(storage_url=temp_sqlite_db, auto_commit=False)
        init_database(config)

        from karenina.storage.models import BenchmarkModel

        with get_session(config) as session:
            benchmark = BenchmarkModel(
                name="Test2",
                description="Test",
                version="1.0.0",
                metadata_json={},
            )
            session.add(benchmark)
            # Should NOT auto-commit

        # Verify it was NOT committed (rolled back)
        with get_session(config) as session:
            from sqlalchemy import select

            result = session.execute(select(BenchmarkModel).where(BenchmarkModel.name == "Test2")).scalar_one_or_none()
            # Should be None because transaction was not committed
            assert result is None

    def test_get_session_rollback_on_error(self, temp_db_config):
        """Test that session rolls back on error."""
        init_database(temp_db_config)

        from karenina.storage.models import BenchmarkModel

        with pytest.raises(RuntimeError), get_session(temp_db_config) as session:
            benchmark = BenchmarkModel(
                name="Test",
                description="Test",
                version="1.0.0",
                metadata_json={},
            )
            session.add(benchmark)
            # Raise an error before commit
            raise RuntimeError("Test error")

        # Verify nothing was committed
        with get_session(temp_db_config) as session:
            from sqlalchemy import select

            result = session.execute(select(BenchmarkModel).where(BenchmarkModel.name == "Test")).scalar_one_or_none()
            assert result is None


class TestInitDatabase:
    """Test init_database function."""

    def test_init_database_creates_tables(self, temp_db_config):
        """Test that init_database creates all tables."""
        init_database(temp_db_config)

        engine = get_engine(temp_db_config)

        # Check that tables exist
        from sqlalchemy import inspect

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        expected_tables = [
            "benchmarks",
            "questions",
            "benchmark_questions",
            "verification_runs",
            "verification_results",
        ]

        for table in expected_tables:
            assert table in tables

    def test_init_database_creates_views(self, temp_db_config):
        """Test that init_database creates all views."""
        init_database(temp_db_config)

        engine = get_engine(temp_db_config)

        # Check that views exist
        from sqlalchemy import inspect

        inspector = inspect(engine)
        views = inspector.get_view_names()

        expected_views = [
            "benchmark_summary_view",
            "verification_run_summary_view",
            "question_usage_view",
            "latest_verification_results_view",
            "model_performance_view",
            "rubric_scores_aggregate_view",
            "verification_history_timeline_view",
            "failed_verifications_view",
            "rubric_traits_by_type_view",
        ]

        for view in expected_views:
            assert view in views

    def test_init_database_idempotent(self, temp_db_config):
        """Test that init_database can be called multiple times safely."""
        init_database(temp_db_config)
        # Should not raise an error
        init_database(temp_db_config)


class TestDropDatabase:
    """Test drop_database function."""

    def test_drop_database_removes_tables(self, temp_db_config):
        """Test that drop_database removes all tables."""
        init_database(temp_db_config)
        drop_database(temp_db_config)

        engine = get_engine(temp_db_config)

        from sqlalchemy import inspect

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        # Should have no tables
        assert len(tables) == 0

    def test_drop_database_removes_views(self, temp_db_config):
        """Test that drop_database removes all views."""
        init_database(temp_db_config)
        drop_database(temp_db_config)

        engine = get_engine(temp_db_config)

        from sqlalchemy import inspect

        inspector = inspect(engine)
        views = inspector.get_view_names()

        # Should have no views
        assert len(views) == 0


class TestResetDatabase:
    """Test reset_database function."""

    def test_reset_database_clears_data(self, temp_db_config):
        """Test that reset_database clears all data."""
        init_database(temp_db_config)

        # Add some data
        from karenina.storage.models import BenchmarkModel

        with get_session(temp_db_config) as session:
            benchmark = BenchmarkModel(
                name="Test",
                description="Test",
                version="1.0.0",
                metadata_json={},
            )
            session.add(benchmark)
            session.commit()

        # Reset database
        reset_database(temp_db_config)

        # Verify data is gone
        with get_session(temp_db_config) as session:
            from sqlalchemy import select

            result = session.execute(select(BenchmarkModel)).scalars().all()
            assert len(result) == 0

    def test_reset_database_recreates_schema(self, temp_db_config):
        """Test that reset_database recreates schema."""
        init_database(temp_db_config)
        reset_database(temp_db_config)

        engine = get_engine(temp_db_config)

        from sqlalchemy import inspect

        inspector = inspect(engine)

        # Tables should still exist
        tables = inspector.get_table_names()
        assert "benchmarks" in tables

        # Views should still exist
        views = inspector.get_view_names()
        assert "benchmark_summary_view" in views


class TestCloseEngine:
    """Test close_engine function."""

    def test_close_engine_disposes(self, temp_db_config):
        """Test that close_engine disposes the engine."""
        engine = get_engine(temp_db_config)
        assert engine is not None

        close_engine(temp_db_config)

        # Getting engine again should create a new one
        new_engine = get_engine(temp_db_config)
        assert new_engine is not engine

    def test_close_engine_safe_if_not_exists(self):
        """Test that close_engine is safe to call even if engine doesn't exist."""
        config = DBConfig(storage_url="sqlite:///nonexistent.db")
        # Should not raise an error
        close_engine(config)
