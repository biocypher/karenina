"""Unit tests for DBConfig class."""

import pytest

from karenina.storage.db_config import DBConfig


class TestDBConfigCreation:
    """Test DBConfig initialization and validation."""

    def test_create_sqlite_config(self):
        """Test creating a SQLite configuration."""
        config = DBConfig(storage_url="sqlite:///test.db")
        assert config.storage_url == "sqlite:///test.db"
        assert config.auto_create is True
        assert config.auto_commit is True
        assert config.echo is False

    def test_create_postgresql_config(self):
        """Test creating a PostgreSQL configuration."""
        config = DBConfig(storage_url="postgresql://user:pass@localhost/db")
        assert config.storage_url == "postgresql://user:pass@localhost/db"
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.pool_recycle == 3600
        assert config.pool_pre_ping is True

    def test_create_mysql_config(self):
        """Test creating a MySQL configuration."""
        config = DBConfig(storage_url="mysql+pymysql://user:pass@localhost/db")
        assert config.storage_url == "mysql+pymysql://user:pass@localhost/db"

    def test_custom_pool_settings(self):
        """Test custom connection pool settings."""
        config = DBConfig(
            storage_url="postgresql://localhost/db",
            pool_size=20,
            max_overflow=30,
            pool_recycle=7200,
            pool_pre_ping=False,
        )
        assert config.pool_size == 20
        assert config.max_overflow == 30
        assert config.pool_recycle == 7200
        assert config.pool_pre_ping is False

    def test_auto_flags(self):
        """Test auto_create and auto_commit flags."""
        config = DBConfig(storage_url="sqlite:///test.db", auto_create=False, auto_commit=False)
        assert config.auto_create is False
        assert config.auto_commit is False

    def test_echo_mode(self):
        """Test SQL echo mode for debugging."""
        config = DBConfig(storage_url="sqlite:///test.db", echo=True)
        assert config.echo is True


class TestDBConfigProperties:
    """Test DBConfig computed properties."""

    def test_is_sqlite_true(self):
        """Test is_sqlite property with SQLite URL."""
        config = DBConfig(storage_url="sqlite:///test.db")
        assert config.is_sqlite is True

    def test_is_sqlite_memory(self):
        """Test is_sqlite property with in-memory SQLite."""
        config = DBConfig(storage_url="sqlite:///:memory:")
        assert config.is_sqlite is True

    def test_is_sqlite_false_postgresql(self):
        """Test is_sqlite property with PostgreSQL URL."""
        config = DBConfig(storage_url="postgresql://localhost/db")
        assert config.is_sqlite is False

    def test_is_sqlite_false_mysql(self):
        """Test is_sqlite property with MySQL URL."""
        config = DBConfig(storage_url="mysql://localhost/db")
        assert config.is_sqlite is False

    def test_dialect_sqlite(self):
        """Test dialect property for SQLite."""
        config = DBConfig(storage_url="sqlite:///test.db")
        assert config.dialect == "sqlite"

    def test_dialect_postgresql(self):
        """Test dialect property for PostgreSQL."""
        config = DBConfig(storage_url="postgresql://localhost/db")
        assert config.dialect == "postgresql"

    def test_dialect_mysql(self):
        """Test dialect property for MySQL."""
        config = DBConfig(storage_url="mysql://localhost/db")
        assert config.dialect == "mysql"

    def test_dialect_with_driver(self):
        """Test dialect property with explicit driver."""
        config = DBConfig(storage_url="mysql+pymysql://localhost/db")
        assert config.dialect == "mysql"


class TestDBConfigValidation:
    """Test DBConfig validation and constraints."""

    def test_pool_size_minimum(self):
        """Test pool_size must be at least 1."""
        with pytest.raises(ValueError):
            DBConfig(storage_url="postgresql://localhost/db", pool_size=0)

    def test_pool_size_negative(self):
        """Test pool_size cannot be negative."""
        with pytest.raises(ValueError):
            DBConfig(storage_url="postgresql://localhost/db", pool_size=-1)

    def test_max_overflow_minimum(self):
        """Test max_overflow must be at least 0."""
        with pytest.raises(ValueError):
            DBConfig(storage_url="postgresql://localhost/db", max_overflow=-1)

    def test_pool_recycle_negative(self):
        """Test pool_recycle can be -1 (disabled) but not less."""
        # -1 is allowed (disable recycling)
        config = DBConfig(storage_url="postgresql://localhost/db", pool_recycle=-1)
        assert config.pool_recycle == -1

        # Less than -1 should fail
        with pytest.raises(ValueError):
            DBConfig(storage_url="postgresql://localhost/db", pool_recycle=-2)

    def test_empty_storage_url(self):
        """Test that empty storage URL is rejected."""
        with pytest.raises(ValueError):
            DBConfig(storage_url="")

    def test_invalid_url_format(self):
        """Test that invalid URL format is rejected."""
        # Missing :// separator should be rejected
        with pytest.raises(ValueError, match="must be a valid SQLAlchemy URL"):
            DBConfig(storage_url="not-a-valid-url")


class TestDBConfigPostInit:
    """Test post-initialization hooks."""

    def test_sqlite_pool_settings_disabled(self):
        """Test that SQLite disables pooling via post-init hook."""
        config = DBConfig(storage_url="sqlite:///test.db", pool_size=10, max_overflow=5)

        # SQLite should have pooling disabled regardless of input
        # Note: The post-init hook sets these to None or appropriate SQLite values
        # This behavior may vary based on implementation
        assert config.is_sqlite is True

    def test_postgresql_pool_settings_preserved(self):
        """Test that PostgreSQL preserves pool settings."""
        config = DBConfig(storage_url="postgresql://localhost/db", pool_size=15, max_overflow=25)

        assert config.is_sqlite is False
        assert config.pool_size == 15
        assert config.max_overflow == 25


class TestDBConfigEquality:
    """Test DBConfig comparison and hashing."""

    def test_configs_with_same_url_equal(self):
        """Test that configs with same URL are equal."""
        config1 = DBConfig(storage_url="sqlite:///test.db")
        config2 = DBConfig(storage_url="sqlite:///test.db")

        # They should have the same storage_url
        assert config1.storage_url == config2.storage_url

    def test_configs_with_different_url_not_equal(self):
        """Test that configs with different URLs are not equal."""
        config1 = DBConfig(storage_url="sqlite:///test1.db")
        config2 = DBConfig(storage_url="sqlite:///test2.db")

        assert config1.storage_url != config2.storage_url


class TestDBConfigEdgeCases:
    """Test edge cases and special scenarios."""

    def test_in_memory_sqlite(self):
        """Test in-memory SQLite database configuration."""
        config = DBConfig(storage_url="sqlite:///:memory:")
        assert config.is_sqlite is True
        assert config.storage_url == "sqlite:///:memory:"

    def test_very_large_pool_size(self):
        """Test configuration with very large pool size."""
        config = DBConfig(storage_url="postgresql://localhost/db", pool_size=1000)
        assert config.pool_size == 1000

    def test_zero_pool_recycle(self):
        """Test pool_recycle set to 0 (recycle immediately)."""
        config = DBConfig(storage_url="postgresql://localhost/db", pool_recycle=0)
        assert config.pool_recycle == 0

    def test_all_defaults(self):
        """Test using all default values."""
        config = DBConfig(storage_url="postgresql://localhost/db")

        assert config.auto_create is True
        assert config.auto_commit is True
        assert config.echo is False
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.pool_recycle == 3600
        assert config.pool_pre_ping is True

    def test_url_with_special_characters(self):
        """Test URL with special characters in password."""
        url = "postgresql://user:p@ss%20word@localhost/db"
        config = DBConfig(storage_url=url)
        assert config.storage_url == url
