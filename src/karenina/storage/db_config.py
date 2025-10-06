"""Database configuration for Karenina storage.

This module defines the DBConfig class for managing database connections
and configuration settings.
"""

from pydantic import BaseModel, Field, field_validator


class DBConfig(BaseModel):
    """Configuration for database connections and operations.

    This class provides settings for connecting to various database management
    systems (SQLite, PostgreSQL, MySQL, etc.) via SQLAlchemy.

    Attributes:
        storage_url: SQLAlchemy database URL (e.g., "sqlite:///example.db",
            "postgresql://user:pass@localhost/dbname")
        auto_create: If True, automatically create database tables and views
            if they don't exist
        auto_commit: If True, automatically commit transactions after operations
        echo: If True, log all SQL statements (useful for debugging)
        pool_size: Number of database connections to keep in the pool (for non-SQLite)
        max_overflow: Maximum number of connections that can be created beyond pool_size
        pool_recycle: Recycle connections after this many seconds (-1 = disabled)
        pool_pre_ping: If True, test connections before using them from the pool
    """

    storage_url: str = Field(
        description="SQLAlchemy database URL (e.g., 'sqlite:///example.db')",
        min_length=1,
    )
    auto_create: bool = Field(
        default=True,
        description="Automatically create database tables and views if they don't exist",
    )
    auto_commit: bool = Field(
        default=True,
        description="Automatically commit transactions after operations",
    )
    echo: bool = Field(
        default=False,
        description="Log all SQL statements for debugging",
    )
    pool_size: int = Field(
        default=5,
        ge=1,
        description="Number of database connections to keep in the pool",
    )
    max_overflow: int = Field(
        default=10,
        ge=0,
        description="Maximum overflow connections beyond pool_size",
    )
    pool_recycle: int = Field(
        default=3600,
        ge=-1,
        description="Recycle connections after this many seconds (-1 = disabled)",
    )
    pool_pre_ping: bool = Field(
        default=True,
        description="Test connections before using them from the pool",
    )

    @field_validator("storage_url")
    @classmethod
    def validate_storage_url(cls, v: str) -> str:
        """Validate that storage URL follows SQLAlchemy format.

        Args:
            v: The storage URL to validate

        Returns:
            The validated storage URL

        Raises:
            ValueError: If URL format is invalid
        """
        if not v or "://" not in v:
            raise ValueError(
                "storage_url must be a valid SQLAlchemy URL "
                "(e.g., 'sqlite:///example.db', 'postgresql://user:pass@host/db')"
            )
        return v

    @property
    def is_sqlite(self) -> bool:
        """Check if this is a SQLite database.

        Returns:
            True if using SQLite, False otherwise
        """
        return self.storage_url.startswith("sqlite:")

    @property
    def dialect(self) -> str:
        """Get the database dialect (sqlite, postgresql, mysql, etc.).

        Returns:
            The database dialect name
        """
        return self.storage_url.split("://")[0].split("+")[0]

    def model_post_init(self, __context: object) -> None:
        """Post-initialization hook to adjust settings for SQLite.

        SQLite doesn't support connection pooling in the same way as other databases,
        so we disable pooling-related features for SQLite.
        """
        if self.is_sqlite:
            # SQLite doesn't benefit from connection pooling
            # Set pool_size to 1 and disable pooling features
            object.__setattr__(self, "pool_size", 1)
            object.__setattr__(self, "max_overflow", 0)

    class Config:
        """Pydantic configuration."""

        frozen = False  # Allow modifications in model_post_init
