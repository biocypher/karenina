"""SQLAlchemy declarative base for Karenina models.

This is separated to avoid circular imports between models.py and generated_models.py.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass
