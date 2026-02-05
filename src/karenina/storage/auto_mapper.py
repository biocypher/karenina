"""Auto-mapper for generating SQLAlchemy models from Pydantic models.

This module provides utilities to automatically generate SQLAlchemy ORM models
from Pydantic BaseModel classes, with support for:
- Flattening nested models into prefixed columns
- Automatic type mapping (str→Text, int→Integer, etc.)
- Index hints via Pydantic Field metadata
- Handling Optional/Union types
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from sqlalchemy import JSON, Boolean, Column, Float, Index, Integer, String, Text

from .utils import is_pydantic_model as _is_pydantic_model
from .utils import unwrap_optional as _unwrap_optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase


# Type mapping from Python/Pydantic types to SQLAlchemy column types
TYPE_MAP: dict[type, Any] = {
    str: Text,
    int: Integer,
    float: Float,
    bool: Boolean,
    dict: JSON,
    list: JSON,
    # datetime types handled specially
}


def _get_field_metadata(field_info: FieldInfo | None) -> dict[str, Any]:
    """Extract custom metadata from Pydantic Field's json_schema_extra."""
    if field_info is None:
        return {}
    extra = field_info.json_schema_extra
    if callable(extra):
        # If it's a callable, we can't extract metadata
        return {}
    return extra or {}


def _get_sqlalchemy_type(python_type: type, metadata: dict[str, Any]) -> Any:
    """Map Python type to SQLAlchemy column type.

    Args:
        python_type: The Python type to map
        metadata: Field metadata that may contain type hints (e.g., max_length)

    Returns:
        SQLAlchemy column type
    """
    # Check for max_length hint for string fields
    max_length = metadata.get("max_length")
    if python_type is str and max_length:
        return String(max_length)

    # Use type map or default to JSON for complex types
    return TYPE_MAP.get(python_type, JSON)


class ColumnSpec:
    """Specification for a generated SQLAlchemy column."""

    def __init__(
        self,
        name: str,
        sa_type: Any,
        nullable: bool = True,
        index: bool = False,
        default: Any = None,
        comment: str | None = None,
    ):
        self.name = name
        self.sa_type = sa_type
        self.nullable = nullable
        self.index = index
        self.default = default
        self.comment = comment

    def to_column(self) -> Column[Any]:
        """Convert to SQLAlchemy Column."""
        kwargs: dict[str, Any] = {
            "nullable": self.nullable,
        }
        if self.index:
            kwargs["index"] = True
        if self.default is not None:
            kwargs["default"] = self.default
        if self.comment:
            kwargs["comment"] = self.comment

        return Column(self.sa_type, **kwargs)

    def __repr__(self) -> str:
        return f"ColumnSpec({self.name}, {self.sa_type}, nullable={self.nullable}, index={self.index})"


class PydanticSQLAlchemyMapper:
    """Auto-generates SQLAlchemy column specifications from Pydantic models.

    This mapper introspects Pydantic BaseModel classes and generates
    corresponding SQLAlchemy column definitions with support for:
    - Recursive flattening of nested models
    - Automatic type mapping
    - Index hints from Field metadata
    - Nullable handling for Optional fields

    Example:
        >>> mapper = PydanticSQLAlchemyMapper()
        >>> columns = mapper.generate_columns(VerificationResult, flatten_config={
        ...     "metadata": {"prefix": "metadata_"},
        ...     "template": {"prefix": "template_", "optional": True},
        ... })
    """

    def __init__(self, type_map: dict[type, Any] | None = None):
        """Initialize the mapper.

        Args:
            type_map: Optional custom type mapping to extend/override defaults
        """
        self.type_map = {**TYPE_MAP}
        if type_map:
            self.type_map.update(type_map)

    def generate_columns(
        self,
        model: type[BaseModel],
        prefix: str = "",
        flatten_nested: bool = True,
        parent_optional: bool = False,
    ) -> dict[str, ColumnSpec]:
        """Generate SQLAlchemy column specifications from Pydantic model fields.

        Args:
            model: The Pydantic model class to introspect
            prefix: Prefix to add to all column names (e.g., "metadata_")
            flatten_nested: Whether to recursively flatten nested models
            parent_optional: Whether the parent field was optional (affects nullability)

        Returns:
            Dictionary mapping column names to ColumnSpec objects
        """
        columns: dict[str, ColumnSpec] = {}

        # Get type hints and field info
        try:
            hints = get_type_hints(model)
        except Exception:
            logger.debug("get_type_hints failed for %s, falling back to model_fields", model, exc_info=True)
            hints = {name: field.annotation for name, field in model.model_fields.items() if field.annotation}

        for field_name, field_type in hints.items():
            field_info = model.model_fields.get(field_name)
            metadata = _get_field_metadata(field_info)

            # Build column name with prefix
            column_name = f"{prefix}{field_name}"

            # Unwrap Optional types
            inner_type, is_optional = _unwrap_optional(field_type)

            # Determine nullability
            # Field is nullable if: it's optional, parent is optional, or has default None
            has_default_none = field_info is not None and field_info.default is None
            nullable = is_optional or parent_optional or has_default_none

            # Check if this is a nested Pydantic model
            if flatten_nested and _is_pydantic_model(inner_type):
                # Recursively generate columns for nested model
                nested_columns = self.generate_columns(
                    model=inner_type,
                    prefix=f"{column_name}_",
                    flatten_nested=True,
                    parent_optional=nullable,
                )
                columns.update(nested_columns)
            else:
                # Generate single column
                sa_type = _get_sqlalchemy_type(inner_type, metadata)
                index = metadata.get("index", False)
                comment = metadata.get("comment")

                columns[column_name] = ColumnSpec(
                    name=column_name,
                    sa_type=sa_type,
                    nullable=nullable,
                    index=index,
                    comment=comment,
                )

        return columns

    def generate_columns_for_result(
        self,
        model: type[BaseModel],
        flatten_config: dict[str, dict[str, Any]],
    ) -> dict[str, ColumnSpec]:
        """Generate columns with custom flatten configuration per field.

        This is designed for the VerificationResult model where each
        component (metadata, template, rubric, etc.) has its own prefix
        and optional status.

        Args:
            model: The root Pydantic model class
            flatten_config: Configuration for each field, e.g.:
                {
                    "metadata": {"prefix": "metadata_", "optional": False},
                    "template": {"prefix": "template_", "optional": True},
                }

        Returns:
            Dictionary mapping column names to ColumnSpec objects
        """
        columns: dict[str, ColumnSpec] = {}

        try:
            hints = get_type_hints(model)
        except Exception:
            logger.debug("get_type_hints failed for %s, falling back to model_fields", model, exc_info=True)
            hints = {name: field.annotation for name, field in model.model_fields.items() if field.annotation}

        for field_name, field_type in hints.items():
            # Get configuration for this field
            config = flatten_config.get(field_name, {})
            prefix = config.get("prefix", f"{field_name}_")
            force_optional = config.get("optional", False)

            # Unwrap Optional types
            inner_type, is_optional = _unwrap_optional(field_type)
            parent_optional = is_optional or force_optional

            if _is_pydantic_model(inner_type):
                # Recursively generate columns for nested model
                nested_columns = self.generate_columns(
                    model=inner_type,
                    prefix=prefix,
                    flatten_nested=True,
                    parent_optional=parent_optional,
                )
                columns.update(nested_columns)
            else:
                # Root-level field (not nested)
                field_info = model.model_fields.get(field_name)
                metadata = _get_field_metadata(field_info)
                sa_type = _get_sqlalchemy_type(inner_type, metadata)

                columns[field_name] = ColumnSpec(
                    name=field_name,
                    sa_type=sa_type,
                    nullable=parent_optional,
                    index=metadata.get("index", False),
                    comment=metadata.get("comment"),
                )

        return columns

    def create_model_class(
        self,
        base: type[DeclarativeBase],
        name: str,
        tablename: str,
        columns: dict[str, ColumnSpec],
        extra_columns: dict[str, Column[Any]] | None = None,
        relationships: dict[str, Any] | None = None,
        table_args: tuple[Any, ...] | None = None,
    ) -> type:
        """Dynamically create a SQLAlchemy ORM model class.

        Args:
            base: SQLAlchemy declarative base class
            name: Name for the generated class
            tablename: Database table name
            columns: Column specifications from generate_columns()
            extra_columns: Additional columns (e.g., id, foreign keys)
            relationships: SQLAlchemy relationship definitions
            table_args: Additional table arguments (indexes, constraints)

        Returns:
            Generated SQLAlchemy model class
        """
        # Build class attributes
        attrs: dict[str, Any] = {
            "__tablename__": tablename,
        }

        # Add extra columns first (id, foreign keys, etc.)
        if extra_columns:
            attrs.update(extra_columns)

        # Add generated columns
        for col_name, col_spec in columns.items():
            attrs[col_name] = col_spec.to_column()

        # Add relationships
        if relationships:
            attrs.update(relationships)

        # Add table args
        if table_args:
            attrs["__table_args__"] = table_args

        # Create and return the class
        return type(name, (base,), attrs)


def generate_indexes_from_columns(
    columns: dict[str, ColumnSpec],  # noqa: ARG001
    tablename: str,
    composite_indexes: list[tuple[str, ...]] | None = None,
) -> list[Index]:
    """Generate SQLAlchemy Index objects from column specifications.

    Args:
        columns: Column specifications
        tablename: Table name for naming indexes
        composite_indexes: List of column name tuples for composite indexes

    Returns:
        List of Index objects
    """
    indexes = []

    # Add composite indexes if specified
    if composite_indexes:
        for col_names in composite_indexes:
            index_name = f"idx_{tablename}_{'_'.join(col_names)}"
            indexes.append(Index(index_name, *col_names))

    return indexes


def get_flat_field_mapping(
    model: type[BaseModel],
    flatten_config: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Get mapping from nested field paths to flat column names.

    Args:
        model: The Pydantic model class
        flatten_config: Flatten configuration

    Returns:
        Dictionary mapping "component.field" to "prefix_field"
    """
    mapping: dict[str, str] = {}

    try:
        hints = get_type_hints(model)
    except Exception:
        logger.debug("get_type_hints failed for %s, falling back to model_fields", model, exc_info=True)
        hints = {name: field.annotation for name, field in model.model_fields.items() if field.annotation}

    for field_name, field_type in hints.items():
        config = flatten_config.get(field_name, {})
        prefix = config.get("prefix", f"{field_name}_")

        inner_type, _ = _unwrap_optional(field_type)

        if _is_pydantic_model(inner_type):
            # Get fields from nested model
            try:
                nested_hints = get_type_hints(inner_type)
            except Exception:
                logger.debug(
                    "get_type_hints failed for nested model %s, falling back to model_fields", inner_type, exc_info=True
                )
                nested_hints = {
                    name: field.annotation
                    for name, field in inner_type.model_fields.items()  # type: ignore[attr-defined]
                    if field.annotation
                }

            for nested_field in nested_hints:
                nested_path = f"{field_name}.{nested_field}"
                column_name = f"{prefix}{nested_field}"
                mapping[nested_path] = column_name
        else:
            # Root-level field
            mapping[field_name] = field_name

    return mapping
