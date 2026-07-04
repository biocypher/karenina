"""Tests for migrate_template_id SQL correctness (Issue 041)."""

import inspect

import pytest

from karenina.storage import migrate_template_id


@pytest.mark.unit
class TestMigrateTemplateIdSQL:
    """Issue 041: SQL must reference bq.template_id, not bq.metadata.template_id."""

    def test_no_invalid_sql_column_reference(self) -> None:
        """The migration SQL must not use bq.metadata.template_id (invalid notation)."""
        source = inspect.getsource(migrate_template_id)
        assert "bq.metadata.template_id" not in source

    def test_correct_sql_column_reference(self) -> None:
        """The migration SQL must use bq.template_id (valid column reference)."""
        source = inspect.getsource(migrate_template_id)
        assert "bq.template_id" in source
