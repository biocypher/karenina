"""Tests verifying DB columns are auto-generated for scenario metadata fields."""

import pytest


@pytest.mark.unit
class TestScenarioDBColumns:
    """Verify that PydanticSQLAlchemyMapper picks up scenario metadata fields."""

    def test_scenario_columns_on_verification_result_model(self):
        """The auto-generated VerificationResultModel should include scenario columns."""
        from karenina.storage.generated_models import VerificationResultModel

        table = VerificationResultModel.__table__
        column_names = {c.name for c in table.columns}

        assert "metadata_scenario_id" in column_names
        assert "metadata_scenario_node" in column_names
        assert "metadata_scenario_turn" in column_names
        assert "metadata_scenario_path" in column_names

    def test_scenario_columns_are_nullable(self):
        """Scenario columns should be nullable (None for standalone questions)."""
        from karenina.storage.generated_models import VerificationResultModel

        table = VerificationResultModel.__table__
        columns_by_name = {c.name: c for c in table.columns}

        for col_name in [
            "metadata_scenario_id",
            "metadata_scenario_node",
            "metadata_scenario_turn",
            "metadata_scenario_path",
        ]:
            col = columns_by_name[col_name]
            assert col.nullable is True, f"{col_name} should be nullable"

    def test_get_column_names_includes_scenario_fields(self):
        """The get_column_names() helper should list scenario columns."""
        from karenina.storage.generated_models import get_column_names

        names = get_column_names()
        assert "metadata_scenario_id" in names
        assert "metadata_scenario_node" in names
        assert "metadata_scenario_turn" in names
        assert "metadata_scenario_path" in names
