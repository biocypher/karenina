"""Tests that _rubric_kind_validation module is importable and functional."""

import pytest


@pytest.mark.unit
class TestRubricKindValidationImport:
    """The validation module should be importable from its new location."""

    def test_import_from_new_path(self) -> None:
        from karenina.schemas.entities._rubric_kind_validation import _validate_template_fields

        assert callable(_validate_template_fields)

    def test_rubric_still_imports_validator(self) -> None:
        """rubric.py should still work after the import path change."""
        from karenina.schemas.entities.rubric import LLMRubricTrait

        assert LLMRubricTrait is not None
