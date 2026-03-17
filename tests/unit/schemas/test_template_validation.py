"""Tests for template kind field type validation."""

import pytest
from pydantic import BaseModel

from karenina.schemas.entities._template_validation import _validate_template_fields


@pytest.mark.unit
class TestValidateTemplateFields:
    def test_accepts_primitives(self):
        class Good(BaseModel):
            count: int
            name: str
            score: float
            found: bool

        _validate_template_fields(Good)  # Should not raise

    def test_accepts_list_of_primitives(self):
        class Good(BaseModel):
            items: list[str]
            counts: list[int]

        _validate_template_fields(Good)

    def test_accepts_optional_with_default(self):
        class Good(BaseModel):
            score: float | None = None
            name: str | None = None

        _validate_template_fields(Good)

    def test_rejects_nested_model(self):
        class Inner(BaseModel):
            x: int

        class Bad(BaseModel):
            nested: Inner

        with pytest.raises(ValueError, match="not allowed"):
            _validate_template_fields(Bad)

    def test_rejects_dict(self):
        class Bad(BaseModel):
            data: dict[str, int]

        with pytest.raises(ValueError, match="not allowed"):
            _validate_template_fields(Bad)

    def test_rejects_optional_without_default(self):
        class Bad(BaseModel):
            score: float | None  # No default

        with pytest.raises(ValueError, match="must have an explicit default"):
            _validate_template_fields(Bad)

    def test_rejects_list_of_model(self):
        class Inner(BaseModel):
            x: int

        class Bad(BaseModel):
            items: list[Inner]

        with pytest.raises(ValueError, match="list must contain primitive"):
            _validate_template_fields(Bad)
