"""Tests for checkpoint trait converter template kind support."""

import pytest
from pydantic import BaseModel, Field

from karenina.schemas.entities.rubric import AgenticRubricTrait


class _CheckpointFindings(BaseModel):
    count: int = Field(description="Count")
    found: bool = Field(description="Found flag")


@pytest.mark.unit
class TestCheckpointTraitConverterTemplateKind:
    def test_agentic_trait_to_rating_roundtrip(self):
        from karenina.utils.checkpoint_trait_converters import (
            _convert_agentic_trait_to_rating,
            _convert_rating_to_agentic_trait,
        )

        trait = AgenticRubricTrait(
            name="test",
            description="Test trait.",
            kind=_CheckpointFindings,
            higher_is_better=None,
            context_mode="trace_only",
        )

        rating = _convert_agentic_trait_to_rating(trait, "global")
        rebuilt = _convert_rating_to_agentic_trait(rating)

        assert rebuilt.is_template_kind is True
        assert rebuilt.higher_is_better is None
        assert set(rebuilt.kind.model_fields.keys()) == {"count", "found"}

    def test_template_kind_rating_has_zero_bounds(self):
        """Template kind traits should have bestRating=0, worstRating=0."""
        from karenina.utils.checkpoint_trait_converters import (
            _convert_agentic_trait_to_rating,
        )

        trait = AgenticRubricTrait(
            name="test",
            description="Test trait.",
            kind=_CheckpointFindings,
            higher_is_better=None,
            context_mode="trace_only",
        )

        rating = _convert_agentic_trait_to_rating(trait, "global")
        assert rating.bestRating == 0.0
        assert rating.worstRating == 0.0

    def test_template_kind_serialized_as_dict(self):
        """Template kind should be serialized as {'type': 'template', 'schema': ...}."""
        from karenina.utils.checkpoint_trait_converters import (
            _convert_agentic_trait_to_rating,
        )

        trait = AgenticRubricTrait(
            name="test",
            description="Test trait.",
            kind=_CheckpointFindings,
            higher_is_better=None,
            context_mode="trace_only",
        )

        rating = _convert_agentic_trait_to_rating(trait, "global")

        # Find the kind property
        kind_prop = next(p for p in rating.additionalProperty if p.name == "kind")
        assert isinstance(kind_prop.value, dict)
        assert kind_prop.value["type"] == "template"
        assert "schema" in kind_prop.value
        assert "properties" in kind_prop.value["schema"]

    def test_non_template_kind_unchanged(self):
        """Boolean/score/literal kinds should still roundtrip correctly."""
        from karenina.utils.checkpoint_trait_converters import (
            _convert_agentic_trait_to_rating,
            _convert_rating_to_agentic_trait,
        )

        trait = AgenticRubricTrait(
            name="bool_test",
            description="Boolean trait.",
            kind="boolean",
            higher_is_better=True,
            context_mode="trace_only",
        )

        rating = _convert_agentic_trait_to_rating(trait, "global")
        rebuilt = _convert_rating_to_agentic_trait(rating)

        assert rebuilt.kind == "boolean"
        assert rebuilt.higher_is_better is True
        assert rebuilt.is_template_kind is False
