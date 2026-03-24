"""Tests for higher_is_better unification across all trait types."""

import cloudpickle
import pytest

from karenina.schemas.entities.rubric import (
    AgenticRubricTrait,
    CallableRubricTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
    Rubric,
)


def _make_callable_code() -> bytes:
    """Create serialized callable code for test CallableRubricTraits."""
    return cloudpickle.dumps(lambda _text: True)


@pytest.mark.unit
class TestHigherIsBetterUnification:
    """Test that all trait types support higher_is_better: bool | None."""

    def test_llm_trait_accepts_none(self):
        """LLMRubricTrait should accept None for higher_is_better."""
        trait = LLMRubricTrait(
            name="test",
            description="test trait",
            kind="boolean",
            higher_is_better=None,
        )
        assert trait.higher_is_better is None

    def test_llm_trait_defaults_to_true(self):
        """LLMRubricTrait should default to True when not specified."""
        trait = LLMRubricTrait(name="test", description="test trait", kind="boolean")
        assert trait.higher_is_better is True

    def test_regex_trait_accepts_none(self):
        """RegexRubricTrait should accept None for higher_is_better."""
        trait = RegexRubricTrait(
            name="test",
            description="test trait",
            pattern=r"\d+",
            higher_is_better=None,
        )
        assert trait.higher_is_better is None

    def test_regex_trait_defaults_to_true(self):
        """RegexRubricTrait should default to True when not specified."""
        trait = RegexRubricTrait(name="test", description="test trait", pattern=r"\d+")
        assert trait.higher_is_better is True

    def test_callable_trait_accepts_none(self):
        """CallableRubricTrait should accept None for higher_is_better."""
        trait = CallableRubricTrait(
            name="test",
            description="test trait",
            callable_code=_make_callable_code(),
            kind="boolean",
            higher_is_better=None,
        )
        assert trait.higher_is_better is None

    def test_callable_trait_defaults_to_true(self):
        """CallableRubricTrait should default to True when not specified."""
        trait = CallableRubricTrait(
            name="test",
            description="test trait",
            callable_code=_make_callable_code(),
            kind="boolean",
        )
        assert trait.higher_is_better is True

    def test_agentic_trait_preserves_none(self):
        """AgenticRubricTrait should preserve explicit None (not coerce to True)."""
        trait = AgenticRubricTrait(
            name="test",
            description="test trait",
            kind="boolean",
            higher_is_better=None,
        )
        assert trait.higher_is_better is None

    def test_metric_trait_has_higher_is_better(self):
        """MetricRubricTrait should have higher_is_better field defaulting to None."""
        trait = MetricRubricTrait(
            name="test",
            description="test trait",
            metrics=["precision"],
            tp_instructions=["check this"],
        )
        assert trait.higher_is_better is None

    def test_metric_trait_accepts_true(self):
        """MetricRubricTrait should accept True for higher_is_better."""
        trait = MetricRubricTrait(
            name="test",
            description="test trait",
            metrics=["precision"],
            tp_instructions=["check this"],
            higher_is_better=True,
        )
        assert trait.higher_is_better is True

    def test_legacy_data_without_field_defaults_to_true(self):
        """Legacy data missing higher_is_better should still get True."""
        trait = LLMRubricTrait.model_validate({"name": "test", "description": "test", "kind": "boolean"})
        assert trait.higher_is_better is True

    def test_get_trait_directionalities_includes_metric(self):
        """Rubric.get_trait_directionalities() should include metric traits."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="clarity",
                    description="clear",
                    kind="boolean",
                    higher_is_better=True,
                )
            ],
            metric_traits=[
                MetricRubricTrait(
                    name="precision",
                    description="prec",
                    metrics=["precision"],
                    tp_instructions=["check this"],
                )
            ],
        )
        dirs = rubric.get_trait_directionalities()
        assert "clarity" in dirs
        assert dirs["clarity"] is True
        assert "precision" in dirs
        assert dirs["precision"] is None

    def test_get_trait_directionalities_none_values(self):
        """Traits with higher_is_better=None should appear as None in directionalities."""
        rubric = Rubric(
            llm_traits=[
                LLMRubricTrait(
                    name="word_count",
                    description="wc",
                    kind="boolean",
                    higher_is_better=None,
                )
            ],
        )
        dirs = rubric.get_trait_directionalities()
        assert dirs["word_count"] is None

    def test_agentic_template_kind_still_none(self):
        """AgenticRubricTrait with template kind should still have higher_is_better=None."""
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            result: str

        trait = AgenticRubricTrait(
            name="test",
            description="test",
            kind=MyOutput,
            higher_is_better=None,
        )
        assert trait.higher_is_better is None


@pytest.mark.unit
class TestRubricFromTraits:
    """Test Rubric.from_traits() factory method."""

    def test_from_mixed_traits(self):
        """from_traits should categorize a flat list into typed trait lists."""
        llm = LLMRubricTrait(name="clarity", description="clear", kind="boolean")
        regex = RegexRubricTrait(name="has_number", description="num", pattern=r"\d+")
        metric = MetricRubricTrait(
            name="precision", description="prec", metrics=["precision"], tp_instructions=["check this"]
        )

        rubric = Rubric.from_traits([llm, regex, metric])

        assert len(rubric.llm_traits) == 1
        assert rubric.llm_traits[0].name == "clarity"
        assert len(rubric.regex_traits) == 1
        assert rubric.regex_traits[0].name == "has_number"
        assert len(rubric.metric_traits) == 1
        assert rubric.metric_traits[0].name == "precision"
        assert len(rubric.callable_traits) == 0
        assert len(rubric.agentic_traits) == 0

    def test_from_empty_traits(self):
        """from_traits with empty list should produce empty Rubric."""
        rubric = Rubric.from_traits([])
        assert len(rubric.llm_traits) == 0
        assert len(rubric.regex_traits) == 0

    def test_from_none_returns_none(self):
        """from_traits with None should return None."""
        assert Rubric.from_traits(None) is None
