"""Tests for DynamicRubric class and merge_dynamic_rubrics function.

Covers:
- Construction with various trait combinations
- Validation (missing summary+description raises ValueError)
- Warning when summary is None but description exists
- get_trait_names
- is_empty
- resolve_concept_text
- merge_dynamic_rubrics (none+none, global only, question only, concatenation, name collision)
"""

import logging

import pytest
from pydantic import ValidationError

from karenina.schemas.entities import (
    AgenticRubricTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
)
from karenina.schemas.entities.rubric import DynamicRubric, merge_dynamic_rubrics

# =============================================================================
# Helpers
# =============================================================================


def _make_llm_trait(name: str, summary: str | None = None, description: str | None = None) -> LLMRubricTrait:
    """Create a minimal LLMRubricTrait for testing."""
    return LLMRubricTrait(
        name=name,
        kind="boolean",
        higher_is_better=True,
        summary=summary,
        description=description,
    )


def _make_regex_trait(name: str, summary: str | None = None, description: str | None = None) -> RegexRubricTrait:
    """Create a minimal RegexRubricTrait for testing."""
    return RegexRubricTrait(
        name=name,
        pattern=r"\btest\b",
        higher_is_better=True,
        summary=summary,
        description=description,
    )


def _make_metric_trait(name: str, summary: str | None = None, description: str | None = None) -> MetricRubricTrait:
    """Create a minimal MetricRubricTrait for testing."""
    return MetricRubricTrait(
        name=name,
        evaluation_mode="tp_only",
        metrics=["precision"],
        tp_instructions=["instruction"],
        summary=summary,
        description=description,
    )


def _make_agentic_trait(name: str, summary: str | None = None, description: str | None = None) -> AgenticRubricTrait:
    """Create a minimal AgenticRubricTrait for testing."""
    return AgenticRubricTrait(
        name=name,
        kind="boolean",
        higher_is_better=True,
        description=description or "placeholder",
        summary=summary,
    )


# =============================================================================
# Construction Tests
# =============================================================================


@pytest.mark.unit
class TestDynamicRubricConstruction:
    """Tests for DynamicRubric construction."""

    def test_empty_construction(self) -> None:
        """DynamicRubric with no traits is valid."""
        dr = DynamicRubric()
        assert dr.llm_traits == []
        assert dr.regex_traits == []
        assert dr.callable_traits == []
        assert dr.metric_traits == []
        assert dr.agentic_traits == []

    def test_construction_with_llm_traits(self) -> None:
        """DynamicRubric accepts LLM traits with summary set."""
        trait = _make_llm_trait("safety", summary="Response safety")
        dr = DynamicRubric(llm_traits=[trait])
        assert len(dr.llm_traits) == 1
        assert dr.llm_traits[0].name == "safety"

    def test_construction_with_regex_traits(self) -> None:
        """DynamicRubric accepts regex traits."""
        trait = _make_regex_trait("has_citation", summary="Citation present")
        dr = DynamicRubric(regex_traits=[trait])
        assert len(dr.regex_traits) == 1

    def test_construction_with_metric_traits(self) -> None:
        """DynamicRubric accepts metric traits."""
        trait = _make_metric_trait("coverage", summary="Entity coverage")
        dr = DynamicRubric(metric_traits=[trait])
        assert len(dr.metric_traits) == 1

    def test_construction_with_agentic_traits(self) -> None:
        """DynamicRubric accepts agentic traits."""
        trait = _make_agentic_trait("code_quality", summary="Code quality check")
        dr = DynamicRubric(agentic_traits=[trait])
        assert len(dr.agentic_traits) == 1

    def test_construction_with_mixed_traits(self) -> None:
        """DynamicRubric accepts a mix of trait types."""
        dr = DynamicRubric(
            llm_traits=[_make_llm_trait("safety", summary="Safety")],
            regex_traits=[_make_regex_trait("format", summary="Format check")],
            metric_traits=[_make_metric_trait("coverage", summary="Coverage")],
        )
        assert len(dr.llm_traits) == 1
        assert len(dr.regex_traits) == 1
        assert len(dr.metric_traits) == 1

    def test_extra_fields_forbidden(self) -> None:
        """DynamicRubric rejects unknown fields."""
        with pytest.raises(ValidationError):
            DynamicRubric(unknown_field="value")


# =============================================================================
# Validation Tests
# =============================================================================


@pytest.mark.unit
class TestDynamicRubricValidation:
    """Tests for DynamicRubric validation (summary/description check)."""

    def test_trait_with_summary_only_passes(self) -> None:
        """Trait with summary but no description passes validation."""
        trait = _make_llm_trait("safety", summary="Response safety")
        dr = DynamicRubric(llm_traits=[trait])
        assert dr.llm_traits[0].summary == "Response safety"

    def test_trait_with_description_only_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Trait with description but no summary logs a warning."""
        trait = _make_llm_trait("safety", description="Check response safety")
        with caplog.at_level(logging.WARNING):
            dr = DynamicRubric(llm_traits=[trait])
        assert dr.llm_traits[0].description == "Check response safety"
        assert any("safety" in record.message and "summary" in record.message for record in caplog.records)

    def test_trait_with_both_summary_and_description_passes(self) -> None:
        """Trait with both summary and description passes validation."""
        trait = _make_llm_trait("safety", summary="Safety", description="Check response safety")
        dr = DynamicRubric(llm_traits=[trait])
        assert dr.llm_traits[0].summary == "Safety"

    def test_trait_with_neither_summary_nor_description_raises(self) -> None:
        """Trait with neither summary nor description raises ValueError."""
        trait = _make_llm_trait("safety")
        with pytest.raises(ValidationError, match="safety"):
            DynamicRubric(llm_traits=[trait])

    def test_mixed_traits_one_missing_both_raises(self) -> None:
        """If any trait across types is missing both summary and description, validation fails."""
        good_trait = _make_llm_trait("safety", summary="Safety")
        bad_trait = _make_regex_trait("format")  # neither summary nor description
        with pytest.raises(ValidationError, match="format"):
            DynamicRubric(llm_traits=[good_trait], regex_traits=[bad_trait])

    def test_regex_description_only_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Regex trait with description but no summary logs a warning."""
        trait = _make_regex_trait("citation", description="Has citations")
        with caplog.at_level(logging.WARNING):
            DynamicRubric(regex_traits=[trait])
        assert any("citation" in record.message for record in caplog.records)

    def test_metric_with_neither_raises(self) -> None:
        """Metric trait with neither summary nor description raises."""
        trait = _make_metric_trait("coverage")
        with pytest.raises(ValidationError, match="coverage"):
            DynamicRubric(metric_traits=[trait])


# =============================================================================
# get_trait_names Tests
# =============================================================================


@pytest.mark.unit
class TestDynamicRubricGetTraitNames:
    """Tests for DynamicRubric.get_trait_names()."""

    def test_empty_returns_empty_list(self) -> None:
        """Empty DynamicRubric returns empty trait names list."""
        dr = DynamicRubric()
        assert dr.get_trait_names() == []

    def test_returns_all_names_across_types(self) -> None:
        """get_trait_names returns names from all trait types."""
        dr = DynamicRubric(
            llm_traits=[_make_llm_trait("safety", summary="Safety")],
            regex_traits=[_make_regex_trait("format", summary="Format")],
            metric_traits=[_make_metric_trait("coverage", summary="Coverage")],
        )
        names = dr.get_trait_names()
        assert "safety" in names
        assert "format" in names
        assert "coverage" in names
        assert len(names) == 3

    def test_preserves_order(self) -> None:
        """get_trait_names returns names in trait-type order: llm, regex, callable, metric, agentic."""
        dr = DynamicRubric(
            llm_traits=[_make_llm_trait("a", summary="A")],
            regex_traits=[_make_regex_trait("b", summary="B")],
            metric_traits=[_make_metric_trait("c", summary="C")],
        )
        names = dr.get_trait_names()
        assert names == ["a", "b", "c"]


# =============================================================================
# is_empty Tests
# =============================================================================


@pytest.mark.unit
class TestDynamicRubricIsEmpty:
    """Tests for DynamicRubric.is_empty()."""

    def test_empty_rubric_is_empty(self) -> None:
        """DynamicRubric with no traits is empty."""
        dr = DynamicRubric()
        assert dr.is_empty() is True

    def test_rubric_with_trait_is_not_empty(self) -> None:
        """DynamicRubric with any trait is not empty."""
        dr = DynamicRubric(llm_traits=[_make_llm_trait("safety", summary="Safety")])
        assert dr.is_empty() is False

    def test_rubric_with_only_regex_trait_is_not_empty(self) -> None:
        """DynamicRubric with only a regex trait is not empty."""
        dr = DynamicRubric(regex_traits=[_make_regex_trait("fmt", summary="Format")])
        assert dr.is_empty() is False


# =============================================================================
# resolve_concept_text Tests
# =============================================================================


@pytest.mark.unit
class TestDynamicRubricResolveConceptText:
    """Tests for DynamicRubric.resolve_concept_text()."""

    def test_returns_summary_when_set(self) -> None:
        """resolve_concept_text returns summary when both are set."""
        trait = _make_llm_trait("safety", summary="Response safety", description="Detailed desc")
        dr = DynamicRubric(llm_traits=[trait])
        assert dr.resolve_concept_text(trait) == "Response safety"

    def test_returns_description_when_summary_is_none(self) -> None:
        """resolve_concept_text falls back to description when summary is None."""
        trait = _make_llm_trait("safety", description="Detailed safety check")
        # Bypass validation (description-only triggers warning, not error)
        dr = DynamicRubric.__new__(DynamicRubric)
        object.__setattr__(dr, "llm_traits", [trait])
        object.__setattr__(dr, "regex_traits", [])
        object.__setattr__(dr, "callable_traits", [])
        object.__setattr__(dr, "metric_traits", [])
        object.__setattr__(dr, "agentic_traits", [])
        assert dr.resolve_concept_text(trait) == "Detailed safety check"

    def test_returns_summary_when_only_summary_set(self) -> None:
        """resolve_concept_text returns summary when only summary is set."""
        trait = _make_llm_trait("safety", summary="Safety concept")
        dr = DynamicRubric(llm_traits=[trait])
        assert dr.resolve_concept_text(trait) == "Safety concept"


# =============================================================================
# merge_dynamic_rubrics Tests
# =============================================================================


@pytest.mark.unit
class TestMergeDynamicRubrics:
    """Tests for merge_dynamic_rubrics function."""

    def test_both_none_returns_none(self) -> None:
        """merge_dynamic_rubrics(None, None) returns None."""
        result = merge_dynamic_rubrics(None, None)
        assert result is None

    def test_global_only_returns_global(self) -> None:
        """When question is None, returns the global DynamicRubric."""
        global_dr = DynamicRubric(llm_traits=[_make_llm_trait("safety", summary="Safety")])
        result = merge_dynamic_rubrics(global_dr, None)
        assert result is global_dr

    def test_question_only_returns_question(self) -> None:
        """When global is None, returns the question DynamicRubric."""
        question_dr = DynamicRubric(regex_traits=[_make_regex_trait("fmt", summary="Format")])
        result = merge_dynamic_rubrics(None, question_dr)
        assert result is question_dr

    def test_concatenation_merges_traits(self) -> None:
        """Both present: traits from both are concatenated in the merged result."""
        global_dr = DynamicRubric(llm_traits=[_make_llm_trait("safety", summary="Safety")])
        question_dr = DynamicRubric(
            llm_traits=[_make_llm_trait("clarity", summary="Clarity")],
            regex_traits=[_make_regex_trait("fmt", summary="Format")],
        )
        result = merge_dynamic_rubrics(global_dr, question_dr)
        assert result is not None
        assert len(result.llm_traits) == 2
        assert len(result.regex_traits) == 1
        names = result.get_trait_names()
        assert "safety" in names
        assert "clarity" in names
        assert "fmt" in names

    def test_name_collision_raises(self) -> None:
        """Duplicate trait names across global and question raise ValueError."""
        global_dr = DynamicRubric(llm_traits=[_make_llm_trait("safety", summary="Safety")])
        question_dr = DynamicRubric(llm_traits=[_make_llm_trait("safety", summary="Safety v2")])
        with pytest.raises(ValueError, match="safety"):
            merge_dynamic_rubrics(global_dr, question_dr)

    def test_cross_type_name_allowed(self) -> None:
        """Cross-type same-name traits merge without error (type-segregated storage)."""
        global_dr = DynamicRubric(llm_traits=[_make_llm_trait("check", summary="LLM check")])
        question_dr = DynamicRubric(regex_traits=[_make_regex_trait("check", summary="Regex check")])
        result = merge_dynamic_rubrics(global_dr, question_dr)
        assert len(result.llm_traits) == 1
        assert len(result.regex_traits) == 1

    def test_merged_result_is_new_instance(self) -> None:
        """Merged result is a new DynamicRubric, not one of the inputs."""
        global_dr = DynamicRubric(llm_traits=[_make_llm_trait("a", summary="A")])
        question_dr = DynamicRubric(llm_traits=[_make_llm_trait("b", summary="B")])
        result = merge_dynamic_rubrics(global_dr, question_dr)
        assert result is not global_dr
        assert result is not question_dr

    def test_empty_plus_populated_returns_populated(self) -> None:
        """Merging an empty DynamicRubric with a populated one works correctly."""
        empty_dr = DynamicRubric()
        populated_dr = DynamicRubric(llm_traits=[_make_llm_trait("safety", summary="Safety")])
        result = merge_dynamic_rubrics(empty_dr, populated_dr)
        assert result is not None
        assert len(result.llm_traits) == 1


# =============================================================================
# _all_traits Tests
# =============================================================================


@pytest.mark.unit
class TestDynamicRubricAllTraits:
    """Tests for DynamicRubric._all_traits() helper."""

    def test_empty_returns_empty(self) -> None:
        """_all_traits on empty DynamicRubric returns empty list."""
        dr = DynamicRubric()
        assert dr._all_traits() == []

    def test_returns_all_trait_objects(self) -> None:
        """_all_traits returns flat list of all trait objects."""
        llm = _make_llm_trait("a", summary="A")
        regex = _make_regex_trait("b", summary="B")
        dr = DynamicRubric(llm_traits=[llm], regex_traits=[regex])
        all_traits = dr._all_traits()
        assert len(all_traits) == 2
        assert llm in all_traits
        assert regex in all_traits
