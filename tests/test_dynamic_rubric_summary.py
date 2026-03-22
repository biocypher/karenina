"""Tests for the summary field on all rubric trait types.

Verifies that:
- Each trait type has a summary field defaulting to None
- Summary can be set on each type
- Summary appears in model_dump() output
- Backward compatibility: deserializing without summary key works
"""

import pytest

from karenina.schemas.entities import (
    AgenticRubricTrait,
    CallableRubricTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
)

# =============================================================================
# LLMRubricTrait summary tests
# =============================================================================


@pytest.mark.unit
class TestLLMRubricTraitSummary:
    """Tests for summary field on LLMRubricTrait."""

    def test_summary_defaults_to_none(self) -> None:
        """LLMRubricTrait summary field defaults to None."""
        trait = LLMRubricTrait(
            name="safety",
            kind="boolean",
            higher_is_better=True,
        )
        assert trait.summary is None

    def test_summary_can_be_set(self) -> None:
        """LLMRubricTrait summary can be assigned a string value."""
        trait = LLMRubricTrait(
            name="safety",
            kind="boolean",
            higher_is_better=True,
            summary="Response safety",
        )
        assert trait.summary == "Response safety"

    def test_summary_appears_in_model_dump(self) -> None:
        """LLMRubricTrait model_dump() includes summary key."""
        trait = LLMRubricTrait(
            name="safety",
            kind="boolean",
            higher_is_better=True,
            summary="Response safety",
        )
        dumped = trait.model_dump()
        assert "summary" in dumped
        assert dumped["summary"] == "Response safety"

    def test_summary_none_in_model_dump(self) -> None:
        """LLMRubricTrait model_dump() includes summary key even when None."""
        trait = LLMRubricTrait(
            name="safety",
            kind="boolean",
            higher_is_better=True,
        )
        dumped = trait.model_dump()
        assert "summary" in dumped
        assert dumped["summary"] is None

    def test_backward_compat_deserialize_without_summary(self) -> None:
        """Deserializing LLMRubricTrait dict without summary key works, defaulting to None."""
        data = {
            "name": "safety",
            "kind": "boolean",
            "higher_is_better": True,
        }
        trait = LLMRubricTrait.model_validate(data)
        assert trait.summary is None


# =============================================================================
# RegexRubricTrait summary tests
# =============================================================================


@pytest.mark.unit
class TestRegexTraitSummary:
    """Tests for summary field on RegexRubricTrait."""

    def test_summary_defaults_to_none(self) -> None:
        """RegexRubricTrait summary field defaults to None."""
        trait = RegexRubricTrait(
            name="has_citation",
            pattern=r"\[\d+\]",
            higher_is_better=True,
        )
        assert trait.summary is None

    def test_summary_can_be_set(self) -> None:
        """RegexRubricTrait summary can be assigned a string value."""
        trait = RegexRubricTrait(
            name="has_citation",
            pattern=r"\[\d+\]",
            higher_is_better=True,
            summary="Citation presence",
        )
        assert trait.summary == "Citation presence"

    def test_summary_appears_in_model_dump(self) -> None:
        """RegexRubricTrait model_dump() includes summary key."""
        trait = RegexRubricTrait(
            name="has_citation",
            pattern=r"\[\d+\]",
            higher_is_better=True,
            summary="Citation presence",
        )
        dumped = trait.model_dump()
        assert "summary" in dumped
        assert dumped["summary"] == "Citation presence"

    def test_summary_none_in_model_dump(self) -> None:
        """RegexRubricTrait model_dump() includes summary key even when None."""
        trait = RegexRubricTrait(
            name="has_citation",
            pattern=r"\[\d+\]",
            higher_is_better=True,
        )
        dumped = trait.model_dump()
        assert "summary" in dumped
        assert dumped["summary"] is None

    def test_backward_compat_deserialize_without_summary(self) -> None:
        """Deserializing RegexRubricTrait dict without summary key works, defaulting to None."""
        data = {
            "name": "has_citation",
            "pattern": r"\[\d+\]",
            "higher_is_better": True,
        }
        trait = RegexRubricTrait.model_validate(data)
        assert trait.summary is None


# =============================================================================
# CallableRubricTrait summary tests
# =============================================================================


@pytest.mark.unit
class TestCallableTraitSummary:
    """Tests for summary field on CallableRubricTrait."""

    def test_summary_defaults_to_none(self) -> None:
        """CallableRubricTrait summary field defaults to None."""
        trait = CallableRubricTrait.from_callable(
            name="word_count",
            func=lambda text: len(text.split()) >= 50,
            kind="boolean",
            higher_is_better=True,
        )
        assert trait.summary is None

    def test_summary_can_be_set_via_from_callable(self) -> None:
        """CallableRubricTrait.from_callable() accepts and stores summary."""
        trait = CallableRubricTrait.from_callable(
            name="word_count",
            func=lambda text: len(text.split()) >= 50,
            kind="boolean",
            higher_is_better=True,
            summary="Minimum word count",
        )
        assert trait.summary == "Minimum word count"

    def test_summary_appears_in_model_dump(self) -> None:
        """CallableRubricTrait model_dump() includes summary key."""
        trait = CallableRubricTrait.from_callable(
            name="word_count",
            func=lambda text: len(text.split()) >= 50,
            kind="boolean",
            higher_is_better=True,
            summary="Minimum word count",
        )
        dumped = trait.model_dump()
        assert "summary" in dumped
        assert dumped["summary"] == "Minimum word count"

    def test_summary_none_in_model_dump(self) -> None:
        """CallableRubricTrait model_dump() includes summary key even when None."""
        trait = CallableRubricTrait.from_callable(
            name="word_count",
            func=lambda text: len(text.split()) >= 50,
            kind="boolean",
            higher_is_better=True,
        )
        dumped = trait.model_dump()
        assert "summary" in dumped
        assert dumped["summary"] is None

    def test_backward_compat_deserialize_without_summary(self) -> None:
        """Round-trip serialization without summary key works, defaulting to None."""
        original = CallableRubricTrait.from_callable(
            name="word_count",
            func=lambda text: len(text.split()) >= 50,
            kind="boolean",
            higher_is_better=True,
        )
        data = original.model_dump()
        data.pop("summary", None)
        restored = CallableRubricTrait.model_validate(data)
        assert restored.summary is None


# =============================================================================
# MetricRubricTrait summary tests
# =============================================================================


@pytest.mark.unit
class TestMetricRubricTraitSummary:
    """Tests for summary field on MetricRubricTrait."""

    def test_summary_defaults_to_none(self) -> None:
        """MetricRubricTrait summary field defaults to None."""
        trait = MetricRubricTrait(
            name="entity_extraction",
            metrics=["precision", "recall"],
            tp_instructions=["mitochondria", "apoptosis"],
        )
        assert trait.summary is None

    def test_summary_can_be_set(self) -> None:
        """MetricRubricTrait summary can be assigned a string value."""
        trait = MetricRubricTrait(
            name="entity_extraction",
            metrics=["precision", "recall"],
            tp_instructions=["mitochondria", "apoptosis"],
            summary="Entity extraction quality",
        )
        assert trait.summary == "Entity extraction quality"

    def test_summary_appears_in_model_dump(self) -> None:
        """MetricRubricTrait model_dump() includes summary key."""
        trait = MetricRubricTrait(
            name="entity_extraction",
            metrics=["precision", "recall"],
            tp_instructions=["mitochondria", "apoptosis"],
            summary="Entity extraction quality",
        )
        dumped = trait.model_dump()
        assert "summary" in dumped
        assert dumped["summary"] == "Entity extraction quality"

    def test_summary_none_in_model_dump(self) -> None:
        """MetricRubricTrait model_dump() includes summary key even when None."""
        trait = MetricRubricTrait(
            name="entity_extraction",
            metrics=["precision", "recall"],
            tp_instructions=["mitochondria", "apoptosis"],
        )
        dumped = trait.model_dump()
        assert "summary" in dumped
        assert dumped["summary"] is None

    def test_backward_compat_deserialize_without_summary(self) -> None:
        """Deserializing MetricRubricTrait dict without summary key works, defaulting to None."""
        data = {
            "name": "entity_extraction",
            "metrics": ["precision", "recall"],
            "tp_instructions": ["mitochondria", "apoptosis"],
        }
        trait = MetricRubricTrait.model_validate(data)
        assert trait.summary is None


# =============================================================================
# AgenticRubricTrait summary tests
# =============================================================================


@pytest.mark.unit
class TestAgenticRubricTraitSummary:
    """Tests for summary field on AgenticRubricTrait."""

    def test_summary_defaults_to_none(self) -> None:
        """AgenticRubricTrait summary field defaults to None."""
        trait = AgenticRubricTrait(
            name="file_verification",
            description="Check that the agent created the expected output file.",
            kind="boolean",
            higher_is_better=True,
        )
        assert trait.summary is None

    def test_summary_can_be_set(self) -> None:
        """AgenticRubricTrait summary can be assigned a string value."""
        trait = AgenticRubricTrait(
            name="file_verification",
            description="Check that the agent created the expected output file.",
            kind="boolean",
            higher_is_better=True,
            summary="Output file present",
        )
        assert trait.summary == "Output file present"

    def test_summary_appears_in_model_dump(self) -> None:
        """AgenticRubricTrait model_dump() includes summary key."""
        trait = AgenticRubricTrait(
            name="file_verification",
            description="Check that the agent created the expected output file.",
            kind="boolean",
            higher_is_better=True,
            summary="Output file present",
        )
        dumped = trait.model_dump()
        assert "summary" in dumped
        assert dumped["summary"] == "Output file present"

    def test_summary_none_in_model_dump(self) -> None:
        """AgenticRubricTrait model_dump() includes summary key even when None."""
        trait = AgenticRubricTrait(
            name="file_verification",
            description="Check that the agent created the expected output file.",
            kind="boolean",
            higher_is_better=True,
        )
        dumped = trait.model_dump()
        assert "summary" in dumped
        assert dumped["summary"] is None

    def test_backward_compat_deserialize_without_summary(self) -> None:
        """Deserializing AgenticRubricTrait dict without summary key works, defaulting to None."""
        data = {
            "name": "file_verification",
            "description": "Check that the agent created the expected output file.",
            "kind": "boolean",
            "higher_is_better": True,
        }
        trait = AgenticRubricTrait.model_validate(data)
        assert trait.summary is None
