"""Tests for checkpoint serialization round-trip fixes.

Covers issues: 171, 078, 172, 079, 173.
"""

import inspect
from datetime import datetime

import pytest

from karenina.schemas.checkpoint import (
    SchemaOrgAnswer,
    SchemaOrgDataFeedItem,
    SchemaOrgPropertyValue,
    SchemaOrgQuestion,
    SchemaOrgRating,
    SchemaOrgSoftwareSourceCode,
)
from karenina.schemas.entities.rubric import (
    AgenticRubricTrait,
    CallableRubricTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
)
from karenina.utils.checkpoint import (
    create_jsonld_benchmark,
    validate_jsonld_benchmark,
)
from karenina.utils.checkpoint_trait_converters import (
    convert_rating_to_rubric_trait,
    convert_rubric_trait_to_rating,
    strip_deep_judgment_config_from_checkpoint,
)

# =============================================================================
# Issue 171: Checkpoint validation must accept agentic rubric trait types
# =============================================================================


@pytest.mark.unit
class TestIssue171AgenticTraitValidation:
    """Checkpoint validation must accept agentic rubric trait types."""

    def test_global_agentic_trait_accepted_at_dataset_level(self) -> None:
        """Global AgenticRubricTrait should be valid at dataset level."""
        benchmark = create_jsonld_benchmark(name="Test")
        benchmark.rating = [
            SchemaOrgRating(
                name="agentic_test",
                bestRating=1,
                worstRating=0,
                additionalType="karenina:GlobalAgenticRubricTrait",
            )
        ]

        is_valid, message = validate_jsonld_benchmark(benchmark)

        assert is_valid is True, f"Expected valid but got: {message}"

    def test_global_agentic_trait_accepted_at_question_level(self) -> None:
        """Global AgenticRubricTrait should be valid at question level."""
        benchmark = create_jsonld_benchmark(name="Test")
        item = SchemaOrgDataFeedItem(
            dateCreated=datetime.now().isoformat(),
            dateModified=datetime.now().isoformat(),
            item=SchemaOrgQuestion(
                text="Test question?",
                acceptedAnswer=SchemaOrgAnswer(text="Answer"),
                hasPart=SchemaOrgSoftwareSourceCode(name="Answer", text="class Answer: pass"),
                rating=[
                    SchemaOrgRating(
                        name="agentic_test",
                        bestRating=1,
                        worstRating=0,
                        additionalType="karenina:GlobalAgenticRubricTrait",
                    )
                ],
            ),
        )
        benchmark.dataFeedElement.append(item)

        is_valid, message = validate_jsonld_benchmark(benchmark)

        assert is_valid is True, f"Expected valid but got: {message}"

    def test_question_specific_agentic_trait_accepted_at_question_level(self) -> None:
        """QuestionSpecificAgenticRubricTrait should be valid at question level."""
        benchmark = create_jsonld_benchmark(name="Test")
        item = SchemaOrgDataFeedItem(
            dateCreated=datetime.now().isoformat(),
            dateModified=datetime.now().isoformat(),
            item=SchemaOrgQuestion(
                text="Test question?",
                acceptedAnswer=SchemaOrgAnswer(text="Answer"),
                hasPart=SchemaOrgSoftwareSourceCode(name="Answer", text="class Answer: pass"),
                rating=[
                    SchemaOrgRating(
                        name="agentic_test",
                        bestRating=1,
                        worstRating=0,
                        additionalType="karenina:QuestionSpecificAgenticRubricTrait",
                    )
                ],
            ),
        )
        benchmark.dataFeedElement.append(item)

        is_valid, message = validate_jsonld_benchmark(benchmark)

        assert is_valid is True, f"Expected valid but got: {message}"

    def test_question_specific_agentic_trait_rejected_at_dataset_level(self) -> None:
        """QuestionSpecific traits should still be rejected at dataset level."""
        benchmark = create_jsonld_benchmark(name="Test")
        benchmark.rating = [
            SchemaOrgRating(
                name="agentic_test",
                bestRating=1,
                worstRating=0,
                additionalType="karenina:QuestionSpecificAgenticRubricTrait",
            )
        ]

        is_valid, message = validate_jsonld_benchmark(benchmark)

        assert is_valid is False


# =============================================================================
# Issue 078: Strip function must cover literal-kind LLM traits
# =============================================================================


@pytest.mark.unit
class TestIssue078StripLiteralLLMTraits:
    """strip_deep_judgment_config_from_checkpoint must cover literal-kind LLM traits."""

    def test_strip_removes_dj_from_global_literal_llm_trait(self) -> None:
        """DJ properties should be stripped from GlobalLLMRubricTrait ratings."""
        checkpoint = create_jsonld_benchmark(name="Test")
        checkpoint.rating = [
            SchemaOrgRating(
                name="literal_trait",
                bestRating=2,
                worstRating=0,
                additionalType="karenina:GlobalLLMRubricTrait",
                additionalProperty=[
                    SchemaOrgPropertyValue(name="deep_judgment_enabled", value=True),
                    SchemaOrgPropertyValue(name="deep_judgment_excerpt_enabled", value=True),
                    SchemaOrgPropertyValue(name="kind", value="literal"),
                    SchemaOrgPropertyValue(name="higher_is_better", value=True),
                ],
            )
        ]

        strip_deep_judgment_config_from_checkpoint(checkpoint)

        remaining_names = [p.name for p in checkpoint.rating[0].additionalProperty]
        assert "deep_judgment_enabled" not in remaining_names
        assert "deep_judgment_excerpt_enabled" not in remaining_names
        assert "kind" in remaining_names
        assert "higher_is_better" in remaining_names

    def test_strip_removes_dj_from_question_specific_literal_llm_trait(self) -> None:
        """DJ properties should be stripped from QuestionSpecificLLMRubricTrait."""
        checkpoint = create_jsonld_benchmark(name="Test")
        item = SchemaOrgDataFeedItem(
            dateCreated=datetime.now().isoformat(),
            dateModified=datetime.now().isoformat(),
            item=SchemaOrgQuestion(
                text="Test?",
                acceptedAnswer=SchemaOrgAnswer(text="A"),
                hasPart=SchemaOrgSoftwareSourceCode(name="A", text="class A: pass"),
                rating=[
                    SchemaOrgRating(
                        name="literal_trait",
                        bestRating=2,
                        worstRating=0,
                        additionalType="karenina:QuestionSpecificLLMRubricTrait",
                        additionalProperty=[
                            SchemaOrgPropertyValue(name="deep_judgment_enabled", value=True),
                            SchemaOrgPropertyValue(name="deep_judgment_excerpt_enabled", value=True),
                            SchemaOrgPropertyValue(name="kind", value="literal"),
                        ],
                    )
                ],
            ),
        )
        checkpoint.dataFeedElement.append(item)

        strip_deep_judgment_config_from_checkpoint(checkpoint)

        remaining = [p.name for p in checkpoint.dataFeedElement[0].item.rating[0].additionalProperty]
        assert "deep_judgment_enabled" not in remaining
        assert "deep_judgment_excerpt_enabled" not in remaining


# =============================================================================
# Issue 172: DJ config defaults
# =============================================================================


@pytest.mark.unit
class TestIssue172DJConfigDefaults:
    """Deep judgment config must be preserved by default and deserialization defaults must match model."""

    def test_excerpt_enabled_defaults_to_true_when_not_in_checkpoint(self) -> None:
        """When DJ properties are absent (old checkpoint), excerpt_enabled should be True (model default)."""
        rating = SchemaOrgRating(
            name="test_trait",
            bestRating=1,
            worstRating=0,
            additionalType="karenina:GlobalRubricTrait",
            additionalProperty=[
                SchemaOrgPropertyValue(name="higher_is_better", value=True),
            ],
        )

        trait = convert_rating_to_rubric_trait(rating)

        assert isinstance(trait, LLMRubricTrait)
        assert trait.deep_judgment_excerpt_enabled is True

    def test_save_default_is_true(self) -> None:
        """BenchmarkBase.save() should default to preserving DJ config."""
        from karenina.benchmark.core.base import BenchmarkBase

        sig = inspect.signature(BenchmarkBase.save)
        default = sig.parameters["save_deep_judgment_config"].default
        assert default is True, f"Expected default True, got {default}"


# =============================================================================
# Issue 079: Summary field must survive round-trip for all trait types
# =============================================================================


@pytest.mark.unit
class TestIssue079SummaryRoundTrip:
    """Summary field must survive serialize/deserialize round-trip for all trait types."""

    def test_llm_boolean_summary_roundtrip(self) -> None:
        """LLMRubricTrait boolean summary survives round-trip."""
        trait = LLMRubricTrait(
            name="test_llm",
            description="Test LLM trait",
            summary="LLM summary text",
            kind="boolean",
        )

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, LLMRubricTrait)
        assert restored.summary == "LLM summary text"

    def test_llm_score_summary_roundtrip(self) -> None:
        """Score-kind LLMRubricTrait summary survives round-trip."""
        trait = LLMRubricTrait(
            name="test_score",
            description="Score trait",
            summary="Score summary",
            kind="score",
            min_score=1,
            max_score=10,
        )

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, LLMRubricTrait)
        assert restored.summary == "Score summary"

    def test_llm_literal_summary_roundtrip(self) -> None:
        """Literal-kind LLMRubricTrait summary survives round-trip."""
        trait = LLMRubricTrait(
            name="test_literal",
            description="Literal trait",
            summary="Literal summary",
            kind="literal",
            classes={"0": "bad", "1": "ok", "2": "good"},
        )

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, LLMRubricTrait)
        assert restored.summary == "Literal summary"

    def test_regex_summary_roundtrip(self) -> None:
        """RegexRubricTrait summary survives round-trip."""
        trait = RegexRubricTrait(
            name="test_regex",
            description="Test regex",
            summary="Regex summary text",
            pattern=r"\d+",
        )

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, RegexRubricTrait)
        assert restored.summary == "Regex summary text"

    def test_callable_summary_roundtrip(self) -> None:
        """CallableRubricTrait summary survives round-trip."""
        trait = CallableRubricTrait(
            name="test_callable",
            description="Test callable",
            summary="Callable summary text",
            kind="boolean",
            callable_code=b"def evaluate(response): return True",
        )

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, CallableRubricTrait)
        assert restored.summary == "Callable summary text"

    def test_metric_summary_roundtrip(self) -> None:
        """MetricRubricTrait summary survives round-trip."""
        trait = MetricRubricTrait(
            name="test_metric",
            description="Test metric",
            summary="Metric summary text",
            metrics=["precision"],
            tp_instructions=["Check for correct items"],
        )

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, MetricRubricTrait)
        assert restored.summary == "Metric summary text"

    def test_agentic_summary_roundtrip(self) -> None:
        """AgenticRubricTrait summary survives round-trip."""
        trait = AgenticRubricTrait(
            name="test_agentic",
            description="Test agentic",
            summary="Agentic summary text",
            kind="boolean",
        )

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, AgenticRubricTrait)
        assert restored.summary == "Agentic summary text"

    def test_none_summary_stays_none(self) -> None:
        """Traits with summary=None should stay None after round-trip."""
        trait = LLMRubricTrait(
            name="no_summary",
            description="No summary",
            kind="boolean",
        )

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, LLMRubricTrait)
        assert restored.summary is None


# =============================================================================
# Issue 173: min_score/max_score must survive round-trip
# =============================================================================


@pytest.mark.unit
class TestIssue173MinMaxScoreRoundTrip:
    """min_score and max_score must survive serialize/deserialize round-trip."""

    def test_llm_boolean_preserves_default_min_max(self) -> None:
        """Boolean LLMRubricTrait should preserve Pydantic default min_score=1, max_score=5."""
        trait = LLMRubricTrait(
            name="bool_trait",
            description="Boolean trait",
            kind="boolean",
        )
        assert trait.min_score == 1
        assert trait.max_score == 5

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, LLMRubricTrait)
        assert restored.min_score == 1, f"Expected min_score=1, got {restored.min_score}"
        assert restored.max_score == 5, f"Expected max_score=5, got {restored.max_score}"

    def test_llm_score_preserves_custom_min_max(self) -> None:
        """Score LLMRubricTrait should preserve custom min_score/max_score."""
        trait = LLMRubricTrait(
            name="score_trait",
            description="Score trait",
            kind="score",
            min_score=0,
            max_score=10,
        )

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, LLMRubricTrait)
        assert restored.min_score == 0
        assert restored.max_score == 10

    def test_agentic_boolean_preserves_default_min_max(self) -> None:
        """Boolean AgenticRubricTrait should preserve Pydantic default min_score=1, max_score=5."""
        trait = AgenticRubricTrait(
            name="agentic_bool",
            description="Agentic boolean",
            kind="boolean",
        )
        assert trait.min_score == 1
        assert trait.max_score == 5

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, AgenticRubricTrait)
        assert restored.min_score == 1, f"Expected min_score=1, got {restored.min_score}"
        assert restored.max_score == 5, f"Expected max_score=5, got {restored.max_score}"

    def test_agentic_score_preserves_custom_min_max(self) -> None:
        """Score AgenticRubricTrait should preserve custom min_score/max_score."""
        trait = AgenticRubricTrait(
            name="agentic_score",
            description="Agentic score",
            kind="score",
            min_score=0,
            max_score=100,
        )

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, AgenticRubricTrait)
        assert restored.min_score == 0
        assert restored.max_score == 100

    def test_llm_literal_min_max_auto_derived(self) -> None:
        """Literal LLMRubricTrait min_score/max_score auto-derived from classes survives round-trip."""
        trait = LLMRubricTrait(
            name="literal_trait",
            description="Literal trait",
            kind="literal",
            classes={"0": "bad", "1": "ok", "2": "good"},
        )
        assert trait.min_score == 0
        assert trait.max_score == 2

        rating = convert_rubric_trait_to_rating(trait, "global")
        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, LLMRubricTrait)
        assert restored.min_score == 0
        assert restored.max_score == 2

    def test_old_checkpoint_without_min_max_gets_pydantic_defaults(self) -> None:
        """Old checkpoints without min_score/max_score get Pydantic defaults, not bestRating/worstRating."""
        rating = SchemaOrgRating(
            name="old_score_trait",
            description="Old trait",
            bestRating=10.0,
            worstRating=1.0,
            additionalType="karenina:GlobalRubricTrait",
            additionalProperty=[
                SchemaOrgPropertyValue(name="higher_is_better", value=True),
            ],
        )

        restored = convert_rating_to_rubric_trait(rating)

        assert isinstance(restored, LLMRubricTrait)
        # No fallback to bestRating/worstRating; Pydantic defaults apply
        assert restored.min_score == 1
        assert restored.max_score == 5
