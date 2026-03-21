"""Round-trip tests for DynamicRubric checkpoint serialization.

Tests verify that DynamicRubric instances survive the serialize/deserialize
cycle through the JSON-LD checkpoint format, preserving all trait types and
the ``summary`` field essential for concept presence checking.
"""

import pytest

from karenina.schemas.entities import LLMRubricTrait, RegexTrait
from karenina.schemas.entities.rubric import DynamicRubric
from karenina.utils.checkpoint import (
    add_global_dynamic_rubric_to_benchmark,
    create_jsonld_benchmark,
    extract_global_dynamic_rubric_from_benchmark,
)
from karenina.utils.checkpoint_trait_converters import (
    convert_dynamic_rubric_to_ratings,
    convert_ratings_to_dynamic_rubric,
)

# ── Converter-level round-trip tests ──────────────────────────────────


@pytest.mark.unit
class TestDynamicRubricConverterRoundTrip:
    """Tests for convert_dynamic_rubric_to_ratings / convert_ratings_to_dynamic_rubric."""

    def test_llm_trait_roundtrip_with_summary(self) -> None:
        """LLM trait with summary survives serialization round-trip."""
        original = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="safety",
                    description="Checks response safety.",
                    summary="Response safety assessment",
                    kind="boolean",
                ),
            ],
        )

        ratings = convert_dynamic_rubric_to_ratings(original, "global")
        restored = convert_ratings_to_dynamic_rubric(ratings)

        assert len(restored.llm_traits) == 1
        trait = restored.llm_traits[0]
        assert trait.name == "safety"
        assert trait.description == "Checks response safety."
        assert trait.summary == "Response safety assessment"
        assert trait.kind == "boolean"

    def test_llm_score_trait_roundtrip(self) -> None:
        """LLM score trait preserves min_score, max_score, and summary."""
        original = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="clarity",
                    description="Rate the clarity of the response.",
                    summary="Clarity rating",
                    kind="score",
                    min_score=1,
                    max_score=10,
                    higher_is_better=True,
                ),
            ],
        )

        ratings = convert_dynamic_rubric_to_ratings(original, "global")
        restored = convert_ratings_to_dynamic_rubric(ratings)

        assert len(restored.llm_traits) == 1
        trait = restored.llm_traits[0]
        assert trait.name == "clarity"
        assert trait.summary == "Clarity rating"
        assert trait.kind == "score"
        assert trait.min_score == 1
        assert trait.max_score == 10

    def test_regex_trait_roundtrip_with_summary(self) -> None:
        """Regex trait with summary survives serialization round-trip."""
        original = DynamicRubric(
            regex_traits=[
                RegexTrait(
                    name="has_citations",
                    description="Response includes numbered citations.",
                    summary="Citation presence",
                    pattern=r"\[\d+\]",
                    case_sensitive=True,
                    invert_result=False,
                ),
            ],
        )

        ratings = convert_dynamic_rubric_to_ratings(original, "global")
        restored = convert_ratings_to_dynamic_rubric(ratings)

        assert len(restored.regex_traits) == 1
        trait = restored.regex_traits[0]
        assert trait.name == "has_citations"
        assert trait.summary == "Citation presence"
        assert trait.pattern == r"\[\d+\]"
        assert trait.case_sensitive is True

    def test_mixed_traits_roundtrip(self) -> None:
        """DynamicRubric with multiple trait types survives round-trip."""
        original = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="safety",
                    description="Checks response safety.",
                    summary="Safety check",
                    kind="boolean",
                ),
                LLMRubricTrait(
                    name="depth",
                    description="Evaluates analytical depth.",
                    summary="Depth of analysis",
                    kind="score",
                    min_score=1,
                    max_score=5,
                ),
            ],
            regex_traits=[
                RegexTrait(
                    name="has_citations",
                    description="Response includes citations.",
                    summary="Citation presence",
                    pattern=r"\[\d+\]",
                ),
            ],
        )

        ratings = convert_dynamic_rubric_to_ratings(original, "global")
        restored = convert_ratings_to_dynamic_rubric(ratings)

        assert len(restored.llm_traits) == 2
        assert len(restored.regex_traits) == 1
        assert restored.llm_traits[0].name == "safety"
        assert restored.llm_traits[0].summary == "Safety check"
        assert restored.llm_traits[1].name == "depth"
        assert restored.llm_traits[1].summary == "Depth of analysis"
        assert restored.regex_traits[0].name == "has_citations"
        assert restored.regex_traits[0].summary == "Citation presence"

    def test_trait_without_summary_roundtrip(self) -> None:
        """Trait with only description (no summary) roundtrips with summary=None."""
        original = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="conciseness",
                    description="Evaluates whether the response is concise.",
                    kind="boolean",
                ),
            ],
        )

        ratings = convert_dynamic_rubric_to_ratings(original, "global")
        restored = convert_ratings_to_dynamic_rubric(ratings)

        assert len(restored.llm_traits) == 1
        trait = restored.llm_traits[0]
        assert trait.name == "conciseness"
        assert trait.summary is None
        assert trait.description == "Evaluates whether the response is concise."

    def test_question_specific_type_tag(self) -> None:
        """Question-specific rubric_type uses the correct @type discriminator."""
        dr = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="safety",
                    description="Checks response safety.",
                    summary="Safety",
                    kind="boolean",
                ),
            ],
        )

        ratings = convert_dynamic_rubric_to_ratings(dr, "question-specific")
        assert len(ratings) == 1
        assert ratings[0].additionalType == "karenina:QuestionSpecificDynamicRubricTrait"

    def test_global_type_tag(self) -> None:
        """Global rubric_type uses the correct @type discriminator."""
        dr = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="safety",
                    description="Checks response safety.",
                    summary="Safety",
                    kind="boolean",
                ),
            ],
        )

        ratings = convert_dynamic_rubric_to_ratings(dr, "global")
        assert len(ratings) == 1
        assert ratings[0].additionalType == "karenina:GlobalDynamicRubricTrait"


# ── Benchmark-level round-trip tests ──────────────────────────────────


@pytest.mark.unit
class TestDynamicRubricBenchmarkRoundTrip:
    """Tests for add/extract global dynamic rubric through a full benchmark checkpoint."""

    def test_add_and_extract_global_dynamic_rubric(self) -> None:
        """DynamicRubric written to a benchmark can be read back identically."""
        benchmark = create_jsonld_benchmark(name="test_benchmark")

        original = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="safety",
                    description="Checks response safety.",
                    summary="Response safety assessment",
                    kind="boolean",
                ),
                LLMRubricTrait(
                    name="depth",
                    description="Rate analytical depth.",
                    summary="Depth rating",
                    kind="score",
                    min_score=1,
                    max_score=5,
                ),
            ],
            regex_traits=[
                RegexTrait(
                    name="has_citations",
                    description="Response includes citations.",
                    summary="Citation check",
                    pattern=r"\[\d+\]",
                ),
            ],
        )

        add_global_dynamic_rubric_to_benchmark(benchmark, original)
        restored = extract_global_dynamic_rubric_from_benchmark(benchmark)

        assert restored is not None
        assert len(restored.llm_traits) == 2
        assert len(restored.regex_traits) == 1

        # Verify LLM traits
        assert restored.llm_traits[0].name == "safety"
        assert restored.llm_traits[0].summary == "Response safety assessment"
        assert restored.llm_traits[0].kind == "boolean"

        assert restored.llm_traits[1].name == "depth"
        assert restored.llm_traits[1].summary == "Depth rating"
        assert restored.llm_traits[1].kind == "score"
        assert restored.llm_traits[1].min_score == 1
        assert restored.llm_traits[1].max_score == 5

        # Verify regex trait
        assert restored.regex_traits[0].name == "has_citations"
        assert restored.regex_traits[0].summary == "Citation check"
        assert restored.regex_traits[0].pattern == r"\[\d+\]"

    def test_extract_returns_none_when_no_dynamic_rubric(self) -> None:
        """Extracting from a benchmark with no dynamic rubric returns None."""
        benchmark = create_jsonld_benchmark(name="empty_benchmark")

        result = extract_global_dynamic_rubric_from_benchmark(benchmark)
        assert result is None

    def test_dynamic_rubric_coexists_with_regular_rubric(self) -> None:
        """Dynamic rubric ratings coexist with regular rubric ratings."""
        from karenina.utils.checkpoint import (
            add_global_rubric_to_benchmark,
            extract_global_rubric_from_benchmark,
        )

        benchmark = create_jsonld_benchmark(name="mixed_benchmark")

        # Add a regular global rubric
        regular_trait = LLMRubricTrait(
            name="regular_safety",
            description="Regular safety check.",
            kind="boolean",
        )
        add_global_rubric_to_benchmark(benchmark, [regular_trait])

        # Add a dynamic rubric
        dynamic = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="dynamic_clarity",
                    description="Dynamic clarity check.",
                    summary="Clarity concept",
                    kind="boolean",
                ),
            ],
        )
        add_global_dynamic_rubric_to_benchmark(benchmark, dynamic)

        # Both should be extractable independently
        regular_traits = extract_global_rubric_from_benchmark(benchmark)
        dynamic_rubric = extract_global_dynamic_rubric_from_benchmark(benchmark)

        assert regular_traits is not None
        assert len(regular_traits) == 1
        assert regular_traits[0].name == "regular_safety"

        assert dynamic_rubric is not None
        assert len(dynamic_rubric.llm_traits) == 1
        assert dynamic_rubric.llm_traits[0].name == "dynamic_clarity"
        assert dynamic_rubric.llm_traits[0].summary == "Clarity concept"

    def test_empty_dynamic_rubric_roundtrip(self) -> None:
        """An empty DynamicRubric produces no ratings and extracts as None."""
        benchmark = create_jsonld_benchmark(name="empty_dr_benchmark")

        empty_dr = DynamicRubric()
        add_global_dynamic_rubric_to_benchmark(benchmark, empty_dr)

        # No dynamic ratings should have been added
        restored = extract_global_dynamic_rubric_from_benchmark(benchmark)
        assert restored is None

    def test_llm_literal_trait_roundtrip(self) -> None:
        """LLM literal trait with classes survives checkpoint round-trip."""
        original = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="severity",
                    description="Rate the severity level.",
                    summary="Severity classification",
                    kind="literal",
                    classes={"low": "Minor issue", "medium": "Moderate issue", "high": "Critical issue"},
                ),
            ],
        )

        benchmark = create_jsonld_benchmark(name="literal_benchmark")
        add_global_dynamic_rubric_to_benchmark(benchmark, original)
        restored = extract_global_dynamic_rubric_from_benchmark(benchmark)

        assert restored is not None
        assert len(restored.llm_traits) == 1
        trait = restored.llm_traits[0]
        assert trait.name == "severity"
        assert trait.summary == "Severity classification"
        assert trait.kind == "literal"
        assert trait.classes == {"low": "Minor issue", "medium": "Moderate issue", "high": "Critical issue"}
