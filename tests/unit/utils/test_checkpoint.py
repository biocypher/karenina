"""Unit tests for checkpoint conversion utilities.

Tests cover:
- generate_question_id() - MD5-based question ID generation
- generate_template_id() - Template hash ID generation
- convert_rubric_trait_to_rating() - Trait to SchemaOrgRating conversion
- convert_rating_to_rubric_trait() - SchemaOrgRating to trait conversion
- create_jsonld_benchmark() - Empty benchmark creation
- validate_jsonld_benchmark() - Benchmark validation
- BenchmarkConversionError exception
"""

from datetime import datetime

import pytest

from karenina.schemas.domain import (
    CallableTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexTrait,
)
from karenina.utils.checkpoint import (
    BenchmarkConversionError,
    convert_rating_to_rubric_trait,
    convert_rubric_trait_to_rating,
    create_jsonld_benchmark,
    generate_question_id,
    generate_template_id,
    validate_jsonld_benchmark,
)

# =============================================================================
# generate_question_id() Tests
# =============================================================================


@pytest.mark.unit
def test_generate_question_id_basic() -> None:
    """Test question ID generation for basic question."""
    question = "What is the capital of France?"
    result = generate_question_id(question)

    assert result.startswith("urn:uuid:question-")
    assert "capital" in result.lower()
    # Should end with 8-char MD5 hex
    parts = result.split("-")
    assert len(parts[-1]) == 8


@pytest.mark.unit
def test_generate_question_id_deterministic() -> None:
    """Test that same question produces same ID."""
    question = "Test question?"
    id1 = generate_question_id(question)
    id2 = generate_question_id(question)

    assert id1 == id2


@pytest.mark.unit
def test_generate_question_id_different_questions() -> None:
    """Test that different questions produce different IDs."""
    id1 = generate_question_id("Question one?")
    id2 = generate_question_id("Question two?")

    assert id1 != id2


@pytest.mark.unit
def test_generate_question_id_with_punctuation() -> None:
    """Test question ID removes some punctuation from prefix."""
    question = "What's BCL2's role? Check: yes."
    result = generate_question_id(question)

    # The function removes ?, ., but keeps '
    assert "bcl2" in result.lower()
    # Hash should be present
    parts = result.split("-")
    assert len(parts[-1]) == 8


@pytest.mark.unit
def test_generate_question_id_long_question() -> None:
    """Test question ID with very long question."""
    # Create a long question (> 50 chars)
    long_question = "a" * 100 + "?"
    result = generate_question_id(long_question)

    # Should still produce valid ID with hash
    assert result.startswith("urn:uuid:question-")
    parts = result.split("-")
    assert len(parts[-1]) == 8  # 8-char hash


@pytest.mark.unit
def test_generate_question_id_empty_question() -> None:
    """Test question ID with empty string."""
    result = generate_question_id("")
    assert result.startswith("urn:uuid:question-")
    # Hash should still be present
    parts = result.split("-")
    assert len(parts[2]) == 8


# =============================================================================
# generate_template_id() Tests
# =============================================================================


@pytest.mark.unit
def test_generate_template_id_basic() -> None:
    """Test template ID generation for basic template."""
    template = "class Answer(BaseAnswer):\n    value: str"
    result = generate_template_id(template)

    # Should be 32-char MD5 hash
    assert len(result) == 32
    assert all(c in "0123456789abcdef" for c in result)


@pytest.mark.unit
def test_generate_template_id_deterministic() -> None:
    """Test that same template produces same ID."""
    template = "class Answer(BaseAnswer):\n    value: str"
    id1 = generate_template_id(template)
    id2 = generate_template_id(template)

    assert id1 == id2


@pytest.mark.unit
def test_generate_template_id_different_templates() -> None:
    """Test that different templates produce different IDs."""
    id1 = generate_template_id("class Answer(BaseAnswer):\n    value: str")
    id2 = generate_template_id("class Answer(BaseAnswer):\n    value: int")

    assert id1 != id2


@pytest.mark.unit
def test_generate_template_id_none_returns_no_template() -> None:
    """Test that None template returns 'no_template'."""
    result = generate_template_id(None)
    assert result == "no_template"


@pytest.mark.unit
def test_generate_template_id_empty_returns_no_template() -> None:
    """Test that empty template returns 'no_template'."""
    result = generate_template_id("")
    assert result == "no_template"


@pytest.mark.unit
def test_generate_template_id_whitespace_only_returns_no_template() -> None:
    """Test that whitespace-only template returns 'no_template'."""
    result = generate_template_id("   \n\t  ")
    assert result == "no_template"


@pytest.mark.unit
def test_generate_template_id_strips_whitespace() -> None:
    """Test that leading/trailing whitespace is stripped before hashing."""
    template1 = "class Answer(BaseAnswer):\n    value: str"
    template2 = "  class Answer(BaseAnswer):\n    value: str  "

    id1 = generate_template_id(template1)
    id2 = generate_template_id(template2)

    assert id1 == id2


# =============================================================================
# convert_rubric_trait_to_rating() - RegexTrait Tests
# =============================================================================


@pytest.mark.unit
def test_convert_regex_trait_to_rating_global() -> None:
    """Test converting global RegexTrait to SchemaOrgRating."""
    trait = RegexTrait(
        name="has_citation",
        pattern=r"\[\d+\]",
        case_sensitive=True,
        invert_result=False,
        higher_is_better=True,
    )

    rating = convert_rubric_trait_to_rating(trait, "global")

    assert rating.name == "has_citation"
    assert rating.bestRating == 1
    assert rating.worstRating == 0
    assert rating.additionalType == "GlobalRegexTrait"

    # Check additionalProperty
    props = {prop.name: prop.value for prop in (rating.additionalProperty or [])}
    assert props["pattern"] == r"\[\d+\]"
    assert props["case_sensitive"] is True
    assert props["invert_result"] is False
    assert props["higher_is_better"] is True


@pytest.mark.unit
def test_convert_regex_trait_to_rating_question_specific() -> None:
    """Test converting question-specific RegexTrait to SchemaOrgRating."""
    trait = RegexTrait(
        name="has_url",
        pattern=r"https?://\S+",
        higher_is_better=True,
    )

    rating = convert_rubric_trait_to_rating(trait, "question-specific")

    assert rating.additionalType == "QuestionSpecificRegexTrait"


# =============================================================================
# convert_rubric_trait_to_rating() - CallableTrait Tests
# =============================================================================


@pytest.mark.unit
def test_convert_callable_trait_boolean_to_rating() -> None:
    """Test converting boolean CallableTrait to SchemaOrgRating."""
    import cloudpickle

    def func(text):
        return "keyword" in text
    code = cloudpickle.dumps(func)

    trait = CallableTrait(
        name="has_keyword",
        kind="boolean",
        callable_code=code,
        invert_result=False,
        higher_is_better=True,
    )

    rating = convert_rubric_trait_to_rating(trait, "global")

    assert rating.name == "has_keyword"
    assert rating.bestRating == 1.0
    assert rating.worstRating == 0.0
    assert rating.additionalType == "GlobalCallableTrait"

    # Check additionalProperty
    props = {prop.name: prop.value for prop in (rating.additionalProperty or [])}
    assert "callable_code" in props
    assert props["kind"] == "boolean"
    assert props["invert_result"] is False


@pytest.mark.unit
def test_convert_callable_trait_score_to_rating() -> None:
    """Test converting score CallableTrait to SchemaOrgRating."""
    import cloudpickle

    def func(text):
        return len(text)
    code = cloudpickle.dumps(func)

    trait = CallableTrait(
        name="length_score",
        kind="score",
        callable_code=code,
        min_score=0,
        max_score=100,
        higher_is_better=True,
    )

    rating = convert_rubric_trait_to_rating(trait, "global")

    assert rating.bestRating == 100.0
    assert rating.worstRating == 0.0

    # Check additionalProperty
    props = {prop.name: prop.value for prop in (rating.additionalProperty or [])}
    assert props["min_score"] == 0
    assert props["max_score"] == 100


@pytest.mark.unit
def test_convert_callable_trait_question_specific() -> None:
    """Test converting question-specific CallableTrait to SchemaOrgRating."""
    import cloudpickle

    trait = CallableTrait(
        name="custom_check",
        kind="boolean",
        callable_code=cloudpickle.dumps(lambda _: True),
    )

    rating = convert_rubric_trait_to_rating(trait, "question-specific")

    assert rating.additionalType == "QuestionSpecificCallableTrait"


# =============================================================================
# convert_rubric_trait_to_rating() - LLMRubricTrait Tests
# =============================================================================


@pytest.mark.unit
def test_convert_llm_trait_boolean_to_rating() -> None:
    """Test converting boolean LLMRubricTrait to SchemaOrgRating."""
    trait = LLMRubricTrait(
        name="clarity",
        kind="boolean",
        description="Is the response clear?",
        higher_is_better=True,
    )

    rating = convert_rubric_trait_to_rating(trait, "global")

    assert rating.name == "clarity"
    assert rating.bestRating == 1
    assert rating.worstRating == 0
    assert rating.additionalType == "GlobalRubricTrait"


@pytest.mark.unit
def test_convert_llm_trait_score_to_rating() -> None:
    """Test converting score LLMRubricTrait to SchemaOrgRating."""
    trait = LLMRubricTrait(
        name="quality",
        kind="score",
        min_score=1,
        max_score=5,
        description="Quality rating",
        higher_is_better=True,
    )

    rating = convert_rubric_trait_to_rating(trait, "global")

    assert rating.bestRating == 5
    assert rating.worstRating == 1


@pytest.mark.unit
def test_convert_llm_trait_with_deep_judgment_settings() -> None:
    """Test converting LLMRubricTrait with deep judgment settings."""
    trait = LLMRubricTrait(
        name="accuracy",
        kind="boolean",
        deep_judgment_enabled=True,
        deep_judgment_excerpt_enabled=True,
        deep_judgment_search_enabled=True,
        deep_judgment_max_excerpts=3,
        deep_judgment_fuzzy_match_threshold=0.85,
        higher_is_better=True,
    )

    rating = convert_rubric_trait_to_rating(trait, "global")

    # Check deep judgment settings in additionalProperty
    props = {prop.name: prop.value for prop in (rating.additionalProperty or [])}
    assert props["deep_judgment_enabled"] is True
    assert props["deep_judgment_excerpt_enabled"] is True
    assert props["deep_judgment_search_enabled"] is True
    assert props["deep_judgment_max_excerpts"] == 3
    assert props["deep_judgment_fuzzy_match_threshold"] == 0.85


# =============================================================================
# convert_rubric_trait_to_rating() - MetricRubricTrait Tests
# =============================================================================


@pytest.mark.unit
def test_convert_metric_trait_to_rating() -> None:
    """Test converting MetricRubricTrait to SchemaOrgRating."""
    trait = MetricRubricTrait(
        name="entity_extraction",
        evaluation_mode="tp_only",
        metrics=["precision", "recall"],
        tp_instructions=["mitochondria", "apoptosis"],
        tn_instructions=["nucleus", "ribosome"],
        repeated_extraction=True,
    )

    rating = convert_rubric_trait_to_rating(trait, "global")

    assert rating.name == "entity_extraction"
    assert rating.bestRating == 1.0
    assert rating.worstRating == 0.0
    assert rating.additionalType == "GlobalMetricRubricTrait"

    # Check additionalProperty
    props = {prop.name: prop.value for prop in (rating.additionalProperty or [])}
    assert props["metrics"] == ["precision", "recall"]
    assert props["evaluation_mode"] == "tp_only"
    assert props["tp_instructions"] == ["mitochondria", "apoptosis"]
    assert props["tn_instructions"] == ["nucleus", "ribosome"]
    assert props["repeated_extraction"] is True


@pytest.mark.unit
def test_convert_metric_trait_question_specific() -> None:
    """Test converting question-specific MetricRubricTrait to SchemaOrgRating."""
    trait = MetricRubricTrait(
        name="local_entity_check",
        evaluation_mode="full_matrix",
        metrics=["f1"],
        tp_instructions=["entity1", "entity2"],
        tn_instructions=["other1", "other2"],
    )

    rating = convert_rubric_trait_to_rating(trait, "question-specific")

    assert rating.additionalType == "QuestionSpecificMetricRubricTrait"


# =============================================================================
# convert_rating_to_rubric_trait() Tests
# =============================================================================


@pytest.mark.unit
def test_convert_rating_to_regex_trait() -> None:
    """Test converting SchemaOrgRating back to RegexTrait."""
    from karenina.schemas.checkpoint import (
        SchemaOrgPropertyValue,
        SchemaOrgRating,
    )

    rating = SchemaOrgRating(
        name="has_email",
        description="Has email pattern",
        bestRating=1,
        worstRating=0,
        additionalType="GlobalRegexTrait",
        additionalProperty=[
            SchemaOrgPropertyValue(name="pattern", value=r"\S+@\S+"),
            SchemaOrgPropertyValue(name="case_sensitive", value=True),
            SchemaOrgPropertyValue(name="invert_result", value=False),
            SchemaOrgPropertyValue(name="higher_is_better", value=True),
        ],
    )

    trait = convert_rating_to_rubric_trait(rating)

    assert isinstance(trait, RegexTrait)
    assert trait.name == "has_email"
    assert trait.pattern == r"\S+@\S+"
    assert trait.case_sensitive is True
    assert trait.invert_result is False
    assert trait.higher_is_better is True


@pytest.mark.unit
def test_convert_rating_to_callable_trait_boolean() -> None:
    """Test converting SchemaOrgRating back to boolean CallableTrait."""
    import base64

    import cloudpickle

    from karenina.schemas.checkpoint import (
        SchemaOrgPropertyValue,
        SchemaOrgRating,
    )

    def func(text):
        return len(text) > 10
    code = cloudpickle.dumps(func)
    code_b64 = base64.b64encode(code).decode("utf-8")

    rating = SchemaOrgRating(
        name="long_text",
        bestRating=1.0,
        worstRating=0.0,
        additionalType="GlobalCallableTrait",
        additionalProperty=[
            SchemaOrgPropertyValue(name="callable_code", value=code_b64),
            SchemaOrgPropertyValue(name="kind", value="boolean"),
            SchemaOrgPropertyValue(name="invert_result", value=False),
            SchemaOrgPropertyValue(name="higher_is_better", value=True),
        ],
    )

    trait = convert_rating_to_rubric_trait(rating)

    assert isinstance(trait, CallableTrait)
    assert trait.name == "long_text"
    assert trait.kind == "boolean"
    assert trait.invert_result is False


@pytest.mark.unit
def test_convert_rating_to_callable_trait_score() -> None:
    """Test converting SchemaOrgRating back to score CallableTrait."""
    import base64

    import cloudpickle

    from karenina.schemas.checkpoint import (
        SchemaOrgPropertyValue,
        SchemaOrgRating,
    )

    def func(text):
        return min(len(text), 100)
    code = cloudpickle.dumps(func)
    code_b64 = base64.b64encode(code).decode("utf-8")

    rating = SchemaOrgRating(
        name="length_clipped",
        bestRating=100.0,
        worstRating=0.0,
        additionalType="QuestionSpecificCallableTrait",
        additionalProperty=[
            SchemaOrgPropertyValue(name="callable_code", value=code_b64),
            SchemaOrgPropertyValue(name="kind", value="score"),
            SchemaOrgPropertyValue(name="min_score", value=0),
            SchemaOrgPropertyValue(name="max_score", value=100),
        ],
    )

    trait = convert_rating_to_rubric_trait(rating)

    assert isinstance(trait, CallableTrait)
    assert trait.kind == "score"
    assert trait.min_score == 0
    assert trait.max_score == 100


@pytest.mark.unit
def test_convert_rating_to_metric_trait() -> None:
    """Test converting SchemaOrgRating back to MetricRubricTrait."""
    import json

    from karenina.schemas.checkpoint import (
        SchemaOrgPropertyValue,
        SchemaOrgRating,
    )

    rating = SchemaOrgRating(
        name="entity_check",
        bestRating=1.0,
        worstRating=0.0,
        additionalType="GlobalMetricRubricTrait",
        additionalProperty=[
            SchemaOrgPropertyValue(name="metrics", value=json.dumps(["precision", "recall"])),
            SchemaOrgPropertyValue(name="evaluation_mode", value="tp_only"),
            SchemaOrgPropertyValue(name="tp_instructions", value=["entity1", "entity2"]),
            SchemaOrgPropertyValue(name="tn_instructions", value=["other1", "other2"]),
            SchemaOrgPropertyValue(name="repeated_extraction", value=True),
        ],
    )

    trait = convert_rating_to_rubric_trait(rating)

    assert isinstance(trait, MetricRubricTrait)
    assert trait.name == "entity_check"
    assert trait.evaluation_mode == "tp_only"
    assert trait.metrics == ["precision", "recall"]
    assert trait.tp_instructions == ["entity1", "entity2"]
    assert trait.tn_instructions == ["other1", "other2"]
    assert trait.repeated_extraction is True


@pytest.mark.unit
def test_convert_rating_to_llm_trait_boolean() -> None:
    """Test converting SchemaOrgRating back to boolean LLMRubricTrait."""
    from karenina.schemas.checkpoint import (
        SchemaOrgPropertyValue,
        SchemaOrgRating,
    )

    rating = SchemaOrgRating(
        name="safety",
        description="Safety check",
        bestRating=1,
        worstRating=0,
        additionalType="GlobalRubricTrait",
        additionalProperty=[
            SchemaOrgPropertyValue(name="deep_judgment_enabled", value=True),
            SchemaOrgPropertyValue(name="higher_is_better", value=True),
        ],
    )

    trait = convert_rating_to_rubric_trait(rating)

    assert isinstance(trait, LLMRubricTrait)
    assert trait.name == "safety"
    assert trait.kind == "boolean"
    assert trait.deep_judgment_enabled is True


@pytest.mark.unit
def test_convert_rating_to_llm_trait_score() -> None:
    """Test converting SchemaOrgRating back to score LLMRubricTrait."""
    from karenina.schemas.checkpoint import SchemaOrgRating

    rating = SchemaOrgRating(
        name="quality",
        bestRating=5,
        worstRating=1,
        additionalType="QuestionSpecificRubricTrait",
    )

    trait = convert_rating_to_rubric_trait(rating)

    assert isinstance(trait, LLMRubricTrait)
    assert trait.name == "quality"
    assert trait.kind == "score"
    assert trait.min_score == 1
    assert trait.max_score == 5


# Note: Testing unsupported trait types (ManualRubricTrait) is not possible
# because Pydantic validates the additionalType field at creation time.
# The convert_rating_to_rubric_trait function handles this case by raising
# ValueError for unsupported types, but we cannot create invalid objects
# to test this path directly.


# =============================================================================
# create_jsonld_benchmark() Tests
# =============================================================================


@pytest.mark.unit
def test_create_jsonld_benchmark_basic() -> None:
    """Test creating a basic empty benchmark."""
    benchmark = create_jsonld_benchmark(
        name="Test Benchmark",
        description="A test benchmark",
    )

    assert benchmark.name == "Test Benchmark"
    assert benchmark.description == "A test benchmark"
    assert benchmark.version == "0.1.0"
    assert benchmark.creator == "Karenina Benchmarking System"
    assert benchmark.dataFeedElement == []
    assert benchmark.rating is None


@pytest.mark.unit
def test_create_jsonld_benchmark_with_custom_params() -> None:
    """Test creating benchmark with custom parameters."""
    benchmark = create_jsonld_benchmark(
        name="Custom Benchmark",
        version="2.0.0",
        creator="Test Creator",
    )

    assert benchmark.name == "Custom Benchmark"
    assert benchmark.version == "2.0.0"
    assert benchmark.creator == "Test Creator"


@pytest.mark.unit
def test_create_jsonld_benchmark_has_ids() -> None:
    """Test that created benchmark has required IDs."""
    benchmark = create_jsonld_benchmark(name="Test")

    assert benchmark.id is not None
    assert benchmark.id.startswith("urn:uuid:karenina-checkpoint-")
    assert benchmark.type == "DataFeed"
    assert benchmark.context is not None


@pytest.mark.unit
def test_create_jsonld_benchmark_has_timestamps() -> None:
    """Test that created benchmark has timestamps."""
    benchmark = create_jsonld_benchmark(name="Test")

    assert benchmark.dateCreated is not None
    assert benchmark.dateModified is not None
    # Should be ISO format strings
    assert "T" in benchmark.dateCreated  # ISO datetime separator


@pytest.mark.unit
def test_create_jsonld_benchmark_empty_questions() -> None:
    """Test that new benchmark has empty questions list."""
    benchmark = create_jsonld_benchmark(name="Test")

    assert benchmark.dataFeedElement == []
    assert len(benchmark.dataFeedElement) == 0


# =============================================================================
# validate_jsonld_benchmark() Tests
# =============================================================================


@pytest.mark.unit
def test_validate_valid_benchmark() -> None:
    """Test validating a valid benchmark."""
    benchmark = create_jsonld_benchmark(name="Valid")

    is_valid, message = validate_jsonld_benchmark(benchmark)

    assert is_valid is True
    assert message == "Valid benchmark"


@pytest.mark.unit
def test_validate_benchmark_missing_name() -> None:
    """Test validation fails when benchmark has no name."""
    benchmark = create_jsonld_benchmark(name="Test")
    benchmark.name = ""

    is_valid, message = validate_jsonld_benchmark(benchmark)

    assert is_valid is False
    assert "name" in message.lower()


@pytest.mark.unit
def test_validate_benchmark_with_question_missing_text() -> None:
    """Test validation fails when question has no text."""
    from karenina.schemas.checkpoint import (
        SchemaOrgAnswer,
        SchemaOrgDataFeedItem,
        SchemaOrgQuestion,
        SchemaOrgSoftwareSourceCode,
    )

    benchmark = create_jsonld_benchmark(name="Test")

    # Add question with empty text
    item = SchemaOrgDataFeedItem(
        dateCreated=datetime.now().isoformat(),
        dateModified=datetime.now().isoformat(),
        item=SchemaOrgQuestion(
            text="",  # Empty text
            acceptedAnswer=SchemaOrgAnswer(text="Answer"),
            hasPart=SchemaOrgSoftwareSourceCode(name="Answer", text="class Answer: pass"),
        ),
    )
    benchmark.dataFeedElement.append(item)

    is_valid, message = validate_jsonld_benchmark(benchmark)

    assert is_valid is False
    assert "missing text" in message.lower()


@pytest.mark.unit
def test_validate_benchmark_accepts_valid_rating_types() -> None:
    """Test validation accepts all valid rating types."""
    from karenina.schemas.checkpoint import (
        SchemaOrgAnswer,
        SchemaOrgDataFeedItem,
        SchemaOrgQuestion,
        SchemaOrgRating,
        SchemaOrgSoftwareSourceCode,
    )

    benchmark = create_jsonld_benchmark(name="Test")

    # Add question with valid global rubric type
    item = SchemaOrgDataFeedItem(
        dateCreated=datetime.now().isoformat(),
        dateModified=datetime.now().isoformat(),
        item=SchemaOrgQuestion(
            text="Test question?",
            acceptedAnswer=SchemaOrgAnswer(text="Answer"),
            hasPart=SchemaOrgSoftwareSourceCode(name="Answer", text="class Answer: pass"),
            rating=[
                SchemaOrgRating(
                    name="test",
                    bestRating=1,
                    worstRating=0,
                    additionalType="GlobalRubricTrait",
                )
            ],
        ),
    )
    benchmark.dataFeedElement.append(item)

    is_valid, message = validate_jsonld_benchmark(benchmark)

    assert is_valid is True


@pytest.mark.unit
def test_validate_benchmark_with_global_rating_invalid_type() -> None:
    """Test validation fails with question-specific trait at global level."""
    from karenina.schemas.checkpoint import SchemaOrgRating

    benchmark = create_jsonld_benchmark(name="Test")

    # Add question-specific trait as global rating (invalid)
    benchmark.rating = [
        SchemaOrgRating(
            name="test",
            bestRating=1,
            worstRating=0,
            additionalType="QuestionSpecificRubricTrait",  # Invalid at global level
        )
    ]

    is_valid, message = validate_jsonld_benchmark(benchmark)

    assert is_valid is False
    assert "global" in message.lower()


# =============================================================================
# BenchmarkConversionError Tests
# =============================================================================


@pytest.mark.unit
def test_benchmark_conversion_error_is_exception() -> None:
    """Test that BenchmarkConversionError is an exception."""
    error = BenchmarkConversionError("Test error")

    assert isinstance(error, Exception)
    assert str(error) == "Test error"


@pytest.mark.unit
def test_benchmark_conversion_error_can_be_raised() -> None:
    """Test that BenchmarkConversionError can be raised and caught."""

    def raise_conversion_error():
        raise BenchmarkConversionError("Conversion failed")

    with pytest.raises(BenchmarkConversionError) as exc_info:
        raise_conversion_error()

    assert str(exc_info.value) == "Conversion failed"
