"""Unit tests for checkpoint JSON-LD schemas.

Tests cover:
- SchemaOrgPerson validation and serialization
- SchemaOrgCreativeWork validation
- SchemaOrgPropertyValue for custom metadata
- SchemaOrgRating for rubric traits
- SchemaOrgSoftwareSourceCode for templates
- SchemaOrgAnswer for raw answers
- SchemaOrgQuestion with all components
- SchemaOrgDataFeedItem for timestamped questions
- JsonLdCheckpoint with full JSON-LD structure
- SCHEMA_ORG_CONTEXT constant
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from karenina.schemas.checkpoint import (
    SCHEMA_ORG_CONTEXT,
    JsonLdCheckpoint,
    SchemaOrgAnswer,
    SchemaOrgCreativeWork,
    SchemaOrgDataFeed,
    SchemaOrgDataFeedItem,
    SchemaOrgPerson,
    SchemaOrgPropertyValue,
    SchemaOrgQuestion,
    SchemaOrgRating,
    SchemaOrgSoftwareSourceCode,
)

# =============================================================================
# SchemaOrgPerson Tests
# =============================================================================


@pytest.mark.unit
def test_schema_org_person_minimal() -> None:
    """Test SchemaOrgPerson with minimal required fields."""
    person = SchemaOrgPerson(name="Alice Smith")

    assert person.type == "Person"
    assert person.name == "Alice Smith"
    assert person.url is None
    assert person.email is None


@pytest.mark.unit
def test_schema_org_person_full() -> None:
    """Test SchemaOrgPerson with all fields."""
    person = SchemaOrgPerson(
        name="Bob Jones",
        url="https://example.com/bob",
        email="bob@example.com",
    )

    assert person.type == "Person"
    assert person.name == "Bob Jones"
    assert person.url == "https://example.com/bob"
    assert person.email == "bob@example.com"


@pytest.mark.unit
def test_schema_org_person_serialization_with_alias() -> None:
    """Test SchemaOrgPerson serializes @type correctly."""
    person = SchemaOrgPerson(name="Test")
    data = person.model_dump(by_alias=True)

    assert data["@type"] == "Person"
    assert "type" not in data


# =============================================================================
# SchemaOrgCreativeWork Tests
# =============================================================================


@pytest.mark.unit
def test_schema_org_creative_work_minimal() -> None:
    """Test SchemaOrgCreativeWork with minimal fields."""
    work = SchemaOrgCreativeWork(name="Research Paper")

    assert work.type == "CreativeWork"
    assert work.name == "Research Paper"
    assert work.url is None
    assert work.author is None
    assert work.datePublished is None


@pytest.mark.unit
def test_schema_org_creative_work_full() -> None:
    """Test SchemaOrgCreativeWork with all fields."""
    work = SchemaOrgCreativeWork(
        name="Gene Expression Study",
        url="https://doi.org/10.1234/study",
        author="Smith et al.",
        datePublished="2023-01-15",
    )

    assert work.type == "CreativeWork"
    assert work.name == "Gene Expression Study"
    assert work.url == "https://doi.org/10.1234/study"
    assert work.author == "Smith et al."
    assert work.datePublished == "2023-01-15"


# =============================================================================
# SchemaOrgPropertyValue Tests
# =============================================================================


@pytest.mark.unit
def test_schema_org_property_value_string() -> None:
    """Test SchemaOrgPropertyValue with string value."""
    prop = SchemaOrgPropertyValue(name="difficulty", value="easy")

    assert prop.type == "PropertyValue"
    assert prop.name == "difficulty"
    assert prop.value == "easy"


@pytest.mark.unit
def test_schema_org_property_value_number() -> None:
    """Test SchemaOrgPropertyValue with numeric value."""
    prop = SchemaOrgPropertyValue(name="priority", value=1)

    assert prop.name == "priority"
    assert prop.value == 1


@pytest.mark.unit
def test_schema_org_property_value_dict() -> None:
    """Test SchemaOrgPropertyValue with complex value."""
    prop = SchemaOrgPropertyValue(
        name="metadata",
        value={"key1": "value1", "key2": 42},
    )

    assert prop.name == "metadata"
    assert prop.value == {"key1": "value1", "key2": 42}


# =============================================================================
# SchemaOrgRating Tests
# =============================================================================


@pytest.mark.unit
def test_schema_org_rating_minimal() -> None:
    """Test SchemaOrgRating with minimal fields."""
    rating = SchemaOrgRating(
        name="clarity",
        bestRating=1.0,
        worstRating=0.0,
        additionalType="GlobalRubricTrait",
    )

    assert rating.type == "Rating"
    assert rating.name == "clarity"
    assert rating.bestRating == 1.0
    assert rating.worstRating == 0.0
    assert rating.ratingValue is None
    assert rating.ratingExplanation is None


@pytest.mark.unit
def test_schema_org_rating_with_evaluation() -> None:
    """Test SchemaOrgRating with evaluation results."""
    rating = SchemaOrgRating(
        name="quality",
        bestRating=5.0,
        worstRating=1.0,
        ratingValue=4.0,
        ratingExplanation="Well-structured response",
        additionalType="QuestionSpecificRubricTrait",
    )

    assert rating.ratingValue == 4.0
    assert rating.ratingExplanation == "Well-structured response"


@pytest.mark.unit
def test_schema_org_rating_all_additional_types() -> None:
    """Test SchemaOrgRating accepts all additionalType values."""
    additional_types = [
        "GlobalRubricTrait",
        "QuestionSpecificRubricTrait",
        "GlobalRegexTrait",
        "QuestionSpecificRegexTrait",
        "GlobalCallableTrait",
        "QuestionSpecificCallableTrait",
        "GlobalMetricRubricTrait",
        "QuestionSpecificMetricRubricTrait",
    ]

    for additional_type in additional_types:
        rating = SchemaOrgRating(
            name="test",
            bestRating=1.0,
            worstRating=0.0,
            additionalType=additional_type,
        )
        assert rating.additionalType == additional_type


@pytest.mark.unit
def test_schema_org_rating_invalid_additional_type_raises_error() -> None:
    """Test SchemaOrgRating rejects invalid additionalType."""
    with pytest.raises(ValidationError):
        SchemaOrgRating(
            name="test",
            bestRating=1.0,
            worstRating=0.0,
            additionalType="InvalidType",  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_schema_org_rating_with_additional_property() -> None:
    """Test SchemaOrgRating with additionalProperty for metric traits."""
    rating = SchemaOrgRating(
        name="entity_extraction",
        bestRating=1.0,
        worstRating=0.0,
        additionalType="GlobalMetricRubricTrait",
        additionalProperty=[
            SchemaOrgPropertyValue(name="tp_instructions", value=["mitochondria", "apoptosis"]),
            SchemaOrgPropertyValue(name="tn_instructions", value=["nucleus", "ribosome"]),
        ],
    )

    assert len(rating.additionalProperty) == 2
    assert rating.additionalProperty[0].name == "tp_instructions"
    assert rating.additionalProperty[0].value == ["mitochondria", "apoptosis"]


# =============================================================================
# SchemaOrgSoftwareSourceCode Tests
# =============================================================================


@pytest.mark.unit
def test_schema_org_software_source_code_minimal() -> None:
    """Test SchemaOrgSoftwareSourceCode with minimal fields."""
    code = SchemaOrgSoftwareSourceCode(
        name="Answer",
        text="class Answer(BaseAnswer):\n    value: str",
    )

    assert code.type == "SoftwareSourceCode"
    assert code.name == "Answer"
    assert code.text.startswith("class Answer")
    assert code.programmingLanguage == "Python"
    assert code.id is None


@pytest.mark.unit
def test_schema_org_software_source_code_with_id() -> None:
    """Test SchemaOrgSoftwareSourceCode with @id."""
    code = SchemaOrgSoftwareSourceCode(
        **{"@id": "template-001"},  # Use alias since id conflicts with Python built-in
        name="ComplexAnswer",
        text="class Answer(BaseAnswer):\n    value: str\n    count: int",
    )

    assert code.id == "template-001"


@pytest.mark.unit
def test_schema_org_software_source_code_custom_repository() -> None:
    """Test SchemaOrgSoftwareSourceCode with custom codeRepository."""
    code = SchemaOrgSoftwareSourceCode(
        name="Answer",
        text="class Answer(BaseAnswer):\n    pass",
        codeRepository="custom-benchmarks",
    )

    assert code.codeRepository == "custom-benchmarks"


# =============================================================================
# SchemaOrgAnswer Tests
# =============================================================================


@pytest.mark.unit
def test_schema_org_answer_minimal() -> None:
    """Test SchemaOrgAnswer with minimal fields."""
    answer = SchemaOrgAnswer(text="BCL2")

    assert answer.type == "Answer"
    assert answer.text == "BCL2"
    assert answer.id is None


@pytest.mark.unit
def test_schema_org_answer_with_id() -> None:
    """Test SchemaOrgAnswer with @id."""
    answer = SchemaOrgAnswer(**{"@id": "answer-001"}, text="Paris")

    assert answer.id == "answer-001"
    assert answer.text == "Paris"


# =============================================================================
# SchemaOrgQuestion Tests
# =============================================================================


@pytest.mark.unit
def test_schema_org_question_minimal() -> None:
    """Test SchemaOrgQuestion with minimal required fields."""
    question = SchemaOrgQuestion(
        text="What is the capital of France?",
        acceptedAnswer=SchemaOrgAnswer(text="Paris"),
        hasPart=SchemaOrgSoftwareSourceCode(
            name="Answer",
            text="class Answer(BaseAnswer):\n    value: str",
        ),
    )

    assert question.type == "Question"
    assert question.text == "What is the capital of France?"
    assert question.acceptedAnswer.text == "Paris"
    assert question.hasPart.name == "Answer"
    assert question.rating is None
    assert question.additionalProperty is None


@pytest.mark.unit
def test_schema_org_question_with_ratings() -> None:
    """Test SchemaOrgQuestion with question-specific rubric traits."""
    question = SchemaOrgQuestion(
        text="Test question",
        acceptedAnswer=SchemaOrgAnswer(text="Answer"),
        hasPart=SchemaOrgSoftwareSourceCode(
            name="Answer",
            text="class Answer(BaseAnswer):\n    pass",
        ),
        rating=[
            SchemaOrgRating(
                name="clarity",
                bestRating=1.0,
                worstRating=0.0,
                additionalType="QuestionSpecificRubricTrait",
            ),
            SchemaOrgRating(
                name="citation",
                bestRating=1.0,
                worstRating=0.0,
                additionalType="QuestionSpecificRegexTrait",
            ),
        ],
    )

    assert len(question.rating) == 2
    assert question.rating[0].name == "clarity"
    assert question.rating[1].name == "citation"


@pytest.mark.unit
def test_schema_org_question_with_custom_properties() -> None:
    """Test SchemaOrgQuestion with additionalProperty metadata."""
    question = SchemaOrgQuestion(
        text="Test question",
        acceptedAnswer=SchemaOrgAnswer(text="Answer"),
        hasPart=SchemaOrgSoftwareSourceCode(
            name="Answer",
            text="class Answer(BaseAnswer):\n    pass",
        ),
        additionalProperty=[
            SchemaOrgPropertyValue(name="category", value="biology"),
            SchemaOrgPropertyValue(name="difficulty", value="hard"),
        ],
    )

    assert len(question.additionalProperty) == 2
    assert question.additionalProperty[0].name == "category"
    assert question.additionalProperty[1].value == "hard"


# =============================================================================
# SchemaOrgDataFeedItem Tests
# =============================================================================


@pytest.mark.unit
def test_schema_org_data_feed_item_minimal() -> None:
    """Test SchemaOrgDataFeedItem with minimal fields."""
    now = datetime.now().isoformat()

    item = SchemaOrgDataFeedItem(
        dateCreated=now,
        dateModified=now,
        item=SchemaOrgQuestion(
            text="Test?",
            acceptedAnswer=SchemaOrgAnswer(text="Answer"),
            hasPart=SchemaOrgSoftwareSourceCode(
                name="Answer",
                text="class Answer(BaseAnswer):\n    pass",
            ),
        ),
    )

    assert item.type == "DataFeedItem"
    assert item.dateCreated == now
    assert item.dateModified == now
    assert item.item.text == "Test?"
    assert item.keywords is None


@pytest.mark.unit
def test_schema_org_data_feed_item_with_keywords() -> None:
    """Test SchemaOrgDataFeedItem with keywords."""
    now = datetime.now().isoformat()

    item = SchemaOrgDataFeedItem(
        dateCreated=now,
        dateModified=now,
        item=SchemaOrgQuestion(
            text="Test?",
            acceptedAnswer=SchemaOrgAnswer(text="Answer"),
            hasPart=SchemaOrgSoftwareSourceCode(
                name="Answer",
                text="class Answer(BaseAnswer):\n    pass",
            ),
        ),
        keywords=["biology", "genetics", "apoptosis"],
    )

    assert item.keywords == ["biology", "genetics", "apoptosis"]


# =============================================================================
# SchemaOrgDataFeed Tests
# =============================================================================


@pytest.mark.unit
def test_schema_org_data_feed_minimal() -> None:
    """Test SchemaOrgDataFeed with minimal fields."""
    now = datetime.now().isoformat()

    feed = SchemaOrgDataFeed(
        name="Test Benchmark",
        dateCreated=now,
        dateModified=now,
        dataFeedElement=[],
    )

    assert feed.type == "DataFeed"
    assert feed.name == "Test Benchmark"
    assert feed.description is None
    assert feed.version is None
    assert feed.creator is None
    assert feed.rating is None
    assert feed.dataFeedElement == []


@pytest.mark.unit
def test_schema_org_data_feed_with_elements() -> None:
    """Test SchemaOrgDataFeed with dataFeedElement items."""
    now = datetime.now().isoformat()

    feed = SchemaOrgDataFeed(
        name="Test Benchmark",
        dateCreated=now,
        dateModified=now,
        dataFeedElement=[
            SchemaOrgDataFeedItem(
                dateCreated=now,
                dateModified=now,
                item=SchemaOrgQuestion(
                    text="Q1?",
                    acceptedAnswer=SchemaOrgAnswer(text="A1"),
                    hasPart=SchemaOrgSoftwareSourceCode(
                        name="Answer",
                        text="class Answer(BaseAnswer):\n    pass",
                    ),
                ),
            ),
            SchemaOrgDataFeedItem(
                dateCreated=now,
                dateModified=now,
                item=SchemaOrgQuestion(
                    text="Q2?",
                    acceptedAnswer=SchemaOrgAnswer(text="A2"),
                    hasPart=SchemaOrgSoftwareSourceCode(
                        name="Answer",
                        text="class Answer(BaseAnswer):\n    pass",
                    ),
                ),
            ),
        ],
    )

    assert len(feed.dataFeedElement) == 2
    assert feed.dataFeedElement[0].item.text == "Q1?"
    assert feed.dataFeedElement[1].item.text == "Q2?"


@pytest.mark.unit
def test_schema_org_data_feed_with_global_ratings() -> None:
    """Test SchemaOrgDataFeed with global rubric traits."""
    now = datetime.now().isoformat()

    feed = SchemaOrgDataFeed(
        name="Test Benchmark",
        dateCreated=now,
        dateModified=now,
        dataFeedElement=[],
        rating=[
            SchemaOrgRating(
                name="safety",
                bestRating=1.0,
                worstRating=0.0,
                additionalType="GlobalRubricTrait",
            ),
        ],
    )

    assert len(feed.rating) == 1
    assert feed.rating[0].name == "safety"


@pytest.mark.unit
def test_schema_org_data_feed_creator_string() -> None:
    """Test SchemaOrgDataFeed with string creator."""
    now = datetime.now().isoformat()

    feed = SchemaOrgDataFeed(
        name="Test Benchmark",
        dateCreated=now,
        dateModified=now,
        dataFeedElement=[],
        creator="Research Lab",
    )

    assert feed.creator == "Research Lab"


@pytest.mark.unit
def test_schema_org_data_feed_creator_person() -> None:
    """Test SchemaOrgDataFeed with SchemaOrgPerson creator."""
    now = datetime.now().isoformat()

    feed = SchemaOrgDataFeed(
        name="Test Benchmark",
        dateCreated=now,
        dateModified=now,
        dataFeedElement=[],
        creator=SchemaOrgPerson(
            name="Alice Smith",
            email="alice@example.com",
        ),
    )

    assert isinstance(feed.creator, SchemaOrgPerson)
    assert feed.creator.name == "Alice Smith"


# =============================================================================
# JsonLdCheckpoint Tests
# =============================================================================


@pytest.mark.unit
def test_json_ld_checkpoint_minimal() -> None:
    """Test JsonLdCheckpoint with minimal fields."""
    context = SCHEMA_ORG_CONTEXT
    now = datetime.now().isoformat()

    checkpoint = JsonLdCheckpoint(
        **{"@context": context},
        name="Test Benchmark",
        dateCreated=now,
        dateModified=now,
        dataFeedElement=[],
    )

    assert checkpoint.context == context
    assert checkpoint.type == "DataFeed"
    assert checkpoint.name == "Test Benchmark"


@pytest.mark.unit
def test_json_ld_checkpoint_full_structure() -> None:
    """Test JsonLdCheckpoint with complete structure."""
    context = SCHEMA_ORG_CONTEXT
    now = datetime.now().isoformat()

    checkpoint = JsonLdCheckpoint(
        **{"@context": context, "@id": "benchmark-001"},
        name="Complete Benchmark",
        description="A comprehensive test benchmark",
        version="1.0.0",
        creator=SchemaOrgPerson(name="Alice", email="alice@example.com"),
        dateCreated=now,
        dateModified=now,
        rating=[
            SchemaOrgRating(
                name="clarity",
                bestRating=1.0,
                worstRating=0.0,
                additionalType="GlobalRubricTrait",
            )
        ],
        dataFeedElement=[
            SchemaOrgDataFeedItem(
                **{"@id": "q001"},
                dateCreated=now,
                dateModified=now,
                item=SchemaOrgQuestion(
                    text="What is BCL2?",
                    acceptedAnswer=SchemaOrgAnswer(text="BCL2 is a gene"),
                    hasPart=SchemaOrgSoftwareSourceCode(
                        **{"@id": "template-001"},
                        name="Answer",
                        text="class Answer(BaseAnswer):\n    value: str",
                    ),
                    rating=[
                        SchemaOrgRating(
                            name="accuracy",
                            bestRating=1.0,
                            worstRating=0.0,
                            additionalType="QuestionSpecificRubricTrait",
                        )
                    ],
                ),
                keywords=["biology", "genetics"],
            )
        ],
        additionalProperty=[
            SchemaOrgPropertyValue(name="domain", value="biology"),
        ],
    )

    assert checkpoint.id == "benchmark-001"
    assert checkpoint.description == "A comprehensive test benchmark"
    assert checkpoint.version == "1.0.0"
    assert len(checkpoint.rating) == 1
    assert len(checkpoint.dataFeedElement) == 1
    assert checkpoint.dataFeedElement[0].id == "q001"
    assert checkpoint.dataFeedElement[0].keywords == ["biology", "genetics"]
    assert len(checkpoint.additionalProperty) == 1


@pytest.mark.unit
def test_json_ld_checkpoint_serialization() -> None:
    """Test JsonLdCheckpoint serializes to JSON correctly."""
    context = SCHEMA_ORG_CONTEXT
    now = datetime.now().isoformat()

    checkpoint = JsonLdCheckpoint(
        **{"@context": context},
        name="Test",
        dateCreated=now,
        dateModified=now,
        dataFeedElement=[],
    )

    # Test by_alias serialization for @type and @id
    data = checkpoint.model_dump(by_alias=True)

    assert "@context" in data
    assert "@type" in data
    assert data["@type"] == "DataFeed"
    assert "type" not in data


@pytest.mark.unit
def test_json_ld_checkpoint_roundtrip() -> None:
    """Test JsonLdCheckpoint can serialize and deserialize."""
    context = SCHEMA_ORG_CONTEXT
    now = datetime.now().isoformat()

    original = JsonLdCheckpoint(
        **{"@context": context},
        name="Roundtrip Test",
        dateCreated=now,
        dateModified=now,
        dataFeedElement=[
            SchemaOrgDataFeedItem(
                dateCreated=now,
                dateModified=now,
                item=SchemaOrgQuestion(
                    text="Test?",
                    acceptedAnswer=SchemaOrgAnswer(text="Answer"),
                    hasPart=SchemaOrgSoftwareSourceCode(
                        name="Answer",
                        text="class Answer(BaseAnswer):\n    pass",
                    ),
                ),
            ),
        ],
    )

    # Serialize
    json_str = original.model_dump_json()

    # Deserialize
    restored = JsonLdCheckpoint.model_validate_json(json_str)

    assert restored.name == original.name
    assert len(restored.dataFeedElement) == len(original.dataFeedElement)


# =============================================================================
# SCHEMA_ORG_CONTEXT Tests
# =============================================================================


@pytest.mark.unit
def test_schema_org_context_structure() -> None:
    """Test SCHEMA_ORG_CONTEXT has required keys."""
    assert "@version" in SCHEMA_ORG_CONTEXT
    assert "@vocab" in SCHEMA_ORG_CONTEXT
    assert SCHEMA_ORG_CONTEXT["@vocab"] == "http://schema.org/"


@pytest.mark.unit
def test_schema_org_context_mappings() -> None:
    """Test SCHEMA_ORG_CONTEXT has required type mappings."""
    assert SCHEMA_ORG_CONTEXT.get("DataFeed") == "DataFeed"
    assert SCHEMA_ORG_CONTEXT.get("Question") == "Question"
    assert SCHEMA_ORG_CONTEXT.get("Answer") == "Answer"
    assert SCHEMA_ORG_CONTEXT.get("Rating") == "Rating"


@pytest.mark.unit
def test_schema_org_context_data_feed_element_set_container() -> None:
    """Test dataFeedElement uses @set container."""
    mapping = SCHEMA_ORG_CONTEXT.get("dataFeedElement")
    assert isinstance(mapping, dict)
    assert mapping.get("@container") == "@set"


@pytest.mark.unit
def test_schema_org_context_rating_set_container() -> None:
    """Test rating uses @set container."""
    mapping = SCHEMA_ORG_CONTEXT.get("rating")
    assert isinstance(mapping, dict)
    assert mapping.get("@container") == "@set"
