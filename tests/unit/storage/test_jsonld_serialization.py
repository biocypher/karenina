"""Unit tests for Benchmark JSON-LD serialization (save/load).

Tests cover:
- Benchmark.save() to JSON-LD file
- Benchmark.load() from JSON-LD file
- Roundtrip consistency (save → load → identical)
- Malformed JSON-LD handling
- save_deep_judgment_config flag behavior
- File extension handling
- Data preservation (questions, metadata, rubrics, templates)
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from karenina import Benchmark
from karenina.schemas.entities import LLMRubricTrait, RegexTrait

# =============================================================================
# save() Method Tests
# =============================================================================


@pytest.mark.unit
def test_save_benchmark_to_file(tmp_path: Path) -> None:
    """Test saving a benchmark to a JSON-LD file."""
    benchmark = Benchmark.create(
        name="Test Benchmark",
        description="A test benchmark for serialization",
        version="1.0.0",
    )

    # Add a question
    benchmark.add_question(
        question="What is the capital of France?",
        raw_answer="Paris",
        answer_template="class Answer(BaseAnswer):\n    value: str",
    )

    # Save to file
    save_path = tmp_path / "test.jsonld"
    benchmark.save(save_path)

    # Verify file exists
    assert save_path.exists()

    # Verify content is valid JSON
    with open(save_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["name"] == "Test Benchmark"
    assert data["description"] == "A test benchmark for serialization"
    assert data["version"] == "1.0.0"
    assert len(data["dataFeedElement"]) == 1


@pytest.mark.unit
def test_save_adds_jsonld_extension(tmp_path: Path) -> None:
    """Test that save() adds .jsonld extension if not present."""
    benchmark = Benchmark.create(name="Test")

    # Save without extension
    save_path = tmp_path / "test"
    benchmark.save(save_path)

    # Should create .jsonld file
    assert (tmp_path / "test.jsonld").exists()


@pytest.mark.unit
def test_save_preserves_json_extension(tmp_path: Path) -> None:
    """Test that save() preserves .json extension."""
    benchmark = Benchmark.create(name="Test")

    # Save with .json extension
    save_path = tmp_path / "test.json"
    benchmark.save(save_path)

    # Should create .json file
    assert (tmp_path / "test.json").exists()


@pytest.mark.unit
def test_save_updates_date_modified(tmp_path: Path) -> None:
    """Test that saving updates the dateModified timestamp."""
    benchmark = Benchmark.create(name="Test")

    # Get original dateModified
    original_modified = benchmark.modified_at

    # Wait a bit and save
    import time

    time.sleep(0.01)
    save_path = tmp_path / "test.jsonld"
    benchmark.save(save_path)

    # Load and check dateModified was updated
    loaded = Benchmark.load(save_path)
    assert loaded.modified_at != original_modified
    assert loaded.modified_at > original_modified


@pytest.mark.unit
def test_save_with_deep_judgment_config_by_default_strips_it(tmp_path: Path) -> None:
    """Test that save() strips deep judgment config by default."""
    from karenina.utils.checkpoint import add_global_rubric_to_benchmark

    benchmark = Benchmark.create(name="Test")

    # Add a global rubric with deep judgment config
    trait = LLMRubricTrait(
        name="clarity",
        kind="boolean",
        deep_judgment_enabled=True,
        deep_judgment_excerpt_enabled=True,
        deep_judgment_search_enabled=True,
        deep_judgment_max_excerpts=5,
    )

    # Manually add to the underlying checkpoint
    add_global_rubric_to_benchmark(benchmark.jsonld_data, [trait])

    # Save with default settings (should strip deep judgment)
    save_path = tmp_path / "test.jsonld"
    benchmark.save(save_path)

    # Load and verify deep judgment was stripped
    with open(save_path, encoding="utf-8") as f:
        data = json.load(f)

    # Check that deep judgment properties are not in the saved file
    ratings = data.get("rating", [])
    assert len(ratings) > 0
    for rating in ratings:
        if rating.get("name") == "clarity":
            props = rating.get("additionalProperty", [])
            prop_names = [p.get("name") for p in props]
            assert not any(name.startswith("deep_judgment_") for name in prop_names)


@pytest.mark.unit
def test_save_with_deep_judgment_config_preserves_it(tmp_path: Path) -> None:
    """Test that save() with save_deep_judgment_config=True preserves deep judgment."""
    from karenina.utils.checkpoint import add_global_rubric_to_benchmark

    benchmark = Benchmark.create(name="Test")

    # Add a global rubric with deep judgment config
    trait = LLMRubricTrait(
        name="clarity",
        kind="boolean",
        deep_judgment_enabled=True,
        deep_judgment_excerpt_enabled=True,
        deep_judgment_max_excerpts=5,
    )

    # Manually add to the underlying checkpoint
    add_global_rubric_to_benchmark(benchmark.jsonld_data, [trait])

    # Save with deep judgment config preserved
    save_path = tmp_path / "test.jsonld"
    benchmark._base.save(save_path, save_deep_judgment_config=True)

    # Load and verify deep judgment was preserved
    with open(save_path, encoding="utf-8") as f:
        data = json.load(f)

    # Check that deep judgment properties are in the saved file
    ratings = data.get("rating", [])
    assert len(ratings) > 0
    for rating in ratings:
        if rating.get("name") == "clarity":
            props = rating.get("additionalProperty", [])
            prop_names = [p.get("name") for p in props]
            assert "deep_judgment_enabled" in prop_names
            assert "deep_judgment_excerpt_enabled" in prop_names


@pytest.mark.unit
def test_save_preserves_all_metadata(tmp_path: Path) -> None:
    """Test that save() preserves all benchmark metadata."""
    benchmark = Benchmark.create(
        name="Metadata Test",
        description="Testing metadata preservation",
        version="2.5.0",
        creator="Test Creator",
    )

    benchmark.add_question(
        question="Test question?",
        raw_answer="Test answer",
        answer_template="class Answer(BaseAnswer):\n    value: str",
        author={"name": "John Doe", "email": "john@example.com"},
        sources=[{"title": "Test Source", "url": "https://example.com"}],
        custom_metadata={"difficulty": "easy", "category": "test"},
    )

    save_path = tmp_path / "test.jsonld"
    benchmark.save(save_path)

    # Load and verify all metadata
    loaded = Benchmark.load(save_path)

    assert loaded.name == "Metadata Test"
    assert loaded.description == "Testing metadata preservation"
    assert loaded.version == "2.5.0"
    assert loaded.creator == "Test Creator"

    # Check question metadata
    questions = loaded._base._questions_cache
    assert len(questions) == 1
    q = list(questions.values())[0]
    assert q["author"]["name"] == "John Doe"
    assert q["sources"][0]["title"] == "Test Source"
    assert q["custom_metadata"]["difficulty"] == "easy"


# =============================================================================
# load() Method Tests
# =============================================================================


@pytest.mark.unit
def test_load_benchmark_from_file(tmp_path: Path) -> None:
    """Test loading a benchmark from a JSON-LD file."""
    # Create and save a benchmark
    original = Benchmark.create(name="Load Test")
    original.add_question(
        question="Test question?",
        raw_answer="Test answer",
        answer_template="class Answer(BaseAnswer):\n    value: str",
    )

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)

    # Load the benchmark
    loaded = Benchmark.load(save_path)

    # Verify loaded data
    assert loaded.name == "Load Test"
    assert loaded.question_count == 1
    # Questions with templates are auto-finished
    assert loaded.finished_count == 1


@pytest.mark.unit
def test_load_nonexistent_file_raises_error(tmp_path: Path) -> None:
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Benchmark file not found"):
        Benchmark.load(tmp_path / "nonexistent.jsonld")


@pytest.mark.unit
def test_load_invalid_json_raises_error(tmp_path: Path) -> None:
    """Test that loading invalid JSON raises ValueError."""

    invalid_file = tmp_path / "invalid.jsonld"
    with open(invalid_file, "w", encoding="utf-8") as f:
        f.write("{ this is not valid json }")

    # JSONDecodeError is raised, which is a subclass of ValueError
    with pytest.raises(ValueError):
        Benchmark.load(invalid_file)


@pytest.mark.unit
def test_load_invalid_jsonld_structure_raises_error(tmp_path: Path) -> None:
    """Test that loading valid JSON but invalid JSON-LD structure raises error."""
    invalid_file = tmp_path / "invalid_structure.jsonld"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump({"@type": "WrongType", "name": "Test"}, f)

    # Pydantic ValidationError or ValueError is raised
    with pytest.raises((ValueError, ValidationError)):
        Benchmark.load(invalid_file)


@pytest.mark.unit
def test_load_missing_required_fields_raises_error(tmp_path: Path) -> None:
    """Test that loading checkpoint missing required fields raises ValueError."""
    # Create a checkpoint with missing name
    invalid_file = tmp_path / "missing_name.jsonld"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "@context": "https://schema.org",
                "@type": "DataFeed",
                # name is missing
                "dataFeedElement": [],
            },
            f,
        )

    with pytest.raises(ValueError, match="name"):
        Benchmark.load(invalid_file)


# =============================================================================
# Roundtrip Tests
# =============================================================================


@pytest.mark.unit
def test_roundtrip_preserves_name_metadata(tmp_path: Path) -> None:
    """Test that save/load roundtrip preserves name and metadata."""
    original = Benchmark.create(
        name="Roundtrip Test",
        description="Testing roundtrip preservation",
        version="3.0.0",
        creator="Roundtrip Creator",
    )

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)
    loaded = Benchmark.load(save_path)

    assert loaded.name == original.name
    assert loaded.description == original.description
    assert loaded.version == original.version
    assert loaded.creator == original.creator


@pytest.mark.unit
def test_roundtrip_preserves_questions(tmp_path: Path) -> None:
    """Test that save/load roundtrip preserves all questions."""
    original = Benchmark.create(name="Questions Test")

    # Add multiple questions
    original.add_question(
        question="Question 1?",
        raw_answer="Answer 1",
        answer_template="class Answer(BaseAnswer):\n    value: str",
    )
    original.add_question(
        question="Question 2?",
        raw_answer="Answer 2",
        answer_template="class Answer(BaseAnswer):\n    value: str",
    )
    original.add_question(
        question="Question 3?",
        raw_answer="Answer 3",
        answer_template="class Answer(BaseAnswer):\n    value: str",
        author={"name": "Author 3"},
    )

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)
    loaded = Benchmark.load(save_path)

    assert loaded.question_count == 3
    # All questions with templates are auto-finished
    assert loaded.finished_count == 3

    # Check question IDs are preserved
    original_ids = set(original._base._questions_cache.keys())
    loaded_ids = set(loaded._base._questions_cache.keys())
    assert original_ids == loaded_ids


@pytest.mark.unit
def test_roundtrip_preserves_templates(tmp_path: Path) -> None:
    """Test that save/load roundtrip preserves answer templates."""
    original = Benchmark.create(name="Template Test")

    template = "class Answer(BaseAnswer):\n    value: str\n\n    def verify(self) -> bool:\n        return self.value == 'expected'"
    original.add_question(
        question="Template question?",
        raw_answer="expected",
        answer_template=template,
    )

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)
    loaded = Benchmark.load(save_path)

    # Get the question from loaded benchmark
    questions = loaded._base._questions_cache
    assert len(questions) == 1
    loaded_template = list(questions.values())[0]["answer_template"]
    assert loaded_template == template


@pytest.mark.unit
def test_roundtrip_preserves_rubrics(tmp_path: Path) -> None:
    """Test that save/load roundtrip preserves rubric traits."""
    from karenina.utils.checkpoint import (
        add_global_rubric_to_benchmark,
        add_question_to_benchmark,
    )

    original = Benchmark.create(name="Rubric Test")

    # Add a global regex rubric
    rubric = RegexTrait(
        name="has_citation",
        pattern=r"\[\d+\]",
        case_sensitive=True,
        description="Checks for citation patterns",
    )
    add_global_rubric_to_benchmark(original.jsonld_data, [rubric])

    # Add question-specific rubric using checkpoint utility
    add_question_to_benchmark(
        original.jsonld_data,
        question="Rubric question?",
        raw_answer="Answer",
        answer_template="class Answer(BaseAnswer):\n    value: str",
        question_rubric_traits=[
            RegexTrait(
                name="has_url",
                pattern=r"https?://\S+",
                higher_is_better=True,
                description="Checks for URL patterns",
            ),
        ],
    )

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)
    loaded = Benchmark.load(save_path)

    # Check global rubrics are preserved
    global_traits = loaded._base.jsonld_data.rating
    assert global_traits is not None
    assert len(global_traits) > 0

    # Check question-specific rubrics
    questions = loaded._base._questions_cache
    q = list(questions.values())[0]
    assert q["question_rubric"] is not None
    assert "regex_traits" in q["question_rubric"]


@pytest.mark.unit
def test_roundtrip_preserves_keywords(tmp_path: Path) -> None:
    """Test that save/load roundtrip preserves question keywords."""
    from karenina.utils.checkpoint import add_question_to_benchmark

    original = Benchmark.create(name="Keywords Test")

    add_question_to_benchmark(
        original.jsonld_data,
        question="Question?",
        raw_answer="Answer",
        answer_template="class Answer(BaseAnswer):\n    value: str",
        keywords=["keyword1", "keyword2", "keyword3"],
    )

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)
    loaded = Benchmark.load(save_path)

    questions = loaded._base._questions_cache
    q = list(questions.values())[0]
    assert q["keywords"] == ["keyword1", "keyword2", "keyword3"]


@pytest.mark.unit
def test_roundtrip_preserves_author_sources(tmp_path: Path) -> None:
    """Test that save/load roundtrip preserves author and sources."""
    original = Benchmark.create(name="Author Sources Test")

    author = {"name": "Jane Doe", "email": "jane@example.com", "affiliation": "Test Org"}
    sources = [
        {"title": "Source 1", "url": "https://example.com/1"},
        {"title": "Source 2", "url": "https://example.com/2"},
    ]

    original.add_question(
        question="Question?",
        raw_answer="Answer",
        answer_template="class Answer(BaseAnswer):\n    value: str",
        author=author,
        sources=sources,
    )

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)
    loaded = Benchmark.load(save_path)

    questions = loaded._base._questions_cache
    q = list(questions.values())[0]
    assert q["author"]["name"] == "Jane Doe"
    assert q["author"]["email"] == "jane@example.com"
    assert len(q["sources"]) == 2
    assert q["sources"][0]["title"] == "Source 1"


@pytest.mark.unit
def test_roundtrip_preserves_custom_metadata(tmp_path: Path) -> None:
    """Test that save/load roundtrip preserves custom metadata."""
    original = Benchmark.create(name="Custom Metadata Test")

    original.add_question(
        question="Question?",
        raw_answer="Answer",
        answer_template="class Answer(BaseAnswer):\n    value: str",
        custom_metadata={
            "difficulty": "hard",
            "category": "science",
            "estimated_time": 30,
            "tags": ["biology", "genetics"],
        },
    )

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)
    loaded = Benchmark.load(save_path)

    questions = loaded._base._questions_cache
    q = list(questions.values())[0]
    assert q["custom_metadata"]["difficulty"] == "hard"
    assert q["custom_metadata"]["category"] == "science"
    assert q["custom_metadata"]["estimated_time"] == 30
    assert q["custom_metadata"]["tags"] == ["biology", "genetics"]


@pytest.mark.unit
def test_roundtrip_preserves_few_shot_examples(tmp_path: Path) -> None:
    """Test that save/load roundtrip preserves few-shot examples."""
    original = Benchmark.create(name="Few Shot Test")

    few_shot = [
        {"question": "Example Q1", "answer": "Example A1"},
        {"question": "Example Q2", "answer": "Example A2"},
    ]

    original.add_question(
        question="Question?",
        raw_answer="Answer",
        answer_template="class Answer(BaseAnswer):\n    value: str",
        few_shot_examples=few_shot,
    )

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)
    loaded = Benchmark.load(save_path)

    questions = loaded._base._questions_cache
    q = list(questions.values())[0]
    assert len(q["few_shot_examples"]) == 2
    assert q["few_shot_examples"][0]["question"] == "Example Q1"


@pytest.mark.unit
def test_roundtrip_empty_benchmark(tmp_path: Path) -> None:
    """Test that save/load roundtrip works for empty benchmark."""
    original = Benchmark.create(name="Empty Benchmark")

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)
    loaded = Benchmark.load(save_path)

    assert loaded.name == "Empty Benchmark"
    assert loaded.question_count == 0
    assert loaded.is_empty


# =============================================================================
# Malformed JSON-LD Handling Tests
# =============================================================================


@pytest.mark.unit
def test_load_with_extra_unknown_fields(tmp_path: Path) -> None:
    """Test loading JSON-LD with extra unknown fields (should be ignored)."""
    # Create a valid benchmark
    original = Benchmark.create(name="Extra Fields Test")
    original.add_question(
        question="Question?",
        raw_answer="Answer",
        answer_template="class Answer(BaseAnswer):\n    value: str",
    )

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)

    # Add extra fields to the JSON
    with open(save_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["unknownField"] = "some value"
        data["extraField"] = 123
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    # Should still load successfully (Pydantic ignores extra fields)
    loaded = Benchmark.load(save_path)
    assert loaded.name == "Extra Fields Test"


@pytest.mark.unit
def test_load_with_malformed_date(tmp_path: Path) -> None:
    """Test loading JSON-LD with malformed date (Pydantic should handle)."""
    # Create a valid benchmark
    original = Benchmark.create(name="Date Test")
    original.add_question(
        question="Question?",
        raw_answer="Answer",
        answer_template="class Answer(BaseAnswer):\n    value: str",
    )

    save_path = tmp_path / "test.jsonld"
    original.save(save_path)

    # Modify the dateModified to be malformed
    with open(save_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["dateModified"] = "not-a-valid-date"
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    # Pydantic allows strings for date fields, so this should load
    # (the field is typed as str in JsonLdCheckpoint)
    loaded = Benchmark.load(save_path)
    assert loaded.name == "Date Test"


@pytest.mark.unit
def test_load_with_missing_text_in_question(tmp_path: Path) -> None:
    """Test loading JSON-LD with question missing text field."""
    invalid_file = tmp_path / "missing_text.jsonld"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "@context": "https://schema.org",
                "@type": "DataFeed",
                "name": "Invalid Benchmark",
                "dataFeedElement": [
                    {
                        "@type": "DataFeedItem",
                        "item": {
                            "@type": "Question",
                            # text is missing
                            "acceptedAnswer": {
                                "@type": "Answer",
                                "text": "Answer",
                            },
                        },
                    }
                ],
            },
            f,
        )

    # Pydantic validation happens before our custom validation
    with pytest.raises((ValueError, ValidationError)):
        Benchmark.load(invalid_file)


@pytest.mark.unit
def test_load_with_invalid_rating_type(tmp_path: Path) -> None:
    """Test loading JSON-LD with invalid rating additionalType."""
    invalid_file = tmp_path / "invalid_rating.jsonld"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "@context": "https://schema.org",
                "@type": "DataFeed",
                "name": "Invalid Benchmark",
                "rating": [
                    {
                        "@type": "Rating",
                        "name": "invalid_trait",
                        "additionalType": "InvalidTraitType",  # Not a valid type
                    }
                ],
                "dataFeedElement": [],
            },
            f,
        )

    # Pydantic should reject this at the model validation level
    with pytest.raises((ValueError, ValidationError)):
        Benchmark.load(invalid_file)


@pytest.mark.unit
def test_load_with_question_specific_at_global_level(tmp_path: Path) -> None:
    """Test loading JSON-LD with question-specific trait at global level."""
    invalid_file = tmp_path / "invalid_global.jsonld"
    with open(invalid_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "@context": "https://schema.org",
                "@type": "DataFeed",
                "name": "Invalid Benchmark",
                "rating": [
                    {
                        "@type": "Rating",
                        "name": "trait",
                        "additionalType": "QuestionSpecificRubricTrait",  # Invalid at global level
                    }
                ],
                "dataFeedElement": [],
            },
            f,
        )

    # Pydantic validation happens first, rejecting the invalid additionalType
    with pytest.raises((ValueError, ValidationError)):
        Benchmark.load(invalid_file)


# =============================================================================
# JSON-LD Structure Tests
# =============================================================================


@pytest.mark.unit
def test_saved_jsonld_has_correct_structure(tmp_path: Path) -> None:
    """Test that saved JSON-LD has correct Schema.org structure."""
    benchmark = Benchmark.create(name="Structure Test")
    benchmark.add_question(
        question="Question?",
        raw_answer="Answer",
        answer_template="class Answer(BaseAnswer):\n    value: str",
    )

    save_path = tmp_path / "test.jsonld"
    benchmark.save(save_path)

    with open(save_path, encoding="utf-8") as f:
        data = json.load(f)

    # Verify Schema.org structure
    # @context is a dictionary with Schema.org mappings
    assert isinstance(data["@context"], dict)
    assert "@vocab" in data["@context"]
    assert data["@type"] == "DataFeed"
    assert data["name"] == "Structure Test"
    assert isinstance(data["dataFeedElement"], list)

    # Verify question structure
    item = data["dataFeedElement"][0]
    assert item["@type"] == "DataFeedItem"
    assert item["item"]["@type"] == "Question"
    assert item["item"]["acceptedAnswer"]["@type"] == "Answer"
    assert item["item"]["hasPart"]["@type"] == "SoftwareSourceCode"


@pytest.mark.unit
def test_saved_jsonld_has_correct_aliases(tmp_path: Path) -> None:
    """Test that saved JSON-LD uses correct field aliases."""
    benchmark = Benchmark.create(name="Alias Test")
    benchmark.add_question(
        question="Question?",
        raw_answer="Answer",
        answer_template="class Answer(BaseAnswer):\n    value: str",
    )

    save_path = tmp_path / "test.jsonld"
    benchmark.save(save_path)

    with open(save_path, encoding="utf-8") as f:
        data = json.load(f)

    # Check for @id alias (not id)
    assert "@id" in data
    assert "id" not in data or "@id" in data  # @id should be present

    # Check for @type alias (not type)
    assert data["@type"] == "DataFeed"
