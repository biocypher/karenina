"""Tests for benchmark functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from karenina.benchmark.benchmark import Benchmark
from karenina.schemas.rubric_class import RubricTrait


def test_create_benchmark():
    """Test creating a new benchmark."""
    benchmark = Benchmark.create(
        name="Test Benchmark", description="A test benchmark", version="1.0.0", creator="Test Suite"
    )

    assert benchmark._checkpoint.name == "Test Benchmark"
    assert benchmark._checkpoint.description == "A test benchmark"
    assert benchmark._checkpoint.version == "1.0.0"
    assert benchmark._checkpoint.creator == "Test Suite"
    assert len(benchmark.get_question_ids()) == 0


def test_add_question():
    """Test adding questions to a benchmark."""
    benchmark = Benchmark.create("Test")

    # Add a question with minimal information
    q_id = benchmark.add_question(question="What is Python?", raw_answer="Python is a programming language")

    assert q_id in benchmark.get_question_ids()
    question = benchmark.get_question(q_id)
    assert question["question"] == "What is Python?"
    assert question["raw_answer"] == "Python is a programming language"
    assert "BaseAnswer" in question["answer_template"]

    # Add a question with full template
    template_code = """from karenina.schemas import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    language_name: str = Field(description="Name of the language")
    paradigm: str = Field(description="Programming paradigm")

    def verify(self) -> bool:
        return self.language_name.lower() == "python"
"""

    q_id2 = benchmark.add_question(
        question="Name a dynamic language", raw_answer="Python", answer_template=template_code, finished=True
    )

    question2 = benchmark.get_question(q_id2)
    assert question2["answer_template"] == template_code
    assert question2["finished"] is True


def test_add_answer_template():
    """Test adding/updating answer templates."""
    benchmark = Benchmark.create("Test")

    q_id = benchmark.add_question(question="What is recursion?", raw_answer="A function that calls itself")

    # Update with a proper template (no imports needed)
    template = """class Answer(BaseAnswer):
    definition: str = Field(description="Definition of recursion")
    example: str = Field(description="Example of recursion")

    def verify(self) -> bool:
        return "itself" in self.definition.lower()
"""

    benchmark.add_answer_template(q_id, template)

    question = benchmark.get_question(q_id)
    assert question["answer_template"] == template

    # Test invalid template (empty template)
    invalid_template = ""
    with pytest.raises(ValueError, match="Invalid template"):
        benchmark.add_answer_template(q_id, invalid_template)


def test_rubric_management():
    """Test global and question-specific rubrics."""
    benchmark = Benchmark.create("Test")

    # Add global rubric trait
    global_trait = RubricTrait(name="clarity", description="Is the response clear?", kind="boolean")
    benchmark.add_global_rubric_trait(global_trait)

    rubric = benchmark.get_global_rubric()
    assert rubric is not None
    assert len(rubric.traits) == 1
    assert rubric.traits[0].name == "clarity"

    # Add question with question-specific rubric
    q_id = benchmark.add_question(
        question="Explain quicksort", raw_answer="Quicksort is a divide-and-conquer algorithm..."
    )

    question_trait = RubricTrait(
        name="complexity_analysis", description="Does it mention time complexity?", kind="boolean"
    )
    benchmark.add_question_rubric_trait(q_id, question_trait)

    # Verify in the benchmark data
    question_data = benchmark.get_question(q_id)
    assert question_data["question_rubric"] is not None
    assert len(question_data["question_rubric"]) == 1
    assert question_data["question_rubric"][0].name == "complexity_analysis"


def test_save_and_load():
    """Test saving and loading benchmarks."""
    # Create a benchmark with data
    benchmark = Benchmark.create(name="Save/Load Test", description="Testing save and load")

    q_id = benchmark.add_question(question="What is AI?", raw_answer="Artificial Intelligence", finished=True)

    benchmark.add_global_rubric_trait(RubricTrait(name="accuracy", description="Is it accurate?", kind="boolean"))

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jsonld", delete=False) as f:
        temp_path = Path(f.name)

    try:
        benchmark.save(temp_path)

        # Load from file
        loaded = Benchmark.load(temp_path)

        # Verify data is preserved
        assert loaded._checkpoint.name == "Save/Load Test"
        assert loaded._checkpoint.description == "Testing save and load"
        assert len(loaded.get_question_ids()) == 1

        loaded_question = loaded.get_question(q_id)
        assert loaded_question["question"] == "What is AI?"
        assert loaded_question["raw_answer"] == "Artificial Intelligence"
        assert loaded_question["finished"] is True

        loaded_rubric = loaded.get_global_rubric()
        assert loaded_rubric is not None
        assert len(loaded_rubric.traits) == 1
        assert loaded_rubric.traits[0].name == "accuracy"

    finally:
        temp_path.unlink()


def test_json_ld_format():
    """Test that the output is valid JSON-LD."""
    benchmark = Benchmark.create("JSON-LD Test")

    benchmark.add_question(question="Test question", raw_answer="Test answer")

    # Save and verify JSON-LD structure
    with tempfile.NamedTemporaryFile(suffix=".jsonld", delete=False) as f:
        temp_path = Path(f.name)

    try:
        benchmark.save(temp_path)

        # Load as raw JSON
        with open(temp_path) as f:
            data = json.load(f)

        # Verify JSON-LD structure
        assert "@context" in data
        assert "@type" in data
        assert data["@type"] == "Dataset"
        assert "hasPart" in data
        assert len(data["hasPart"]) == 1

        # Verify question structure
        item = data["hasPart"][0]
        assert item["@type"] == "DataFeedItem"
        assert "item" in item
        assert item["item"]["@type"] == "Question"
        assert "acceptedAnswer" in item["item"]
        assert item["item"]["acceptedAnswer"]["@type"] == "Answer"
        assert "hasPart" in item["item"]
        assert item["item"]["hasPart"]["@type"] == "SoftwareSourceCode"

    finally:
        temp_path.unlink()


def test_get_finished_templates():
    """Test getting finished templates for verification."""
    benchmark = Benchmark.create("Test")

    # Add unfinished question
    benchmark.add_question(question="Unfinished question", raw_answer="Answer", finished=False)

    # Add finished question
    template = """class Answer(BaseAnswer):
    response: str = Field(description="Response")

    def verify(self) -> bool:
        return True
"""

    benchmark.add_question(question="Finished question", raw_answer="Answer", answer_template=template, finished=True)

    # Get finished templates
    templates = benchmark.get_finished_templates()
    assert len(templates) == 1
    assert templates[0].question_text == "Finished question"
    assert templates[0].template_code == template
    assert templates[0].finished is True


def test_validate_benchmark():
    """Test benchmark validation."""
    benchmark = Benchmark.create("Test")

    # Empty benchmark should be valid
    is_valid, msg = benchmark.validate()
    assert is_valid is True

    # Add valid question
    benchmark.add_question(question="Valid question", raw_answer="Valid answer")

    is_valid, msg = benchmark.validate()
    assert is_valid is True

    # Add question with invalid template
    q_id = benchmark.add_question(question="Question with bad template", raw_answer="Answer")

    # Directly modify the template to be invalid (bypassing validation)
    for item in benchmark._checkpoint.hasPart:
        if benchmark._get_item_id(item) == q_id:
            item.item.hasPart.text = ""  # Empty template should fail validation
            break

    benchmark._rebuild_cache()

    is_valid, msg = benchmark.validate()
    assert is_valid is False
    assert "Invalid template" in msg


def test_metadata_handling():
    """Test custom metadata and author/sources."""
    benchmark = Benchmark.create("Test")

    author = {"name": "John Doe", "email": "john@example.com"}
    sources = [
        {"title": "Python Documentation", "url": "https://docs.python.org"},
        {"title": "Wikipedia", "url": "https://wikipedia.org"},
    ]
    custom_meta = {"difficulty": "easy", "topic": "programming"}

    q_id = benchmark.add_question(
        question="What is Python?",
        raw_answer="A programming language",
        author=author,
        sources=sources,
        custom_metadata=custom_meta,
    )

    question = benchmark.get_question(q_id)
    assert question["author"] == author
    assert question["sources"] == sources
    assert question["custom_metadata"] == custom_meta

    # Test round-trip through save/load
    with tempfile.NamedTemporaryFile(suffix=".jsonld", delete=False) as f:
        temp_path = Path(f.name)

    try:
        benchmark.save(temp_path)
        loaded = Benchmark.load(temp_path)

        loaded_q = loaded.get_question(q_id)
        assert loaded_q["author"] == author
        assert loaded_q["sources"] == sources
        assert loaded_q["custom_metadata"] == custom_meta

    finally:
        temp_path.unlink()


def test_question_id_generation():
    """Test that question IDs are generated consistently."""
    benchmark = Benchmark.create("Test")

    # Add same question twice
    q1_id = benchmark.add_question(question="What is Python?", raw_answer="A language")

    q2_id = benchmark.add_question(
        question="What is Python?",  # Same question
        raw_answer="Different answer",
    )

    # IDs should be based on question text, so should be similar
    assert "python" in q1_id.lower()
    assert "python" in q2_id.lower()

    # But we should have 2 questions
    assert len(benchmark.get_question_ids()) == 2


def test_magic_methods():
    """Test magic methods for better usability."""
    benchmark = Benchmark.create("Test Benchmark", "Testing magic methods")

    # Test __str__ and __repr__
    str_repr = str(benchmark)
    assert "Test Benchmark" in str_repr
    assert "0 questions" in str_repr
    assert "0.0% complete" in str_repr

    repr_str = repr(benchmark)
    assert "Benchmark(" in repr_str
    assert "name='Test Benchmark'" in repr_str

    # Test __len__
    assert len(benchmark) == 0

    q1_id = benchmark.add_question("What is AI?", "Artificial Intelligence")
    q2_id = benchmark.add_question("What is ML?", "Machine Learning")

    assert len(benchmark) == 2

    # Test __contains__
    assert q1_id in benchmark
    assert q2_id in benchmark
    assert "nonexistent" not in benchmark

    # Test __getitem__
    question = benchmark[q1_id]
    assert question["question"] == "What is AI?"

    with pytest.raises(ValueError):
        _ = benchmark["nonexistent"]

    # Test __iter__
    questions_from_iter = list(benchmark)
    assert len(questions_from_iter) == 2
    assert all("question" in q for q in questions_from_iter)


def test_property_accessors():
    """Test property accessors for common attributes."""
    benchmark = Benchmark.create("Test", "Description", "2.0.0", "Test Creator")

    # Test getter properties
    assert benchmark.name == "Test"
    assert benchmark.description == "Description"
    assert benchmark.version == "2.0.0"
    assert benchmark.creator == "Test Creator"
    assert benchmark.question_count == 0
    assert benchmark.finished_count == 0
    assert benchmark.is_empty is True
    assert benchmark.is_complete is False

    # Test setter properties
    benchmark.name = "New Name"
    benchmark.description = "New Description"
    benchmark.version = "3.0.0"
    benchmark.creator = "New Creator"

    assert benchmark.name == "New Name"
    assert benchmark.description == "New Description"
    assert benchmark.version == "3.0.0"
    assert benchmark.creator == "New Creator"

    # Add questions and test counts
    q1_id = benchmark.add_question("Q1", "A1")
    _ = benchmark.add_question("Q2", "A2")

    assert benchmark.question_count == 2
    assert benchmark.finished_count == 0
    assert benchmark.is_empty is False
    assert benchmark.is_complete is False

    # Mark one as finished
    benchmark.mark_finished(q1_id)

    assert benchmark.finished_count == 1
    assert benchmark.is_complete is False


def test_statistics_and_summary():
    """Test statistics and summary methods."""
    benchmark = Benchmark.create("Stats Test")

    # Empty benchmark
    assert benchmark.get_progress() == 0.0
    summary = benchmark.get_summary()
    assert summary["question_count"] == 0
    assert summary["finished_count"] == 0
    assert summary["progress_percentage"] == 0.0

    # Add questions with templates
    q1_id = benchmark.add_question("Q1", "A1")
    _ = benchmark.add_question("Q2", "A2")

    template = """class Answer(BaseAnswer):
    response: str = Field(description="The response")

    def verify(self) -> bool:
        return True
"""
    benchmark.add_answer_template(q1_id, template)
    benchmark.mark_finished(q1_id)

    # Test progress
    assert benchmark.get_progress() == 50.0

    # Test summary
    summary = benchmark.get_summary()
    assert summary["question_count"] == 2
    assert summary["finished_count"] == 1
    assert summary["has_template_count"] == 1
    assert summary["progress_percentage"] == 50.0

    # Test detailed statistics
    stats = benchmark.get_statistics()
    assert "avg_template_length" in stats
    assert stats["avg_template_length"] > 0


def test_filtering_and_search():
    """Test filtering and search capabilities."""
    benchmark = Benchmark.create("Filter Test")

    # Add various questions
    q1_id = benchmark.add_question("What is Python programming?", "A language")
    _ = benchmark.add_question("What is Java?", "Another language")
    _ = benchmark.add_question("Explain Python decorators", "Functions that modify functions")

    # Add template to one question
    template = """class Answer(BaseAnswer):
    response: str = Field(description="The response")

    def verify(self) -> bool:
        return True
"""
    benchmark.add_answer_template(q1_id, template)
    benchmark.mark_finished(q1_id)

    # Test search
    python_questions = benchmark.search_questions("Python")
    assert len(python_questions) == 2

    java_questions = benchmark.search_questions("Java")
    assert len(java_questions) == 1

    # Test filtering
    finished = benchmark.filter_questions(finished=True)
    assert len(finished) == 1

    unfinished = benchmark.filter_questions(finished=False)
    assert len(unfinished) == 2

    with_template = benchmark.filter_questions(has_template=True)
    assert len(with_template) == 1

    without_template = benchmark.filter_questions(has_template=False)
    assert len(without_template) == 2

    # Test helper methods
    missing_templates = benchmark.get_missing_templates()
    assert len(missing_templates) == 2

    unfinished_questions = benchmark.get_unfinished_questions()
    assert len(unfinished_questions) == 2


def test_bulk_operations():
    """Test bulk operations."""
    benchmark = Benchmark.create("Bulk Test")

    # Test bulk question addition
    questions_data = [
        {"question": "Q1", "raw_answer": "A1"},
        {"question": "Q2", "raw_answer": "A2", "finished": True},
        {"question": "Q3", "raw_answer": "A3"},
    ]

    question_ids = benchmark.add_questions_batch(questions_data)
    assert len(question_ids) == 3
    assert len(benchmark) == 3

    # Test bulk status changes
    benchmark.mark_finished_batch([question_ids[0], question_ids[2]])
    finished = benchmark.filter_questions(finished=True)
    assert len(finished) == 3  # All should be finished now

    benchmark.mark_unfinished_batch([question_ids[1]])
    finished = benchmark.filter_questions(finished=True)
    assert len(finished) == 2

    # Test apply global template
    template = """class Answer(BaseAnswer):
    response: str = Field(description="Global template response")

    def verify(self) -> bool:
        return True
"""
    updated_ids = benchmark.apply_global_template(template)
    assert len(updated_ids) == 3  # All questions should get template

    with_templates = benchmark.filter_questions(has_template=True)
    assert len(with_templates) == 3


def test_template_management():
    """Test enhanced template management."""
    benchmark = Benchmark.create("Template Test")

    q1_id = benchmark.add_question("Q1", "A1")
    q2_id = benchmark.add_question("Q2", "A2")

    # Test has_template
    assert not benchmark.has_template(q1_id)
    assert not benchmark.has_template(q2_id)

    # Add template
    template1 = """class Answer(BaseAnswer):
    response1: str = Field(description="First response")

    def verify(self) -> bool:
        return True
"""
    benchmark.add_answer_template(q1_id, template1)

    assert benchmark.has_template(q1_id)
    assert not benchmark.has_template(q2_id)

    # Test get_template
    retrieved = benchmark.get_template(q1_id)
    assert retrieved == template1

    with pytest.raises(ValueError):
        benchmark.get_template(q2_id)

    # Test update_template (alias for add_answer_template)
    template2 = """class Answer(BaseAnswer):
    response2: str = Field(description="Second response")

    def verify(self) -> bool:
        return True
"""
    benchmark.update_template(q1_id, template2)
    assert benchmark.get_template(q1_id) == template2

    # Test copy_template
    benchmark.copy_template(q1_id, q2_id)
    assert benchmark.has_template(q2_id)
    assert benchmark.get_template(q2_id) == template2


def test_status_management():
    """Test status management methods."""
    benchmark = Benchmark.create("Status Test")

    q1_id = benchmark.add_question("Q1", "A1")
    _ = benchmark.add_question("Q2", "A2")

    # Initially unfinished
    assert benchmark.finished_count == 0

    # Test mark_finished
    benchmark.mark_finished(q1_id)
    assert benchmark.finished_count == 1

    # Test mark_unfinished
    benchmark.mark_unfinished(q1_id)
    assert benchmark.finished_count == 0

    # Test toggle_finished
    status = benchmark.toggle_finished(q1_id)
    assert status is True
    assert benchmark.finished_count == 1

    status = benchmark.toggle_finished(q1_id)
    assert status is False
    assert benchmark.finished_count == 0

    with pytest.raises(ValueError):
        benchmark.toggle_finished("nonexistent")


def test_clear_and_remove_operations():
    """Test clear and remove operations."""
    from karenina.schemas.rubric_class import RubricTrait

    benchmark = Benchmark.create("Clear Test")

    # Add content
    q1_id = benchmark.add_question("Q1", "A1")
    _ = benchmark.add_question("Q2", "A2")
    benchmark.add_global_rubric_trait(RubricTrait(name="test", description="test", kind="boolean"))

    assert len(benchmark) == 2

    # Test remove_question
    removed = benchmark.remove_question(q1_id)
    assert removed is True
    assert len(benchmark) == 1
    assert q1_id not in benchmark

    removed = benchmark.remove_question("nonexistent")
    assert removed is False

    # Test clear_questions
    count = benchmark.clear_questions()
    assert count == 1
    assert len(benchmark) == 0

    # Test clear_global_rubric
    cleared = benchmark.clear_global_rubric()
    assert cleared is True

    cleared = benchmark.clear_global_rubric()  # Already cleared
    assert cleared is False


def test_export_methods():
    """Test export capabilities."""
    benchmark = Benchmark.create("Export Test", "Test description")

    q1_id = benchmark.add_question("What is Python?", "A programming language")
    template = """class Answer(BaseAnswer):
    response: str = Field(description="Global template response")

    def verify(self) -> bool:
        return True
"""
    benchmark.add_answer_template(q1_id, template)
    benchmark.mark_finished(q1_id)

    # Test to_dict
    data = benchmark.to_dict()
    assert "metadata" in data
    assert "statistics" in data
    assert "questions" in data
    assert data["metadata"]["name"] == "Export Test"

    # Test to_csv
    csv_data = benchmark.to_csv()
    assert "Question ID" in csv_data
    assert "What is Python?" in csv_data

    # Test to_markdown
    markdown = benchmark.to_markdown()
    assert "# Export Test" in markdown
    assert "What is Python?" in markdown
    assert "✅" in markdown  # Finished indicator

    # Test clone
    cloned = benchmark.clone()
    assert cloned.name == benchmark.name
    assert len(cloned) == len(benchmark)
    assert cloned is not benchmark  # Different objects


def test_validation_and_health():
    """Test validation and health check methods."""
    benchmark = Benchmark.create("Health Test")

    # Empty benchmark health
    health = benchmark.get_health_report()
    assert health["health_score"] == 0.0
    assert health["health_status"] == "critical"

    # Add question
    q1_id = benchmark.add_question("Q1", "A1")

    # Test template validation
    valid_template = """class Answer(BaseAnswer):
    response: str = Field(description="Valid response")

    def verify(self) -> bool:
        return True
"""
    invalid_template = "class Answer(BaseAnswer: pass"  # Missing closing paren

    benchmark.add_answer_template(q1_id, valid_template)

    valid, errors = benchmark.validate_templates()
    assert valid is True
    assert len(errors) == 0

    # Test with invalid template - should raise exception during add
    with pytest.raises(ValueError, match="Invalid template"):
        benchmark.add_answer_template(q1_id, invalid_template)

    # Template should still be valid since invalid one was rejected
    valid, errors = benchmark.validate_templates()
    assert valid is True
    assert len(errors) == 0
    benchmark.mark_finished(q1_id)

    readiness = benchmark.check_readiness()
    assert readiness["ready_for_verification"] is True
    assert readiness["has_questions"] is True
    assert readiness["all_have_templates"] is True
    assert readiness["all_finished"] is True

    # Health should improve
    health = benchmark.get_health_report()
    assert health["health_score"] >= 90.0
    assert health["health_status"] == "excellent"


def test_comparison_methods():
    """Test benchmark comparison."""
    b1 = Benchmark.create("Test1", "Desc1", "1.0.0")
    b2 = Benchmark.create("Test1", "Desc1", "1.0.0")
    b3 = Benchmark.create("Test2", "Desc2", "2.0.0")

    # Empty benchmarks with same metadata should be equal
    assert b1 == b2
    assert b1 != b3

    # Add same content to both
    b1.add_question("Q1", "A1")
    b2.add_question("Q1", "A1")

    # Should still be equal (question IDs might differ but content is same)
    # Note: This tests structural equality, not exact ID equality
    assert b1.name == b2.name
    assert b1.question_count == b2.question_count

    # Add different content
    b1.add_question("Q2", "A2")

    # Should not be equal due to different question counts
    assert b1.question_count != b2.question_count


def test_metadata_properties():
    """Test all metadata property getters and setters."""
    benchmark = Benchmark.create("Original Name", "Original Description", "1.0.0", "Original Creator")

    # Test basic properties
    assert benchmark.name == "Original Name"
    assert benchmark.description == "Original Description"
    assert benchmark.version == "1.0.0"
    assert benchmark.creator == "Original Creator"

    # Test ID property
    assert benchmark.id is not None  # Auto-generated
    benchmark.id = "custom-benchmark-id"
    assert benchmark.id == "custom-benchmark-id"

    # Test timestamp properties
    _ = benchmark.created_at
    original_modified = benchmark.modified_at

    # Test setters update dateModified
    benchmark.name = "New Name"
    assert benchmark.name == "New Name"
    assert benchmark.modified_at != original_modified

    benchmark.description = "New Description"
    assert benchmark.description == "New Description"

    benchmark.version = "2.0.0"
    assert benchmark.version == "2.0.0"

    benchmark.creator = "New Creator"
    assert benchmark.creator == "New Creator"

    # Test timestamp setters
    custom_created = "2023-01-01T00:00:00.000000"
    custom_modified = "2023-12-31T23:59:59.999999"

    benchmark.created_at = custom_created
    benchmark.modified_at = custom_modified

    assert benchmark.created_at == custom_created
    assert benchmark.modified_at == custom_modified


def test_custom_properties():
    """Test custom property management."""
    benchmark = Benchmark.create("Custom Props Test")

    # Initially only has default benchmark_format_version property
    initial_props = benchmark.get_all_custom_properties()
    assert "benchmark_format_version" in initial_props
    assert benchmark.get_custom_property("nonexistent") is None

    # Set single property
    benchmark.set_custom_property("category", "machine_learning")
    assert benchmark.get_custom_property("category") == "machine_learning"

    # Set multiple properties
    props = {
        "difficulty": "advanced",
        "estimated_time": 45,
        "requires_gpu": True,
        "tags": ["classification", "neural_networks"],
    }
    benchmark.set_multiple_custom_properties(props)

    # Verify all properties
    all_props = benchmark.get_all_custom_properties()
    assert all_props["category"] == "machine_learning"
    assert all_props["difficulty"] == "advanced"
    assert all_props["estimated_time"] == 45
    assert all_props["requires_gpu"] is True
    assert all_props["tags"] == ["classification", "neural_networks"]

    # Update existing property
    benchmark.set_custom_property("difficulty", "expert")
    assert benchmark.get_custom_property("difficulty") == "expert"

    # Remove property
    removed = benchmark.remove_custom_property("estimated_time")
    assert removed is True
    assert benchmark.get_custom_property("estimated_time") is None

    # Try to remove non-existent property
    removed = benchmark.remove_custom_property("nonexistent")
    assert removed is False

    # Verify remaining properties
    remaining = benchmark.get_all_custom_properties()
    assert "estimated_time" not in remaining
    assert len(remaining) == 5  # benchmark_format_version, category, difficulty, requires_gpu, tags


def test_metadata_persistence():
    """Test metadata persistence through save/load."""
    import tempfile
    from pathlib import Path

    benchmark = Benchmark.create("Persistence Test")

    # Set various metadata
    benchmark.id = "test-benchmark-123"
    benchmark.description = "Testing metadata persistence"
    benchmark.version = "3.1.4"
    benchmark.creator = "Test Suite"

    # Set custom properties
    benchmark.set_multiple_custom_properties(
        {
            "domain": "natural_language_processing",
            "paper_reference": "https://arxiv.org/abs/1234.5678",
            "license": "MIT",
            "contributors": ["Alice", "Bob", "Charlie"],
        }
    )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jsonld", delete=False) as f:
        temp_path = Path(f.name)

    try:
        benchmark.save(temp_path)

        # Load and verify all metadata is preserved
        loaded = Benchmark.load(temp_path)

        assert loaded.id == "test-benchmark-123"
        assert loaded.name == "Persistence Test"
        assert loaded.description == "Testing metadata persistence"
        assert loaded.version == "3.1.4"
        assert loaded.creator == "Test Suite"
        assert loaded.created_at == benchmark.created_at
        assert loaded.modified_at == benchmark.modified_at

        # Verify custom properties
        loaded_props = loaded.get_all_custom_properties()
        assert loaded_props["domain"] == "natural_language_processing"
        assert loaded_props["paper_reference"] == "https://arxiv.org/abs/1234.5678"
        assert loaded_props["license"] == "MIT"
        assert loaded_props["contributors"] == ["Alice", "Bob", "Charlie"]

    finally:
        temp_path.unlink()


def test_question_metadata():
    """Test question metadata management."""
    benchmark = Benchmark.create("Question Metadata Test")

    # Add a question with metadata
    q_id = benchmark.add_question(
        question="What is machine learning?",
        raw_answer="ML is a subset of AI that enables computers to learn from data.",
        author={"name": "Dr. Smith", "email": "smith@university.edu"},
        sources=[
            {"title": "Introduction to ML", "url": "https://example.com/ml-intro"},
            {"title": "ML Textbook", "url": "https://example.com/ml-book"},
        ],
        custom_metadata={"difficulty": "beginner", "topic": "artificial_intelligence", "estimated_time": 15},
    )

    # Test get_question_metadata
    metadata = benchmark.get_question_metadata(q_id)
    assert metadata["id"] == q_id
    assert metadata["question"] == "What is machine learning?"
    assert metadata["raw_answer"] == "ML is a subset of AI that enables computers to learn from data."
    assert metadata["finished"] is False
    assert metadata["author"]["name"] == "Dr. Smith"
    assert len(metadata["sources"]) == 2
    assert metadata["custom_metadata"]["difficulty"] == "beginner"
    assert metadata["has_template"] is False  # Default template doesn't count
    assert metadata["has_rubric"] is False

    # Test timestamps
    timestamps = benchmark.get_question_timestamps(q_id)
    assert "created" in timestamps
    assert "modified" in timestamps

    # Test author access
    author = benchmark.get_question_author(q_id)
    assert author["name"] == "Dr. Smith"
    assert author["email"] == "smith@university.edu"

    # Test sources access
    sources = benchmark.get_question_sources(q_id)
    assert len(sources) == 2
    assert sources[0]["title"] == "Introduction to ML"

    # Test custom properties
    difficulty = benchmark.get_question_custom_property(q_id, "difficulty")
    assert difficulty == "beginner"

    topic = benchmark.get_question_custom_property(q_id, "topic")
    assert topic == "artificial_intelligence"

    # Test nonexistent property
    nonexistent = benchmark.get_question_custom_property(q_id, "nonexistent")
    assert nonexistent is None


def test_question_metadata_updates():
    """Test updating question metadata."""
    benchmark = Benchmark.create("Metadata Updates Test")

    q_id = benchmark.add_question("Original question", "Original answer")

    # Test basic field updates
    benchmark.update_question_metadata(
        q_id,
        question="Updated question text",
        raw_answer="Updated answer text",
        author={"name": "New Author", "affiliation": "New University"},
        sources=[{"title": "New Source", "url": "https://new-source.com"}],
    )

    metadata = benchmark.get_question_metadata(q_id)
    assert metadata["question"] == "Updated question text"
    assert metadata["raw_answer"] == "Updated answer text"
    assert metadata["author"]["name"] == "New Author"
    assert metadata["sources"][0]["title"] == "New Source"

    # Test custom metadata updates
    benchmark.update_question_metadata(
        q_id, custom_metadata={"new_field": "new_value", "complexity": "high", "validated": True}
    )

    assert benchmark.get_question_custom_property(q_id, "new_field") == "new_value"
    assert benchmark.get_question_custom_property(q_id, "complexity") == "high"
    assert benchmark.get_question_custom_property(q_id, "validated") is True

    # Test individual custom property operations
    benchmark.set_question_custom_property(q_id, "priority", "urgent")
    assert benchmark.get_question_custom_property(q_id, "priority") == "urgent"

    # Test removing custom property
    removed = benchmark.remove_question_custom_property(q_id, "complexity")
    assert removed is True
    assert benchmark.get_question_custom_property(q_id, "complexity") is None

    # Test removing nonexistent property
    removed = benchmark.remove_question_custom_property(q_id, "nonexistent")
    assert removed is False

    # Test author and sources setters
    benchmark.set_question_author(q_id, {"name": "Final Author", "role": "Expert"})
    author = benchmark.get_question_author(q_id)
    assert author["name"] == "Final Author"
    assert author["role"] == "Expert"

    benchmark.set_question_sources(
        q_id, [{"title": "Source A", "type": "paper"}, {"title": "Source B", "type": "book"}]
    )
    sources = benchmark.get_question_sources(q_id)
    assert len(sources) == 2
    assert sources[1]["title"] == "Source B"


def test_question_metadata_persistence():
    """Test question metadata persistence through save/load."""
    import tempfile
    from datetime import datetime
    from pathlib import Path

    benchmark = Benchmark.create("Question Persistence Test")

    # Add question with comprehensive metadata
    q_id = benchmark.add_question(
        question="What is deep learning?",
        raw_answer="Deep learning uses neural networks with multiple layers.",
        author={"name": "Prof. Johnson", "institution": "AI Institute"},
        sources=[
            {"title": "Deep Learning Book", "authors": ["Goodfellow", "Bengio", "Courville"]},
            {"title": "Nature Paper", "doi": "10.1038/nature123456"},
        ],
        custom_metadata={
            "field": "computer_science",
            "subfield": "machine_learning",
            "difficulty_level": "advanced",
            "prerequisites": ["linear_algebra", "calculus", "statistics"],
            "estimated_hours": 8.5,
        },
    )

    # Add template to make it more complete
    template = """class Answer(BaseAnswer):
    explanation: str = Field(description='Explanation')

    def verify(self) -> bool:
        return True
"""
    benchmark.add_answer_template(q_id, template)
    benchmark.mark_finished(q_id)

    # Get timestamps after all modifications
    original_timestamps = benchmark.get_question_timestamps(q_id)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jsonld", delete=False) as f:
        temp_path = Path(f.name)

    try:
        benchmark.save(temp_path)

        # Load and verify all question metadata is preserved
        loaded = Benchmark.load(temp_path)

        loaded_metadata = loaded.get_question_metadata(q_id)

        # Verify basic fields
        assert loaded_metadata["question"] == "What is deep learning?"
        assert loaded_metadata["raw_answer"] == "Deep learning uses neural networks with multiple layers."
        assert loaded_metadata["finished"] is True
        assert loaded_metadata["has_template"] is True

        # Verify author
        author = loaded.get_question_author(q_id)
        assert author["name"] == "Prof. Johnson"
        assert author["institution"] == "AI Institute"

        # Verify sources
        sources = loaded.get_question_sources(q_id)
        assert len(sources) == 2
        assert sources[0]["title"] == "Deep Learning Book"
        assert sources[1]["doi"] == "10.1038/nature123456"

        # Verify custom metadata
        assert loaded.get_question_custom_property(q_id, "field") == "computer_science"
        assert loaded.get_question_custom_property(q_id, "subfield") == "machine_learning"
        assert loaded.get_question_custom_property(q_id, "estimated_hours") == 8.5
        assert loaded.get_question_custom_property(q_id, "prerequisites") == [
            "linear_algebra",
            "calculus",
            "statistics",
        ]

        # Verify timestamps are preserved
        loaded_timestamps = loaded.get_question_timestamps(q_id)
        assert original_timestamps["created"] == loaded_timestamps["created"]
        # Modified timestamp may differ slightly due to save process, but should be close
        assert (
            abs(
                datetime.fromisoformat(original_timestamps["modified"]).timestamp()
                - datetime.fromisoformat(loaded_timestamps["modified"]).timestamp()
            )
            < 1.0
        )  # Within 1 second

    finally:
        temp_path.unlink()


def test_question_metadata_error_handling():
    """Test error handling for question metadata operations."""
    benchmark = Benchmark.create("Error Handling Test")

    # Test with nonexistent question ID
    with pytest.raises(ValueError, match="Question not found"):
        benchmark.get_question_metadata("nonexistent")

    with pytest.raises(ValueError, match="Question not found"):
        benchmark.update_question_metadata("nonexistent", question="Updated")

    with pytest.raises(ValueError, match="Question not found"):
        benchmark.set_question_custom_property("nonexistent", "key", "value")

    with pytest.raises(ValueError, match="Question not found"):
        benchmark.remove_question_custom_property("nonexistent", "key")

    with pytest.raises(ValueError, match="Question not found"):
        benchmark.get_question_author("nonexistent")

    with pytest.raises(ValueError, match="Question not found"):
        benchmark.get_question_sources("nonexistent")

    with pytest.raises(ValueError, match="Question not found"):
        benchmark.get_question_timestamps("nonexistent")


def test_benchmark_getitem_by_index():
    """Test accessing questions by integer index."""
    benchmark = Benchmark.create("Index Test")

    # Add some questions
    q1_id = benchmark.add_question("Question 1", "Answer 1")
    q2_id = benchmark.add_question("Question 2", "Answer 2")
    q3_id = benchmark.add_question("Question 3", "Answer 3")

    # Test positive indexing
    first_question = benchmark[0]
    assert first_question.text == "Question 1"
    assert first_question.acceptedAnswer.text == "Answer 1"
    assert first_question.id == q1_id

    second_question = benchmark[1]
    assert second_question.text == "Question 2"
    assert second_question.acceptedAnswer.text == "Answer 2"
    assert second_question.id == q2_id

    third_question = benchmark[2]
    assert third_question.text == "Question 3"
    assert third_question.acceptedAnswer.text == "Answer 3"
    assert third_question.id == q3_id

    # Test negative indexing
    last_question = benchmark[-1]
    assert last_question.text == "Question 3"
    assert last_question.id == q3_id

    second_to_last = benchmark[-2]
    assert second_to_last.text == "Question 2"
    assert second_to_last.id == q2_id


def test_benchmark_getitem_by_slice():
    """Test accessing questions by slice."""
    benchmark = Benchmark.create("Slice Test")

    # Add questions
    for i in range(5):
        benchmark.add_question(f"Question {i}", f"Answer {i}")

    # Test simple slice
    first_three = benchmark[:3]
    assert len(first_three) == 3
    assert first_three[0].text == "Question 0"
    assert first_three[1].text == "Question 1"
    assert first_three[2].text == "Question 2"

    # Test slice with start and end
    middle_slice = benchmark[1:4]
    assert len(middle_slice) == 3
    assert middle_slice[0].text == "Question 1"
    assert middle_slice[1].text == "Question 2"
    assert middle_slice[2].text == "Question 3"

    # Test slice with step
    every_other = benchmark[::2]
    assert len(every_other) == 3  # indices 0, 2, 4
    assert every_other[0].text == "Question 0"
    assert every_other[1].text == "Question 2"
    assert every_other[2].text == "Question 4"

    # Test negative slice
    last_two = benchmark[-2:]
    assert len(last_two) == 2
    assert last_two[0].text == "Question 3"
    assert last_two[1].text == "Question 4"


def test_benchmark_getitem_by_string_id():
    """Test accessing questions by string ID (existing behavior)."""
    benchmark = Benchmark.create("String ID Test")

    q_id = benchmark.add_question("Test Question", "Test Answer")

    # Access by string ID
    question = benchmark[q_id]
    assert question.text == "Test Question"
    assert question.acceptedAnswer.text == "Test Answer"
    assert question.id == q_id


def test_benchmark_getitem_edge_cases():
    """Test edge cases and error conditions for __getitem__."""
    benchmark = Benchmark.create("Edge Cases Test")

    # Test with empty benchmark
    with pytest.raises(IndexError, match="Question index 0 out of range"):
        benchmark[0]

    # Add a question
    benchmark.add_question("Only Question", "Only Answer")

    # Test out of bounds
    with pytest.raises(IndexError, match="Question index 1 out of range"):
        benchmark[1]

    with pytest.raises(IndexError, match="Question index -2 out of range"):
        benchmark[-2]

    # Test invalid key types
    with pytest.raises(TypeError, match="Invalid key type"):
        benchmark[1.5]  # float

    with pytest.raises(TypeError, match="Invalid key type"):
        benchmark[None]  # None

    # Test nonexistent string ID
    with pytest.raises(ValueError, match="Question not found"):
        benchmark["nonexistent-id"]


def test_benchmark_getitem_returns_schema_org_question():
    """Test that __getitem__ returns proper SchemaOrgQuestion objects."""
    from karenina.schemas.checkpoint import SchemaOrgQuestion

    benchmark = Benchmark.create("Schema Test")

    # Add question with custom metadata and template
    template_code = """from karenina.schemas import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    response: str = Field(description="The response")

    def verify(self) -> bool:
        return True
"""

    q_id = benchmark.add_question(
        question="Test Question with Template",
        raw_answer="Test Answer",
        answer_template=template_code,
        finished=True,
        custom_metadata={"category": "test", "difficulty": 5},
    )

    # Test by index
    question_by_index = benchmark[0]
    assert isinstance(question_by_index, SchemaOrgQuestion)
    assert question_by_index.text == "Test Question with Template"
    assert question_by_index.acceptedAnswer.text == "Test Answer"
    assert question_by_index.hasPart.text == template_code

    # Check additional properties (custom metadata)
    assert question_by_index.additionalProperty is not None
    prop_dict = {prop.name: prop.value for prop in question_by_index.additionalProperty}
    assert prop_dict["finished"] is True
    assert prop_dict["custom_category"] == "test"
    assert prop_dict["custom_difficulty"] == 5

    # Test by string ID
    question_by_id = benchmark[q_id]
    assert isinstance(question_by_id, SchemaOrgQuestion)
    assert question_by_id.text == question_by_index.text
    assert question_by_id.id == question_by_index.id

    # Test by slice
    questions_by_slice = benchmark[:1]
    assert len(questions_by_slice) == 1
    assert isinstance(questions_by_slice[0], SchemaOrgQuestion)
    assert questions_by_slice[0].text == "Test Question with Template"


def test_benchmark_getitem_with_rubric_traits():
    """Test __getitem__ with question-specific rubric traits."""
    benchmark = Benchmark.create("Rubric Test")

    # Add question
    q_id = benchmark.add_question("Question with Rubric", "Answer with Rubric")

    # Add question-specific rubric trait
    rubric_trait = RubricTrait(
        name="accuracy", description="How accurate is the answer?", kind="score", min_score=1, max_score=5
    )
    benchmark.add_question_rubric_trait(q_id, rubric_trait)

    # Test access
    question = benchmark[0]
    assert question.rating is not None
    assert len(question.rating) == 1
    assert question.rating[0].name == "accuracy"
    assert question.rating[0].description == "How accurate is the answer?"
    assert question.rating[0].bestRating == 5.0
    assert question.rating[0].worstRating == 1.0
    assert question.rating[0].additionalType == "QuestionSpecificRubricTrait"
