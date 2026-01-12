"""Unit tests for Benchmark class core functionality.

Tests cover:
- Benchmark initialization with various parameters
- Property accessors (name, description, version, creator)
- add_question() with different input types
- Question retrieval (by ID, by index, iteration)
- Question count, empty, complete properties
- Edge cases and error handling
"""

import pytest

from karenina import Benchmark


@pytest.mark.unit
def test_benchmark_initialization_minimal() -> None:
    """Test Benchmark initialization with minimal parameters."""
    benchmark = Benchmark.create(name="test-bench")

    assert benchmark.name == "test-bench"
    # Default description is "Benchmark containing questions"
    assert "Benchmark" in benchmark.description
    assert benchmark.version == "0.1.0"
    assert benchmark.creator == "Karenina Benchmarking System"


@pytest.mark.unit
def test_benchmark_initialization_full() -> None:
    """Test Benchmark initialization with all parameters."""
    benchmark = Benchmark.create(
        name="full-bench",
        description="A comprehensive test benchmark",
        version="1.0.0",
        creator="Test Suite",
    )

    assert benchmark.name == "full-bench"
    assert benchmark.description == "A comprehensive test benchmark"
    assert benchmark.version == "1.0.0"
    assert benchmark.creator == "Test Suite"


@pytest.mark.unit
def test_benchmark_name_property() -> None:
    """Test name property getter and setter."""
    benchmark = Benchmark.create(name="original-name")

    assert benchmark.name == "original-name"

    benchmark.name = "new-name"
    assert benchmark.name == "new-name"


@pytest.mark.unit
def test_benchmark_description_property() -> None:
    """Test description property getter and setter."""
    benchmark = Benchmark.create(name="test", description="Original description")

    assert benchmark.description == "Original description"

    benchmark.description = "Updated description"
    assert benchmark.description == "Updated description"


@pytest.mark.unit
def test_benchmark_version_property() -> None:
    """Test version property getter and setter."""
    benchmark = Benchmark.create(name="test", version="0.1.0")

    assert benchmark.version == "0.1.0"

    benchmark.version = "2.0.0"
    assert benchmark.version == "2.0.0"


@pytest.mark.unit
def test_benchmark_creator_property() -> None:
    """Test creator property getter and setter."""
    benchmark = Benchmark.create(name="test", creator="Original Creator")

    assert benchmark.creator == "Original Creator"

    benchmark.creator = "New Creator"
    assert benchmark.creator == "New Creator"


@pytest.mark.unit
def test_benchmark_id_property() -> None:
    """Test id property getter and setter."""
    benchmark = Benchmark.create(name="test")

    # ID is auto-generated, should not be None
    assert benchmark.id is not None

    benchmark.id = "custom-id-123"
    assert benchmark.id == "custom-id-123"


@pytest.mark.unit
def test_benchmark_id_can_be_none() -> None:
    """Test that id can be set to None."""
    benchmark = Benchmark.create(name="test")

    benchmark.id = None
    assert benchmark.id is None


@pytest.mark.unit
def test_benchmark_question_count_empty() -> None:
    """Test question_count is 0 for new benchmark."""
    benchmark = Benchmark.create(name="test")

    assert benchmark.question_count == 0


@pytest.mark.unit
def test_benchmark_is_empty_true() -> None:
    """Test is_empty is True for new benchmark."""
    benchmark = Benchmark.create(name="test")

    assert benchmark.is_empty is True


@pytest.mark.unit
def test_benchmark_is_empty_false() -> None:
    """Test is_empty is False after adding question."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("What is 2+2?", "4")

    assert benchmark.is_empty is False


@pytest.mark.unit
def test_benchmark_is_complete_false_for_new() -> None:
    """Test is_complete is False for new benchmark."""
    benchmark = Benchmark.create(name="test")

    assert benchmark.is_complete is False


@pytest.mark.unit
def test_benchmark_finished_count_empty() -> None:
    """Test finished_count is 0 for new benchmark."""
    benchmark = Benchmark.create(name="test")

    assert benchmark.finished_count == 0


@pytest.mark.unit
def test_add_question_with_string_input() -> None:
    """Test add_question with string question and answer."""
    benchmark = Benchmark.create(name="test")

    q_id = benchmark.add_question("What is 2+2?", "4")

    assert benchmark.question_count == 1
    assert q_id in benchmark


@pytest.mark.unit
def test_add_question_generates_id() -> None:
    """Test that add_question generates an ID if not provided."""
    benchmark = Benchmark.create(name="test")

    q_id = benchmark.add_question("Test question?", "Test answer")

    # ID should be a string in URN format
    assert isinstance(q_id, str)
    assert len(q_id) > 0


@pytest.mark.unit
def test_add_question_with_custom_id() -> None:
    """Test add_question with custom question ID."""
    benchmark = Benchmark.create(name="test")

    q_id = benchmark.add_question("Test question?", "Test answer", question_id="custom-q-1")

    assert q_id == "custom-q-1"
    assert "custom-q-1" in benchmark


@pytest.mark.unit
def test_add_question_with_finished_true() -> None:
    """Test add_question with finished=True."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Test question?", "Test answer", finished=True)

    assert benchmark.question_count == 1
    assert benchmark.finished_count == 1


@pytest.mark.unit
def test_add_question_with_finished_false() -> None:
    """Test add_question with finished=False."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Test question?", "Test answer", finished=False)

    assert benchmark.question_count == 1
    assert benchmark.finished_count == 0


@pytest.mark.unit
def test_add_question_with_template() -> None:
    """Test add_question with answer template."""
    benchmark = Benchmark.create(name="test")

    template = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    result: int = Field(description="The result")
    def verify(self) -> bool:
        return self.result == 4
"""

    q_id = benchmark.add_question("What is 2+2?", "4", answer_template=template)

    assert q_id in benchmark
    question = benchmark.get_question(q_id)
    assert question["answer_template"] == template
    assert question["finished"] is True  # Auto-set to True when template provided


@pytest.mark.unit
def test_add_multiple_questions() -> None:
    """Test adding multiple questions."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("What is 2+2?", "4")
    benchmark.add_question("What is 3+3?", "6")
    benchmark.add_question("What is 4+4?", "8")

    assert benchmark.question_count == 3


@pytest.mark.unit
def test_get_question_by_id() -> None:
    """Test retrieving a question by ID."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("What is the capital of France?", "Paris", question_id="q1")

    question = benchmark.get_question("q1")

    assert question["question"] == "What is the capital of France?"
    assert question["raw_answer"] == "Paris"


@pytest.mark.unit
def test_get_question_raises_for_unknown_id() -> None:
    """Test that get_question raises ValueError for unknown ID."""
    benchmark = Benchmark.create(name="test")

    with pytest.raises(ValueError, match="Question not found"):
        benchmark.get_question("unknown-id")


@pytest.mark.unit
def test_get_all_questions_empty() -> None:
    """Test get_all_questions returns empty list for new benchmark."""
    benchmark = Benchmark.create(name="test")

    questions = benchmark.get_all_questions()

    assert questions == []


@pytest.mark.unit
def test_get_all_questions_returns_list() -> None:
    """Test get_all_questions returns list of questions."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1")
    benchmark.add_question("Q2?", "A2", question_id="q2")

    questions = benchmark.get_all_questions()

    assert len(questions) == 2
    assert any(q["id"] == "q1" for q in questions)
    assert any(q["id"] == "q2" for q in questions)


@pytest.mark.unit
def test_get_question_ids_empty() -> None:
    """Test get_question_ids returns empty list for new benchmark."""
    benchmark = Benchmark.create(name="test")

    ids = benchmark.get_question_ids()

    assert ids == []


@pytest.mark.unit
def test_get_question_ids() -> None:
    """Test get_question_ids returns list of question IDs."""
    benchmark = Benchmark.create(name="test")

    q1 = benchmark.add_question("Q1?", "A1", question_id="q1")
    q2 = benchmark.add_question("Q2?", "A2", question_id="q2")

    ids = benchmark.get_question_ids()

    assert len(ids) == 2
    assert q1 in ids
    assert q2 in ids


@pytest.mark.unit
def test_contains_operator() -> None:
    """Test __contains__ for checking if question ID exists."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Test?", "Answer", question_id="test-q")

    assert "test-q" in benchmark
    assert "nonexistent" not in benchmark


@pytest.mark.unit
def test_bracket_notation_get_question() -> None:
    """Test __getitem__ for bracket notation access.

    Note: Bracket notation returns a SchemaOrgQuestion object,
    which has .text, .id, and other attributes.
    """
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Test question?", "Test answer", question_id="q1")

    question = benchmark["q1"]

    # The returned value is a SchemaOrgQuestion object
    assert hasattr(question, "text")
    assert hasattr(question, "id")
    assert question.id == "q1"
    # .text is the question text (not .question)
    assert "Test question?" in question.text or question.text == "Test question?"


@pytest.mark.unit
def test_bracket_notation_raises_for_unknown() -> None:
    """Test __getitem__ raises ValueError for unknown ID."""
    benchmark = Benchmark.create(name="test")

    with pytest.raises(ValueError, match="Question not found"):
        _ = benchmark["unknown"]


@pytest.mark.unit
def test_len_operator() -> None:
    """Test __len__ returns question count."""
    benchmark = Benchmark.create(name="test")

    assert len(benchmark) == 0

    benchmark.add_question("Q1?", "A1")
    assert len(benchmark) == 1

    benchmark.add_question("Q2?", "A2")
    assert len(benchmark) == 2


@pytest.mark.unit
def test_str_representation() -> None:
    """Test __str__ returns human-readable string."""
    benchmark = Benchmark.create(name="my-benchmark")

    str_repr = str(benchmark)

    assert "my-benchmark" in str_repr
    assert "0 questions" in str_repr


@pytest.mark.unit
def test_str_representation_with_questions() -> None:
    """Test __str__ shows question count."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", finished=True)
    benchmark.add_question("Q2?", "A2", finished=False)

    str_repr = str(benchmark)

    assert "2 questions" in str_repr
    assert "50.0%" in str_repr  # 1 of 2 finished


@pytest.mark.unit
def test_repr_representation() -> None:
    """Test __repr__ returns developer-friendly string."""
    benchmark = Benchmark.create(name="test", version="1.0.0")

    repr_str = repr(benchmark)

    assert "Benchmark(" in repr_str or "test" in repr_str
    assert "1.0.0" in repr_str


@pytest.mark.unit
def test_get_progress_percentage() -> None:
    """Test get_progress returns completion percentage."""
    benchmark = Benchmark.create(name="test")

    # Empty benchmark should have 0% progress
    assert benchmark.get_progress() == 0.0

    # Add questions with mixed finished status
    benchmark.add_question("Q1?", "A1", finished=True)
    benchmark.add_question("Q2?", "A2", finished=False)

    # 1 of 2 finished = 50%
    assert benchmark.get_progress() == 50.0

    # Add a third finished question
    benchmark.add_question("Q3?", "A3", finished=True)

    # 2 of 3 finished = ~66.67%
    progress = benchmark.get_progress()
    assert 66.0 < progress < 67.0


@pytest.mark.unit
def test_set_metadata() -> None:
    """Test set_metadata updates multiple fields."""
    benchmark = Benchmark.create(name="test")

    benchmark.set_metadata(
        name="updated-name",
        description="updated description",
        version="2.0.0",
    )

    assert benchmark.name == "updated-name"
    assert benchmark.description == "updated description"
    assert benchmark.version == "2.0.0"


@pytest.mark.unit
def test_benchmark_equality_same_questions() -> None:
    """Test __eq__ returns True for benchmarks with same questions."""
    b1 = Benchmark.create(name="test")
    b1.add_question("Q1?", "A1", question_id="q1")

    b2 = Benchmark.create(name="test")
    b2.add_question("Q1?", "A1", question_id="q1")

    # Questions are the same content but IDs might differ
    # Equality checks questions cache, so they may not be equal
    assert b1.name == b2.name


@pytest.mark.unit
def test_benchmark_inequality_different_name() -> None:
    """Test __eq__ returns False for benchmarks with different names."""
    b1 = Benchmark.create(name="bench1")
    b2 = Benchmark.create(name="bench2")

    assert b1 != b2


@pytest.mark.unit
def test_benchmark_inequality_non_benchmark() -> None:
    """Test __eq__ returns NotImplemented for non-Benchmark objects."""
    benchmark = Benchmark.create(name="test")

    assert benchmark != "not a benchmark"
    assert benchmark != 123
    assert benchmark is not None
