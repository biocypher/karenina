"""Unit tests for Benchmark filtering and querying methods.

Tests cover:
- filter_questions() with various criteria
- filter_by_metadata() with dot notation
- filter_by_custom_metadata() with AND/OR logic
- search_questions() with text search options
- get_questions_by_author()
- get_questions_with_rubric()
- get_finished_questions() and get_unfinished_questions()
- count_by_field() grouping
"""

import pytest

from karenina import Benchmark


@pytest.mark.unit
def test_filter_questions_finished_only() -> None:
    """Test filter_questions with finished=True."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)
    benchmark.add_question("Q3?", "A3", question_id="q3", finished=True)

    results = benchmark.filter_questions(finished=True)

    assert len(results) == 2
    assert any(q["id"] == "q1" for q in results)
    assert any(q["id"] == "q3" for q in results)
    assert not any(q["id"] == "q2" for q in results)


@pytest.mark.unit
def test_filter_questions_unfinished_only() -> None:
    """Test filter_questions with finished=False."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)
    benchmark.add_question("Q3?", "A3", question_id="q3", finished=False)

    results = benchmark.filter_questions(finished=False)

    assert len(results) == 2
    assert any(q["id"] == "q2" for q in results)
    assert any(q["id"] == "q3" for q in results)
    assert not any(q["id"] == "q1" for q in results)


@pytest.mark.unit
def test_filter_questions_has_template() -> None:
    """Test filter_questions with has_template=True."""
    benchmark = Benchmark.create(name="test")

    # Valid template with verify() method
    template = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The answer")

    def verify(self) -> bool:
        return self.value == "A1"
"""

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=True)
    benchmark.add_answer_template("q1", template)

    results = benchmark.filter_questions(has_template=True)

    assert len(results) == 1
    assert results[0]["id"] == "q1"


@pytest.mark.unit
def test_filter_questions_no_template() -> None:
    """Test filter_questions with has_template=False."""
    benchmark = Benchmark.create(name="test")

    # Valid template with verify() method
    template = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The answer")

    def verify(self) -> bool:
        return self.value == "A1"
"""

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=True)
    benchmark.add_answer_template("q1", template)

    results = benchmark.filter_questions(has_template=False)

    assert len(results) == 1
    assert results[0]["id"] == "q2"


@pytest.mark.unit
def test_filter_questions_by_author() -> None:
    """Test filter_questions with author filter."""
    benchmark = Benchmark.create(name="test")

    author1 = {"name": "Alice", "email": "alice@example.com"}
    author2 = {"name": "Bob", "email": "bob@example.com"}

    benchmark.add_question("Q1?", "A1", question_id="q1", author=author1)
    benchmark.add_question("Q2?", "A2", question_id="q2", author=author2)
    benchmark.add_question("Q3?", "A3", question_id="q3", author=author1)

    results = benchmark.filter_questions(author="Alice")

    assert len(results) == 2
    assert all(q["author"]["name"] == "Alice" for q in results)


@pytest.mark.unit
def test_filter_questions_combined_criteria() -> None:
    """Test filter_questions with multiple criteria."""
    benchmark = Benchmark.create(name="test")

    author = {"name": "Alice", "email": "alice@example.com"}
    # Valid template with verify() method
    template = """
from karenina.schemas.domain import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: str = Field(description="The answer")

    def verify(self) -> bool:
        return self.value == "A1"
"""

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True, author=author)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False, author=author)
    benchmark.add_answer_template("q1", template)

    # Filter for finished AND has_template AND author=Alice
    results = benchmark.filter_questions(finished=True, has_template=True, author="Alice")

    assert len(results) == 1
    assert results[0]["id"] == "q1"


@pytest.mark.unit
def test_filter_questions_no_criteria() -> None:
    """Test filter_questions with no criteria returns all questions."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)

    results = benchmark.filter_questions()

    assert len(results) == 2


@pytest.mark.unit
def test_get_finished_questions() -> None:
    """Test get_finished_questions returns only finished questions."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)
    benchmark.add_question("Q3?", "A3", question_id="q3", finished=True)

    results = benchmark.get_finished_questions()

    assert len(results) == 2
    assert all(q["finished"] is True for q in results)


@pytest.mark.unit
def test_get_finished_questions_ids_only() -> None:
    """Test get_finished_questions with ids_only=True."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)
    benchmark.add_question("Q3?", "A3", question_id="q3", finished=True)

    ids = benchmark.get_finished_questions(ids_only=True)

    assert isinstance(ids, list)
    assert len(ids) == 2
    assert "q1" in ids
    assert "q3" in ids
    assert "q2" not in ids


@pytest.mark.unit
def test_get_unfinished_questions() -> None:
    """Test get_unfinished_questions returns only unfinished questions."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)
    benchmark.add_question("Q3?", "A3", question_id="q3", finished=False)

    results = benchmark.get_unfinished_questions()

    assert len(results) == 2
    assert all(q.get("finished") is not True for q in results)


@pytest.mark.unit
def test_get_unfinished_questions_ids_only() -> None:
    """Test get_unfinished_questions with ids_only=True."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)
    benchmark.add_question("Q3?", "A3", question_id="q3", finished=False)

    ids = benchmark.get_unfinished_questions(ids_only=True)

    assert isinstance(ids, list)
    assert len(ids) == 2
    assert "q2" in ids
    assert "q3" in ids
    assert "q1" not in ids


@pytest.mark.unit
def test_get_questions_by_author() -> None:
    """Test get_questions_by_author filters by author name."""
    benchmark = Benchmark.create(name="test")

    author1 = {"name": "Alice", "email": "alice@example.com"}
    author2 = {"name": "Bob", "email": "bob@example.com"}

    benchmark.add_question("Q1?", "A1", question_id="q1", author=author1)
    benchmark.add_question("Q2?", "A2", question_id="q2", author=author2)
    benchmark.add_question("Q3?", "A3", question_id="q3", author=author1)

    results = benchmark.get_questions_by_author("Alice")

    assert len(results) == 2
    assert all(q["author"]["name"] == "Alice" for q in results)


@pytest.mark.unit
def test_get_questions_with_rubric() -> None:
    """Test get_questions_with_rubric returns questions with rubrics."""
    from karenina.schemas.domain import RegexTrait

    benchmark = Benchmark.create(name="test")

    # Add questions
    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=True)
    benchmark.add_question("Q3?", "A3", question_id="q3", finished=True)

    # Add rubric to q1 only
    trait = RegexTrait(name="test", pattern=r"\w+", higher_is_better=True)
    benchmark.add_question_rubric_trait("q1", trait)

    results = benchmark.get_questions_with_rubric()

    assert len(results) == 1
    assert results[0]["id"] == "q1"


@pytest.mark.unit
def test_search_questions_single_query() -> None:
    """Test search_questions with single query string."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("What is the capital of France?", "Paris", question_id="q1")
    benchmark.add_question("What is 2+2?", "4", question_id="q2")
    benchmark.add_question("Who wrote Hamlet?", "Shakespeare", question_id="q3")

    results = benchmark.search_questions("capital")

    assert len(results) == 1
    assert "capital" in results[0]["question"].lower()


@pytest.mark.unit
def test_search_questions_multiple_queries() -> None:
    """Test search_questions with list of queries."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("What is the capital of France?", "Paris", question_id="q1")
    benchmark.add_question("What is the population of Tokyo?", "37 million", question_id="q2")
    benchmark.add_question("Who wrote Hamlet?", "Shakespeare", question_id="q3")

    # Search for questions containing "capital" OR "population"
    results = benchmark.search_questions(["capital", "population"], match_all=False)

    assert len(results) == 2


@pytest.mark.unit
def test_search_questions_match_all() -> None:
    """Test search_questions with match_all=True."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("What is the capital of France and its population?", "Paris", question_id="q1")
    benchmark.add_question("What is the capital of France?", "Paris", question_id="q2")
    benchmark.add_question("What is the population of Tokyo?", "37 million", question_id="q3")

    # Must contain both "capital" and "population"
    results = benchmark.search_questions(["capital", "population"], match_all=True)

    assert len(results) == 1
    assert results[0]["id"] == "q1"


@pytest.mark.unit
def test_search_questions_case_insensitive() -> None:
    """Test search_questions is case-insensitive by default."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("What is the CAPITAL of France?", "Paris", question_id="q1")
    benchmark.add_question("What is 2+2?", "4", question_id="q2")

    results = benchmark.search_questions("capital")

    assert len(results) == 1


@pytest.mark.unit
def test_search_questions_case_sensitive() -> None:
    """Test search_questions with case_sensitive=True."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("What is the CAPITAL of France?", "Paris", question_id="q1")
    benchmark.add_question("What is the capital of France?", "Paris", question_id="q2")

    results = benchmark.search_questions("CAPITAL", case_sensitive=True)

    assert len(results) == 1
    assert results[0]["id"] == "q1"


@pytest.mark.unit
def test_search_questions_specific_fields() -> None:
    """Test search_questions with specific fields."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("What is the capital of France?", "Paris is the capital", question_id="q1")
    benchmark.add_question("What is 2+2?", "Four", question_id="q2")

    # Search only in raw_answer field
    results = benchmark.search_questions("capital", fields=["raw_answer"])

    assert len(results) == 1
    assert "capital" in results[0]["raw_answer"].lower()


@pytest.mark.unit
def test_search_questions_regex() -> None:
    """Test search_questions with regex=True."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("What is the price of item ABC123?", "$10", question_id="q1")
    benchmark.add_question("What is the price of item XYZ?", "$20", question_id="q2")
    benchmark.add_question("What is 2+2?", "4", question_id="q3")

    # Search for alphanumeric product codes
    results = benchmark.search_questions(r"\b[A-Z]{3}\d{3}\b", regex=True)

    assert len(results) == 1
    assert results[0]["id"] == "q1"


@pytest.mark.unit
def test_filter_by_metadata_field() -> None:
    """Test filter_by_metadata with dot notation."""
    benchmark = Benchmark.create(name="test")

    author1 = {"name": "Alice", "email": "alice@example.com"}
    author2 = {"name": "Bob", "email": "bob@example.com"}

    benchmark.add_question("Q1?", "A1", question_id="q1", author=author1)
    benchmark.add_question("Q2?", "A2", question_id="q2", author=author2)

    results = benchmark.filter_by_metadata("author.name", "Alice")

    assert len(results) == 1
    assert results[0]["author"]["name"] == "Alice"


@pytest.mark.unit
def test_filter_by_metadata_nested_field() -> None:
    """Test filter_by_metadata with nested dot notation."""
    benchmark = Benchmark.create(name="test")

    custom_meta1 = {"difficulty": "easy", "category": {"name": "math", "level": 1}}
    custom_meta2 = {"difficulty": "hard", "category": {"name": "literature", "level": 3}}

    benchmark.add_question("Q1?", "A1", question_id="q1", custom_metadata=custom_meta1)
    benchmark.add_question("Q2?", "A2", question_id="q2", custom_metadata=custom_meta2)

    results = benchmark.filter_by_metadata("custom_metadata.category.name", "math")

    assert len(results) == 1
    assert results[0]["id"] == "q1"


@pytest.mark.unit
def test_filter_by_metadata_contains_mode() -> None:
    """Test filter_by_metadata with match_mode='contains'."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question(
        "What is Python?",
        "A language",
        question_id="q1",
        custom_metadata={"category": "programming language"},
    )
    benchmark.add_question(
        "What is Java?",
        "A language",
        question_id="q2",
        custom_metadata={"category": "language"},
    )

    results = benchmark.filter_by_metadata("custom_metadata.category", "programming", match_mode="contains")

    assert len(results) == 1
    assert results[0]["id"] == "q1"


@pytest.mark.unit
def test_filter_by_custom_metadata_and() -> None:
    """Test filter_by_custom_metadata with AND logic (default)."""
    benchmark = Benchmark.create(name="test")

    custom_meta1 = {"difficulty": "easy", "category": "math"}
    custom_meta2 = {"difficulty": "easy", "category": "literature"}
    custom_meta3 = {"difficulty": "hard", "category": "math"}

    benchmark.add_question("Q1?", "A1", question_id="q1", custom_metadata=custom_meta1)
    benchmark.add_question("Q2?", "A2", question_id="q2", custom_metadata=custom_meta2)
    benchmark.add_question("Q3?", "A3", question_id="q3", custom_metadata=custom_meta3)

    # Must have both difficulty=easy AND category=math
    results = benchmark.filter_by_custom_metadata(difficulty="easy", category="math")

    assert len(results) == 1
    assert results[0]["id"] == "q1"


@pytest.mark.unit
def test_filter_by_custom_metadata_or() -> None:
    """Test filter_by_custom_metadata with OR logic."""
    benchmark = Benchmark.create(name="test")

    custom_meta1 = {"difficulty": "easy", "category": "math"}
    custom_meta2 = {"difficulty": "hard", "category": "literature"}
    custom_meta3 = {"difficulty": "medium", "category": "science"}

    benchmark.add_question("Q1?", "A1", question_id="q1", custom_metadata=custom_meta1)
    benchmark.add_question("Q2?", "A2", question_id="q2", custom_metadata=custom_meta2)
    benchmark.add_question("Q3?", "A3", question_id="q3", custom_metadata=custom_meta3)

    # Must have difficulty=easy OR category=literature
    results = benchmark.filter_by_custom_metadata(match_all=False, difficulty="easy", category="literature")

    assert len(results) == 2
    question_ids = {q["id"] for q in results}
    assert question_ids == {"q1", "q2"}


@pytest.mark.unit
def test_count_by_field() -> None:
    """Test count_by_field groups by field value."""
    benchmark = Benchmark.create(name="test")

    author1 = {"name": "Alice", "email": "alice@example.com"}
    author2 = {"name": "Bob", "email": "bob@example.com"}

    benchmark.add_question("Q1?", "A1", question_id="q1", author=author1)
    benchmark.add_question("Q2?", "A2", question_id="q2", author=author2)
    benchmark.add_question("Q3?", "A3", question_id="q3", author=author1)

    counts = benchmark.count_by_field("author.name")

    assert counts.get("Alice") == 2
    assert counts.get("Bob") == 1


@pytest.mark.unit
def test_count_by_field_nested() -> None:
    """Test count_by_field with nested custom_metadata."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question(
        "Q1?",
        "A1",
        question_id="q1",
        custom_metadata={"category": {"name": "math", "level": 1}},
    )
    benchmark.add_question(
        "Q2?",
        "A2",
        question_id="q2",
        custom_metadata={"category": {"name": "literature", "level": 2}},
    )
    benchmark.add_question(
        "Q3?",
        "A3",
        question_id="q3",
        custom_metadata={"category": {"name": "math", "level": 1}},
    )

    counts = benchmark.count_by_field("custom_metadata.category.name")

    assert counts.get("math") == 2
    assert counts.get("literature") == 1


@pytest.mark.unit
def test_count_by_field_with_subset() -> None:
    """Test count_by_field with subset of questions."""
    benchmark = Benchmark.create(name="test")

    author1 = {"name": "Alice", "email": "alice@example.com"}
    author2 = {"name": "Bob", "email": "bob@example.com"}

    benchmark.add_question("Q1?", "A1", question_id="q1", author=author1)
    benchmark.add_question("Q2?", "A2", question_id="q2", author=author2)
    benchmark.add_question("Q3?", "A3", question_id="q3", author=author1)

    all_questions = benchmark.get_all_questions()
    subset = all_questions[:2]

    counts = benchmark.count_by_field("author.name", questions=subset)

    # Only count from the subset (q1=Alice, q2=Bob)
    assert counts.get("Alice") == 1
    assert counts.get("Bob") == 1


@pytest.mark.unit
def test_filter_questions_custom_filter() -> None:
    """Test filter_questions with custom_filter callable."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Short?", "A", question_id="q1", finished=True)
    benchmark.add_question("This is a very long question text?", "B", question_id="q2", finished=True)
    benchmark.add_question("Medium length?", "C", question_id="q3", finished=True)

    # Custom filter: only questions with text < 20 chars
    results = benchmark.filter_questions(finished=True, custom_filter=lambda q: len(q.get("question", "")) < 20)

    assert len(results) == 2
    question_ids = {q["id"] for q in results}
    assert question_ids == {"q1", "q3"}


@pytest.mark.unit
def test_get_all_questions_ids_only() -> None:
    """Test get_all_questions with ids_only=True."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)

    ids = benchmark.get_all_questions(ids_only=True)

    assert isinstance(ids, list)
    assert len(ids) == 2
    assert "q1" in ids
    assert "q2" in ids


@pytest.mark.unit
def test_get_all_questions_full_objects() -> None:
    """Test get_all_questions with ids_only=False (default)."""
    benchmark = Benchmark.create(name="test")

    benchmark.add_question("Q1?", "A1", question_id="q1", finished=True)
    benchmark.add_question("Q2?", "A2", question_id="q2", finished=False)

    questions = benchmark.get_all_questions(ids_only=False)

    assert isinstance(questions, list)
    assert len(questions) == 2
    assert all(isinstance(q, dict) for q in questions)
    assert questions[0]["id"] == "q1"
    assert questions[1]["id"] == "q2"
