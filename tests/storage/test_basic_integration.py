"""Basic integration tests for database storage functionality.

Tests the core workflow: create benchmark → save to DB → load from DB → verify.
"""

import tempfile
from pathlib import Path

import pytest

from karenina.benchmark import Benchmark
from karenina.schemas.question_class import Question
from karenina.storage import DBConfig, load_benchmark, save_benchmark


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield f"sqlite:///{db_path}"

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


def test_save_and_load_simple_benchmark(temp_db: str):
    """Test saving and loading a simple benchmark with questions."""
    # Create a benchmark
    benchmark = Benchmark.create(
        name="Test Benchmark",
        description="A test benchmark for database storage",
        version="1.0.0",
        creator="Test Suite",
    )

    # Add some questions
    q1 = Question(
        question="What is 2 + 2?",
        raw_answer="4",
        tags=["math", "basic"],
    )
    benchmark.add_question(q1)

    q2 = Question(
        question="What is the capital of France?",
        raw_answer="Paris",
        tags=["geography"],
    )
    benchmark.add_question(q2)

    # Save to database
    save_benchmark(benchmark, temp_db)

    # Load from database
    loaded_benchmark = load_benchmark("Test Benchmark", temp_db)

    # Verify the loaded benchmark
    assert loaded_benchmark.name == "Test Benchmark"
    assert loaded_benchmark.description == "A test benchmark for database storage"
    assert loaded_benchmark.version == "1.0.0"

    # Verify questions were loaded
    questions = loaded_benchmark.get_all_questions()
    assert len(questions) == 2

    # Find and verify each question
    q1_loaded = next((q for q in questions if "2 + 2" in q["question"]), None)
    assert q1_loaded is not None
    assert q1_loaded["raw_answer"] == "4"

    q2_loaded = next((q for q in questions if "France" in q["question"]), None)
    assert q2_loaded is not None
    assert q2_loaded["raw_answer"] == "Paris"


def test_save_benchmark_with_template(temp_db: str):
    """Test saving and loading a benchmark with answer templates."""
    # Create benchmark
    benchmark = Benchmark.create(
        name="Template Test",
        description="Testing template storage",
    )

    # Add question with template
    template_code = """class Answer(BaseAnswer):
    value: int

    def model_post_init(self, __context):
        self.correct = {"value": 4}

    def verify(self) -> bool:
        return self.value == self.correct["value"]
"""

    q = Question(question="What is 2+2?", raw_answer="4")
    q_id = benchmark.add_question(q, answer_template=template_code, finished=True)

    # Save to database
    save_benchmark(benchmark, temp_db)

    # Load from database
    loaded_benchmark = load_benchmark("Template Test", temp_db)

    # Verify template was saved
    question = loaded_benchmark.get_question(q_id)
    assert question["answer_template"] is not None
    assert "def verify" in question["answer_template"]
    assert question["finished"] is True


def test_update_existing_benchmark(temp_db: str):
    """Test updating an existing benchmark in the database."""
    # Create and save initial benchmark
    benchmark = Benchmark.create(name="Update Test", version="1.0.0")
    benchmark.add_question("Question 1?", "Answer 1")
    save_benchmark(benchmark, temp_db)

    # Modify and save again
    benchmark.description = "Updated description"
    benchmark.add_question("Question 2?", "Answer 2")
    save_benchmark(benchmark, temp_db)

    # Load and verify
    loaded = load_benchmark("Update Test", temp_db)
    assert loaded.description == "Updated description"
    assert len(loaded.get_all_questions()) == 2


def test_db_config_object(temp_db: str):
    """Test using DBConfig object instead of string."""
    db_config = DBConfig(storage_url=temp_db, auto_create=True)

    benchmark = Benchmark.create(name="Config Test")
    benchmark.add_question("Test?", "Yes")

    # Save using DBConfig
    save_benchmark(benchmark, db_config)

    # Load using DBConfig
    loaded = load_benchmark("Config Test", db_config)
    assert loaded.name == "Config Test"


def test_load_with_config_flag(temp_db: str):
    """Test loading benchmark with load_config=True."""
    benchmark = Benchmark.create(name="Config Flag Test")
    benchmark.add_question("Test?", "Answer")
    save_benchmark(benchmark, temp_db)

    # Load with config flag
    loaded, db_config = load_benchmark("Config Flag Test", temp_db, load_config=True)

    assert loaded.name == "Config Flag Test"
    assert isinstance(db_config, DBConfig)
    assert db_config.storage_url == temp_db


def test_benchmark_save_to_db_method(temp_db: str):
    """Test the Benchmark.save_to_db() method."""
    benchmark = Benchmark.create(name="Method Test")
    benchmark.add_question("Q?", "A")

    # Use the save_to_db method
    result = benchmark.save_to_db(temp_db)
    assert result is benchmark  # Should return self for chaining

    # Verify it was saved
    loaded = Benchmark.load_from_db("Method Test", temp_db)
    assert loaded.name == "Method Test"


def test_load_nonexistent_benchmark(temp_db: str):
    """Test loading a benchmark that doesn't exist."""
    with pytest.raises(ValueError, match="not found"):
        load_benchmark("Nonexistent", temp_db)
