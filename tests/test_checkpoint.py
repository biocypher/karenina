"""Tests for checkpoint functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from karenina.benchmark.checkpoint import Checkpoint
from karenina.schemas.rubric_class import RubricTrait


def test_create_checkpoint():
    """Test creating a new checkpoint."""
    checkpoint = Checkpoint.create(
        name="Test Benchmark", description="A test benchmark", version="1.0.0", creator="Test Suite"
    )

    assert checkpoint._checkpoint.name == "Test Benchmark"
    assert checkpoint._checkpoint.description == "A test benchmark"
    assert checkpoint._checkpoint.version == "1.0.0"
    assert checkpoint._checkpoint.creator == "Test Suite"
    assert len(checkpoint.get_question_ids()) == 0


def test_add_question():
    """Test adding questions to a checkpoint."""
    checkpoint = Checkpoint.create("Test")

    # Add a question with minimal information
    q_id = checkpoint.add_question(question="What is Python?", raw_answer="Python is a programming language")

    assert q_id in checkpoint.get_question_ids()
    question = checkpoint.get_question(q_id)
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

    q_id2 = checkpoint.add_question(
        question="Name a dynamic language", raw_answer="Python", answer_template=template_code, finished=True
    )

    question2 = checkpoint.get_question(q_id2)
    assert question2["answer_template"] == template_code
    assert question2["finished"] is True


def test_add_answer_template():
    """Test adding/updating answer templates."""
    checkpoint = Checkpoint.create("Test")

    q_id = checkpoint.add_question(question="What is recursion?", raw_answer="A function that calls itself")

    # Update with a proper template (no imports needed)
    template = """class Answer(BaseAnswer):
    definition: str = Field(description="Definition of recursion")
    example: str = Field(description="Example of recursion")

    def verify(self) -> bool:
        return "itself" in self.definition.lower()
"""

    checkpoint.add_answer_template(q_id, template)

    question = checkpoint.get_question(q_id)
    assert question["answer_template"] == template

    # Test invalid template
    invalid_template = "not a valid python class"
    with pytest.raises(ValueError, match="Invalid template"):
        checkpoint.add_answer_template(q_id, invalid_template)


def test_rubric_management():
    """Test global and question-specific rubrics."""
    checkpoint = Checkpoint.create("Test")

    # Add global rubric trait
    global_trait = RubricTrait(name="clarity", description="Is the response clear?", kind="boolean")
    checkpoint.add_global_rubric_trait(global_trait)

    rubric = checkpoint.get_global_rubric()
    assert rubric is not None
    assert len(rubric.traits) == 1
    assert rubric.traits[0].name == "clarity"

    # Add question with question-specific rubric
    q_id = checkpoint.add_question(
        question="Explain quicksort", raw_answer="Quicksort is a divide-and-conquer algorithm..."
    )

    question_trait = RubricTrait(
        name="complexity_analysis", description="Does it mention time complexity?", kind="boolean"
    )
    checkpoint.add_question_rubric_trait(q_id, question_trait)

    # Verify in the checkpoint data
    question_data = checkpoint.get_question(q_id)
    assert question_data["question_rubric"] is not None
    assert len(question_data["question_rubric"]) == 1
    assert question_data["question_rubric"][0].name == "complexity_analysis"


def test_save_and_load():
    """Test saving and loading checkpoints."""
    # Create a checkpoint with data
    checkpoint = Checkpoint.create(name="Save/Load Test", description="Testing save and load")

    q_id = checkpoint.add_question(question="What is AI?", raw_answer="Artificial Intelligence", finished=True)

    checkpoint.add_global_rubric_trait(RubricTrait(name="accuracy", description="Is it accurate?", kind="boolean"))

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jsonld", delete=False) as f:
        temp_path = Path(f.name)

    try:
        checkpoint.save(temp_path)

        # Load from file
        loaded = Checkpoint.load(temp_path)

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
    checkpoint = Checkpoint.create("JSON-LD Test")

    checkpoint.add_question(question="Test question", raw_answer="Test answer")

    # Save and verify JSON-LD structure
    with tempfile.NamedTemporaryFile(suffix=".jsonld", delete=False) as f:
        temp_path = Path(f.name)

    try:
        checkpoint.save(temp_path)

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
    checkpoint = Checkpoint.create("Test")

    # Add unfinished question
    checkpoint.add_question(question="Unfinished question", raw_answer="Answer", finished=False)

    # Add finished question
    template = """class Answer(BaseAnswer):
    response: str = Field(description="Response")

    def verify(self) -> bool:
        return True
"""

    checkpoint.add_question(question="Finished question", raw_answer="Answer", answer_template=template, finished=True)

    # Get finished templates
    templates = checkpoint.get_finished_templates()
    assert len(templates) == 1
    assert templates[0].question_text == "Finished question"
    assert templates[0].template_code == template
    assert templates[0].finished is True


def test_validate_checkpoint():
    """Test checkpoint validation."""
    checkpoint = Checkpoint.create("Test")

    # Empty checkpoint should be valid
    is_valid, msg = checkpoint.validate()
    assert is_valid is True

    # Add valid question
    checkpoint.add_question(question="Valid question", raw_answer="Valid answer")

    is_valid, msg = checkpoint.validate()
    assert is_valid is True

    # Add question with invalid template
    q_id = checkpoint.add_question(question="Question with bad template", raw_answer="Answer")

    # Directly modify the template to be invalid (bypassing validation)
    for item in checkpoint._checkpoint.hasPart:
        if checkpoint._get_item_id(item) == q_id:
            item.item.hasPart.text = "invalid python code"
            break

    checkpoint._rebuild_cache()

    is_valid, msg = checkpoint.validate()
    assert is_valid is False
    assert "Invalid template" in msg


def test_metadata_handling():
    """Test custom metadata and author/sources."""
    checkpoint = Checkpoint.create("Test")

    author = {"name": "John Doe", "email": "john@example.com"}
    sources = [
        {"title": "Python Documentation", "url": "https://docs.python.org"},
        {"title": "Wikipedia", "url": "https://wikipedia.org"},
    ]
    custom_meta = {"difficulty": "easy", "topic": "programming"}

    q_id = checkpoint.add_question(
        question="What is Python?",
        raw_answer="A programming language",
        author=author,
        sources=sources,
        custom_metadata=custom_meta,
    )

    question = checkpoint.get_question(q_id)
    assert question["author"] == author
    assert question["sources"] == sources
    assert question["custom_metadata"] == custom_meta

    # Test round-trip through save/load
    with tempfile.NamedTemporaryFile(suffix=".jsonld", delete=False) as f:
        temp_path = Path(f.name)

    try:
        checkpoint.save(temp_path)
        loaded = Checkpoint.load(temp_path)

        loaded_q = loaded.get_question(q_id)
        assert loaded_q["author"] == author
        assert loaded_q["sources"] == sources
        assert loaded_q["custom_metadata"] == custom_meta

    finally:
        temp_path.unlink()


def test_question_id_generation():
    """Test that question IDs are generated consistently."""
    checkpoint = Checkpoint.create("Test")

    # Add same question twice
    q1_id = checkpoint.add_question(question="What is Python?", raw_answer="A language")

    q2_id = checkpoint.add_question(
        question="What is Python?",  # Same question
        raw_answer="Different answer",
    )

    # IDs should be based on question text, so should be similar
    assert "python" in q1_id.lower()
    assert "python" in q2_id.lower()

    # But we should have 2 questions
    assert len(checkpoint.get_question_ids()) == 2
