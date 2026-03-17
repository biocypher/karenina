from unittest.mock import patch

import pytest

from karenina import Benchmark
from karenina.benchmark.authoring.answers.generator import generate_answer_template
from karenina.schemas.entities import Question
from karenina.utils.checkpoint import extract_questions_from_benchmark


@pytest.mark.unit
def test_question_schema_answer_notes():
    """Test that Question schema correctly stores answer_notes."""
    q = Question(question="What is BCL2?", raw_answer="A gene", answer_notes="Focus on its role in apoptosis.")
    assert q.answer_notes == "Focus on its role in apoptosis."

    # Test serialization
    data = q.model_dump()
    assert data["answer_notes"] == "Focus on its role in apoptosis."


@pytest.mark.unit
def test_question_manager_add_question_with_notes():
    """Test that QuestionManager.add_question handles answer_notes."""
    benchmark = Benchmark.create(name="test")

    # Via kwargs
    q_id1 = benchmark.add_question("Q1", "A1", answer_notes="Note 1")
    q_data1 = benchmark.get_question(q_id1)
    assert q_data1["answer_notes"] == "Note 1"

    # Via Question object
    q_obj = Question(question="Q2", raw_answer="A2", answer_notes="Note 2")
    q_id2 = benchmark.add_question(q_obj)
    q_data2 = benchmark.get_question(q_id2)
    assert q_data2["answer_notes"] == "Note 2"


@pytest.mark.unit
def test_checkpoint_persistence_answer_notes():
    """Test that answer_notes are preserved in JSON-LD checkpoints."""
    benchmark = Benchmark.create(name="test")
    benchmark.add_question("Q1", "A1", answer_notes="Important note")

    # Extract questions (simulates loading from checkpoint)
    questions = extract_questions_from_benchmark(benchmark._base._checkpoint)
    assert questions[0]["answer_notes"] == "Important note"


@pytest.mark.unit
def test_generator_uses_answer_notes():
    """Test that generate_answer_template passes answer_notes to underlying logic."""
    q_obj = Question(
        question="What is the capital of France?", raw_answer="Paris", answer_notes="Mention the region if possible."
    )

    with patch("karenina.benchmark.authoring.answers.generator._generate_structured_outputs") as mock_gen:
        # Provide a complete mock response including field_descriptions
        mock_gen.return_value = {
            "attributes": [{"name": "is_correct", "type": "bool", "ground_truth": True}],
            "field_descriptions": {"is_correct": "Whether the answer is correct."},
        }

        generate_answer_template(question_obj=q_obj, model="claude-haiku-4-5", interface="langchain")

        # Verify it was called with answer_notes
        args, kwargs = mock_gen.call_args
        assert kwargs["answer_notes"] == "Mention the region if possible."
