import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from karenina.answers.generator import (
    generate_answer_template,
    generate_answer_templates_from_questions_file,
    load_answer_templates_from_json,
)


@pytest.fixture
def mock_llm() -> None:
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value.content = """
    ```python
    class Answer(BaseAnswer):
        answer: bool = Field(description="Answer contains whether the condition is true or false")

        def model_post_init(self, __context):
            self.correct = True
    ```
    """
    return mock


def test_generate_answer_template(mock_llm) -> None:
    """Test generating an answer template."""
    # Mock at the _build_chain level to return pre-parsed results
    from karenina.answers.generator import AttributeDescriptions, GroundTruthField, GroundTruthSpec

    # Phase 1 result: ground truth specification
    gt_spec = GroundTruthSpec(attributes=[GroundTruthField(name="answer", type="bool", ground_truth=True)])

    # Phase 2 result: field descriptions
    fd_spec = AttributeDescriptions(field_descriptions={"answer": "Whether the condition is true or false"})

    # Create a mock chain that returns these results
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [gt_spec, fd_spec]

    with patch("karenina.answers.generator._build_chain", return_value=mock_chain):
        result = generate_answer_template(
            question="Test question?",
            raw_answer="Yes",
            model="test-model",
            model_provider="test_provider",
            temperature=0,
        )

        # Verify the chain was called twice (ground truth + field descriptions)
        assert mock_chain.invoke.call_count == 2

        # Verify the result contains the expected code
        assert "class Answer" in result
        assert "BaseAnswer" in result
        assert "answer: bool" in result


def test_generate_answer_templates_from_questions_file() -> None:
    """Test generating answer templates from a questions file."""
    from karenina.answers.generator import AttributeDescriptions, GroundTruthField, GroundTruthSpec

    # Create a temporary questions.py file
    questions_content = """
from karenina.schemas.domain import Question

question_1 = Question(
    question="Test question 1?",
    raw_answer="Yes",
    tags=[],
)

all_questions = [question_1]
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(questions_content)
        tmp_path = tmp.name

    try:
        # Mock the chain results
        gt_spec = GroundTruthSpec(attributes=[GroundTruthField(name="answer", type="bool", ground_truth=True)])
        fd_spec = AttributeDescriptions(field_descriptions={"answer": "Whether the condition is true or false"})

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [gt_spec, fd_spec] * 2  # Called twice for both tests

        with patch("karenina.answers.generator._build_chain", return_value=mock_chain):
            # Test without return_blocks
            result = generate_answer_templates_from_questions_file(
                tmp_path,
                model="test-model",
                model_provider="test_provider",
            )
            assert len(result) == 1
            # Get the auto-generated ID
            import hashlib

            expected_id = hashlib.md5(b"Test question 1?").hexdigest()
            assert expected_id in result

            # Test with return_blocks
            result, blocks = generate_answer_templates_from_questions_file(
                tmp_path,
                model="test-model",
                model_provider="test_provider",
                return_blocks=True,
            )
            assert len(result) == 1
            assert len(blocks) == 1
            # Get the auto-generated ID
            import hashlib

            expected_id = hashlib.md5(b"Test question 1?").hexdigest()
            assert expected_id in result
            assert "class Answer" in blocks[expected_id]

    finally:
        os.unlink(tmp_path)


def test_load_answer_templates_from_json() -> None:
    """Test loading answer templates from a JSON file."""
    # Create a temporary JSON file with code blocks
    code_blocks = {
        "test1": """
class Answer(BaseAnswer):
    answer: bool = Field(description="Answer contains whether the condition is true or false")

    def model_post_init(self, __context):
        self.correct = True
""",
        "test2": """
class Answer(BaseAnswer):
    answer: str = Field(description="Answer contains the string response")

    def model_post_init(self, __context):
        self.correct = True
""",
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(code_blocks, tmp)
        tmp_path = tmp.name

    try:
        # Test without return_blocks
        result = load_answer_templates_from_json(tmp_path)
        assert len(result) == 2
        assert "test1" in result
        assert "test2" in result

        # Verify the Answer classes were created correctly
        answer1 = result["test1"](answer=True)
        assert answer1.id == "test1"
        assert answer1.correct is True

        answer2 = result["test2"](answer="foo")
        assert answer2.id == "test2"
        assert answer2.correct is True

        # Test with return_blocks
        result, blocks = load_answer_templates_from_json(tmp_path, return_blocks=True)
        assert len(result) == 2
        assert len(blocks) == 2
        assert "test1" in result
        assert "test2" in result
        assert "class Answer" in blocks["test1"]
        assert "class Answer" in blocks["test2"]

    finally:
        os.unlink(tmp_path)


def test_load_answer_templates_from_json_invalid_file() -> None:
    """Test loading answer templates from an invalid JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp.write("invalid json content")
        tmp_path = tmp.name

    try:
        with pytest.raises(json.JSONDecodeError):
            load_answer_templates_from_json(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_load_answer_templates_from_json_invalid_code() -> None:
    """Test loading answer templates with invalid code blocks."""
    code_blocks = {"test1": "invalid python code"}

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(code_blocks, tmp)
        tmp_path = tmp.name

    try:
        with pytest.raises(SyntaxError):
            load_answer_templates_from_json(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_generate_answer_templates_reader_integration() -> None:
    """Test that the generator properly integrates with the new reader module."""
    from karenina.answers.generator import AttributeDescriptions, GroundTruthField, GroundTruthSpec
    from karenina.questions.reader import read_questions_from_file

    # Create a temporary questions.py file
    questions_content = """
from karenina.schemas.domain import Question

question_1 = Question(
    question="Integration test question?",
    raw_answer="Yes",
    tags=["integration"],
)

all_questions = [question_1]
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(questions_content)
        tmp_path = tmp.name

    try:
        # First test that the reader works independently
        questions = read_questions_from_file(tmp_path)
        assert len(questions) == 1
        import hashlib

        expected_id = hashlib.md5(b"Integration test question?").hexdigest()
        assert questions[0].id == expected_id

        # Mock the chain results
        gt_spec = GroundTruthSpec(attributes=[GroundTruthField(name="answer", type="bool", ground_truth=True)])
        fd_spec = AttributeDescriptions(field_descriptions={"answer": "Integration test answer"})

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [gt_spec, fd_spec]

        # Test that the generator uses the reader correctly
        with patch("karenina.answers.generator._build_chain", return_value=mock_chain):
            result = generate_answer_templates_from_questions_file(tmp_path)
            assert len(result) == 1
            assert expected_id in result

            # Verify the Answer class was created correctly
            Answer = result[expected_id]
            answer_instance = Answer(answer=True)
            assert answer_instance.id == expected_id
            # The new structured generation sets correct as a dictionary
            assert answer_instance.correct == {"answer": True}

    finally:
        os.unlink(tmp_path)


def test_generate_answer_templates_reader_with_dict_compatibility() -> None:
    """Test that the generator works with the updated reader function (backward compatibility)."""
    from karenina.answers.generator import AttributeDescriptions, GroundTruthField, GroundTruthSpec
    from karenina.questions.reader import read_questions_from_file

    # Create a temporary questions.py file
    questions_content = """
from karenina.schemas.domain import Question

question_1 = Question(
    question="Dictionary compatibility test?",
    raw_answer="Yes",
    tags=["compatibility"],
)

all_questions = [question_1]
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(questions_content)
        tmp_path = tmp.name

    try:
        # Test that reader can return both formats
        questions_list = read_questions_from_file(tmp_path, return_dict=False)
        questions_dict = read_questions_from_file(tmp_path, return_dict=True)

        assert len(questions_list) == 1
        assert len(questions_dict) == 1
        import hashlib

        expected_id = hashlib.md5(b"Dictionary compatibility test?").hexdigest()
        assert expected_id in questions_dict
        assert questions_list[0].id == questions_dict[expected_id].id

        # Mock the chain results
        gt_spec = GroundTruthSpec(attributes=[GroundTruthField(name="answer", type="bool", ground_truth=True)])
        fd_spec = AttributeDescriptions(field_descriptions={"answer": "Dictionary compatibility test answer"})

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [gt_spec, fd_spec]

        # Test that the generator still works with the reader (uses default list behavior)
        with patch("karenina.answers.generator._build_chain", return_value=mock_chain):
            result = generate_answer_templates_from_questions_file(tmp_path)
            assert len(result) == 1
            assert expected_id in result

    finally:
        os.unlink(tmp_path)
