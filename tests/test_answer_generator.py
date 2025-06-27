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
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value.content = """
    ```python
    class Answer(BaseAnswer):
        answer: bool = Field(description="Answer contains whether the condition is true or false")

        def model_post_init(self, __context):
            self.id = "test_id"
            self.correct = True
    ```
    """
    return mock


def test_generate_answer_template(mock_llm):
    """Test generating an answer template."""
    with patch("karenina.llm.interface.init_chat_model", return_value=mock_llm):
        result = generate_answer_template(
            question="Test question?",
            question_json='{"id": "test_id", "question": "Test question?", "raw_answer": "Yes", "tags": []}',
            model="test-model",
            model_provider="test_provider",
            temperature=0,
        )

        # Verify the LLM was called with correct parameters
        mock_llm.invoke.assert_called_once()
        messages = mock_llm.invoke.call_args[0][0]
        assert len(messages) == 2
        assert "Test question?" in messages[1].content
        assert "test_id" in messages[1].content

        # Verify the result contains the expected code
        assert "class Answer" in result
        assert "BaseAnswer" in result
        assert "answer: bool" in result


def test_generate_answer_templates_from_questions_file():
    """Test generating answer templates from a questions file."""
    # Create a temporary questions.py file
    questions_content = """
from karenina.schemas.question_class import Question

question_1 = Question(
    id="test1",
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
        # Mock the LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = """
        ```python
        class Answer(BaseAnswer):
            answer: bool = Field(description="Answer contains whether the condition is true or false")

            def model_post_init(self, __context):
                self.id = "test1"
                self.correct = True
        ```
        """

        with patch("karenina.llm.interface.init_chat_model", return_value=mock_llm):
            # Test without return_blocks
            result = generate_answer_templates_from_questions_file(
                tmp_path,
                model="test-model",
                model_provider="test_provider",
            )
            assert len(result) == 1
            assert "test1" in result

            # Test with return_blocks
            result, blocks = generate_answer_templates_from_questions_file(
                tmp_path,
                model="test-model",
                model_provider="test_provider",
                return_blocks=True,
            )
            assert len(result) == 1
            assert len(blocks) == 1
            assert "test1" in result
            assert "class Answer" in blocks["test1"]

    finally:
        os.unlink(tmp_path)


def test_load_answer_templates_from_json():
    """Test loading answer templates from a JSON file."""
    # Create a temporary JSON file with code blocks
    code_blocks = {
        "test1": """
class Answer(BaseAnswer):
    answer: bool = Field(description="Answer contains whether the condition is true or false")

    def model_post_init(self, __context):
        self.id = "test1"
        self.correct = True
""",
        "test2": """
class Answer(BaseAnswer):
    answer: str = Field(description="Answer contains the string response")

    def model_post_init(self, __context):
        self.id = "test2"
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


def test_load_answer_templates_from_json_invalid_file():
    """Test loading answer templates from an invalid JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp.write("invalid json content")
        tmp_path = tmp.name

    try:
        with pytest.raises(json.JSONDecodeError):
            load_answer_templates_from_json(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_load_answer_templates_from_json_invalid_code():
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


def test_generate_answer_templates_reader_integration():
    """Test that the generator properly integrates with the new reader module."""
    from karenina.questions.reader import read_questions_from_file

    # Create a temporary questions.py file
    questions_content = """
from karenina.schemas.question_class import Question

question_1 = Question(
    id="integration_test",
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
        assert questions[0].id == "integration_test"

        # Mock the LLM response for the generator
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = """
        ```python
        class Answer(BaseAnswer):
            answer: bool = Field(description="Integration test answer")

            def model_post_init(self, __context):
                self.id = "integration_test"
                self.correct = True
        ```
        """

        # Test that the generator uses the reader correctly
        with patch("karenina.llm.interface.init_chat_model", return_value=mock_llm):
            result = generate_answer_templates_from_questions_file(tmp_path)
            assert len(result) == 1
            assert "integration_test" in result

            # Verify the Answer class was created correctly
            Answer = result["integration_test"]
            answer_instance = Answer(answer=True)
            assert answer_instance.id == "integration_test"
            assert answer_instance.correct is True

    finally:
        os.unlink(tmp_path)


def test_generate_answer_templates_reader_with_dict_compatibility():
    """Test that the generator works with the updated reader function (backward compatibility)."""
    from karenina.questions.reader import read_questions_from_file

    # Create a temporary questions.py file
    questions_content = """
from karenina.schemas.question_class import Question

question_1 = Question(
    id="dict_test",
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
        assert "dict_test" in questions_dict
        assert questions_list[0].id == questions_dict["dict_test"].id

        # Mock the LLM response for the generator
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = """
        ```python
        class Answer(BaseAnswer):
            answer: bool = Field(description="Dictionary compatibility test answer")

            def model_post_init(self, __context):
                self.id = "dict_test"
                self.correct = True
        ```
        """

        # Test that the generator still works with the reader (uses default list behavior)
        with patch("karenina.llm.interface.init_chat_model", return_value=mock_llm):
            result = generate_answer_templates_from_questions_file(tmp_path)
            assert len(result) == 1
            assert "dict_test" in result

    finally:
        os.unlink(tmp_path)
