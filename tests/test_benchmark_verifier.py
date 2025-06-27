"""Tests for benchmark verification system."""

from unittest.mock import Mock, patch

from karenina.benchmark.models import VerificationConfig
from karenina.benchmark.verifier import run_question_verification, validate_answer_template


def test_validate_answer_template_valid():
    """Test validation of a valid answer template."""
    template_code = '''
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Answer for a simple question."""
    response: str = Field(description="The answer response")
    
    def verify(self):
        return len(self.response) > 0
'''

    is_valid, error_msg, Answer = validate_answer_template(template_code)

    assert is_valid is True
    assert error_msg is None
    assert Answer is not None
    assert hasattr(Answer, "verify")


def test_validate_answer_template_with_literal():
    """Test validation of a template using Literal type."""
    template_code = '''
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field
from typing import Literal

class Answer(BaseAnswer):
    """Answer with Literal type."""
    status: Literal["success", "failure"] = Field(description="The status")
    response: str = Field(description="The response")
    
    def verify(self):
        return self.status in ["success", "failure"] and len(self.response) > 0
'''

    is_valid, error_msg, Answer = validate_answer_template(template_code)

    assert is_valid is True
    assert error_msg is None
    assert Answer is not None
    assert hasattr(Answer, "verify")


def test_validate_answer_template_no_answer_class():
    """Test validation fails when no Answer class is found."""
    template_code = """
from karenina.schemas.answer_class import BaseAnswer

class SomeOtherClass(BaseAnswer):
    pass
"""

    is_valid, error_msg, Answer = validate_answer_template(template_code)

    assert is_valid is False
    assert "No 'Answer' class found" in error_msg
    assert Answer is None


def test_validate_answer_template_no_verify_method():
    """Test validation fails when Answer class has no verify method."""
    template_code = """
from karenina.schemas.answer_class import BaseAnswer

class Answer(BaseAnswer):
    response: str = "test"
"""

    is_valid, error_msg, Answer = validate_answer_template(template_code)

    assert is_valid is False
    assert "does not have a 'verify' method" in error_msg
    assert Answer is None


def test_validate_answer_template_invalid_syntax():
    """Test validation fails with invalid Python syntax."""
    template_code = """
from karenina.schemas.answer_class import BaseAnswer

class Answer(BaseAnswer):
    def verify(self
        return True  # Missing closing parenthesis
"""

    is_valid, error_msg, Answer = validate_answer_template(template_code)

    assert is_valid is False
    assert "Error executing template code" in error_msg
    assert Answer is None


@patch("karenina.llm.interface.init_chat_model")
def test_run_question_verification_template_validation_failure(mock_init_model):
    """Test verification with invalid template."""
    config = VerificationConfig(
        answering_model_provider="google_genai",
        answering_model_name="gemini-2.0-flash",
        answering_temperature=0.1,
        answering_interface="langchain",
        parsing_model_provider="google_genai",
        parsing_model_name="gemini-2.0-flash",
        parsing_temperature=0.1,
        parsing_interface="langchain",
    )

    invalid_template = "invalid python code"

    results = run_question_verification(
        question_id="test_id", question_text="What is 2+2?", template_code=invalid_template, config=config
    )

    # Get the single result from the dictionary
    assert len(results) == 1
    result = list(results.values())[0]

    assert result.success is False
    assert "Template validation failed" in result.error
    assert result.question_id == "test_id"
    assert result.question_text == "What is 2+2?"


@patch("karenina.llm.interface.init_chat_model")
def test_run_question_verification_success(mock_init_model):
    """Test successful verification run."""
    # Mock the LLM responses
    mock_answering_llm = Mock()
    mock_parsing_llm = Mock()
    mock_structured_llm = Mock()

    mock_answering_llm.invoke.return_value.content = "The answer is 4"
    mock_parsing_llm.with_structured_output.return_value = mock_structured_llm

    # Create a mock answer instance
    mock_answer = Mock()
    mock_answer.model_dump.return_value = {"response": "4"}
    mock_answer.verify.return_value = True
    mock_structured_llm.invoke.return_value = mock_answer

    # Setup the init_chat_model_unified mock to return different models
    def mock_init_side_effect(*args, **kwargs):
        if kwargs.get("model") == "gemini-2.0-flash":
            return mock_answering_llm if "answering" in str(kwargs) else mock_parsing_llm
        return mock_parsing_llm

    mock_init_model.side_effect = (
        lambda **kwargs: mock_answering_llm if mock_init_model.call_count <= 1 else mock_parsing_llm
    )

    config = VerificationConfig(
        answering_model_provider="google_genai",
        answering_model_name="gemini-2.0-flash",
        answering_temperature=0.1,
        answering_interface="langchain",
        parsing_model_provider="google_genai",
        parsing_model_name="gemini-2.0-flash",
        parsing_temperature=0.1,
        parsing_interface="langchain",
    )

    valid_template = """
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    response: str = Field(description="The answer")
    
    def verify(self):
        return True
"""

    results = run_question_verification(
        question_id="test_id", question_text="What is 2+2?", template_code=valid_template, config=config
    )

    # Get the single result from the dictionary
    assert len(results) == 1
    result = list(results.values())[0]

    assert result.success is True
    assert result.error is None
    assert result.question_id == "test_id"
    assert result.question_text == "What is 2+2?"
    assert result.raw_llm_response == "The answer is 4"
    assert result.parsed_response == {"response": "4"}
    assert result.verify_result is True
