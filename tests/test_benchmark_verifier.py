"""Tests for benchmark verification system."""

from unittest.mock import Mock, patch

from karenina.benchmark.models import VerificationConfig
from karenina.benchmark.verification.runner import _strip_markdown_fences, _system_prompt_compose
from karenina.benchmark.verifier import run_question_verification, validate_answer_template


def test_validate_answer_template_valid() -> None:
    """Test validation of a valid answer template."""
    template_code = '''
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    """Answer for a simple question."""
    response: str = Field(description="The answer response")

    def model_post_init(self, __context):
        self.correct = {"expected": "test"}

    def verify(self):
        return len(self.response) > 0
'''

    is_valid, error_msg, Answer = validate_answer_template(template_code)

    assert is_valid is True
    assert error_msg is None
    assert Answer is not None
    assert hasattr(Answer, "verify")
    assert "model_post_init" in Answer.__dict__


def test_validate_answer_template_with_literal() -> None:
    """Test validation of a template using Literal type."""
    template_code = '''
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field
from typing import Literal

class Answer(BaseAnswer):
    """Answer with Literal type."""
    status: Literal["success", "failure"] = Field(description="The status")
    response: str = Field(description="The response")

    def model_post_init(self, __context):
        self.correct = {"status": "success", "response": "expected"}

    def verify(self):
        return self.status in ["success", "failure"] and len(self.response) > 0
'''

    is_valid, error_msg, Answer = validate_answer_template(template_code)

    assert is_valid is True
    assert error_msg is None
    assert Answer is not None
    assert hasattr(Answer, "verify")


def test_validate_answer_template_no_answer_class() -> None:
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


def test_validate_answer_template_no_verify_method() -> None:
    """Test validation fails when Answer class has no verify method."""
    template_code = """
from karenina.schemas.answer_class import BaseAnswer

class Answer(BaseAnswer):
    response: str = "test"
    correct: dict = {}
"""

    is_valid, error_msg, Answer = validate_answer_template(template_code)

    assert is_valid is False
    assert "does not have a 'verify' method" in error_msg
    assert Answer is None


def test_validate_answer_template_invalid_syntax() -> None:
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
def test_run_question_verification_template_validation_failure(mock_init_model) -> None:
    """Test verification with invalid template."""
    config = VerificationConfig(
        answering_model_provider="openai",
        answering_model_name="gpt-4.1-mini",
        answering_temperature=0.1,
        answering_interface="langchain",
        parsing_model_provider="openai",
        parsing_model_name="gpt-4.1-mini",
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
def test_run_question_verification_success(mock_init_model) -> None:
    """Test successful verification run."""
    # Mock the LLM responses
    mock_answering_llm = Mock()
    mock_parsing_llm = Mock()

    mock_answering_llm.invoke.return_value.content = "The answer is 4"
    mock_parsing_llm.invoke.return_value.content = '{"response": "4"}'

    # Create a mock answer instance
    mock_answer = Mock()
    mock_answer.model_dump.return_value = {"response": "4"}
    mock_answer.verify.return_value = True
    mock_answer.verify_regex.return_value = {"success": True, "results": {}, "details": {}}

    # Setup the init_chat_model_unified mock to return different models
    def mock_init_side_effect(*args, **kwargs) -> None:
        if kwargs.get("model") == "gpt-4.1-mini":
            return mock_answering_llm if "answering" in str(kwargs) else mock_parsing_llm
        return mock_parsing_llm

    mock_init_model.side_effect = (
        lambda **_kwargs: mock_answering_llm if mock_init_model.call_count <= 1 else mock_parsing_llm
    )

    # Mock the PydanticOutputParser
    with patch("karenina.benchmark.verification.runner.PydanticOutputParser") as mock_parser_class:
        mock_parser = Mock()
        mock_parser.get_format_instructions.return_value = "Format the response as JSON with the following structure..."
        mock_parser.parse.return_value = mock_answer
        mock_parser_class.return_value = mock_parser

        config = VerificationConfig(
            answering_model_provider="openai",
            answering_model_name="gpt-4.1-mini",
            answering_temperature=0.1,
            answering_interface="langchain",
            parsing_model_provider="openai",
            parsing_model_name="gpt-4.1-mini",
            parsing_temperature=0.1,
            parsing_interface="langchain",
        )

        valid_template = """
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    response: str = Field(description="The answer")

    def model_post_init(self, __context):
        self.correct = {"response": "expected"}

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
        assert result.parsed_llm_response == {"response": "4"}
        # Note: parsed_gt_response would contain the 'correct' field content if it exists
        assert result.verify_result is True


@patch("karenina.llm.interface.init_chat_model")
def test_run_question_verification_markdown_fenced_json(mock_init_model) -> None:
    """Test successful parsing of markdown-fenced JSON response."""
    # Mock the LLM responses with markdown fences
    mock_answering_llm = Mock()
    mock_parsing_llm = Mock()

    mock_answering_llm.invoke.return_value.content = "The answer is 4"
    mock_parsing_llm.invoke.return_value.content = '```json\n{"response": "4"}\n```'

    # Create a mock answer instance
    mock_answer = Mock()
    mock_answer.model_dump.return_value = {"response": "4"}
    mock_answer.verify.return_value = True
    mock_answer.verify_regex.return_value = {"success": True, "results": {}, "details": {}}

    mock_init_model.side_effect = (
        lambda **_kwargs: mock_answering_llm if mock_init_model.call_count <= 1 else mock_parsing_llm
    )

    # Mock the PydanticOutputParser
    with patch("karenina.benchmark.verification.runner.PydanticOutputParser") as mock_parser_class:
        mock_parser = Mock()
        mock_parser.get_format_instructions.return_value = "Format the response as JSON with the following structure..."
        mock_parser.parse.return_value = mock_answer
        mock_parser_class.return_value = mock_parser

        config = VerificationConfig(
            answering_model_provider="openai",
            answering_model_name="gpt-4.1-mini",
            answering_temperature=0.1,
            answering_interface="langchain",
            parsing_model_provider="openai",
            parsing_model_name="gpt-4.1-mini",
            parsing_temperature=0.1,
            parsing_interface="langchain",
        )

        valid_template = """
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    response: str = Field(description="The answer")

    def model_post_init(self, __context):
        self.correct = {"response": "expected"}

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
        assert result.parsed_llm_response == {"response": "4"}
        # Note: parsed_gt_response would contain the 'correct' field content if it exists
        assert result.verify_result is True

        # Verify that the parser was called with cleaned JSON (without markdown fences)
        mock_parser.parse.assert_called_once_with('{"response": "4"}')


@patch("karenina.llm.interface.init_chat_model")
def test_run_question_verification_malformed_json(mock_init_model) -> None:
    """Test handling of malformed JSON from LLM."""
    # Mock the LLM responses with malformed JSON
    mock_answering_llm = Mock()
    mock_parsing_llm = Mock()

    mock_answering_llm.invoke.return_value.content = "The answer is 4"
    mock_parsing_llm.invoke.return_value.content = '{"response": "4"'  # Missing closing brace

    mock_init_model.side_effect = (
        lambda **_kwargs: mock_answering_llm if mock_init_model.call_count <= 1 else mock_parsing_llm
    )

    # Mock the PydanticOutputParser to raise an exception
    with patch("karenina.benchmark.verification.runner.PydanticOutputParser") as mock_parser_class:
        mock_parser = Mock()
        mock_parser.get_format_instructions.return_value = "Format the response as JSON with the following structure..."
        mock_parser.parse.side_effect = Exception("Invalid JSON format")
        mock_parser_class.return_value = mock_parser

        config = VerificationConfig(
            answering_model_provider="openai",
            answering_model_name="gpt-4.1-mini",
            answering_temperature=0.1,
            answering_interface="langchain",
            parsing_model_provider="openai",
            parsing_model_name="gpt-4.1-mini",
            parsing_temperature=0.1,
            parsing_interface="langchain",
        )

        valid_template = """
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    response: str = Field(description="The answer")

    def model_post_init(self, __context):
        self.correct = {"response": "expected"}

    def verify(self):
        return True
"""

        results = run_question_verification(
            question_id="test_id", question_text="What is 2+2?", template_code=valid_template, config=config
        )

        # Get the single result from the dictionary
        assert len(results) == 1
        result = list(results.values())[0]

        assert result.success is False
        assert "Parsing failed: Invalid JSON format" in result.error
        assert result.question_id == "test_id"
        assert result.question_text == "What is 2+2?"
        assert result.raw_llm_response == "The answer is 4"


def test_strip_markdown_fences() -> None:
    """Test markdown fence stripping functionality."""
    # Test with ```json fences
    json_with_fences = '```json\n{"response": "test"}\n```'
    assert _strip_markdown_fences(json_with_fences) == '{"response": "test"}'

    # Test with regular ``` fences
    json_with_regular_fences = '```\n{"response": "test"}\n```'
    assert _strip_markdown_fences(json_with_regular_fences) == '{"response": "test"}'

    # Test with only opening fence
    json_with_opening_fence = '```json\n{"response": "test"}'
    assert _strip_markdown_fences(json_with_opening_fence) == '{"response": "test"}'

    # Test with only closing fence
    json_with_closing_fence = '{"response": "test"}\n```'
    assert _strip_markdown_fences(json_with_closing_fence) == '{"response": "test"}'

    # Test with no fences
    plain_json = '{"response": "test"}'
    assert _strip_markdown_fences(plain_json) == '{"response": "test"}'

    # Test with non-string input
    assert _strip_markdown_fences(None) is None
    assert _strip_markdown_fences(123) == 123


def test_system_prompt_compose() -> None:
    """Test system prompt composition functionality."""
    # Test with both system prompt and format instructions
    system_prompt = "You are a helpful assistant."
    format_instructions = "Please format your response as JSON."
    result = _system_prompt_compose(system_prompt, format_instructions)

    expected = """<general_instructions>
You are a helpful assistant.
</general_instructions>

<format_instructions>
Please format your response as JSON.
</format_instructions>
"""
    assert result == expected

    # Test with None system prompt
    result = _system_prompt_compose(None, format_instructions)
    expected = """<general_instructions>

</general_instructions>

<format_instructions>
Please format your response as JSON.
</format_instructions>
"""
    assert result == expected

    # Test with empty system prompt
    result = _system_prompt_compose("", format_instructions)
    expected = """<general_instructions>

</general_instructions>

<format_instructions>
Please format your response as JSON.
</format_instructions>
"""
    assert result == expected


def test_validate_answer_template_no_correct_field() -> None:
    """Test validation passes when Answer class has no correct field (optional)."""
    template_code = """
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    response: str = Field(description="The answer response")

    def verify(self):
        return len(self.response) > 0
"""

    is_valid, error_msg, Answer = validate_answer_template(template_code)

    assert is_valid is True
    assert error_msg is None
    assert Answer is not None


def test_validate_answer_template_model_post_init_dict() -> None:
    """Test validation passes with model_post_init assigning dict to correct."""
    template_code = """
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    response: str = Field(description="The answer response")

    def model_post_init(self, __context):
        self.correct = {"expected": "test"}

    def verify(self):
        return len(self.response) > 0
"""

    is_valid, error_msg, Answer = validate_answer_template(template_code)
    assert is_valid is True
    assert error_msg is None
    assert Answer is not None


def test_validate_answer_template_correct_non_dict() -> None:
    """Test validation fails when model_post_init assigns non-dict to correct."""
    template_code = """
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    response: str = Field(description="The answer response")

    def model_post_init(self, __context):
        self.correct = "not a dict"

    def verify(self):
        return len(self.response) > 0
"""

    is_valid, error_msg, Answer = validate_answer_template(template_code)

    assert is_valid is False
    assert "model_post_init must assign 'self.correct' as a dictionary" in error_msg
    assert Answer is None
