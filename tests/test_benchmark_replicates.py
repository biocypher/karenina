"""Tests for benchmark replication functionality."""

from unittest.mock import Mock, patch

from karenina.benchmark.models import ModelConfiguration, VerificationConfig
from karenina.benchmark.verifier import run_question_verification


def test_verification_config_replicate_count():
    """Test that VerificationConfig properly handles replicate_count."""
    config = VerificationConfig(
        answering_models=[
            ModelConfiguration(
                id="test-answering",
                model_provider="test",
                model_name="test-model",
                temperature=0.1,
                interface="langchain",
                system_prompt="Test prompt",
            )
        ],
        parsing_models=[
            ModelConfiguration(
                id="test-parsing",
                model_provider="test",
                model_name="test-model",
                temperature=0.1,
                interface="langchain",
                system_prompt="Test prompt",
            )
        ],
        replicate_count=3,
    )

    assert config.replicate_count == 3


@patch("karenina.llm.interface.init_chat_model")
def test_run_question_verification_with_replicates(mock_init_model):
    """Test that verification runs multiple replicates when configured."""
    # Mock the LLM responses
    mock_answering_llm = Mock()
    mock_parsing_llm = Mock()

    mock_answering_llm.invoke.return_value.content = "The answer is 4"
    mock_parsing_llm.invoke.return_value.content = '{"response": "4"}'

    # Create a mock answer instance
    mock_answer = Mock()
    mock_answer.model_dump.return_value = {"response": "4"}
    mock_answer.verify.return_value = True

    # Setup the init_chat_model_unified mock
    mock_init_model.side_effect = [mock_answering_llm, mock_parsing_llm] * 6  # 3 replicates × 2 calls each

    # Mock the PydanticOutputParser
    with patch("karenina.benchmark.verification.runner.PydanticOutputParser") as mock_parser_class:
        mock_parser = Mock()
        mock_parser.get_format_instructions.return_value = "Format the response as JSON with the following structure..."
        mock_parser.parse.return_value = mock_answer
        mock_parser_class.return_value = mock_parser

        config = VerificationConfig(
            answering_models=[
                ModelConfiguration(
                    id="test-answering",
                    model_provider="test",
                    model_name="test-model",
                    temperature=0.1,
                    interface="langchain",
                    system_prompt="Test prompt",
                )
            ],
            parsing_models=[
                ModelConfiguration(
                    id="test-parsing",
                    model_provider="test",
                    model_name="test-model",
                    temperature=0.1,
                    interface="langchain",
                    system_prompt="Test prompt",
                )
            ],
            replicate_count=3,
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

        # Should have 3 results (1 combination × 3 replicates)
        assert len(results) == 3

        # Check that replicate numbers are properly set
        expected_keys = [
            "test_id_test-answering_test-parsing_rep1",
            "test_id_test-answering_test-parsing_rep2",
            "test_id_test-answering_test-parsing_rep3",
        ]

        for key in expected_keys:
            assert key in results
            result = results[key]
            assert result.success is True
            assert result.question_id == "test_id"

            # Extract replicate number from key
            replicate_num = int(key.split("_rep")[1])
            assert result.answering_replicate == replicate_num
            assert result.parsing_replicate == replicate_num


@patch("karenina.llm.interface.init_chat_model")
def test_run_question_verification_single_replicate(mock_init_model):
    """Test that single replicate doesn't include replicate numbers."""
    # Mock the LLM responses
    mock_answering_llm = Mock()
    mock_parsing_llm = Mock()

    mock_answering_llm.invoke.return_value.content = "The answer is 4"
    mock_parsing_llm.invoke.return_value.content = '{"response": "4"}'

    # Create a mock answer instance
    mock_answer = Mock()
    mock_answer.model_dump.return_value = {"response": "4"}
    mock_answer.verify.return_value = True

    # Setup the init_chat_model_unified mock
    mock_init_model.side_effect = [mock_answering_llm, mock_parsing_llm]

    # Mock the PydanticOutputParser
    with patch("karenina.benchmark.verification.runner.PydanticOutputParser") as mock_parser_class:
        mock_parser = Mock()
        mock_parser.get_format_instructions.return_value = "Format the response as JSON with the following structure..."
        mock_parser.parse.return_value = mock_answer
        mock_parser_class.return_value = mock_parser

        config = VerificationConfig(
            answering_models=[
                ModelConfiguration(
                    id="test-answering",
                    model_provider="test",
                    model_name="test-model",
                    temperature=0.1,
                    interface="langchain",
                    system_prompt="Test prompt",
                )
            ],
            parsing_models=[
                ModelConfiguration(
                    id="test-parsing",
                    model_provider="test",
                    model_name="test-model",
                    temperature=0.1,
                    interface="langchain",
                    system_prompt="Test prompt",
                )
            ],
            replicate_count=1,
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

        # Should have 1 result
        assert len(results) == 1

        # Key should not include replicate number
        expected_key = "test_id_test-answering_test-parsing"
        assert expected_key in results

        result = results[expected_key]
        assert result.success is True
        assert result.question_id == "test_id"

        # Replicate fields should be None for single replicate
        assert result.answering_replicate is None
        assert result.parsing_replicate is None


def test_verification_config_default_replicate_count():
    """Test that VerificationConfig defaults to 1 replicate."""
    config = VerificationConfig(
        answering_models=[
            ModelConfiguration(
                id="test-answering",
                model_provider="test",
                model_name="test-model",
                temperature=0.1,
                interface="langchain",
                system_prompt="Test prompt",
            )
        ],
        parsing_models=[
            ModelConfiguration(
                id="test-parsing",
                model_provider="test",
                model_name="test-model",
                temperature=0.1,
                interface="langchain",
                system_prompt="Test prompt",
            )
        ],
    )

    assert config.replicate_count == 1
