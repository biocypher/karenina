"""Basic tests for few-shot prompting functionality."""

import pytest

from karenina.benchmark.models import ModelConfig, VerificationConfig
from karenina.benchmark.verification.verification_utils import _construct_few_shot_prompt
from karenina.schemas.question_class import Question


def test_construct_few_shot_prompt_disabled() -> None:
    """Test that when few-shot is disabled, original question is returned."""
    question_text = "What is 2 + 2?"
    examples = [
        {"question": "What is 1 + 1?", "answer": "2"},
        {"question": "What is 3 + 3?", "answer": "6"},
    ]

    result = _construct_few_shot_prompt(question_text, examples, few_shot_enabled=False)

    assert result == question_text


def test_construct_few_shot_prompt_enabled() -> None:
    """Test few-shot prompt construction with examples."""
    question_text = "What is 2 + 2?"
    examples = [
        {"question": "What is 1 + 1?", "answer": "2"},
        {"question": "What is 3 + 3?", "answer": "6"},
    ]

    result = _construct_few_shot_prompt(question_text, examples, few_shot_enabled=True)

    expected = (
        "Question: What is 1 + 1?\n"
        "Answer: 2\n\n"
        "Question: What is 3 + 3?\n"
        "Answer: 6\n\n"
        "Question: What is 2 + 2?\n"
        "Answer:"
    )
    assert result == expected


def test_question_schema_with_few_shot_examples() -> None:
    """Test creating a Question with few-shot examples."""
    question = Question(
        question="What is the capital of France?",
        raw_answer="Paris",
        few_shot_examples=[
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
            {"question": "What is the capital of Italy?", "answer": "Rome"},
        ],
    )

    assert question.question == "What is the capital of France?"
    assert question.raw_answer == "Paris"
    assert len(question.few_shot_examples) == 2
    assert question.few_shot_examples[0]["question"] == "What is the capital of Germany?"
    assert question.few_shot_examples[0]["answer"] == "Berlin"


def test_question_schema_without_few_shot_examples() -> None:
    """Test creating a Question without few-shot examples."""
    question = Question(
        question="What is the capital of France?",
        raw_answer="Paris",
    )

    assert question.question == "What is the capital of France?"
    assert question.raw_answer == "Paris"
    assert question.few_shot_examples is None


def test_verification_config_with_few_shot_settings() -> None:
    """Test VerificationConfig with few-shot settings."""
    answering_model = ModelConfig(
        id="test-answering",
        model_provider="test",
        model_name="test-model",
        temperature=0.1,
        interface="langchain",
        system_prompt="Test answering prompt",
    )

    parsing_model = ModelConfig(
        id="test-parsing",
        model_provider="test",
        model_name="test-model",
        temperature=0.1,
        interface="langchain",
        system_prompt="Test parsing prompt",
    )

    config = VerificationConfig(
        answering_models=[answering_model],
        parsing_models=[parsing_model],
        few_shot_enabled=True,
        few_shot_mode="k-shot",
        few_shot_k=5,
    )

    assert config.few_shot_enabled is True
    assert config.few_shot_mode == "k-shot"
    assert config.few_shot_k == 5


def test_verification_config_few_shot_defaults() -> None:
    """Test that few-shot is disabled by default in VerificationConfig."""
    answering_model = ModelConfig(
        id="test-answering",
        model_provider="test",
        model_name="test-model",
        temperature=0.1,
        interface="langchain",
        system_prompt="Test answering prompt",
    )

    parsing_model = ModelConfig(
        id="test-parsing",
        model_provider="test",
        model_name="test-model",
        temperature=0.1,
        interface="langchain",
        system_prompt="Test parsing prompt",
    )

    config = VerificationConfig(
        answering_models=[answering_model],
        parsing_models=[parsing_model],
    )

    # Few-shot should be disabled by default (new API)
    assert config.is_few_shot_enabled() is False
    assert config.get_few_shot_config() is None

    # Legacy fields should be None
    assert config.few_shot_enabled is None
    assert config.few_shot_mode is None
    assert config.few_shot_k is None


def test_verification_config_k_shot_validation() -> None:
    """Test that k-shot validation works correctly."""
    answering_model = ModelConfig(
        id="test-answering",
        model_provider="test",
        model_name="test-model",
        temperature=0.1,
        interface="langchain",
        system_prompt="Test answering prompt",
    )

    parsing_model = ModelConfig(
        id="test-parsing",
        model_provider="test",
        model_name="test-model",
        temperature=0.1,
        interface="langchain",
        system_prompt="Test parsing prompt",
    )

    # Test that k must be positive for k-shot mode (updated error message)
    with pytest.raises(ValueError, match="Global few-shot k value must be at least 1"):
        VerificationConfig(
            answering_models=[answering_model],
            parsing_models=[parsing_model],
            few_shot_enabled=True,
            few_shot_mode="k-shot",
            few_shot_k=0,
        )
