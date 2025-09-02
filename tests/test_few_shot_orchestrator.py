"""Tests for few-shot prompting in the verification orchestrator."""

from karenina.benchmark.models import ModelConfig, VerificationConfig
from karenina.benchmark.verification.orchestrator import _create_verification_task


def test_create_verification_task_with_few_shot():
    """Test that _create_verification_task includes few-shot parameters."""
    # Create test models
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

    few_shot_examples = [
        {"question": "What is 1 + 1?", "answer": "2"},
        {"question": "What is 3 + 3?", "answer": "6"},
    ]

    task = _create_verification_task(
        question_id="test-question",
        question_text="What is 2 + 2?",
        template_code="class Answer(BaseModel): answer: int",
        answering_model=answering_model,
        parsing_model=parsing_model,
        run_name="test-run",
        job_id="test-job",
        answering_replicate=1,
        parsing_replicate=1,
        rubric=None,
        keywords=None,
        few_shot_examples=few_shot_examples,
        few_shot_enabled=True,
    )

    # Verify the task contains few-shot parameters
    assert task["few_shot_examples"] == few_shot_examples
    assert task["few_shot_enabled"] is True

    # Verify other standard parameters are still present
    assert task["question_id"] == "test-question"
    assert task["question_text"] == "What is 2 + 2?"
    assert task["answering_model"] == answering_model
    assert task["parsing_model"] == parsing_model


def test_create_verification_task_without_few_shot():
    """Test that _create_verification_task works without few-shot parameters."""
    # Create test models
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

    task = _create_verification_task(
        question_id="test-question",
        question_text="What is 2 + 2?",
        template_code="class Answer(BaseModel): answer: int",
        answering_model=answering_model,
        parsing_model=parsing_model,
        run_name="test-run",
        job_id="test-job",
        answering_replicate=1,
        parsing_replicate=1,
        rubric=None,
        keywords=None,
        few_shot_examples=None,
        few_shot_enabled=False,
    )

    # Verify the task contains few-shot parameters with default values
    assert task["few_shot_examples"] is None
    assert task["few_shot_enabled"] is False

    # Verify other standard parameters are still present
    assert task["question_id"] == "test-question"
    assert task["question_text"] == "What is 2 + 2?"


def test_verification_config_includes_few_shot_in_api_format():
    """Test that VerificationConfig includes few-shot settings."""
    # This test verifies the structure is correct for API calls
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

    # Test that the config has all the expected few-shot attributes
    assert hasattr(config, "few_shot_enabled")
    assert hasattr(config, "few_shot_mode")
    assert hasattr(config, "few_shot_k")

    assert config.few_shot_enabled is True
    assert config.few_shot_mode == "k-shot"
    assert config.few_shot_k == 5
