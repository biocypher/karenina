"""Integration tests for AdapterParallelInvoker with LLMTraitEvaluator.

Tests verify that when a ParserPort adapter is present, the evaluator correctly
uses AdapterParallelInvoker instead of falling back to sequential execution.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from karenina.benchmark.verification.evaluators.rubric_llm_trait_evaluator import (
    LLMTraitEvaluator,
)
from karenina.schemas.domain.rubric import LLMRubricTrait
from karenina.schemas.workflow.models import ModelConfig

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    return MagicMock()


@pytest.fixture
def mock_model_config() -> ModelConfig:
    """Create a mock model config."""
    return ModelConfig(
        id="test-model",
        model_name="claude-sonnet-4-20250514",
        model_provider="anthropic",
        interface="claude_agent_sdk",
    )


@pytest.fixture
def sample_question() -> str:
    """Sample question text for testing."""
    return "What is the capital of France?"


@pytest.fixture
def sample_answer() -> str:
    """Sample answer text for testing."""
    return "The capital of France is Paris."


@pytest.fixture
def boolean_traits() -> list[LLMRubricTrait]:
    """Create sample boolean traits for testing."""
    return [
        LLMRubricTrait(
            name="conciseness",
            description="Is the response concise?",
            kind="boolean",
            higher_is_better=True,
        ),
        LLMRubricTrait(
            name="accuracy",
            description="Is the response accurate?",
            kind="boolean",
            higher_is_better=True,
        ),
        LLMRubricTrait(
            name="completeness",
            description="Is the response complete?",
            kind="boolean",
            higher_is_better=True,
        ),
    ]


@pytest.fixture
def literal_traits() -> list[LLMRubricTrait]:
    """Create sample literal traits for testing."""
    return [
        LLMRubricTrait(
            name="sentiment",
            description="Classify the sentiment",
            kind="literal",
            classes={
                "positive": "Response has positive sentiment",
                "neutral": "Response is neutral",
                "negative": "Response has negative sentiment",
            },
            higher_is_better=True,
        ),
        LLMRubricTrait(
            name="complexity",
            description="Classify the complexity",
            kind="literal",
            classes={
                "simple": "Simple explanation",
                "medium": "Moderate complexity",
                "complex": "Complex explanation",
            },
            higher_is_better=True,
        ),
    ]


class MockBooleanResult(BaseModel):
    """Mock boolean result for testing."""

    result: bool


class MockClassificationResult(BaseModel):
    """Mock classification result for testing."""

    classification: str


# =============================================================================
# AdapterParallelInvoker Integration Tests
# =============================================================================


@pytest.mark.integration
def test_evaluator_uses_adapter_parallel_invoker_when_adapter_present(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    boolean_traits: list[LLMRubricTrait],
) -> None:
    """Test that evaluate_sequential uses AdapterParallelInvoker when adapter is present."""
    # Mock the get_parser factory to return a mock adapter
    mock_adapter = MagicMock()

    with patch("karenina.adapters.factory.get_parser", return_value=mock_adapter):
        evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

        # Verify adapter was set
        assert evaluator._parser_adapter is not None

        # Now test that AdapterParallelInvoker is used
        with patch("karenina.adapters.parallel.AdapterParallelInvoker") as mock_invoker_class:
            mock_invoker = MagicMock()
            mock_invoker_class.return_value = mock_invoker

            # Setup mock results for boolean traits
            mock_results = []
            for _ in boolean_traits:
                mock_results.append((MockBooleanResult(result=True), {}, None))

            mock_invoker.invoke_batch.return_value = mock_results

            results, usage_list = evaluator.evaluate_sequential(sample_question, sample_answer, boolean_traits)

            # Verify AdapterParallelInvoker was used
            mock_invoker_class.assert_called_once()
            mock_invoker.invoke_batch.assert_called_once()

            # Verify results
            assert len(results) == len(boolean_traits)
            for trait in boolean_traits:
                assert trait.name in results


@pytest.mark.integration
def test_evaluator_uses_adapter_parallel_invoker_for_literal_traits(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    literal_traits: list[LLMRubricTrait],
) -> None:
    """Test that evaluate_literal_sequential uses AdapterParallelInvoker when adapter is present."""
    mock_adapter = MagicMock()

    with patch("karenina.adapters.factory.get_parser", return_value=mock_adapter):
        evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

        with patch("karenina.adapters.parallel.AdapterParallelInvoker") as mock_invoker_class:
            mock_invoker = MagicMock()
            mock_invoker_class.return_value = mock_invoker

            # Setup mock results for literal traits
            mock_results = []
            for trait in literal_traits:
                class_names = list(trait.classes.keys())
                mock_results.append((MockClassificationResult(classification=class_names[0]), {}, None))

            mock_invoker.invoke_batch.return_value = mock_results

            scores, labels, usage_list = evaluator.evaluate_literal_sequential(
                sample_question, sample_answer, literal_traits
            )

            # Verify AdapterParallelInvoker was used
            mock_invoker_class.assert_called_once()
            mock_invoker.invoke_batch.assert_called_once()

            # Verify results
            assert len(scores) == len(literal_traits)
            assert len(labels) == len(literal_traits)


@pytest.mark.integration
def test_adapter_parallel_invoker_handles_errors_gracefully(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    boolean_traits: list[LLMRubricTrait],
) -> None:
    """Test that adapter parallel evaluation handles errors gracefully."""
    mock_adapter = MagicMock()

    with patch("karenina.adapters.factory.get_parser", return_value=mock_adapter):
        evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

        with patch("karenina.adapters.parallel.AdapterParallelInvoker") as mock_invoker_class:
            mock_invoker = MagicMock()
            mock_invoker_class.return_value = mock_invoker

            # Setup mock results with one error
            mock_results = [
                (MockBooleanResult(result=True), {}, None),
                (None, None, ValueError("Test error")),
                (MockBooleanResult(result=False), {}, None),
            ]

            mock_invoker.invoke_batch.return_value = mock_results

            results, usage_list = evaluator.evaluate_sequential(sample_question, sample_answer, boolean_traits)

            # Verify the errored trait has None
            assert results[boolean_traits[1].name] is None

            # Verify other traits have valid results
            assert results[boolean_traits[0].name] is True
            assert results[boolean_traits[2].name] is False


@pytest.mark.integration
def test_adapter_parallel_invoker_literal_handles_errors(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    literal_traits: list[LLMRubricTrait],
) -> None:
    """Test that literal adapter parallel evaluation handles errors gracefully."""
    mock_adapter = MagicMock()

    with patch("karenina.adapters.factory.get_parser", return_value=mock_adapter):
        evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

        with patch("karenina.adapters.parallel.AdapterParallelInvoker") as mock_invoker_class:
            mock_invoker = MagicMock()
            mock_invoker_class.return_value = mock_invoker

            # Setup mock results with one error
            # literal_traits[0] = "sentiment" with classes: positive, neutral, negative
            # literal_traits[1] = "complexity" with classes: simple, medium, complex
            mock_results = [
                (None, None, ValueError("Test error")),  # Error for sentiment
                (
                    MockClassificationResult(classification="medium"),
                    {},
                    None,
                ),  # "medium" is class index 1 for complexity
            ]

            mock_invoker.invoke_batch.return_value = mock_results

            scores, labels, usage_list = evaluator.evaluate_literal_sequential(
                sample_question, sample_answer, literal_traits
            )

            # Verify the errored trait has score -1 and error label
            assert scores[literal_traits[0].name] == -1
            assert "[EVALUATION_ERROR:" in labels[literal_traits[0].name]

            # Verify other traits have valid results
            # "medium" is index 1 in complexity classes: [simple, medium, complex]
            assert scores[literal_traits[1].name] == 1


@pytest.mark.integration
def test_adapter_parallel_invoker_tasks_converted_correctly(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    boolean_traits: list[LLMRubricTrait],
) -> None:
    """Test that message-based tasks are correctly converted to prompt-text tasks."""
    mock_adapter = MagicMock()

    with patch("karenina.adapters.factory.get_parser", return_value=mock_adapter):
        evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

        with patch("karenina.adapters.parallel.AdapterParallelInvoker") as mock_invoker_class:
            mock_invoker = MagicMock()
            mock_invoker_class.return_value = mock_invoker

            # Setup mock results
            mock_results = [(MockBooleanResult(result=True), {}, None) for _ in boolean_traits]
            mock_invoker.invoke_batch.return_value = mock_results

            evaluator.evaluate_sequential(sample_question, sample_answer, boolean_traits)

            # Get the tasks that were passed to invoke_batch
            call_args = mock_invoker.invoke_batch.call_args
            adapter_tasks = call_args[0][0]  # First positional arg

            # Verify tasks are (prompt_text, model_class) tuples
            assert len(adapter_tasks) == len(boolean_traits)
            for prompt_text, model_class in adapter_tasks:
                # prompt_text should be a string (combined from messages)
                assert isinstance(prompt_text, str)
                # Should contain content from system and user messages
                assert len(prompt_text) > 0
                # model_class should be a type
                assert isinstance(model_class, type)


@pytest.mark.integration
def test_adapter_parallel_invoker_max_workers_passed_correctly(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    boolean_traits: list[LLMRubricTrait],
) -> None:
    """Test that max_workers is passed correctly to AdapterParallelInvoker."""
    mock_adapter = MagicMock()

    with patch("karenina.adapters.factory.get_parser", return_value=mock_adapter):
        evaluator = LLMTraitEvaluator(
            mock_llm,
            async_enabled=True,
            async_max_workers=8,
            model_config=mock_model_config,
        )

        with patch("karenina.adapters.parallel.AdapterParallelInvoker") as mock_invoker_class:
            mock_invoker = MagicMock()
            mock_invoker_class.return_value = mock_invoker

            mock_results = [(MockBooleanResult(result=True), {}, None) for _ in boolean_traits]
            mock_invoker.invoke_batch.return_value = mock_results

            evaluator.evaluate_sequential(sample_question, sample_answer, boolean_traits)

            # Verify AdapterParallelInvoker was created with correct max_workers
            mock_invoker_class.assert_called_once()
            _, kwargs = mock_invoker_class.call_args
            assert kwargs.get("max_workers") == 8


@pytest.mark.integration
def test_evaluator_async_disabled_uses_true_sequential_with_adapter(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    boolean_traits: list[LLMRubricTrait],
) -> None:
    """Test that with async_enabled=False, true sequential is used even with adapter."""
    mock_adapter = MagicMock()

    with patch("karenina.adapters.factory.get_parser", return_value=mock_adapter):
        evaluator = LLMTraitEvaluator(mock_llm, async_enabled=False, model_config=mock_model_config)

        # Mock invoke_with_structured_output since true sequential calls it directly
        with patch(
            "karenina.benchmark.verification.evaluators.rubric_parsing.invoke_with_structured_output"
        ) as mock_invoke:
            mock_result = MagicMock()
            mock_result.result = True
            mock_invoke.return_value = (mock_result, {"total_tokens": 10})

            evaluator.evaluate_sequential(sample_question, sample_answer, boolean_traits)

            # Verify invoke_with_structured_output was called (true sequential)
            # It should be called once per trait
            assert mock_invoke.call_count == len(boolean_traits)
