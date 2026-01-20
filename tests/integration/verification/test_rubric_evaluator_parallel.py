"""Integration tests for rubric evaluator parallel invocation.

Tests verify that the parallel invocation feature works correctly
with LLMTraitEvaluator when using sequential trait evaluation mode.

The evaluator uses LLMParallelInvoker for concurrent execution and
LLMPort.with_structured_output() for structured parsing.
"""

from unittest.mock import MagicMock, patch

import pytest

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
    """Create a mock LLM (LLMPort) for testing."""
    mock = MagicMock()
    # Setup with_structured_output chain for serial mode
    mock_structured = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"result": true}'
    mock_response.usage = None
    mock_response.raw = MagicMock(result=True)
    mock_structured.invoke.return_value = mock_response
    mock.with_structured_output.return_value = mock_structured
    return mock


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
def score_traits() -> list[LLMRubricTrait]:
    """Create sample score traits for testing."""
    return [
        LLMRubricTrait(
            name="clarity",
            description="Rate the clarity of the response",
            kind="score",
            min_score=1,
            max_score=5,
            higher_is_better=True,
        ),
        LLMRubricTrait(
            name="helpfulness",
            description="Rate how helpful the response is",
            kind="score",
            min_score=1,
            max_score=5,
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


# =============================================================================
# Async Configuration Tests
# =============================================================================


@pytest.mark.integration
def test_evaluator_parallel_mode_enabled_by_default(mock_llm: MagicMock, mock_model_config: ModelConfig) -> None:
    """Test that parallel mode is enabled by default."""
    evaluator = LLMTraitEvaluator(mock_llm, model_config=mock_model_config)

    assert evaluator._async_enabled is True


@pytest.mark.integration
def test_evaluator_parallel_mode_explicit_disabled(mock_llm: MagicMock, mock_model_config: ModelConfig) -> None:
    """Test disabling parallel mode explicitly."""
    evaluator = LLMTraitEvaluator(mock_llm, async_enabled=False, model_config=mock_model_config)

    assert evaluator._async_enabled is False


@pytest.mark.integration
def test_evaluator_parallel_mode_custom_workers(mock_llm: MagicMock, mock_model_config: ModelConfig) -> None:
    """Test setting custom max_workers."""
    evaluator = LLMTraitEvaluator(mock_llm, async_max_workers=8, model_config=mock_model_config)

    assert evaluator._async_max_workers == 8


@pytest.mark.integration
def test_evaluator_env_var_overrides(mock_llm: MagicMock, mock_model_config: ModelConfig) -> None:
    """Test that env vars are respected when params not provided."""
    with patch.dict(
        "os.environ",
        {"KARENINA_ASYNC_ENABLED": "false", "KARENINA_ASYNC_MAX_WORKERS": "16"},
    ):
        evaluator = LLMTraitEvaluator(mock_llm, model_config=mock_model_config)

        assert evaluator._async_enabled is False
        assert evaluator._async_max_workers == 16


@pytest.mark.integration
def test_evaluator_explicit_params_override_env(mock_llm: MagicMock, mock_model_config: ModelConfig) -> None:
    """Test that explicit params override env vars."""
    with patch.dict(
        "os.environ",
        {"KARENINA_ASYNC_ENABLED": "false", "KARENINA_ASYNC_MAX_WORKERS": "16"},
    ):
        evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, async_max_workers=4, model_config=mock_model_config)

        assert evaluator._async_enabled is True
        assert evaluator._async_max_workers == 4


# =============================================================================
# Sequential Boolean/Score Trait Evaluation Tests
# =============================================================================


@pytest.mark.integration
def test_evaluate_sequential_uses_parallel_invoker(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    boolean_traits: list[LLMRubricTrait],
) -> None:
    """Test that evaluate_sequential uses LLMParallelInvoker when async_enabled."""
    evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

    with patch("karenina.adapters.llm_parallel.LLMParallelInvoker") as mock_invoker_class:
        mock_invoker = MagicMock()
        mock_invoker_class.return_value = mock_invoker

        # Setup mock results for boolean traits
        mock_results = []
        for _ in boolean_traits:
            mock_result = MagicMock()
            mock_result.result = True  # Boolean result
            mock_results.append((mock_result, {"total_tokens": 10}, None))

        mock_invoker.invoke_batch.return_value = mock_results

        results, usage_list = evaluator.evaluate_sequential(sample_question, sample_answer, boolean_traits)

        # Verify LLMParallelInvoker was used
        mock_invoker_class.assert_called_once()
        mock_invoker.invoke_batch.assert_called_once()

        # Verify results
        assert len(results) == len(boolean_traits)
        for trait in boolean_traits:
            assert trait.name in results


@pytest.mark.integration
def test_evaluate_sequential_falls_back_when_disabled(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    boolean_traits: list[LLMRubricTrait],
) -> None:
    """Test that evaluate_sequential uses true sequential when async_enabled=False."""
    # Setup mock LLM for serial mode (with_structured_output chain)
    mock_structured = MagicMock()
    mock_response = MagicMock()
    mock_response.usage = None
    mock_result = MagicMock()
    mock_result.result = True
    mock_response.raw = mock_result
    mock_structured.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value = mock_structured

    evaluator = LLMTraitEvaluator(mock_llm, async_enabled=False, model_config=mock_model_config)

    results, usage_list = evaluator.evaluate_sequential(sample_question, sample_answer, boolean_traits)

    # Verify with_structured_output was called for each trait
    assert mock_llm.with_structured_output.call_count == len(boolean_traits)

    # Verify results
    assert len(results) == len(boolean_traits)
    for trait in boolean_traits:
        assert results[trait.name] is True


@pytest.mark.integration
def test_evaluate_sequential_handles_partial_errors(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    boolean_traits: list[LLMRubricTrait],
) -> None:
    """Test that parallel evaluation handles partial errors gracefully."""
    evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

    with patch("karenina.adapters.llm_parallel.LLMParallelInvoker") as mock_invoker_class:
        mock_invoker = MagicMock()
        mock_invoker_class.return_value = mock_invoker

        # Setup mock results with one error
        mock_results = []
        for i, _ in enumerate(boolean_traits):
            if i == 1:
                mock_results.append((None, None, ValueError("Test error")))
            else:
                mock_result = MagicMock()
                mock_result.result = True
                mock_results.append((mock_result, {"total_tokens": 10}, None))

        mock_invoker.invoke_batch.return_value = mock_results

        results, usage_list = evaluator.evaluate_sequential(sample_question, sample_answer, boolean_traits)

        # Verify the errored trait has None
        assert results[boolean_traits[1].name] is None

        # Verify other traits have valid results
        for i, trait in enumerate(boolean_traits):
            if i != 1:
                assert results[trait.name] is not None


@pytest.mark.integration
def test_evaluate_sequential_score_traits(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    score_traits: list[LLMRubricTrait],
) -> None:
    """Test parallel evaluation of score traits."""
    evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

    with patch("karenina.adapters.llm_parallel.LLMParallelInvoker") as mock_invoker_class:
        mock_invoker = MagicMock()
        mock_invoker_class.return_value = mock_invoker

        # Setup mock results for score traits
        mock_results = []
        for i, _ in enumerate(score_traits):
            mock_result = MagicMock()
            mock_result.score = 3 + i  # Numeric score
            mock_results.append((mock_result, {"total_tokens": 10}, None))

        mock_invoker.invoke_batch.return_value = mock_results

        results, usage_list = evaluator.evaluate_sequential(sample_question, sample_answer, score_traits)

        # Verify results are numeric
        for trait in score_traits:
            assert isinstance(results[trait.name], int)


# =============================================================================
# Sequential Literal Trait Evaluation Tests
# =============================================================================


@pytest.mark.integration
def test_evaluate_literal_sequential_uses_parallel_invoker(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    literal_traits: list[LLMRubricTrait],
) -> None:
    """Test that evaluate_literal_sequential uses LLMParallelInvoker."""
    evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

    with patch("karenina.adapters.llm_parallel.LLMParallelInvoker") as mock_invoker_class:
        mock_invoker = MagicMock()
        mock_invoker_class.return_value = mock_invoker

        # Setup mock results for literal traits
        mock_results = []
        for trait in literal_traits:
            mock_result = MagicMock()
            class_names = list(trait.classes.keys())
            mock_result.classification = class_names[0]  # First class
            mock_results.append((mock_result, {"total_tokens": 10}, None))

        mock_invoker.invoke_batch.return_value = mock_results

        scores, labels, usage_list = evaluator.evaluate_literal_sequential(
            sample_question, sample_answer, literal_traits
        )

        # Verify LLMParallelInvoker was used
        mock_invoker_class.assert_called_once()
        mock_invoker.invoke_batch.assert_called_once()

        # Verify results
        assert len(scores) == len(literal_traits)
        assert len(labels) == len(literal_traits)


@pytest.mark.integration
def test_evaluate_literal_sequential_falls_back_when_disabled(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    literal_traits: list[LLMRubricTrait],
) -> None:
    """Test that evaluate_literal_sequential uses true sequential when async_enabled=False."""
    # Setup mock LLM for serial mode
    mock_structured = MagicMock()
    mock_response = MagicMock()
    mock_response.usage = None
    mock_result = MagicMock()
    mock_result.classification = "positive"  # First class name
    mock_response.raw = mock_result
    mock_structured.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value = mock_structured

    evaluator = LLMTraitEvaluator(mock_llm, async_enabled=False, model_config=mock_model_config)

    scores, labels, usage_list = evaluator.evaluate_literal_sequential(sample_question, sample_answer, literal_traits)

    # Verify with_structured_output was called for each literal trait
    assert mock_llm.with_structured_output.call_count == len(literal_traits)

    # Verify results
    assert len(scores) == len(literal_traits)
    assert len(labels) == len(literal_traits)


@pytest.mark.integration
def test_evaluate_literal_sequential_handles_errors(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    literal_traits: list[LLMRubricTrait],
) -> None:
    """Test that literal parallel evaluation handles errors gracefully."""
    evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

    with patch("karenina.adapters.llm_parallel.LLMParallelInvoker") as mock_invoker_class:
        mock_invoker = MagicMock()
        mock_invoker_class.return_value = mock_invoker

        # Setup mock results with one error
        mock_results = []
        for i, trait in enumerate(literal_traits):
            if i == 0:
                mock_results.append((None, None, ValueError("Test error")))
            else:
                mock_result = MagicMock()
                class_names = list(trait.classes.keys())
                mock_result.classification = class_names[1] if len(class_names) > 1 else class_names[0]
                mock_results.append((mock_result, {"total_tokens": 10}, None))

        mock_invoker.invoke_batch.return_value = mock_results

        scores, labels, usage_list = evaluator.evaluate_literal_sequential(
            sample_question, sample_answer, literal_traits
        )

        # Verify the errored trait has score -1 and error label
        assert scores[literal_traits[0].name] == -1
        assert "[EVALUATION_ERROR:" in labels[literal_traits[0].name]

        # Verify other traits have valid results
        for i, trait in enumerate(literal_traits):
            if i != 0:
                assert scores[trait.name] >= 0


@pytest.mark.integration
def test_evaluate_literal_sequential_empty_list(mock_model_config: ModelConfig) -> None:
    """Test that empty literal traits list returns empty results."""
    mock_llm = MagicMock()

    evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

    # Pass empty list or list with no literal traits
    non_literal_traits = [LLMRubricTrait(name="test", description="test", kind="boolean")]

    scores, labels, usage_list = evaluator.evaluate_literal_sequential("question", "answer", non_literal_traits)

    # Should return empty results
    assert scores == {}
    assert labels == {}
    assert usage_list == []


# =============================================================================
# Usage Metadata Tests
# =============================================================================


@pytest.mark.integration
def test_evaluate_sequential_preserves_usage_metadata(
    mock_llm: MagicMock,
    mock_model_config: ModelConfig,
    sample_question: str,
    sample_answer: str,
    boolean_traits: list[LLMRubricTrait],
) -> None:
    """Test that usage metadata is preserved per-trait in sequential evaluation."""
    evaluator = LLMTraitEvaluator(mock_llm, async_enabled=True, model_config=mock_model_config)

    with patch("karenina.adapters.llm_parallel.LLMParallelInvoker") as mock_invoker_class:
        mock_invoker = MagicMock()
        mock_invoker_class.return_value = mock_invoker

        # Setup mock results with different usage per trait
        mock_results = []
        for i, _ in enumerate(boolean_traits):
            mock_result = MagicMock()
            mock_result.result = True
            mock_results.append((mock_result, {"total_tokens": 10 + i, "input_tokens": 5}, None))

        mock_invoker.invoke_batch.return_value = mock_results

        results, usage_list = evaluator.evaluate_sequential(sample_question, sample_answer, boolean_traits)

        # Verify usage list has one entry per trait
        assert len(usage_list) == len(boolean_traits)

        # Verify each entry has different token counts
        for i, usage in enumerate(usage_list):
            assert usage.get("total_tokens") == 10 + i
