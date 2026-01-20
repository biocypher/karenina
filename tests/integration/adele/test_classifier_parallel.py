"""Integration tests for ADeLe classifier parallel invocation.

Tests verify that the parallel invocation feature works correctly
with the QuestionClassifier when using sequential trait evaluation mode.
"""

from unittest.mock import MagicMock, patch

import pytest

from karenina.integrations.adele.classifier import QuestionClassifier
from karenina.integrations.adele.traits import get_adele_trait

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    mock = MagicMock()
    return mock


@pytest.fixture
def sample_question() -> str:
    """Sample question text for testing."""
    return "What gene causes breast cancer?"


@pytest.fixture
def sample_traits() -> list:
    """Get a subset of ADeLe traits for testing."""
    return [
        get_adele_trait("attention_and_scan"),
        get_adele_trait("volume"),
        get_adele_trait("atypicality"),
    ]


# =============================================================================
# Parallel Mode Tests
# =============================================================================


@pytest.mark.integration
def test_classifier_parallel_mode_enabled_by_default(mock_llm: MagicMock) -> None:
    """Test that parallel mode is enabled by default when no env var set."""
    with patch.dict("os.environ", {}, clear=True):
        classifier = QuestionClassifier(llm=mock_llm)

        assert classifier._async_enabled is True
        assert classifier._async_max_workers == 2  # Default


@pytest.mark.integration
def test_classifier_parallel_mode_explicit_disabled() -> None:
    """Test disabling parallel mode explicitly."""
    mock_llm = MagicMock()
    classifier = QuestionClassifier(llm=mock_llm, async_enabled=False)

    assert classifier._async_enabled is False


@pytest.mark.integration
def test_classifier_parallel_mode_custom_workers() -> None:
    """Test setting custom max_workers."""
    mock_llm = MagicMock()
    classifier = QuestionClassifier(llm=mock_llm, async_max_workers=8)

    assert classifier._async_max_workers == 8


@pytest.mark.integration
def test_classifier_env_var_overrides(mock_llm: MagicMock) -> None:
    """Test that env vars are respected when params not provided."""
    with patch.dict(
        "os.environ",
        {"KARENINA_ASYNC_ENABLED": "false", "KARENINA_ASYNC_MAX_WORKERS": "16"},
    ):
        classifier = QuestionClassifier(llm=mock_llm)

        assert classifier._async_enabled is False
        assert classifier._async_max_workers == 16


@pytest.mark.integration
def test_classifier_explicit_params_override_env(mock_llm: MagicMock) -> None:
    """Test that explicit params override env vars."""
    with patch.dict(
        "os.environ",
        {"KARENINA_ASYNC_ENABLED": "false", "KARENINA_ASYNC_MAX_WORKERS": "16"},
    ):
        classifier = QuestionClassifier(llm=mock_llm, async_enabled=True, async_max_workers=4)

        assert classifier._async_enabled is True
        assert classifier._async_max_workers == 4


# =============================================================================
# Sequential Mode Classification Tests
# =============================================================================


@pytest.mark.integration
def test_classifier_sequential_mode_with_parallel_execution(
    mock_llm: MagicMock,
    sample_question: str,
    sample_traits: list,
) -> None:
    """Test that sequential mode uses LLMParallelInvoker when async_enabled."""
    classifier = QuestionClassifier(
        llm=mock_llm,
        trait_eval_mode="sequential",
        async_enabled=True,
    )

    # Mock the parallel invoker (now in adapters module)
    with patch("karenina.adapters.llm_parallel.LLMParallelInvoker") as mock_invoker_class:
        mock_invoker = MagicMock()
        mock_invoker_class.return_value = mock_invoker

        # Setup mock results - one per trait
        mock_results = []
        for i, trait in enumerate(sample_traits):
            # Create a mock result with a class name
            mock_result = MagicMock()
            class_names = list(trait.classes.keys())
            mock_result.classification = class_names[i % len(class_names)]
            mock_results.append((mock_result, {"total_tokens": 10}, None))

        mock_invoker.invoke_batch.return_value = mock_results

        # Execute classification
        result = classifier._classify_single_sequential(sample_question, sample_traits, question_id="test-1")

        # Verify LLMParallelInvoker was used
        mock_invoker_class.assert_called_once()
        mock_invoker.invoke_batch.assert_called_once()

        # Verify all traits have results
        assert len(result.scores) == len(sample_traits)
        assert len(result.labels) == len(sample_traits)


@pytest.mark.integration
def test_classifier_sequential_mode_without_parallel_execution(
    sample_question: str,
    sample_traits: list,
) -> None:
    """Test that sequential mode falls back when async_enabled=False."""
    # Create a mock LLM that supports with_structured_output
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    classifier = QuestionClassifier(
        llm=mock_llm,
        trait_eval_mode="sequential",
        async_enabled=False,
    )

    # Setup mock responses for the structured LLM
    mock_responses = []
    for i, trait in enumerate(sample_traits):
        mock_response = MagicMock()
        mock_result = MagicMock()
        class_names = list(trait.classes.keys())
        mock_result.classification = class_names[i % len(class_names)]
        mock_response.raw = mock_result
        mock_response.usage = MagicMock(total_tokens=10, input_tokens=5, output_tokens=5)
        mock_responses.append(mock_response)

    mock_structured_llm.invoke.side_effect = mock_responses

    # Execute classification
    result = classifier._classify_single_sequential(sample_question, sample_traits, question_id="test-1")

    # Verify with_structured_output().invoke() was called for each trait
    assert mock_structured_llm.invoke.call_count == len(sample_traits)

    # Verify all traits have results
    assert len(result.scores) == len(sample_traits)
    assert len(result.labels) == len(sample_traits)


@pytest.mark.integration
def test_classifier_parallel_handles_partial_errors(
    mock_llm: MagicMock,
    sample_question: str,
    sample_traits: list,
) -> None:
    """Test that parallel classification handles partial errors gracefully."""
    classifier = QuestionClassifier(
        llm=mock_llm,
        trait_eval_mode="sequential",
        async_enabled=True,
    )

    with patch("karenina.adapters.llm_parallel.LLMParallelInvoker") as mock_invoker_class:
        mock_invoker = MagicMock()
        mock_invoker_class.return_value = mock_invoker

        # Setup mock results with one error
        mock_results = []
        for i, trait in enumerate(sample_traits):
            if i == 1:  # Fail the second trait
                mock_results.append((None, None, ValueError("Test error")))
            else:
                mock_result = MagicMock()
                class_names = list(trait.classes.keys())
                mock_result.classification = class_names[0]
                mock_results.append((mock_result, {"total_tokens": 10}, None))

        mock_invoker.invoke_batch.return_value = mock_results

        result = classifier._classify_single_sequential(sample_question, sample_traits, question_id="test-1")

        # Verify the errored trait has score -1
        assert result.scores[sample_traits[1].name] == -1
        assert "[ERROR:" in result.labels[sample_traits[1].name]

        # Verify other traits have valid scores
        for i, trait in enumerate(sample_traits):
            if i != 1:
                assert result.scores[trait.name] >= 0


@pytest.mark.integration
def test_classifier_parallel_usage_aggregation(
    mock_llm: MagicMock,
    sample_question: str,
    sample_traits: list,
) -> None:
    """Test that usage metadata is aggregated correctly."""
    classifier = QuestionClassifier(
        llm=mock_llm,
        trait_eval_mode="sequential",
        async_enabled=True,
    )

    with patch("karenina.adapters.llm_parallel.LLMParallelInvoker") as mock_invoker_class:
        mock_invoker = MagicMock()
        mock_invoker_class.return_value = mock_invoker

        # Setup mock results with usage metadata
        mock_results = []
        for i, trait in enumerate(sample_traits):
            mock_result = MagicMock()
            class_names = list(trait.classes.keys())
            mock_result.classification = class_names[0]
            mock_results.append(
                (mock_result, {"total_tokens": 10 + i, "input_tokens": 5, "output_tokens": 5 + i}, None)
            )

        mock_invoker.invoke_batch.return_value = mock_results

        result = classifier._classify_single_sequential(sample_question, sample_traits, question_id="test-1")

        # Verify usage metadata is aggregated
        assert result.usage_metadata["calls"] == len(sample_traits)
        assert result.usage_metadata["total_tokens"] == sum(10 + i for i in range(len(sample_traits)))


# =============================================================================
# Batch Classification Tests
# =============================================================================


@pytest.mark.integration
def test_classifier_batch_with_sequential_mode(mock_llm: MagicMock) -> None:
    """Test batch classification uses sequential mode with parallel invocation."""
    classifier = QuestionClassifier(
        llm=mock_llm,
        trait_eval_mode="sequential",
        async_enabled=True,
    )

    questions = [
        ("q1", "What gene causes breast cancer?"),
        ("q2", "How does CRISPR work?"),
    ]

    with patch.object(classifier, "_classify_single_sequential") as mock_classify:
        mock_result = MagicMock()
        mock_result.scores = {"attention_and_scan": 3}
        mock_result.labels = {"attention_and_scan": "middle_high"}
        mock_result.usage_metadata = {"calls": 1, "total_tokens": 10}
        mock_classify.return_value = mock_result

        results = classifier.classify_batch(questions)

        # Verify sequential classification was called for each question
        assert mock_classify.call_count == len(questions)
        assert len(results) == len(questions)
