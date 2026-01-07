"""Unit tests for LLMFeedbackGenerator.

These tests verify the feedback generation logic without making real LLM calls.
Run with: uv run pytest tests/integrations/gepa/test_feedback_generator.py -v
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from karenina.integrations.gepa.data_types import KareninaDataInst, KareninaTrajectory
from karenina.schemas.workflow.models import ModelConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def feedback_model_config():
    """Valid ModelConfig for feedback LLM."""
    return ModelConfig(
        id="feedback-model",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.7,
        interface="langchain",
    )


@pytest.fixture
def mock_data_inst():
    """Create a mock KareninaDataInst."""
    return KareninaDataInst(
        question_id="test-q1",
        question_text="What is the gene symbol for BCL2?",
        raw_answer="BCL2",
        template_code="class Answer(BaseAnswer): pass",
    )


@pytest.fixture
def mock_model_config():
    """Create a mock ModelConfig for trajectories."""
    return ModelConfig(
        id="test-model",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        interface="langchain",
    )


@pytest.fixture
def mock_verification_result_failed():
    """Create a mock failed VerificationResult."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = False
    result.template.raw_llm_response = "The gene is B-cell lymphoma 2"
    result.template.parsed_llm_response = {"gene_symbol": "B-cell lymphoma 2"}
    result.rubric = None
    result.metadata = MagicMock()
    result.metadata.error = None
    return result


@pytest.fixture
def mock_verification_result_passed():
    """Create a mock passed VerificationResult."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.template.raw_llm_response = "The gene symbol is BCL2"
    result.template.parsed_llm_response = {"gene_symbol": "BCL2"}
    result.rubric = None
    result.metadata = MagicMock()
    result.metadata.error = None
    return result


@pytest.fixture
def failed_trajectory(mock_data_inst, mock_model_config, mock_verification_result_failed):
    """Create a failed KareninaTrajectory."""
    return KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-haiku-4-5",
        model_config=mock_model_config,
        optimized_components={"answering_system_prompt": "You are helpful."},
        verification_result=mock_verification_result_failed,
        score=0.0,
        raw_llm_response="The gene is B-cell lymphoma 2, which is involved in apoptosis regulation.",
        parsing_error=None,
        failed_fields=["gene_symbol"],
        rubric_scores=None,
    )


@pytest.fixture
def passed_trajectory(mock_data_inst, mock_model_config, mock_verification_result_passed):
    """Create a passed KareninaTrajectory."""
    return KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-sonnet-4-5",
        model_config=mock_model_config,
        optimized_components={"answering_system_prompt": "You are helpful."},
        verification_result=mock_verification_result_passed,
        score=1.0,
        raw_llm_response="The gene symbol is BCL2.",
        parsing_error=None,
        failed_fields=None,
        rubric_scores=None,
    )


@pytest.fixture
def sample_rubric_scores():
    """Sample rubric scores for testing."""
    return {
        "consequentiality": True,
        "clarity": 0.8,
        "precision_recall": {"precision": 0.9, "recall": 0.7, "f1": 0.78},
        "format_compliance": False,
    }


# =============================================================================
# Step 1: Initialization Tests
# =============================================================================


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_init_with_valid_config(mock_init_model, feedback_model_config):
    """LLMFeedbackGenerator initializes with valid ModelConfig."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    mock_init_model.return_value = mock_llm

    generator = LLMFeedbackGenerator(feedback_model_config)

    assert generator.llm == mock_llm
    assert generator.model_config == feedback_model_config
    mock_init_model.assert_called_once()


def test_init_missing_model_name_raises():
    """ModelConfig validates that model_name is required for non-manual interfaces.

    Note: This validation happens at the ModelConfig level, not LLMFeedbackGenerator.
    LLMFeedbackGenerator has a redundant check for defense in depth.
    """
    # ModelConfig itself validates model_name for non-manual interfaces
    with pytest.raises(Exception):  # pydantic ValidationError
        ModelConfig(
            id="test",
            model_name=None,
            interface="langchain",
        )


def test_init_missing_provider_for_langchain_raises():
    """Raises ValueError when provider missing for langchain interface."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    config = ModelConfig(
        id="test",
        model_provider=None,  # Missing for langchain
        model_name="claude-haiku-4-5",
        interface="langchain",
    )

    with pytest.raises(ValueError, match="model_provider is required"):
        LLMFeedbackGenerator(config)


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_init_llm_failure_raises_runtime_error(mock_init_model, feedback_model_config):
    """Raises RuntimeError when init_chat_model_unified fails."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_init_model.side_effect = Exception("API key not found")

    with pytest.raises(RuntimeError, match="Failed to initialize feedback LLM"):
        LLMFeedbackGenerator(feedback_model_config)


# =============================================================================
# Step 2: Prompt Building Tests
# =============================================================================


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_build_single_feedback_prompt_format(mock_init_model, feedback_model_config, failed_trajectory):
    """Verify prompt structure for single trajectory analysis."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_init_model.return_value = Mock()
    generator = LLMFeedbackGenerator(feedback_model_config)

    prompt = generator._build_single_feedback_prompt(failed_trajectory)

    # Verify key sections are present
    assert "## Question" in prompt
    assert "What is the gene symbol for BCL2?" in prompt
    assert "## Expected Answer" in prompt
    assert "BCL2" in prompt
    assert "## Model Response" in prompt
    assert "claude-haiku-4-5" in prompt
    assert "B-cell lymphoma 2" in prompt
    assert "## Verification Result" in prompt
    assert "Failed Fields: gene_symbol" in prompt


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_build_differential_feedback_prompt_includes_full_traces(
    mock_init_model, feedback_model_config, failed_trajectory, passed_trajectory
):
    """Verify successful traces are included WITHOUT truncation."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_init_model.return_value = Mock()
    generator = LLMFeedbackGenerator(feedback_model_config)

    prompt = generator._build_differential_feedback_prompt(failed_trajectory, [passed_trajectory])

    # Verify full successful trace is included (no truncation)
    assert "SUCCESSFUL TRACES" in prompt
    assert "claude-sonnet-4-5" in prompt
    assert "The gene symbol is BCL2." in prompt  # Full response, not truncated

    # Verify failed trace is included
    assert "FAILED TRACE" in prompt
    assert "claude-haiku-4-5" in prompt
    assert "B-cell lymphoma 2" in prompt


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_build_differential_feedback_prompt_multiple_successes(
    mock_init_model, feedback_model_config, failed_trajectory, passed_trajectory, mock_data_inst, mock_model_config
):
    """Verify all successful trajectories are included."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_init_model.return_value = Mock()
    generator = LLMFeedbackGenerator(feedback_model_config)

    # Create a second successful trajectory
    mock_result2 = MagicMock()
    mock_result2.template = MagicMock()
    mock_result2.template.verify_result = True
    mock_result2.template.raw_llm_response = "BCL2 is the official symbol"
    mock_result2.template.parsed_llm_response = {"gene_symbol": "BCL2"}

    passed_trajectory_2 = KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="gpt-4o",
        model_config=mock_model_config,
        optimized_components={},
        verification_result=mock_result2,
        score=1.0,
        raw_llm_response="BCL2 is the official symbol for the gene.",
    )

    prompt = generator._build_differential_feedback_prompt(failed_trajectory, [passed_trajectory, passed_trajectory_2])

    # Both successful models should be included
    assert "claude-sonnet-4-5" in prompt
    assert "gpt-4o" in prompt
    assert "Successful Trace 1" in prompt
    assert "Successful Trace 2" in prompt


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_build_rubric_feedback_prompt_format(
    mock_init_model, feedback_model_config, failed_trajectory, sample_rubric_scores
):
    """Verify rubric scores are formatted correctly."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_init_model.return_value = Mock()
    generator = LLMFeedbackGenerator(feedback_model_config)

    prompt = generator._build_rubric_feedback_prompt(failed_trajectory, sample_rubric_scores)

    # Verify rubric scores are formatted
    assert "## Rubric Evaluation Results" in prompt
    assert "consequentiality: PASSED" in prompt
    assert "clarity: 0.80" in prompt
    assert "format_compliance: FAILED" in prompt


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_build_rubric_feedback_identifies_failed_traits(
    mock_init_model, feedback_model_config, failed_trajectory, sample_rubric_scores
):
    """Verify failed traits are highlighted."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_init_model.return_value = Mock()
    generator = LLMFeedbackGenerator(feedback_model_config)

    prompt = generator._build_rubric_feedback_prompt(failed_trajectory, sample_rubric_scores)

    # Verify failed traits are identified
    assert "Failed/Low-Scoring Traits" in prompt
    assert "format_compliance" in prompt


# =============================================================================
# Step 3: Feedback Generation Tests (mocked LLM)
# =============================================================================


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_generate_single_feedback_calls_llm(mock_init_model, feedback_model_config, failed_trajectory):
    """Verify LLM is invoked with correct messages."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="The model failed to extract the gene symbol correctly.")
    mock_init_model.return_value = mock_llm

    generator = LLMFeedbackGenerator(feedback_model_config)
    feedback = generator.generate_single_feedback(failed_trajectory)

    # Verify LLM was called
    mock_llm.invoke.assert_called_once()

    # Verify messages structure
    call_args = mock_llm.invoke.call_args[0][0]
    assert len(call_args) == 2  # SystemMessage + HumanMessage
    assert "SystemMessage" in str(type(call_args[0]))
    assert "HumanMessage" in str(type(call_args[1]))

    # Verify feedback content
    assert feedback == "The model failed to extract the gene symbol correctly."


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_generate_differential_feedback_calls_llm(
    mock_init_model, feedback_model_config, failed_trajectory, passed_trajectory
):
    """Verify differential analysis prompt is sent to LLM."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="The successful model used the official gene symbol.")
    mock_init_model.return_value = mock_llm

    generator = LLMFeedbackGenerator(feedback_model_config)
    feedback = generator.generate_differential_feedback(failed_trajectory, [passed_trajectory])

    mock_llm.invoke.assert_called_once()
    assert "The successful model used the official gene symbol." in feedback


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_generate_rubric_feedback_calls_llm(
    mock_init_model, feedback_model_config, failed_trajectory, sample_rubric_scores
):
    """Verify rubric analysis prompt is sent to LLM."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Format compliance failed due to incorrect structure.")
    mock_init_model.return_value = mock_llm

    generator = LLMFeedbackGenerator(feedback_model_config)
    feedback = generator.generate_rubric_feedback(failed_trajectory, sample_rubric_scores)

    mock_llm.invoke.assert_called_once()
    assert "Format compliance failed" in feedback


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_generate_complete_feedback_combines_all(
    mock_init_model, feedback_model_config, failed_trajectory, passed_trajectory, sample_rubric_scores
):
    """Verify template + rubric feedback are combined correctly."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    # First call for differential, second for rubric
    mock_llm.invoke.side_effect = [
        Mock(content="Differential analysis result."),
        Mock(content="Rubric analysis result."),
    ]
    mock_init_model.return_value = mock_llm

    generator = LLMFeedbackGenerator(feedback_model_config)
    feedback = generator.generate_complete_feedback(
        failed_trajectory=failed_trajectory,
        successful_trajectories=[passed_trajectory],
        rubric_scores=sample_rubric_scores,
    )

    # Verify both sections are present
    assert "--- TEMPLATE VERIFICATION FEEDBACK ---" in feedback
    assert "Differential analysis result." in feedback
    assert "--- RUBRIC EVALUATION FEEDBACK ---" in feedback
    assert "Rubric analysis result." in feedback

    # Verify LLM was called twice
    assert mock_llm.invoke.call_count == 2


# =============================================================================
# Step 4: Edge Case Tests
# =============================================================================


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_generate_complete_feedback_no_successes(mock_init_model, feedback_model_config, failed_trajectory):
    """Falls back to single feedback when no successful traces."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Single trajectory analysis.")
    mock_init_model.return_value = mock_llm

    generator = LLMFeedbackGenerator(feedback_model_config)
    feedback = generator.generate_complete_feedback(
        failed_trajectory=failed_trajectory,
        successful_trajectories=None,  # No successes
        rubric_scores=None,
    )

    # Should use single feedback, not differential
    assert "--- TEMPLATE VERIFICATION FEEDBACK ---" in feedback
    assert "Single trajectory analysis." in feedback
    # No rubric section
    assert "--- RUBRIC EVALUATION FEEDBACK ---" not in feedback

    # LLM called only once (no rubric)
    assert mock_llm.invoke.call_count == 1


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_generate_complete_feedback_no_rubrics(
    mock_init_model, feedback_model_config, failed_trajectory, passed_trajectory
):
    """Omits rubric section when rubric_scores is None."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Differential analysis.")
    mock_init_model.return_value = mock_llm

    generator = LLMFeedbackGenerator(feedback_model_config)
    feedback = generator.generate_complete_feedback(
        failed_trajectory=failed_trajectory,
        successful_trajectories=[passed_trajectory],
        rubric_scores=None,  # No rubrics
    )

    assert "--- TEMPLATE VERIFICATION FEEDBACK ---" in feedback
    assert "--- RUBRIC EVALUATION FEEDBACK ---" not in feedback


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_generate_complete_feedback_empty_rubric_scores(mock_init_model, feedback_model_config, failed_trajectory):
    """Handles empty rubric_scores dict."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Single analysis.")
    mock_init_model.return_value = mock_llm

    generator = LLMFeedbackGenerator(feedback_model_config)
    feedback = generator.generate_complete_feedback(
        failed_trajectory=failed_trajectory,
        successful_trajectories=None,
        rubric_scores={},  # Empty dict - treated as falsy
    )

    # Empty dict should be treated as no rubrics
    assert "--- RUBRIC EVALUATION FEEDBACK ---" not in feedback


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_response_content_extraction_with_content_attr(mock_init_model, feedback_model_config, failed_trajectory):
    """Handles response with .content attribute."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Response with content attribute."
    mock_llm.invoke.return_value = mock_response
    mock_init_model.return_value = mock_llm

    generator = LLMFeedbackGenerator(feedback_model_config)
    feedback = generator.generate_single_feedback(failed_trajectory)

    assert feedback == "Response with content attribute."


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_response_content_extraction_without_content_attr(mock_init_model, feedback_model_config, failed_trajectory):
    """Handles response without .content attribute (falls back to str())."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    # Create a response without .content
    mock_response = "Plain string response"
    mock_llm.invoke.return_value = mock_response
    mock_init_model.return_value = mock_llm

    generator = LLMFeedbackGenerator(feedback_model_config)
    feedback = generator.generate_single_feedback(failed_trajectory)

    assert feedback == "Plain string response"


# =============================================================================
# Step 5: Additional Validation Tests
# =============================================================================


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_trajectory_with_parsing_error(mock_init_model, feedback_model_config, mock_data_inst, mock_model_config):
    """Test feedback generation when trajectory has parsing error."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Parsing error analysis.")
    mock_init_model.return_value = mock_llm

    # Create trajectory with parsing error
    mock_result = MagicMock()
    mock_result.template = MagicMock()
    mock_result.template.verify_result = False
    mock_result.template.parsed_llm_response = None

    trajectory = KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-haiku-4-5",
        model_config=mock_model_config,
        optimized_components={},
        verification_result=mock_result,
        score=0.0,
        raw_llm_response="Invalid response",
        parsing_error="Failed to parse JSON: Unexpected token",
        failed_fields=None,
    )

    generator = LLMFeedbackGenerator(feedback_model_config)
    prompt = generator._build_single_feedback_prompt(trajectory)

    assert "Parsing Error: Failed to parse JSON" in prompt


@patch("karenina.integrations.gepa.feedback.init_chat_model_unified")
def test_metric_trait_rubric_scores(mock_init_model, feedback_model_config, failed_trajectory):
    """Test rubric feedback with metric traits (precision/recall/f1)."""
    from karenina.integrations.gepa.feedback import LLMFeedbackGenerator

    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Metric analysis.")
    mock_init_model.return_value = mock_llm

    metric_scores = {
        "entity_extraction": {"precision": 0.3, "recall": 0.8, "f1": 0.45},
    }

    generator = LLMFeedbackGenerator(feedback_model_config)
    prompt = generator._build_rubric_feedback_prompt(failed_trajectory, metric_scores)

    # Metric traits should be formatted with all values
    assert "entity_extraction" in prompt
    assert "precision: 0.30" in prompt
    assert "recall: 0.80" in prompt
    assert "f1: 0.45" in prompt

    # Low F1 should be identified as failed
    assert "Failed/Low-Scoring Traits" in prompt
    assert "entity_extraction" in prompt
