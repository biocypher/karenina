"""Integration tests for KareninaAdapter feedback generation.

These tests verify the integration between KareninaAdapter and LLMFeedbackGenerator.
Run with: uv run pytest tests/integrations/gepa/test_adapter_feedback.py -v
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from karenina.integrations.gepa.config import ObjectiveConfig, OptimizationTarget
from karenina.integrations.gepa.data_types import KareninaDataInst, KareninaTrajectory
from karenina.schemas.workflow.models import ModelConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def feedback_model_config():
    """ModelConfig for feedback LLM."""
    return ModelConfig(
        id="feedback-model",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.7,
        interface="langchain",
    )


@pytest.fixture
def answering_model_config():
    """ModelConfig for answering LLM."""
    return ModelConfig(
        id="answering-model",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
        interface="langchain",
    )


@pytest.fixture
def mock_benchmark():
    """Create a mock Benchmark."""
    benchmark = MagicMock()
    benchmark.run_verification = MagicMock(return_value=MagicMock())
    return benchmark


@pytest.fixture
def mock_verification_config(answering_model_config):
    """Create a mock VerificationConfig."""
    config = MagicMock()
    config.answering_models = [answering_model_config]
    config.model_copy = MagicMock(return_value=config)
    return config


@pytest.fixture
def objective_config():
    """Default ObjectiveConfig for tests."""
    return ObjectiveConfig()


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
def mock_trajectory_failed(mock_data_inst, answering_model_config):
    """Create a failed KareninaTrajectory."""
    mock_result = MagicMock()
    mock_result.template = MagicMock()
    mock_result.template.verify_result = False
    mock_result.template.parsed_llm_response = None

    return KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-haiku-4-5",
        model_config=answering_model_config,
        optimized_components={"answering_system_prompt": "You are helpful."},
        verification_result=mock_result,
        score=0.0,
        raw_llm_response="The gene is B-cell lymphoma 2.",
        parsing_error=None,
        failed_fields=["gene_symbol"],
        rubric_scores={"clarity": 0.8},
    )


@pytest.fixture
def mock_trajectory_passed(mock_data_inst, answering_model_config):
    """Create a passed KareninaTrajectory."""
    mock_result = MagicMock()
    mock_result.template = MagicMock()
    mock_result.template.verify_result = True
    mock_result.template.parsed_llm_response = {"gene_symbol": "BCL2"}

    return KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-sonnet-4-5",
        model_config=answering_model_config,
        optimized_components={"answering_system_prompt": "You are helpful."},
        verification_result=mock_result,
        score=1.0,
        raw_llm_response="The gene symbol is BCL2.",
        parsing_error=None,
        failed_fields=None,
        rubric_scores=None,
    )


# =============================================================================
# Step 1: Adapter Initialization with Feedback Generator
# =============================================================================


@patch("karenina.integrations.gepa.adapter.LLMFeedbackGenerator")
def test_adapter_init_with_feedback_model(
    mock_generator_class, mock_benchmark, mock_verification_config, feedback_model_config, objective_config
):
    """Adapter creates LLMFeedbackGenerator when config provided."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    mock_generator_instance = Mock()
    mock_generator_class.return_value = mock_generator_instance

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=feedback_model_config,
    )

    assert adapter.feedback_generator == mock_generator_instance
    mock_generator_class.assert_called_once_with(feedback_model_config)


def test_adapter_init_without_feedback_model(mock_benchmark, mock_verification_config, objective_config):
    """Adapter.feedback_generator is None when no config."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=None,
    )

    assert adapter.feedback_generator is None


@patch("karenina.integrations.gepa.adapter.LLMFeedbackGenerator")
def test_adapter_differential_analysis_flag(
    mock_generator_class, mock_benchmark, mock_verification_config, feedback_model_config, objective_config
):
    """enable_differential_analysis is respected."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    mock_generator_class.return_value = Mock()

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=feedback_model_config,
        enable_differential_analysis=False,
    )

    assert adapter.enable_differential_analysis is False


# =============================================================================
# Step 2: make_reflective_dataset Integration
# =============================================================================


@patch("karenina.integrations.gepa.adapter.LLMFeedbackGenerator")
def test_make_reflective_dataset_uses_llm_feedback(
    mock_generator_class,
    mock_benchmark,
    mock_verification_config,
    feedback_model_config,
    mock_trajectory_failed,
    objective_config,
):
    """When feedback_generator set, uses LLM-generated feedback."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    # Setup mock generator
    mock_generator = Mock()
    mock_generator.generate_complete_feedback.return_value = "LLM generated feedback"
    mock_generator_class.return_value = mock_generator

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=feedback_model_config,
    )

    # Create mock EvaluationBatch with one failed trajectory
    mock_eval_batch = MagicMock()
    mock_eval_batch.trajectories = [mock_trajectory_failed]

    result = adapter.make_reflective_dataset(
        candidate={"answering_system_prompt": "You are helpful."},
        eval_batch=mock_eval_batch,
        components_to_update=["answering_system_prompt"],
    )

    # Verify LLM feedback was used
    mock_generator.generate_complete_feedback.assert_called_once()
    assert "LLM generated feedback" in result["answering_system_prompt"][0]["Feedback"]


def test_make_reflective_dataset_fallback_programmatic(
    mock_benchmark, mock_verification_config, mock_trajectory_failed, objective_config
):
    """When no feedback_generator, uses programmatic feedback."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=None,  # No feedback generator
    )

    mock_eval_batch = MagicMock()
    mock_eval_batch.trajectories = [mock_trajectory_failed]

    result = adapter.make_reflective_dataset(
        candidate={"answering_system_prompt": "You are helpful."},
        eval_batch=mock_eval_batch,
        components_to_update=["answering_system_prompt"],
    )

    # Verify programmatic feedback (contains expected fields)
    feedback = result["answering_system_prompt"][0]["Feedback"]
    assert "Failed fields: gene_symbol" in feedback
    assert "Expected answer: BCL2" in feedback


@patch("karenina.integrations.gepa.adapter.LLMFeedbackGenerator")
def test_make_reflective_dataset_differential_enabled(
    mock_generator_class,
    mock_benchmark,
    mock_verification_config,
    feedback_model_config,
    mock_trajectory_failed,
    mock_trajectory_passed,
    objective_config,
):
    """Passes successful trajectories when differential enabled."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    mock_generator = Mock()
    mock_generator.generate_complete_feedback.return_value = "Differential feedback"
    mock_generator_class.return_value = mock_generator

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=feedback_model_config,
        enable_differential_analysis=True,
    )

    mock_eval_batch = MagicMock()
    mock_eval_batch.trajectories = [mock_trajectory_failed, mock_trajectory_passed]

    adapter.make_reflective_dataset(
        candidate={"answering_system_prompt": "You are helpful."},
        eval_batch=mock_eval_batch,
        components_to_update=["answering_system_prompt"],
    )

    # Verify successful_trajectories was passed (not None)
    call_kwargs = mock_generator.generate_complete_feedback.call_args[1]
    assert call_kwargs["successful_trajectories"] is not None
    assert len(call_kwargs["successful_trajectories"]) == 1


@patch("karenina.integrations.gepa.adapter.LLMFeedbackGenerator")
def test_make_reflective_dataset_differential_disabled(
    mock_generator_class,
    mock_benchmark,
    mock_verification_config,
    feedback_model_config,
    mock_trajectory_failed,
    mock_trajectory_passed,
    objective_config,
):
    """Passes None for successes when differential disabled."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    mock_generator = Mock()
    mock_generator.generate_complete_feedback.return_value = "Single feedback"
    mock_generator_class.return_value = mock_generator

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=feedback_model_config,
        enable_differential_analysis=False,  # Disabled
    )

    mock_eval_batch = MagicMock()
    mock_eval_batch.trajectories = [mock_trajectory_failed, mock_trajectory_passed]

    adapter.make_reflective_dataset(
        candidate={"answering_system_prompt": "You are helpful."},
        eval_batch=mock_eval_batch,
        components_to_update=["answering_system_prompt"],
    )

    # Verify successful_trajectories is None when disabled
    call_kwargs = mock_generator.generate_complete_feedback.call_args[1]
    assert call_kwargs["successful_trajectories"] is None


@patch("karenina.integrations.gepa.adapter.LLMFeedbackGenerator")
def test_make_reflective_dataset_includes_rubric_scores(
    mock_generator_class,
    mock_benchmark,
    mock_verification_config,
    feedback_model_config,
    mock_trajectory_failed,
    objective_config,
):
    """Passes rubric_scores from trajectory to generator."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    mock_generator = Mock()
    mock_generator.generate_complete_feedback.return_value = "Rubric feedback"
    mock_generator_class.return_value = mock_generator

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=feedback_model_config,
    )

    mock_eval_batch = MagicMock()
    mock_eval_batch.trajectories = [mock_trajectory_failed]

    adapter.make_reflective_dataset(
        candidate={"answering_system_prompt": "You are helpful."},
        eval_batch=mock_eval_batch,
        components_to_update=["answering_system_prompt"],
    )

    # Verify rubric_scores was passed
    call_kwargs = mock_generator.generate_complete_feedback.call_args[1]
    assert call_kwargs["rubric_scores"] == {"clarity": 0.8}


# =============================================================================
# Step 3: Multi-Model Scenarios
# =============================================================================


@patch("karenina.integrations.gepa.adapter.LLMFeedbackGenerator")
def test_differential_analysis_one_success_one_failure(
    mock_generator_class,
    mock_benchmark,
    mock_verification_config,
    feedback_model_config,
    mock_data_inst,
    answering_model_config,
    objective_config,
):
    """Differential feedback generated when models differ."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    mock_generator = Mock()
    mock_generator.generate_complete_feedback.return_value = "Differential analysis result"
    mock_generator_class.return_value = mock_generator

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=feedback_model_config,
        enable_differential_analysis=True,
    )

    # Create one failed and one passed trajectory for same question
    failed_result = MagicMock()
    failed_result.template.verify_result = False

    passed_result = MagicMock()
    passed_result.template.verify_result = True
    passed_result.template.parsed_llm_response = {"gene": "BCL2"}

    failed_traj = KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-haiku-4-5",
        model_config=answering_model_config,
        optimized_components={},
        verification_result=failed_result,
        score=0.0,
        raw_llm_response="Wrong answer",
    )

    passed_traj = KareninaTrajectory(
        data_inst=mock_data_inst,  # Same question
        model_name="claude-sonnet-4-5",
        model_config=answering_model_config,
        optimized_components={},
        verification_result=passed_result,
        score=1.0,
        raw_llm_response="Correct answer: BCL2",
    )

    mock_eval_batch = MagicMock()
    mock_eval_batch.trajectories = [failed_traj, passed_traj]

    adapter.make_reflective_dataset(
        candidate={},
        eval_batch=mock_eval_batch,
        components_to_update=["answering_system_prompt"],
    )

    # Verify differential feedback was generated
    call_kwargs = mock_generator.generate_complete_feedback.call_args[1]
    assert call_kwargs["successful_trajectories"] is not None
    assert len(call_kwargs["successful_trajectories"]) == 1
    assert call_kwargs["successful_trajectories"][0].model_name == "claude-sonnet-4-5"


@patch("karenina.integrations.gepa.adapter.LLMFeedbackGenerator")
def test_differential_analysis_all_fail(
    mock_generator_class,
    mock_benchmark,
    mock_verification_config,
    feedback_model_config,
    mock_data_inst,
    answering_model_config,
    objective_config,
):
    """Single feedback when all models fail (no differential)."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    mock_generator = Mock()
    mock_generator.generate_complete_feedback.return_value = "Single feedback"
    mock_generator_class.return_value = mock_generator

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=feedback_model_config,
        enable_differential_analysis=True,
    )

    # Create two failed trajectories
    failed_result = MagicMock()
    failed_result.template.verify_result = False

    failed_traj_1 = KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-haiku-4-5",
        model_config=answering_model_config,
        optimized_components={},
        verification_result=failed_result,
        score=0.0,
        raw_llm_response="Wrong 1",
    )

    failed_traj_2 = KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-sonnet-4-5",
        model_config=answering_model_config,
        optimized_components={},
        verification_result=failed_result,
        score=0.0,
        raw_llm_response="Wrong 2",
    )

    mock_eval_batch = MagicMock()
    mock_eval_batch.trajectories = [failed_traj_1, failed_traj_2]

    adapter.make_reflective_dataset(
        candidate={},
        eval_batch=mock_eval_batch,
        components_to_update=["answering_system_prompt"],
    )

    # Verify no successful trajectories (empty list, not differential)
    call_kwargs = mock_generator.generate_complete_feedback.call_args[1]
    # When differential is enabled but no successes, should pass empty list
    # which is falsy, so generate_complete_feedback will use single feedback
    assert call_kwargs["successful_trajectories"] == []


def test_differential_analysis_all_pass(
    mock_benchmark, mock_verification_config, mock_data_inst, answering_model_config, objective_config
):
    """No feedback generated when all pass (no failures)."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=None,
    )

    # Create two passed trajectories
    passed_result = MagicMock()
    passed_result.template.verify_result = True

    passed_traj_1 = KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-haiku-4-5",
        model_config=answering_model_config,
        optimized_components={},
        verification_result=passed_result,
        score=1.0,
        raw_llm_response="Correct 1",
    )

    passed_traj_2 = KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-sonnet-4-5",
        model_config=answering_model_config,
        optimized_components={},
        verification_result=passed_result,
        score=1.0,
        raw_llm_response="Correct 2",
    )

    mock_eval_batch = MagicMock()
    mock_eval_batch.trajectories = [passed_traj_1, passed_traj_2]

    result = adapter.make_reflective_dataset(
        candidate={},
        eval_batch=mock_eval_batch,
        components_to_update=["answering_system_prompt"],
    )

    # No feedback generated since no failures
    assert result["answering_system_prompt"] == []


# =============================================================================
# Step 4: Programmatic Fallback Tests
# =============================================================================


def test_programmatic_feedback_includes_knowledge_distillation(
    mock_benchmark, mock_verification_config, mock_data_inst, answering_model_config, objective_config
):
    """Programmatic fallback includes knowledge distillation from successes."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_verification_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=objective_config,
        feedback_model_config=None,  # No LLM feedback
    )

    failed_result = MagicMock()
    failed_result.template.verify_result = False

    passed_result = MagicMock()
    passed_result.template.verify_result = True

    failed_traj = KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-haiku-4-5",
        model_config=answering_model_config,
        optimized_components={},
        verification_result=failed_result,
        score=0.0,
        raw_llm_response="Wrong answer",
        failed_fields=["gene"],
    )

    passed_traj = KareninaTrajectory(
        data_inst=mock_data_inst,
        model_name="claude-sonnet-4-5",
        model_config=answering_model_config,
        optimized_components={},
        verification_result=passed_result,
        score=1.0,
        raw_llm_response="The gene symbol is BCL2 which is involved in apoptosis.",
    )

    mock_eval_batch = MagicMock()
    mock_eval_batch.trajectories = [failed_traj, passed_traj]

    result = adapter.make_reflective_dataset(
        candidate={},
        eval_batch=mock_eval_batch,
        components_to_update=["answering_system_prompt"],
    )

    # Verify knowledge distillation is included
    feedback = result["answering_system_prompt"][0]["Feedback"]
    assert "claude-sonnet-4-5" in feedback
    assert "succeeded" in feedback
    assert "The gene symbol is BCL2" in feedback[:100] or "Their responses included" in feedback
