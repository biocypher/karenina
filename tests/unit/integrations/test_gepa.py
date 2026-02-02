"""Unit tests for GEPA-Karenina integration module.

Tests cover:
- config.py: Pydantic models for GEPA configuration
- data_types.py: Dataclasses for GEPA data structures
- splitting.py: Benchmark splitting utilities
- scoring.py: Score computation utilities

All tests use mocks and avoid external dependencies.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

# Import config after data_types to avoid forward reference issues
from karenina.integrations.gepa.config import (  # noqa: E402
    FrontierType,
    MetricObjectiveConfig,
    ObjectiveConfig,
    OptimizationConfig,
    OptimizationTarget,
    TraitSelectionMode,
)
from karenina.integrations.gepa.data_types import (
    BenchmarkSplit,
    KareninaDataInst,
    KareninaOutput,
    KareninaTrajectory,
)
from karenina.integrations.gepa.scoring import (
    compute_improvement,
    compute_objective_scores,
    extract_failed_fields,
)

# Import ModelConfig to fix OptimizationConfig forward reference
from karenina.schemas.workflow.models import ModelConfig  # noqa: F401

# Rebuild OptimizationConfig to resolve forward references
OptimizationConfig.model_rebuild()


# =============================================================================
# TraitSelectionMode Tests
# =============================================================================


@pytest.mark.unit
def test_trait_selection_mode_enum_values() -> None:
    """Test TraitSelectionMode enum has expected values."""
    assert TraitSelectionMode.ALL.value == "all"
    assert TraitSelectionMode.NONE.value == "none"
    assert TraitSelectionMode.CUSTOM.value == "custom"


@pytest.mark.unit
def test_trait_selection_mode_comparison() -> None:
    """Test TraitSelectionMode enum comparison."""
    assert TraitSelectionMode.ALL == TraitSelectionMode.ALL
    assert TraitSelectionMode.ALL != TraitSelectionMode.NONE
    assert TraitSelectionMode.CUSTOM == "custom"


# =============================================================================
# MetricObjectiveConfig Tests
# =============================================================================


@pytest.mark.unit
def test_metric_objective_config_defaults() -> None:
    """Test MetricObjectiveConfig default values."""
    config = MetricObjectiveConfig()
    assert config.include_precision is False
    assert config.include_recall is False
    assert config.include_f1 is True


@pytest.mark.unit
def test_metric_objective_config_get_enabled_metrics_default() -> None:
    """Test get_enabled_metrics returns only f1 by default."""
    config = MetricObjectiveConfig()
    assert config.get_enabled_metrics() == ["f1"]


@pytest.mark.unit
def test_metric_objective_config_get_enabled_metrics_all() -> None:
    """Test get_enabled_metrics returns all metrics when enabled."""
    config = MetricObjectiveConfig(
        include_precision=True,
        include_recall=True,
        include_f1=True,
    )
    assert config.get_enabled_metrics() == ["precision", "recall", "f1"]


@pytest.mark.unit
def test_metric_objective_config_get_enabled_metrics_partial() -> None:
    """Test get_enabled_metrics with partial selection."""
    config = MetricObjectiveConfig(
        include_precision=True,
        include_recall=False,
        include_f1=True,
    )
    assert config.get_enabled_metrics() == ["precision", "f1"]


@pytest.mark.unit
def test_metric_objective_config_get_enabled_metrics_empty() -> None:
    """Test get_enabled_metrics returns empty list when none enabled."""
    config = MetricObjectiveConfig(
        include_precision=False,
        include_recall=False,
        include_f1=False,
    )
    assert config.get_enabled_metrics() == []


# =============================================================================
# ObjectiveConfig Tests
# =============================================================================


@pytest.mark.unit
def test_objective_config_defaults() -> None:
    """Test ObjectiveConfig default values."""
    config = ObjectiveConfig()
    assert config.include_template is True
    assert config.trait_mode == TraitSelectionMode.ALL
    assert config.selected_traits is None
    assert config.metric_config.include_f1 is True


@pytest.mark.unit
def test_objective_config_custom_mode_without_traits_raises() -> None:
    """Test CUSTOM mode without selected_traits raises error."""
    with pytest.raises(ValueError, match="selected_traits must be provided"):
        ObjectiveConfig(trait_mode=TraitSelectionMode.CUSTOM)


@pytest.mark.unit
def test_objective_config_custom_mode_with_traits_succeeds() -> None:
    """Test CUSTOM mode with selected_traits succeeds."""
    config = ObjectiveConfig(
        trait_mode=TraitSelectionMode.CUSTOM,
        selected_traits=["clarity", "safety"],
    )
    assert config.selected_traits == ["clarity", "safety"]


@pytest.mark.unit
def test_objective_config_no_objectives_raises() -> None:
    """Test configuration with no objectives raises error."""
    with pytest.raises(ValueError, match="must include at least one objective"):
        ObjectiveConfig(
            include_template=False,
            trait_mode=TraitSelectionMode.NONE,
        )


@pytest.mark.unit
def test_objective_config_should_include_trait_all() -> None:
    """Test should_include_trait with ALL mode."""
    config = ObjectiveConfig(trait_mode=TraitSelectionMode.ALL)
    assert config.should_include_trait("clarity") is True
    assert config.should_include_trait("safety") is True


@pytest.mark.unit
def test_objective_config_should_include_trait_none() -> None:
    """Test should_include_trait with NONE mode."""
    config = ObjectiveConfig(trait_mode=TraitSelectionMode.NONE)
    assert config.should_include_trait("clarity") is False
    assert config.should_include_trait("safety") is False


@pytest.mark.unit
def test_objective_config_should_include_trait_custom() -> None:
    """Test should_include_trait with CUSTOM mode."""
    config = ObjectiveConfig(
        trait_mode=TraitSelectionMode.CUSTOM,
        selected_traits=["clarity", "safety"],
    )
    assert config.should_include_trait("clarity") is True
    assert config.should_include_trait("safety") is True
    assert config.should_include_trait("accuracy") is False


# =============================================================================
# OptimizationTarget Tests
# =============================================================================


@pytest.mark.unit
def test_optimization_target_enum_values() -> None:
    """Test OptimizationTarget enum has expected values."""
    assert OptimizationTarget.ANSWERING_SYSTEM_PROMPT.value == "answering_system_prompt"
    assert OptimizationTarget.PARSING_INSTRUCTIONS.value == "parsing_instructions"
    assert OptimizationTarget.MCP_TOOL_DESCRIPTIONS.value == "mcp_tool_descriptions"


# =============================================================================
# OptimizationConfig Tests
# =============================================================================


@pytest.mark.unit
def test_optimization_config_defaults() -> None:
    """Test OptimizationConfig default values."""
    config = OptimizationConfig(targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT])
    assert config.frontier_type == "objective"
    assert config.reflection_model == "openai/gpt-4o"
    assert config.max_metric_calls == 150
    assert config.candidate_selection_strategy == "pareto"
    assert config.enable_differential_analysis is True
    assert config.train_ratio == 0.8
    assert config.val_ratio == 0.2
    assert config.test_ratio is None
    assert config.split_seed is None


@pytest.mark.unit
def test_optimization_config_split_ratios_sum_to_one() -> None:
    """Test validation with valid split ratios."""
    config = OptimizationConfig(
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        train_ratio=0.7,
        val_ratio=0.3,
    )
    assert config.train_ratio == 0.7
    assert config.val_ratio == 0.3


@pytest.mark.unit
def test_optimization_config_split_ratios_with_test_sum_to_one() -> None:
    """Test validation with valid split ratios including test."""
    config = OptimizationConfig(
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    assert config.train_ratio == 0.6
    assert config.val_ratio == 0.2
    assert config.test_ratio == 0.2


@pytest.mark.unit
def test_optimization_config_invalid_split_ratios_raises() -> None:
    """Test validation with invalid split ratios raises error."""
    with pytest.raises(ValueError, match="must equal 1.0"):
        OptimizationConfig(
            targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
            train_ratio=0.5,
            val_ratio=0.6,
        )


@pytest.mark.unit
def test_optimization_config_invalid_split_ratios_with_test_raises() -> None:
    """Test validation with invalid split ratios including test raises error."""
    with pytest.raises(ValueError, match="must equal 1.0"):
        OptimizationConfig(
            targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.3,
        )


@pytest.mark.unit
def test_optimization_config_max_metric_calls_minimum() -> None:
    """Test max_metric_calls has minimum constraint."""
    with pytest.raises(ValidationError):
        OptimizationConfig(
            targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
            max_metric_calls=0,
        )


@pytest.mark.unit
def test_optimization_config_train_ratio_bounds() -> None:
    """Test train_ratio respects bounds."""
    with pytest.raises(ValidationError):
        OptimizationConfig(
            targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
            train_ratio=1.5,
        )
    with pytest.raises(ValidationError):
        OptimizationConfig(
            targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
            train_ratio=-0.1,
        )


@pytest.mark.unit
def test_optimization_config_auto_seed_answering_prompt() -> None:
    """Test default seed prompt is set for answering target."""
    config = OptimizationConfig(
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        seed_answering_prompt=None,
    )
    assert config.seed_answering_prompt == "You are a helpful assistant."


@pytest.mark.unit
def test_optimization_config_auto_seed_parsing_instructions() -> None:
    """Test default seed instructions are set for parsing target."""
    config = OptimizationConfig(
        targets=[OptimizationTarget.PARSING_INSTRUCTIONS],
        seed_parsing_instructions=None,
    )
    assert config.seed_parsing_instructions == "Extract the answer from the response following the schema."


@pytest.mark.unit
def test_optimization_config_get_seed_candidate_answering() -> None:
    """Test get_seed_candidate for answering target."""
    config = OptimizationConfig(
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        seed_answering_prompt="Be helpful.",
    )
    candidate = config.get_seed_candidate()
    assert candidate == {"answering_system_prompt": "Be helpful."}


@pytest.mark.unit
def test_optimization_config_get_seed_candidate_parsing() -> None:
    """Test get_seed_candidate for parsing target."""
    config = OptimizationConfig(
        targets=[OptimizationTarget.PARSING_INSTRUCTIONS],
        seed_parsing_instructions="Extract data.",
    )
    candidate = config.get_seed_candidate()
    assert candidate == {"parsing_instructions": "Extract data."}


@pytest.mark.unit
def test_optimization_config_get_seed_candidate_mcp_tools() -> None:
    """Test get_seed_candidate for MCP tools target."""
    config = OptimizationConfig(
        targets=[OptimizationTarget.MCP_TOOL_DESCRIPTIONS],
        seed_mcp_tool_descriptions={
            "search": "Search the web",
            "calculate": "Do math",
        },
    )
    candidate = config.get_seed_candidate()
    assert candidate == {
        "mcp_tool_search": "Search the web",
        "mcp_tool_calculate": "Do math",
    }


@pytest.mark.unit
def test_optimization_config_get_seed_candidate_multiple_targets() -> None:
    """Test get_seed_candidate for multiple targets."""
    config = OptimizationConfig(
        targets=[
            OptimizationTarget.ANSWERING_SYSTEM_PROMPT,
            OptimizationTarget.PARSING_INSTRUCTIONS,
        ],
        seed_answering_prompt="Be helpful.",
        seed_parsing_instructions="Extract data.",
    )
    candidate = config.get_seed_candidate()
    assert candidate == {
        "answering_system_prompt": "Be helpful.",
        "parsing_instructions": "Extract data.",
    }


# =============================================================================
# FrontierType Tests
# =============================================================================


@pytest.mark.unit
def test_frontier_type_values() -> None:
    """Test FrontierType literal values."""
    valid_types: list[FrontierType] = ["instance", "objective", "hybrid", "cartesian"]
    for ft in valid_types:
        assert ft in ["instance", "objective", "hybrid", "cartesian"]


# =============================================================================
# KareninaDataInst Tests
# =============================================================================


@pytest.mark.unit
def test_data_inst_construction() -> None:
    """Test KareninaDataInst construction."""
    inst = KareninaDataInst(
        question_id="q-1",
        question_text="What is 2+2?",
        raw_answer="4",
        template_code="class Answer(BaseAnswer): value: str",
    )
    assert inst.question_id == "q-1"
    assert inst.question_text == "What is 2+2?"
    assert inst.raw_answer == "4"
    assert inst.template_code == "class Answer(BaseAnswer): value: str"
    assert inst.rubric is None
    assert inst.few_shot_examples is None
    assert inst.metadata == {}


@pytest.mark.unit
def test_data_inst_with_optional_fields() -> None:
    """Test KareninaDataInst with optional fields."""
    inst = KareninaDataInst(
        question_id="q-1",
        question_text="What is 2+2?",
        raw_answer="4",
        template_code="class Answer(BaseAnswer): value: str",
        rubric={"clarity": {"type": "llm", "max_score": 5}},
        few_shot_examples=[{"question": "1+1", "answer": "2"}],
        metadata={"author": "alice", "tags": ["math"]},
    )
    assert inst.rubric == {"clarity": {"type": "llm", "max_score": 5}}
    assert inst.few_shot_examples == [{"question": "1+1", "answer": "2"}]
    assert inst.metadata == {"author": "alice", "tags": ["math"]}


@pytest.mark.unit
def test_data_inst_to_dict() -> None:
    """Test KareninaDataInst to_dict method."""
    inst = KareninaDataInst(
        question_id="q-1",
        question_text="What is 2+2?",
        raw_answer="4",
        template_code="class Answer(BaseAnswer): value: str",
        rubric={"clarity": {"type": "llm"}},
        few_shot_examples=[{"q": "1+1"}],
        metadata={"author": "alice"},
    )
    result = inst.to_dict()
    assert result == {
        "question_id": "q-1",
        "question_text": "What is 2+2?",
        "raw_answer": "4",
        "template_code": "class Answer(BaseAnswer): value: str",
        "rubric": {"clarity": {"type": "llm"}},
        "few_shot_examples": [{"q": "1+1"}],
        "metadata": {"author": "alice"},
    }


@pytest.mark.unit
def test_data_inst_metadata_default_factory() -> None:
    """Test metadata default_factory creates independent dicts."""
    inst1 = KareninaDataInst(
        question_id="q-1",
        question_text="Q1",
        raw_answer="A1",
        template_code="code1",
    )
    inst2 = KareninaDataInst(
        question_id="q-2",
        question_text="Q2",
        raw_answer="A2",
        template_code="code2",
    )
    inst1.metadata["author"] = "alice"
    inst2.metadata["author"] = "bob"
    assert inst1.metadata["author"] == "alice"
    assert inst2.metadata["author"] == "bob"


# =============================================================================
# KareninaTrajectory Tests
# =============================================================================


@pytest.mark.unit
def test_trajectory_construction() -> None:
    """Test KareninaTrajectory construction."""
    data_inst = KareninaDataInst(
        question_id="q-1",
        question_text="What is 2+2?",
        raw_answer="4",
        template_code="code",
    )
    model_config = MagicMock()

    trajectory = KareninaTrajectory(
        data_inst=data_inst,
        model_name="gpt-4",
        model_config=model_config,
        optimized_components={"answering_system_prompt": "Be helpful."},
        verification_result=MagicMock(template=MagicMock(verify_result=True)),
    )
    assert trajectory.model_name == "gpt-4"
    assert trajectory.raw_llm_response is None
    assert trajectory.parsing_error is None
    assert trajectory.failed_fields is None
    assert trajectory.rubric_scores is None


@pytest.mark.unit
def test_trajectory_passed_true() -> None:
    """Test trajectory passed() returns True when verification passed."""
    data_inst = KareninaDataInst(
        question_id="q-1",
        question_text="Q",
        raw_answer="A",
        template_code="code",
    )
    trajectory = KareninaTrajectory(
        data_inst=data_inst,
        model_name="gpt-4",
        model_config=MagicMock(),
        optimized_components={},
        verification_result=MagicMock(template=MagicMock(verify_result=True)),
    )
    assert trajectory.passed() is True


@pytest.mark.unit
def test_trajectory_passed_false() -> None:
    """Test trajectory passed() returns False when verification failed."""
    data_inst = KareninaDataInst(
        question_id="q-1",
        question_text="Q",
        raw_answer="A",
        template_code="code",
    )
    trajectory = KareninaTrajectory(
        data_inst=data_inst,
        model_name="gpt-4",
        model_config=MagicMock(),
        optimized_components={},
        verification_result=MagicMock(template=MagicMock(verify_result=False)),
    )
    assert trajectory.passed() is False


@pytest.mark.unit
def test_trajectory_passed_no_template() -> None:
    """Test trajectory passed() returns False when no template."""
    data_inst = KareninaDataInst(
        question_id="q-1",
        question_text="Q",
        raw_answer="A",
        template_code="code",
    )
    trajectory = KareninaTrajectory(
        data_inst=data_inst,
        model_name="gpt-4",
        model_config=MagicMock(),
        optimized_components={},
        verification_result=MagicMock(template=None),
    )
    assert trajectory.passed() is False


@pytest.mark.unit
def test_trajectory_to_feedback_dict() -> None:
    """Test trajectory to_feedback_dict method."""
    data_inst = KareninaDataInst(
        question_id="q-1",
        question_text="What is 2+2?",
        raw_answer="4",
        template_code="code",
    )
    trajectory = KareninaTrajectory(
        data_inst=data_inst,
        model_name="gpt-4",
        model_config=MagicMock(),
        optimized_components={},
        verification_result=MagicMock(template=None),
        raw_llm_response="The answer is 4.",
        parsing_error="Invalid schema",
        failed_fields=["value"],
    )
    result = trajectory.to_feedback_dict()
    assert result["Inputs"] == {"question": "What is 2+2?", "model": "gpt-4"}
    assert result["Generated Outputs"] == "The answer is 4."
    assert "Parsing error: Invalid schema" in result["Feedback"]
    assert "Failed fields: value" in result["Feedback"]
    assert "Expected answer: 4" in result["Feedback"]


@pytest.mark.unit
def test_trajectory_to_feedback_dict_minimal() -> None:
    """Test trajectory to_feedback_dict with minimal data."""
    data_inst = KareninaDataInst(
        question_id="q-1",
        question_text="Q",
        raw_answer="A",
        template_code="code",
    )
    trajectory = KareninaTrajectory(
        data_inst=data_inst,
        model_name="gpt-4",
        model_config=MagicMock(),
        optimized_components={},
        verification_result=MagicMock(template=None),
    )
    result = trajectory.to_feedback_dict()
    assert result["Inputs"] == {"question": "Q", "model": "gpt-4"}
    assert result["Generated Outputs"] == "(no response)"
    assert "Expected answer: A" in result["Feedback"]


# =============================================================================
# KareninaOutput Tests
# =============================================================================


@pytest.mark.unit
def test_output_construction_defaults() -> None:
    """Test KareninaOutput construction with defaults."""
    output = KareninaOutput()
    assert output.answering_system_prompt is None
    assert output.parsing_instructions is None
    assert output.mcp_tool_descriptions is None
    assert output.train_score == 0.0
    assert output.val_score == 0.0
    assert output.test_score is None
    assert output.baseline_score == 0.0
    assert output.improvement == 0.0
    assert output.total_generations == 0
    assert output.total_metric_calls == 0
    assert output.best_generation == 0


@pytest.mark.unit
def test_output_with_all_fields() -> None:
    """Test KareninaOutput with all fields."""
    output = KareninaOutput(
        answering_system_prompt="Optimized prompt",
        parsing_instructions="Optimized instructions",
        mcp_tool_descriptions={"search": "Find info"},
        train_score=0.8,
        val_score=0.75,
        test_score=0.72,
        baseline_score=0.6,
        improvement=0.25,
        total_generations=10,
        total_metric_calls=150,
        best_generation=7,
    )
    assert output.answering_system_prompt == "Optimized prompt"
    assert output.val_score == 0.75
    assert output.test_score == 0.72
    assert output.improvement == 0.25
    assert output.best_generation == 7


@pytest.mark.unit
def test_output_get_optimized_prompts_answering() -> None:
    """Test get_optimized_prompts for answering prompt."""
    output = KareninaOutput(
        answering_system_prompt="Be helpful.",
    )
    result = output.get_optimized_prompts()
    assert result == {"answering_system_prompt": "Be helpful."}


@pytest.mark.unit
def test_output_get_optimized_prompts_parsing() -> None:
    """Test get_optimized_prompts for parsing instructions."""
    output = KareninaOutput(
        parsing_instructions="Extract data.",
    )
    result = output.get_optimized_prompts()
    assert result == {"parsing_instructions": "Extract data."}


@pytest.mark.unit
def test_output_get_optimized_prompts_mcp_tools() -> None:
    """Test get_optimized_prompts for MCP tools."""
    output = KareninaOutput(
        mcp_tool_descriptions={"search": "Find info", "calc": "Math"},
    )
    result = output.get_optimized_prompts()
    assert result == {
        "mcp_tool_search": "Find info",
        "mcp_tool_calc": "Math",
    }


@pytest.mark.unit
def test_output_get_optimized_prompts_all() -> None:
    """Test get_optimized_prompts with all components."""
    output = KareninaOutput(
        answering_system_prompt="Be helpful.",
        parsing_instructions="Extract data.",
        mcp_tool_descriptions={"search": "Find info"},
    )
    result = output.get_optimized_prompts()
    assert result == {
        "answering_system_prompt": "Be helpful.",
        "parsing_instructions": "Extract data.",
        "mcp_tool_search": "Find info",
    }


@pytest.mark.unit
def test_output_get_optimized_prompts_empty() -> None:
    """Test get_optimized_prompts when nothing is set."""
    output = KareninaOutput()
    result = output.get_optimized_prompts()
    assert result == {}


# =============================================================================
# BenchmarkSplit Tests
# =============================================================================


@pytest.mark.unit
def test_benchmark_split_construction() -> None:
    """Test BenchmarkSplit construction."""
    train = [
        KareninaDataInst(
            question_id="q-1",
            question_text="Q1",
            raw_answer="A1",
            template_code="code1",
        )
    ]
    val = [
        KareninaDataInst(
            question_id="q-2",
            question_text="Q2",
            raw_answer="A2",
            template_code="code2",
        )
    ]
    split = BenchmarkSplit(train=train, val=val)
    assert len(split.train) == 1
    assert len(split.val) == 1
    assert split.test is None
    assert split.seed is None


@pytest.mark.unit
def test_benchmark_split_with_test_set() -> None:
    """Test BenchmarkSplit with test set."""
    train = [
        KareninaDataInst(
            question_id="q-1",
            question_text="Q1",
            raw_answer="A1",
            template_code="code1",
        )
    ]
    val = [
        KareninaDataInst(
            question_id="q-2",
            question_text="Q2",
            raw_answer="A2",
            template_code="code2",
        )
    ]
    test = [
        KareninaDataInst(
            question_id="q-3",
            question_text="Q3",
            raw_answer="A3",
            template_code="code3",
        )
    ]
    split = BenchmarkSplit(train=train, val=val, test=test, seed=42)
    assert split.test is not None
    assert len(split.test) == 1
    assert split.seed == 42


@pytest.mark.unit
def test_benchmark_split_empty_train_raises() -> None:
    """Test BenchmarkSplit raises on empty train set."""
    val = [
        KareninaDataInst(
            question_id="q-1",
            question_text="Q",
            raw_answer="A",
            template_code="code",
        )
    ]
    with pytest.raises(ValueError, match="Training set cannot be empty"):
        BenchmarkSplit(train=[], val=val)


@pytest.mark.unit
def test_benchmark_split_empty_val_raises() -> None:
    """Test BenchmarkSplit raises on empty val set."""
    train = [
        KareninaDataInst(
            question_id="q-1",
            question_text="Q",
            raw_answer="A",
            template_code="code",
        )
    ]
    with pytest.raises(ValueError, match="Validation set cannot be empty"):
        BenchmarkSplit(train=train, val=[])


@pytest.mark.unit
def test_benchmark_split_train_ids() -> None:
    """Test train_ids property."""
    train = [
        KareninaDataInst(
            question_id="q-1",
            question_text="Q1",
            raw_answer="A1",
            template_code="code1",
        ),
        KareninaDataInst(
            question_id="q-2",
            question_text="Q2",
            raw_answer="A2",
            template_code="code2",
        ),
    ]
    val = [
        KareninaDataInst(
            question_id="q-3",
            question_text="Q3",
            raw_answer="A3",
            template_code="code3",
        )
    ]
    split = BenchmarkSplit(train=train, val=val)
    assert split.train_ids == ["q-1", "q-2"]
    assert split.val_ids == ["q-3"]


@pytest.mark.unit
def test_benchmark_split_test_ids() -> None:
    """Test test_ids property."""
    train = [
        KareninaDataInst(
            question_id="q-1",
            question_text="Q1",
            raw_answer="A1",
            template_code="code1",
        )
    ]
    val = [
        KareninaDataInst(
            question_id="q-2",
            question_text="Q2",
            raw_answer="A2",
            template_code="code2",
        )
    ]
    test = [
        KareninaDataInst(
            question_id="q-3",
            question_text="Q3",
            raw_answer="A3",
            template_code="code3",
        )
    ]
    split = BenchmarkSplit(train=train, val=val, test=test)
    assert split.test_ids == ["q-3"]


@pytest.mark.unit
def test_benchmark_split_test_ids_when_no_test() -> None:
    """Test test_ids returns None when no test set."""
    train = [
        KareninaDataInst(
            question_id="q-1",
            question_text="Q1",
            raw_answer="A1",
            template_code="code1",
        )
    ]
    val = [
        KareninaDataInst(
            question_id="q-2",
            question_text="Q2",
            raw_answer="A2",
            template_code="code2",
        )
    ]
    split = BenchmarkSplit(train=train, val=val)
    assert split.test_ids is None


@pytest.mark.unit
def test_benchmark_split_summary() -> None:
    """Test summary method."""
    train = [
        KareninaDataInst(
            question_id="q-1",
            question_text="Q1",
            raw_answer="A1",
            template_code="code1",
        ),
        KareninaDataInst(
            question_id="q-2",
            question_text="Q2",
            raw_answer="A2",
            template_code="code2",
        ),
    ]
    val = [
        KareninaDataInst(
            question_id="q-3",
            question_text="Q3",
            raw_answer="A3",
            template_code="code3",
        )
    ]
    split = BenchmarkSplit(train=train, val=val, seed=42)
    summary = split.summary()
    assert "Train: 2 questions" in summary
    assert "Val: 1 questions" in summary
    assert "Seed: 42" in summary


@pytest.mark.unit
def test_benchmark_split_summary_with_test() -> None:
    """Test summary method with test set."""
    train = [
        KareninaDataInst(
            question_id="q-1",
            question_text="Q1",
            raw_answer="A1",
            template_code="code1",
        )
    ]
    val = [
        KareninaDataInst(
            question_id="q-2",
            question_text="Q2",
            raw_answer="A2",
            template_code="code2",
        )
    ]
    test = [
        KareninaDataInst(
            question_id="q-3",
            question_text="Q3",
            raw_answer="A3",
            template_code="code3",
        )
    ]
    split = BenchmarkSplit(train=train, val=val, test=test)
    summary = split.summary()
    assert "Train: 1 questions" in summary
    assert "Val: 1 questions" in summary
    assert "Test: 1 questions" in summary


# =============================================================================
# compute_objective_scores Tests
# =============================================================================


@pytest.mark.unit
def test_compute_objective_scores_template_only() -> None:
    """Test compute_objective_scores with template only."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.rubric = None

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.NONE,
    )

    scores = compute_objective_scores(result, "gpt-4", config)
    assert scores == {"gpt-4:template": 1.0}


@pytest.mark.unit
def test_compute_objective_scores_template_false() -> None:
    """Test compute_objective_scores with template False."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = False
    result.rubric = None

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.NONE,
    )

    scores = compute_objective_scores(result, "gpt-4", config)
    assert scores == {"gpt-4:template": 0.0}


@pytest.mark.unit
def test_compute_objective_scores_no_template() -> None:
    """Test compute_objective_scores with no template."""
    result = MagicMock()
    result.template = None
    result.rubric = None

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.NONE,
    )

    scores = compute_objective_scores(result, "gpt-4", config)
    assert scores == {"gpt-4:template": 0.0}


@pytest.mark.unit
def test_compute_objective_scores_with_bool_traits() -> None:
    """Test compute_objective_scores with boolean rubric traits."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.rubric = MagicMock()
    result.rubric.rubric_evaluation_performed = True
    result.rubric.get_all_trait_scores.return_value = {
        "has_citation": True,
        "is_concise": False,
    }

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.ALL,
    )

    scores = compute_objective_scores(result, "gpt-4", config)
    assert scores["gpt-4:template"] == 1.0
    assert scores["gpt-4:has_citation"] == 1.0
    assert scores["gpt-4:is_concise"] == 0.0


@pytest.mark.unit
def test_compute_objective_scores_with_int_traits() -> None:
    """Test compute_objective_scores with integer rubric traits."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.rubric = MagicMock()
    result.rubric.rubric_evaluation_performed = True
    result.rubric.get_all_trait_scores.return_value = {
        "clarity": 4,
        "accuracy": 3,
    }

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.ALL,
    )

    scores = compute_objective_scores(result, "gpt-4", config)
    assert scores["gpt-4:template"] == 1.0
    assert scores["gpt-4:clarity"] == 0.8  # 4/5
    assert scores["gpt-4:accuracy"] == 0.6  # 3/5


@pytest.mark.unit
def test_compute_objective_scores_with_custom_max_scores() -> None:
    """Test compute_objective_scores with custom max scores."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.rubric = MagicMock()
    result.rubric.rubric_evaluation_performed = True
    result.rubric.get_all_trait_scores.return_value = {
        "clarity": 8,
        "accuracy": 15,
    }

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.ALL,
    )

    trait_max_scores = {"clarity": 10, "accuracy": 20}
    scores = compute_objective_scores(result, "gpt-4", config, trait_max_scores=trait_max_scores)
    assert scores["gpt-4:clarity"] == 0.8  # 8/10
    assert scores["gpt-4:accuracy"] == 0.75  # 15/20


@pytest.mark.unit
def test_compute_objective_scores_with_inverted_traits() -> None:
    """Test compute_objective_scores with lower-is-better traits."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.rubric = MagicMock()
    result.rubric.rubric_evaluation_performed = True
    result.rubric.get_all_trait_scores.return_value = {
        "error_rate": 2,
    }

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.ALL,
    )

    # error_rate: lower is better
    trait_directionalities = {"error_rate": False}
    scores = compute_objective_scores(
        result,
        "gpt-4",
        config,
        trait_max_scores={"error_rate": 10},
        trait_directionalities=trait_directionalities,
    )
    # Score of 2/10 = 0.2, inverted to 0.8 (lower is better)
    assert scores["gpt-4:error_rate"] == 0.8


@pytest.mark.unit
def test_compute_objective_scores_inverted_bool_trait() -> None:
    """Test compute_objective_scores with inverted boolean trait."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.rubric = MagicMock()
    result.rubric.rubric_evaluation_performed = True
    result.rubric.get_all_trait_scores.return_value = {
        "has_errors": True,
    }

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.ALL,
    )

    # has_errors: lower is better (True should score low)
    trait_directionalities = {"has_errors": False}
    scores = compute_objective_scores(
        result,
        "gpt-4",
        config,
        trait_directionalities=trait_directionalities,
    )
    # True becomes 1.0, inverted to 0.0
    assert scores["gpt-4:has_errors"] == 0.0


@pytest.mark.unit
def test_compute_objective_scores_with_metric_traits() -> None:
    """Test compute_objective_scores with metric traits."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.rubric = MagicMock()
    result.rubric.rubric_evaluation_performed = True
    result.rubric.get_all_trait_scores.return_value = {
        "recall": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
    }

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.ALL,
        metric_config=MetricObjectiveConfig(
            include_precision=True,
            include_recall=True,
            include_f1=True,
        ),
    )

    scores = compute_objective_scores(result, "gpt-4", config)
    assert "gpt-4:recall_precision" in scores
    assert "gpt-4:recall_recall" in scores
    assert "gpt-4:recall_f1" in scores
    assert scores["gpt-4:recall_precision"] == 0.9
    assert scores["gpt-4:recall_recall"] == 0.8
    assert scores["gpt-4:recall_f1"] == 0.85


@pytest.mark.unit
def test_compute_objective_scores_metric_partial() -> None:
    """Test compute_objective_scores with partial metric selection."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.rubric = MagicMock()
    result.rubric.rubric_evaluation_performed = True
    result.rubric.get_all_trait_scores.return_value = {
        "recall": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
    }

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.ALL,
        metric_config=MetricObjectiveConfig(
            include_precision=False,
            include_recall=False,
            include_f1=True,
        ),
    )

    scores = compute_objective_scores(result, "gpt-4", config)
    # Only F1 should be included
    assert "gpt-4:recall_f1" in scores
    assert "gpt-4:recall_precision" not in scores
    assert "gpt-4:recall_recall" not in scores


@pytest.mark.unit
def test_compute_objective_scores_custom_trait_selection() -> None:
    """Test compute_objective_scores with CUSTOM trait selection."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.rubric = MagicMock()
    result.rubric.rubric_evaluation_performed = True
    result.rubric.get_all_trait_scores.return_value = {
        "clarity": 4,
        "safety": 5,
        "accuracy": 3,
    }

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.CUSTOM,
        selected_traits=["clarity", "safety"],
    )

    scores = compute_objective_scores(result, "gpt-4", config)
    assert "gpt-4:template" in scores
    assert "gpt-4:clarity" in scores
    assert "gpt-4:safety" in scores
    assert "gpt-4:accuracy" not in scores


@pytest.mark.unit
def test_compute_objective_scores_no_rubric() -> None:
    """Test compute_objective_scores when rubric is None."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.rubric = None

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.ALL,
    )

    scores = compute_objective_scores(result, "gpt-4", config)
    assert scores == {"gpt-4:template": 1.0}


@pytest.mark.unit
def test_compute_objective_scores_rubric_not_performed() -> None:
    """Test compute_objective_scores when rubric evaluation not performed."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.verify_result = True
    result.rubric = MagicMock()
    result.rubric.rubric_evaluation_performed = False

    config = ObjectiveConfig(
        include_template=True,
        trait_mode=TraitSelectionMode.ALL,
    )

    scores = compute_objective_scores(result, "gpt-4", config)
    assert scores == {"gpt-4:template": 1.0}


# =============================================================================
# extract_failed_fields Tests
# =============================================================================


@pytest.mark.unit
def test_extract_failed_fields_no_template() -> None:
    """Test extract_failed_fields when template is None."""
    result = MagicMock()
    result.template = None
    failed = extract_failed_fields(result)
    assert failed == []


@pytest.mark.unit
def test_extract_failed_fields_no_field_results() -> None:
    """Test extract_failed_fields when field_results not present."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.field_results = None
    failed = extract_failed_fields(result)
    assert failed == []


@pytest.mark.unit
def test_extract_failed_fields_bool_failures() -> None:
    """Test extract_failed_fields with boolean field results."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.field_results = {
        "value": True,
        "confidence": False,
        "reasoning": True,
    }
    failed = extract_failed_fields(result)
    assert failed == ["confidence"]


@pytest.mark.unit
def test_extract_failed_fields_dict_failures() -> None:
    """Test extract_failed_fields with dict field results."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.field_results = {
        "value": {"passed": True},
        "confidence": {"passed": False},
        "reasoning": {"passed": True},
    }
    failed = extract_failed_fields(result)
    assert failed == ["confidence"]


@pytest.mark.unit
def test_extract_failed_fields_mixed() -> None:
    """Test extract_failed_fields with mixed field results."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.field_results = {
        "value": True,
        "confidence": {"passed": False},
        "reasoning": False,
        "format": {"passed": True},
    }
    failed = extract_failed_fields(result)
    assert set(failed) == {"confidence", "reasoning"}


@pytest.mark.unit
def test_extract_failed_fields_empty() -> None:
    """Test extract_failed_fields with no failures."""
    result = MagicMock()
    result.template = MagicMock()
    result.template.field_results = {
        "value": True,
        "confidence": {"passed": True},
    }
    failed = extract_failed_fields(result)
    assert failed == []


# =============================================================================
# compute_improvement Tests
# =============================================================================


@pytest.mark.unit
def test_compute_improvement_normal() -> None:
    """Test compute_improvement with normal values."""
    improvement = compute_improvement(0.6, 0.75)
    assert improvement == pytest.approx(0.25)


@pytest.mark.unit
def test_compute_improvement_zero_baseline() -> None:
    """Test compute_improvement when baseline is zero."""
    improvement = compute_improvement(0.0, 0.5)
    assert improvement == 0.5


@pytest.mark.unit
def test_compute_improvement_negative() -> None:
    """Test compute_improvement when optimized is worse."""
    improvement = compute_improvement(0.8, 0.6)
    assert improvement == pytest.approx(-0.25)


@pytest.mark.unit
def test_compute_improvement_equal() -> None:
    """Test compute_improvement when values are equal."""
    improvement = compute_improvement(0.7, 0.7)
    assert improvement == 0.0
