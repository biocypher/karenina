"""GEPA Integration Tests.

These tests verify the GEPA integration for karenina benchmark optimization.
Run with: uv run pytest tests/integrations/gepa/test_gepa_integration.py -v

Note: E2E tests (Steps 8-11) require API keys and make real API calls.
Mark them with @pytest.mark.slow or @pytest.mark.e2e for selective running.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# =============================================================================
# Step 1: Verify GEPA Dependency Availability
# =============================================================================


def test_gepa_available():
    """Step 1: Verify GEPA is installed and available."""
    from karenina.integrations.gepa import GEPA_AVAILABLE

    assert GEPA_AVAILABLE, "GEPA not installed. Run: pip install karenina[gepa]"


# =============================================================================
# Step 2: Unit Test - Scoring Functions
# =============================================================================


def test_compute_single_score_template_only():
    """Test scoring with template pass (no rubric)."""
    from karenina.integrations.gepa import compute_single_score

    # Mock a passing VerificationResult
    mock_result = MagicMock()
    mock_result.template = MagicMock()
    mock_result.template.verify_result = True
    mock_result.rubric = None

    score = compute_single_score(mock_result, template_weight=0.7, rubric_weight=0.3)
    assert score == 1.0, f"Expected 1.0 for passing template, got {score}"


def test_compute_single_score_template_fail():
    """Test scoring with template fail."""
    from karenina.integrations.gepa import compute_single_score

    mock_result = MagicMock()
    mock_result.template = MagicMock()
    mock_result.template.verify_result = False
    mock_result.rubric = None

    score = compute_single_score(mock_result, template_weight=0.7, rubric_weight=0.3)
    assert score == 0.0, f"Expected 0.0 for failing template, got {score}"


def test_compute_improvement_positive():
    """Test improvement calculation with positive improvement."""
    from karenina.integrations.gepa import compute_improvement

    improvement = compute_improvement(baseline=0.5, optimized=0.6)
    expected = (0.6 - 0.5) / 0.5  # 0.2 = 20%
    assert abs(improvement - expected) < 1e-6, f"Expected {expected}, got {improvement}"


def test_compute_improvement_zero_baseline():
    """Test improvement calculation with zero baseline."""
    from karenina.integrations.gepa import compute_improvement

    improvement = compute_improvement(baseline=0.0, optimized=0.5)
    assert improvement == 0.5, f"Expected 0.5 for zero baseline, got {improvement}"


# =============================================================================
# Step 3: Unit Test - Benchmark Splitting
# =============================================================================


@pytest.fixture
def benchmark_path():
    """Path to test benchmark."""
    return Path("/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld")


@pytest.mark.skipif(
    not Path("/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld").exists(),
    reason="Test benchmark not found",
)
def test_split_benchmark_default(benchmark_path):
    """Test default 80/20 split."""
    from karenina.benchmark import Benchmark
    from karenina.integrations.gepa import split_benchmark

    benchmark = Benchmark.load(benchmark_path)
    split = split_benchmark(benchmark, seed=42)

    total = len(split.train) + len(split.val)
    assert total == 129, f"Expected 129 total, got {total}"
    assert len(split.train) >= 100, f"Expected ~103 train, got {len(split.train)}"
    assert len(split.val) >= 24, f"Expected ~26 val, got {len(split.val)}"
    assert split.test is None, "Test set should be None for default split"


@pytest.mark.skipif(
    not Path("/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld").exists(),
    reason="Test benchmark not found",
)
def test_split_benchmark_reproducibility(benchmark_path):
    """Test that split is reproducible with seed."""
    from karenina.benchmark import Benchmark
    from karenina.integrations.gepa import split_benchmark

    benchmark = Benchmark.load(benchmark_path)
    split1 = split_benchmark(benchmark, seed=42)
    split2 = split_benchmark(benchmark, seed=42)

    ids1 = [d.question_id for d in split1.train]
    ids2 = [d.question_id for d in split2.train]
    assert ids1 == ids2, "Splits with same seed should be identical"


@pytest.mark.skipif(
    not Path("/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld").exists(),
    reason="Test benchmark not found",
)
def test_split_benchmark_with_test_set(benchmark_path):
    """Test 70/15/15 split with test set."""
    from karenina.benchmark import Benchmark
    from karenina.integrations.gepa import split_benchmark

    benchmark = Benchmark.load(benchmark_path)
    split = split_benchmark(benchmark, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert split.test is not None, "Test set should not be None"
    total = len(split.train) + len(split.val) + len(split.test)
    assert total == 129, f"Expected 129 total, got {total}"


@pytest.mark.skipif(
    not Path("/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld").exists(),
    reason="Test benchmark not found",
)
def test_data_inst_fields_populated(benchmark_path):
    """Test that KareninaDataInst fields are properly populated."""
    from karenina.benchmark import Benchmark
    from karenina.integrations.gepa import split_benchmark

    benchmark = Benchmark.load(benchmark_path)
    split = split_benchmark(benchmark, seed=42)
    inst = split.train[0]

    assert inst.question_id, "question_id should be populated"
    assert inst.question_text, "question_text should be populated"
    assert inst.raw_answer is not None, "raw_answer should be populated"
    assert inst.template_code, "template_code should be populated"


# =============================================================================
# Step 4: Unit Test - Data Types
# =============================================================================


def test_karenina_data_inst_serialization():
    """Test KareninaDataInst serialization."""
    from karenina.integrations.gepa import KareninaDataInst

    inst = KareninaDataInst(
        question_id="test-123",
        question_text="What is 2+2?",
        raw_answer="4",
        template_code="class Answer(BaseAnswer): ...",
    )
    d = inst.to_dict()

    assert d["question_id"] == "test-123"
    assert "question_text" in d


def test_benchmark_split_validation():
    """Test BenchmarkSplit validation."""
    from karenina.integrations.gepa import BenchmarkSplit, KareninaDataInst

    inst = KareninaDataInst(
        question_id="test-123",
        question_text="What is 2+2?",
        raw_answer="4",
        template_code="class Answer(BaseAnswer): ...",
    )

    # Empty train should raise
    with pytest.raises(ValueError):
        BenchmarkSplit(train=[], val=[inst])


def test_benchmark_split_summary():
    """Test BenchmarkSplit summary method."""
    from karenina.integrations.gepa import BenchmarkSplit, KareninaDataInst

    inst = KareninaDataInst(
        question_id="test-123",
        question_text="What is 2+2?",
        raw_answer="4",
        template_code="class Answer(BaseAnswer): ...",
    )
    split = BenchmarkSplit(train=[inst], val=[inst])
    summary = split.summary()

    assert "train" in summary.lower() or "1" in summary


def test_karenina_output_get_optimized_prompts():
    """Test KareninaOutput.get_optimized_prompts method."""
    from karenina.integrations.gepa import KareninaOutput

    output = KareninaOutput(
        answering_system_prompt="You are helpful",
        train_score=0.85,
        val_score=0.82,
        improvement=0.15,
    )
    prompts = output.get_optimized_prompts()

    assert "answering_system_prompt" in prompts


# =============================================================================
# Step 5: Unit Test - SQLite Tracking
# =============================================================================


def test_tracker_log_and_retrieve_run():
    """Test logging and retrieving optimization runs."""
    from karenina.integrations.gepa import OptimizationRun, OptimizationTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tracking.db"
        tracker = OptimizationTracker(db_path)

        run = OptimizationRun(
            benchmark_name="test_benchmark",
            targets=["answering_system_prompt"],
            seed_prompts={"answering_system_prompt": "You are helpful"},
            optimized_prompts={"answering_system_prompt": "You are an expert..."},
            train_score=0.85,
            val_score=0.82,
            improvement=0.15,
            reflection_model="gpt-4o",
            metric_calls=50,
            best_generation=3,
            total_generations=5,
        )
        run_id = tracker.log_run(run)

        retrieved = tracker.get_run(run_id)
        assert retrieved.benchmark_name == "test_benchmark"
        assert retrieved.val_score == 0.82


def test_tracker_list_runs():
    """Test listing optimization runs."""
    from karenina.integrations.gepa import OptimizationRun, OptimizationTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tracking.db"
        tracker = OptimizationTracker(db_path)

        run = OptimizationRun(
            benchmark_name="test_benchmark",
            targets=["answering_system_prompt"],
            seed_prompts={},
            optimized_prompts={},
            train_score=0.85,
            val_score=0.82,
            improvement=0.15,
        )
        tracker.log_run(run)

        runs = tracker.list_runs(benchmark_name="test_benchmark")
        assert len(runs) == 1


def test_tracker_export_json():
    """Test exporting history as JSON."""
    from karenina.integrations.gepa import OptimizationRun, OptimizationTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tracking.db"
        tracker = OptimizationTracker(db_path)

        run = OptimizationRun(
            benchmark_name="test_benchmark",
            targets=["answering_system_prompt"],
            seed_prompts={},
            optimized_prompts={},
            train_score=0.85,
            val_score=0.82,
            improvement=0.15,
        )
        tracker.log_run(run)

        history_json = tracker.export_history(format="json")
        assert len(history_json) > 0


def test_tracker_export_csv():
    """Test exporting history as CSV."""
    from karenina.integrations.gepa import OptimizationRun, OptimizationTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tracking.db"
        tracker = OptimizationTracker(db_path)

        run = OptimizationRun(
            benchmark_name="test_benchmark",
            targets=["answering_system_prompt"],
            seed_prompts={},
            optimized_prompts={},
            train_score=0.85,
            val_score=0.82,
            improvement=0.15,
        )
        tracker.log_run(run)

        history_csv = tracker.export_history(format="csv")
        assert "benchmark_name" in history_csv.split("\n")[0]


# =============================================================================
# Step 6: Unit Test - Export Functions
# =============================================================================


def test_export_and_load_prompts_json():
    """Test exporting and loading prompts JSON."""
    from karenina.integrations.gepa import export_prompts_json, load_prompts_json

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "prompts.json"

        optimized = {"answering_system_prompt": "You are an expert bioinformatician..."}
        metadata = {"benchmark": "test", "improvement": 0.15}
        export_prompts_json(optimized, metadata, output_path)

        loaded_prompts, loaded_meta = load_prompts_json(output_path)
        assert loaded_prompts["answering_system_prompt"] == optimized["answering_system_prompt"]
        assert loaded_meta["improvement"] == 0.15


# =============================================================================
# Step 7: Integration Test - KareninaAdapter (mocked)
# =============================================================================


def test_adapter_inject_candidate():
    """Test that _inject_candidate injects system prompt correctly."""
    from karenina.integrations.gepa import GEPA_AVAILABLE

    if not GEPA_AVAILABLE:
        pytest.skip("GEPA not available")

    from karenina.integrations.gepa import KareninaAdapter, OptimizationTarget

    # Setup mock benchmark and config
    mock_benchmark = MagicMock()
    mock_config = MagicMock()
    mock_model = MagicMock()
    mock_model.model_name = "claude-haiku-4-5"
    mock_model.system_prompt = None
    mock_config.answering_models = [mock_model]
    mock_config.model_copy = MagicMock(return_value=mock_config)

    adapter = KareninaAdapter(
        benchmark=mock_benchmark,
        base_config=mock_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
    )

    candidate = {"answering_system_prompt": "You are an expert bioinformatician."}
    injected_config = adapter._inject_candidate(candidate)

    assert injected_config.answering_models[0].system_prompt == "You are an expert bioinformatician."


# =============================================================================
# Step 8-11: E2E Tests (require API keys)
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld").exists(),
    reason="Test benchmark not found",
)
def test_e2e_single_verification_run(benchmark_path):
    """Step 8: E2E test - single verification run with real API."""
    from karenina.benchmark import Benchmark
    from karenina.integrations.gepa import KareninaAdapter, OptimizationTarget, split_benchmark
    from karenina.schemas.workflow.models import ModelConfig
    from karenina.schemas.workflow.verification.config import VerificationConfig

    benchmark = Benchmark.load(benchmark_path)

    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="answer-haiku",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                system_prompt="You are a helpful assistant.",
            )
        ],
        parsing_models=[
            ModelConfig(
                id="parser-haiku",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
            )
        ],
        evaluation_mode="template_only",
    )

    split = split_benchmark(benchmark, seed=42)
    test_batch = split.train[:3]

    adapter = KareninaAdapter(
        benchmark=benchmark,
        base_config=config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
    )

    candidate = {"answering_system_prompt": "You are a helpful bioinformatics assistant."}
    result = adapter.evaluate(test_batch, candidate, capture_traces=True)

    assert len(result.scores) == 3
    assert result.trajectories is not None


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld").exists(),
    reason="Test benchmark not found",
)
def test_e2e_full_gepa_optimization(benchmark_path):
    """Step 9: E2E test - full GEPA optimization loop."""
    import gepa

    from karenina.benchmark import Benchmark
    from karenina.integrations.gepa import KareninaAdapter, OptimizationTarget, questions_to_data_insts
    from karenina.schemas.workflow.models import ModelConfig
    from karenina.schemas.workflow.verification.config import VerificationConfig

    benchmark = Benchmark.load(benchmark_path)
    all_ids = benchmark.get_question_ids()[:5]
    train_insts = questions_to_data_insts(benchmark, all_ids[:3])
    val_insts = questions_to_data_insts(benchmark, all_ids[3:5])

    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="answer-haiku",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                system_prompt="You are a helpful assistant.",
            )
        ],
        parsing_models=[
            ModelConfig(
                id="parser-haiku",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
            )
        ],
        evaluation_mode="template_only",
    )

    adapter = KareninaAdapter(
        benchmark=benchmark,
        base_config=config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
    )

    seed = {"answering_system_prompt": "You are a helpful assistant."}

    result = gepa.optimize(
        seed_candidate=seed,
        trainset=train_insts,
        valset=val_insts,
        adapter=adapter,
        reflection_lm="anthropic/claude-haiku-4-5",
        max_metric_calls=5,
    )

    assert result.best_candidate is not None
    assert len(result.val_aggregate_scores) > 0


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld").exists(),
    reason="Test benchmark not found",
)
def test_e2e_benchmark_optimize_api(benchmark_path):
    """Step 10: E2E test - Benchmark.optimize() high-level API."""
    from karenina.benchmark import Benchmark
    from karenina.schemas.workflow.models import ModelConfig
    from karenina.schemas.workflow.verification.config import VerificationConfig

    benchmark = Benchmark.load(benchmark_path)

    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="answer-haiku",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                system_prompt="You are a helpful assistant.",
            )
        ],
        parsing_models=[
            ModelConfig(
                id="parser-haiku",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
            )
        ],
        evaluation_mode="template_only",
    )

    result = benchmark.optimize(
        targets=["answering_system_prompt"],
        config=config,
        train_ratio=0.6,
        val_ratio=0.4,
        seed=42,
        reflection_model="anthropic/claude-haiku-4-5",
        max_metric_calls=5,
    )

    assert result.val_score is not None
    assert result.improvement is not None


# =============================================================================
# Step 8b: E2E Tests - LLM Feedback Generation
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld").exists(),
    reason="Test benchmark not found",
)
def test_e2e_feedback_generator_single_model(benchmark_path):
    """E2E test - feedback generation with single model (no differential)."""
    from karenina.benchmark import Benchmark
    from karenina.integrations.gepa import LLMFeedbackGenerator, KareninaAdapter, OptimizationTarget, questions_to_data_insts
    from karenina.schemas.workflow.models import ModelConfig
    from karenina.schemas.workflow.verification.config import VerificationConfig

    benchmark = Benchmark.load(benchmark_path)
    test_question_ids = benchmark.get_question_ids()[:2]

    # Single model config
    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="haiku",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                temperature=0.0,
            )
        ],
        parsing_models=[
            ModelConfig(
                id="parser",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
            )
        ],
    )

    # Feedback model
    feedback_model = ModelConfig(
        id="feedback",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        temperature=0.7,
    )

    # Create adapter with feedback
    adapter = KareninaAdapter(
        benchmark=benchmark,
        base_config=config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        feedback_model_config=feedback_model,
        enable_differential_analysis=False,  # Single model, no differential
    )

    data_insts = questions_to_data_insts(benchmark, test_question_ids)

    # Evaluate
    eval_result = adapter.evaluate(
        batch=data_insts,
        candidate={"answering_system_prompt": "You are a helpful assistant."},
        capture_traces=True,
    )

    # Make reflective dataset
    reflective_data = adapter.make_reflective_dataset(
        candidate={"answering_system_prompt": "You are a helpful assistant."},
        eval_batch=eval_result,
        components_to_update=["answering_system_prompt"],
    )

    # Verify we have feedback if there were failures
    failed_count = sum(1 for s in eval_result.scores if s < 1.0)
    feedback_items = reflective_data.get("answering_system_prompt", [])

    if failed_count > 0:
        assert len(feedback_items) > 0, "Should have feedback for failures"
        # Verify feedback structure
        for item in feedback_items:
            assert "Feedback" in item
            assert len(item["Feedback"]) > 50, "Feedback should be substantive"
            assert "TEMPLATE VERIFICATION FEEDBACK" in item["Feedback"]


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld").exists(),
    reason="Test benchmark not found",
)
def test_e2e_feedback_generator_differential_analysis(benchmark_path):
    """E2E test - differential analysis with two models (haiku vs sonnet)."""
    from karenina.benchmark import Benchmark
    from karenina.integrations.gepa import KareninaAdapter, OptimizationTarget, questions_to_data_insts
    from karenina.schemas.workflow.models import ModelConfig
    from karenina.schemas.workflow.verification.config import VerificationConfig

    benchmark = Benchmark.load(benchmark_path)
    test_question_ids = benchmark.get_question_ids()[:2]

    # Two-model config for differential analysis
    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="haiku",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                temperature=0.0,
            ),
            ModelConfig(
                id="sonnet",
                model_name="claude-sonnet-4-5",
                model_provider="anthropic",
                temperature=0.0,
            ),
        ],
        parsing_models=[
            ModelConfig(
                id="parser",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
            )
        ],
    )

    feedback_model = ModelConfig(
        id="feedback",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        temperature=0.7,
    )

    adapter = KareninaAdapter(
        benchmark=benchmark,
        base_config=config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        feedback_model_config=feedback_model,
        enable_differential_analysis=True,  # Enable differential
    )

    data_insts = questions_to_data_insts(benchmark, test_question_ids)

    eval_result = adapter.evaluate(
        batch=data_insts,
        candidate={"answering_system_prompt": "You are a helpful assistant."},
        capture_traces=True,
    )

    reflective_data = adapter.make_reflective_dataset(
        candidate={"answering_system_prompt": "You are a helpful assistant."},
        eval_batch=eval_result,
        components_to_update=["answering_system_prompt"],
    )

    feedback_items = reflective_data.get("answering_system_prompt", [])

    # If we have feedback, verify it's substantive
    if feedback_items:
        for item in feedback_items:
            assert "Feedback" in item
            feedback_text = item["Feedback"]
            assert len(feedback_text) > 50, "Feedback should be substantive"
            # Should contain template verification section
            assert "TEMPLATE VERIFICATION FEEDBACK" in feedback_text


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("/Users/carli/Projects/karenina-monorepo/local_data/data/checkpoints/karenina_checkpoint_ot_bench_v5.4.jsonld").exists(),
    reason="Test benchmark not found",
)
def test_e2e_feedback_generator_with_rubrics(benchmark_path):
    """E2E test - feedback generation with rubric evaluation enabled."""
    from karenina.benchmark import Benchmark
    from karenina.integrations.gepa import KareninaAdapter, OptimizationTarget, questions_to_data_insts
    from karenina.schemas.workflow.models import ModelConfig
    from karenina.schemas.workflow.verification.config import VerificationConfig

    benchmark = Benchmark.load(benchmark_path)
    test_question_ids = benchmark.get_question_ids()[:2]

    config = VerificationConfig(
        answering_models=[
            ModelConfig(
                id="haiku",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
                temperature=0.0,
            )
        ],
        parsing_models=[
            ModelConfig(
                id="parser",
                model_name="claude-haiku-4-5",
                model_provider="anthropic",
            )
        ],
        use_rubrics=True,  # Enable rubric evaluation
    )

    feedback_model = ModelConfig(
        id="feedback",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        temperature=0.7,
    )

    adapter = KareninaAdapter(
        benchmark=benchmark,
        base_config=config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        feedback_model_config=feedback_model,
    )

    data_insts = questions_to_data_insts(benchmark, test_question_ids)

    eval_result = adapter.evaluate(
        batch=data_insts,
        candidate={"answering_system_prompt": "You are a helpful assistant."},
        capture_traces=True,
    )

    reflective_data = adapter.make_reflective_dataset(
        candidate={"answering_system_prompt": "You are a helpful assistant."},
        eval_batch=eval_result,
        components_to_update=["answering_system_prompt"],
    )

    feedback_items = reflective_data.get("answering_system_prompt", [])

    # Test passes if:
    # - No failures (no feedback needed)
    # - Or failures with rubric feedback
    if feedback_items:
        for item in feedback_items:
            feedback_text = item["Feedback"]
            # Should have template feedback section
            assert "TEMPLATE VERIFICATION FEEDBACK" in feedback_text
            # If rubrics present in trajectory, should have rubric section
            # (May not always be present if no rubrics failed)


def test_cli_help():
    """Step 11: Test CLI optimize --help works."""
    import subprocess

    result = subprocess.run(
        ["uv", "run", "karenina", "optimize", "--help"],
        capture_output=True,
        text=True,
        cwd="/Users/carli/Projects/karenina-monorepo/karenina",
    )

    assert result.returncode == 0
    assert "optimize" in result.stdout.lower()
    assert "--target" in result.stdout
