"""
Integration test for the new trait system (RegexTrait + CallableTrait).

This test verifies that the refactored trait system works end-to-end:
- Loading checkpoints with RegexTrait and CallableTrait
- Running verification with all four trait types (LLM, Regex, Callable, Metric)
- Results properly split into regex_trait_scores and callable_trait_scores
- Database storage and retrieval works correctly
- Export functionality handles all trait types
"""

from pathlib import Path

import pytest

from karenina.benchmark import Benchmark
from karenina.schemas import VerificationConfig


@pytest.fixture
def enriched_checkpoint_path():
    """Path to the enriched checkpoint with all trait types."""
    return Path("/Users/carli/Projects/karenina_dev/checkpoints/latest_rubric_advanced_with_callables.jsonld")


@pytest.fixture
def rubric_config_path():
    """Path to the rubric verification config."""
    return Path("/Users/carli/Projects/karenina_dev/presets/gpt-oss-003-8000-rubrics.json")


@pytest.fixture
def benchmark(enriched_checkpoint_path):
    """Load the enriched benchmark."""
    return Benchmark.load(enriched_checkpoint_path)


def test_checkpoint_loads_with_all_trait_types(benchmark):
    """Verify checkpoint loads correctly with all four trait types."""
    global_rubric = benchmark.get_global_rubric()

    assert global_rubric is not None, "Global rubric should exist"

    # Check all trait types are present
    assert len(global_rubric.traits) == 4, "Should have 4 LLM traits"
    assert len(global_rubric.regex_traits) == 2, "Should have 2 Regex traits"
    assert len(global_rubric.callable_traits) == 2, "Should have 2 Callable traits"
    assert len(global_rubric.metric_traits) == 0, "Should have 0 Metric traits"

    # Verify callable trait details
    callable_names = {t.name for t in global_rubric.callable_traits}
    assert "Contains Citations" in callable_names
    assert "Response Quality" in callable_names

    # Check boolean vs score
    citation_trait = next(t for t in global_rubric.callable_traits if t.name == "Contains Citations")
    quality_trait = next(t for t in global_rubric.callable_traits if t.name == "Response Quality")

    assert citation_trait.kind == "boolean"
    assert quality_trait.kind == "score"
    assert quality_trait.min_score == 1
    assert quality_trait.max_score == 5


def test_callable_trait_evaluation(benchmark):
    """Verify callable traits can evaluate text correctly."""
    global_rubric = benchmark.get_global_rubric()

    # Test boolean callable
    citation_trait = next(t for t in global_rubric.callable_traits if t.name == "Contains Citations")

    assert citation_trait.evaluate("This is a fact [1].") is True
    assert citation_trait.evaluate("No citations here.") is False

    # Test score callable
    quality_trait = next(t for t in global_rubric.callable_traits if t.name == "Response Quality")

    short_score = quality_trait.evaluate("Short.")
    long_score = quality_trait.evaluate(
        "This is a comprehensive answer that provides extensive detail. "
        "It covers multiple aspects thoroughly. Each point is clear. "
        "The response demonstrates understanding. Multiple perspectives are considered."
    )

    assert isinstance(short_score, int)
    assert isinstance(long_score, int)
    assert 1 <= short_score <= 5
    assert 1 <= long_score <= 5
    assert long_score > short_score, "Longer answer should score higher"


def test_verification_with_all_trait_types(benchmark, rubric_config_path):
    """Run verification and verify all trait types are evaluated."""
    # Load config
    config = VerificationConfig.from_preset(rubric_config_path)

    # Get first question only for quick test
    questions = benchmark.get_all_questions(ids_only=False)
    finished_questions = [q for q in questions if q.get("finished") and q.get("answer_template")]

    if not finished_questions:
        pytest.skip("No finished questions with templates available")

    test_question_id = finished_questions[0]["id"]

    # Run verification on one question
    print(f"\nRunning verification on question: {test_question_id}")
    result_set = benchmark.run_verification(config, question_ids=[test_question_id])

    assert len(result_set.results) > 0, "Should have at least one result"

    # Get the first result from the set
    result = result_set.results[0]

    # Verify result has rubric evaluation
    assert result.rubric is not None, "Result should have rubric evaluation"
    assert result.rubric.rubric_evaluation_performed is True

    # Verify trait scores are split correctly
    print("\nVerifying trait score separation:")

    # Check LLM trait scores
    if result.rubric.llm_trait_scores:
        print(f"  ✅ LLM trait scores: {len(result.rubric.llm_trait_scores)} traits")
        assert isinstance(result.rubric.llm_trait_scores, dict)

    # Check Regex trait scores
    if result.rubric.regex_trait_scores:
        print(f"  ✅ Regex trait scores: {len(result.rubric.regex_trait_scores)} traits")
        assert isinstance(result.rubric.regex_trait_scores, dict)
        for trait_name, score in result.rubric.regex_trait_scores.items():
            assert isinstance(score, bool), f"Regex trait {trait_name} should return bool"

    # Check Callable trait scores
    if result.rubric.callable_trait_scores:
        print(f"  ✅ Callable trait scores: {len(result.rubric.callable_trait_scores)} traits")
        assert isinstance(result.rubric.callable_trait_scores, dict)

        for trait_name, score in result.rubric.callable_trait_scores.items():
            # Score can be bool or int depending on trait kind
            assert isinstance(score, bool | int), f"Callable trait {trait_name} should return bool or int"
            print(f"     - {trait_name}: {score} ({type(score).__name__})")

    # Verify get_all_trait_scores() includes all types
    all_scores = result.rubric.get_all_trait_scores()
    print(f"\n  Total traits in get_all_trait_scores(): {len(all_scores)}")
    assert len(all_scores) > 0, "Should have aggregated trait scores"


def test_database_storage_with_new_traits(benchmark, rubric_config_path, tmp_path):
    """Verify results can store and retrieve results with new trait types."""
    # Get first question
    questions = benchmark.get_all_questions(ids_only=False)
    finished_questions = [q for q in questions if q.get("finished") and q.get("answer_template")]

    if not finished_questions:
        pytest.skip("No finished questions with templates available")

    test_question_id = finished_questions[0]["id"]

    # Load config
    config = VerificationConfig.from_preset(rubric_config_path)

    # Run verification
    result_set = benchmark.run_verification(config, question_ids=[test_question_id])

    # Store results in memory
    benchmark.store_verification_results(result_set, run_name="test_trait_storage")

    # Retrieve results
    loaded_results = benchmark.get_verification_results(question_ids=[test_question_id], run_name="test_trait_storage")

    assert len(loaded_results) > 0, "Should have stored results"

    # Verify trait scores persisted correctly
    original_result = result_set.results[0]
    loaded_result = list(loaded_results.values())[0]

    if original_result.rubric:
        assert loaded_result.rubric is not None

        # Compare regex trait scores
        if original_result.rubric.regex_trait_scores:
            assert loaded_result.rubric.regex_trait_scores == original_result.rubric.regex_trait_scores
            print("  ✅ Regex trait scores stored correctly")

        # Compare callable trait scores
        if original_result.rubric.callable_trait_scores:
            assert loaded_result.rubric.callable_trait_scores == original_result.rubric.callable_trait_scores
            print("  ✅ Callable trait scores stored correctly")


def test_export_with_all_trait_types(benchmark, rubric_config_path, tmp_path):
    """Verify export functionality handles all trait types."""
    # Get first question
    questions = benchmark.get_all_questions(ids_only=False)
    finished_questions = [q for q in questions if q.get("finished") and q.get("answer_template")]

    if not finished_questions:
        pytest.skip("No finished questions with templates available")

    test_question_id = finished_questions[0]["id"]

    # Load config
    config = VerificationConfig.from_preset(rubric_config_path)

    # Run verification
    result_set = benchmark.run_verification(config, question_ids=[test_question_id])

    # Store results so we can export them
    benchmark.store_verification_results(result_set, run_name="test_export")

    # Export to CSV
    csv_path = tmp_path / "results.csv"
    global_rubric = benchmark.get_global_rubric()
    benchmark.export_verification_results_to_file(
        file_path=csv_path,
        question_ids=[test_question_id],
        run_name="test_export",
        format="csv",
        global_rubric=global_rubric,
    )

    assert csv_path.exists(), "CSV file should be created"

    # Read CSV and verify trait columns exist
    import csv

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        assert len(rows) > 0, "Should have at least one row"

        # Check that rubric trait columns exist
        headers = reader.fieldnames

        # Should have columns for all trait types
        rubric_columns = [h for h in headers if h.startswith("rubric_")]
        assert len(rubric_columns) > 0, "Should have rubric trait columns"

        print(f"\n  CSV export created with {len(rubric_columns)} rubric columns")


if __name__ == "__main__":
    # Run tests manually for debugging

    checkpoint_path = Path(
        "/Users/carli/Projects/karenina_dev/checkpoints/latest_rubric_advanced_with_callables.jsonld"
    )
    config_path = Path("/Users/carli/Projects/karenina_dev/presets/gpt-oss-003-8000-rubrics.json")

    benchmark = Benchmark.load(checkpoint_path)

    print("=" * 80)
    print("Integration Test: New Trait System")
    print("=" * 80)

    print("\n1. Testing checkpoint loading...")
    test_checkpoint_loads_with_all_trait_types(benchmark)
    print("   ✅ Checkpoint loaded with all trait types")

    print("\n2. Testing callable trait evaluation...")
    test_callable_trait_evaluation(benchmark)
    print("   ✅ Callable traits evaluate correctly")

    print("\n3. Testing verification with all trait types...")
    test_verification_with_all_trait_types(benchmark, config_path)
    print("   ✅ Verification works with all trait types")

    print("\n" + "=" * 80)
    print("All integration tests passed! ✅")
    print("=" * 80)
