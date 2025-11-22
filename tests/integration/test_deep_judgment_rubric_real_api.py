"""
Real API integration tests for Deep Judgment Rubrics.

These tests make actual LLM API calls following the pattern from run-verification-info skill.
They use real preset configs and checkpoint to test deep judgment functionality.

WARNING: These tests:
- Make real API calls and consume tokens
- Require API keys to be set in environment
- May be slow (several seconds per test)
- Should be run selectively, not in every test run

Run with: pytest tests/integration/test_deep_judgment_rubric_real_api.py -v -s
Skip with: pytest -m "not real_api"

Resources used (from run-verification-info skill):
- Preset: /Users/carli/Projects/karenina_dev/presets/gpt-oss-001-8000.json
- Checkpoint: /Users/carli/Projects/karenina_dev/checkpoints/latest_rubric_advanced_with_callables.jsonld
"""

from pathlib import Path

import pytest

from karenina.benchmark import Benchmark
from karenina.schemas import (
    LLMRubricTrait,
    Rubric,
    VerificationConfig,
)

# Paths from run-verification-info skill
PRESET_GPT_OSS = Path("/Users/carli/Projects/karenina_dev/presets/gpt-oss-001-8000.json")
CHECKPOINT_WITH_RUBRICS = Path(
    "/Users/carli/Projects/karenina_dev/checkpoints/latest_rubric_advanced_with_callables.jsonld"
)


@pytest.mark.real_api
@pytest.mark.skipif(
    not PRESET_GPT_OSS.exists() or not CHECKPOINT_WITH_RUBRICS.exists(),
    reason="Preset config or checkpoint not available",
)
def test_deep_judgment_rubrics_end_to_end_real_api():
    """
    End-to-end test with REAL LLM API calls for deep judgment rubrics.

    This test:
    - Loads benchmark from checkpoint with existing rubrics
    - Modifies rubric traits to enable deep judgment
    - Runs actual verification with real LLM calls on one question
    - Verifies deep judgment results are populated correctly

    Expected behavior:
    - Excerpts are extracted and validated
    - Reasoning is generated for each trait
    - Scores are extracted
    - Metadata tracks model calls and stages
    """
    print("\n" + "=" * 70)
    print("REAL API TEST: Deep Judgment Rubrics End-to-End")
    print("=" * 70)

    # Load benchmark from checkpoint
    print(f"\nLoading checkpoint: {CHECKPOINT_WITH_RUBRICS.name}")
    benchmark = Benchmark.load(CHECKPOINT_WITH_RUBRICS)
    print(f"✓ Benchmark loaded: {benchmark.name}")

    questions = benchmark.get_all_questions(ids_only=False)
    print(f"✓ Questions in benchmark: {len(questions)}")

    # Get existing global rubric
    rubric = benchmark.get_global_rubric()
    print(f"✓ Global rubric: {len(rubric.llm_traits)} LLM traits, {len(rubric.regex_traits)} regex traits")

    # Load verification config from preset
    print(f"\nLoading config from: {PRESET_GPT_OSS}")
    config = VerificationConfig.from_preset(PRESET_GPT_OSS)

    # Ensure rubric evaluation is enabled
    config.rubric_enabled = True

    # Enable deep judgment for ALL LLM traits using new configuration system
    config.deep_judgment_rubric_mode = "enable_all"
    config.deep_judgment_rubric_global_excerpts = True
    config.deep_judgment_rubric_max_excerpts_default = 3
    config.deep_judgment_rubric_fuzzy_match_threshold_default = 0.75
    config.deep_judgment_rubric_excerpt_retry_attempts_default = 2

    print(f"✓ Config loaded: {config.answering_models[0].model_name}")
    print(f"  Evaluation mode: {config.evaluation_mode}")
    print(f"  Rubric enabled: {config.rubric_enabled}")
    print(f"  Deep judgment rubric mode: {config.deep_judgment_rubric_mode}")
    print(f"  Deep judgment global excerpts: {config.deep_judgment_rubric_global_excerpts}")

    # Run verification on first finished question only
    finished_questions = [q for q in questions if q.get("finished")]
    if not finished_questions:
        pytest.skip("No finished questions in checkpoint")

    question_id = finished_questions[0]["id"]
    print(f"\nRunning verification on question: {question_id}")
    print(f"  Question: {finished_questions[0]['question'][:80]}...")
    print("⚠️  Making real LLM API calls...")

    result_set = benchmark.run_verification(
        config=config,
        question_ids=[question_id],
    )

    # Should have exactly one result
    assert len(result_set.results) == 1, f"Expected 1 result, got {len(result_set.results)}"
    result = result_set.results[0]

    print("\n✓ Verification completed")
    print(f"  Result question: {result.metadata.question_id}")

    # Verify basic structure
    assert result.deep_judgment_rubric is not None, "Deep judgment rubric data missing"
    assert result.deep_judgment_rubric.deep_judgment_rubric_performed is True, "Deep judgment not performed"

    print("✓ Deep judgment rubric performed")

    # With enable_all mode, ALL LLM traits should be evaluated with deep judgment
    dj_rubric = result.deep_judgment_rubric

    # Should have excerpts for traits (enable_all with excerpts enabled)
    assert dj_rubric.extracted_rubric_excerpts is not None, "No excerpts extracted"

    # Get all LLM traits from rubric
    all_llm_trait_names = [t.name for t in rubric.llm_traits]
    print(f"✓ LLM traits in rubric: {all_llm_trait_names}")

    # Verify at least one trait has excerpts (some might have none if answer doesn't support them)
    assert len(dj_rubric.extracted_rubric_excerpts) > 0, "No traits have excerpts"

    # Test one trait in detail (Clarity)
    if "Clarity" in dj_rubric.extracted_rubric_excerpts:
        excerpts = dj_rubric.extracted_rubric_excerpts["Clarity"]
        if len(excerpts) > 0:
            assert all("text" in e for e in excerpts), "Excerpt missing 'text' field"
            assert all("confidence" in e for e in excerpts), "Excerpt missing 'confidence' field"
            assert all("similarity_score" in e for e in excerpts), "Excerpt missing 'similarity_score'"

            print(f"✓ Excerpts extracted for Clarity: {len(excerpts)} excerpts")
            for i, excerpt in enumerate(excerpts[:2]):  # Show first 2
                excerpt_text = excerpt["text"][:60] if len(excerpt["text"]) > 60 else excerpt["text"]
                print(f'  [{i}] "{excerpt_text}..." (similarity: {excerpt["similarity_score"]:.2f})')

    # Should have reasoning for ALL LLM traits
    assert dj_rubric.rubric_trait_reasoning is not None, "No reasoning generated"
    for trait_name in all_llm_trait_names:
        assert trait_name in dj_rubric.rubric_trait_reasoning, f"Reasoning missing for {trait_name}"
        reasoning = dj_rubric.rubric_trait_reasoning[trait_name]
        assert len(reasoning) > 0, f"Empty reasoning for {trait_name}"

    print(f"✓ Reasoning generated for all {len(all_llm_trait_names)} LLM traits")

    # Should have scores for ALL LLM traits
    assert dj_rubric.deep_judgment_rubric_scores is not None, "No scores generated"
    for trait_name in all_llm_trait_names:
        assert trait_name in dj_rubric.deep_judgment_rubric_scores, f"Score missing for {trait_name}"
        score = dj_rubric.deep_judgment_rubric_scores[trait_name]
        # Boolean traits return True/False, score traits return 1-5
        assert isinstance(score, int | bool), f"Expected int/bool score for {trait_name}, got {type(score)}"

    print(f"✓ All {len(all_llm_trait_names)} LLM traits evaluated with deep judgment")

    # With enable_all mode, there should be NO standard scores (all are deep judgment)
    assert dj_rubric.standard_rubric_scores is None or len(dj_rubric.standard_rubric_scores) == 0, (
        "Expected no standard scores with enable_all mode"
    )

    print("✓ No standard trait scores (all traits use deep judgment)")

    # Verify metadata
    assert dj_rubric.trait_metadata is not None, "No trait metadata"
    assert "Clarity" in dj_rubric.trait_metadata, "Metadata missing for Clarity"
    metadata = dj_rubric.trait_metadata["Clarity"]
    assert "model_calls" in metadata, "model_calls missing from metadata"
    assert "stages_completed" in metadata, "stages_completed missing from metadata"
    assert metadata["model_calls"] >= 3, f"Expected >= 3 model calls, got {metadata['model_calls']}"

    print(f"✓ Metadata: {metadata['model_calls']} model calls, stages: {metadata['stages_completed']}")

    # Verify no auto-fail (excerpts should have validated)
    if dj_rubric.traits_without_valid_excerpts:
        assert "Clarity" not in dj_rubric.traits_without_valid_excerpts, "Clarity should not have failed validation"

    print("✓ No auto-fail triggered")

    # Summary
    print("\n" + "=" * 70)
    print("✅ REAL API TEST PASSED")
    print("=" * 70)
    print(f"Total deep judgment model calls: {dj_rubric.total_deep_judgment_model_calls}")
    print(f"Traits evaluated with deep judgment: {len(dj_rubric.deep_judgment_rubric_scores)}")
    print(f"Traits evaluated with standard method: {len(dj_rubric.standard_rubric_scores)}")
    print("=" * 70)


@pytest.mark.real_api
@pytest.mark.skipif(
    not PRESET_GPT_OSS.exists() or not CHECKPOINT_WITH_RUBRICS.exists(),
    reason="Preset config or checkpoint not available",
)
def test_deep_judgment_rubrics_dataframe_export_real_api():
    """
    Test dataframe export with real deep judgment rubrics data from API calls.

    Validates:
    - Standard export with include_deep_judgment=True
    - Detailed export with get_rubric_judgments_results()
    """
    print("\n" + "=" * 70)
    print("REAL API TEST: Dataframe Export with Deep Judgment")
    print("=" * 70)

    # Create benchmark with question
    benchmark = Benchmark.create(
        name="Deep Judgment Export Test",
    )

    test_question = "What does the BCL-2 protein do?"
    test_answer = (
        "BCL-2 is an anti-apoptotic protein that prevents programmed cell death. "
        "It blocks apoptosis by sequestering pro-apoptotic proteins like BAX and BAK."
    )

    benchmark.add_question(question=test_question, raw_answer=test_answer)

    # Create rubric with deep judgment
    rubric = Rubric(
        name="Export Test Rubric",
        traits=[
            LLMRubricTrait(
                name="MentionsAntiApoptotic",
                description="Does the answer mention BCL-2's anti-apoptotic role?",
                kind="binary",
                deep_judgment_enabled=True,
                deep_judgment_excerpt_enabled=True,
                deep_judgment_max_excerpts=2,
            ),
        ],
    )

    benchmark.set_global_rubric(rubric)

    # Run verification with new API
    config = VerificationConfig.from_preset(PRESET_GPT_OSS)
    config.rubric_enabled = True

    question_ids = [benchmark.get_all_questions(ids_only=True)[0]]
    print("Running verification with real API calls...")

    result_sets = benchmark.run_verification_new(
        config=config,
        question_ids=question_ids,
    )

    # Get result set for the question
    assert len(result_sets) == 1, f"Expected 1 result set, got {len(result_sets)}"
    question_id = list(result_sets.keys())[0]
    result_set = result_sets[question_id]

    print(f"✓ Verification completed for question: {question_id}")

    # Test standard export WITH deep judgment columns
    print("\nTesting standard export with include_deep_judgment=True...")
    rubric_results = result_set.get_rubrics(include_deep_judgment=True)
    df_standard = rubric_results.to_dataframe(trait_type="llm_score")

    # Should have deep judgment columns
    assert "trait_reasoning" in df_standard.columns, "Missing trait_reasoning column"
    assert "trait_excerpts" in df_standard.columns, "Missing trait_excerpts column"
    assert "trait_had_excerpts" in df_standard.columns, "Missing trait_had_excerpts column"

    print("✓ Deep judgment columns present in standard export")
    print(f"  Columns: {list(df_standard.columns)}")

    # Verify content
    assert len(df_standard) > 0, "Empty dataframe"
    row = df_standard.iloc[0]
    assert row["trait_reasoning"] is not None, "No reasoning in row"
    assert len(row["trait_reasoning"]) > 0, "Empty reasoning"

    print(f"✓ Standard export row count: {len(df_standard)}")
    print(f"  Trait: {row['trait_name']}")
    print(f"  Score: {row['trait_score']}")
    print(f'  Reasoning: "{row["trait_reasoning"][:80]}..."')

    # Test detailed export (excerpt explosion)
    print("\nTesting detailed export (excerpt explosion)...")
    rubric_judgments = result_set.get_rubric_judgments_results()
    df_detailed = rubric_judgments.to_dataframe()

    # Should have excerpt-level columns
    expected_cols = [
        "trait_name",
        "trait_score",
        "excerpt_index",
        "excerpt_text",
        "excerpt_confidence",
        "excerpt_similarity_score",
        "trait_reasoning",
        "trait_model_calls",
        "trait_excerpt_retries",
        "trait_had_excerpts",
    ]
    for col in expected_cols:
        assert col in df_detailed.columns, f"Missing column: {col}"

    print("✓ All expected columns present in detailed export")

    # Should have multiple rows (one per excerpt)
    assert len(df_detailed) > 0, "Empty detailed dataframe"

    print(f"✓ Detailed export row count: {len(df_detailed)} (one per excerpt)")

    # Verify excerpt data populated
    if len(df_detailed) > 0:
        first_row = df_detailed.iloc[0]
        if first_row["trait_had_excerpts"]:
            assert first_row["excerpt_text"] is not None, "Missing excerpt_text"
            assert first_row["excerpt_confidence"] is not None, "Missing excerpt_confidence"
            assert first_row["excerpt_similarity_score"] is not None, "Missing excerpt_similarity_score"

            print(f'  Excerpt 0: "{first_row["excerpt_text"][:60]}..."')
            print(f"  Confidence: {first_row['excerpt_confidence']}")
            print(f"  Similarity: {first_row['excerpt_similarity_score']:.2f}")

    print("\n" + "=" * 70)
    print("✅ DATAFRAME EXPORT TEST PASSED")
    print("=" * 70)
    print(f"Standard export: {len(df_standard)} rows, {len(df_standard.columns)} columns")
    print(f"Detailed export: {len(df_detailed)} rows, {len(df_detailed.columns)} columns")
    print("=" * 70)


if __name__ == "__main__":
    # For manual testing
    print("Running Deep Judgment Rubrics Real API Integration Tests...")
    print("\n⚠️  WARNING: These tests make real API calls and consume tokens!")
    print("=" * 70)

    test_deep_judgment_rubrics_end_to_end_real_api()
    print()
    test_deep_judgment_rubrics_dataframe_export_real_api()

    print("\n" + "=" * 70)
    print("✅ All real API integration tests completed!")
    print("=" * 70)
