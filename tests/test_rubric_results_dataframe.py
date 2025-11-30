"""Tests for RubricResults.to_dataframe() method."""

import pytest

from karenina.schemas.workflow.rubric_results import RubricResults
from karenina.schemas.workflow.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultRubric,
)


def test_to_dataframe_llm_score_traits():
    """Test DataFrame creation for LLM score traits (1-5 scale)."""
    metadata = VerificationResultMetadata(
        question_id="q1",
        template_id="template1",
        completed_without_errors=True,
        error=None,
        question_text="What is photosynthesis?",
        keywords=["biology"],
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=2.5,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        llm_trait_scores={
            "clarity": 4,
            "accuracy": 5,
            "completeness": 3,
        },
        evaluation_rubric={"test": "rubric"},
    )

    result = VerificationResult(metadata=metadata, rubric=rubric)
    rubric_results = RubricResults(results=[result])

    # Test llm_score type filter
    df = rubric_results.to_dataframe(trait_type="llm_score")

    assert len(df) == 3, "Should have 3 rows (one per LLM trait)"
    assert set(df["trait_name"].values) == {"clarity", "accuracy", "completeness"}
    assert all(df["trait_type"] == "llm_score")

    # Check clarity trait
    clarity_row = df[df["trait_name"] == "clarity"].iloc[0]
    assert clarity_row["trait_score"] == 4
    assert clarity_row["question_id"] == "q1"
    assert clarity_row["completed_without_errors"]
    assert clarity_row["replicate"] == 1


def test_to_dataframe_llm_binary_traits():
    """Test DataFrame creation for LLM binary traits."""
    metadata = VerificationResultMetadata(
        question_id="q2",
        template_id="template2",
        completed_without_errors=True,
        error=None,
        question_text="Binary test",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=1.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        llm_trait_scores={
            "is_relevant": True,
            "is_complete": False,
        },
    )

    result = VerificationResult(metadata=metadata, rubric=rubric)
    rubric_results = RubricResults(results=[result])

    # Test llm_binary type filter
    df = rubric_results.to_dataframe(trait_type="llm_binary")

    assert len(df) == 2, "Should have 2 rows"
    assert set(df["trait_name"].values) == {"is_relevant", "is_complete"}
    assert all(df["trait_type"] == "llm_binary")

    # Check boolean values
    relevant_row = df[df["trait_name"] == "is_relevant"].iloc[0]
    assert relevant_row["trait_score"]


def test_to_dataframe_regex_traits():
    """Test DataFrame creation for regex traits."""
    metadata = VerificationResultMetadata(
        question_id="q3",
        template_id="template3",
        completed_without_errors=True,
        error=None,
        question_text="Regex traits test",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=1.5,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        regex_trait_scores={
            "has_citation": True,
            "mentions_regulatory": False,
        },
    )

    result = VerificationResult(metadata=metadata, rubric=rubric)
    rubric_results = RubricResults(results=[result])

    df = rubric_results.to_dataframe(trait_type="regex")

    assert len(df) == 2, "Should have 2 rows"
    assert set(df["trait_name"].values) == {"has_citation", "mentions_regulatory"}
    assert all(df["trait_type"] == "regex")

    # Check values
    citation_row = df[df["trait_name"] == "has_citation"].iloc[0]
    assert citation_row["trait_score"]


def test_to_dataframe_metric_traits_explosion():
    """Test DataFrame creation for metric traits (EXPLODED by metric)."""
    metadata = VerificationResultMetadata(
        question_id="q4",
        template_id="template4",
        completed_without_errors=True,
        error=None,
        question_text="Metric traits test",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=2.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        metric_trait_scores={
            "entity_extraction": {
                "precision": 0.85,
                "recall": 0.92,
                "f1": 0.88,
            },
            "feature_identification": {
                "precision": 1.0,
                "recall": 0.75,
                "f1": 0.86,
            },
        },
        metric_trait_confusion_lists={
            "entity_extraction": {
                "tp": ["Paris", "France"],
                "fp": ["approximately"],
                "fn": ["Europe"],
                "tn": [],
            }
        },
    )

    result = VerificationResult(metadata=metadata, rubric=rubric)
    rubric_results = RubricResults(results=[result])

    df = rubric_results.to_dataframe(trait_type="metric")

    # Should have 6 rows: 2 traits × 3 metrics each
    assert len(df) == 6
    assert all(df["trait_type"] == "metric")

    # Check metric names
    assert set(df["metric_name"].values) == {"precision", "recall", "f1"}

    # Check entity_extraction precision row
    entity_precision = df[(df["trait_name"] == "entity_extraction") & (df["metric_name"] == "precision")].iloc[0]
    assert entity_precision["trait_score"] == 0.85
    assert entity_precision["confusion_tp"] == ["Paris", "France"]
    assert entity_precision["confusion_fp"] == ["approximately"]
    assert entity_precision["confusion_fn"] == ["Europe"]
    assert entity_precision["confusion_tn"] == []

    # Check feature_identification has no confusion lists
    feature_recall = df[(df["trait_name"] == "feature_identification") & (df["metric_name"] == "recall")].iloc[0]
    assert feature_recall["trait_score"] == 0.75
    assert feature_recall["confusion_tp"] is None


def test_to_dataframe_all_trait_types():
    """Test DataFrame with all trait types combined."""
    metadata = VerificationResultMetadata(
        question_id="q5",
        template_id="template5",
        completed_without_errors=True,
        error=None,
        question_text="All traits test",
        keywords=["combined"],
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=3.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        llm_trait_scores={
            "clarity": 4,
        },
        regex_trait_scores={
            "has_citation": True,
        },
        metric_trait_scores={
            "entity_extraction": {
                "precision": 0.85,
                "recall": 0.92,
            },
        },
    )

    result = VerificationResult(metadata=metadata, rubric=rubric)
    rubric_results = RubricResults(results=[result])

    df = rubric_results.to_dataframe(trait_type="all")

    # Should have 4 rows: 1 LLM + 1 regex + 2 metrics
    assert len(df) == 4

    # Check trait types
    assert set(df["trait_type"].values) == {"llm_score", "regex", "metric"}

    # Verify each type is present
    assert len(df[df["trait_type"] == "llm_score"]) == 1
    assert len(df[df["trait_type"] == "regex"]) == 1
    assert len(df[df["trait_type"] == "metric"]) == 2


def test_to_dataframe_no_rubric_data():
    """Test DataFrame creation when result has no rubric data."""
    metadata = VerificationResultMetadata(
        question_id="q6",
        template_id="template6",
        completed_without_errors=False,
        error="Rubric evaluation failed",
        question_text="Failed rubric",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=1.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    result = VerificationResult(metadata=metadata, rubric=None)
    rubric_results = RubricResults(results=[result])

    df = rubric_results.to_dataframe()

    # Should have one row with minimal data
    assert len(df) == 1
    assert not df.iloc[0]["completed_without_errors"]
    assert df.iloc[0]["error"] == "Rubric evaluation failed"
    assert df.iloc[0]["trait_name"] is None
    assert df.iloc[0]["trait_score"] is None


def test_to_dataframe_multiple_results():
    """Test DataFrame with multiple verification results."""
    results = []

    for i in range(3):
        metadata = VerificationResultMetadata(
            question_id=f"q{i + 1}",
            template_id=f"template{i + 1}",
            completed_without_errors=True,
            error=None,
            question_text=f"Question {i + 1}",
            keywords=None,
            answering_model="gpt-4",
            parsing_model="gpt-4-mini",
            execution_time=2.0,
            timestamp="2024-01-15T10:30:00",
            answering_replicate=1,
        )

        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={
                "clarity": i + 3,  # 3, 4, 5
            },
        )

        results.append(VerificationResult(metadata=metadata, rubric=rubric))

    rubric_results = RubricResults(results=results)
    df = rubric_results.to_dataframe(trait_type="llm_score")

    # Should have 3 rows (one per result)
    assert len(df) == 3
    assert set(df["question_id"].values) == {"q1", "q2", "q3"}
    assert all(df["trait_name"] == "clarity")
    assert set(df["trait_score"].values) == {3, 4, 5}


def test_to_dataframe_mixed_llm_types():
    """Test DataFrame with mixed LLM trait types (score and binary)."""
    metadata = VerificationResultMetadata(
        question_id="q7",
        template_id="template7",
        completed_without_errors=True,
        error=None,
        question_text="Mixed LLM traits",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=2.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        llm_trait_scores={
            "clarity": 4,  # int score
            "accuracy": 5,  # int score
            "is_relevant": True,  # bool
            "is_complete": False,  # bool
        },
    )

    result = VerificationResult(metadata=metadata, rubric=rubric)
    rubric_results = RubricResults(results=[result])

    # Test "llm" type (all LLM traits)
    df_llm = rubric_results.to_dataframe(trait_type="llm")
    assert len(df_llm) == 4

    # Test filtering by llm_score
    df_score = rubric_results.to_dataframe(trait_type="llm_score")
    assert len(df_score) == 2
    assert set(df_score["trait_name"].values) == {"clarity", "accuracy"}

    # Test filtering by llm_binary
    df_binary = rubric_results.to_dataframe(trait_type="llm_binary")
    assert len(df_binary) == 2
    assert set(df_binary["trait_name"].values) == {"is_relevant", "is_complete"}


def test_to_dataframe_column_ordering():
    """Test that DataFrame columns follow the correct ordering."""
    metadata = VerificationResultMetadata(
        question_id="q8",
        template_id="template8",
        completed_without_errors=True,
        error=None,
        question_text="Column order test",
        keywords=["test"],
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        answering_system_prompt="Answer questions",
        parsing_system_prompt="Parse responses",
        execution_time=2.5,
        timestamp="2024-01-15T10:30:00",
        run_name="test_run",
        answering_replicate=1,
    )

    rubric = VerificationResultRubric(
        rubric_evaluation_performed=True,
        llm_trait_scores={"clarity": 4},
        evaluation_rubric={"test": "rubric"},
    )

    result = VerificationResult(metadata=metadata, rubric=rubric)
    rubric_results = RubricResults(results=[result])

    df = rubric_results.to_dataframe()
    row = df.iloc[0]

    # Verify Status columns
    assert row["completed_without_errors"]
    assert row["error"] is None

    # Verify Identification metadata
    assert row["question_id"] == "q8"
    assert row["template_id"] == "template8"
    assert row["question_text"] == "Column order test"
    assert row["keywords"] == ["test"]
    assert row["replicate"] == 1

    # Verify Model Configuration
    assert row["answering_model"] == "gpt-4"
    assert row["parsing_model"] == "gpt-4-mini"
    assert row["answering_system_prompt"] == "Answer questions"
    assert row["parsing_system_prompt"] == "Parse responses"

    # Verify Rubric Data
    assert row["trait_name"] == "clarity"
    assert row["trait_score"] == 4
    assert row["trait_type"] == "llm_score"

    # Verify Execution Metadata
    assert row["execution_time"] == 2.5
    assert row["timestamp"] == "2024-01-15T10:30:00"
    assert row["run_name"] == "test_run"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
