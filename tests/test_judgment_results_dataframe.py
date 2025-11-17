"""Tests for JudgmentResults.to_dataframe() method."""

import pytest

from karenina.schemas.workflow.judgment_results import JudgmentResults
from karenina.schemas.workflow.verification import (
    VerificationResult,
    VerificationResultDeepJudgment,
    VerificationResultMetadata,
    VerificationResultTemplate,
)


def test_to_dataframe_basic_explosion():
    """Test DataFrame creation with attribute × excerpt explosion."""
    metadata = VerificationResultMetadata(
        question_id="q1",
        template_id="template1",
        completed_without_errors=True,
        error=None,
        question_text="Where is Paris?",
        keywords=["geography"],
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=2.5,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    template = VerificationResultTemplate(
        raw_llm_response="Paris is in France",
        parsed_gt_response={"location": ["Paris", "France"], "person": []},
        parsed_llm_response={"location": ["Paris", "France"], "person": []},
        template_verification_performed=True,
        recursion_limit_reached=False,
    )

    deep_judgment = VerificationResultDeepJudgment(
        deep_judgment_enabled=True,
        deep_judgment_performed=True,
        extracted_excerpts={
            "location": [
                {"text": "Paris", "confidence": "high", "similarity_score": 0.95},
                {"text": "France", "confidence": "medium", "similarity_score": 0.85},
            ],
            "person": [],  # No excerpts for this attribute
        },
        attribute_reasoning={
            "location": "Found clear location entities",
            "person": "No person entities mentioned",
        },
        deep_judgment_model_calls=2,
        deep_judgment_excerpt_retry_count=0,
    )

    result = VerificationResult(metadata=metadata, template=template, deep_judgment=deep_judgment)
    judgment_results = JudgmentResults(results=[result])

    df = judgment_results.to_dataframe()

    # Should have 3 rows: 2 excerpts for location + 1 for person (no excerpts)
    assert len(df) == 3

    # Check location attribute rows (2 excerpts)
    location_rows = df[df["attribute_name"] == "location"]
    assert len(location_rows) == 2
    assert set(location_rows["excerpt_index"].values) == {0, 1}
    assert set(location_rows["excerpt_text"].values) == {"Paris", "France"}

    # Check person attribute row (no excerpts)
    person_row = df[df["attribute_name"] == "person"].iloc[0]
    import pandas as pd

    assert pd.isna(person_row["excerpt_index"])  # Pandas converts None to NaN for numeric columns
    assert person_row["excerpt_text"] is None
    assert not person_row["attribute_has_excerpts"]


def test_to_dataframe_with_search_enhancement():
    """Test DataFrame with search-enhanced deep judgment."""
    metadata = VerificationResultMetadata(
        question_id="q2",
        template_id="template2",
        completed_without_errors=True,
        error=None,
        question_text="Test search",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=3.0,
        timestamp="2024-01-15T10:32:00",
        answering_replicate=1,
    )

    template = VerificationResultTemplate(
        raw_llm_response="Test response",
        parsed_gt_response={"fact": ["test"]},
        parsed_llm_response={"fact": ["test"]},
        template_verification_performed=True,
        recursion_limit_reached=False,
    )

    deep_judgment = VerificationResultDeepJudgment(
        deep_judgment_enabled=True,
        deep_judgment_performed=True,
        deep_judgment_search_enabled=True,
        extracted_excerpts={
            "fact": [
                {
                    "text": "test",
                    "confidence": "high",
                    "similarity_score": 0.9,
                    "search_results": "External validation text",
                    "hallucination_risk": "low",
                    "hallucination_justification": "Supported by external sources",
                }
            ]
        },
        hallucination_risk_assessment={"fact": "low"},
        deep_judgment_model_calls=3,
        deep_judgment_excerpt_retry_count=1,
    )

    result = VerificationResult(metadata=metadata, template=template, deep_judgment=deep_judgment)
    judgment_results = JudgmentResults(results=[result])

    df = judgment_results.to_dataframe()

    assert len(df) == 1

    row = df.iloc[0]
    assert row["deep_judgment_search_enabled"]
    assert row["excerpt_search_results"] == "External validation text"
    assert row["excerpt_hallucination_risk"] == "low"
    assert row["excerpt_hallucination_justification"] == "Supported by external sources"
    assert row["attribute_overall_risk"] == "low"


def test_to_dataframe_no_judgment_data():
    """Test DataFrame when result has no deep judgment data."""
    metadata = VerificationResultMetadata(
        question_id="q3",
        template_id="template3",
        completed_without_errors=False,
        error="Deep judgment failed",
        question_text="Failed question",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=1.0,
        timestamp="2024-01-15T10:35:00",
        answering_replicate=1,
    )

    result = VerificationResult(metadata=metadata, template=None, deep_judgment=None)
    judgment_results = JudgmentResults(results=[result])

    df = judgment_results.to_dataframe()

    # Should have one row with minimal data
    assert len(df) == 1
    assert not df.iloc[0]["completed_without_errors"]
    assert df.iloc[0]["error"] == "Deep judgment failed"
    assert not df.iloc[0]["deep_judgment_performed"]
    assert df.iloc[0]["attribute_name"] is None


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

        template = VerificationResultTemplate(
            raw_llm_response=f"Response {i + 1}",
            parsed_gt_response={"attr1": [f"value{i + 1}"]},
            parsed_llm_response={"attr1": [f"value{i + 1}"]},
            template_verification_performed=True,
            recursion_limit_reached=False,
        )

        deep_judgment = VerificationResultDeepJudgment(
            deep_judgment_enabled=True,
            deep_judgment_performed=True,
            extracted_excerpts={"attr1": [{"text": f"value{i + 1}", "confidence": "high", "similarity_score": 0.9}]},
            deep_judgment_model_calls=1,
            deep_judgment_excerpt_retry_count=0,
        )

        results.append(VerificationResult(metadata=metadata, template=template, deep_judgment=deep_judgment))

    judgment_results = JudgmentResults(results=results)
    df = judgment_results.to_dataframe()

    # Should have 3 rows (one attribute with one excerpt per result)
    assert len(df) == 3
    assert set(df["question_id"].values) == {"q1", "q2", "q3"}
    assert all(df["attribute_name"] == "attr1")


def test_to_dataframe_attribute_match():
    """Test attribute_match logic with matching and non-matching values."""
    metadata = VerificationResultMetadata(
        question_id="q4",
        template_id="template4",
        completed_without_errors=True,
        error=None,
        question_text="Attribute match test",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=2.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    template = VerificationResultTemplate(
        raw_llm_response="Test",
        parsed_gt_response={
            "match_attr": ["value"],
            "nomatch_attr": ["gt_value"],
        },
        parsed_llm_response={
            "match_attr": ["value"],
            "nomatch_attr": ["llm_value"],
        },
        template_verification_performed=True,
        recursion_limit_reached=False,
    )

    deep_judgment = VerificationResultDeepJudgment(
        deep_judgment_enabled=True,
        deep_judgment_performed=True,
        extracted_excerpts={
            "match_attr": [{"text": "value", "confidence": "high", "similarity_score": 0.9}],
            "nomatch_attr": [{"text": "llm_value", "confidence": "medium", "similarity_score": 0.7}],
        },
        deep_judgment_model_calls=2,
        deep_judgment_excerpt_retry_count=0,
    )

    result = VerificationResult(metadata=metadata, template=template, deep_judgment=deep_judgment)
    judgment_results = JudgmentResults(results=[result])

    df = judgment_results.to_dataframe()

    # Check matching attribute
    match_row = df[df["attribute_name"] == "match_attr"].iloc[0]
    assert match_row["attribute_match"]
    assert match_row["gt_attribute_value"] == ["value"]
    assert match_row["llm_attribute_value"] == ["value"]

    # Check non-matching attribute
    nomatch_row = df[df["attribute_name"] == "nomatch_attr"].iloc[0]
    assert not nomatch_row["attribute_match"]
    assert nomatch_row["gt_attribute_value"] == ["gt_value"]
    assert nomatch_row["llm_attribute_value"] == ["llm_value"]


def test_to_dataframe_column_ordering():
    """Test that DataFrame columns follow the correct ordering."""
    metadata = VerificationResultMetadata(
        question_id="q5",
        template_id="template5",
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
        job_id="job-123",
        answering_replicate=1,
    )

    template = VerificationResultTemplate(
        raw_llm_response="Test response",
        parsed_gt_response={"attr": ["value"]},
        parsed_llm_response={"attr": ["value"]},
        template_verification_performed=True,
        recursion_limit_reached=False,
        answering_mcp_servers=["filesystem", "brave-search"],
    )

    deep_judgment = VerificationResultDeepJudgment(
        deep_judgment_enabled=True,
        deep_judgment_performed=True,
        deep_judgment_search_enabled=True,
        extracted_excerpts={
            "attr": [
                {
                    "text": "value",
                    "confidence": "high",
                    "similarity_score": 0.95,
                    "search_results": "Search text",
                    "hallucination_risk": "none",
                    "hallucination_justification": "Strong evidence",
                }
            ]
        },
        attribute_reasoning={"attr": "Clear evidence found"},
        hallucination_risk_assessment={"attr": "none"},
        deep_judgment_stages_completed=["excerpts", "reasoning", "search"],
        deep_judgment_model_calls=3,
        deep_judgment_excerpt_retry_count=1,
    )

    result = VerificationResult(metadata=metadata, template=template, deep_judgment=deep_judgment)
    judgment_results = JudgmentResults(results=[result])

    df = judgment_results.to_dataframe()
    row = df.iloc[0]

    # Verify Status columns
    assert row["completed_without_errors"]
    assert row["error"] is None
    assert not row["recursion_limit_reached"]

    # Verify Identification metadata
    assert row["question_id"] == "q5"
    assert row["template_id"] == "template5"
    assert row["question_text"] == "Column order test"
    assert row["keywords"] == ["test"]
    assert row["replicate"] == 1
    assert row["answering_mcp_servers"] == ["filesystem", "brave-search"]

    # Verify Model Configuration
    assert row["answering_model"] == "gpt-4"
    assert row["parsing_model"] == "gpt-4-mini"
    assert row["answering_system_prompt"] == "Answer questions"
    assert row["parsing_system_prompt"] == "Parse responses"

    # Verify Response Data
    assert row["raw_llm_response"] == "Test response"
    assert row["parsed_gt_response"] == {"attr": ["value"]}
    assert row["parsed_llm_response"] == {"attr": ["value"]}

    # Verify Deep Judgment Configuration
    assert row["deep_judgment_enabled"]
    assert row["deep_judgment_performed"]
    assert row["deep_judgment_search_enabled"]

    # Verify Attribute Information
    assert row["attribute_name"] == "attr"
    assert row["gt_attribute_value"] == ["value"]
    assert row["llm_attribute_value"] == ["value"]
    assert row["attribute_match"]

    # Verify Excerpt Information
    assert row["excerpt_index"] == 0
    assert row["excerpt_text"] == "value"
    assert row["excerpt_confidence"] == "high"
    assert row["excerpt_similarity_score"] == 0.95

    # Verify Search Enhancement
    assert row["excerpt_search_results"] == "Search text"
    assert row["excerpt_hallucination_risk"] == "none"
    assert row["excerpt_hallucination_justification"] == "Strong evidence"

    # Verify Attribute Metadata
    assert row["attribute_reasoning"] == "Clear evidence found"
    assert row["attribute_overall_risk"] == "none"
    assert row["attribute_has_excerpts"]

    # Verify Processing Metrics
    assert row["deep_judgment_model_calls"] == 3
    assert row["deep_judgment_excerpt_retries"] == 1
    assert row["stages_completed"] == ["excerpts", "reasoning", "search"]

    # Verify Execution Metadata
    assert row["execution_time"] == 2.5
    assert row["timestamp"] == "2024-01-15T10:30:00"
    assert row["run_name"] == "test_run"
    assert row["job_id"] == "job-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
