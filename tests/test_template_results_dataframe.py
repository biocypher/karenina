"""Tests for TemplateResults.to_dataframe() method."""

import pytest

from karenina.schemas.workflow.template_results import TemplateResults
from karenina.schemas.workflow.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)


def test_to_dataframe_basic_field_explosion():
    """Test that to_dataframe() creates one row per field in parsed responses."""
    # Create a mock VerificationResult with parsed responses
    metadata = VerificationResultMetadata(
        question_id="q1",
        template_id="template1",
        completed_without_errors=True,
        error=None,
        question_text="What is the answer?",
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
        parsing_replicate=None,
    )

    template = VerificationResultTemplate(
        raw_llm_response="The answer is 42 (number)",
        parsed_gt_response={"answer": "42", "unit": "number"},
        parsed_llm_response={"answer": "42", "unit": "integer"},
        template_verification_performed=True,
        verify_result=False,  # Failed because unit doesn't match
        recursion_limit_reached=False,
        answering_mcp_servers=["filesystem", "brave-search"],
    )

    result = VerificationResult(metadata=metadata, template=template)

    # Create TemplateResults with single result
    template_results = TemplateResults(results=[result])

    # Convert to DataFrame
    df = template_results.to_dataframe()

    # Assertions
    assert len(df) == 2, "Should have 2 rows (one per field)"

    # Check field names
    assert set(df["field_name"].values) == {"answer", "unit"}

    # Check answer field row
    answer_row = df[df["field_name"] == "answer"].iloc[0]
    assert answer_row["gt_value"] == "42"
    assert answer_row["llm_value"] == "42"
    assert answer_row["field_match"]  # Values match
    assert answer_row["field_type"] == "str"

    # Check unit field row
    unit_row = df[df["field_name"] == "unit"].iloc[0]
    assert unit_row["gt_value"] == "number"
    assert unit_row["llm_value"] == "integer"
    assert not unit_row["field_match"]  # Values don't match
    assert unit_row["field_type"] == "str"

    # Check Status columns (first columns)
    assert answer_row["completed_without_errors"]
    assert answer_row["error"] is None
    assert not answer_row["recursion_limit_reached"]

    # Check Identification metadata
    assert answer_row["question_id"] == "q1"
    assert answer_row["template_id"] == "template1"
    assert answer_row["question_text"] == "What is the answer?"
    assert answer_row["keywords"] == ["test"]
    assert answer_row["replicate"] == 1
    assert answer_row["answering_mcp_servers"] == ["filesystem", "brave-search"]

    # Check Model Configuration
    assert answer_row["answering_model"] == "gpt-4"
    assert answer_row["parsing_model"] == "gpt-4-mini"

    # Check Template Response
    assert answer_row["raw_llm_response"] == "The answer is 42 (number)"

    # Check Execution Metadata (last columns)
    assert answer_row["execution_time"] == 2.5
    assert answer_row["timestamp"] == "2024-01-15T10:30:00"
    assert answer_row["run_name"] == "test_run"
    assert answer_row["job_id"] == "job-123"


def test_to_dataframe_empty_parsed_responses():
    """Test DataFrame creation when parsed responses are empty."""
    metadata = VerificationResultMetadata(
        question_id="q2",
        template_id="template2",
        completed_without_errors=True,
        error=None,
        question_text="Empty question",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=1.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    template = VerificationResultTemplate(
        raw_llm_response="No structured response",
        parsed_gt_response={},  # Empty
        parsed_llm_response={},  # Empty
        template_verification_performed=True,
        recursion_limit_reached=False,
    )

    result = VerificationResult(metadata=metadata, template=template)
    template_results = TemplateResults(results=[result])

    df = template_results.to_dataframe()

    # Should have one row with None field data
    assert len(df) == 1
    assert df.iloc[0]["field_name"] is None
    assert df.iloc[0]["gt_value"] is None
    assert df.iloc[0]["llm_value"] is None
    assert df.iloc[0]["field_match"] is None


def test_to_dataframe_no_template_data():
    """Test DataFrame creation when result has no template data."""
    metadata = VerificationResultMetadata(
        question_id="q3",
        template_id="template3",
        completed_without_errors=False,
        error="Model API timeout",
        question_text="Failed question",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=30.0,
        timestamp="2024-01-15T10:32:00",
        answering_replicate=1,
    )

    result = VerificationResult(metadata=metadata, template=None)
    template_results = TemplateResults(results=[result])

    df = template_results.to_dataframe()

    # Should have one row with minimal data
    assert len(df) == 1
    assert not df.iloc[0]["completed_without_errors"]
    assert df.iloc[0]["error"] == "Model API timeout"
    assert df.iloc[0]["field_name"] is None
    assert df.iloc[0]["raw_llm_response"] is None


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
            parsed_gt_response={"field1": f"value{i + 1}"},
            parsed_llm_response={"field1": f"value{i + 1}"},
            template_verification_performed=True,
            recursion_limit_reached=False,
        )

        results.append(VerificationResult(metadata=metadata, template=template))

    template_results = TemplateResults(results=results)
    df = template_results.to_dataframe()

    # Should have 3 rows (one field per result)
    assert len(df) == 3
    assert set(df["question_id"].values) == {"q1", "q2", "q3"}
    assert all(df["field_name"] == "field1")


def test_to_dataframe_field_match_comparison():
    """Test field_match logic with various data types."""
    metadata = VerificationResultMetadata(
        question_id="q_types",
        template_id="template_types",
        completed_without_errors=True,
        error=None,
        question_text="Type test",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=2.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    template = VerificationResultTemplate(
        raw_llm_response="Various types",
        parsed_gt_response={
            "str_match": "hello",
            "str_nomatch": "world",
            "int_match": 42,
            "int_nomatch": 100,
            "list_match": [1, 2, 3],
            "list_nomatch": [1, 2],
            "none_both": None,
            "none_gt_only": None,
        },
        parsed_llm_response={
            "str_match": "hello",
            "str_nomatch": "universe",
            "int_match": 42,
            "int_nomatch": 200,
            "list_match": [1, 2, 3],
            "list_nomatch": [1, 2, 3],
            "none_both": None,
            "none_llm_only": None,
        },
        template_verification_performed=True,
        recursion_limit_reached=False,
    )

    result = VerificationResult(metadata=metadata, template=template)
    template_results = TemplateResults(results=[result])

    df = template_results.to_dataframe()

    # Check field matches
    str_match_row = df[df["field_name"] == "str_match"].iloc[0]
    assert str_match_row["field_match"]

    str_nomatch_row = df[df["field_name"] == "str_nomatch"].iloc[0]
    assert not str_nomatch_row["field_match"]

    int_match_row = df[df["field_name"] == "int_match"].iloc[0]
    assert int_match_row["field_match"]

    list_match_row = df[df["field_name"] == "list_match"].iloc[0]
    assert list_match_row["field_match"]

    list_nomatch_row = df[df["field_name"] == "list_nomatch"].iloc[0]
    assert not list_nomatch_row["field_match"]

    # Both None should match
    none_both_row = df[df["field_name"] == "none_both"].iloc[0]
    assert none_both_row["field_match"]

    # One None should not match
    none_gt_only_row = df[df["field_name"] == "none_gt_only"].iloc[0]
    assert not none_gt_only_row["field_match"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
