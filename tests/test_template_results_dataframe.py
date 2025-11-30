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


def test_to_regex_dataframe_basic():
    """Test regex DataFrame creation with pattern explosion."""
    metadata = VerificationResultMetadata(
        question_id="q_regex",
        template_id="template_regex",
        completed_without_errors=True,
        error=None,
        question_text="Regex test",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=2.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    template = VerificationResultTemplate(
        raw_llm_response="The answer is 42 units",
        parsed_gt_response={"value": "42"},
        parsed_llm_response={"value": "42"},
        template_verification_performed=True,
        recursion_limit_reached=False,
        regex_validations_performed=True,
        regex_validation_results={
            "number_pattern": True,
            "unit_pattern": True,
            "date_pattern": False,
        },
        regex_validation_details={
            "number_pattern": {
                "pattern": r"\d+",
                "match_start": 14,
                "match_end": 16,
                "full_match": "42",
            },
            "unit_pattern": {
                "pattern": r"units?",
                "match_start": 17,
                "match_end": 22,
                "full_match": "units",
            },
            "date_pattern": {
                "pattern": r"\d{4}-\d{2}-\d{2}",
            },
        },
        regex_extraction_results={
            "number_pattern": "42",
            "unit_pattern": "units",
            "date_pattern": None,
        },
    )

    result = VerificationResult(metadata=metadata, template=template)
    template_results = TemplateResults(results=[result])

    df = template_results.to_regex_dataframe()

    # Should have 3 rows (one per pattern)
    assert len(df) == 3

    # Check matched patterns
    matched_df = df[df["matched"]]
    assert len(matched_df) == 2
    assert set(matched_df["pattern_name"].values) == {"number_pattern", "unit_pattern"}

    # Check number pattern row
    number_row = df[df["pattern_name"] == "number_pattern"].iloc[0]
    assert number_row["matched"]
    assert number_row["pattern_regex"] == r"\d+"
    assert number_row["extracted_value"] == "42"
    assert number_row["match_start"] == 14
    assert number_row["match_end"] == 16
    assert number_row["full_match"] == "42"

    # Check failed pattern
    date_row = df[df["pattern_name"] == "date_pattern"].iloc[0]
    assert not date_row["matched"]
    assert date_row["extracted_value"] is None


def test_to_regex_dataframe_no_patterns():
    """Test regex DataFrame when no patterns are tested."""
    metadata = VerificationResultMetadata(
        question_id="q_no_regex",
        template_id="template_no_regex",
        completed_without_errors=True,
        error=None,
        question_text="No regex test",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=1.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    template = VerificationResultTemplate(
        raw_llm_response="Test response",
        parsed_gt_response={},
        parsed_llm_response={},
        template_verification_performed=True,
        recursion_limit_reached=False,
        regex_validations_performed=False,
    )

    result = VerificationResult(metadata=metadata, template=template)
    template_results = TemplateResults(results=[result])

    df = template_results.to_regex_dataframe()

    # Should be empty (no patterns tested)
    assert len(df) == 0


def test_to_usage_dataframe_exploded():
    """Test usage DataFrame with stage explosion (default)."""
    metadata = VerificationResultMetadata(
        question_id="q_usage",
        template_id="template_usage",
        completed_without_errors=True,
        error=None,
        question_text="Usage test",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=5.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    template = VerificationResultTemplate(
        raw_llm_response="Test response",
        parsed_gt_response={"value": "test"},
        parsed_llm_response={"value": "test"},
        template_verification_performed=True,
        recursion_limit_reached=False,
        usage_metadata={
            "answer_generation": {
                "input_tokens": 150,
                "output_tokens": 50,
                "total_tokens": 200,
                "model": "gpt-4.1-mini",
                "input_token_details": {"audio": 0, "cache_read": 10},
                "output_token_details": {"audio": 0, "reasoning": 5},
            },
            "parsing": {
                "input_tokens": 80,
                "output_tokens": 20,
                "total_tokens": 100,
                "model": "gpt-4.1-mini",
            },
            "total": {
                "input_tokens": 230,
                "output_tokens": 70,
                "total_tokens": 300,
            },
        },
        agent_metrics={
            "iterations": 3,
            "tool_calls": 5,
            "tools_used": ["mcp__brave_search", "mcp__filesystem"],
            "suspect_failed_tool_calls": 1,
        },
    )

    result = VerificationResult(metadata=metadata, template=template)
    template_results = TemplateResults(results=[result])

    df = template_results.to_usage_dataframe()

    # Should have 2 rows (answer_generation + parsing, excluding total)
    assert len(df) == 2
    assert set(df["usage_stage"].values) == {"answer_generation", "parsing"}

    # Check answer_generation row
    answer_row = df[df["usage_stage"] == "answer_generation"].iloc[0]
    assert answer_row["input_tokens"] == 150
    assert answer_row["output_tokens"] == 50
    assert answer_row["total_tokens"] == 200
    assert answer_row["model_used"] == "gpt-4.1-mini"
    assert answer_row["input_cache_read_tokens"] == 10
    assert answer_row["output_reasoning_tokens"] == 5
    assert answer_row["agent_iterations"] == 3
    assert answer_row["agent_tool_calls"] == 5

    # Check parsing row (no agent metrics in details)
    parsing_row = df[df["usage_stage"] == "parsing"].iloc[0]
    assert parsing_row["input_tokens"] == 80
    assert parsing_row["output_tokens"] == 20
    assert parsing_row["total_tokens"] == 100


def test_to_usage_dataframe_totals_only():
    """Test usage DataFrame with totals_only=True."""
    metadata = VerificationResultMetadata(
        question_id="q_totals",
        template_id="template_totals",
        completed_without_errors=True,
        error=None,
        question_text="Totals test",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=5.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    template = VerificationResultTemplate(
        raw_llm_response="Test response",
        parsed_gt_response={"value": "test"},
        parsed_llm_response={"value": "test"},
        template_verification_performed=True,
        recursion_limit_reached=False,
        usage_metadata={
            "answer_generation": {"input_tokens": 150, "output_tokens": 50, "total_tokens": 200},
            "parsing": {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
            "total": {"input_tokens": 230, "output_tokens": 70, "total_tokens": 300},
        },
    )

    result = VerificationResult(metadata=metadata, template=template)
    template_results = TemplateResults(results=[result])

    df = template_results.to_usage_dataframe(totals_only=True)

    # Should have 1 row (total only)
    assert len(df) == 1
    assert df.iloc[0]["usage_stage"] is None  # usage_stage is None for totals_only
    assert df.iloc[0]["input_tokens"] == 230
    assert df.iloc[0]["output_tokens"] == 70
    assert df.iloc[0]["total_tokens"] == 300


def test_to_usage_dataframe_no_usage_data():
    """Test usage DataFrame when no usage data exists."""
    metadata = VerificationResultMetadata(
        question_id="q_no_usage",
        template_id="template_no_usage",
        completed_without_errors=True,
        error=None,
        question_text="No usage test",
        keywords=None,
        answering_model="gpt-4",
        parsing_model="gpt-4-mini",
        execution_time=1.0,
        timestamp="2024-01-15T10:30:00",
        answering_replicate=1,
    )

    template = VerificationResultTemplate(
        raw_llm_response="Test response",
        parsed_gt_response={},
        parsed_llm_response={},
        template_verification_performed=True,
        recursion_limit_reached=False,
        usage_metadata=None,
    )

    result = VerificationResult(metadata=metadata, template=template)
    template_results = TemplateResults(results=[result])

    df = template_results.to_usage_dataframe()

    # Should be empty (no usage data)
    assert len(df) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
