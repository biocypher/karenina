"""Tests for deep-judgment fields in export functionality.

This module tests that deep-judgment metadata is properly included in
JSON and CSV exports.
"""

import csv
import json
from datetime import datetime
from io import StringIO

from karenina.benchmark.exporter import export_verification_results_csv, export_verification_results_json
from karenina.benchmark.models import ModelConfig, VerificationConfig, VerificationJob, VerificationResult


def test_deep_judgment_in_json_export():
    """Test that deep-judgment fields are included in JSON export."""
    # Create a verification result with deep-judgment data
    result = VerificationResult(
        question_id="a" * 32,
        template_id="t" * 32,
        success=True,
        verify_result=True,
        question_text="What is the drug target?",
        raw_llm_response="The drug targets BCL-2 protein.",
        answering_model="openai/gpt-4.1-mini",
        parsing_model="openai/gpt-4.1-mini",
        parsed_gt_response={"drug_target": "BCL-2"},
        parsed_llm_response={"drug_target": "BCL-2"},
        execution_time=1.5,
        timestamp=datetime.now().isoformat(),
        answering_system_prompt="You are helpful.",
        parsing_system_prompt="Parse this.",
        # Deep-judgment fields
        deep_judgment_enabled=True,
        deep_judgment_performed=True,
        extracted_excerpts={"drug_target": [{"text": "targets BCL-2", "confidence": "high", "similarity_score": 0.95}]},
        attribute_reasoning={"drug_target": "The excerpt clearly states BCL-2 as the target."},
        deep_judgment_stages_completed=["excerpts", "reasoning", "parameters"],
        deep_judgment_model_calls=3,
        deep_judgment_excerpt_retry_count=0,
        attributes_without_excerpts=[],
    )

    # Create a mock job with minimal config
    parsing_model = ModelConfig(
        id="test-parser",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.0,
        system_prompt="Parse this.",
    )
    config = VerificationConfig(
        answering_models=[],
        parsing_models=[parsing_model],
        parsing_only=True,
    )
    job = VerificationJob(
        job_id="test_job",
        run_name="Test Run",
        status="completed",
        config=config,
        total_questions=1,
        successful_count=1,
        failed_count=0,
        start_time=1234567890.0,
        end_time=1234567900.0,
    )

    # Export to JSON
    json_output = export_verification_results_json(job, {"a" * 32: result})
    export_data = json.loads(json_output)

    # Verify deep-judgment fields are present
    result_data = export_data["results"]["a" * 32]
    assert "deep_judgment_enabled" in result_data
    assert result_data["deep_judgment_enabled"] is True
    assert result_data["deep_judgment_performed"] is True
    assert result_data["extracted_excerpts"] == result.extracted_excerpts
    assert result_data["attribute_reasoning"] == result.attribute_reasoning
    assert result_data["deep_judgment_stages_completed"] == ["excerpts", "reasoning", "parameters"]
    assert result_data["deep_judgment_model_calls"] == 3
    assert result_data["deep_judgment_excerpt_retry_count"] == 0
    assert result_data["attributes_without_excerpts"] == []


def test_deep_judgment_in_csv_export():
    """Test that deep-judgment fields are included in CSV export."""
    # Create a verification result with deep-judgment data
    result = VerificationResult(
        question_id="b" * 32,
        template_id="t" * 32,
        success=True,
        verify_result=True,
        question_text="What is the mechanism?",
        raw_llm_response="The mechanism is inhibition.",
        answering_model="openai/gpt-4.1-mini",
        parsing_model="openai/gpt-4.1-mini",
        parsed_gt_response={"mechanism": "inhibition"},
        parsed_llm_response={"mechanism": "inhibition"},
        execution_time=1.2,
        timestamp=datetime.now().isoformat(),
        answering_system_prompt="You are helpful.",
        parsing_system_prompt="Parse this.",
        # Deep-judgment fields
        deep_judgment_enabled=True,
        deep_judgment_performed=True,
        extracted_excerpts={
            "mechanism": [{"text": "mechanism is inhibition", "confidence": "high", "similarity_score": 0.92}]
        },
        attribute_reasoning={"mechanism": "Direct statement about the mechanism."},
        deep_judgment_stages_completed=["excerpts", "reasoning", "parameters"],
        deep_judgment_model_calls=3,
        deep_judgment_excerpt_retry_count=1,
        attributes_without_excerpts=[],
    )

    # Create a mock job with minimal config
    parsing_model = ModelConfig(
        id="test-parser",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.0,
        system_prompt="Parse this.",
    )
    config = VerificationConfig(
        answering_models=[],
        parsing_models=[parsing_model],
        parsing_only=True,
    )
    job = VerificationJob(
        job_id="test_job_2",
        run_name="Test Run 2",
        status="completed",
        config=config,
        total_questions=1,
        successful_count=1,
        failed_count=0,
        start_time=1234567890.0,
        end_time=1234567900.0,
    )

    # Export to CSV
    csv_output = export_verification_results_csv(job, {"b" * 32: result})

    # Parse CSV to verify deep-judgment fields
    csv_reader = csv.DictReader(StringIO(csv_output))
    rows = list(csv_reader)

    assert len(rows) == 1
    row = rows[0]

    # Check that deep-judgment columns exist and have correct values
    assert "deep_judgment_enabled" in row
    assert row["deep_judgment_enabled"] == "True"
    assert row["deep_judgment_performed"] == "True"
    assert "extracted_excerpts" in row
    assert "mechanism" in row["extracted_excerpts"]
    assert "attribute_reasoning" in row
    assert row["deep_judgment_model_calls"] == "3"
    assert row["deep_judgment_excerpt_retry_count"] == "1"
    assert "attributes_without_excerpts" in row


def test_deep_judgment_with_empty_excerpts_export():
    """Test that empty excerpt lists (refusal scenarios) are properly exported."""
    # Create a result with empty excerpts
    result = VerificationResult(
        question_id="c" * 32,
        template_id="t" * 32,
        success=True,
        verify_result=False,
        question_text="What is confidential?",
        raw_llm_response="I cannot provide that information.",
        answering_model="openai/gpt-4.1-mini",
        parsing_model="openai/gpt-4.1-mini",
        parsed_gt_response=None,
        parsed_llm_response=None,
        execution_time=0.8,
        timestamp=datetime.now().isoformat(),
        answering_system_prompt="You are helpful.",
        parsing_system_prompt="Parse this.",
        # Deep-judgment with empty excerpts
        deep_judgment_enabled=True,
        deep_judgment_performed=True,
        extracted_excerpts={"confidential_data": []},  # Empty list
        attribute_reasoning={"confidential_data": "No corroborating excerpts found."},
        deep_judgment_stages_completed=["excerpts", "reasoning", "parameters"],
        deep_judgment_model_calls=3,
        deep_judgment_excerpt_retry_count=2,
        attributes_without_excerpts=["confidential_data"],
    )

    # Create a mock job with minimal config
    parsing_model = ModelConfig(
        id="test-parser",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.0,
        system_prompt="Parse this.",
    )
    config = VerificationConfig(
        answering_models=[],
        parsing_models=[parsing_model],
        parsing_only=True,
    )
    job = VerificationJob(
        job_id="test_job_3",
        run_name="Test Run 3",
        status="completed",
        config=config,
        total_questions=1,
        successful_count=1,
        failed_count=0,
        start_time=1234567890.0,
        end_time=1234567900.0,
    )

    # Export to JSON
    json_output = export_verification_results_json(job, {"c" * 32: result})
    export_data = json.loads(json_output)

    result_data = export_data["results"]["c" * 32]
    assert result_data["extracted_excerpts"] == {"confidential_data": []}
    assert result_data["attributes_without_excerpts"] == ["confidential_data"]
    assert result_data["deep_judgment_excerpt_retry_count"] == 2

    # Export to CSV
    csv_output = export_verification_results_csv(job, {"c" * 32: result})
    csv_reader = csv.DictReader(StringIO(csv_output))
    rows = list(csv_reader)

    row = rows[0]
    assert "confidential_data" in row["attributes_without_excerpts"]
    assert row["deep_judgment_excerpt_retry_count"] == "2"
