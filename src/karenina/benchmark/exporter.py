"""Export functionality for verification results."""

import csv
import json
import time
from io import StringIO
from typing import Any

from .models import VerificationJob, VerificationResult


def get_karenina_version() -> str:
    """Get the current Karenina version."""
    try:
        import karenina

        return getattr(karenina, "__version__", "unknown")
    except ImportError:
        return "unknown"


def export_verification_results_json(job: VerificationJob, results: dict[str, VerificationResult]) -> str:
    """
    Export verification results to JSON format with metadata.

    Args:
        job: The verification job
        results: Dictionary of verification results

    Returns:
        JSON string with results and metadata
    """
    export_data = {
        "metadata": {
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "karenina_version": get_karenina_version(),
            "job_id": job.job_id,
            "verification_config": {
                "answering_model": {
                    "provider": job.config.answering_model_provider,
                    "name": job.config.answering_model_name,
                    "temperature": job.config.answering_temperature,
                    "interface": job.config.answering_interface,
                },
                "parsing_model": {
                    "provider": job.config.parsing_model_provider,
                    "name": job.config.parsing_model_name,
                    "temperature": job.config.parsing_temperature,
                    "interface": job.config.parsing_interface,
                },
            },
            "job_summary": {
                "total_questions": job.total_questions,
                "successful_count": job.successful_count,
                "failed_count": job.failed_count,
                "start_time": job.start_time,
                "end_time": job.end_time,
                "total_duration": job.end_time - job.start_time if job.end_time and job.start_time else None,
            },
        },
        "results": {},
    }

    # Convert results to serializable format
    for question_id, result in results.items():
        export_data["results"][question_id] = {
            "question_id": result.question_id,
            "success": result.success,
            "error": result.error,
            "question_text": result.question_text,
            "raw_llm_response": result.raw_llm_response,
            "parsed_response": result.parsed_response,
            "verify_result": _serialize_verification_result(result.verify_result),
            "verify_granular_result": _serialize_verification_result(result.verify_granular_result),
            "verify_rubric": result.verify_rubric,
            "answering_model": result.answering_model,
            "parsing_model": result.parsing_model,
            "answering_replicate": result.answering_replicate,
            "parsing_replicate": result.parsing_replicate,
            "execution_time": result.execution_time,
            "timestamp": result.timestamp,
            "answering_system_prompt": result.answering_system_prompt,
            "parsing_system_prompt": result.parsing_system_prompt,
            "run_name": result.run_name,
            "job_id": result.job_id,
        }

    return json.dumps(export_data, indent=2, ensure_ascii=False)


def export_verification_results_csv(job: VerificationJob, results: dict[str, VerificationResult]) -> str:
    """
    Export verification results to CSV format.

    Args:
        job: The verification job
        results: Dictionary of verification results

    Returns:
        CSV string with results
    """
    output = StringIO()

    # Collect all unique rubric trait names across all results
    all_rubric_traits: set[str] = set()
    for result in results.values():
        if result.verify_rubric:
            all_rubric_traits.update(result.verify_rubric.keys())

    # Sort trait names for consistent column ordering
    sorted_traits = sorted(all_rubric_traits)

    # Define CSV headers with all result fields + dynamic rubric columns
    headers = [
        "question_id",
        "success",
        "error",
        "question_text",
        "raw_llm_response",
        "parsed_response",
        "verify_result",
        "verify_granular_result",
    ]

    # Add rubric trait columns (prefixed with 'rubric_')
    headers.extend([f"rubric_{trait}" for trait in sorted_traits])

    # Add remaining standard columns
    headers.extend(
        [
            "answering_model",
            "parsing_model",
            "answering_replicate",
            "parsing_replicate",
            "execution_time",
            "timestamp",
            "answering_system_prompt",
            "parsing_system_prompt",
            "run_name",
            "export_timestamp",
            "karenina_version",
            "job_id",
        ]
    )

    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()

    # Metadata for each row
    export_timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    karenina_version = get_karenina_version()

    # Write data rows
    for _question_id, result in results.items():
        row = {
            "question_id": result.question_id,
            "success": result.success,
            "error": result.error or "",
            "question_text": result.question_text,
            "raw_llm_response": result.raw_llm_response,
            "parsed_response": json.dumps(result.parsed_response) if result.parsed_response else "",
            "verify_result": _serialize_verification_result(result.verify_result),
            "verify_granular_result": _serialize_verification_result(result.verify_granular_result),
            "answering_model": result.answering_model,
            "parsing_model": result.parsing_model,
            "answering_replicate": result.answering_replicate or "",
            "parsing_replicate": result.parsing_replicate or "",
            "execution_time": result.execution_time,
            "timestamp": result.timestamp,
            "answering_system_prompt": result.answering_system_prompt or "",
            "parsing_system_prompt": result.parsing_system_prompt or "",
            "run_name": result.run_name or "",
            "export_timestamp": export_timestamp,
            "karenina_version": karenina_version,
            "job_id": job.job_id,
        }

        # Add rubric trait values
        for trait in sorted_traits:
            rubric_value = ""
            if result.verify_rubric and trait in result.verify_rubric:
                rubric_value = str(result.verify_rubric[trait])
            row[f"rubric_{trait}"] = rubric_value

        writer.writerow(row)

    return output.getvalue()


def _serialize_verification_result(result: Any) -> str:
    """
    Serialize verification result to string for export.

    Args:
        result: The verification result (can be any type)

    Returns:
        String representation of the result
    """
    if result is None:
        return ""

    try:
        # Try to serialize as JSON if it's a complex object
        if isinstance(result, dict | list):
            return json.dumps(result)
        elif hasattr(result, "model_dump"):
            # Pydantic model
            return json.dumps(result.model_dump())
        elif hasattr(result, "dict"):
            # Pydantic model (older version)
            return json.dumps(result.dict())
        else:
            # Simple types or string representation
            return str(result)
    except Exception:
        # Fallback to string representation
        return str(result)


def create_export_filename(job: VerificationJob, format: str) -> str:
    """
    Create a filename for the export based on job info and timestamp.

    Args:
        job: The verification job
        format: Export format ('json' or 'csv')

    Returns:
        Suggested filename
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    answering_model = f"{job.config.answering_model_provider}_{job.config.answering_model_name}".replace("/", "_")
    parsing_model = f"{job.config.parsing_model_provider}_{job.config.parsing_model_name}".replace("/", "_")

    return f"verification_results_{timestamp}_{answering_model}_to_{parsing_model}.{format}"
