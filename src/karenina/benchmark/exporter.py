"""Export functionality for verification results."""

import csv
import json
import logging
import time
from io import StringIO
from typing import Any, Protocol

from .models import VerificationJob, VerificationResult


class HasTraitNames(Protocol):
    """Protocol for objects that can provide trait names."""

    def get_trait_names(self) -> list[str]:
        """Get list of trait names from the rubric."""
        ...


# Set up logger
logger = logging.getLogger(__name__)


def _validate_trait_name(trait_name: str) -> bool:
    """Validate that a trait name is safe for use in CSV headers and JSON.

    Args:
        trait_name: The trait name to validate

    Returns:
        True if the trait name is valid, False otherwise
    """
    if not trait_name or not isinstance(trait_name, str):
        return False

    # Check for reasonable length (CSV headers should be reasonable)
    if len(trait_name) > 255:
        logger.warning("Trait name too long (>255 chars): %s...", trait_name[:50])
        return False

    # Check for problematic characters that might cause CSV parsing issues
    problematic_chars = ["\n", "\r", "\0"]
    if any(char in trait_name for char in problematic_chars):
        logger.warning("Trait name contains problematic characters: %s", repr(trait_name))
        return False

    return True


def _safe_json_serialize(data: Any, question_id: str, field_name: str) -> str:
    """Safely serialize data to JSON string with error handling.

    Args:
        data: The data to serialize
        question_id: The question ID for logging context
        field_name: The field name for logging context

    Returns:
        JSON string or fallback representation
    """
    if not data:
        return ""

    try:
        return json.dumps(data)
    except (TypeError, ValueError) as e:
        logger.warning(
            "Failed to serialize %s for question %s (%s: %s). Using string representation instead.",
            field_name,
            question_id,
            type(e).__name__,
            e,
        )
        try:
            return str(data)
        except Exception as str_error:
            logger.error(
                "Critical: Failed to convert %s to string for question %s: %s", field_name, question_id, str_error
            )
            return f"<serialization_failed:{type(data).__name__}>"


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
            "completed_without_errors": result.completed_without_errors,
            "error": result.error,
            "question_text": result.question_text,
            "raw_llm_response": result.raw_llm_response,
            "parsed_gt_response": result.parsed_gt_response,
            "parsed_llm_response": result.parsed_llm_response,
            "template_verification_performed": result.template_verification_performed,
            "verify_result": _serialize_verification_result(result.verify_result),
            "verify_granular_result": _serialize_verification_result(result.verify_granular_result),
            "rubric_evaluation_performed": result.rubric_evaluation_performed,
            "verify_rubric": result.verify_rubric,
            "keywords": result.keywords,
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
            # Embedding check fields
            "embedding_check_performed": result.embedding_check_performed,
            "embedding_similarity_score": result.embedding_similarity_score,
            "embedding_override_applied": result.embedding_override_applied,
            "embedding_model_used": result.embedding_model_used,
            # MCP server fields
            "answering_mcp_servers": result.answering_mcp_servers,
            # Deep-judgment fields
            "deep_judgment_enabled": result.deep_judgment_enabled,
            "deep_judgment_performed": result.deep_judgment_performed,
            "extracted_excerpts": result.extracted_excerpts,
            "attribute_reasoning": result.attribute_reasoning,
            "deep_judgment_stages_completed": result.deep_judgment_stages_completed,
            "deep_judgment_model_calls": result.deep_judgment_model_calls,
            "deep_judgment_excerpt_retry_count": result.deep_judgment_excerpt_retry_count,
            "attributes_without_excerpts": result.attributes_without_excerpts,
            # Search-enhanced deep-judgment fields
            "deep_judgment_search_enabled": result.deep_judgment_search_enabled,
            "hallucination_risk_assessment": result.hallucination_risk_assessment,
            # Metric trait fields
            "metric_trait_confusion_lists": result.metric_trait_confusion_lists,
            "metric_trait_metrics": result.metric_trait_metrics,
        }

    return json.dumps(export_data, indent=2, ensure_ascii=False)


def export_verification_results_csv(
    job: VerificationJob, results: dict[str, VerificationResult], global_rubric: HasTraitNames | None = None
) -> str:
    """
    Export verification results to CSV format with rubric consolidation.

    Args:
        job: The verification job
        results: Dictionary of verification results
        global_rubric: Optional global rubric object that implements HasTraitNames protocol
                      for distinguishing global vs question-specific traits. If None,
                      all rubric traits will be consolidated into question_specific_rubrics.

    Returns:
        CSV string with results. Global rubric traits appear as dedicated columns
        (rubric_TraitName), while question-specific traits are consolidated into
        a single JSON column (question_specific_rubrics).

    Note:
        The function gracefully handles errors in trait name extraction and JSON
        serialization, logging warnings and continuing with fallback values.
    """
    # Input validation
    if not results:
        logger.warning("No results provided for CSV export. Generating empty CSV.")
        # Return minimal CSV with headers only
        output = StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerow(
            [
                "question_id",
                "success",
                "error",
                "question_text",
                "raw_llm_response",
                "keywords",
                "export_timestamp",
                "karenina_version",
                "job_id",
            ]
        )
        return output.getvalue()

    if not isinstance(results, dict):
        raise ValueError(f"Results must be a dictionary, got {type(results).__name__}")

    # Log export summary
    logger.info("Starting CSV export for %d results", len(results))

    output = StringIO()

    # Collect all unique rubric trait names across all results with validation
    all_rubric_traits: set[str] = set()
    invalid_trait_count = 0
    for result in results.values():
        if result.verify_rubric:
            for trait_name in result.verify_rubric:
                if _validate_trait_name(trait_name):
                    all_rubric_traits.add(trait_name)
                else:
                    invalid_trait_count += 1
                    logger.warning("Skipping invalid trait name '%s' in question %s", trait_name, result.question_id)

    if invalid_trait_count > 0:
        logger.info("Skipped %d invalid trait names during CSV export", invalid_trait_count)

    # Determine global vs question-specific rubrics
    global_trait_names: set[str] = set()
    if global_rubric:
        try:
            if hasattr(global_rubric, "get_trait_names") and callable(global_rubric.get_trait_names):
                trait_names = global_rubric.get_trait_names()
                if isinstance(trait_names, list):
                    # Validate each trait name from global rubric
                    valid_global_traits = []
                    for trait_name in trait_names:
                        if _validate_trait_name(trait_name):
                            valid_global_traits.append(trait_name)
                        else:
                            logger.warning("Skipping invalid global trait name '%s' from global_rubric", trait_name)
                    global_trait_names = set(valid_global_traits)

                    if len(valid_global_traits) != len(trait_names):
                        logger.info(
                            "Global rubric had %d traits, %d were valid for CSV export",
                            len(trait_names),
                            len(valid_global_traits),
                        )
                else:
                    logger.warning(
                        "Global rubric get_trait_names() returned %s instead of list. "
                        "All rubric traits will be treated as question-specific.",
                        type(trait_names).__name__,
                    )
            else:
                logger.warning(
                    "Global rubric object does not have a callable get_trait_names method. "
                    "All rubric traits will be treated as question-specific."
                )
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(
                "Error accessing global rubric trait names (%s: %s). "
                "All rubric traits will be treated as question-specific.",
                type(e).__name__,
                e,
            )
            # Continue with empty set - graceful degradation

    # Separate traits into global and question-specific (with performance optimization)
    global_traits = sorted(all_rubric_traits.intersection(global_trait_names))
    question_specific_traits = sorted(all_rubric_traits - global_trait_names)

    # Pre-compute set for faster lookups during row processing
    question_specific_traits_set = set(question_specific_traits)

    # Log export configuration
    logger.debug(
        "CSV export configuration: %d global traits, %d question-specific traits, %d total results",
        len(global_traits),
        len(question_specific_traits),
        len(results),
    )

    # Define CSV headers with all result fields + dynamic rubric columns
    headers = [
        "question_id",
        "success",
        "error",
        "question_text",
        "raw_llm_response",
        "parsed_gt_response",
        "parsed_llm_response",
        "template_verification_performed",
        "verify_result",
        "verify_granular_result",
        "rubric_evaluation_performed",
        "keywords",
    ]

    # Add global rubric trait columns (prefixed with 'rubric_')
    headers.extend([f"rubric_{trait}" for trait in global_traits])

    # Add single column for question-specific rubrics
    if question_specific_traits:
        headers.append("question_specific_rubrics")

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
            # Embedding check fields
            "embedding_check_performed",
            "embedding_similarity_score",
            "embedding_override_applied",
            "embedding_model_used",
            # MCP server fields
            "answering_mcp_servers",
            # Deep-judgment fields
            "deep_judgment_enabled",
            "deep_judgment_performed",
            "extracted_excerpts",
            "attribute_reasoning",
            "deep_judgment_stages_completed",
            "deep_judgment_model_calls",
            "deep_judgment_excerpt_retry_count",
            "attributes_without_excerpts",
            # Search-enhanced deep-judgment fields
            "deep_judgment_search_enabled",
            "hallucination_risk_assessment",
            # Metric trait fields
            "metric_trait_confusion_lists",
            "metric_trait_metrics",
        ]
    )

    writer: csv.DictWriter[str] = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()

    # Metadata for each row
    export_timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    karenina_version = get_karenina_version()

    # Write data rows
    for _question_id, result in results.items():
        row = {
            "question_id": result.question_id,
            "completed_without_errors": result.completed_without_errors,
            "error": result.error or "",
            "question_text": result.question_text,
            "raw_llm_response": result.raw_llm_response,
            "parsed_gt_response": _safe_json_serialize(
                result.parsed_gt_response, result.question_id, "parsed_gt_response"
            ),
            "parsed_llm_response": _safe_json_serialize(
                result.parsed_llm_response, result.question_id, "parsed_llm_response"
            ),
            "template_verification_performed": result.template_verification_performed,
            "verify_result": _serialize_verification_result(result.verify_result),
            "verify_granular_result": _serialize_verification_result(result.verify_granular_result),
            "rubric_evaluation_performed": result.rubric_evaluation_performed,
            "keywords": _safe_json_serialize(result.keywords, result.question_id, "keywords"),
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
            # Embedding check fields
            "embedding_check_performed": result.embedding_check_performed,
            "embedding_similarity_score": result.embedding_similarity_score or "",
            "embedding_override_applied": result.embedding_override_applied,
            "embedding_model_used": result.embedding_model_used or "",
            # MCP server fields
            "answering_mcp_servers": _safe_json_serialize(
                result.answering_mcp_servers, result.question_id, "answering_mcp_servers"
            ),
            # Deep-judgment fields
            "deep_judgment_enabled": result.deep_judgment_enabled,
            "deep_judgment_performed": result.deep_judgment_performed,
            "extracted_excerpts": _safe_json_serialize(
                result.extracted_excerpts, result.question_id, "extracted_excerpts"
            ),
            "attribute_reasoning": _safe_json_serialize(
                result.attribute_reasoning, result.question_id, "attribute_reasoning"
            ),
            "deep_judgment_stages_completed": _safe_json_serialize(
                result.deep_judgment_stages_completed, result.question_id, "deep_judgment_stages_completed"
            ),
            "deep_judgment_model_calls": result.deep_judgment_model_calls,
            "deep_judgment_excerpt_retry_count": result.deep_judgment_excerpt_retry_count,
            "attributes_without_excerpts": _safe_json_serialize(
                result.attributes_without_excerpts, result.question_id, "attributes_without_excerpts"
            ),
            # Search-enhanced deep-judgment fields
            "deep_judgment_search_enabled": result.deep_judgment_search_enabled,
            "hallucination_risk_assessment": _safe_json_serialize(
                result.hallucination_risk_assessment, result.question_id, "hallucination_risk_assessment"
            ),
            # Metric trait fields
            "metric_trait_confusion_lists": _safe_json_serialize(
                result.metric_trait_confusion_lists, result.question_id, "metric_trait_confusion_lists"
            ),
            "metric_trait_metrics": _safe_json_serialize(
                result.metric_trait_metrics, result.question_id, "metric_trait_metrics"
            ),
        }

        # Add global rubric trait values (optimized with dictionary comprehension)
        if result.verify_rubric:
            # Use pre-computed set for faster membership testing
            for trait in global_traits:
                row[f"rubric_{trait}"] = str(result.verify_rubric.get(trait, ""))
        else:
            # Set all global traits to empty when no rubric data
            for trait in global_traits:
                row[f"rubric_{trait}"] = ""

        # Add question-specific rubrics as JSON (optimized)
        if question_specific_traits_set:
            if result.verify_rubric:
                # Use dictionary comprehension for better performance
                question_specific_rubrics = {
                    trait: result.verify_rubric[trait]
                    for trait in question_specific_traits_set
                    if trait in result.verify_rubric
                }
            else:
                question_specific_rubrics = {}

            # Safe JSON serialization with error handling
            serialized = _safe_json_serialize(
                question_specific_rubrics, result.question_id, "question_specific_rubrics"
            )
            row["question_specific_rubrics"] = serialized if serialized else "{}"

        writer.writerow(row)

    # Log completion summary
    result_count = len(results)
    logger.info("CSV export completed successfully for %d results", result_count)

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
