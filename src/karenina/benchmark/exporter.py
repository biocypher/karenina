"""Export functionality for verification execution results.

This module provides functions for exporting verification execution results
(VerificationResult objects) to CSV and JSON formats. These exports focus on
the OUTPUTS of verification runs - what happened when questions were verified.

Key Functions:
- export_verification_results_json(): Export complete verification results as JSON
- export_verification_results_csv(): Export verification results as CSV with rubric columns
- create_export_filename(): Generate filename for exports

Note: This module is distinct from benchmark/core/exports.py, which handles
exporting benchmark STRUCTURE/METADATA (questions, templates, rubrics definition),
not verification execution results.

Usage:
    from karenina.benchmark import export_verification_results_csv, export_verification_results_json

    # Export verification job results
    json_export = export_verification_results_json(job, results)
    csv_export = export_verification_results_csv(job, results, global_rubric)
"""

import csv
import json
import logging
import time
from io import StringIO
from typing import Any, Protocol

from ..schemas.workflow import VerificationJob, VerificationResultSet


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


def export_verification_results_json(job: VerificationJob, results: VerificationResultSet) -> str:
    """
    Export verification results to JSON format with metadata.

    Args:
        job: The verification job
        results: VerificationResultSet containing all verification results

    Returns:
        JSON string with results and metadata
    """
    export_data: dict[str, Any] = {
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
        "results": [],
    }

    # Convert results to serializable format with nested structure
    for result in results:
        # Use Pydantic's model_dump to serialize nested structure, then apply custom serialization to verify_result
        result_dict = result.model_dump(mode="json")

        # Apply custom serialization to verify_result fields if present
        if result.template and result.template.verify_result is not None:
            result_dict["template"]["verify_result"] = _serialize_verification_result(result.template.verify_result)
        if result.template and result.template.verify_granular_result is not None:
            result_dict["template"]["verify_granular_result"] = _serialize_verification_result(
                result.template.verify_granular_result
            )

        export_data["results"].append(result_dict)

    return json.dumps(export_data, indent=2, ensure_ascii=False)


def export_verification_results_csv(
    job: VerificationJob, results: VerificationResultSet, global_rubric: HasTraitNames | None = None
) -> str:
    """
    Export verification results to CSV format with rubric consolidation.

    Args:
        job: The verification job
        results: VerificationResultSet containing all verification results
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
    if not results or len(results) == 0:
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

    # Log export summary
    logger.info("Starting CSV export for %d results", len(results))

    output = StringIO()

    # Collect all unique rubric trait names across all results with validation
    all_rubric_traits: set[str] = set()
    invalid_trait_count = 0
    for result in results:
        if result.rubric:
            # Collect from all trait score dicts (llm, regex, callable, metric)
            for trait_dict in [
                result.rubric.llm_trait_scores,
                result.rubric.regex_trait_scores,
                result.rubric.callable_trait_scores,
                result.rubric.metric_trait_scores,
            ]:
                if trait_dict:
                    for trait_name in trait_dict:
                        if _validate_trait_name(trait_name):
                            all_rubric_traits.add(trait_name)
                        else:
                            invalid_trait_count += 1
                            logger.warning(
                                "Skipping invalid trait name '%s' in question %s",
                                trait_name,
                                result.metadata.question_id,
                            )

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
            # LLM usage tracking fields
            "usage_metadata",
            "agent_metrics",
        ]
    )

    writer: csv.DictWriter[str] = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()

    # Metadata for each row
    export_timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    karenina_version = get_karenina_version()

    # Write data rows
    for result in results:
        # Access fields from nested structure
        metadata = result.metadata
        template = result.template
        rubric = result.rubric
        deep_judgment = result.deep_judgment

        row = {
            # Metadata fields
            "question_id": metadata.question_id,
            "success": metadata.completed_without_errors,  # Header uses 'success', not 'completed_without_errors'
            "error": metadata.error or "",
            "question_text": metadata.question_text,
            "keywords": _safe_json_serialize(metadata.keywords, metadata.question_id, "keywords"),
            "answering_model": metadata.answering_model,
            "parsing_model": metadata.parsing_model,
            "answering_replicate": metadata.answering_replicate or "",
            "parsing_replicate": metadata.parsing_replicate or "",
            "execution_time": metadata.execution_time,
            "timestamp": metadata.timestamp,
            "run_name": metadata.run_name or "",
            # Template fields
            "raw_llm_response": template.raw_llm_response if template else "",
            "parsed_gt_response": _safe_json_serialize(
                template.parsed_gt_response if template else None, metadata.question_id, "parsed_gt_response"
            ),
            "parsed_llm_response": _safe_json_serialize(
                template.parsed_llm_response if template else None, metadata.question_id, "parsed_llm_response"
            ),
            "template_verification_performed": template.template_verification_performed if template else False,
            "verify_result": _serialize_verification_result(template.verify_result if template else None),
            "verify_granular_result": _serialize_verification_result(
                template.verify_granular_result if template else None
            ),
            "answering_system_prompt": metadata.answering_system_prompt or "",
            "parsing_system_prompt": metadata.parsing_system_prompt or "",
            "embedding_check_performed": template.embedding_check_performed if template else False,
            "embedding_similarity_score": template.embedding_similarity_score or "" if template else "",
            "embedding_override_applied": template.embedding_override_applied if template else False,
            "embedding_model_used": template.embedding_model_used or "" if template else "",
            "answering_mcp_servers": _safe_json_serialize(
                template.answering_mcp_servers if template else None, metadata.question_id, "answering_mcp_servers"
            ),
            "usage_metadata": _safe_json_serialize(
                template.usage_metadata if template else None, metadata.question_id, "usage_metadata"
            )
            if template and template.usage_metadata
            else "",
            "agent_metrics": _safe_json_serialize(
                template.agent_metrics if template else None, metadata.question_id, "agent_metrics"
            )
            if template and template.agent_metrics
            else "",
            # Rubric fields
            "rubric_evaluation_performed": rubric.rubric_evaluation_performed if rubric else False,
            "metric_trait_confusion_lists": _safe_json_serialize(
                rubric.metric_trait_confusion_lists if rubric else None,
                metadata.question_id,
                "metric_trait_confusion_lists",
            ),
            "metric_trait_metrics": _safe_json_serialize(
                rubric.metric_trait_scores if rubric else None, metadata.question_id, "metric_trait_metrics"
            ),
            # Deep-judgment fields
            "deep_judgment_enabled": deep_judgment.deep_judgment_enabled if deep_judgment else False,
            "deep_judgment_performed": deep_judgment.deep_judgment_performed if deep_judgment else False,
            "extracted_excerpts": _safe_json_serialize(
                deep_judgment.extracted_excerpts if deep_judgment else None, metadata.question_id, "extracted_excerpts"
            ),
            "attribute_reasoning": _safe_json_serialize(
                deep_judgment.attribute_reasoning if deep_judgment else None,
                metadata.question_id,
                "attribute_reasoning",
            ),
            "deep_judgment_stages_completed": _safe_json_serialize(
                deep_judgment.deep_judgment_stages_completed if deep_judgment else None,
                metadata.question_id,
                "deep_judgment_stages_completed",
            ),
            "deep_judgment_model_calls": deep_judgment.deep_judgment_model_calls if deep_judgment else 0,
            "deep_judgment_excerpt_retry_count": deep_judgment.deep_judgment_excerpt_retry_count
            if deep_judgment
            else 0,
            "attributes_without_excerpts": _safe_json_serialize(
                deep_judgment.attributes_without_excerpts if deep_judgment else None,
                metadata.question_id,
                "attributes_without_excerpts",
            ),
            "deep_judgment_search_enabled": deep_judgment.deep_judgment_search_enabled if deep_judgment else False,
            "hallucination_risk_assessment": _safe_json_serialize(
                deep_judgment.hallucination_risk_assessment if deep_judgment else None,
                metadata.question_id,
                "hallucination_risk_assessment",
            ),
            # Export metadata
            "export_timestamp": export_timestamp,
            "karenina_version": karenina_version,
            "job_id": job.job_id,
        }

        # Add global rubric trait values from all trait score dicts
        if rubric:
            # Merge all trait scores into a unified dict for CSV export
            merged_traits: dict[str, Any] = {}
            if rubric.llm_trait_scores:
                merged_traits.update(rubric.llm_trait_scores)
            if rubric.regex_trait_scores:
                merged_traits.update(rubric.regex_trait_scores)
            if rubric.callable_trait_scores:
                merged_traits.update(rubric.callable_trait_scores)
            if rubric.metric_trait_scores:
                merged_traits.update(rubric.metric_trait_scores)

            # Use pre-computed set for faster membership testing
            for trait in global_traits:
                row[f"rubric_{trait}"] = str(merged_traits.get(trait, ""))
        else:
            # Set all global traits to empty when no rubric data
            for trait in global_traits:
                row[f"rubric_{trait}"] = ""

        # Add question-specific rubrics as JSON (optimized)
        if question_specific_traits_set:
            if rubric and merged_traits:
                # Use dictionary comprehension for better performance
                question_specific_rubrics = {
                    trait: merged_traits[trait] for trait in question_specific_traits_set if trait in merged_traits
                }
            else:
                question_specific_rubrics = {}

            # Safe JSON serialization with error handling
            serialized = _safe_json_serialize(
                question_specific_rubrics, metadata.question_id, "question_specific_rubrics"
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
