"""Results I/O manager for verification result export and import."""

import csv
import json
import logging
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...schemas.entities import Rubric

from ...schemas.verification import VerificationResult

logger = logging.getLogger(__name__)


class ResultsIOManager:
    """Manager for verification result export and import operations.

    Handles CSV and JSON serialization/deserialization logic for ResultsManager.
    """

    @staticmethod
    def export_to_json(results: dict[str, VerificationResult]) -> str:
        """
        Export results to JSON format matching frontend format.

        Args:
            results: Dictionary of verification results

        Returns:
            JSON string with results array including row_index
        """
        results_array = []
        for index, (_, result) in enumerate(results.items(), 1):
            result_dict = result.model_dump()
            result_dict["row_index"] = index
            # Replace success with "abstained" when abstention is detected
            if result.abstention_detected and result.abstention_override_applied:
                result_dict["success"] = "abstained"
            results_array.append(result_dict)
        return json.dumps(results_array, indent=2, ensure_ascii=False)

    @staticmethod
    def export_to_csv(results: dict[str, VerificationResult], global_rubric: "Rubric | None" = None) -> str:
        """
        Export results to CSV format matching frontend format exactly.

        Args:
            results: Dictionary of verification results
            global_rubric: Optional global rubric for trait separation

        Returns:
            CSV string in frontend format
        """
        if not results:
            return "row_index,question_id,question_text,success,error,execution_time,timestamp,embedding_check_performed,embedding_similarity_score,embedding_override_applied,embedding_model_used\n"

        # Extract all unique rubric trait names from results
        all_rubric_trait_names: set[str] = set()
        for result in results.values():
            if result.verify_rubric:
                all_rubric_trait_names.update(result.verify_rubric.keys())

        # Determine global vs question-specific rubrics
        global_trait_names = set()
        if global_rubric and hasattr(global_rubric, "traits"):
            for trait in global_rubric.llm_traits:
                global_trait_names.add(trait.name)

        # Separate traits into global and question-specific
        global_traits: list[str] = sorted([trait for trait in all_rubric_trait_names if trait in global_trait_names])
        question_specific_traits: list[str] = sorted(
            [trait for trait in all_rubric_trait_names if trait not in global_trait_names]
        )

        # Create headers for global rubrics only
        global_rubric_headers = [f"rubric_{trait}" for trait in global_traits]

        headers = [
            "row_index",
            "question_id",
            "question_text",
            "raw_llm_response",
            "parsed_gt_response",
            "parsed_llm_response",
            "verify_result",
            "verify_granular_result",
            *global_rubric_headers,
            *(["question_specific_rubrics"] if question_specific_traits else []),
            "rubric_summary",
            "answering_model",
            "parsing_model",
            "replicate",
            "answering_system_prompt",
            "parsing_system_prompt",
            "success",
            "error",
            "execution_time",
            "timestamp",
            "run_name",
            "job_id",
            # Embedding check fields
            "embedding_check_performed",
            "embedding_similarity_score",
            "embedding_override_applied",
            "embedding_model_used",
        ]

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)

        for index, result in enumerate(results.values(), start=1):
            row = ResultsIOManager._create_csv_row(index, result, global_traits, question_specific_traits)
            writer.writerow(row)

        return output.getvalue()

    @staticmethod
    def _create_csv_row(
        index: int,
        result: VerificationResult,
        global_traits: list[str],
        question_specific_traits: list[str],
    ) -> list[Any]:
        """Create a CSV row for a single verification result."""
        # Extract global rubric trait values
        global_rubric_values: list[str] = []
        for trait_name in global_traits:
            if result.verify_rubric and trait_name in result.verify_rubric:
                value = result.verify_rubric[trait_name]
                global_rubric_values.append(str(value) if value is not None else "")
            else:
                global_rubric_values.append("")

        # Create question-specific rubrics JSON
        question_specific_rubrics: dict[str, int | bool] = {}
        if result.verify_rubric:
            for trait_name in question_specific_traits:
                if trait_name in result.verify_rubric:
                    question_specific_rubrics[trait_name] = result.verify_rubric[trait_name]

        question_specific_rubrics_value = json.dumps(question_specific_rubrics) if question_specific_traits else ""

        # Create rubric summary
        rubric_summary = ""
        if result.verify_rubric:
            traits = list(result.verify_rubric.items())
            passed_traits = sum(
                1
                for name, value in traits
                if (isinstance(value, bool) and value) or (isinstance(value, int | float) and value >= 3)
            )
            rubric_summary = f"{passed_traits}/{len(traits)}"

        return [
            index,  # row_index
            ResultsIOManager._escape_csv_field(result.metadata.question_id),
            ResultsIOManager._escape_csv_field(result.metadata.question_text),
            ResultsIOManager._escape_csv_field(result.template.raw_llm_response if result.template else ""),
            ResultsIOManager._escape_csv_field(
                json.dumps(result.template.parsed_gt_response)
                if result.template and result.template.parsed_gt_response
                else ""
            ),
            ResultsIOManager._escape_csv_field(
                json.dumps(result.template.parsed_llm_response)
                if result.template and result.template.parsed_llm_response
                else ""
            ),
            ResultsIOManager._escape_csv_field(
                json.dumps(result.template.verify_result)
                if result.template and result.template.verify_result is not None
                else "N/A"
            ),
            ResultsIOManager._escape_csv_field(
                json.dumps(result.template.verify_granular_result)
                if result.template and result.template.verify_granular_result is not None
                else "N/A"
            ),
            *[ResultsIOManager._escape_csv_field(value) for value in global_rubric_values],
            *([question_specific_rubrics_value] if question_specific_traits else []),
            ResultsIOManager._escape_csv_field(rubric_summary),
            ResultsIOManager._escape_csv_field(result.metadata.answering_model),
            ResultsIOManager._escape_csv_field(result.metadata.parsing_model),
            ResultsIOManager._escape_csv_field(result.metadata.replicate or ""),
            ResultsIOManager._escape_csv_field(result.metadata.answering_system_prompt or ""),
            ResultsIOManager._escape_csv_field(result.metadata.parsing_system_prompt or ""),
            ResultsIOManager._escape_csv_field(
                "abstained"
                if result.template
                and result.template.abstention_detected
                and result.template.abstention_override_applied
                else result.metadata.completed_without_errors
            ),
            ResultsIOManager._escape_csv_field(result.metadata.error or ""),
            ResultsIOManager._escape_csv_field(result.metadata.execution_time),
            ResultsIOManager._escape_csv_field(result.metadata.timestamp),
            ResultsIOManager._escape_csv_field(result.metadata.run_name or ""),
            # Embedding check fields
            ResultsIOManager._escape_csv_field(result.template.embedding_check_performed if result.template else False),
            ResultsIOManager._escape_csv_field(result.template.embedding_similarity_score if result.template else ""),
            ResultsIOManager._escape_csv_field(
                result.template.embedding_override_applied if result.template else False
            ),
            ResultsIOManager._escape_csv_field(result.template.embedding_model_used if result.template else ""),
        ]

    @staticmethod
    def _escape_csv_field(field: Any) -> str:
        """
        Escape CSV field content matching frontend logic.

        Args:
            field: Field value to escape

        Returns:
            Escaped string value
        """
        if field is None:
            return ""
        str_field = str(field)
        if "," in str_field or '"' in str_field or "\n" in str_field:
            return '"' + str_field.replace('"', '""') + '"'
        return str_field

    @staticmethod
    def load_from_json(file_path: Path) -> dict[str, VerificationResult]:
        """
        Load results from JSON file supporting multiple formats.

        Args:
            file_path: Path to the JSON file

        Returns:
            Dictionary of loaded verification results
        """
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        results = {}

        # Handle different JSON formats
        if isinstance(data, list):
            # Frontend format: array with row_index
            for item in data:
                if isinstance(item, dict):
                    # Remove row_index if present and create key
                    item_copy = dict(item)
                    row_index = item_copy.pop("row_index", None)
                    question_id = item_copy.get("question_id", "unknown")
                    result_key = f"{question_id}_{row_index}" if row_index else question_id
                    try:
                        results[result_key] = VerificationResult(**item_copy)
                    except Exception:
                        logger.warning("Skipping malformed JSON item at key %s", result_key, exc_info=True)
                        continue
        elif isinstance(data, dict):
            # Handle server format with metadata wrapper
            results_data = data["results"] if "results" in data and "metadata" in data else data

            # Reconstruct VerificationResult objects
            for result_key, result_data in results_data.items():
                if isinstance(result_data, dict):
                    try:
                        results[result_key] = VerificationResult(**result_data)
                    except Exception:
                        logger.warning("Skipping malformed result at key %s", result_key, exc_info=True)
                        continue

        return results

    @staticmethod
    def load_from_csv(file_path: Path) -> dict[str, VerificationResult]:
        """
        Load results from CSV file supporting frontend format.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary of loaded verification results
        """
        results = {}

        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                result_key, processed_row = ResultsIOManager._process_csv_row(row)
                if processed_row:
                    try:
                        results[result_key] = VerificationResult(**processed_row)
                    except Exception:
                        logger.warning("Skipping malformed CSV row at key %s", result_key, exc_info=True)
                        continue

        return results

    @staticmethod
    def _process_csv_row(row: dict[str, str]) -> tuple[str, dict[str, Any] | None]:
        """
        Process a single CSV row into a format suitable for VerificationResult.

        Args:
            row: Dictionary from csv.DictReader

        Returns:
            Tuple of (result_key, processed_row_dict) or (result_key, None) if row is invalid
        """
        # Get row_index for key generation
        row_index = row.get("row_index", "")
        question_id = row.get("question_id", "unknown")
        result_key = f"{question_id}_{row_index}" if row_index else question_id

        # Process rubric data
        verify_rubric = {}

        # Extract global rubric traits (columns starting with "rubric_")
        for key, value in row.items():
            if key.startswith("rubric_") and value:
                trait_name = key[len("rubric_") :]
                try:
                    # Try to convert to number first, then boolean
                    if isinstance(value, str) and value.isdigit():
                        verify_rubric[trait_name] = int(value)
                    elif isinstance(value, str) and value.lower() in ("true", "false"):
                        verify_rubric[trait_name] = value.lower() == "true"
                    else:
                        verify_rubric[trait_name] = value  # type: ignore[assignment]
                except (ValueError, AttributeError):
                    logger.debug("Could not convert rubric trait value for %s, using raw value", trait_name)
                    verify_rubric[trait_name] = value  # type: ignore[assignment]

        # Extract question-specific rubrics from JSON column
        if "question_specific_rubrics" in row and row["question_specific_rubrics"]:
            try:
                question_specific = json.loads(row["question_specific_rubrics"])
                if isinstance(question_specific, dict):
                    verify_rubric.update(question_specific)
            except json.JSONDecodeError:
                logger.debug("Could not parse question_specific_rubrics JSON for row")

        # Convert JSON strings back to objects
        processed_row: dict[str, Any] = {}
        for field, value in row.items():
            if field.startswith("rubric_") or field in [
                "row_index",
                "question_specific_rubrics",
                "rubric_summary",
            ]:
                continue  # Skip these fields as they're processed separately

            if (
                field in ["parsed_gt_response", "parsed_llm_response", "verify_result", "verify_granular_result"]
                and value
                and value != "N/A"
            ):
                try:
                    processed_row[field] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    logger.debug("Could not parse JSON for field %s, using raw value", field)
                    processed_row[field] = value
            elif field == "execution_time" and value:
                try:
                    processed_row[field] = float(value)
                except ValueError:
                    logger.debug("Could not parse execution_time '%s', defaulting to 0.0", value)
                    processed_row[field] = 0.0
            elif field == "success" and value:
                # Handle "abstained" status as True (since abstention overrides are successful responses)
                if value.lower() == "abstained":
                    processed_row[field] = True
                else:
                    processed_row[field] = value.lower() in ("true", "1", "yes")
            elif field == "replicate" and value:
                try:
                    processed_row[field] = int(value)
                except ValueError:
                    processed_row[field] = None
            else:
                processed_row[field] = value if value else None

        # Add the processed rubric data
        if verify_rubric:
            processed_row["verify_rubric"] = verify_rubric

        return result_key, processed_row
