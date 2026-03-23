"""Results I/O manager for verification result export and import."""

import csv
import json
import logging
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from karenina.schemas.entities import Rubric

from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

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
            if result.template and result.template.abstention_detected and result.template.abstention_override_applied:
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
            if result.rubric:
                scores = result.rubric.get_all_trait_scores()
                if scores:
                    all_rubric_trait_names.update(scores.keys())

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
        # Get all trait scores from rubric sub-object
        all_trait_scores = result.rubric.get_all_trait_scores() if result.rubric else None

        # Extract global rubric trait values
        global_rubric_values: list[str] = []
        for trait_name in global_traits:
            if all_trait_scores and trait_name in all_trait_scores:
                value = all_trait_scores[trait_name]
                global_rubric_values.append(str(value) if value is not None else "")
            else:
                global_rubric_values.append("")

        # Create question-specific rubrics JSON
        question_specific_rubrics: dict[str, int | bool | float | str | list[Any] | dict[str, float] | None] = {}
        if all_trait_scores:
            for trait_name in question_specific_traits:
                if trait_name in all_trait_scores:
                    question_specific_rubrics[trait_name] = all_trait_scores[trait_name]

        question_specific_rubrics_value = json.dumps(question_specific_rubrics) if question_specific_traits else ""

        # Create rubric summary
        rubric_summary = ""
        if all_trait_scores:
            traits = list(all_trait_scores.items())
            passed_traits = sum(
                1
                for name, value in traits
                if (isinstance(value, bool) and value) or (isinstance(value, int | float) and value >= 3)
            )
            rubric_summary = f"{passed_traits}/{len(traits)}"

        return [
            index,  # row_index
            result.metadata.question_id,
            result.metadata.question_text,
            result.template.raw_llm_response if result.template else "",
            (
                json.dumps(result.template.parsed_gt_response)
                if result.template and result.template.parsed_gt_response
                else ""
            ),
            (
                json.dumps(result.template.parsed_llm_response)
                if result.template and result.template.parsed_llm_response
                else ""
            ),
            (
                json.dumps(result.template.verify_result)
                if result.template and result.template.verify_result is not None
                else "N/A"
            ),
            (
                json.dumps(result.template.verify_granular_result)
                if result.template and result.template.verify_granular_result is not None
                else "N/A"
            ),
            *global_rubric_values,
            *([question_specific_rubrics_value] if question_specific_traits else []),
            rubric_summary,
            result.metadata.answering_model,
            result.metadata.parsing_model,
            result.metadata.replicate or "",
            result.metadata.answering_system_prompt or "",
            result.metadata.parsing_system_prompt or "",
            (
                "abstained"
                if result.template
                and result.template.abstention_detected
                and result.template.abstention_override_applied
                else result.metadata.completed_without_errors
            ),
            result.metadata.error or "",
            result.metadata.execution_time,
            result.metadata.timestamp,
            result.metadata.run_name or "",
            "",  # job_id: not stored in VerificationResult, populated by server layer
            # Embedding check fields
            result.template.embedding_check_performed if result.template else False,
            result.template.embedding_similarity_score if result.template else "",
            result.template.embedding_override_applied if result.template else False,
            result.template.embedding_model_used if result.template else "",
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
        """Load results from CSV file, reconstructing nested VerificationResult objects.

        Parses the flat CSV columns back into the nested sub-objects
        (metadata, template, rubric) that VerificationResult expects.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Dictionary of loaded verification results keyed by "{question_id}_{row_index}".
        """
        results: dict[str, VerificationResult] = {}

        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=1):
                try:
                    result_key, verification_result = ResultsIOManager._process_csv_row(row)
                    if verification_result is not None:
                        results[result_key] = verification_result
                    else:
                        logger.warning("Skipping invalid CSV row %d: processing returned None", row_num)
                except Exception:
                    logger.warning("Skipping malformed CSV row %d", row_num, exc_info=True)
                    continue

        logger.info("Loaded %d results from CSV file %s", len(results), file_path)
        return results

    @staticmethod
    def _parse_model_identity(display_string: str | None) -> ModelIdentity:
        """Parse a ModelIdentity display string back into a ModelIdentity object.

        Handles the format "interface:model_name" or "interface:model_name +[tool1, tool2]".

        Args:
            display_string: The display string from CSV export, or None/empty.

        Returns:
            Reconstructed ModelIdentity.
        """
        if not display_string:
            return ModelIdentity(interface="unknown", model_name="unknown")

        tools: list[str] = []
        base = display_string

        # Split on " +[" to extract tools
        if " +[" in display_string:
            base, tools_part = display_string.split(" +[", 1)
            tools = [t.strip() for t in tools_part.rstrip("]").split(",") if t.strip()]

        # Split base on ":" to get interface and model_name
        parts = base.split(":", 1)
        interface = parts[0] if parts else "unknown"
        model_name = parts[1] if len(parts) > 1 else "unknown"

        return ModelIdentity(interface=interface, model_name=model_name, tools=tools)

    @staticmethod
    def _parse_rubric_traits(row: dict[str, str]) -> dict[str, int | bool]:
        """Extract rubric trait scores from CSV row columns.

        Collects traits from "rubric_*" columns and the "question_specific_rubrics"
        JSON column, converting values to int or bool.

        Args:
            row: Dictionary from csv.DictReader.

        Returns:
            Dict of trait name to score (int or bool). Empty dict if no traits found.
        """
        trait_scores: dict[str, int | bool] = {}

        # Extract global rubric traits (columns starting with "rubric_")
        for key, value in row.items():
            if key == "rubric_summary":
                continue
            if key.startswith("rubric_") and value:
                trait_name = key[len("rubric_") :]
                try:
                    if value.isdigit():
                        trait_scores[trait_name] = int(value)
                    elif value.lower() in ("true", "false"):
                        trait_scores[trait_name] = value.lower() == "true"
                    else:
                        # Try float then int conversion for numeric strings like "4.0"
                        trait_scores[trait_name] = int(float(value))
                except (ValueError, AttributeError):
                    logger.debug("Could not convert rubric trait value for %s, skipping", trait_name)

        # Extract question-specific rubrics from JSON column
        qs_raw = row.get("question_specific_rubrics", "")
        if qs_raw:
            try:
                question_specific = json.loads(qs_raw)
                if isinstance(question_specific, dict):
                    for name, value in question_specific.items():
                        if isinstance(value, bool | int):
                            trait_scores[name] = value
                        elif isinstance(value, float) or isinstance(value, str) and value.isdigit():
                            trait_scores[name] = int(value)
            except json.JSONDecodeError:
                logger.debug("Could not parse question_specific_rubrics JSON")

        return trait_scores

    @staticmethod
    def _parse_json_field(value: str | None) -> Any:
        """Parse a JSON string field, returning None for empty or N/A values.

        Args:
            value: Raw string from CSV cell.

        Returns:
            Parsed JSON value, or None if the value is empty/N/A/unparseable.
        """
        if not value or value == "N/A":
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _process_csv_row(row: dict[str, str]) -> tuple[str, VerificationResult | None]:
        """Process a single CSV row into a VerificationResult with nested sub-objects.

        Reconstructs the metadata, template, and rubric sub-objects from the
        flat CSV columns that export_to_csv produces.

        Args:
            row: Dictionary from csv.DictReader.

        Returns:
            Tuple of (result_key, VerificationResult) or (result_key, None) if
            the row cannot be parsed.
        """
        # Generate result key
        row_index = row.get("row_index", "")
        question_id = row.get("question_id", "unknown")
        result_key = f"{question_id}_{row_index}" if row_index else question_id

        # Parse model identities from display strings
        answering = ResultsIOManager._parse_model_identity(row.get("answering_model"))
        parsing = ResultsIOManager._parse_model_identity(row.get("parsing_model"))

        # Parse execution_time
        execution_time = 0.0
        if row.get("execution_time"):
            try:
                execution_time = float(row["execution_time"])
            except ValueError:
                logger.debug("Could not parse execution_time '%s', defaulting to 0.0", row.get("execution_time"))

        # Parse success/completed_without_errors
        success_raw = row.get("success", "")
        if success_raw:
            completed = True if success_raw.lower() == "abstained" else success_raw.lower() in ("true", "1", "yes")
        else:
            completed = True

        # Parse replicate
        replicate: int | None = None
        if row.get("replicate"):
            try:
                replicate = int(row["replicate"])
            except ValueError:
                replicate = None

        # Build timestamp (required field; fall back to current time if missing)
        timestamp = row.get("timestamp") or ""

        # Compute result_id from available data
        result_id = VerificationResultMetadata.compute_result_id(
            question_id=question_id,
            answering=answering,
            parsing=parsing,
            timestamp=timestamp,
            replicate=replicate,
        )

        # Build metadata sub-object
        metadata = VerificationResultMetadata(
            question_id=question_id,
            template_id="no_template",
            completed_without_errors=completed,
            error=row.get("error") or None,
            question_text=row.get("question_text", ""),
            answering=answering,
            parsing=parsing,
            answering_system_prompt=row.get("answering_system_prompt") or None,
            parsing_system_prompt=row.get("parsing_system_prompt") or None,
            execution_time=execution_time,
            timestamp=timestamp,
            result_id=result_id,
            run_name=row.get("run_name") or None,
            replicate=replicate,
        )

        # Build template sub-object (if any template-related data exists)
        template: VerificationResultTemplate | None = None
        raw_llm_response = row.get("raw_llm_response", "")
        verify_result_raw = row.get("verify_result")
        parsed_gt = ResultsIOManager._parse_json_field(row.get("parsed_gt_response"))
        parsed_llm = ResultsIOManager._parse_json_field(row.get("parsed_llm_response"))
        verify_result = ResultsIOManager._parse_json_field(verify_result_raw)
        verify_granular = ResultsIOManager._parse_json_field(row.get("verify_granular_result"))

        # Parse embedding fields
        embedding_check = row.get("embedding_check_performed", "").lower() in ("true", "1")
        embedding_score: float | None = None
        if row.get("embedding_similarity_score"):
            try:
                embedding_score = float(row["embedding_similarity_score"])
            except ValueError:
                embedding_score = None
        embedding_override = row.get("embedding_override_applied", "").lower() in ("true", "1")
        embedding_model = row.get("embedding_model_used") or None

        has_template_data = bool(
            raw_llm_response or parsed_gt or parsed_llm or verify_result is not None or verify_granular is not None
        )
        if has_template_data:
            template = VerificationResultTemplate(
                raw_llm_response=raw_llm_response or "",
                parsed_gt_response=parsed_gt,
                parsed_llm_response=parsed_llm,
                verify_result=verify_result,
                verify_granular_result=verify_granular,
                embedding_check_performed=embedding_check,
                embedding_similarity_score=embedding_score,
                embedding_override_applied=embedding_override,
                embedding_model_used=embedding_model,
            )

        # Build rubric sub-object (if any rubric traits exist)
        rubric: VerificationResultRubric | None = None
        trait_scores = ResultsIOManager._parse_rubric_traits(row)
        if trait_scores:
            rubric = VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores=trait_scores,
            )

        verification_result = VerificationResult(
            metadata=metadata,
            template=template,
            rubric=rubric,
        )

        return result_key, verification_result
