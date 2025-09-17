"""Results management functionality for benchmarks."""

import csv
import json
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...schemas.rubric_class import Rubric
    from .base import BenchmarkBase

from ..models import VerificationResult


class ResultsManager:
    """Manager for verification result export and import (no checkpoint storage)."""

    def __init__(self, base: "BenchmarkBase") -> None:
        """Initialize with reference to benchmark base."""
        self.base = base
        # Store results in memory only - they are NOT saved to checkpoint
        self._in_memory_results: dict[str, dict[str, VerificationResult]] = {}

    def store_verification_results(
        self,
        results: dict[str, VerificationResult],
        run_name: str | None = None,
    ) -> None:
        """
        Store verification results in memory (NOT in checkpoint).

        Args:
            results: Dictionary of verification results to store
            run_name: Optional run name for organizing results
        """
        # Create a timestamp-based key if no run name provided
        if run_name is None:
            run_name = f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Store in memory only
        self._in_memory_results[run_name] = results

    def get_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Get verification results from memory for specific questions and/or runs.

        Args:
            question_ids: Optional list of question IDs to filter by
            run_name: Optional run name to filter by

        Returns:
            Dictionary of verification results
        """
        all_results = {}

        # Filter by run name if provided
        runs_to_check = [run_name] if run_name else list(self._in_memory_results.keys())

        for run in runs_to_check:
            if run not in self._in_memory_results:
                continue

            for result_key, result in self._in_memory_results[run].items():
                # Check if this matches question ID filter
                if question_ids is not None and result.question_id not in question_ids:
                    continue

                all_results[result_key] = result

        return all_results

    def get_verification_history(self, question_id: str | None = None) -> dict[str, dict[str, VerificationResult]]:
        """
        Get verification history organized by run name from memory.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping run names to their results
        """
        history = {}

        for run_name, results in self._in_memory_results.items():
            run_results = {}
            for result_key, result in results.items():
                # Check if this matches question ID filter
                if question_id is not None and result.question_id != question_id:
                    continue

                run_results[result_key] = result

            if run_results:  # Only add if we have results
                history[run_name] = run_results

        return history

    def clear_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> int:
        """
        Clear verification results from memory.

        Args:
            question_ids: Optional list of question IDs to clear (None = all)
            run_name: Optional run name to clear (None = all runs)

        Returns:
            Number of result entries that were cleared
        """
        cleared_count = 0

        # Filter by run name
        runs_to_check = [run_name] if run_name else list(self._in_memory_results.keys())

        for run in runs_to_check:
            if run not in self._in_memory_results:
                continue

            if question_ids is None:
                # Clear all results for this run
                cleared_count += len(self._in_memory_results[run])
                del self._in_memory_results[run]
            else:
                # Clear specific questions
                results_to_remove = []
                for result_key, result in self._in_memory_results[run].items():
                    if result.question_id in question_ids:
                        results_to_remove.append(result_key)
                        cleared_count += 1

                for key in results_to_remove:
                    del self._in_memory_results[run][key]

                # Remove run if no results left
                if not self._in_memory_results[run]:
                    del self._in_memory_results[run]

        return cleared_count

    def export_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        format: str = "json",
        global_rubric: "Rubric | None" = None,
    ) -> str:
        """
        Export verification results in specified format matching frontend format.

        Args:
            question_ids: Optional list of question IDs to export
            run_name: Optional run name to export
            format: Export format ("json" or "csv")
            global_rubric: Optional global rubric for CSV export

        Returns:
            Exported data as string

        Raises:
            ValueError: If format is not supported
        """
        results = self.get_verification_results(question_ids, run_name)

        if format.lower() == "json":
            # Convert to frontend JSON format (array with row_index)
            results_array = []
            for index, (_, result) in enumerate(results.items(), 1):
                result_dict = result.model_dump()
                result_dict["row_index"] = index
                results_array.append(result_dict)
            return json.dumps(results_array, indent=2, ensure_ascii=False)

        elif format.lower() == "csv":
            # Convert to frontend CSV format
            return self._export_to_frontend_csv(results, global_rubric)

        else:
            raise ValueError(f"Unsupported export format: {format}. Supported formats: json, csv")

    def get_verification_summary(self, run_name: str | None = None) -> dict[str, Any]:
        """
        Get summary statistics for verification results.

        Args:
            run_name: Optional run name to filter by

        Returns:
            Dictionary with verification statistics
        """
        results = self.get_verification_results(run_name=run_name)

        if not results:
            return {
                "total_results": 0,
                "successful_count": 0,
                "failed_count": 0,
                "success_rate": 0.0,
                "unique_questions": 0,
                "average_execution_time": 0.0,
                "model_combinations": 0,
            }

        successful_count = sum(1 for r in results.values() if r.success)
        failed_count = len(results) - successful_count
        unique_questions = len({r.question_id for r in results.values()})
        total_execution_time = sum(r.execution_time for r in results.values() if r.execution_time)
        average_execution_time = total_execution_time / len(results) if results else 0.0

        # Count unique model combinations
        model_combinations = len({f"{r.answering_model}:{r.parsing_model}" for r in results.values()})

        return {
            "total_results": len(results),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "success_rate": (successful_count / len(results)) * 100 if results else 0.0,
            "unique_questions": unique_questions,
            "average_execution_time": round(average_execution_time, 2),
            "total_execution_time": round(total_execution_time, 2),
            "model_combinations": model_combinations,
        }

    def get_results_by_question(self, question_id: str) -> dict[str, VerificationResult]:
        """
        Get all verification results for a specific question.

        Args:
            question_id: The question ID

        Returns:
            Dictionary of verification results for the question
        """
        return self.get_verification_results(question_ids=[question_id])

    def get_results_by_run(self, run_name: str) -> dict[str, VerificationResult]:
        """
        Get all verification results for a specific run.

        Args:
            run_name: The run name

        Returns:
            Dictionary of verification results for the run
        """
        return self.get_verification_results(run_name=run_name)

    def get_latest_results(self, question_id: str | None = None) -> dict[str, VerificationResult]:
        """
        Get the most recent verification results.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary of the most recent verification results
        """
        history = self.get_verification_history(question_id)

        if not history:
            return {}

        # Find the most recent run by parsing timestamps from run names
        latest_run = None
        latest_timestamp = None

        for run_name in history:
            # Try to extract timestamp from run name
            if "_" in run_name:
                try:
                    timestamp_part = run_name.split("_")[-1]
                    # Try to parse as YYYYMMDD_HHMMSS format
                    if len(timestamp_part) >= 8:
                        timestamp = datetime.strptime(timestamp_part[:8], "%Y%m%d")
                        if latest_timestamp is None or timestamp > latest_timestamp:
                            latest_timestamp = timestamp
                            latest_run = run_name
                except (ValueError, IndexError):
                    continue

        # If we couldn't parse timestamps, just get the last one alphabetically
        if latest_run is None and history:
            latest_run = max(history.keys())

        return history.get(latest_run, {}) if latest_run else {}

    def has_results(self, question_id: str | None = None, run_name: str | None = None) -> bool:
        """
        Check if verification results exist in memory.

        Args:
            question_id: Optional question ID to check
            run_name: Optional run name to check

        Returns:
            True if results exist, False otherwise
        """
        results = self.get_verification_results(question_ids=[question_id] if question_id else None, run_name=run_name)
        return len(results) > 0

    def get_all_run_names(self) -> list[str]:
        """
        Get all verification run names from memory.

        Returns:
            List of run names
        """
        return sorted(self._in_memory_results.keys())

    def get_results_statistics_by_run(self) -> dict[str, dict[str, Any]]:
        """
        Get verification statistics for each run.

        Returns:
            Dictionary mapping run names to their statistics
        """
        run_stats = {}
        for run_name in self.get_all_run_names():
            run_stats[run_name] = self.get_verification_summary(run_name)
        return run_stats

    def export_results_to_file(
        self,
        file_path: Path,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        format: str | None = None,
        global_rubric: "Rubric | None" = None,
    ) -> None:
        """
        Export verification results directly to a file.

        Args:
            file_path: Path where to save the results file
            question_ids: Optional list of question IDs to export
            run_name: Optional run name to export
            format: Export format ("json" or "csv"), auto-detected from extension if None
            global_rubric: Optional global rubric for CSV export

        Raises:
            ValueError: If format cannot be determined or is not supported
        """
        file_path = Path(file_path)

        # Auto-detect format from file extension
        if format is None:
            if file_path.suffix.lower() == ".json":
                format = "json"
            elif file_path.suffix.lower() == ".csv":
                format = "csv"
            else:
                raise ValueError(
                    f"Cannot determine format from extension '{file_path.suffix}'. Please specify format explicitly."
                )

        # Export data
        exported_data = self.export_verification_results(question_ids, run_name, format, global_rubric)

        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(exported_data)

    def load_results_from_file(self, file_path: Path, run_name: str | None = None) -> dict[str, VerificationResult]:
        """
        Load verification results from a previously exported file.

        Args:
            file_path: Path to the results file
            run_name: Optional run name to assign to loaded results

        Returns:
            Dictionary of loaded verification results

        Raises:
            ValueError: If file format is not supported or file is malformed
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")

        # Determine format from extension
        if file_path.suffix.lower() == ".json":
            return self._load_results_from_json(file_path, run_name)
        elif file_path.suffix.lower() == ".csv":
            return self._load_results_from_csv(file_path, run_name)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .json, .csv")

    def _export_to_frontend_csv(
        self, results: dict[str, VerificationResult], global_rubric: "Rubric | None" = None
    ) -> str:
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
            for trait in global_rubric.traits:
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
            "answering_replicate",
            "parsing_replicate",
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

        for index, result in enumerate(results.values(), 1):
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

            row = [
                index,  # row_index
                self._escape_csv_field(result.question_id),
                self._escape_csv_field(result.question_text),
                self._escape_csv_field(result.raw_llm_response),
                self._escape_csv_field(json.dumps(result.parsed_gt_response) if result.parsed_gt_response else ""),
                self._escape_csv_field(json.dumps(result.parsed_llm_response) if result.parsed_llm_response else ""),
                self._escape_csv_field(json.dumps(result.verify_result) if result.verify_result is not None else "N/A"),
                self._escape_csv_field(
                    json.dumps(result.verify_granular_result) if result.verify_granular_result is not None else "N/A"
                ),
                *[self._escape_csv_field(value) for value in global_rubric_values],
                *([question_specific_rubrics_value] if question_specific_traits else []),
                self._escape_csv_field(rubric_summary),
                self._escape_csv_field(result.answering_model),
                self._escape_csv_field(result.parsing_model),
                self._escape_csv_field(result.answering_replicate or ""),
                self._escape_csv_field(result.parsing_replicate or ""),
                self._escape_csv_field(result.answering_system_prompt or ""),
                self._escape_csv_field(result.parsing_system_prompt or ""),
                self._escape_csv_field(result.success),
                self._escape_csv_field(result.error or ""),
                self._escape_csv_field(result.execution_time),
                self._escape_csv_field(result.timestamp),
                self._escape_csv_field(result.run_name or ""),
                self._escape_csv_field(result.job_id or ""),
                # Embedding check fields
                self._escape_csv_field(result.embedding_check_performed),
                self._escape_csv_field(result.embedding_similarity_score or ""),
                self._escape_csv_field(result.embedding_override_applied),
                self._escape_csv_field(result.embedding_model_used or ""),
            ]
            writer.writerow(row)

        return output.getvalue()

    def _escape_csv_field(self, field: Any) -> str:
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

    def _load_results_from_json(self, file_path: Path, run_name: str | None = None) -> dict[str, VerificationResult]:
        """Load results from JSON file supporting multiple formats."""
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
                        # Skip malformed items
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
                        # Skip malformed results
                        continue

        # Store in memory if run_name provided
        if run_name:
            self._in_memory_results[run_name] = results

        return results

    def _load_results_from_csv(self, file_path: Path, run_name: str | None = None) -> dict[str, VerificationResult]:
        """Load results from CSV file supporting frontend format."""
        results = {}

        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
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
                            verify_rubric[trait_name] = value  # type: ignore[assignment]

                # Extract question-specific rubrics from JSON column
                if "question_specific_rubrics" in row and row["question_specific_rubrics"]:
                    try:
                        question_specific = json.loads(row["question_specific_rubrics"])
                        if isinstance(question_specific, dict):
                            verify_rubric.update(question_specific)
                    except json.JSONDecodeError:
                        pass

                # Convert JSON strings back to objects
                processed_row = {}
                for field, value in row.items():
                    if field.startswith("rubric_") or field in [
                        "row_index",
                        "question_specific_rubrics",
                        "rubric_summary",
                    ]:
                        continue  # Skip these fields as they're processed separately

                    if (
                        field
                        in ["parsed_gt_response", "parsed_llm_response", "verify_result", "verify_granular_result"]
                        and value
                        and value != "N/A"
                    ):
                        try:
                            processed_row[field] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            processed_row[field] = value
                    elif field == "execution_time" and value:
                        try:
                            processed_row[field] = float(value)
                        except ValueError:
                            processed_row[field] = 0.0
                    elif field == "success" and value:
                        processed_row[field] = value.lower() in ("true", "1", "yes")
                    elif field in ["answering_replicate", "parsing_replicate"] and value:
                        try:
                            processed_row[field] = int(value)
                        except ValueError:
                            processed_row[field] = None
                    else:
                        processed_row[field] = value if value else None

                # Add the processed rubric data
                if verify_rubric:
                    processed_row["verify_rubric"] = verify_rubric

                try:
                    results[result_key] = VerificationResult(**processed_row)
                except Exception:
                    # Skip malformed rows
                    continue

        # Store in memory if run_name provided
        if run_name:
            self._in_memory_results[run_name] = results

        return results
