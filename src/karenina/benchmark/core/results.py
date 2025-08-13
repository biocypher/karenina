"""Results management functionality for benchmarks."""

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BenchmarkBase
    from .metadata import MetadataManager

from ..models import VerificationResult


class ResultsManager:
    """Manager for verification result storage, retrieval, and export."""

    def __init__(self, base: "BenchmarkBase", metadata_manager: "MetadataManager") -> None:
        """Initialize with reference to benchmark base and metadata manager."""
        self.base = base
        self.metadata_manager = metadata_manager

    def store_verification_results(
        self,
        results: dict[str, VerificationResult],
        run_name: str | None = None,
    ) -> None:
        """
        Store verification results in the benchmark metadata.

        Args:
            results: Dictionary of verification results to store
            run_name: Optional run name for organizing results
        """
        # Store results in benchmark custom properties
        results_data = {}
        for result_key, result in results.items():
            results_data[result_key] = result.model_dump()

        # Create a timestamp-based key if no run name provided
        if run_name is None:
            run_name = f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Store in benchmark custom properties
        verification_results_key = f"verification_results_{run_name}"
        self.metadata_manager.set_custom_property(verification_results_key, results_data)

    def get_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> dict[str, VerificationResult]:
        """
        Get verification results for specific questions and/or runs.

        Args:
            question_ids: Optional list of question IDs to filter by
            run_name: Optional run name to filter by

        Returns:
            Dictionary of verification results
        """
        all_results = {}

        # Get all verification result properties
        all_props = self.metadata_manager.get_all_custom_properties()

        for prop_name, prop_value in all_props.items():
            if prop_name.startswith("verification_results_"):
                # Check if this matches the run name filter
                if run_name is not None:
                    expected_key = f"verification_results_{run_name}"
                    if prop_name != expected_key:
                        continue

                # Parse the stored results
                try:
                    stored_results = prop_value
                    if isinstance(stored_results, dict):
                        for result_key, result_data in stored_results.items():
                            # Reconstruct VerificationResult object
                            if isinstance(result_data, dict):
                                verification_result = VerificationResult(**result_data)

                                # Check if this matches question ID filter
                                if question_ids is not None and verification_result.question_id not in question_ids:
                                    continue

                                all_results[result_key] = verification_result
                except Exception:
                    # Skip malformed result data
                    continue

        return all_results

    def get_verification_history(self, question_id: str | None = None) -> dict[str, dict[str, VerificationResult]]:
        """
        Get verification history organized by run name.

        Args:
            question_id: Optional question ID to filter by

        Returns:
            Dictionary mapping run names to their results
        """
        history = {}

        # Get all verification result properties
        all_props = self.metadata_manager.get_all_custom_properties()

        for prop_name, prop_value in all_props.items():
            if prop_name.startswith("verification_results_"):
                run_name = prop_name[len("verification_results_") :]

                try:
                    stored_results = prop_value
                    if isinstance(stored_results, dict):
                        run_results = {}
                        for result_key, result_data in stored_results.items():
                            if isinstance(result_data, dict):
                                verification_result = VerificationResult(**result_data)

                                # Check if this matches question ID filter
                                if question_id is not None and verification_result.question_id != question_id:
                                    continue

                                run_results[result_key] = verification_result

                        if run_results:  # Only add if we have results
                            history[run_name] = run_results
                except Exception:
                    # Skip malformed result data
                    continue

        return history

    def clear_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> int:
        """
        Clear verification results.

        Args:
            question_ids: Optional list of question IDs to clear (None = all)
            run_name: Optional run name to clear (None = all runs)

        Returns:
            Number of result entries that were cleared
        """
        cleared_count = 0

        # Get all verification result properties
        all_props = self.metadata_manager.get_all_custom_properties()
        props_to_remove = []
        props_to_update = {}

        for prop_name, prop_value in all_props.items():
            if prop_name.startswith("verification_results_"):
                # Check if this matches the run name filter
                if run_name is not None:
                    expected_key = f"verification_results_{run_name}"
                    if prop_name != expected_key:
                        continue

                try:
                    stored_results = prop_value
                    if isinstance(stored_results, dict):
                        updated_results = {}

                        for result_key, result_data in stored_results.items():
                            if isinstance(result_data, dict):
                                verification_result = VerificationResult(**result_data)

                                # Check if this should be cleared
                                should_clear = False
                                if question_ids is None:
                                    should_clear = True  # Clear all
                                elif verification_result.question_id in question_ids:
                                    should_clear = True

                                if should_clear:
                                    cleared_count += 1
                                else:
                                    updated_results[result_key] = result_data

                        # Update or remove the property
                        if not updated_results:
                            props_to_remove.append(prop_name)
                        else:
                            props_to_update[prop_name] = updated_results
                except Exception:
                    # Skip malformed result data
                    continue

        # Apply the changes
        for prop_name in props_to_remove:
            self.metadata_manager.remove_custom_property(prop_name)

        for prop_name, updated_value in props_to_update.items():
            self.metadata_manager.set_custom_property(prop_name, updated_value)

        return cleared_count

    def export_verification_results(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
        format: str = "json",
    ) -> str:
        """
        Export verification results in specified format.

        Args:
            question_ids: Optional list of question IDs to export
            run_name: Optional run name to export
            format: Export format ("json" or "csv")

        Returns:
            Exported data as string

        Raises:
            ValueError: If format is not supported
        """
        results = self.get_verification_results(question_ids, run_name)

        if format.lower() == "json":
            # Convert to JSON
            export_data = {}
            for result_key, result in results.items():
                export_data[result_key] = result.model_dump()
            return json.dumps(export_data, indent=2, ensure_ascii=False)

        elif format.lower() == "csv":
            # Convert to CSV
            import csv
            from io import StringIO

            output = StringIO()
            writer = csv.writer(output)

            # Header
            if results:
                # Get field names from first result
                sample_result = next(iter(results.values()))
                fieldnames = list(sample_result.model_dump().keys())
                writer.writerow(["result_key"] + fieldnames)

                # Data rows
                for result_key, result in results.items():
                    row_data = [result_key]
                    result_dict = result.model_dump()
                    for field in fieldnames:
                        value = result_dict.get(field, "")
                        # Convert complex objects to string
                        if isinstance(value, dict | list):
                            value = json.dumps(value)
                        row_data.append(str(value))
                    writer.writerow(row_data)

            return output.getvalue()

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
        Check if verification results exist.

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
        Get all verification run names.

        Returns:
            List of run names
        """
        all_props = self.metadata_manager.get_all_custom_properties()
        run_names = []

        for prop_name in all_props:
            if prop_name.startswith("verification_results_"):
                run_name = prop_name[len("verification_results_") :]
                run_names.append(run_name)

        return sorted(run_names)

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
