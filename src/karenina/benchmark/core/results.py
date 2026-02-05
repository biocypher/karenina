"""Results management functionality for benchmarks."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...schemas.entities import Rubric
    from .base import BenchmarkBase

from ...schemas.verification import VerificationResult
from .results_io import ResultsIOManager

logger = logging.getLogger(__name__)


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
                if question_ids is not None and result.metadata.question_id not in question_ids:
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
                if question_id is not None and result.metadata.question_id != question_id:
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
                    if result.metadata.question_id in question_ids:
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
            return ResultsIOManager.export_to_json(results)
        elif format.lower() == "csv":
            return ResultsIOManager.export_to_csv(results, global_rubric)
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

        successful_count = sum(1 for r in results.values() if r.metadata.completed_without_errors)
        failed_count = sum(1 for r in results.values() if not r.metadata.completed_without_errors)
        unique_questions = len({r.metadata.question_id for r in results.values()})
        total_execution_time = sum(r.metadata.execution_time for r in results.values() if r.metadata.execution_time)
        total_result_count = len(results)
        average_execution_time = total_execution_time / total_result_count if total_result_count else 0.0

        # Count unique model combinations
        model_combinations = len({f"{r.metadata.answering_model}:{r.metadata.parsing_model}" for r in results.values()})

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
                    logger.debug("Could not parse timestamp from run name %s", run_name)
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

        # Determine format from extension and delegate to handler
        if file_path.suffix.lower() == ".json":
            results = ResultsIOManager.load_from_json(file_path)
        elif file_path.suffix.lower() == ".csv":
            results = ResultsIOManager.load_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .json, .csv")

        # Store in memory if run_name provided
        if run_name:
            self._in_memory_results[run_name] = results

        return results
