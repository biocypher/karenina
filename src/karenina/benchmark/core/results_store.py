"""Standalone results store for accumulating verification results across runs.

ResultsStore holds VerificationResultSets keyed by run name, providing
query, filter, summary, and export capabilities independent of the
Benchmark facade.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from karenina.schemas.results.verification_result_set import VerificationResultSet
from karenina.schemas.verification import VerificationResult

logger = logging.getLogger(__name__)

__all__ = ["ResultsStore"]


class ResultsStore:
    """Accumulates VerificationResultSets keyed by run name.

    Each call to ``add()`` stores a full result set under a unique run name.
    Query methods allow filtering by run, question, or recency. The store is
    independent of the Benchmark facade and can be composed with it or used
    on its own.

    Example:
        ```python
        store = ResultsStore()
        store.add(result_set, run_name="gpt4o_baseline")
        store.add(result_set, run_name="claude_baseline")
        latest = store.get_latest()
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty results store."""
        self._runs: dict[str, VerificationResultSet] = {}

    # ========================================================================
    # Add
    # ========================================================================

    def add(
        self,
        result_set: VerificationResultSet,
        run_name: str | None = None,
    ) -> str:
        """Add a result set to the store.

        Args:
            result_set: The verification result set to store.
            run_name: Key for this run. If None, a timestamped name is
                generated automatically (``run_YYYY-MM-DDTHH:MM:SS``).

        Returns:
            The run name under which the result set was stored.

        Raises:
            ValueError: If ``run_name`` already exists in the store.
        """
        if run_name is None:
            run_name = f"run_{datetime.now(tz=UTC).strftime('%Y-%m-%dT%H:%M:%S')}"
            logger.info("Auto-generated run name: %s", run_name)

        if run_name in self._runs:
            raise ValueError(
                f"Run name '{run_name}' already exists. Use a unique name or omit run_name to auto-generate one."
            )

        self._runs[run_name] = result_set
        logger.info(
            "Added %d results under run '%s'",
            len(result_set.results),
            run_name,
        )
        return run_name

    # ========================================================================
    # Query
    # ========================================================================

    def get_by_run(self, run_name: str) -> VerificationResultSet:
        """Return the result set for a given run.

        Args:
            run_name: The run to look up.

        Returns:
            The VerificationResultSet stored under ``run_name``.

        Raises:
            KeyError: If the run name does not exist.
        """
        if run_name not in self._runs:
            raise KeyError(f"No run named '{run_name}' in store")
        return self._runs[run_name]

    def get_by_question(
        self,
        question_id: str,
    ) -> dict[str, list[VerificationResult]]:
        """Collect results for a question across all runs.

        Args:
            question_id: The question ID to search for.

        Returns:
            Dict mapping run name to a list of matching VerificationResult
            objects. Runs with no matches are omitted.
        """
        matches: dict[str, list[VerificationResult]] = {}
        for run_name, result_set in self._runs.items():
            hits = [r for r in result_set.results if r.metadata.question_id == question_id]
            if hits:
                matches[run_name] = hits
        return matches

    def get_latest(
        self,
        question_id: str | None = None,
    ) -> dict[str, VerificationResult]:
        """Return the most recent result per question.

        Iterates runs in reverse insertion order and collects the first
        result encountered for each question ID.

        Args:
            question_id: If provided, only return the latest result for
                this specific question.

        Returns:
            Dict mapping question ID to the most recent VerificationResult.
        """
        latest: dict[str, VerificationResult] = {}
        for result_set in reversed(list(self._runs.values())):
            for result in result_set.results:
                qid = result.metadata.question_id
                if question_id is not None and qid != question_id:
                    continue
                if qid not in latest:
                    latest[qid] = result
        return latest

    def has_results(
        self,
        question_id: str | None = None,
        run_name: str | None = None,
    ) -> bool:
        """Check whether results exist, optionally filtered.

        Args:
            question_id: If provided, check only for this question.
            run_name: If provided, check only in this run.

        Returns:
            True if at least one matching result exists.
        """
        if not self._runs:
            return False

        runs_to_check = {run_name: self._runs[run_name]} if run_name and run_name in self._runs else self._runs
        if run_name and run_name not in self._runs:
            return False

        for result_set in runs_to_check.values():
            for result in result_set.results:
                if question_id is None or result.metadata.question_id == question_id:
                    return True
        return False

    def get_all_runs(self) -> list[str]:
        """Return run names in insertion order.

        Returns:
            List of run name strings.
        """
        return list(self._runs.keys())

    # ========================================================================
    # Clear
    # ========================================================================

    def clear(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> int:
        """Remove results from the store.

        Args:
            question_ids: If provided, remove only results with these
                question IDs (from all runs, or from ``run_name`` if given).
            run_name: If provided, scope the clear to this run only.

        Returns:
            Number of individual results removed.
        """
        cleared = 0
        runs_to_check = [run_name] if run_name else list(self._runs.keys())

        for rn in runs_to_check:
            if rn not in self._runs:
                continue

            if question_ids is None:
                cleared += len(self._runs[rn].results)
                del self._runs[rn]
            else:
                original = self._runs[rn].results
                kept = [r for r in original if r.metadata.question_id not in question_ids]
                cleared += len(original) - len(kept)
                if kept:
                    self._runs[rn] = VerificationResultSet(results=kept)
                else:
                    del self._runs[rn]

        logger.info("Cleared %d results", cleared)
        return cleared

    # ========================================================================
    # Summary and Statistics
    # ========================================================================

    def get_summary(self, run_name: str | None = None) -> dict[str, Any]:
        """Return aggregate statistics across stored results.

        Args:
            run_name: If provided, scope statistics to this run.

        Returns:
            Dict with keys: total_results, total_runs, unique_questions,
            completed_count, failed_count, success_rate.
        """
        if run_name is not None:
            if run_name not in self._runs:
                return self._empty_summary()
            runs = {run_name: self._runs[run_name]}
        else:
            runs = self._runs

        if not runs:
            return self._empty_summary()

        all_results = [r for rs in runs.values() for r in rs.results]
        total = len(all_results)
        completed = sum(1 for r in all_results if r.metadata.completed_without_errors)
        failed = total - completed
        question_ids = {r.metadata.question_id for r in all_results}

        return {
            "total_results": total,
            "total_runs": len(runs),
            "unique_questions": len(question_ids),
            "completed_count": completed,
            "failed_count": failed,
            "success_rate": (completed / total * 100) if total else 0.0,
        }

    def get_statistics_by_run(self) -> dict[str, dict[str, Any]]:
        """Return per-run summary statistics.

        Returns:
            Dict mapping each run name to its summary dict
            (same shape as ``get_summary``).
        """
        return {rn: self.get_summary(run_name=rn) for rn in self._runs}

    # ========================================================================
    # Export and Import
    # ========================================================================

    def export(
        self,
        question_ids: list[str] | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        """Export results as a JSON-serializable dict.

        Args:
            question_ids: If provided, include only results with these
                question IDs.
            run_name: If provided, include only this run.

        Returns:
            Dict with a ``runs`` key mapping run names to lists of
            serialized result dicts. If any scenario execution results
            carry ``outcome_results``, a ``scenario_outcomes`` key is
            added at the top level mapping run name to a dict of
            ``scenario_id -> outcome_results``.
        """
        runs_to_export = {run_name: self._runs[run_name]} if run_name and run_name in self._runs else dict(self._runs)

        exported_runs: dict[str, list[dict[str, Any]]] = {}
        scenario_outcomes: dict[str, dict[str, dict[str, bool | int | float]]] = {}
        for rn, result_set in runs_to_export.items():
            results_list = []
            for result in result_set.results:
                if question_ids and result.metadata.question_id not in question_ids:
                    continue
                results_list.append(result.model_dump())
            if results_list:
                exported_runs[rn] = results_list

            if result_set.scenario_results:
                per_scenario: dict[str, dict[str, bool | int | float]] = {}
                for er in result_set.scenario_results:
                    outcomes = getattr(er, "outcome_results", None)
                    if outcomes:
                        per_scenario[er.scenario_id] = dict(outcomes)
                if per_scenario:
                    scenario_outcomes[rn] = per_scenario

        output: dict[str, Any] = {"runs": exported_runs}
        if scenario_outcomes:
            output["scenario_outcomes"] = scenario_outcomes
        return output

    def export_to_file(
        self,
        file_path: str | Path,
        **kwargs: Any,
    ) -> None:
        """Write exported results to a JSON file.

        Args:
            file_path: Destination path for the JSON file.
            **kwargs: Passed through to ``export()``.
        """
        file_path = Path(file_path)
        data = self.export(**kwargs)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Exported results to %s", file_path)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "ResultsStore":
        """Load a ResultsStore from a previously exported JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            A new ResultsStore populated with the loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file content is not valid.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if "runs" not in data:
            raise ValueError("Invalid results file: missing 'runs' key")

        store = cls()
        for rn, results_list in data["runs"].items():
            result_objects = [VerificationResult.model_validate(rd) for rd in results_list]
            result_set = VerificationResultSet(results=result_objects)
            store._runs[rn] = result_set

        logger.info(
            "Loaded %d runs from %s",
            len(store._runs),
            file_path,
        )
        return store

    # ========================================================================
    # Private Helpers
    # ========================================================================

    @staticmethod
    def _empty_summary() -> dict[str, Any]:
        """Return a summary dict with zero values.

        Returns:
            Dict with all summary fields set to zero.
        """
        return {
            "total_results": 0,
            "total_runs": 0,
            "unique_questions": 0,
            "completed_count": 0,
            "failed_count": 0,
            "success_rate": 0.0,
        }

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        total = sum(len(rs.results) for rs in self._runs.values())
        return f"ResultsStore(runs={len(self._runs)}, total_results={total})"

    def __len__(self) -> int:
        """Return total number of individual results across all runs."""
        return sum(len(rs.results) for rs in self._runs.values())
