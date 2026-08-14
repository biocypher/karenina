"""DataFrame builders for multi-turn scenario results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCENARIO_COLUMNS = [
    "scenario_id",
    "status",
    "scenario_path",
    "turn_count",
    "replicate",
    "terminal_failure_node",
    "terminal_failure_category",
    "terminal_failure_stage",
    "terminal_failure_reason",
]

TURN_COLUMNS = [
    "scenario_id",
    "status",
    "replicate",
    "scenario_turn",
    "node_id",
    "question_text",
    "raw_response",
    "parsed_fields",
    "verify_result",
    "verification_result_id",
]

OUTCOME_COLUMNS = [
    "scenario_id",
    "status",
    "replicate",
    "outcome_name",
    "outcome_value",
    "outcome_type",
]


def _value(obj: object, name: str, default: Any = None) -> Any:
    """Read a field from an object or mapping."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _outcome_type(value: bool | int | float) -> str:
    """Return a stable scalar type label for an outcome value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    return "number"


class ScenarioDataFrameBuilder:
    """Convert live or loaded scenario execution results to DataFrames.

    Args:
        results: Scenario execution objects or validated compact scenario
            records.
    """

    def __init__(self, results: list[object]) -> None:
        """Initialize the builder with scenario execution results.

        Args:
            results: Live scenario objects or validated compact records.
        """
        self._results = results

    def build_scenario_dataframe(self) -> Any:
        """Build one row per scenario execution.

        Outcome values are added as columns prefixed with ``outcome_``.

        Returns:
            A pandas DataFrame with scenario status, path, failures, and
            outcomes.
        """
        import pandas as pd

        outcome_names = sorted(
            {str(name) for result in self._results for name in (_value(result, "outcome_results", {}) or {})}
        )
        columns = [*SCENARIO_COLUMNS, *(f"outcome_{name}" for name in outcome_names)]
        rows: list[dict[str, Any]] = []
        for result in self._results:
            terminal_failure = _value(result, "terminal_failure")
            path = _value(result, "path", []) or []
            outcomes = _value(result, "outcome_results", {}) or {}
            row = {
                "scenario_id": _value(result, "scenario_id"),
                "status": _value(result, "status"),
                "scenario_path": "->".join(str(node) for node in path),
                "turn_count": _value(result, "turn_count"),
                "replicate": _value(result, "replicate"),
                "terminal_failure_node": _value(terminal_failure, "node_id"),
                "terminal_failure_category": _value(terminal_failure, "category"),
                "terminal_failure_stage": _value(terminal_failure, "stage"),
                "terminal_failure_reason": _value(terminal_failure, "reason"),
            }
            row.update({f"outcome_{name}": outcomes.get(name) for name in outcome_names})
            rows.append(row)

        frame = pd.DataFrame(rows, columns=columns)
        for column in ("turn_count", "replicate"):
            frame[column] = frame[column].astype(pd.Int64Dtype())
        return frame

    def build_turn_dataframe(self) -> Any:
        """Build one row per executed scenario turn.

        Returns:
            A pandas DataFrame containing turn content and result links.
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for result in self._results:
            for turn_index, turn in enumerate(_value(result, "history", []) or []):
                rows.append(
                    {
                        "scenario_id": _value(result, "scenario_id"),
                        "status": _value(result, "status"),
                        "replicate": _value(result, "replicate"),
                        "scenario_turn": turn_index,
                        "node_id": _value(turn, "node_id"),
                        "question_text": _value(turn, "question_text"),
                        "raw_response": _value(turn, "raw_response"),
                        "parsed_fields": _value(turn, "parsed_fields", {}) or {},
                        "verify_result": _value(turn, "verify_result"),
                        "verification_result_id": _value(turn, "verification_result_id"),
                    }
                )

        frame = pd.DataFrame(rows, columns=TURN_COLUMNS)
        for column in ("replicate", "scenario_turn"):
            frame[column] = frame[column].astype(pd.Int64Dtype())
        return frame

    def build_outcome_dataframe(self) -> Any:
        """Build one row per evaluated scenario outcome criterion.

        Returns:
            A pandas DataFrame with scalar outcome values and type labels.
        """
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for result in self._results:
            outcomes = _value(result, "outcome_results", {}) or {}
            for name, value in sorted(outcomes.items()):
                rows.append(
                    {
                        "scenario_id": _value(result, "scenario_id"),
                        "status": _value(result, "status"),
                        "replicate": _value(result, "replicate"),
                        "outcome_name": name,
                        "outcome_value": value,
                        "outcome_type": _outcome_type(value),
                    }
                )

        frame = pd.DataFrame(rows, columns=OUTCOME_COLUMNS)
        frame["replicate"] = frame["replicate"].astype(pd.Int64Dtype())
        return frame
