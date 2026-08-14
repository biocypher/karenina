"""Tests for validated complete result-set loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from karenina.benchmark import ResultsIOManager
from karenina.schemas.results import ScenarioResultRecord, ScenarioTurnRecord
from tests.schemas._metadata_factory import make_metadata
from tests.unit.benchmark.core._result_factory import make_result


def _scenario_payload() -> dict[str, object]:
    """Return a valid compact scenario record payload."""
    return ScenarioResultRecord(
        scenario_id="scenario-1",
        status="completed",
        path=["ask"],
        turn_count=1,
        history=[
            ScenarioTurnRecord(
                node_id="ask",
                question_text="Question?",
                raw_response="Answer.",
                parsed_fields={"answer": "Answer"},
                verify_result=True,
                verification_result_id="result-1",
            )
        ],
        outcome_results={"initial_correct": True},
    ).model_dump(mode="json")


@pytest.mark.unit
class TestLoadResultSetFromJson:
    """Validate results, metadata, and scenario records together."""

    def test_loads_complete_scenario_export(self, tmp_path: Path) -> None:
        result = make_result(metadata=make_metadata())
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                {
                    "metadata": {"regime": "mcp", "difficulty": "hard"},
                    "results": [result.model_dump(mode="json")],
                    "scenario_results": [_scenario_payload()],
                }
            ),
            encoding="utf-8",
        )

        loaded = ResultsIOManager.load_result_set_from_json(path)

        assert len(loaded.results) == 1
        assert loaded.metadata == {"regime": "mcp", "difficulty": "hard"}
        assert loaded.scenario_results is not None
        assert loaded.scenario_results[0].scenario_id == "scenario-1"
        assert len(loaded.get_scenario_results().to_turn_dataframe()) == 1

    def test_rejects_malformed_scenario_record(self, tmp_path: Path) -> None:
        result = make_result(metadata=make_metadata())
        scenario = _scenario_payload()
        scenario["unexpected"] = True
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                {
                    "metadata": {},
                    "results": [result.model_dump(mode="json")],
                    "scenario_results": [scenario],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Invalid scenario result at index 0"):
            ResultsIOManager.load_result_set_from_json(path)

    def test_rejects_malformed_verification_row(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        path.write_text(json.dumps({"metadata": {}, "results": [{"bad": "row"}]}), encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid verification result at index 0"):
            ResultsIOManager.load_result_set_from_json(path)
