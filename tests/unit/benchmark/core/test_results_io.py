"""CSV round-trip tests for failure + caveats shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from karenina.benchmark.core.results_io import ResultsIOManager
from karenina.schemas.results.caveat import Caveat
from karenina.schemas.results.failure import Failure, FailureCategory
from tests.schemas._metadata_factory import make_metadata
from tests.unit.benchmark.core._result_factory import make_result


@pytest.mark.unit
class TestResultsIOCSVRoundTrip:
    """CSV round-trip must preserve ``Failure`` and ``Caveat`` metadata."""

    def test_csv_roundtrip_preserves_failure_and_caveats(self, tmp_path: Path) -> None:
        md = make_metadata(
            failure=Failure(
                category=FailureCategory.TIMEOUT,
                stage="generate_answer",
                reason="gone",
                details={"error_message": "timed out after 120s"},
            ),
            caveats=[Caveat.RETRIES_USED, Caveat.PARTIAL_CONTENT],
        )
        result = make_result(metadata=md)

        path = tmp_path / "r.csv"
        csv_output = ResultsIOManager.export_to_csv({md.question_id: result})
        path.write_text(csv_output, encoding="utf-8")
        loaded = ResultsIOManager.load_from_csv(path)

        # Loaded key is f"{question_id}_{row_index}", e.g. "q1_1".
        expected_key = f"{md.question_id}_1"
        assert expected_key in loaded
        round_trip_md = loaded[expected_key].metadata
        assert round_trip_md.failure is not None
        assert round_trip_md.failure.category is FailureCategory.TIMEOUT
        assert round_trip_md.failure.stage == "generate_answer"
        assert round_trip_md.failure.reason == "gone"
        assert round_trip_md.failure.details == {"error_message": "timed out after 120s"}
        assert Caveat.RETRIES_USED in round_trip_md.caveats
        assert Caveat.PARTIAL_CONTENT in round_trip_md.caveats

    def test_csv_pass_has_empty_failure_columns(self, tmp_path: Path) -> None:
        md = make_metadata(failure=None, caveats=[])
        result = make_result(metadata=md)

        path = tmp_path / "r.csv"
        csv_output = ResultsIOManager.export_to_csv({md.question_id: result})
        path.write_text(csv_output, encoding="utf-8")
        loaded = ResultsIOManager.load_from_csv(path)

        expected_key = f"{md.question_id}_1"
        round_trip_md = loaded[expected_key].metadata
        assert round_trip_md.failure is None
        assert round_trip_md.caveats == []

    def test_csv_header_uses_failure_columns(self) -> None:
        """The CSV header must expose failure_* + caveats columns, not legacy ones."""
        md = make_metadata(failure=None, caveats=[])
        result = make_result(metadata=md)

        csv_output = ResultsIOManager.export_to_csv({md.question_id: result})
        header_line = csv_output.splitlines()[0]
        header_cols = header_line.split(",")

        assert "failure_category" in header_cols
        assert "failure_group" in header_cols
        assert "failure_stage" in header_cols
        assert "failure_reason" in header_cols
        assert "failure_details_json" in header_cols
        assert "caveats" in header_cols

        for legacy in ("error", "error_category", "failed_stage", "completed_without_errors"):
            assert legacy not in header_cols, f"Legacy column {legacy!r} must be removed"
