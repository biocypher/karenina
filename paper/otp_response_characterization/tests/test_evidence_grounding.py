"""Tests for evidence-grounding panel construction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from paper.common.qa_results import REFERENCE_JUDGE
from paper.otp_response_characterization.analyses.evidence_grounding import (
    KEY_COLUMNS,
    build_panel_rows,
    load_evidence_scores,
)


def _response_rows() -> pd.DataFrame:
    statuses = [
        ("outside", "other", "pass", False),
        ("incorrect", REFERENCE_JUDGE, "fail-content", False),
        ("regex", REFERENCE_JUDGE, "pass", True),
        ("tool-less", REFERENCE_JUDGE, "pass", False),
        ("unscored", REFERENCE_JUDGE, "pass", False),
        ("grounded", REFERENCE_JUDGE, "pass", False),
        ("ungrounded", REFERENCE_JUDGE, "pass", False),
    ]
    return pd.DataFrame(
        [
            {
                "question_id": name,
                "answerer": "answerer",
                "judge": judge,
                "replicate": 1,
                "outcome_class": outcome,
                "EmptyTrace": regex,
                "EmptyTrailingAI": False,
                "NoAIFinalMessage": False,
            }
            for name, judge, outcome, regex in statuses
        ]
    )


def _no_tool_rows(response: pd.DataFrame) -> pd.DataFrame:
    frame = response[KEY_COLUMNS].copy()
    frame["no_tool_call"] = frame["question_id"] == "tool-less"
    frame["empty_trace"] = False
    return frame


@pytest.mark.unit
class TestEvidenceGrounding:
    """Validate null handling and the mutually exclusive panel partition."""

    def test_panel_status_precedence_is_exclusive(self) -> None:
        response = _response_rows()
        scores = {
            (str(row.question_id), str(row.answerer), str(row.judge), int(row.replicate)): (
                True
                if row.question_id == "grounded"
                else False
                if row.question_id == "ungrounded"
                else None
            )
            for row in response.itertuples(index=False)
        }
        panel = build_panel_rows(response, _no_tool_rows(response), scores)
        assert dict(zip(panel["question_id"], panel["panel_status"], strict=True)) == {
            "outside": "outside_reference_judge",
            "incorrect": "not_correct",
            "regex": "regex_characterized",
            "tool-less": "tool_less",
            "unscored": "unscored",
            "grounded": "grounded",
            "ungrounded": "ungrounded",
        }

    def test_score_loader_preserves_null(self, tmp_path: Path) -> None:
        path = tmp_path / "scores.jsonl"
        record = {
            "key": {
                "question_id": "q1",
                "answering": {"model_name": "answerer"},
                "parsing": {"model_name": "judge"},
                "replicate": 1,
                "result_id": "result",
            },
            "rubric_addon": None,
        }
        path.write_text(json.dumps(record) + "\n")
        assert load_evidence_scores(path) == {("q1", "answerer", "judge", 1): None}

    def test_incomplete_score_join_fails(self) -> None:
        response = _response_rows()
        with pytest.raises(ValueError, match="incomplete"):
            build_panel_rows(response, _no_tool_rows(response), {})
