"""Tests for trace-length and token transformations."""

from __future__ import annotations

import pandas as pd
import pytest

from paper.otp_response_characterization.analyses.trace_tokens import (
    common_correct_questions,
    reference_rows,
    right_wrong_tokens,
    token_lookup,
    trace_length_stats,
    trace_length_summary,
)


def _longform() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    outcomes = {
        ("q1", "m1", 1): "pass",
        ("q1", "m1", 2): "pass",
        ("q1", "m2", 1): "pass",
        ("q1", "m2", 2): "pass",
        ("q2", "m1", 1): "pass",
        ("q2", "m1", 2): "pass",
        ("q2", "m2", 1): "pass",
        ("q2", "m2", 2): "fail-content",
    }
    for (question_id, answerer, replicate), outcome in outcomes.items():
        base = {
            "question_id": question_id,
            "answerer": answerer,
            "regime": "mcp",
            "replicate": replicate,
            "outcome_class": outcome,
            "trace_length": 4 + replicate,
        }
        rows.append({**base, "judge": "claude-opus-4-6", "tokens_answerer": None})
        rows.append(
            {
                **base,
                "judge": "token-source",
                "tokens_answerer": 100 * replicate + (50 if answerer == "m2" else 0),
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.unit
class TestTraceTokens:
    def test_sibling_tokens_enrich_reference_rows(self) -> None:
        reference = reference_rows(_longform())
        assert len(reference) == 8
        assert reference["tokens_answerer"].notna().all()

    def test_conflicting_sibling_tokens_fail(self) -> None:
        frame = _longform()
        duplicate = frame.iloc[[1]].copy()
        duplicate["judge"] = "other-token-source"
        duplicate["tokens_answerer"] = 999
        with pytest.raises(ValueError, match="conflicting"):
            token_lookup(pd.concat([frame, duplicate], ignore_index=True))

    def test_common_correct_and_summaries_use_current_input(self) -> None:
        reference = reference_rows(_longform())
        assert common_correct_questions(reference, ["m1", "m2"]) == {"q1"}

        stats = trace_length_stats(reference, ["m1", "m2"])
        summary = trace_length_summary(reference, ["m1", "m2"])
        tokens = right_wrong_tokens(reference, ["m1", "m2"])

        assert stats.set_index("answerer")["n"].to_dict() == {"m1": 2, "m2": 2}
        assert list(summary.columns) == ["answerer", "mean", "median", "p25", "p75"]
        m2 = tokens.set_index("answerer").loc["m2"]
        assert m2["n_right"] == 3
        assert m2["n_wrong"] == 1
