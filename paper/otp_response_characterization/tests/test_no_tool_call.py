"""Tests for deterministic no-tool-call transformations."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from karenina.schemas.config import ModelConfig
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from paper.otp_response_characterization.analyses.no_tool_call import (
    build_no_tool_summary,
    score_no_tool_rows,
)


def _row(question_id: str, outcome: str, trace: str, answerer: str) -> VerificationResult:
    answering = ModelIdentity(interface="openai_endpoint", model_name=answerer, tools=["otp"])
    parsing = ModelIdentity(interface="openai_endpoint", model_name="claude-opus-4-6")
    categories = {
        "fail-content": FailureCategory.CONTENT,
        "abstain": FailureCategory.ABSTENTION,
        "infra": FailureCategory.UNEXPECTED_ERROR,
        "mystery": FailureCategory.UNEXPECTED_ERROR,
    }
    failure = Failure(category=categories[outcome], stage="test", reason=outcome) if outcome != "pass" else None
    timestamp = datetime.now(UTC).isoformat()
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="template",
            failure=failure,
            question_text="Question",
            answering=answering,
            parsing=parsing,
            execution_time=0.1,
            timestamp=timestamp,
            result_id=VerificationResultMetadata.compute_result_id(
                question_id=question_id,
                answering=answering,
                parsing=parsing,
                timestamp=timestamp,
                replicate=1,
            ),
            run_name="test",
            replicate=1,
        ),
        template=VerificationResultTemplate(raw_llm_response=trace),
    )


def _parsing_model() -> ModelConfig:
    return ModelConfig(
        id="judge",
        model_provider="openai",
        model_name="judge",
        interface="openai_endpoint",
        endpoint_base_url="http://localhost:9999",
        endpoint_api_key="EMPTY",
    )


@pytest.mark.unit
class TestNoToolCall:
    def test_excludes_empty_traces_and_partitions_outcomes(self) -> None:
        scores = score_no_tool_rows(
            [
                _row("pass", "pass", "answer", "gpt-oss-120b"),
                _row("fail", "fail-content", "wrong", "qwen3.5-a3b"),
                _row("abstain", "abstain", "declined", "claude-sonnet-4-6"),
                _row("infra", "infra", "cut short", "qwen3.6-a3b"),
                _row("empty", "infra", "", "claude-haiku-4-5-20251001"),
                _row(
                    "tool",
                    "pass",
                    "--- Tool Message (call_id:x) ---\nresult",
                    "claude-opus-4-6",
                ),
            ],
            _parsing_model(),
        )
        summary = build_no_tool_summary(scores).set_index("metric")["value"]

        assert summary["mcp_no_tool_nonempty_count"] == 4
        assert summary["mcp_empty_trace_count"] == 1
        partition = sum(
            summary[name]
            for name in (
                "mcp_no_tool_pass_count",
                "mcp_no_tool_fail_count",
                "mcp_no_tool_abstain_count",
                "mcp_no_tool_infra_count",
            )
        )
        assert partition == summary["mcp_no_tool_nonempty_count"]
        answerer_passes = summary[
            summary.index.str.endswith("_no_tool_pass_count")
            & ~summary.index.str.startswith("mcp_")
            & ~summary.index.str.contains("_ref_judge_")
        ].sum()
        assert answerer_passes == summary["mcp_no_tool_pass_count"]

    def test_rejects_unknown_outcome(self) -> None:
        scores = score_no_tool_rows(
            [_row("q1", "mystery", "answer", "gpt-oss-120b")],
            _parsing_model(),
        )
        scores.loc[0, "outcome_class"] = "mystery"
        with pytest.raises(ValueError, match="mystery"):
            build_no_tool_summary(scores)
