"""Tests for deterministic response-shape and failure-tree transformations."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from karenina.schemas.config import ModelConfig
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from paper.otp_response_characterization.analyses.failure_tree import (
    FLAG_COLUMNS,
    KEY_COLUMNS,
    build_failure_tree_edges,
    build_instance_flags,
    score_response_shapes,
    summarize_response_shapes,
)


def _row(
    question_id: str,
    regime: str,
    outcome: str,
    trace: str,
    parser: str = "claude-opus-4-6",
) -> VerificationResult:
    answering = ModelIdentity(
        interface="openai_endpoint",
        model_name="model-a",
        tools=["otp"] if regime == "mcp" else [],
    )
    parsing = ModelIdentity(interface="openai_endpoint", model_name=parser)
    categories = {
        "fail-content": FailureCategory.CONTENT,
        "abstain": FailureCategory.ABSTENTION,
        "infra": FailureCategory.UNEXPECTED_ERROR,
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
        template=VerificationResultTemplate(
            raw_llm_response=trace,
            verify_result=outcome == "pass",
        ),
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
class TestResponseShapeScoring:
    def test_parser_siblings_collapse_to_exclusive_flags(self) -> None:
        blank = "--- AI Message ---\n"
        scores = score_response_shapes(
            [
                _row("q1", "mcp", "infra", blank),
                _row("q1", "mcp", "infra", blank, parser="judge-b"),
                _row("q2", "mcp", "infra", ""),
                _row("q3", "mcp", "infra", "--- Tool Message (call_id:x) ---\nresult"),
            ],
            "mcp",
            _parsing_model(),
        )
        flags = build_instance_flags(scores)

        assert len(flags) == 3
        assert flags[KEY_COLUMNS].duplicated().sum() == 0
        assert flags[["no_usable_output", "blank_final_assistant", "tool_loop_cutoff"]].sum(axis=1).eq(1).all()
        assert flags["technical_failure"].all()

    def test_summaries_distinguish_judge_and_generated_answer_counts(self) -> None:
        blank = "--- AI Message ---\n"
        scores = score_response_shapes(
            [
                _row("q1", "mcp", "infra", blank),
                _row("q1", "mcp", "infra", blank, parser="judge-b"),
                _row("q2", "mcp", "pass", "answer"),
            ],
            "mcp",
            _parsing_model(),
        )
        summary, unique, by_answerer = summarize_response_shapes(scores)
        blank_summary = summary[summary["trait"] == "EmptyTrailingAI"].iloc[0]
        blank_unique = unique[unique["trait"] == "EmptyTrailingAI"].iloc[0]

        assert blank_summary["hits"] == 2
        assert blank_unique["unique_instances"] == 1
        assert by_answerer[by_answerer["trait"] == "EmptyTrailingAI"]["unique_instances"].sum() == 1


@pytest.mark.unit
class TestFailureTree:
    def test_every_parent_is_partitioned_by_its_children(self) -> None:
        blank = "--- AI Message ---\n"
        rows = [
            _row("p-content", "parametric", "fail-content", "wrong"),
            _row("p-abstain", "parametric", "abstain", "declined"),
            _row("p-technical", "parametric", "infra", blank),
            _row("m-content", "mcp", "fail-content", "wrong"),
            _row("m-abstain", "mcp", "abstain", "declined"),
            _row("m-empty", "mcp", "infra", ""),
            _row("m-blank", "mcp", "infra", blank),
            _row("m-cutoff", "mcp", "infra", "--- Tool Message (call_id:x) ---\nresult"),
            _row("m-pass", "mcp", "pass", "correct"),
        ]
        scores = pd.concat(
            [
                score_response_shapes(
                    [row for row in rows if not row.metadata.answering.tools],
                    "parametric",
                    _parsing_model(),
                ),
                score_response_shapes(
                    [row for row in rows if row.metadata.answering.tools],
                    "mcp",
                    _parsing_model(),
                ),
            ],
            ignore_index=True,
        )
        flags = build_instance_flags(scores)
        edges = build_failure_tree_edges(
            scores,
            flags,
            {"answer_present_no_final_message": 1},
        )

        counts = edges.set_index("child")["count"].to_dict()
        assert counts["parametric_failed"] == 3
        assert counts["mcp_failed"] == 5
        assert counts["mcp_malformed"] == 3
        assert (
            counts["mcp_empty_trace"] + counts["mcp_blank_final"] + counts["mcp_tool_cutoff"] == counts["mcp_malformed"]
        )
        assert (
            counts["mcp_no_answer"] + counts["mcp_wrong_result"] + counts["mcp_answer_present"]
            == counts["mcp_blank_final"]
        )
        assert list(edges.columns) == ["edge_order", "parent", "child", "condition", "count"]

    def test_rejects_nonexclusive_flags(self) -> None:
        scores = score_response_shapes(
            [_row("q1", "mcp", "infra", "")],
            "mcp",
            _parsing_model(),
        )
        flags = build_instance_flags(scores)
        flags.loc[0, FLAG_COLUMNS] = True
        with pytest.raises(ValueError, match="MCP technical split"):
            build_failure_tree_edges(scores, flags, {})
