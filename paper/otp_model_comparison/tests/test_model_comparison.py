"""Synthetic tests for the Open Targets model comparison."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    ModelIdentity,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from paper.otp_model_comparison.analysis import LONG_FORM_COLUMNS, build_long_form, outcome_class, write_analysis
from paper.otp_model_comparison.config import ANSWERER_NAMES, build_config
from paper.otp_model_comparison.run import _mcp_specs


def _row(*, regime: str, failed: bool, tokens: int = 10) -> VerificationResult:
    answering = ModelIdentity(
        interface="openai_endpoint",
        model_name="gpt-oss-120b",
        tools=["otp"] if regime == "mcp" else [],
    )
    parsing = ModelIdentity(interface="langchain", model_name="claude-opus-4-6")
    timestamp = datetime.now(UTC).isoformat()
    failure = Failure(category=FailureCategory.CONTENT, stage="VerifyTemplate", reason="wrong") if failed else None
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id="q1",
            template_id="template",
            question_text="Question",
            answering=answering,
            parsing=parsing,
            replicate=1,
            run_name=regime,
            execution_time=1.0,
            timestamp=timestamp,
            failure=failure,
            result_id=VerificationResultMetadata.compute_result_id(
                question_id="q1",
                answering=answering,
                parsing=parsing,
                replicate=1,
                timestamp=timestamp,
            ),
        ),
        template=VerificationResultTemplate(
            raw_llm_response="answer",
            trace_messages=[{"role": "assistant", "content": "answer"}],
            usage_metadata={"answer_generation": {"total_tokens": tokens}},
        ),
    )


@pytest.mark.unit
class TestModelComparisonConfig:
    def test_builds_square_configured_roster(self) -> None:
        config = build_config(None)

        assert len(config.answering_models) == len(ANSWERER_NAMES)
        assert len(config.parsing_models) == len(ANSWERER_NAMES)
        assert config.replicate_count == 3
        assert config.evaluation_mode == "template_only"
        assert config.abstention_enabled is True
        assert config.sufficiency_enabled is False
        assert config.request_timeout == 180.0
        assert config.async_enabled is True
        assert config.async_max_workers == 16
        assert config.retry_policy.timeout.max_attempts == 4
        assert config.retry_policy.connection.max_attempts == 5
        assert config.retry_policy.timeout_escalation is not None
        assert config.retry_policy.timeout_escalation.max_timeout == 900.0
        assert isinstance(config.answerer_concurrency_limits, dict)
        assert set(config.answerer_concurrency_limits) == {
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
        }

    def test_mcp_arm_attaches_one_url_per_answerer(self) -> None:
        urls = {name: f"http://127.0.0.1:{8000 + index}/mcp" for index, name in enumerate(ANSWERER_NAMES)}

        config = build_config(urls)

        assert [model.mcp_urls_dict for model in config.answering_models] == [
            {"otp": urls[name]} for name in ANSWERER_NAMES
        ]
        assert all(model.mcp_urls_dict is None for model in config.parsing_models)
        for model in config.answering_models:
            assert model.agent_timeout == 900
            assert model.mcp_http_timeout == 240.0
            assert model.mcp_sse_read_timeout == 600.0
            assert model.agent_middleware is not None
            assert model.agent_middleware.limits.model_call_limit == 30
            assert model.agent_middleware.limits.tool_call_limit == 60
            assert model.agent_middleware.model_retry.max_retries == 10
            assert model.agent_middleware.tool_retry.max_retries == 1

    def test_mcp_server_specs_use_configured_timeouts(self, tmp_path: Path) -> None:
        spec = _mcp_specs(["gpt-oss-120b"], tmp_path)[0]

        assert spec.startup_timeout == 300.0
        assert spec.command[-2:] == ("--timeout", "180")


@pytest.mark.unit
class TestModelComparisonAnalysis:
    def test_classifies_content_failure_and_builds_schema(self) -> None:
        passed = _row(regime="parametric", failed=False)
        failed = _row(regime="mcp", failed=True)

        assert outcome_class(passed) == "pass"
        assert outcome_class(failed) == "fail-content"
        frame = build_long_form([passed], "parametric")
        assert list(frame.columns) == LONG_FORM_COLUMNS
        assert frame.iloc[0]["tokens_answerer"] == 10

    def test_empty_arm_keeps_analysis_schema(self, tmp_path: Path) -> None:
        parametric = build_long_form([_row(regime="parametric", failed=False)], "parametric")
        mcp = build_long_form([], "mcp")

        write_analysis(parametric, mcp, tmp_path)

        long_form = pd.read_csv(tmp_path / "results_long_form.tsv", sep="\t")
        assert set(long_form["regime"]) == {"parametric"}

    def test_writes_analysis_tables_with_consistent_counts(self, tmp_path: Path) -> None:
        parametric = build_long_form([_row(regime="parametric", failed=False)], "parametric")
        mcp = build_long_form([_row(regime="mcp", failed=True)], "mcp")

        write_analysis(parametric, mcp, tmp_path)

        long_form = pd.read_csv(tmp_path / "results_long_form.tsv", sep="\t")
        counts = pd.read_csv(tmp_path / "outcome_counts.tsv", sep="\t")
        assert len(long_form) == 2
        assert counts["rows"].sum() == 2
        assert {path.name for path in tmp_path.iterdir()} == {
            "results_long_form.tsv",
            "outcome_counts.tsv",
            "pass_rate_by_replicate.tsv",
            "answer_tokens.tsv",
            "question_pass_counts.tsv",
        }
