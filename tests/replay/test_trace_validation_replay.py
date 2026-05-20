"""Replay-specific behavior for MCP trace validation."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.benchmark.verification.stages.pipeline.trace_validation_autofail import TraceValidationAutoFailStage
from karenina.replay import ReplayEntry
from karenina.schemas.config import ModelConfig


@pytest.mark.integration
def test_mcp_replay_with_parsed_fields_skips_tool_only_trace_autofail():
    context = VerificationContext(
        question_id="q",
        template_id="t",
        question_text="hi",
        template_code="class Answer: pass",
        answering_model=ModelConfig(
            id="qwen",
            model_name="qwen",
            interface="openai_endpoint",
            endpoint_base_url="http://vllm:8000",
            endpoint_api_key="EMPTY",
            mcp_urls_dict={"otp": "http://127.0.0.1:8765/mcp"},
        ),
        parsing_model=ModelConfig(id="parser", model_name="parser", model_provider="anthropic"),
    )
    context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "--- AI Message ---\n\nTool Calls:\n  search(...)")
    context.set_artifact(
        ArtifactKeys.REPLAY_ENTRY,
        ReplayEntry(raw_trace="ignored", parsed_answer_fields={"answer": True}),
    )

    TraceValidationAutoFailStage().execute(context)

    assert context.get_artifact(ArtifactKeys.TRACE_VALIDATION_FAILED) is False
    assert context.get_artifact(ArtifactKeys.TRACE_VALIDATION_ERROR) is None
    assert context.get_result_field(ArtifactKeys.FAILED_STAGE) is None
    assert context.get_result_field(ArtifactKeys.VERIFY_RESULT) is None


@pytest.mark.integration
def test_mcp_replay_with_captured_verdict_hydrates_verify_result():
    context = VerificationContext(
        question_id="q",
        template_id="t",
        question_text="hi",
        template_code="class Answer: pass",
        answering_model=ModelConfig(
            id="qwen",
            model_name="qwen",
            interface="openai_endpoint",
            endpoint_base_url="http://vllm:8000",
            endpoint_api_key="EMPTY",
            mcp_urls_dict={"otp": "http://127.0.0.1:8765/mcp"},
        ),
        parsing_model=ModelConfig(id="parser", model_name="parser", model_provider="anthropic"),
    )
    context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "--- AI Message ---\n\nTool Calls:\n  search(...)")
    context.set_artifact(
        ArtifactKeys.REPLAY_ENTRY,
        ReplayEntry(raw_trace="ignored", verify_result=False),
    )

    TraceValidationAutoFailStage().execute(context)

    assert context.get_artifact(ArtifactKeys.TRACE_VALIDATION_FAILED) is False
    assert context.get_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT) is False
    assert context.get_result_field(ArtifactKeys.VERIFY_RESULT) is False
    assert context.get_result_field(ArtifactKeys.FAILED_STAGE) is None
