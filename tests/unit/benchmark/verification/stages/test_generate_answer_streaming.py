"""Tests for streaming path and agent timeout handling in GenerateAnswerStage."""

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.benchmark.verification.stages.pipeline.generate_answer import GenerateAnswerStage
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.llm import LLMResponse
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config import ModelConfig

# The generate_answer stage imports AdapterRegistry inline (lazy import inside execute()),
# so we must patch it at its source module rather than at the stage module.
_REGISTRY_PATCH_TARGET = "karenina.adapters.registry.AdapterRegistry.get_spec"


def _make_context() -> VerificationContext:
    """Create a minimal VerificationContext for streaming/timeout tests."""
    model = ModelConfig(
        id="test",
        model_name="test-model",
        model_provider="openai",
        system_prompt="You are helpful.",
        request_timeout=30.0,
    )
    return VerificationContext(
        question_id="q1",
        template_id="t1",
        question_text="What is 2+2?",
        template_code="class Answer(BaseAnswer): value: str",
        answering_model=model,
        parsing_model=model,
    )


def _make_mock_llm(*, supports_streaming: bool = True) -> MagicMock:
    """Create a mock LLMPort with configurable streaming support."""
    mock = MagicMock()
    mock.capabilities = PortCapabilities(supports_streaming=supports_streaming)
    return mock


def _make_agent_result(
    *,
    timeout_reached: bool = False,
    raw_trace: str = "--- AI Message ---\nSome response",
    turns: int = 1,
    limit_reached: bool = False,
) -> MagicMock:
    """Create a mock AgentResult with configurable timeout behavior."""
    result = MagicMock()
    result.timeout_reached = timeout_reached
    result.raw_trace = raw_trace
    result.limit_reached = limit_reached
    result.turns = turns
    result.usage = None
    result.trace_messages = None
    return result


@pytest.mark.unit
class TestStreamingLLMPath:
    """Tests for the LLMPort streaming invocation path in generate_answer."""

    def test_streaming_partial_response_sets_timeout_and_usage_artifacts(self) -> None:
        """When stream_invoke returns a partial response, RESPONSE_TIMEOUT_PARTIAL
        and USAGE_UNAVAILABLE artifacts should be set."""
        ctx = _make_context()
        stage = GenerateAnswerStage()
        mock_llm = _make_mock_llm(supports_streaming=True)

        partial_response = LLMResponse(
            content="partial content here",
            usage=UsageMetadata(),
            is_partial=True,
            usage_unavailable=True,
        )
        mock_llm.stream_invoke.return_value = partial_response

        with (
            patch(
                "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
                return_value=mock_llm,
            ),
            patch(_REGISTRY_PATCH_TARGET, return_value=None),
        ):
            stage.execute(ctx)

        # stream_invoke should have been called (not invoke)
        mock_llm.stream_invoke.assert_called_once()
        mock_llm.invoke.assert_not_called()

        # Partial response artifacts
        assert ctx.get_artifact(ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL) is True
        assert ctx.get_result_field("response_timeout_partial") is True

        # Usage unavailable flag
        assert ctx.get_artifact(ArtifactKeys.USAGE_UNAVAILABLE) is True
        assert ctx.get_result_field(ArtifactKeys.USAGE_UNAVAILABLE) is True

        # Raw response should still be captured
        assert "partial content here" in ctx.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)

    def test_streaming_no_content_raises_timeout_error(self) -> None:
        """When stream_invoke returns partial with empty content, a TimeoutError
        should propagate and mark_error on the context."""
        ctx = _make_context()
        stage = GenerateAnswerStage()
        mock_llm = _make_mock_llm(supports_streaming=True)

        empty_partial = LLMResponse(
            content="",
            usage=UsageMetadata(),
            is_partial=True,
            usage_unavailable=True,
        )
        mock_llm.stream_invoke.return_value = empty_partial

        with (
            patch(
                "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
                return_value=mock_llm,
            ),
            patch(_REGISTRY_PATCH_TARGET, return_value=None),
        ):
            stage.execute(ctx)

        # The TimeoutError is caught by the outer except block and marks error
        assert ctx.error is not None
        assert "timed out" in ctx.error.lower() or "TimeoutError" in ctx.error
        assert ctx.completed_without_errors is False

    def test_non_streaming_fallback_uses_invoke(self) -> None:
        """When LLM does not support streaming, invoke() should be called
        instead of stream_invoke()."""
        ctx = _make_context()
        stage = GenerateAnswerStage()
        mock_llm = _make_mock_llm(supports_streaming=False)

        normal_response = LLMResponse(
            content="The answer is 4",
            usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        mock_llm.invoke.return_value = normal_response

        with (
            patch(
                "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
                return_value=mock_llm,
            ),
            patch(_REGISTRY_PATCH_TARGET, return_value=None),
        ):
            stage.execute(ctx)

        # invoke() called, not stream_invoke()
        mock_llm.invoke.assert_called_once()
        mock_llm.stream_invoke.assert_not_called()

        # Response should be stored normally
        assert "The answer is 4" in ctx.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
        assert ctx.error is None

    def test_streaming_complete_response_no_timeout_artifacts(self) -> None:
        """A complete (non-partial) streaming response should not set
        timeout or usage_unavailable artifacts."""
        ctx = _make_context()
        stage = GenerateAnswerStage()
        mock_llm = _make_mock_llm(supports_streaming=True)

        complete_response = LLMResponse(
            content="The answer is 4",
            usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
            is_partial=False,
            usage_unavailable=False,
        )
        mock_llm.stream_invoke.return_value = complete_response

        with (
            patch(
                "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
                return_value=mock_llm,
            ),
            patch(_REGISTRY_PATCH_TARGET, return_value=None),
        ):
            stage.execute(ctx)

        # No timeout artifacts should be set
        assert ctx.get_artifact(ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL) is None
        assert ctx.get_artifact(ArtifactKeys.USAGE_UNAVAILABLE) is None
        assert ctx.error is None

    def test_streaming_passes_request_timeout(self) -> None:
        """stream_invoke should receive the request_timeout from the model config."""
        ctx = _make_context()
        ctx.answering_model.request_timeout = 42.0
        stage = GenerateAnswerStage()
        mock_llm = _make_mock_llm(supports_streaming=True)

        mock_llm.stream_invoke.return_value = LLMResponse(
            content="ok",
            usage=UsageMetadata(),
        )

        with (
            patch(
                "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
                return_value=mock_llm,
            ),
            patch(_REGISTRY_PATCH_TARGET, return_value=None),
        ):
            stage.execute(ctx)

        call_kwargs = mock_llm.stream_invoke.call_args
        assert call_kwargs[1]["timeout"] == 42.0


@pytest.mark.unit
class TestAgentTimeoutPath:
    """Tests for agent timeout_reached handling in generate_answer."""

    def _run_agent_stage(self, ctx: VerificationContext, agent_result: MagicMock) -> None:
        """Helper to run GenerateAnswerStage with a mocked AgentPort.

        Forces the use_agent path by setting an MCP URL on the model config
        and patching get_agent to return a mock that yields the given result.
        """
        stage = GenerateAnswerStage()
        mock_agent = MagicMock()
        mock_agent.run.return_value = agent_result

        # Force the agent path by adding an MCP server URL
        ctx.answering_model.mcp_urls_dict = {"test-server": "http://localhost:8080"}

        with (
            patch(
                "karenina.benchmark.verification.stages.pipeline.generate_answer.get_agent",
                return_value=mock_agent,
            ),
            patch(_REGISTRY_PATCH_TARGET, return_value=None),
        ):
            stage.execute(ctx)

    def test_agent_timeout_with_trace_sets_partial_and_usage_unavailable(self) -> None:
        """When agent times out but has a partial trace, RESPONSE_TIMEOUT_PARTIAL
        and USAGE_UNAVAILABLE should be set, and the trace should be stored."""
        ctx = _make_context()
        result = _make_agent_result(
            timeout_reached=True,
            raw_trace="--- AI Message ---\nPartial response from agent",
            turns=3,
        )

        self._run_agent_stage(ctx, result)

        # Timeout artifacts should be set
        assert ctx.get_artifact(ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL) is True
        assert ctx.get_result_field("response_timeout_partial") is True
        assert ctx.get_artifact(ArtifactKeys.USAGE_UNAVAILABLE) is True
        assert ctx.get_result_field(ArtifactKeys.USAGE_UNAVAILABLE) is True

        # The partial trace should still be stored as the raw response
        raw = ctx.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
        assert "Partial response from agent" in raw

        # Pipeline should continue (no error)
        assert ctx.error is None

    def test_agent_timeout_with_no_trace_marks_error(self) -> None:
        """When agent times out with no trace at all, the context should be
        marked with an error and the pipeline should stop."""
        ctx = _make_context()
        result = _make_agent_result(
            timeout_reached=True,
            raw_trace="",
            turns=0,
        )

        self._run_agent_stage(ctx, result)

        # Error should be set
        assert ctx.error is not None
        assert "timed out" in ctx.error.lower()
        assert ctx.completed_without_errors is False

        # RAW_LLM_RESPONSE should be empty string (set explicitly in the early return)
        assert ctx.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE) == ""
        assert ctx.get_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED) is False

    def test_agent_no_timeout_does_not_set_timeout_artifacts(self) -> None:
        """When agent completes normally (no timeout), no timeout artifacts
        should be set."""
        ctx = _make_context()
        result = _make_agent_result(
            timeout_reached=False,
            raw_trace="--- AI Message ---\nComplete response",
            turns=2,
        )

        self._run_agent_stage(ctx, result)

        # No timeout artifacts
        assert ctx.get_artifact(ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL) is None
        assert ctx.get_artifact(ArtifactKeys.USAGE_UNAVAILABLE) is None
        assert ctx.error is None

        # Normal response stored
        assert "Complete response" in ctx.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)
