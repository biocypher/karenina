"""Tests verifying GenerateAnswerStage invokes its adapter exactly once.

After the retry-error-harmonization refactor, the adapter returned by
``get_llm`` handles retries internally via ``RetryExecutor``.
``StreamingTimeoutError`` propagates up to the stage's exception handler,
which calls ``context.mark_error()``. The stage itself must make exactly
one call to ``stream_invoke`` or ``invoke`` — any in-stage retry loop is a
regression. These tests pin that contract by counting real invocations.
"""

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.benchmark.verification.stages.pipeline.generate_answer import GenerateAnswerStage
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.llm import LLMResponse
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config import ModelConfig

_REGISTRY_PATCH_TARGET = "karenina.adapters.registry.AdapterRegistry.get_spec"


def _make_context() -> VerificationContext:
    """Create a minimal VerificationContext for no-retry-loop tests."""
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


@pytest.mark.unit
class TestNoRetryLoopInGenerateAnswer:
    """Verify that GenerateAnswerStage does not retry stream_invoke internally."""

    def test_stream_invoke_called_exactly_once(self) -> None:
        """stream_invoke should be called exactly once, not in a retry loop."""
        ctx = _make_context()
        stage = GenerateAnswerStage()
        mock_llm = _make_mock_llm(supports_streaming=True)

        mock_llm.stream_invoke.return_value = LLMResponse(
            content="The answer is 4",
            usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        )

        with (
            patch(
                "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
                return_value=mock_llm,
            ),
            patch(_REGISTRY_PATCH_TARGET, return_value=None),
        ):
            stage.execute(ctx)

        assert mock_llm.stream_invoke.call_count == 1

    def test_invoke_called_exactly_once_when_no_streaming(self) -> None:
        """invoke should be called exactly once when streaming is not supported."""
        ctx = _make_context()
        stage = GenerateAnswerStage()
        mock_llm = _make_mock_llm(supports_streaming=False)

        mock_llm.invoke.return_value = LLMResponse(
            content="The answer is 4",
            usage=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        )

        with (
            patch(
                "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
                return_value=mock_llm,
            ),
            patch(_REGISTRY_PATCH_TARGET, return_value=None),
        ):
            stage.execute(ctx)

        assert mock_llm.invoke.call_count == 1

    def test_streaming_timeout_error_propagates_to_exception_handler(self) -> None:
        """When stream_invoke raises an exception (e.g. StreamingTimeoutError),
        it should propagate to the stage's exception handler and mark_error."""
        ctx = _make_context()
        stage = GenerateAnswerStage()
        mock_llm = _make_mock_llm(supports_streaming=True)

        mock_llm.stream_invoke.side_effect = TimeoutError("streaming timeout: no content received")

        with (
            patch(
                "karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm",
                return_value=mock_llm,
            ),
            patch(_REGISTRY_PATCH_TARGET, return_value=None),
        ):
            stage.execute(ctx)

        # Should be called exactly once (no retry)
        assert mock_llm.stream_invoke.call_count == 1
        # Error should be marked on context
        assert ctx.completed_without_errors is False
        assert ctx.error is not None
