"""Tests for ParseTemplateStage error category propagation.

These tests cover issue 191: parser-call exceptions must reach
ParseTemplateStage so the stage can classify them via ErrorRegistry,
instead of defaulting to ErrorCategory.PERMANENT and bypassing
user-configured retry policies.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import Field

from karenina.benchmark.verification.evaluators.template.evaluator import TemplateEvaluator
from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    VerificationContext,
)
from karenina.benchmark.verification.stages.pipeline.parse_template import ParseTemplateStage
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.answer import BaseAnswer
from karenina.utils.errors import ErrorCategory


class _Answer(BaseAnswer):
    """Plain answer template used by these tests."""

    drug_target: str = Field(default="", description="The drug target")


def _make_model_config() -> ModelConfig:
    return ModelConfig(
        id="test-model",
        model_provider="openai",
        model_name="gpt-test",
        temperature=0.0,
    )


def _make_context() -> VerificationContext:
    """Build a minimal VerificationContext that ParseTemplateStage can run on."""
    ctx = VerificationContext(
        question_id="q1",
        template_id="t1",
        question_text="What is the drug target?",
        template_code="class _Answer(BaseAnswer): ...",
        answering_model=_make_model_config(),
        parsing_model=_make_model_config(),
    )
    ctx.set_artifact(ArtifactKeys.RAW_ANSWER, _Answer)
    ctx.set_artifact(ArtifactKeys.ANSWER, _Answer)
    ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "BCL-2 is the target.")
    return ctx


# ============================================================================
# Evaluator: ParseResult.error_exception is populated on failure
# ============================================================================


@pytest.mark.unit
class TestEvaluatorPreservesException:
    """TemplateEvaluator must store the original exception alongside the message."""

    def _build_evaluator(self, parser_side_effect: BaseException) -> TemplateEvaluator:
        with (
            patch("karenina.benchmark.verification.evaluators.template.evaluator.get_llm") as mock_get_llm,
            patch("karenina.benchmark.verification.evaluators.template.evaluator.get_parser") as mock_get_parser,
        ):
            mock_get_llm.return_value = MagicMock()
            mock_parser = MagicMock()
            mock_parser.capabilities = MagicMock()
            mock_parser.parse_to_pydantic.side_effect = parser_side_effect
            mock_get_parser.return_value = mock_parser
            return TemplateEvaluator(model_config=_make_model_config(), answer_class=_Answer)

    def test_timeout_is_preserved(self):
        """A TimeoutError raised by the parser appears in ParseResult.error_exception."""
        exc = TimeoutError("Request timed out")
        evaluator = self._build_evaluator(exc)
        result = evaluator.parse_response(
            raw_response="BCL-2 is the target.",
            question_text="What is the drug target?",
        )
        assert result.success is False
        assert result.error_exception is exc
        assert result.error is not None
        assert "Request timed out" in result.error

    def test_value_error_is_preserved(self):
        """A non-transient parser failure also propagates the exception."""
        exc = ValueError("schema mismatch")
        evaluator = self._build_evaluator(exc)
        result = evaluator.parse_response(
            raw_response="BCL-2",
            question_text="What is the drug target?",
        )
        assert result.success is False
        assert result.error_exception is exc


# ============================================================================
# Stage: ParseTemplateStage.execute classifies the preserved exception
# ============================================================================


@pytest.mark.unit
class TestParseTemplateStageErrorCategory:
    """ParseTemplateStage must route the exception through ErrorRegistry."""

    @patch("karenina.benchmark.verification.stages.pipeline.parse_template.TemplateEvaluator")
    def test_timeout_classified_as_timeout(self, mock_evaluator_cls):
        """Parser TimeoutError lands in ctx.error_category=TIMEOUT, not PERMANENT."""
        from karenina.benchmark.verification.evaluators.template.results import ParseResult

        mock_evaluator = MagicMock()
        mock_evaluator.model_str = "gpt-test"
        timeout_exc = TimeoutError("Request timed out")
        failed = ParseResult()
        failed.success = False
        failed.error = f"Parsing failed: {timeout_exc}"
        failed.error_exception = timeout_exc
        mock_evaluator.parse_response.return_value = failed
        mock_evaluator_cls.return_value = mock_evaluator

        stage = ParseTemplateStage()
        ctx = _make_context()
        stage.execute(ctx)

        assert ctx.error is not None
        assert "Request timed out" in ctx.error
        assert ctx.error_category == ErrorCategory.TIMEOUT

    @patch("karenina.benchmark.verification.stages.pipeline.parse_template.TemplateEvaluator")
    def test_streaming_timeout_subclass_classified_as_timeout(self, mock_evaluator_cls):
        """StreamingTimeoutError (subclass of TimeoutError) is also TIMEOUT."""
        from karenina.benchmark.verification.evaluators.template.results import ParseResult
        from karenina.exceptions import StreamingTimeoutError

        mock_evaluator = MagicMock()
        mock_evaluator.model_str = "gpt-test"
        exc = StreamingTimeoutError("stream stalled after 600s")
        failed = ParseResult()
        failed.success = False
        failed.error = f"Parsing failed: {exc}"
        failed.error_exception = exc
        mock_evaluator.parse_response.return_value = failed
        mock_evaluator_cls.return_value = mock_evaluator

        stage = ParseTemplateStage()
        ctx = _make_context()
        stage.execute(ctx)

        assert ctx.error_category == ErrorCategory.TIMEOUT

    @patch("karenina.benchmark.verification.stages.pipeline.parse_template.TemplateEvaluator")
    def test_value_error_classified_as_permanent(self, mock_evaluator_cls):
        """A non-transient exception still routes to PERMANENT."""
        from karenina.benchmark.verification.evaluators.template.results import ParseResult

        mock_evaluator = MagicMock()
        mock_evaluator.model_str = "gpt-test"
        exc = ValueError("schema mismatch")
        failed = ParseResult()
        failed.success = False
        failed.error = f"Parsing failed: {exc}"
        failed.error_exception = exc
        mock_evaluator.parse_response.return_value = failed
        mock_evaluator_cls.return_value = mock_evaluator

        stage = ParseTemplateStage()
        ctx = _make_context()
        stage.execute(ctx)

        assert ctx.error_category == ErrorCategory.PERMANENT

    @patch("karenina.benchmark.verification.stages.pipeline.parse_template.TemplateEvaluator")
    def test_no_exception_falls_back_to_permanent(self, mock_evaluator_cls):
        """When error_exception is None (e.g. trace extraction error), default to PERMANENT."""
        from karenina.benchmark.verification.evaluators.template.results import ParseResult

        mock_evaluator = MagicMock()
        mock_evaluator.model_str = "gpt-test"
        failed = ParseResult()
        failed.success = False
        failed.error = "Could not extract final AI message from trace"
        failed.error_exception = None
        mock_evaluator.parse_response.return_value = failed
        mock_evaluator_cls.return_value = mock_evaluator

        stage = ParseTemplateStage()
        ctx = _make_context()
        stage.execute(ctx)

        assert ctx.error_category == ErrorCategory.PERMANENT
