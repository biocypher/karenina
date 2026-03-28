"""Tests for VerifyTemplateStage bug fixes.

Issue 085: Regex-only verify() exception treated as fatal vs non-fatal in standard path.
"""

import logging

import pytest

from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.benchmark.verification.stages.pipeline.verify_template import VerifyTemplateStage
from karenina.schemas.config import ModelConfig


def _make_context(**overrides) -> VerificationContext:
    """Create a minimal VerificationContext for testing."""
    defaults = {
        "question_id": "q1",
        "template_id": "t1",
        "question_text": "What is X?",
        "template_code": "class Answer: ...",
        "answering_model": ModelConfig(id="test-model", model_name="test-model", interface="langchain"),
        "parsing_model": ModelConfig(id="test-model", model_name="test-model", interface="langchain"),
    }
    defaults.update(overrides)
    return VerificationContext(**defaults)


@pytest.mark.unit
class TestVerifyTemplateRegexOnlyExceptionHandling:
    """Issue 085: Regex-only path should handle verify() exceptions non-fatally."""

    def test_regex_only_verify_exception_is_non_fatal(self, caplog):
        """When verify() raises in the regex-only path, it should NOT mark_error (fatal).

        Before the fix, the exception would propagate to the outer try/except
        which calls context.mark_error(), making it fatal. After the fix, the
        exception is caught inline and field_verification_result is set to False.
        """

        class FakeAnswer:
            _raw_trace = None

            def verify(self):
                raise ValueError("something went wrong in verify")

            def verify_regex(self, _raw_response):
                return {"success": True, "results": {}, "details": {}}

        stage = VerifyTemplateStage()
        context = _make_context()

        # Set up artifacts: parsed_answer and raw_llm_response present, no evaluator (regex-only)
        context.set_artifact(ArtifactKeys.PARSED_ANSWER, FakeAnswer())
        context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "some response text")
        # No TEMPLATE_EVALUATOR set: triggers regex-only path

        with caplog.at_level(logging.WARNING):
            stage.execute(context)

        # Should NOT have marked a fatal error
        assert context.error is None, f"Expected non-fatal handling but got fatal error: {context.error}"
        # Field verification result should be False (exception treated as failure)
        field_result = context.get_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT)
        assert field_result is False

        # Combined verify_result should be False (field failed)
        verify_result = context.get_artifact(ArtifactKeys.VERIFY_RESULT)
        assert verify_result is False

        # Warning should have been logged
        assert any("field verification error" in record.message.lower() for record in caplog.records), (
            f"Expected warning about field verification error, got: {[r.message for r in caplog.records]}"
        )

    def test_standard_path_verify_exception_is_non_fatal(self, caplog):
        """Standard path already handles verify() exceptions non-fatally via evaluator.

        This test confirms the existing behavior is preserved.
        """
        from karenina.benchmark.verification.evaluators.template.results import (
            FieldVerificationResult,
            RegexVerificationResult,
        )

        class FakeEvaluator:
            def verify_fields(self, _parsed_answer):
                result = FieldVerificationResult()
                result.error = "Field verification failed: something went wrong"
                # success remains False
                return result

            def verify_regex(self, _parsed_answer, _raw_response):
                result = RegexVerificationResult()
                result.success = True
                result.results = {}
                result.details = {}
                return result

        class FakeAnswer:
            _raw_trace = None

        stage = VerifyTemplateStage()
        context = _make_context()

        context.set_artifact(ArtifactKeys.PARSED_ANSWER, FakeAnswer())
        context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "some response text")
        context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, FakeEvaluator())

        with caplog.at_level(logging.WARNING):
            stage.execute(context)

        # Should NOT have marked a fatal error
        assert context.error is None
        # Field verification result should be False
        field_result = context.get_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT)
        assert field_result is False

    def test_regex_only_verify_success_works(self):
        """Regex-only path with successful verify() should work normally."""

        class FakeAnswer:
            _raw_trace = None

            def verify(self):
                return True

            def verify_regex(self, _raw_response):
                return {"success": True, "results": {"field1": True}, "details": {}}

        stage = VerifyTemplateStage()
        context = _make_context()

        context.set_artifact(ArtifactKeys.PARSED_ANSWER, FakeAnswer())
        context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "some response text")

        stage.execute(context)

        assert context.error is None
        assert context.get_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT) is True
        assert context.get_artifact(ArtifactKeys.VERIFY_RESULT) is True
