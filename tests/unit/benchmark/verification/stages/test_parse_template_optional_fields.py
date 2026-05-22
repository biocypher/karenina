"""Classical template parsing tolerates null sub-question extractions."""

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    VerificationContext,
)
from karenina.benchmark.verification.stages.pipeline.parse_template import (
    ParseTemplateStage,
)
from karenina.ports import ParsePortResult, UsageMetadata
from karenina.ports.capabilities import PortCapabilities
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives import BooleanMatch, NumericTolerance


class ThreeFieldClassicalAnswer(BaseAnswer):
    """Three-field template mirroring grouped QA sub-question scoring."""

    q1: float = VerifiedField(
        description="First sub-answer.",
        ground_truth=1.5,
        verify_with=NumericTolerance(tolerance=0.1),
    )
    q2: bool = VerifiedField(
        description="Second sub-answer.",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
    q3: float = VerifiedField(
        description="Third sub-answer.",
        ground_truth=2.0,
        verify_with=NumericTolerance(tolerance=0.1),
    )


class FalseGTClassicalAnswer(BaseAnswer):
    """Boolean with ground truth False for null-vs-False regression coverage."""

    q2: bool = VerifiedField(
        description="Did X happen?",
        ground_truth=False,
        verify_with=BooleanMatch(),
    )


def _make_context(answer_cls: type[BaseAnswer]) -> VerificationContext:
    ctx = VerificationContext(
        question_id="q1",
        template_id="t1",
        question_text="Compute stuff.",
        template_code=f"class {answer_cls.__name__}(BaseAnswer): ...",
        answering_model=ModelConfig(
            id="test",
            model_provider="openai",
            model_name="test-model",
            interface="langchain",
        ),
        parsing_model=ModelConfig(
            id="test",
            model_provider="openai",
            model_name="test-model",
            interface="langchain",
        ),
        use_full_trace_for_template=True,
    )
    ctx.set_artifact(ArtifactKeys.RAW_ANSWER, answer_cls)
    ctx.set_artifact(ArtifactKeys.ANSWER, answer_cls)
    ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "Final answer text.")
    return ctx


@pytest.mark.unit
class TestClassicalParseOptionalFields:
    """Null extractions in classical parsing surface as None in field_results."""

    @patch("karenina.benchmark.verification.evaluators.template.evaluator.get_llm")
    @patch("karenina.benchmark.verification.evaluators.template.evaluator.get_parser")
    def test_null_field_does_not_fail_whole_record(self, mock_get_parser, mock_get_llm):
        mock_get_llm.return_value = MagicMock()

        mock_parser = MagicMock()
        mock_parser.capabilities = PortCapabilities(supports_structured_output=True)

        def parse_to_pydantic(_messages, schema):
            return ParsePortResult(
                parsed=schema(q1=1.5, q2=None, q3=2.0),
                usage=UsageMetadata(),
            )

        mock_parser.parse_to_pydantic.side_effect = parse_to_pydantic
        mock_get_parser.return_value = mock_parser

        ctx = _make_context(ThreeFieldClassicalAnswer)
        ParseTemplateStage().execute(ctx)

        assert ctx.error is None, ctx.error
        parsed = ctx.get_artifact(ArtifactKeys.PARSED_ANSWER)
        assert isinstance(parsed, ThreeFieldClassicalAnswer)
        assert parsed.__dict__.get("_null_fields") == {"q2"}

        field_results = parsed._compute_field_results()
        assert field_results == {"q1": True, "q2": None, "q3": True}
        assert parsed.verify() is False
        assert parsed.verify_granular() == pytest.approx(2 / 3)

        schema_used = mock_parser.parse_to_pydantic.call_args.args[1]
        assert schema_used.__name__ == "ThreeFieldClassicalAnswer__Relaxed"

    @patch("karenina.benchmark.verification.evaluators.template.evaluator.get_llm")
    @patch("karenina.benchmark.verification.evaluators.template.evaluator.get_parser")
    def test_null_extraction_with_false_ground_truth_stays_none(
        self,
        mock_get_parser,
        mock_get_llm,
    ):
        mock_get_llm.return_value = MagicMock()

        mock_parser = MagicMock()
        mock_parser.capabilities = PortCapabilities(supports_structured_output=True)

        def parse_to_pydantic(_messages, schema):
            return ParsePortResult(
                parsed=schema(q2=None),
                usage=UsageMetadata(),
            )

        mock_parser.parse_to_pydantic.side_effect = parse_to_pydantic
        mock_get_parser.return_value = mock_parser

        ctx = _make_context(FalseGTClassicalAnswer)
        ParseTemplateStage().execute(ctx)

        assert ctx.error is None, ctx.error
        parsed = ctx.get_artifact(ArtifactKeys.PARSED_ANSWER)
        field_results = parsed._compute_field_results()

        assert field_results["q2"] is None
        assert field_results["q2"] is not False
        assert field_results["q2"] is not True
        assert parsed.verify() is False
