"""Fix C: agentic-parse-template tolerates null sub-question extractions.

The agentic parser passes a relaxed sibling schema (every VerifiedField
becomes Optional[T] = None) to the LLM-side parser. Fields the LLM
returns as ``null`` are stored on the strict template via
``_null_fields``, so ``_compute_field_results`` reports them as ``None``
rather than False, preserving partial credit on the other fields.

Two regressions guarded here:

1. With a 3-field template and ``q2`` returned as null, q1 and q3 are
   scored normally and field_results["q2"] is exactly ``None``.

2. **Critical**: when ground truth for a boolean is ``False`` and the
   model returned ``None``, field_results["q2"] MUST stay ``None``
   (not become ``False``). Scoring it as False would silently look
   correct against the False ground truth. The bug Fix C exists to
   prevent.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    VerificationContext,
)
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives import BooleanMatch, NumericTolerance


class ThreeFieldAnswer(BaseAnswer):
    """Three-field template; one bool, two floats. Mirrors BixBench shape."""

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


class FalseGTAnswer(BaseAnswer):
    """Boolean with ground truth = False. Used for the critical regression."""

    q2: bool = VerifiedField(
        description="Did X happen?",
        ground_truth=False,
        verify_with=BooleanMatch(),
    )


def _make_context(answer_cls: type[BaseAnswer]) -> VerificationContext:
    """Minimal VerificationContext for agentic-parse stage tests."""
    ctx = VerificationContext(
        question_id="q1",
        template_id="t1",
        question_text="Compute stuff.",
        template_code=f"class {answer_cls.__name__}(BaseAnswer): ...",
        answering_model=ModelConfig(id="test", model_name="test-model", interface="claude_agent_sdk"),
        parsing_model=ModelConfig(id="test", model_name="test-model", interface="claude_agent_sdk"),
        agentic_parsing=True,
        agentic_judge_context="workspace_only",
        workspace_path=Path("/tmp/test_workspace"),
    )
    ctx.set_artifact(ArtifactKeys.RAW_ANSWER, answer_cls)
    ctx.set_artifact(ArtifactKeys.ANSWER, answer_cls)
    ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "raw trace")
    return ctx


@pytest.mark.unit
class TestAgenticParseOptionalFields:
    """Null extractions surface as None in field_results, not False."""

    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_agent")
    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_parser")
    def test_null_field_does_not_fail_whole_record(self, mock_get_parser, mock_get_agent):
        """q1 + q3 score normally; q2 (null in extraction) is None."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.benchmark.verification.utils.schema_builder import (
            build_extraction_relaxed_class,
        )
        from karenina.ports import AgentResult, ParsePortResult, UsageMetadata

        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="ok",
            raw_trace="investigation trace",
            trace_messages=[],
            usage=UsageMetadata(),
            turns=2,
            limit_reached=False,
        )
        mock_get_agent.return_value = mock_agent

        # Parser returns a *relaxed* instance with q2=None; the production
        # code path calls build_extraction_relaxed_class(answer_class) and
        # passes that to parse_to_pydantic, so we build the same class here.
        relaxed_cls = build_extraction_relaxed_class(ThreeFieldAnswer)
        relaxed_instance = relaxed_cls(q1=1.5, q2=None, q3=2.0)

        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = ParsePortResult(
            parsed=relaxed_instance,
            usage=UsageMetadata(),
        )
        mock_get_parser.return_value = mock_parser

        ctx = _make_context(ThreeFieldAnswer)
        AgenticParseTemplateStage().execute(ctx)

        assert ctx.error is None, ctx.error
        stored = ctx.get_artifact(ArtifactKeys.PARSED_ANSWER)
        assert isinstance(stored, ThreeFieldAnswer)
        # q2 was nulled, so it is present in _null_fields.
        assert stored.__dict__.get("_null_fields") == {"q2"}

        # Tri-valued field_results: q1 True, q2 None, q3 True
        field_results = stored._compute_field_results()
        assert field_results["q1"] is True
        assert field_results["q2"] is None
        assert field_results["q3"] is True

        # AllOf default: any non-True (including None) makes verify() False.
        assert stored.verify() is False
        # Granular: n_true / n_total = 2 / 3
        assert stored.verify_granular() == pytest.approx(2 / 3)

    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_agent")
    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_parser")
    def test_null_extraction_with_false_ground_truth_stays_none(self, mock_get_parser, mock_get_agent):
        """CRITICAL: GT=False + extracted=None MUST yield None, not False.

        Without Fix C, a null extraction could be silently scored as False
        and "accidentally match" a False ground truth. The user called this
        out explicitly. It is the load-bearing regression test for the
        whole tri-valued design.
        """
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.benchmark.verification.utils.schema_builder import (
            build_extraction_relaxed_class,
        )
        from karenina.ports import AgentResult, ParsePortResult, UsageMetadata

        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="ok",
            raw_trace="investigation trace",
            trace_messages=[],
            usage=UsageMetadata(),
            turns=2,
            limit_reached=False,
        )
        mock_get_agent.return_value = mock_agent

        relaxed_cls = build_extraction_relaxed_class(FalseGTAnswer)
        relaxed_instance = relaxed_cls(q2=None)

        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = ParsePortResult(
            parsed=relaxed_instance,
            usage=UsageMetadata(),
        )
        mock_get_parser.return_value = mock_parser

        ctx = _make_context(FalseGTAnswer)
        AgenticParseTemplateStage().execute(ctx)

        assert ctx.error is None, ctx.error
        stored = ctx.get_artifact(ArtifactKeys.PARSED_ANSWER)
        field_results = stored._compute_field_results()

        # The bug Fix C prevents: BooleanMatch(None, False) might return True
        # (silently treating a null extraction as a False match). The
        # tri-valued contract says field_results[q2] is None, not False/True.
        assert field_results["q2"] is None, f"q2 with null extraction must be None, got {field_results['q2']!r}"
        # Strict identity to rule out tricky truthiness bugs
        assert field_results["q2"] is not False
        assert field_results["q2"] is not True

        # verify() rejects None at the leaf (soft-False)
        assert stored.verify() is False
