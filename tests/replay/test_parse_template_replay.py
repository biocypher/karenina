"""Unit tests for the parse bypass in ParseTemplateStage."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    VerificationContext,
)
from karenina.benchmark.verification.stages.pipeline.parse_template import (
    ParseTemplateStage,
)
from karenina.replay import ReplayEntry
from karenina.schemas.config import ModelConfig

SIMPLE_TEMPLATE = """
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives.comparisons import BooleanMatch


class Answer(BaseAnswer):
    mechanism: str = VerifiedField(
        description="Mechanism",
        verify_with=BooleanMatch(),
        ground_truth="X",
        default="X",
    )

    def verify(self) -> bool:
        return self.mechanism == "X"
"""


def _build_context_with_answer_cls() -> VerificationContext:
    """Build a context and pre-run template validation so ANSWER is set."""
    from karenina.benchmark.verification.stages.pipeline.validate_template import (
        ValidateTemplateStage,
    )

    ans = ModelConfig(id="a", model_name="m", model_provider="anthropic")
    parse = ModelConfig(id="p", model_name="m", model_provider="anthropic")
    context = VerificationContext(
        question_id="q",
        template_id="t",
        question_text="hi",
        template_code=SIMPLE_TEMPLATE,
        answering_model=ans,
        parsing_model=parse,
    )
    context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "raw")
    ValidateTemplateStage().execute(context)
    return context


@pytest.mark.integration
class TestParseTemplateReplayBypass:
    def test_bypass_skips_judge_when_parsed_fields_injected(self, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise AssertionError("get_parser must not be called on a parsed-fields hit")

        monkeypatch.setattr(
            "karenina.benchmark.verification.stages.pipeline.parse_template.get_parser",
            _boom,
            raising=False,
        )

        context = _build_context_with_answer_cls()
        entry = ReplayEntry(
            raw_trace="raw",
            parsed_answer_fields={"mechanism": "X"},
        )
        context.set_artifact(ArtifactKeys.REPLAY_ENTRY, entry)

        ParseTemplateStage().execute(context)
        parsed = context.get_artifact(ArtifactKeys.PARSED_ANSWER)
        assert parsed is not None
        assert parsed.mechanism == "X"

    def test_bypass_strict_raises_replay_hydration_error(self):
        context = _build_context_with_answer_cls()
        context.replay_parse_on_hydration_mismatch = "strict"
        entry = ReplayEntry(
            raw_trace="raw",
            parsed_answer_fields={"mechanism": 999},
        )
        context.set_artifact(ArtifactKeys.REPLAY_ENTRY, entry)

        from karenina.replay.exceptions import ReplayHydrationError

        with pytest.raises(ReplayHydrationError):
            ParseTemplateStage().execute(context)

    def test_bypass_falls_through_on_validation_error_default_policy(self, caplog):
        """With default fall_through policy, an invalid parsed_answer_fields
        logs a warning and the bypass returns control to the live judge."""
        context = _build_context_with_answer_cls()
        # Default policy is "fall_through".
        assert context.replay_parse_on_hydration_mismatch == "fall_through"
        entry = ReplayEntry(
            raw_trace="raw",
            parsed_answer_fields={"mechanism": 999},  # wrong type, model_validate will fail
        )
        context.set_artifact(ArtifactKeys.REPLAY_ENTRY, entry)

        with caplog.at_level("WARNING"):
            ParseTemplateStage().execute(context)

        assert any(
            "failed validation" in rec.message.lower() or "hydration" in rec.message.lower() for rec in caplog.records
        ), f"expected fall-through warning, got: {[r.message for r in caplog.records]}"
        # The bypass should NOT have set PARSING_MODEL_STR to "replay (no LLM)"
        # because we fell through to the live path.
        parsing_model_str = context.get_artifact(ArtifactKeys.PARSING_MODEL_STR)
        assert parsing_model_str != "replay (no LLM)"
