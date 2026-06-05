"""Tests for DynamicParseTemplateStage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.ports import LLMResponse, UsageMetadata
from karenina.replay import ReplayEntry
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives import BooleanMatch, NumericTolerance


class DynamicAnswer(BaseAnswer):
    solved: bool = VerifiedField(
        description="Whether the task was solved.",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )


class DynamicFalseGTAnswer(BaseAnswer):
    solved: bool = VerifiedField(
        description="Whether the task was solved.",
        ground_truth=False,
        verify_with=BooleanMatch(),
    )


class DynamicNumberAnswer(BaseAnswer):
    value: float = VerifiedField(
        description="A numeric result.",
        ground_truth=2.0,
        verify_with=NumericTolerance(tolerance=0.1),
    )


def _make_context(answer_cls: type[BaseAnswer] = DynamicAnswer, **overrides) -> VerificationContext:
    defaults = {
        "question_id": "q1",
        "template_id": "t1",
        "question_text": "Fix the bug and report whether it was solved.",
        "template_code": f"class {answer_cls.__name__}(BaseAnswer): ...",
        "answering_model": ModelConfig(id="answerer", model_name="test-answerer", interface="claude_agent_sdk"),
        "parsing_model": ModelConfig(id="parser", model_name="test-parser", interface="claude_agent_sdk"),
        "agentic_parsing": True,
        "agentic_parsing_trigger": "dynamic",
        "agentic_judge_context": "workspace_only",
        "workspace_path": Path("/tmp/test_workspace"),
    }
    defaults.update(overrides)
    ctx = VerificationContext(**defaults)
    ctx.set_artifact(ArtifactKeys.RAW_ANSWER, answer_cls)
    ctx.set_artifact(ArtifactKeys.ANSWER, answer_cls)
    ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "Final answer: solved=true")
    return ctx


def _llm(content: str, usage: UsageMetadata | None = None) -> MagicMock:
    mock = MagicMock()
    mock.capabilities.supports_system_messages = True
    mock.invoke.return_value = LLMResponse(
        content=content,
        usage=usage or UsageMetadata(),
    )
    return mock


@pytest.mark.unit
class TestDynamicParseTemplateDirectPath:
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.run_investigation")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.get_llm")
    def test_sufficient_verdict_stores_parsed_answer_without_agent(self, mock_get_llm, mock_run_investigation):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        mock_get_llm.return_value = _llm(
            '{"reasoning":"final message says solved true","sufficient":true,"answer":{"solved":true}}'
        )
        ctx = _make_context()

        DynamicParseTemplateStage().execute(ctx)

        assert ctx.error is None
        stored = ctx.get_artifact(ArtifactKeys.PARSED_ANSWER)
        assert isinstance(stored, DynamicAnswer)
        assert stored.solved is True
        assert ctx.get_artifact(ArtifactKeys.DYNAMIC_PARSING_PERFORMED) is True
        assert ctx.get_result_field(ArtifactKeys.DYNAMIC_PARSE_DECISION) == "direct"
        assert ctx.get_result_field(ArtifactKeys.DYNAMIC_DECISION_REASONING) == "final message says solved true"
        assert ctx.get_artifact(ArtifactKeys.AGENTIC_PARSING_PERFORMED, False) is False
        mock_run_investigation.assert_not_called()

    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.run_investigation")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.get_llm")
    def test_markdown_fenced_json_is_accepted(self, mock_get_llm, mock_run_investigation):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        mock_get_llm.return_value = _llm(
            '```json\n{"reasoning":"inside fence","sufficient":true,"answer":{"solved":true}}\n```'
        )
        ctx = _make_context()

        DynamicParseTemplateStage().execute(ctx)

        assert ctx.error is None
        assert ctx.get_artifact(ArtifactKeys.PARSED_ANSWER).solved is True
        assert ctx.get_result_field(ArtifactKeys.DYNAMIC_PARSE_DECISION) == "direct"
        mock_run_investigation.assert_not_called()

    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.run_investigation")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.get_llm")
    def test_null_field_preserves_tri_valued_none(self, mock_get_llm, mock_run_investigation):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        mock_get_llm.return_value = _llm(
            '{"reasoning":"final message says value unavailable","sufficient":true,"answer":{"solved":null}}'
        )
        ctx = _make_context(answer_cls=DynamicFalseGTAnswer)

        DynamicParseTemplateStage().execute(ctx)

        stored = ctx.get_artifact(ArtifactKeys.PARSED_ANSWER)
        assert stored.__dict__.get("_null_fields") == {"solved"}
        assert stored._compute_field_results()["solved"] is None
        assert mock_run_investigation.call_count == 0

    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.get_llm")
    def test_replay_parsed_fields_bypass_decision_call(self, mock_get_llm):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        ctx = _make_context()
        ctx.set_artifact(
            ArtifactKeys.REPLAY_ENTRY,
            ReplayEntry(raw_trace="raw", parsed_answer_fields={"solved": True}),
        )

        DynamicParseTemplateStage().execute(ctx)

        assert ctx.error is None
        stored = ctx.get_artifact(ArtifactKeys.PARSED_ANSWER)
        assert isinstance(stored, DynamicAnswer)
        assert stored.solved is True
        assert ctx.get_artifact(ArtifactKeys.PARSING_MODEL_STR) == "replay (no LLM)"
        assert ctx.get_result_field(ArtifactKeys.DYNAMIC_PARSE_DECISION) == "replay"
        mock_get_llm.assert_not_called()
