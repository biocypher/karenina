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
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.close_adapter")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.get_llm")
    def test_decision_adapter_is_closed(self, mock_get_llm, mock_close_adapter):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        llm = _llm('{"reasoning":"direct","sufficient":true,"answer":{"solved":true}}')
        mock_get_llm.return_value = llm
        ctx = _make_context()

        DynamicParseTemplateStage().execute(ctx)

        mock_close_adapter.assert_called_once_with(llm)

    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.close_adapter")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.PromptAssembler")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.get_llm")
    def test_decision_adapter_is_closed_when_prompt_assembly_fails(
        self,
        mock_get_llm,
        mock_prompt_assembler,
        mock_close_adapter,
    ):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        llm = _llm('{"reasoning":"direct","sufficient":true,"answer":{"solved":true}}')
        mock_get_llm.return_value = llm
        mock_prompt_assembler.return_value.assemble.side_effect = RuntimeError("prompt assembly failed")
        ctx = _make_context()

        DynamicParseTemplateStage().execute(ctx)

        assert "Dynamic parsing decision failed" in (ctx.error or "")
        mock_close_adapter.assert_called_once_with(llm)

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


@pytest.mark.unit
class TestDynamicParseTemplateEscalation:
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.run_extraction")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.run_investigation")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.get_llm")
    def test_insufficient_verdict_escalates_with_reasoning(
        self,
        mock_get_llm,
        mock_run_investigation,
        mock_run_extraction,
    ):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        mock_get_llm.return_value = _llm(
            '{"reasoning":"answer is in result.json","sufficient":false}',
            UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        mock_run_investigation.return_value = (
            "investigation trace",
            False,
            UsageMetadata(input_tokens=20, output_tokens=7, total_tokens=27),
        )
        mock_run_extraction.return_value = (
            DynamicAnswer(solved=True),
            UsageMetadata(input_tokens=8, output_tokens=3, total_tokens=11),
        )
        ctx = _make_context()

        DynamicParseTemplateStage().execute(ctx)

        assert ctx.error is None
        assert ctx.get_result_field(ArtifactKeys.DYNAMIC_PARSE_DECISION) == "escalated"
        assert ctx.get_result_field(ArtifactKeys.DYNAMIC_DECISION_REASONING) == "answer is in result.json"
        assert ctx.get_artifact(ArtifactKeys.AGENTIC_PARSING_PERFORMED) is True
        assert ctx.get_result_field(ArtifactKeys.INVESTIGATION_TRACE) == "investigation trace"
        mock_run_investigation.assert_called_once()
        assert mock_run_investigation.call_args.kwargs["screening_reasoning"] == "answer is in result.json"

        usage = ctx.get_artifact(ArtifactKeys.USAGE_TRACKER).get_total_summary()
        assert usage["dynamic_parsing_decision"]["input_tokens"] == 10
        assert usage["agentic_parsing_investigation"]["input_tokens"] == 20
        assert usage["agentic_parsing_extraction"]["input_tokens"] == 8
        assert usage["total"]["total_tokens"] == 53

    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.run_extraction")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.run_investigation")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.get_llm")
    def test_malformed_decision_escalates_and_sets_caveat_flag(
        self,
        mock_get_llm,
        mock_run_investigation,
        mock_run_extraction,
    ):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        mock_get_llm.return_value = _llm("not json")
        mock_run_investigation.return_value = ("investigation trace", False, UsageMetadata())
        mock_run_extraction.return_value = (DynamicAnswer(solved=True), UsageMetadata())
        ctx = _make_context()

        DynamicParseTemplateStage().execute(ctx)

        assert ctx.error is None
        assert ctx.get_result_field(ArtifactKeys.DYNAMIC_PARSE_DECISION) == "escalated"
        assert ctx.get_result_field(ArtifactKeys.PARSE_DECISION_MALFORMED) is True
        assert mock_run_investigation.call_args.kwargs["screening_reasoning"] is None

    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.run_extraction")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.run_investigation")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.get_llm")
    def test_validation_error_escalates_and_sets_caveat_flag(
        self,
        mock_get_llm,
        mock_run_investigation,
        mock_run_extraction,
    ):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        mock_get_llm.return_value = _llm(
            '{"reasoning":"bad type","sufficient":true,"answer":{"value":{"not":"a float"}}}'
        )
        mock_run_investigation.return_value = ("investigation trace", False, UsageMetadata())
        mock_run_extraction.return_value = (DynamicNumberAnswer(value=2.0), UsageMetadata())
        ctx = _make_context(answer_cls=DynamicNumberAnswer)

        DynamicParseTemplateStage().execute(ctx)

        assert ctx.error is None
        assert ctx.get_result_field(ArtifactKeys.DYNAMIC_PARSE_DECISION) == "escalated"
        assert ctx.get_result_field(ArtifactKeys.PARSE_DECISION_MALFORMED) is True

    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.run_extraction")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.run_investigation")
    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.get_llm")
    def test_final_message_extraction_error_escalates_without_decision_call(
        self,
        mock_get_llm,
        mock_run_investigation,
        mock_run_extraction,
    ):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        mock_run_investigation.return_value = ("investigation trace", False, UsageMetadata())
        mock_run_extraction.return_value = (DynamicAnswer(solved=True), UsageMetadata())
        ctx = _make_context()
        ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, [])

        DynamicParseTemplateStage().execute(ctx)

        assert ctx.error is None
        assert ctx.get_result_field(ArtifactKeys.DYNAMIC_PARSE_DECISION) == "escalated"
        mock_get_llm.assert_not_called()
        mock_run_investigation.assert_called_once()

    @patch("karenina.benchmark.verification.stages.pipeline.dynamic_parse_template.get_llm")
    def test_decision_infra_error_marks_standard_failure(self, mock_get_llm):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage
        from karenina.utils.errors import ErrorCategory

        mock_get_llm.side_effect = TimeoutError("decision timed out")
        ctx = _make_context()

        DynamicParseTemplateStage().execute(ctx)

        assert ctx.error is not None
        assert "Dynamic parsing decision failed" in ctx.error
        assert ctx.error_category == ErrorCategory.TIMEOUT


@pytest.mark.unit
class TestDynamicParseTemplateShouldRun:
    def test_should_run_requires_dynamic_trigger(self):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        ctx = _make_context(agentic_parsing_trigger="always")
        assert DynamicParseTemplateStage().should_run(ctx) is False

    def test_should_run_does_not_skip_when_sufficiency_never_ran(self):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        ctx = _make_context()
        assert ctx.get_artifact(ArtifactKeys.SUFFICIENCY_DETECTED) is None
        assert DynamicParseTemplateStage().should_run(ctx) is True

    def test_should_run_skips_only_explicit_insufficient(self):
        from karenina.benchmark.verification.stages.pipeline.dynamic_parse_template import DynamicParseTemplateStage

        ctx = _make_context()
        ctx.set_artifact(ArtifactKeys.SUFFICIENCY_DETECTED, False)
        assert DynamicParseTemplateStage().should_run(ctx) is False
