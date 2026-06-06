"""Tests for AgenticParseTemplateStage."""

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
from karenina.schemas.primitives import BooleanMatch
from karenina.utils.errors import ErrorCategory


class MockAnswer(BaseAnswer):
    """Test answer template."""

    test_field: bool = VerifiedField(
        description="A test field.",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )


def _make_context(**overrides) -> VerificationContext:
    """Create a minimal VerificationContext for testing."""
    defaults = {
        "question_id": "q1",
        "template_id": "t1",
        "question_text": "Fix the bug",
        "template_code": "class MockAnswer(BaseAnswer): ...",
        "answering_model": ModelConfig(
            id="test",
            model_name="test-model",
            interface="claude_agent_sdk",
        ),
        "parsing_model": ModelConfig(
            id="test",
            model_name="test-model",
            interface="claude_agent_sdk",
        ),
        "agentic_parsing": True,
        "agentic_judge_context": "workspace_only",
        "workspace_path": Path("/tmp/test_workspace"),
    }
    defaults.update(overrides)
    ctx = VerificationContext(**defaults)
    ctx.set_artifact(ArtifactKeys.RAW_ANSWER, MockAnswer)
    ctx.set_artifact(ArtifactKeys.ANSWER, MockAnswer)
    ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "test trace")
    return ctx


@pytest.mark.unit
class TestAgenticParseTemplateStage:
    def test_name(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        stage = AgenticParseTemplateStage()
        assert stage.name == "AgenticParseTemplate"

    def test_should_run_requires_agentic_parsing(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        stage = AgenticParseTemplateStage()
        ctx = _make_context(agentic_parsing=False)
        assert stage.should_run(ctx) is False

    def test_should_run_true_when_agentic_parsing_enabled(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        stage = AgenticParseTemplateStage()
        ctx = _make_context(agentic_parsing=True)
        assert stage.should_run(ctx) is True

    def test_should_not_run_if_error(self):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        stage = AgenticParseTemplateStage()
        ctx = _make_context()
        ctx.mark_error("previous error")
        assert stage.should_run(ctx) is False

    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_agent")
    def test_investigation_prompt_warns_about_large_files(self, mock_get_agent):
        from karenina.benchmark.verification.stages.helpers.agentic_parse_helpers import (
            run_investigation,
        )
        from karenina.ports import AgentResult, UsageMetadata

        mock_agent = MagicMock()
        mock_agent.capabilities.supports_code_execution = False
        mock_agent.capabilities.supports_system_messages = True
        mock_agent.run.return_value = AgentResult(
            final_response='{"test_field": true}',
            raw_trace='{"test_field": true}',
            trace_messages=[],
            usage=UsageMetadata(),
            turns=1,
            limit_reached=False,
        )
        mock_get_agent.return_value = mock_agent

        run_investigation(_make_context(), {"properties": {"test_field": {"type": "boolean"}}})

        messages = mock_agent.run.call_args.kwargs["messages"]
        prompt_text = "\n".join(str(message.content) for message in messages)
        assert "file sizes" in prompt_text
        assert "context window" in prompt_text
        assert "Read large files carefully" in prompt_text

    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_agent")
    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_parser")
    def test_execute_calls_agent_then_parser(self, mock_get_parser, mock_get_agent):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.ports import AgentResult, ParsePortResult, UsageMetadata

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response='{"test_field": true}',
            raw_trace="investigation trace",
            trace_messages=[],
            usage=UsageMetadata(),
            turns=3,
            limit_reached=False,
        )
        mock_get_agent.return_value = mock_agent

        # Mock parser
        mock_parser = MagicMock()
        parsed_answer = MockAnswer(test_field=True)
        mock_parser.parse_to_pydantic.return_value = ParsePortResult(
            parsed=parsed_answer,
            usage=UsageMetadata(),
        )
        mock_get_parser.return_value = mock_parser

        stage = AgenticParseTemplateStage()
        ctx = _make_context()
        stage.execute(ctx)

        # Agent was called with workspace_path
        mock_agent.run.assert_called_once()
        call_kwargs = mock_agent.run.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        agent_config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        system_prompt = messages[0].text
        assert agent_config.workspace_path == Path("/tmp/test_workspace")
        assert "Look only for artifacts that appear to contain final reported results" in system_prompt
        assert "Do not run scripts, notebooks, or commands" in system_prompt

        # Parser was called
        mock_parser.parse_to_pydantic.assert_called_once()

        # Results stored. After Fix C the stage rebuilds the strict answer
        # via model_construct, so identity is no longer preserved; match
        # by class + field values instead.
        stored = ctx.get_artifact(ArtifactKeys.PARSED_ANSWER)
        assert isinstance(stored, MockAnswer)
        assert stored.test_field is True
        assert ctx.get_artifact(ArtifactKeys.AGENTIC_PARSING_PERFORMED) is True
        assert ctx.get_artifact(ArtifactKeys.INVESTIGATION_TRACE) == "investigation trace"

    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_agent")
    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_parser")
    def test_execute_tracks_agentic_parsing_usage_separately(self, mock_get_parser, mock_get_agent):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.ports import AgentResult, ParsePortResult, UsageMetadata

        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response='{"test_field": true}',
            raw_trace="investigation trace",
            trace_messages=[],
            usage=UsageMetadata(input_tokens=100, output_tokens=20, total_tokens=120),
            turns=3,
            limit_reached=False,
        )
        mock_get_agent.return_value = mock_agent

        mock_parser = MagicMock()
        parsed_answer = MockAnswer(test_field=True)
        mock_parser.parse_to_pydantic.return_value = ParsePortResult(
            parsed=parsed_answer,
            usage=UsageMetadata(input_tokens=30, output_tokens=10, total_tokens=40),
        )
        mock_get_parser.return_value = mock_parser

        stage = AgenticParseTemplateStage()
        ctx = _make_context()
        stage.execute(ctx)

        usage_tracker = ctx.get_artifact(ArtifactKeys.USAGE_TRACKER)
        usage_metadata = usage_tracker.get_total_summary()

        assert usage_metadata["agentic_parsing_investigation"]["input_tokens"] == 100
        assert usage_metadata["agentic_parsing_investigation"]["output_tokens"] == 20
        assert usage_metadata["agentic_parsing_extraction"]["input_tokens"] == 30
        assert usage_metadata["agentic_parsing_extraction"]["output_tokens"] == 10
        assert usage_metadata["total"]["input_tokens"] == 130
        assert usage_metadata["total"]["output_tokens"] == 30
        assert usage_metadata["total"]["total_tokens"] == 160

    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_agent")
    def test_execute_marks_error_on_agent_failure(self, mock_get_agent):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("agent failed")
        mock_get_agent.return_value = mock_agent

        stage = AgenticParseTemplateStage()
        ctx = _make_context()
        stage.execute(ctx)

        assert ctx.error is not None
        assert "agent failed" in ctx.error

    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_agent")
    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_parser")
    def test_execute_marks_error_on_extraction_failure(self, mock_get_parser, mock_get_agent):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.ports import AgentResult, UsageMetadata

        # Agent succeeds
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response='{"test_field": true}',
            raw_trace="investigation trace",
            trace_messages=[],
            usage=UsageMetadata(),
            turns=3,
            limit_reached=False,
        )
        mock_get_agent.return_value = mock_agent

        # Parser fails
        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.side_effect = RuntimeError("parser failed")
        mock_get_parser.return_value = mock_parser

        stage = AgenticParseTemplateStage()
        ctx = _make_context()
        stage.execute(ctx)

        assert ctx.error is not None
        assert "extraction failed" in ctx.error.lower()
        # Investigation trace should still be stored
        assert ctx.get_artifact(ArtifactKeys.INVESTIGATION_TRACE) == "investigation trace"

    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_agent")
    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_parser")
    def test_extraction_uses_final_investigation_report_not_full_trace(self, mock_get_parser, mock_get_agent):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.ports import AgentResult, ParsePortResult, UsageMetadata

        huge_tool_output = "RAW_CSV_LINE," * 20_000
        final_report = '```json\n{"test_field": true}\n```'
        investigation_trace = (
            "--- AI Message ---\nTool Calls:\nread_file(...)\n"
            "--- Tool Message (call_id: read) ---\n"
            f"{huge_tool_output}\n"
            "--- AI Message ---\n"
            f"{final_report}"
        )

        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response=final_report,
            raw_trace=investigation_trace,
            trace_messages=[],
            usage=UsageMetadata(),
            turns=3,
            limit_reached=False,
        )
        mock_get_agent.return_value = mock_agent

        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = ParsePortResult(
            parsed=MockAnswer(test_field=True),
            usage=UsageMetadata(),
        )
        mock_get_parser.return_value = mock_parser

        ctx = _make_context()
        AgenticParseTemplateStage().execute(ctx)

        messages = mock_parser.parse_to_pydantic.call_args.args[0]
        user_text = next(message.text for message in messages if message.role.value == "user")
        assert final_report in user_text
        assert "RAW_CSV_LINE" not in user_text

    def test_extraction_input_falls_back_to_bounded_trace_excerpt(self):
        from karenina.benchmark.verification.stages.helpers.agentic_parse_helpers import (
            prepare_extraction_input,
        )

        trace_without_final_report = (
            "--- AI Message ---\nTool Calls:\nread_file(...)\n--- Tool Message (call_id: read) ---\n" + ("x" * 120_000)
        )

        prepared = prepare_extraction_input(trace_without_final_report)

        assert len(prepared) <= 60_000
        assert "investigation report truncated" in prepared

    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_agent")
    @patch("karenina.benchmark.verification.stages.helpers.agentic_parse_helpers.get_parser")
    def test_extraction_connection_error_is_retryable_connection(self, mock_get_parser, mock_get_agent):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.ports import AgentResult, UsageMetadata

        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response='{"test_field": true}',
            raw_trace='--- AI Message ---\n{"test_field": true}',
            trace_messages=[],
            usage=UsageMetadata(),
            turns=3,
            limit_reached=False,
        )
        mock_get_agent.return_value = mock_agent

        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.side_effect = RuntimeError("Connection error.")
        mock_get_parser.return_value = mock_parser

        ctx = _make_context()
        AgenticParseTemplateStage().execute(ctx)

        assert ctx.error is not None
        assert ctx.error_category == ErrorCategory.CONNECTION
