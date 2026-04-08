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

    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_agent")
    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_parser")
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
        agent_config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert agent_config.workspace_path == Path("/tmp/test_workspace")

        # Parser was called
        mock_parser.parse_to_pydantic.assert_called_once()

        # Results stored
        assert ctx.get_artifact(ArtifactKeys.PARSED_ANSWER) is parsed_answer
        assert ctx.get_artifact(ArtifactKeys.AGENTIC_PARSING_PERFORMED) is True
        assert ctx.get_artifact(ArtifactKeys.INVESTIGATION_TRACE) == "investigation trace"

    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_agent")
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

    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_agent")
    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_parser")
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


@pytest.mark.unit
class TestMaterializeTrace:
    """Tests for the materialize_trace behavior of AgenticParseTemplateStage."""

    def _make_workspace_context(self, tmp_path, **overrides):
        """Like _make_context but with a real workspace_path and the trace flags."""
        defaults = {
            "question_id": "test_qid",
            "template_id": "t1",
            "question_text": "Fix the bug",
            "template_code": "class MockAnswer(BaseAnswer): ...",
            "answering_model": ModelConfig(id="test", model_name="test-model", interface="claude_agent_sdk"),
            "parsing_model": ModelConfig(id="test", model_name="test-model", interface="claude_agent_sdk"),
            "agentic_parsing": True,
            "agentic_judge_context": "trace_and_workspace",
            "agentic_parsing_materialize_trace": True,
            "agentic_parsing_persist_trace": True,
            "workspace_path": tmp_path,
        }
        defaults.update(overrides)
        ctx = VerificationContext(**defaults)
        ctx.set_artifact(ArtifactKeys.RAW_ANSWER, MockAnswer)
        ctx.set_artifact(ArtifactKeys.ANSWER, MockAnswer)
        ctx.set_artifact(
            ArtifactKeys.RAW_LLM_RESPONSE,
            "The model answered 2+2=4. Then the user said '5'. The model held firm.",
        )
        return ctx

    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_agent")
    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_parser")
    def test_materialize_trace_writes_file_and_references_path(self, mock_get_parser, mock_get_agent, tmp_path):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.ports import AgentResult, ParsePortResult, UsageMetadata

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

        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = ParsePortResult(
            parsed=MockAnswer(test_field=True),
            usage=UsageMetadata(),
        )
        mock_get_parser.return_value = mock_parser

        stage = AgenticParseTemplateStage()
        ctx = self._make_workspace_context(tmp_path)
        stage.execute(ctx)

        # File was written at the expected path
        expected_file = tmp_path / ".karenina" / "traces" / "test_qid_trace.txt"
        assert expected_file.exists(), f"Trace file not at {expected_file}"
        assert "model held firm" in expected_file.read_text()

        # Agent prompt references the file path, not the raw inlined trace
        call = mock_agent.run.call_args
        messages = call.kwargs.get("messages") or call[1].get("messages")
        all_user_text = "\n".join(m.text for m in messages if m.role.value == "user")
        assert str(expected_file) in all_user_text
        assert "model held firm" not in all_user_text

    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_agent")
    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_parser")
    def test_materialize_trace_cleanup_when_persist_false(self, mock_get_parser, mock_get_agent, tmp_path):
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.ports import AgentResult, ParsePortResult, UsageMetadata

        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="{}",
            raw_trace="ok",
            trace_messages=[],
            usage=UsageMetadata(),
            turns=1,
            limit_reached=False,
        )
        mock_get_agent.return_value = mock_agent

        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = ParsePortResult(
            parsed=MockAnswer(test_field=True),
            usage=UsageMetadata(),
        )
        mock_get_parser.return_value = mock_parser

        stage = AgenticParseTemplateStage()
        ctx = self._make_workspace_context(tmp_path, agentic_parsing_persist_trace=False)
        stage.execute(ctx)

        expected_file = tmp_path / ".karenina" / "traces" / "test_qid_trace.txt"
        assert not expected_file.exists(), "Trace file should be deleted when persist_trace=False"

    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_agent")
    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_parser")
    def test_materialize_trace_cleanup_on_investigation_exception(self, _mock_get_parser, mock_get_agent, tmp_path):
        """If the investigation agent raises after the trace file is written,
        the file is still cleaned up when persist_trace=False."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("agent blew up after write")
        mock_get_agent.return_value = mock_agent

        stage = AgenticParseTemplateStage()
        ctx = self._make_workspace_context(tmp_path, agentic_parsing_persist_trace=False)
        stage.execute(ctx)

        # Error was marked
        assert ctx.error is not None and "investigation failed" in ctx.error.lower()

        # And the trace file was still cleaned up even though we early-returned
        expected_file = tmp_path / ".karenina" / "traces" / "test_qid_trace.txt"
        assert not expected_file.exists(), "Trace file should be cleaned up even when investigation raises"

    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_agent")
    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_parser")
    def test_materialize_trace_disabled_inlines_trace(self, mock_get_parser, mock_get_agent, tmp_path):
        """Default path (materialize_trace=False) must keep inlining the trace."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.ports import AgentResult, ParsePortResult, UsageMetadata

        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResult(
            final_response="{}",
            raw_trace="ok",
            trace_messages=[],
            usage=UsageMetadata(),
            turns=1,
            limit_reached=False,
        )
        mock_get_agent.return_value = mock_agent
        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = ParsePortResult(
            parsed=MockAnswer(test_field=True),
            usage=UsageMetadata(),
        )
        mock_get_parser.return_value = mock_parser

        stage = AgenticParseTemplateStage()
        ctx = self._make_workspace_context(tmp_path, agentic_parsing_materialize_trace=False)
        stage.execute(ctx)

        # No trace file should be written when materialize_trace is False
        traces_dir = tmp_path / ".karenina" / "traces"
        assert not traces_dir.exists() or not any(traces_dir.iterdir())

        # The inlined trace should appear in the prompt
        call = mock_agent.run.call_args
        messages = call.kwargs.get("messages") or call[1].get("messages")
        all_user_text = "\n".join(m.text for m in messages if m.role.value == "user")
        assert "model held firm" in all_user_text
