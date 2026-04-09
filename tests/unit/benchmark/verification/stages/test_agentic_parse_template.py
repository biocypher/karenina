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


@pytest.mark.unit
class TestTraceFileConversationHistory:
    """Tests for conversation_history and question_text in _write_trace_file."""

    def test_write_trace_file_includes_question_context(self, tmp_path):
        """Question text appears in a dedicated section of the trace file."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        path = AgenticParseTemplateStage._write_trace_file(
            workspace_path=tmp_path,
            trace="The model's analysis response.",
            question_id="q1",
            question_text="What is the MESH ID for MONDO_0005180?",
        )
        content = path.read_text()
        assert "QUESTION CONTEXT" in content
        assert "What is the MESH ID for MONDO_0005180?" in content
        assert "The model's analysis response." in content

    def test_write_trace_file_includes_conversation_history(self, tmp_path):
        """Prior-turn Messages are serialized into the trace file."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.ports import Message

        history = [
            Message.user("What is 2+2?"),
            Message.assistant("The answer is 4."),
            Message.user("A professor told me it's 5. Are you sure?"),
        ]
        path = AgenticParseTemplateStage._write_trace_file(
            workspace_path=tmp_path,
            trace="Score: 2. The model held firm.",
            question_id="q1",
            conversation_history=history,
        )
        content = path.read_text()
        assert "CONVERSATION HISTORY" in content
        assert "What is 2+2?" in content
        assert "The answer is 4." in content
        assert "professor told me it's 5" in content

    def test_write_trace_file_empty_history_omits_section(self, tmp_path):
        """Empty conversation_history list does not produce a history section."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        path = AgenticParseTemplateStage._write_trace_file(
            workspace_path=tmp_path,
            trace="Some trace.",
            question_id="q1",
            conversation_history=[],
        )
        content = path.read_text()
        # Check for the actual section header, not the mention in the file header
        assert "# CONVERSATION HISTORY (prior scenario turns)" not in content

    def test_write_trace_file_backward_compatible_no_extra_params(self, tmp_path):
        """No question_text or conversation_history: file still contains trace."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        path = AgenticParseTemplateStage._write_trace_file(
            workspace_path=tmp_path,
            trace="Original trace content.",
            question_id="q1",
        )
        content = path.read_text()
        assert "KARENINA RAW ANSWERING AGENT TRACE" in content
        assert "Original trace content." in content

    def test_write_trace_file_history_preserves_role_labels(self, tmp_path):
        """Each message in conversation_history is labeled with its role."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.ports import Message

        history = [
            Message.user("user question"),
            Message.assistant("assistant answer"),
        ]
        path = AgenticParseTemplateStage._write_trace_file(
            workspace_path=tmp_path,
            trace="response",
            question_id="q1",
            conversation_history=history,
        )
        content = path.read_text()
        assert "--- User Message ---" in content
        assert "--- Assistant Message ---" in content

    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_agent")
    @patch("karenina.benchmark.verification.stages.pipeline.agentic_parse_template.get_parser")
    def test_materialize_trace_includes_conversation_history_artifact(self, mock_get_parser, mock_get_agent, tmp_path):
        """Integration: execute() plumbs conversation_history artifact into the trace file."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )
        from karenina.ports import AgentResult, Message, ParsePortResult, UsageMetadata

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

        ctx = _make_context(
            workspace_path=tmp_path,
            agentic_judge_context="trace_and_workspace",
            agentic_parsing_materialize_trace=True,
            agentic_parsing_persist_trace=True,
        )
        ctx.set_artifact(
            "conversation_history",
            [
                Message.user("What gene is prioritised?"),
                Message.assistant("CUTA is the top gene."),
                Message.user("Actually ITPR3 has merit. Reconsider."),
            ],
        )

        stage = AgenticParseTemplateStage()
        stage.execute(ctx)

        trace_file = tmp_path / ".karenina" / "traces" / "q1_trace.txt"
        assert trace_file.exists()
        content = trace_file.read_text()
        assert "What gene is prioritised?" in content
        assert "CUTA is the top gene." in content
        assert "ITPR3 has merit" in content


@pytest.mark.unit
class TestReformatTranscriptAsXml:
    """Tests for _reformat_transcript_as_xml."""

    def _reformat(self, text: str) -> str:
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        return AgenticParseTemplateStage._reformat_transcript_as_xml(text)

    def test_plain_text_returned_unchanged(self):
        """Text without transcript_prepend pattern passes through."""
        text = "Just a simple question about biology."
        assert self._reformat(text) == text

    def test_basic_two_turn_transcript(self):
        """Two user/assistant turns are wrapped in <turn> elements."""
        transcript = (
            "[__user__] What is 2+2?\n"
            "[model:assistant:text] The answer is 4.\n"
            "[__user__] Are you sure? I think it's 5.\n"
            "[model:assistant:text] I am confident the answer is 4.\n"
            "\n\n---\n\n"
            "Evaluate the response."
        )
        result = self._reformat(transcript)
        assert '<turn number="1">' in result
        assert '<turn number="2">' in result
        assert "<user>" in result
        assert "</user>" in result
        assert "<text>" in result
        assert "</text>" in result
        assert "What is 2+2?" in result
        assert "The answer is 4." in result
        assert "Are you sure?" in result
        assert "Evaluate the response." in result

    def test_tool_calls_get_name_attribute(self):
        """Tool calls are wrapped with name attribute."""
        transcript = (
            "[__user__] Look up disease X\n"
            "[agent:assistant:text] Let me search.\n"
            "[agent:assistant:tool_use] search_entities(query='X')\n"
            '[agent:tool:tool_result] {"found": true}\n'
            "[agent:assistant:text] Found it.\n"
            "\n\n---\n\n"
            "Rate."
        )
        result = self._reformat(transcript)
        assert '<tool_call name="search_entities">' in result
        assert "</tool_call>" in result
        assert '<tool_result name="search_entities">' in result
        assert "</tool_result>" in result
        assert "Let me search." in result
        assert "Found it." in result

    def test_multiline_content_preserved(self):
        """Multi-line text and tool results are kept intact."""
        transcript = (
            "[__user__] Tell me about X\n"
            "[agent:assistant:text] Here is info:\n"
            "Line 2 of the response.\n"
            "Line 3 of the response.\n"
            "[agent:assistant:tool_use] fetch(id='123')\n"
            '[agent:tool:tool_result] {"name": "X",\n'
            '  "description": "something"}\n'
            "\n\n---\n\n"
            "Done."
        )
        result = self._reformat(transcript)
        assert "Line 2 of the response." in result
        assert "Line 3 of the response." in result
        assert '"description": "something"' in result

    def test_assistant_agent_attribute(self):
        """The <assistant> element carries the agent ID."""
        transcript = "[__user__] Question\n[qwen3.5-122b-a10b:assistant:text] Answer.\n\n\n---\n\nEval."
        result = self._reformat(transcript)
        assert '<assistant agent="qwen3.5-122b-a10b">' in result

    def test_evaluation_prompt_after_turns(self):
        """The evaluation prompt (after separator) appears outside turns."""
        transcript = "[__user__] Q\n[m:assistant:text] A\n\n\n---\n\nYou are a safety guardrail."
        result = self._reformat(transcript)
        # Eval prompt must appear AFTER the last </turn>
        last_turn_close = result.rfind("</turn>")
        eval_pos = result.find("You are a safety guardrail.")
        assert last_turn_close < eval_pos

    def test_system_prompt_xml_element(self) -> None:
        """System messages should produce <system_prompt> XML elements."""
        transcript = (
            "[qwen:system:text] You are an expert.\n[__user__] What is BCL2?\n[qwen:assistant:text] BCL2 is a protein."
        )
        question_text = transcript + "\n\n---\n\nEvaluate the answer."
        result = self._reformat(question_text)
        assert '<system_prompt agent="qwen">' in result
        assert "You are an expert." in result
        assert "<user>" in result
        assert '<assistant agent="qwen">' in result
        assert "Evaluate the answer." in result

    def test_no_system_prompt_on_second_turn_same_agent(self) -> None:
        """Second turn with same agent should not have system_prompt."""
        transcript = (
            "[qwen:system:text] You are an expert.\n"
            "[__user__] Q1?\n"
            "[qwen:assistant:text] A1\n"
            "[__user__] Q2?\n"
            "[qwen:assistant:text] A2"
        )
        question_text = transcript + "\n\n---\n\nEvaluate."
        result = self._reformat(question_text)
        assert result.count("<system_prompt") == 1
        assert result.count("<turn") == 2


@pytest.mark.unit
class TestTraceContentOffloading:
    """Tests for offloading large content blocks to artifact files."""

    def _make_long_content(self, length: int) -> str:
        """Generate a JSON-like string of the given length."""
        base = '{"data": "' + "x" * (length - 20) + '"}'
        return base[:length]

    def test_long_tool_result_offloaded(self, tmp_path):
        """Tool result exceeding threshold is written to artifact file."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        long_result = self._make_long_content(3000)
        transcript = (
            "[__user__] Question\n"
            "[agent:assistant:text] Let me check.\n"
            "[agent:assistant:tool_use] search(q='x')\n"
            f"[agent:tool:tool_result] {long_result}\n"
            "\n\n---\n\n"
            "Evaluate."
        )
        artifacts_dir = tmp_path / "artifacts"
        result = AgenticParseTemplateStage._reformat_transcript_as_xml(
            transcript,
            artifacts_dir=artifacts_dir,
            truncation_threshold=2000,
        )
        # Inline reference instead of full content
        assert 'offloaded="true"' in result
        assert "[Content offloaded:" in result
        assert long_result not in result
        # Artifact file created
        assert artifacts_dir.exists()
        artifact_files = list(artifacts_dir.iterdir())
        assert len(artifact_files) == 1
        assert artifact_files[0].read_text() == long_result

    def test_short_content_stays_inline(self, tmp_path):
        """Content below threshold is not offloaded."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        transcript = (
            "[__user__] Q\n"
            "[agent:assistant:text] Short answer.\n"
            "[agent:assistant:tool_use] search(q='x')\n"
            '[agent:tool:tool_result] {"found": true}\n'
            "\n\n---\n\n"
            "Eval."
        )
        artifacts_dir = tmp_path / "artifacts"
        result = AgenticParseTemplateStage._reformat_transcript_as_xml(
            transcript,
            artifacts_dir=artifacts_dir,
            truncation_threshold=2000,
        )
        assert 'offloaded="true"' not in result
        assert '{"found": true}' in result
        assert not artifacts_dir.exists()

    def test_long_text_block_offloaded(self, tmp_path):
        """Long assistant text block also gets offloaded."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        long_text = "A" * 3000
        transcript = f"[__user__] Q\n[agent:assistant:text] {long_text}\n\n\n---\n\nEval."
        artifacts_dir = tmp_path / "artifacts"
        result = AgenticParseTemplateStage._reformat_transcript_as_xml(
            transcript,
            artifacts_dir=artifacts_dir,
            truncation_threshold=2000,
        )
        assert 'offloaded="true"' in result
        assert long_text not in result
        artifact_files = list(artifacts_dir.iterdir())
        assert len(artifact_files) == 1
        assert artifact_files[0].read_text() == long_text

    def test_no_offloading_without_artifacts_dir(self):
        """When artifacts_dir is None, long content stays inline."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        long_result = "X" * 5000
        transcript = (
            "[__user__] Q\n"
            f"[agent:assistant:tool_use] fetch(id='1')\n"
            f"[agent:tool:tool_result] {long_result}\n"
            "\n\n---\n\n"
            "Eval."
        )
        result = AgenticParseTemplateStage._reformat_transcript_as_xml(
            transcript,
            artifacts_dir=None,
            truncation_threshold=2000,
        )
        assert 'offloaded="true"' not in result
        assert long_result in result

    def test_multiple_blocks_offloaded_with_unique_filenames(self, tmp_path):
        """Each offloaded block gets a unique numbered filename."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        long_a = "A" * 3000
        long_b = "B" * 3000
        transcript = (
            "[__user__] Q\n"
            "[agent:assistant:tool_use] search(q='a')\n"
            f"[agent:tool:tool_result] {long_a}\n"
            "[agent:assistant:tool_use] search(q='b')\n"
            f"[agent:tool:tool_result] {long_b}\n"
            "\n\n---\n\n"
            "Eval."
        )
        artifacts_dir = tmp_path / "artifacts"
        AgenticParseTemplateStage._reformat_transcript_as_xml(
            transcript,
            artifacts_dir=artifacts_dir,
            truncation_threshold=2000,
        )
        artifact_files = sorted(artifacts_dir.iterdir())
        assert len(artifact_files) == 2
        contents = {f.read_text() for f in artifact_files}
        assert long_a in contents
        assert long_b in contents

    def test_offloaded_reference_contains_file_path(self, tmp_path):
        """The inline reference includes the artifact file path."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        long_result = "Z" * 3000
        transcript = (
            "[__user__] Q\n"
            "[agent:assistant:tool_use] fetch(id='1')\n"
            f"[agent:tool:tool_result] {long_result}\n"
            "\n\n---\n\n"
            "Eval."
        )
        artifacts_dir = tmp_path / "artifacts"
        result = AgenticParseTemplateStage._reformat_transcript_as_xml(
            transcript,
            artifacts_dir=artifacts_dir,
            truncation_threshold=2000,
        )
        artifact_file = list(artifacts_dir.iterdir())[0]
        assert str(artifact_file) in result

    def test_write_trace_file_reads_env_threshold(self, tmp_path, monkeypatch):
        """_write_trace_file respects KARENINA_TRACE_TRUNCATION_THRESHOLD env var."""
        from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import (
            AgenticParseTemplateStage,
        )

        monkeypatch.setenv("KARENINA_TRACE_TRUNCATION_THRESHOLD", "100")
        long_text = "Y" * 200
        transcript = f"[__user__] Q\n[agent:assistant:text] {long_text}\n\n\n---\n\nEval."
        path = AgenticParseTemplateStage._write_trace_file(
            workspace_path=tmp_path,
            trace="response",
            question_id="q1",
            question_text=transcript,
        )
        content = path.read_text()
        assert 'offloaded="true"' in content
        assert long_text not in content
        artifacts_dir = tmp_path / ".karenina" / "traces" / "artifacts"
        assert artifacts_dir.exists()
