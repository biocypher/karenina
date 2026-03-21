"""Tests for AgenticTraitEvaluator."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from karenina.ports import AgentResult, Message, Role
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.entities.rubric import AgenticRubricTrait
from karenina.schemas.outputs.rubric import SingleBooleanScore, SingleNumericScore


def _make_trait(
    name: str = "test_trait",
    kind: str = "boolean",
    context_mode: str = "trace_only",
) -> AgenticRubricTrait:
    """Create a test AgenticRubricTrait."""
    return AgenticRubricTrait(
        name=name,
        description="Check quality.",
        kind=kind,
        context_mode=context_mode,
    )


def _make_model_config() -> ModelConfig:
    """Create a test ModelConfig with agent-capable interface."""
    return ModelConfig(
        id="test-agent-model",
        model_name="test-model",
        model_provider="anthropic",
        interface="claude_agent_sdk",
    )


def _make_agent_result(
    raw_trace: str = "Investigation complete.",
    turns: int = 3,
    limit_reached: bool = False,
) -> AgentResult:
    """Create a realistic AgentResult for test mocking."""
    return AgentResult(
        final_response="Summary of findings.",
        raw_trace=raw_trace,
        trace_messages=[Message.assistant("Summary of findings.")],
        usage=UsageMetadata(input_tokens=100, output_tokens=50),
        turns=turns,
        limit_reached=limit_reached,
    )


def _make_parse_result(parsed: object) -> MagicMock:
    """Create a mock ParsePortResult."""
    mock = MagicMock()
    mock.parsed = parsed
    return mock


@pytest.mark.unit
class TestAgenticTraitEvaluator:
    """Tests for the agentic trait evaluator."""

    def test_evaluate_single_boolean_trait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single boolean trait returns bool score and trace.

        trace_only + workspace_path=None routes through LLMPort, not AgentPort.
        """
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait
        from karenina.ports.llm import LLMResponse

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = LLMResponse(
            content="I investigated and found the code is clean.",
            usage=UsageMetadata(input_tokens=100, output_tokens=50),
        )

        fake_parser = MagicMock()
        parsed_score = SingleBooleanScore(result=True)
        fake_parser.parse_to_pydantic.return_value = _make_parse_result(parsed_score)

        monkeypatch.setattr(agentic_trait, "get_llm", lambda _model: fake_llm)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        trait = _make_trait()

        score, trace = evaluator.evaluate_trait(
            trait=trait,
            question_text="Write a function.",
            raw_llm_response="def foo(): pass",
            workspace_path=None,
        )

        assert score is True
        assert trace is not None
        assert "investigated" in trace

    def test_evaluate_single_score_trait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single score trait returns int score (trace_only via LLMPort)."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait
        from karenina.ports.llm import LLMResponse

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = LLMResponse(
            content="Code quality is moderate.",
            usage=UsageMetadata(input_tokens=100, output_tokens=50),
        )

        fake_parser = MagicMock()
        parsed_score = SingleNumericScore(score=3)
        fake_parser.parse_to_pydantic.return_value = _make_parse_result(parsed_score)

        monkeypatch.setattr(agentic_trait, "get_llm", lambda _model: fake_llm)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        trait = _make_trait(kind="score")

        score, trace = evaluator.evaluate_trait(
            trait=trait,
            question_text="Write a function.",
            raw_llm_response="def foo(): pass",
            workspace_path=None,
        )

        assert score == 3

    def test_evaluate_trait_agent_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM failure returns None score and None trace (trace_only via LLMPort)."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = RuntimeError("LLM timed out")

        monkeypatch.setattr(agentic_trait, "get_llm", lambda _model: fake_llm)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        trait = _make_trait()

        score, trace = evaluator.evaluate_trait(
            trait=trait,
            question_text="Write a function.",
            raw_llm_response="def foo(): pass",
            workspace_path=None,
        )

        assert score is None
        assert trace is None

    def test_extraction_failure_preserves_trace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parser failure returns None score but preserves the investigation trace."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait
        from karenina.ports.llm import LLMResponse

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = LLMResponse(
            content="I found several issues.",
            usage=UsageMetadata(input_tokens=100, output_tokens=50),
        )

        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.side_effect = RuntimeError("Parse failed")

        monkeypatch.setattr(agentic_trait, "get_llm", lambda _model: fake_llm)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        trait = _make_trait()

        score, trace = evaluator.evaluate_trait(
            trait=trait,
            question_text="Write a function.",
            raw_llm_response="def foo(): pass",
            workspace_path=None,
        )

        assert score is None
        assert trace == "I found several issues."

    def test_workspace_only_mode_excludes_trace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """workspace_only context_mode does not include raw_llm_response in prompt."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        fake_agent = MagicMock()
        fake_agent.run.return_value = _make_agent_result(
            raw_trace="Checked workspace.",
            turns=2,
            limit_reached=False,
        )
        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.return_value = _make_parse_result(
            SingleBooleanScore(result=True),
        )

        monkeypatch.setattr(agentic_trait, "get_agent", lambda _model: fake_agent)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        trait = _make_trait(context_mode="workspace_only")

        evaluator.evaluate_trait(
            trait=trait,
            question_text="Write a function.",
            raw_llm_response="def foo(): pass",
            workspace_path="/tmp/workspace",
        )

        # Verify the user message sent to agent does NOT contain the trace
        call_args = fake_agent.run.call_args
        messages = call_args.kwargs.get("messages", call_args[0][0] if call_args[0] else [])
        user_msgs = [m for m in messages if m.role == Role.USER]
        assert len(user_msgs) == 1
        user_text = user_msgs[0].text
        assert "def foo(): pass" not in user_text
        assert "/tmp/workspace" in user_text

    def test_trace_only_mode_excludes_workspace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """trace_only context_mode does not include workspace path in prompt."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        fake_agent = MagicMock()
        fake_agent.run.return_value = _make_agent_result(
            raw_trace="Checked trace only.",
        )
        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.return_value = _make_parse_result(
            SingleBooleanScore(result=True),
        )

        monkeypatch.setattr(agentic_trait, "get_agent", lambda _model: fake_agent)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        trait = _make_trait(context_mode="trace_only")

        evaluator.evaluate_trait(
            trait=trait,
            question_text="Write a function.",
            raw_llm_response="def foo(): pass",
            workspace_path="/tmp/workspace",
        )

        call_args = fake_agent.run.call_args
        messages = call_args.kwargs.get("messages", call_args[0][0] if call_args[0] else [])
        user_msgs = [m for m in messages if m.role == Role.USER]
        user_text = user_msgs[0].text
        assert "def foo(): pass" in user_text
        assert "/tmp/workspace" not in user_text

    def test_trace_and_workspace_mode_includes_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """trace_and_workspace context_mode includes both trace and workspace."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        fake_agent = MagicMock()
        fake_agent.run.return_value = _make_agent_result(
            raw_trace="Checked both.",
        )
        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.return_value = _make_parse_result(
            SingleBooleanScore(result=True),
        )

        monkeypatch.setattr(agentic_trait, "get_agent", lambda _model: fake_agent)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        trait = _make_trait(context_mode="trace_and_workspace")

        evaluator.evaluate_trait(
            trait=trait,
            question_text="Write a function.",
            raw_llm_response="def foo(): pass",
            workspace_path="/tmp/workspace",
        )

        call_args = fake_agent.run.call_args
        messages = call_args.kwargs.get("messages", call_args[0][0] if call_args[0] else [])
        user_msgs = [m for m in messages if m.role == Role.USER]
        user_text = user_msgs[0].text
        assert "def foo(): pass" in user_text
        assert "/tmp/workspace" in user_text

    def test_run_extraction_is_public(self) -> None:
        """run_extraction is a public method (needed by Task 8 shared strategy)."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        assert hasattr(evaluator, "run_extraction")
        assert callable(evaluator.run_extraction)


class _TestFindings(BaseModel):
    count: int = Field(description="Count")
    items: list[str] = Field(description="Items")


@pytest.mark.unit
class TestExtractTemplate:
    def test_extract_template_returns_model_dump(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Template extraction returns the Pydantic model_dump dict."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        trait = AgenticRubricTrait(
            name="test",
            description="Test.",
            kind=_TestFindings,
            higher_is_better=None,
            context_mode="trace_only",
        )

        mock_parsed = _TestFindings(count=5, items=["a", "b"])
        mock_parse_result = _make_parse_result(mock_parsed)

        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = mock_parse_result

        monkeypatch.setattr(agentic_trait, "get_parser", lambda _: mock_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(_make_model_config())
        result = evaluator.run_extraction(trait, "Investigation trace text")

        assert result == {"count": 5, "items": ["a", "b"]}
        mock_parser.parse_to_pydantic.assert_called_once()
        call_args = mock_parser.parse_to_pydantic.call_args
        assert call_args[0][1] is _TestFindings  # Second arg is the class


@pytest.mark.unit
class TestTraceMaterialization:
    """Tests for trace file path support in agentic trait evaluation."""

    def test_evaluate_trait_accepts_trace_file_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """evaluate_trait() should accept an optional trace_file_path parameter."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        trait = AgenticRubricTrait(
            name="test_mat",
            description="Check quality.",
            kind="boolean",
            materialize_trace=True,
            context_mode="trace_only",
        )

        mock_agent_result = _make_agent_result(raw_trace="investigation output")
        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_agent_result
        monkeypatch.setattr(agentic_trait, "get_agent", lambda _: mock_agent)

        mock_parsed = SingleBooleanScore(result=True)
        mock_parse_result = _make_parse_result(mock_parsed)
        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = mock_parse_result
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _: mock_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(_make_model_config())
        score, trace = evaluator.evaluate_trait(
            trait,
            question_text="Q",
            raw_llm_response="Response",
            workspace_path=None,
            trace_file_path=Path("/tmp/test_trace.txt"),
        )
        assert score is True

    def test_materialize_trace_replaces_inline_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When materialize_trace=True and trace_file_path given, prompt uses file reference."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        trait = AgenticRubricTrait(
            name="test_mat_prompt",
            description="Check quality.",
            kind="boolean",
            materialize_trace=True,
            context_mode="trace_only",
        )

        mock_agent_result = _make_agent_result(raw_trace="investigation output")
        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_agent_result
        monkeypatch.setattr(agentic_trait, "get_agent", lambda _: mock_agent)

        mock_parsed = SingleBooleanScore(result=True)
        mock_parse_result = _make_parse_result(mock_parsed)
        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = mock_parse_result
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _: mock_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(_make_model_config())
        evaluator.evaluate_trait(
            trait,
            question_text="Q",
            raw_llm_response="The full inline response text here",
            workspace_path=None,
            trace_file_path=Path("/tmp/test_trace.txt"),
        )

        # The agent prompt should reference the file path, not the inline response
        call_args = mock_agent.run.call_args
        messages = call_args.kwargs.get("messages", call_args[0][0] if call_args[0] else [])
        user_msgs = [m for m in messages if m.role == Role.USER]
        user_text = user_msgs[0].text
        assert "/tmp/test_trace.txt" in user_text
        assert "The full inline response text here" not in user_text

    def test_no_trace_file_path_uses_inline_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without trace_file_path, the prompt includes inline trace content.

        trace_only + workspace_path=None + no trace_file routes via LLMPort.
        """
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait
        from karenina.ports.llm import LLMResponse

        trait = AgenticRubricTrait(
            name="test_no_mat",
            description="Check quality.",
            kind="boolean",
            materialize_trace=True,
            context_mode="trace_only",
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = LLMResponse(
            content="investigation output",
            usage=UsageMetadata(input_tokens=100, output_tokens=50),
        )
        monkeypatch.setattr(agentic_trait, "get_llm", lambda _: mock_llm)

        mock_parsed = SingleBooleanScore(result=True)
        mock_parse_result = _make_parse_result(mock_parsed)
        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = mock_parse_result
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _: mock_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(_make_model_config())
        evaluator.evaluate_trait(
            trait,
            question_text="Q",
            raw_llm_response="Inline trace content",
            workspace_path=None,
            trace_file_path=None,
        )

        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        user_msgs = [m for m in messages if m.role == Role.USER]
        user_text = user_msgs[0].text
        assert "Inline trace content" in user_text

    def test_materialize_false_ignores_trace_file_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When materialize_trace=False, trace_file_path is ignored; inline content is used.

        trace_only + workspace_path=None + materialize_trace=False routes via LLMPort.
        """
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait
        from karenina.ports.llm import LLMResponse

        trait = AgenticRubricTrait(
            name="test_no_flag",
            description="Check quality.",
            kind="boolean",
            materialize_trace=False,
            context_mode="trace_only",
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = LLMResponse(
            content="investigation output",
            usage=UsageMetadata(input_tokens=100, output_tokens=50),
        )
        monkeypatch.setattr(agentic_trait, "get_llm", lambda _: mock_llm)

        mock_parsed = SingleBooleanScore(result=True)
        mock_parse_result = _make_parse_result(mock_parsed)
        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = mock_parse_result
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _: mock_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(_make_model_config())
        evaluator.evaluate_trait(
            trait,
            question_text="Q",
            raw_llm_response="Inline trace content",
            workspace_path=None,
            trace_file_path=Path("/tmp/test_trace.txt"),
        )

        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        user_msgs = [m for m in messages if m.role == Role.USER]
        user_text = user_msgs[0].text
        assert "Inline trace content" in user_text
        assert "/tmp/test_trace.txt" not in user_text
