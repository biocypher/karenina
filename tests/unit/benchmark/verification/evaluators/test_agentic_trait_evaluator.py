"""Tests for AgenticTraitEvaluator."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from karenina.ports import AgentResult, Message, Role, ToolUseContent
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

        evaluation = evaluator.evaluate_trait(
            trait=trait,
            question_text="Write a function.",
            raw_llm_response="def foo(): pass",
            workspace_path=None,
        )

        assert evaluation.score is True
        assert evaluation.trace is not None
        assert "investigated" in evaluation.trace

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

        evaluation = evaluator.evaluate_trait(
            trait=trait,
            question_text="Write a function.",
            raw_llm_response="def foo(): pass",
            workspace_path=None,
        )

        assert evaluation.score == 3

    def test_evaluate_trait_agent_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM failure returns None score and None trace (trace_only via LLMPort)."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = RuntimeError("LLM timed out")

        monkeypatch.setattr(agentic_trait, "get_llm", lambda _model: fake_llm)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        trait = _make_trait()

        evaluation = evaluator.evaluate_trait(
            trait=trait,
            question_text="Write a function.",
            raw_llm_response="def foo(): pass",
            workspace_path=None,
        )

        assert evaluation.score is None
        assert evaluation.trace is None

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

        evaluation = evaluator.evaluate_trait(
            trait=trait,
            question_text="Write a function.",
            raw_llm_response="def foo(): pass",
            workspace_path=None,
        )

        assert evaluation.score is None
        assert evaluation.trace == "I found several issues."

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


class _TraceFractions(BaseModel):
    included_message_count: int = Field(description="Included messages")
    environment_setup: float = Field(description="Environment setup fraction")


@pytest.mark.unit
class TestExtractTemplate:
    def test_extract_template_uses_local_json_without_parser(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Template extraction validates final investigation JSON locally."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        trait = AgenticRubricTrait(
            name="trace_category_fractions_v4",
            description="Test.",
            kind=_TraceFractions,
            higher_is_better=None,
            context_mode="trace_only",
        )

        parser = MagicMock()
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _: parser)

        trace = (
            "--- AI Message ---\n"
            "Final classification:\n"
            "```json\n"
            '{"included_message_count": 3, "environment_setup": 0.25}\n'
            "```\n"
        )

        evaluator = agentic_trait.AgenticTraitEvaluator(_make_model_config())
        result = evaluator.run_extraction(trait, trace)

        assert result == {"included_message_count": 3, "environment_setup": 0.25}
        parser.parse_to_pydantic.assert_not_called()

    def test_extract_template_falls_back_to_parser_for_prose_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Parser remains a backstop for collected prose/table investigations."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        trait = AgenticRubricTrait(
            name="trace_category_fractions_v4",
            description="Test.",
            kind=_TraceFractions,
            higher_is_better=None,
            context_mode="trace_only",
        )

        mock_parser = MagicMock()
        mock_parser.parse_to_pydantic.return_value = _make_parse_result(
            _TraceFractions(included_message_count=23, environment_setup=0.1304)
        )

        monkeypatch.setattr(agentic_trait, "get_parser", lambda _: mock_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(_make_model_config())
        result = evaluator.run_extraction(
            trait,
            "## Investigation Summary\n\nincluded messages: 23\nenvironment_setup: 0.1304\n",
        )

        assert result == {"included_message_count": 23, "environment_setup": 0.1304}
        mock_parser.parse_to_pydantic.assert_called_once()

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
        evaluation = evaluator.evaluate_trait(
            trait,
            question_text="Q",
            raw_llm_response="Response",
            workspace_path=None,
            trace_file_path=Path("/tmp/test_trace.txt"),
        )
        assert evaluation.score is True

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


def _make_mcp_model_config() -> ModelConfig:
    """Create a judge ModelConfig with one MCP server."""
    return ModelConfig(
        id="test-agent-model",
        model_name="test-model",
        model_provider="anthropic",
        interface="claude_agent_sdk",
        mcp_urls_dict={"citations": "http://localhost:8765/mcp"},
    )


@pytest.mark.unit
class TestJudgeMcpServers:
    """A judge with MCP servers investigates through AgentPort with those servers."""

    def test_trace_only_judge_with_mcp_uses_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """trace_only with no workspace takes the agent path when mcp_urls_dict is set."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        fake_agent = MagicMock()
        fake_agent.run.return_value = _make_agent_result(raw_trace="Looked up the citation.")
        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.return_value = _make_parse_result(SingleBooleanScore(result=True))

        def _no_llm(_model: ModelConfig) -> None:
            raise AssertionError("an MCP judge must not use the single-call LLM path")

        monkeypatch.setattr(agentic_trait, "get_agent", lambda _model: fake_agent)
        monkeypatch.setattr(agentic_trait, "get_llm", _no_llm)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_mcp_model_config())
        evaluation = evaluator.evaluate_trait(
            trait=_make_trait(context_mode="trace_only"),
            question_text="Q",
            raw_llm_response="See PMID 123.",
            workspace_path=None,
        )

        assert evaluation.score is True
        assert evaluation.trace == "Looked up the citation."
        fake_agent.run.assert_called_once()
        assert fake_agent.run.call_args.kwargs["mcp_servers"] == {
            "citations": {"type": "http", "url": "http://localhost:8765/mcp"},
        }

    def test_trace_only_judge_without_mcp_keeps_llm_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A built-in-only judge (no MCP) still makes a single LLM call."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait
        from karenina.ports.llm import LLMResponse

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = LLMResponse(
            content="Verdict: fine.",
            usage=UsageMetadata(input_tokens=10, output_tokens=5),
        )
        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.return_value = _make_parse_result(SingleBooleanScore(result=True))

        def _no_agent(_model: ModelConfig) -> None:
            raise AssertionError("a judge without MCP servers must keep the LLM path")

        monkeypatch.setattr(agentic_trait, "get_agent", _no_agent)
        monkeypatch.setattr(agentic_trait, "get_llm", lambda _model: fake_llm)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        evaluation = evaluator.evaluate_trait(
            trait=_make_trait(context_mode="trace_only"),
            question_text="Q",
            raw_llm_response="Answer.",
            workspace_path=None,
        )

        assert evaluation.score is True
        fake_llm.invoke.assert_called_once()

    @pytest.mark.parametrize(
        ("model_config", "expected_servers"),
        [
            (_make_model_config(), {}),
            (
                _make_mcp_model_config(),
                {"citations": {"type": "http", "url": "http://localhost:8765/mcp"}},
            ),
        ],
    )
    def test_workspace_agent_receives_judge_mcp_servers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_config: ModelConfig,
        expected_servers: dict[str, dict[str, str]],
    ) -> None:
        """The agent path passes mcp_servers built exactly like generate_answer."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        fake_agent = MagicMock()
        fake_agent.run.return_value = _make_agent_result()
        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.return_value = _make_parse_result(SingleBooleanScore(result=True))

        monkeypatch.setattr(agentic_trait, "get_agent", lambda _model: fake_agent)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=model_config)
        evaluator.evaluate_trait(
            trait=_make_trait(context_mode="workspace_only"),
            question_text="Q",
            raw_llm_response="Answer.",
            workspace_path="/tmp/workspace",
        )

        assert fake_agent.run.call_args.kwargs["mcp_servers"] == expected_servers

    def test_mcp_judge_prompt_mentions_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without file or code tools, an MCP judge is told to use its tools, not only the text."""
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait
        from karenina.ports import PortCapabilities

        fake_agent = MagicMock()
        fake_agent.capabilities = PortCapabilities()
        fake_agent.run.return_value = _make_agent_result()
        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.return_value = _make_parse_result(SingleBooleanScore(result=True))

        monkeypatch.setattr(agentic_trait, "get_agent", lambda _model: fake_agent)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_mcp_model_config())
        evaluator.evaluate_trait(
            trait=_make_trait(context_mode="trace_only"),
            question_text="Q",
            raw_llm_response="Answer.",
            workspace_path=None,
        )

        messages = fake_agent.run.call_args.kwargs["messages"]
        system_text = next(m.text for m in messages if m.role == Role.SYSTEM)
        assert "tools available to you" in system_text
        assert "Use only the response text" not in system_text

    def test_build_mcp_servers_empty_without_urls(self) -> None:
        """build_mcp_servers returns an empty mapping when no MCP servers are configured."""
        from karenina.benchmark.verification.evaluators.rubric.agentic_trait import build_mcp_servers

        assert build_mcp_servers(_make_model_config()) == {}


@pytest.mark.unit
class TestJudgeUsage:
    """The evaluator returns the judge's investigation and extraction usage."""

    def test_llm_path_returns_llm_and_parser_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait
        from karenina.ports import ParsePortResult
        from karenina.ports.llm import LLMResponse

        llm_usage = UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150, web_search_queries=2)
        parser_usage = UsageMetadata(input_tokens=20, output_tokens=5, total_tokens=25)

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = LLMResponse(content="Looks right.", usage=llm_usage)
        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.return_value = ParsePortResult(
            parsed=SingleBooleanScore(result=True),
            usage=parser_usage,
        )
        monkeypatch.setattr(agentic_trait, "get_llm", lambda _model: fake_llm)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        evaluation = evaluator.evaluate_trait(
            trait=_make_trait(),
            question_text="Q",
            raw_llm_response="Answer.",
            workspace_path=None,
        )

        assert evaluation.investigation_usage is llm_usage
        assert evaluation.extraction_usage is parser_usage

    def test_agent_path_returns_agent_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        agent_result = _make_agent_result()
        fake_agent = MagicMock()
        fake_agent.run.return_value = agent_result
        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.return_value = _make_parse_result(SingleBooleanScore(result=True))
        monkeypatch.setattr(agentic_trait, "get_agent", lambda _model: fake_agent)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        evaluation = evaluator.evaluate_trait(
            trait=_make_trait(context_mode="workspace_only"),
            question_text="Q",
            raw_llm_response="Answer.",
            workspace_path="/tmp/workspace",
        )

        assert evaluation.investigation_usage is agent_result.usage

    def test_local_template_extraction_has_no_parser_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        parser = MagicMock()
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: parser)
        trait = AgenticRubricTrait(
            name="findings",
            description="Test.",
            kind=_TestFindings,
            higher_is_better=None,
            context_mode="trace_only",
        )

        evaluator = agentic_trait.AgenticTraitEvaluator(_make_model_config())
        evaluator.run_extraction(trait, '```json\n{"count": 1, "items": ["a"]}\n```')

        assert evaluator.last_extraction_usage is None
        parser.parse_to_pydantic.assert_not_called()


@pytest.mark.unit
class TestInvestigationMessages:
    """The evaluator keeps the judge's structured investigation messages."""

    def test_agent_path_keeps_trace_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait

        tool_call = Message(
            role=Role.ASSISTANT,
            content=[ToolUseContent(id="call_1", name="europepmc_search", input={"query": "PMID 123"})],
        )
        agent_result = AgentResult(
            final_response="Citation checks out.",
            raw_trace="--- AI Message ---\nCitation checks out.",
            trace_messages=[
                tool_call,
                Message.tool_result("call_1", "Found PMID 123."),
                Message.assistant("Citation checks out."),
            ],
            usage=UsageMetadata(input_tokens=10, output_tokens=5),
            turns=2,
            limit_reached=False,
        )
        fake_agent = MagicMock()
        fake_agent.run.return_value = agent_result
        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.return_value = _make_parse_result(SingleBooleanScore(result=True))
        monkeypatch.setattr(agentic_trait, "get_agent", lambda _model: fake_agent)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_mcp_model_config())
        evaluation = evaluator.evaluate_trait(
            trait=_make_trait(),
            question_text="Q",
            raw_llm_response="See PMID 123.",
            workspace_path=None,
        )

        assert evaluation.messages == agent_result.trace_messages

    def test_llm_path_stores_single_assistant_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from karenina.benchmark.verification.evaluators.rubric import agentic_trait
        from karenina.ports.llm import LLMResponse

        fake_llm = MagicMock()
        fake_llm.invoke.return_value = LLMResponse(
            content="The answer is well supported.",
            usage=UsageMetadata(input_tokens=10, output_tokens=5),
        )
        fake_parser = MagicMock()
        fake_parser.parse_to_pydantic.side_effect = RuntimeError("Parse failed")
        monkeypatch.setattr(agentic_trait, "get_llm", lambda _model: fake_llm)
        monkeypatch.setattr(agentic_trait, "get_parser", lambda _model: fake_parser)

        evaluator = agentic_trait.AgenticTraitEvaluator(model_config=_make_model_config())
        evaluation = evaluator.evaluate_trait(
            trait=_make_trait(),
            question_text="Q",
            raw_llm_response="Answer.",
            workspace_path=None,
        )

        # Extraction failed, but the investigation record is kept.
        assert evaluation.score is None
        assert evaluation.messages is not None
        assert len(evaluation.messages) == 1
        assert evaluation.messages[0].role == Role.ASSISTANT
        assert evaluation.messages[0].text == "The answer is well supported."
