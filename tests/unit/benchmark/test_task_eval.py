"""Tests for TaskEval redesign: Message-aware evaluation with merge strategies."""

from typing import Any

import pytest
from pydantic import ValidationError

from karenina.benchmark.task_eval import LogEvent, StepEval, TaskEval, TaskEvalResult
from karenina.benchmark.task_eval.helpers import (
    convert_string_logs_to_messages,
    merge_logs_and_traces,
)
from karenina.ports.messages import Message, Role, ToolUseContent

# =============================================================================
# LogEvent Model Tests
# =============================================================================


@pytest.mark.unit
class TestLogEvent:
    """Tests for LogEvent model after cleanup."""

    def test_text_only_event(self):
        """LogEvent with just text and level."""
        event = LogEvent(level="info", text="hello")
        assert event.text == "hello"
        assert event.level == "info"
        assert event.trace_messages is None
        assert event.tags is None
        assert event.timestamp  # auto-generated

    def test_trace_only_event(self):
        """LogEvent with trace_messages and empty text (default)."""
        msgs = [Message.assistant("response")]
        event = LogEvent(level="info", trace_messages=msgs)
        assert event.text == ""
        assert event.trace_messages is not None
        assert len(event.trace_messages) == 1
        assert event.trace_messages[0].role == Role.ASSISTANT

    def test_text_defaults_to_empty(self):
        """text field defaults to empty string."""
        event = LogEvent(level="debug")
        assert event.text == ""

    def test_removed_fields_not_in_schema(self):
        """Fields that were removed are no longer in the model schema."""
        field_names = LogEvent.model_fields.keys()
        for removed in ["payload", "question_id", "is_agent_output", "output_type", "is_dict_structured", "dict_keys"]:
            assert removed not in field_names, f"{removed} should have been removed from LogEvent"

    def test_tags_preserved(self):
        """Tags field still works."""
        event = LogEvent(level="info", text="test", tags=["tag1", "tag2"])
        assert event.tags == ["tag1", "tag2"]


# =============================================================================
# log() and log_trace() Tests
# =============================================================================


@pytest.mark.unit
class TestLogAndLogTrace:
    """Tests for TaskEval.log() and TaskEval.log_trace() methods."""

    def test_log_global(self):
        """log() adds to global_logs."""
        task = TaskEval()
        task.log("hello")
        assert len(task.global_logs) == 1
        assert task.global_logs[0].text == "hello"

    def test_log_step(self):
        """log() with step_id adds to step_logs."""
        task = TaskEval()
        task.log("step output", step_id="s1", target="step")
        assert "s1" in task.step_logs
        assert len(task.step_logs["s1"]) == 1
        assert task.step_logs["s1"][0].text == "step output"
        assert len(task.global_logs) == 0  # target="step" should not add to global

    def test_log_both(self):
        """log() with target='both' adds to global and step."""
        task = TaskEval()
        task.log("both", step_id="s1", target="both")
        assert len(task.global_logs) == 1
        assert len(task.step_logs["s1"]) == 1

    def test_log_no_dict_support(self):
        """log() no longer accepts dict; only str."""
        task = TaskEval()
        with pytest.raises(ValidationError):
            task.log({"key": "val"})  # pyright: ignore[reportArgumentType]

    def test_log_trace_with_messages(self):
        """log_trace() stores Message objects in trace_messages."""
        task = TaskEval()
        msgs = [Message.assistant("hello"), Message.user("follow up")]
        task.log_trace(msgs)
        assert len(task.global_logs) == 1
        assert task.global_logs[0].trace_messages is not None
        assert len(task.global_logs[0].trace_messages) == 2
        assert task.global_logs[0].text == ""  # no text for trace-only

    def test_log_trace_with_string(self):
        """log_trace() auto-wraps string as assistant Message."""
        task = TaskEval()
        task.log_trace("simple text")
        assert len(task.global_logs) == 1
        event = task.global_logs[0]
        assert event.trace_messages is not None
        assert len(event.trace_messages) == 1
        assert event.trace_messages[0].role == Role.ASSISTANT
        assert event.trace_messages[0].text == "simple text"

    def test_log_trace_step_targeting(self):
        """log_trace() respects target parameter."""
        task = TaskEval()
        task.log_trace("only step", step_id="s1", target="step")
        assert len(task.global_logs) == 0
        assert len(task.step_logs["s1"]) == 1

    def test_log_trace_global_only(self):
        """log_trace() with target='global' doesn't add to step."""
        task = TaskEval()
        task.log_trace("global only", step_id="s1", target="global")
        assert len(task.global_logs) == 1
        assert "s1" not in task.step_logs


# =============================================================================
# Merge Strategy Tests
# =============================================================================


@pytest.mark.unit
class TestMergeStrategies:
    """Tests for merge_logs_and_traces helper."""

    def test_concatenate_text_only(self):
        """Concatenate strategy with text-only logs produces Messages."""
        logs = [
            LogEvent(level="info", text="first"),
            LogEvent(level="info", text="second"),
        ]
        text, messages = merge_logs_and_traces(logs, strategy="concatenate")
        assert messages is not None
        assert len(messages) == 2
        assert "first" in text
        assert "second" in text

    def test_concatenate_traces_only_logs(self):
        """Concatenate strategy with trace-only logs."""
        msgs = [Message.assistant("traced response")]
        logs = [LogEvent(level="info", trace_messages=msgs)]
        text, messages = merge_logs_and_traces(logs, strategy="concatenate")
        assert messages is not None
        assert len(messages) == 1
        assert "traced response" in text

    def test_concatenate_mixed(self):
        """Concatenate strategy combines text and trace logs."""
        logs = [
            LogEvent(level="info", text="text log"),
            LogEvent(level="info", trace_messages=[Message.assistant("trace log")]),
        ]
        text, messages = merge_logs_and_traces(logs, strategy="concatenate")
        assert messages is not None
        assert len(messages) == 2
        assert "text log" in text
        assert "trace log" in text

    def test_traces_only_ignores_text(self):
        """traces_only strategy ignores text-only logs."""
        logs = [
            LogEvent(level="info", text="ignored"),
            LogEvent(level="info", trace_messages=[Message.assistant("kept")]),
        ]
        text, messages = merge_logs_and_traces(logs, strategy="traces_only")
        assert messages is not None
        assert len(messages) == 1
        assert "ignored" not in text
        assert "kept" in text

    def test_traces_only_empty_when_no_traces(self):
        """traces_only returns empty when only text logs exist."""
        logs = [LogEvent(level="info", text="text only")]
        text, messages = merge_logs_and_traces(logs, strategy="traces_only")
        assert text == ""
        assert messages is None

    def test_empty_logs(self):
        """Empty log list returns empty results."""
        text, messages = merge_logs_and_traces([], strategy="concatenate")
        assert text == ""
        assert messages is None

    def test_default_strategy_is_concatenate(self):
        """Default strategy parameter is 'concatenate'."""
        logs = [LogEvent(level="info", text="hello")]
        text, messages = merge_logs_and_traces(logs)
        assert messages is not None
        assert "hello" in text


# =============================================================================
# Agent Metrics Tests
# =============================================================================


@pytest.mark.unit
class TestAgentMetrics:
    """Tests for automatic agent metrics extraction from Message traces."""

    def test_metrics_extracted_from_trace(self):
        """Agent metrics are extracted when trace_messages contain tool calls."""
        from karenina.benchmark.verification.utils.trace_agent_metrics import (
            extract_agent_metrics_from_messages,
        )

        messages = [
            Message.assistant(
                "Let me search for that", tool_calls=[ToolUseContent(id="call_1", name="search", input={"q": "test"})]
            ),
            Message.tool_result("call_1", "search results here"),
            Message.assistant("The answer is 42"),
        ]
        metrics = extract_agent_metrics_from_messages(messages)
        assert metrics["iterations"] == 2
        assert metrics["tool_calls"] == 1
        assert "search" in metrics["tools_used"]

    def test_metrics_none_for_text_only(self):
        """No metrics when only text logs (no Messages)."""
        logs = [LogEvent(level="info", text="just text")]
        _, messages = merge_logs_and_traces(logs)
        # Text-only logs are converted to Messages in concatenate mode
        # but those are simple assistant messages with no tool calls
        if messages:
            from karenina.benchmark.verification.utils.trace_agent_metrics import (
                extract_agent_metrics_from_messages,
            )

            metrics = extract_agent_metrics_from_messages(messages)
            assert metrics["tool_calls"] == 0
            assert metrics["tools_used"] == []


# =============================================================================
# Helpers Tests
# =============================================================================


@pytest.mark.unit
class TestHelpers:
    """Tests for helper functions."""

    def test_convert_string_logs_to_messages(self):
        """convert_string_logs_to_messages wraps each text as assistant Message."""
        messages = convert_string_logs_to_messages(["hello", "world"])
        assert len(messages) == 2
        assert all(m.role == Role.ASSISTANT for m in messages)
        assert messages[0].text == "hello"
        assert messages[1].text == "world"

    def test_convert_string_logs_skips_empty(self):
        """convert_string_logs_to_messages skips empty strings."""
        messages = convert_string_logs_to_messages(["hello", "", "world"])
        assert len(messages) == 2

    def test_convert_string_logs_empty_input(self):
        """convert_string_logs_to_messages returns empty list for empty input."""
        messages = convert_string_logs_to_messages([])
        assert messages == []


# =============================================================================
# Evaluation Loop Tests (mock run_single_model_verification)
# =============================================================================


@pytest.mark.unit
class TestEvaluationLoop:
    """Tests for the evaluation loop with mocked verification pipeline."""

    def _make_mock_verification_result(self):
        """Create a mock VerificationResult for testing."""
        from karenina.schemas.verification import VerificationResult
        from karenina.schemas.verification.model_identity import ModelIdentity
        from karenina.schemas.verification.result_components import VerificationResultMetadata

        metadata = VerificationResultMetadata(
            question_id="test_q",
            template_id="test_t",
            completed_without_errors=True,
            question_text="test question",
            answering=ModelIdentity(interface="mock", model_name="mock"),
            parsing=ModelIdentity(interface="mock", model_name="mock"),
            execution_time=0.1,
            timestamp="2026-01-01T00:00:00",
            result_id="abcd1234abcd1234",
        )
        return VerificationResult(metadata=metadata)

    def test_rubric_only_with_traces(self, monkeypatch):
        """rubric_only mode works with trace_messages."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        mock_result = self._make_mock_verification_result()

        def mock_run(*args, **kwargs):
            # Verify trace_messages are passed through in cached_answer_data
            cached = kwargs.get("cached_answer_data", {})
            assert cached.get("trace_messages") is not None
            return mock_result

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )

        from karenina.schemas.config import ModelConfig
        from karenina.schemas.verification import VerificationConfig

        task = TaskEval(task_id="test")
        task.log_trace([Message.assistant("traced output")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="quality",
                        description="Is the response high quality?",
                        kind="boolean",
                    ),
                ]
            )
        )

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test_parser",
                    model_provider="mock",
                    model_name="mock",
                    interface="langchain",
                )
            ],
            parsing_only=True,
        )

        result = task.evaluate(config)
        assert result.global_eval is not None
        assert len(result.global_eval.verification_results) > 0

    def test_template_and_rubric_mode(self, monkeypatch):
        """template_and_rubric mode passes both template and rubric."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        mock_result = self._make_mock_verification_result()
        call_kwargs: list[dict[str, Any]] = []

        def mock_run(*args, **kwargs):
            call_kwargs.append(kwargs)
            return mock_result

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )

        from karenina.schemas.config import ModelConfig
        from karenina.schemas.verification import VerificationConfig

        template_code = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: int = Field(description="The numeric answer")

    def verify(self) -> bool:
        return self.value == 4
"""

        task = TaskEval(task_id="test")
        task.log("The answer is 4")
        task.add_question(
            {
                "id": "q1",
                "question": "What is 2+2?",
                "raw_answer": "4",
                "answer_template": template_code,
            }
        )
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="clarity",
                        description="Is the answer clear?",
                        kind="boolean",
                    ),
                ]
            )
        )

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test_parser",
                    model_provider="mock",
                    model_name="mock",
                    interface="langchain",
                )
            ],
            parsing_only=True,
        )

        result = task.evaluate(config)
        assert result.global_eval is not None
        assert len(call_kwargs) == 1
        assert call_kwargs[0]["evaluation_mode"] == "template_and_rubric"
        assert call_kwargs[0]["rubric"] is not None
        assert call_kwargs[0]["template_code"] == template_code

    def test_merge_strategy_override(self, monkeypatch):
        """evaluate() merge_strategy parameter overrides instance default."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        mock_result = self._make_mock_verification_result()
        captured_cached: list[dict[str, Any]] = []

        def mock_run(*args, **kwargs):
            captured_cached.append(kwargs.get("cached_answer_data", {}))
            return mock_result

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )

        from karenina.schemas.config import ModelConfig
        from karenina.schemas.verification import VerificationConfig

        task = TaskEval(task_id="test", merge_strategy="concatenate")
        task.log("text log")
        task.log_trace([Message.assistant("trace log")])
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="quality",
                        description="Quality check",
                        kind="boolean",
                    ),
                ]
            )
        )

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test_parser",
                    model_provider="mock",
                    model_name="mock",
                    interface="langchain",
                )
            ],
            parsing_only=True,
        )

        # With traces_only, text log should be excluded
        result = task.evaluate(config, merge_strategy="traces_only")
        assert result.global_eval is not None
        # The response text should only contain the trace, not the text log
        if captured_cached:
            raw = captured_cached[0].get("raw_llm_response", "")
            assert "text log" not in raw

    def test_no_templates_no_rubrics_raises(self):
        """ValueError raised when neither templates nor rubrics are provided."""
        from karenina.schemas.config import ModelConfig
        from karenina.schemas.verification import VerificationConfig

        task = TaskEval()
        task.log("some output")
        task.add_question({"id": "q1", "question": "test?", "raw_answer": "yes"})

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test_parser",
                    model_provider="mock",
                    model_name="mock",
                    interface="langchain",
                )
            ],
            parsing_only=True,
        )

        with pytest.raises(ValueError, match="Must provide either answer templates"):
            task.evaluate(config)

    def test_agent_metrics_passed_through(self, monkeypatch):
        """Agent metrics from trace Messages are passed to cached_answer_data."""
        from karenina.schemas.entities.rubric import LLMRubricTrait, Rubric

        mock_result = self._make_mock_verification_result()
        captured_cached: list[dict[str, Any]] = []

        def mock_run(*args, **kwargs):
            captured_cached.append(kwargs.get("cached_answer_data", {}))
            return mock_result

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )

        from karenina.schemas.config import ModelConfig
        from karenina.schemas.verification import VerificationConfig

        task = TaskEval(task_id="test")
        # Log trace with tool calls so metrics have something to extract
        task.log_trace(
            [
                Message.assistant(
                    "I'll search",
                    tool_calls=[
                        ToolUseContent(id="c1", name="search", input={"q": "test"}),
                    ],
                ),
                Message.tool_result("c1", "results"),
                Message.assistant("Done"),
            ]
        )
        task.add_rubric(
            Rubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="quality",
                        description="Quality check",
                        kind="boolean",
                    ),
                ]
            )
        )

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test_parser",
                    model_provider="mock",
                    model_name="mock",
                    interface="langchain",
                )
            ],
            parsing_only=True,
        )

        task.evaluate(config)
        assert len(captured_cached) == 1
        agent_metrics = captured_cached[0].get("agent_metrics")
        assert agent_metrics is not None
        assert agent_metrics["iterations"] == 2
        assert agent_metrics["tool_calls"] == 1


# =============================================================================
# Agentic Trait Plumbing Tests
# =============================================================================


@pytest.mark.unit
class TestAgenticTraitPlumbing:
    """Tests that agentic rubric traits are fully plumbed through TaskEval."""

    def _make_mock_verification_result(self):
        """Create a mock VerificationResult for testing."""
        from karenina.schemas.verification import VerificationResult
        from karenina.schemas.verification.model_identity import ModelIdentity
        from karenina.schemas.verification.result_components import VerificationResultMetadata

        metadata = VerificationResultMetadata(
            question_id="test_q",
            template_id="test_t",
            completed_without_errors=True,
            question_text="test question",
            answering=ModelIdentity(interface="mock", model_name="mock"),
            parsing=ModelIdentity(interface="mock", model_name="mock"),
            execution_time=0.1,
            timestamp="2026-01-01T00:00:00",
            result_id="abcd1234abcd1234",
        )
        return VerificationResult(metadata=metadata)

    def test_detect_evaluation_mode_agentic_only(self):
        """Rubric with only agentic_traits is detected as rubric mode."""
        from karenina.benchmark.task_eval.task_eval import EvaluationContext
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric

        rubric = Rubric(
            agentic_traits=[
                AgenticRubricTrait(
                    name="lib_check",
                    description="Check which library was used",
                    kind="boolean",
                    higher_is_better=True,
                ),
            ],
        )
        context = EvaluationContext(
            questions=[],
            logs=[LogEvent(level="info", text="output")],
            merged_rubric=rubric,
        )

        task = TaskEval()
        mode = task._detect_evaluation_mode(context)
        assert mode == "rubric_only"

    def test_merge_rubrics_preserves_agentic_traits(self):
        """Merging rubrics preserves agentic_traits."""
        from karenina.schemas.entities.rubric import AgenticRubricTrait, LLMRubricTrait, Rubric

        rubric1 = Rubric(
            llm_traits=[
                LLMRubricTrait(name="clarity", description="Is it clear?", kind="boolean"),
            ],
            agentic_traits=[
                AgenticRubricTrait(
                    name="lib_check",
                    description="Check library",
                    kind="boolean",
                    higher_is_better=True,
                ),
            ],
        )
        rubric2 = Rubric(
            agentic_traits=[
                AgenticRubricTrait(
                    name="tool_usage",
                    description="Check tool usage",
                    kind="boolean",
                    higher_is_better=True,
                ),
            ],
        )

        task = TaskEval()
        merged = task._merge_rubrics([rubric1, rubric2])

        assert merged is not None
        assert len(merged.agentic_traits) == 2
        agentic_names = {t.name for t in merged.agentic_traits}
        assert agentic_names == {"lib_check", "tool_usage"}

    def test_merge_rubrics_detects_duplicate_agentic_names(self):
        """Merging rubrics with duplicate agentic trait names raises ValueError."""
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric

        rubric1 = Rubric(
            agentic_traits=[
                AgenticRubricTrait(
                    name="same_name",
                    description="First",
                    kind="boolean",
                    higher_is_better=True,
                ),
            ],
        )
        rubric2 = Rubric(
            agentic_traits=[
                AgenticRubricTrait(
                    name="same_name",
                    description="Second",
                    kind="boolean",
                    higher_is_better=True,
                ),
            ],
        )

        task = TaskEval()
        with pytest.raises(ValueError, match="Duplicate rubric trait names"):
            task._merge_rubrics([rubric1, rubric2])

    def test_detect_evaluation_mode_template_and_agentic(self):
        """Template + agentic-only rubric detects as template_and_rubric."""
        from karenina.benchmark.task_eval.task_eval import EvaluationContext
        from karenina.schemas.entities.rubric import AgenticRubricTrait, Rubric

        rubric = Rubric(
            agentic_traits=[
                AgenticRubricTrait(
                    name="lib_check",
                    description="Check library",
                    kind="boolean",
                    higher_is_better=True,
                ),
            ],
        )

        template_code = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    value: int = Field(description="The answer")
    def verify(self) -> bool:
        return self.value == 4
"""
        context = EvaluationContext(
            questions=[{"id": "q1", "question": "test", "answer_template": template_code}],
            logs=[LogEvent(level="info", text="output")],
            merged_rubric=rubric,
        )

        task = TaskEval()
        mode = task._detect_evaluation_mode(context)
        assert mode == "template_and_rubric"


# =============================================================================
# Dynamic Rubric Plumbing Tests
# =============================================================================


@pytest.mark.unit
class TestDynamicRubricPlumbing:
    """Tests that DynamicRubric is fully plumbed through TaskEval."""

    def _make_mock_verification_result(self):
        """Create a mock VerificationResult for testing."""
        from karenina.schemas.verification import VerificationResult
        from karenina.schemas.verification.model_identity import ModelIdentity
        from karenina.schemas.verification.result_components import VerificationResultMetadata

        metadata = VerificationResultMetadata(
            question_id="test_q",
            template_id="test_t",
            completed_without_errors=True,
            question_text="test question",
            answering=ModelIdentity(interface="mock", model_name="mock"),
            parsing=ModelIdentity(interface="mock", model_name="mock"),
            execution_time=0.1,
            timestamp="2026-01-01T00:00:00",
            result_id="abcd1234abcd1234",
        )
        return VerificationResult(metadata=metadata)

    def test_add_dynamic_rubric_global(self):
        """add_dynamic_rubric stores in global_dynamic_rubrics."""
        from karenina.schemas.entities.rubric import DynamicRubric, LLMRubricTrait

        task = TaskEval()
        dr = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="safety",
                    summary="safety check",
                    description="Is the response safe?",
                    kind="boolean",
                ),
            ],
        )
        task.add_dynamic_rubric(dr)
        assert len(task.global_dynamic_rubrics) == 1

    def test_add_dynamic_rubric_step(self):
        """add_dynamic_rubric with step_id stores in step_dynamic_rubrics."""
        from karenina.schemas.entities.rubric import DynamicRubric, LLMRubricTrait

        task = TaskEval()
        dr = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="safety",
                    summary="safety check",
                    description="Is the response safe?",
                    kind="boolean",
                ),
            ],
        )
        task.add_dynamic_rubric(dr, step_id="s1")
        assert "s1" in task.step_dynamic_rubrics
        assert len(task.step_dynamic_rubrics["s1"]) == 1

    def test_dynamic_rubric_passed_to_runner(self, monkeypatch):
        """DynamicRubric is passed through to run_single_model_verification."""
        from karenina.schemas.entities.rubric import DynamicRubric, LLMRubricTrait

        mock_result = self._make_mock_verification_result()
        call_kwargs: list[dict[str, Any]] = []

        def mock_run(*args, **kwargs):
            call_kwargs.append(kwargs)
            return mock_result

        monkeypatch.setattr(
            "karenina.benchmark.verification.runner.run_single_model_verification",
            mock_run,
        )

        from karenina.schemas.config import ModelConfig
        from karenina.schemas.verification import VerificationConfig

        task = TaskEval(task_id="test")
        task.log("Some response about drug interactions")
        task.add_rubric(
            __import__("karenina.schemas.entities.rubric", fromlist=["Rubric"]).Rubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="quality",
                        description="Is it good?",
                        kind="boolean",
                    ),
                ],
            )
        )
        task.add_dynamic_rubric(
            DynamicRubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="interaction_safety",
                        summary="drug interaction warnings",
                        description="Answer True if drug interaction warnings present.",
                        kind="boolean",
                    ),
                ],
            )
        )

        config = VerificationConfig(
            parsing_models=[
                ModelConfig(
                    id="test_parser",
                    model_provider="mock",
                    model_name="mock",
                    interface="langchain",
                )
            ],
            parsing_only=True,
        )

        task.evaluate(config)
        assert len(call_kwargs) == 1
        assert call_kwargs[0]["dynamic_rubric"] is not None
        assert len(call_kwargs[0]["dynamic_rubric"].llm_traits) == 1
        assert call_kwargs[0]["dynamic_rubric"].llm_traits[0].name == "interaction_safety"

    def test_dynamic_rubric_triggers_rubric_mode(self):
        """DynamicRubric alone (no static rubric) triggers rubric evaluation mode."""
        from karenina.schemas.entities.rubric import DynamicRubric, LLMRubricTrait

        task = TaskEval()
        dr = DynamicRubric(
            llm_traits=[
                LLMRubricTrait(
                    name="safety",
                    summary="safety check",
                    description="Is it safe?",
                    kind="boolean",
                ),
            ],
        )
        task.add_dynamic_rubric(dr)

        context = task._get_evaluation_context(step_id=None)
        mode = task._detect_evaluation_mode(context)
        assert mode == "rubric_only"


# =============================================================================
# Question Normalization Tests
# =============================================================================


@pytest.mark.unit
class TestQuestionNormalization:
    """Tests that _normalize_question extracts all relevant fields from Question objects."""

    def test_extracts_question_rubric(self):
        """_normalize_question extracts question_rubric from Question objects."""
        from karenina.schemas.entities import Question

        q = Question(
            question="What is BCL2?",
            raw_answer="BCL2 is a protein.",
            question_rubric={"llm_traits": [{"name": "tone", "description": "Formal?", "kind": "boolean"}]},
        )

        task = TaskEval()
        normalized = task._normalize_question(q)
        assert "question_rubric" in normalized
        assert normalized["question_rubric"] is not None

    def test_extracts_question_dynamic_rubric(self):
        """_normalize_question extracts question_dynamic_rubric from Question objects."""
        from karenina.schemas.entities import Question

        q = Question(
            question="What is BCL2?",
            raw_answer="BCL2 is a protein.",
            question_dynamic_rubric={
                "llm_traits": [
                    {
                        "name": "interaction_safety",
                        "summary": "drug interactions",
                        "description": "Mentions interactions?",
                        "kind": "boolean",
                    }
                ]
            },
        )

        task = TaskEval()
        normalized = task._normalize_question(q)
        assert "question_dynamic_rubric" in normalized
        assert normalized["question_dynamic_rubric"] is not None


# =============================================================================
# Import Smoke Test
# =============================================================================


@pytest.mark.unit
class TestImports:
    """Verify public API imports work."""

    def test_public_imports(self):
        """All public symbols importable from karenina.benchmark.task_eval."""
        from karenina.benchmark.task_eval import LogEvent, TaskEval

        assert TaskEval is not None
        assert LogEvent is not None
        assert StepEval is not None
        assert TaskEvalResult is not None

    def test_instantiate_and_log(self):
        """Basic smoke test: instantiate, log, log_trace."""
        task = TaskEval(task_id="smoke")
        task.log("text entry")
        task.log_trace("string trace")
        task.log_trace([Message.assistant("message trace")])

        assert len(task.global_logs) == 3
        assert task.global_logs[0].text == "text entry"
        assert task.global_logs[1].trace_messages is not None
        assert task.global_logs[2].trace_messages is not None
