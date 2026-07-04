"""Unit tests for PlaceholderRetryAutoFailStage.

The stage detects LangChain ModelRetryMiddleware exhaustion placeholders that
would otherwise be parsed as content failures, and reclassifies them as
connection failures via the autofail rule path. See daily note
``2026-04-25-mcp-connection-error-misclassification.md`` for the original bug.
"""

import pytest

from karenina.benchmark.verification.failure_classifier import classify_failure
from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    VerificationContext,
)
from karenina.benchmark.verification.stages.pipeline.placeholder_retry_autofail import (
    PlaceholderRetryAutoFailStage,
)
from karenina.ports.messages import Message
from karenina.schemas.config import ModelConfig
from karenina.schemas.results.failure import FailureCategory, FailureGroup
from karenina.utils.errors import ErrorCategory


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        id="test-model",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
    )


@pytest.fixture
def context(model_config: ModelConfig) -> VerificationContext:
    return VerificationContext(
        question_id="q-1",
        template_id="t-1",
        question_text="Q?",
        template_code="class Answer(BaseAnswer): ...",
        answering_model=model_config,
        parsing_model=model_config,
    )


@pytest.fixture
def stage() -> PlaceholderRetryAutoFailStage:
    return PlaceholderRetryAutoFailStage()


@pytest.mark.unit
class TestPlaceholderRetryAutoFail:
    def test_placeholder_message_triggers_reclassification(
        self, stage: PlaceholderRetryAutoFailStage, context: VerificationContext
    ) -> None:
        placeholder = Message.assistant("Model call failed after 3 attempts with APIConnectionError: Connection error.")
        context.set_artifact(ArtifactKeys.TRACE_MESSAGES, [placeholder])

        stage.execute(context)

        assert context.error is not None
        assert context.error_category is ErrorCategory.CONNECTION
        assert context.error_stage == "generate_answer"
        assert context.get_result_field(ArtifactKeys.FAILED_STAGE) == "PlaceholderRetryAutoFail"
        assert context.get_artifact(ArtifactKeys.VERIFY_RESULT) is False
        assert context.get_result_field(ArtifactKeys.VERIFY_RESULT) is False

    def test_placeholder_classifies_as_connection_failure(
        self, stage: PlaceholderRetryAutoFailStage, context: VerificationContext
    ) -> None:
        placeholder = Message.assistant("Model call failed after 3 attempts with APIConnectionError: Connection error.")
        context.set_artifact(ArtifactKeys.TRACE_MESSAGES, [placeholder])

        stage.execute(context)
        failure = classify_failure(context)

        assert failure is not None
        assert failure.category is FailureCategory.CONNECTION
        assert failure.group is FailureGroup.RETRY_EXHAUSTED
        assert failure.stage == "PlaceholderRetryAutoFail"

    def test_placeholder_dict_form_also_triggers(
        self, stage: PlaceholderRetryAutoFailStage, context: VerificationContext
    ) -> None:
        # Post-serialization (e.g. when loaded from a stored result), trace messages
        # are dicts shaped like Message.to_dict(). The detector must handle both.
        placeholder = {
            "role": "assistant",
            "content": "Model call failed after 3 attempts with APIConnectionError: Connection error.",
            "block_index": 0,
        }
        context.set_artifact(ArtifactKeys.TRACE_MESSAGES, [placeholder])

        stage.execute(context)

        assert context.error_category is ErrorCategory.CONNECTION
        assert context.get_result_field(ArtifactKeys.FAILED_STAGE) == "PlaceholderRetryAutoFail"

    def test_normal_assistant_message_does_not_trigger(
        self, stage: PlaceholderRetryAutoFailStage, context: VerificationContext
    ) -> None:
        normal = Message.assistant("BCL-2 is the drug target. The mechanism is well-studied.")
        context.set_artifact(ArtifactKeys.TRACE_MESSAGES, [normal])

        stage.execute(context)

        assert context.error is None
        assert context.error_category is None
        assert context.get_result_field(ArtifactKeys.FAILED_STAGE) is None

    def test_multi_message_trace_with_placeholder_last_triggers(
        self, stage: PlaceholderRetryAutoFailStage, context: VerificationContext
    ) -> None:
        # Mid-trace failure: model successfully ran tools, then the final
        # synthesis call exhausted retries. The placeholder lands as the last
        # AIMessage. The model never produced a real final answer.
        msgs = [
            Message.user("What proteins interact with BRCA1?"),
            Message.assistant("Let me query Open Targets."),
            Message.assistant("Model call failed after 3 attempts with APIConnectionError: Connection error."),
        ]
        context.set_artifact(ArtifactKeys.TRACE_MESSAGES, msgs)

        stage.execute(context)

        assert context.error_category is ErrorCategory.CONNECTION
        assert context.get_result_field(ArtifactKeys.FAILED_STAGE) == "PlaceholderRetryAutoFail"

    def test_placeholder_not_at_end_does_not_trigger(
        self, stage: PlaceholderRetryAutoFailStage, context: VerificationContext
    ) -> None:
        # If the placeholder is somewhere mid-trace but the agent recovered and
        # produced a real final answer, this is not a terminal failure. (This
        # path is unlikely in practice with on_failure="continue" because the
        # placeholder usually ends the loop, but the detector must not fire on
        # it just in case.)
        msgs = [
            Message.assistant("Model call failed after 3 attempts with APIConnectionError: ..."),
            Message.assistant("Recovered. The answer is BCL-2."),
        ]
        context.set_artifact(ArtifactKeys.TRACE_MESSAGES, msgs)

        stage.execute(context)

        assert context.error is None
        assert context.get_result_field(ArtifactKeys.FAILED_STAGE) is None

    def test_empty_trace_does_not_trigger(
        self, stage: PlaceholderRetryAutoFailStage, context: VerificationContext
    ) -> None:
        context.set_artifact(ArtifactKeys.TRACE_MESSAGES, [])

        stage.execute(context)

        assert context.error is None
        assert context.get_result_field(ArtifactKeys.FAILED_STAGE) is None

    def test_should_run_returns_false_when_trace_messages_missing(
        self, stage: PlaceholderRetryAutoFailStage, context: VerificationContext
    ) -> None:
        # No trace_messages artifact set
        assert stage.should_run(context) is False

    def test_should_run_returns_false_when_context_already_has_error(
        self, stage: PlaceholderRetryAutoFailStage, context: VerificationContext
    ) -> None:
        context.set_artifact(ArtifactKeys.TRACE_MESSAGES, [Message.assistant("x")])
        context.mark_error("prior failure", category=ErrorCategory.PERMANENT)

        assert stage.should_run(context) is False
