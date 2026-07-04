"""Tests for PromptConfig.generation wiring in GenerateAnswerStage."""

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.benchmark.verification.stages.pipeline.generate_answer import GenerateAnswerStage
from karenina.ports import Message
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.prompt_config import PromptConfig


def _make_context(
    system_prompt: str | None = "You are helpful.",
    prompt_config: PromptConfig | None = None,
) -> VerificationContext:
    """Create a minimal VerificationContext for testing."""
    model = ModelConfig(
        id="test",
        model_name="test",
        model_provider="openai",
        system_prompt=system_prompt,
    )
    ctx = VerificationContext(
        question_id="q1",
        template_id="t1",
        question_text="What is 2+2?",
        template_code="class Answer(BaseAnswer): value: str",
        answering_model=model,
        parsing_model=model,
        prompt_config=prompt_config,
    )
    return ctx


def _capture_adapter_messages(context: VerificationContext) -> list[Message]:
    """Run GenerateAnswerStage.execute() with mocked adapter, return captured messages."""
    stage = GenerateAnswerStage()
    captured: list[Message] = []

    from karenina.ports.capabilities import PortCapabilities

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "4"
    mock_response.usage = None
    mock_llm.invoke.side_effect = lambda msgs: (captured.extend(msgs), mock_response)[1]
    mock_llm.capabilities = PortCapabilities(supports_streaming=False)

    with patch("karenina.benchmark.verification.stages.pipeline.generate_answer.get_llm", return_value=mock_llm):
        stage.execute(context)

    return captured


@pytest.mark.unit
class TestPromptConfigGeneration:
    """Test that PromptConfig.generation is injected into the system message."""

    def test_generation_instructions_appended_to_system_prompt(self) -> None:
        """PromptConfig.generation should appear in the system message."""
        ctx = _make_context(
            system_prompt="You are helpful.",
            prompt_config=PromptConfig(generation="Focus on clinical accuracy."),
        )
        messages = _capture_adapter_messages(ctx)
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert "You are helpful." in system_msgs[0].text
        assert "Focus on clinical accuracy." in system_msgs[0].text

    def test_no_prompt_config_uses_system_prompt_only(self) -> None:
        """Without PromptConfig, only ModelConfig.system_prompt is used."""
        ctx = _make_context(system_prompt="You are helpful.", prompt_config=None)
        messages = _capture_adapter_messages(ctx)
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].text == "You are helpful."

    def test_generation_none_does_not_change_system_prompt(self) -> None:
        """PromptConfig with generation=None should not modify the system message."""
        ctx = _make_context(
            system_prompt="You are helpful.",
            prompt_config=PromptConfig(generation=None),
        )
        messages = _capture_adapter_messages(ctx)
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].text == "You are helpful."

    def test_no_system_prompt_generation_only(self) -> None:
        """When ModelConfig.system_prompt is None, only generation instructions appear."""
        ctx = _make_context(
            system_prompt=None,
            prompt_config=PromptConfig(generation="Be concise."),
        )
        messages = _capture_adapter_messages(ctx)
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].text == "Be concise."
