"""Tests for PromptAssembler."""

import pytest

from karenina.benchmark.verification.prompts.assembler import PromptAssembler
from karenina.benchmark.verification.prompts.task_types import PromptTask
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.messages import Role


@pytest.mark.unit
class TestPromptAssemblerNoSystemPrompt:
    """Test PromptAssembler with supports_system_prompt=False."""

    def _make_assembler(self, supports_system: bool) -> PromptAssembler:
        """Create a PromptAssembler with the given system prompt capability.

        Uses a fake interface name so no adapter instructions are registered,
        keeping the test isolated to the system-prompt fallback logic.

        Args:
            supports_system: Whether to enable system prompt support.

        Returns:
            Configured PromptAssembler instance.
        """
        return PromptAssembler(
            task=PromptTask.GENERATION,
            interface="__test_no_adapter__",
            capabilities=PortCapabilities(supports_system_prompt=supports_system),
        )

    def test_system_text_prepended_to_user_when_no_system_support(self):
        """Verify system text is prepended to user text when system prompts unsupported."""
        assembler = self._make_assembler(supports_system=False)

        messages = assembler.assemble(
            system_text="You are a helpful assistant.",
            user_text="What is 2+2?",
        )

        assert len(messages) == 1
        msg = messages[0]
        assert msg.role == Role.USER
        assert "You are a helpful assistant." in msg.text
        assert "What is 2+2?" in msg.text
        # System text comes before user text
        assert msg.text.index("You are a helpful assistant.") < msg.text.index("What is 2+2?")

    def test_no_system_message_present_when_no_system_support(self):
        """Verify no message with Role.SYSTEM exists in the output."""
        assembler = self._make_assembler(supports_system=False)

        messages = assembler.assemble(
            system_text="System instructions here.",
            user_text="User query here.",
        )

        system_messages = [m for m in messages if m.role == Role.SYSTEM]
        assert system_messages == []

    def test_empty_system_text_returns_user_only(self):
        """When system text is empty, user text is returned unchanged."""
        assembler = self._make_assembler(supports_system=False)

        messages = assembler.assemble(
            system_text="",
            user_text="Just a user message.",
        )

        assert len(messages) == 1
        assert messages[0].role == Role.USER
        assert messages[0].text == "Just a user message."

    def test_system_support_true_produces_separate_messages(self):
        """Contrast: with system support, system and user are separate messages."""
        assembler = self._make_assembler(supports_system=True)

        messages = assembler.assemble(
            system_text="You are a helpful assistant.",
            user_text="What is 2+2?",
        )

        assert len(messages) == 2
        assert messages[0].role == Role.SYSTEM
        assert messages[0].text == "You are a helpful assistant."
        assert messages[1].role == Role.USER
        assert messages[1].text == "What is 2+2?"
