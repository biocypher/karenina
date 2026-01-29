"""Prompt assembler combining task, adapter, and user instructions.

The PromptAssembler implements the tri-section prompt assembly pattern:
1. Task instructions — the base system/user text for a specific pipeline call
2. Adapter instructions — text appended based on the LLM backend
3. User instructions — optional per-task-type text from PromptConfig

All instruction sources are treated uniformly as append operations:
adapter text is appended first, then user instructions. The assembler
builds a list[Message] respecting adapter capabilities (e.g., prepending
system text to user text when the adapter does not support system prompts).
"""

from __future__ import annotations

from dataclasses import dataclass

from karenina.benchmark.verification.prompts.task_types import PromptTask
from karenina.ports.adapter_instruction import AdapterInstructionRegistry
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.messages import Message


@dataclass
class PromptAssembler:
    """Combines task, adapter, and user instructions into prompt messages.

    Attributes:
        task: The PromptTask identifying which pipeline call this is for.
        interface: The adapter interface name (e.g., "langchain", "claude_tool").
        capabilities: The adapter's declared capabilities.
    """

    task: PromptTask
    interface: str
    capabilities: PortCapabilities

    def assemble(
        self,
        system_text: str,
        user_text: str,
        user_instructions: str | None = None,
        instruction_context: dict[str, object] | None = None,
    ) -> list[Message]:
        """Assemble prompt messages from the three instruction sections.

        Args:
            system_text: Base system prompt text (task instructions).
            user_text: Base user prompt text.
            user_instructions: Optional user-provided instructions to append
                to the system text.
            instruction_context: Optional context dict passed to adapter
                instruction factories (e.g., JSON schema, model info).

        Returns:
            List of Message objects ready for LLM invocation.
        """
        system_text, user_text = self._apply_instructions(
            system_text, user_text, user_instructions, instruction_context
        )

        if not self.capabilities.supports_system_prompt:
            # Prepend system text to user text when system prompts not supported
            combined = f"{system_text}\n\n{user_text}" if system_text else user_text
            return [Message.user(combined)]

        messages: list[Message] = []
        if system_text:
            messages.append(Message.system(system_text))
        messages.append(Message.user(user_text))
        return messages

    def assemble_text(
        self,
        system_text: str,
        user_text: str,
        user_instructions: str | None = None,
        instruction_context: dict[str, object] | None = None,
    ) -> tuple[str, str]:
        """Assemble prompt text without building Message objects.

        Same tri-section logic as assemble() but returns raw strings.
        Used by multi-stage flows (e.g., deep judgment) that need text
        for intermediate processing before final message construction.

        Args:
            system_text: Base system prompt text (task instructions).
            user_text: Base user prompt text.
            user_instructions: Optional user-provided instructions to append
                to the system text.
            instruction_context: Optional context dict passed to adapter
                instruction factories.

        Returns:
            Tuple of (system_text, user_text) after all instructions applied.
        """
        return self._apply_instructions(system_text, user_text, user_instructions, instruction_context)

    def _apply_instructions(
        self,
        system_text: str,
        user_text: str,
        user_instructions: str | None,
        instruction_context: dict[str, object] | None,
    ) -> tuple[str, str]:
        """Append adapter and user instructions to the prompt texts.

        Order of operations:
        1. Append adapter instruction text (system_addition, user_addition)
        2. Append user instructions to system text

        All instruction sources are treated uniformly as append operations.

        Args:
            system_text: Base system prompt text.
            user_text: Base user prompt text.
            user_instructions: Optional user instructions to append.
            instruction_context: Optional context for instruction factories.

        Returns:
            Tuple of (modified_system_text, modified_user_text).
        """
        # 1. Append adapter instructions
        factories = AdapterInstructionRegistry.get_instructions(self.interface, self.task.value)
        for factory in factories:
            instruction = factory(**(instruction_context or {}))
            if instruction.system_addition:
                system_text = (
                    f"{system_text}\n\n{instruction.system_addition}" if system_text else instruction.system_addition
                )
            if instruction.user_addition:
                user_text = f"{user_text}\n\n{instruction.user_addition}" if user_text else instruction.user_addition

        # 2. Append user instructions to system text
        if user_instructions:
            system_text = f"{system_text}\n\n{user_instructions}" if system_text else user_instructions

        return system_text, user_text
