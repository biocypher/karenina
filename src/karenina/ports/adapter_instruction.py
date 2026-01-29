"""Adapter instruction protocol and registry.

Adapter instructions allow each LLM backend to inject adapter-specific
modifications into prompts. For example, a LangChain adapter might add
JSON schema formatting instructions to parsing prompts.

The registry maps (interface_name, PromptTask value) pairs to lists of
instruction factories. Adapters register instructions in their registration.py
files. An empty result from the registry means no adapter modifications
are needed for that combination.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class AdapterInstruction(Protocol):
    """Protocol for adapter-specific prompt modifications.

    An adapter instruction takes system and user text and returns
    modified versions. Instructions are applied in registration order.
    """

    def apply(self, system_text: str, user_text: str) -> tuple[str, str]:  # noqa: vulture
        """Apply this instruction to the prompt texts.

        Args:
            system_text: The current system prompt text.
            user_text: The current user prompt text.

        Returns:
            A tuple of (modified_system_text, modified_user_text).
        """
        ...


InstructionFactory = Callable[..., AdapterInstruction]
"""A callable that produces an AdapterInstruction instance.

Factories allow deferred construction of instructions with context-specific
parameters (e.g., JSON schema, model capabilities).
"""


class AdapterInstructionRegistry:
    """Maps (interface, task) pairs to lists of instruction factories.

    Adapters register instruction factories in their registration.py files.
    The PromptAssembler queries the registry to apply adapter-specific
    modifications during prompt assembly.

    This is a class-level registry — all adapters share a single global
    registry, keyed by (interface_name, task_value) tuples.
    """

    _registry: dict[tuple[str, str], list[InstructionFactory]] = {}

    @classmethod
    def register(cls, interface: str, task: str, factory: InstructionFactory) -> None:
        """Register an instruction factory for an (interface, task) pair.

        Args:
            interface: The adapter interface name (e.g., "langchain", "claude_tool").
            task: The PromptTask value string (e.g., "parsing", "generation").
            factory: A callable that produces an AdapterInstruction instance.
        """
        key = (interface, task)
        if key not in cls._registry:
            cls._registry[key] = []
        cls._registry[key].append(factory)

    @classmethod
    def get_instructions(cls, interface: str, task: str) -> list[InstructionFactory]:
        """Retrieve instruction factories for an (interface, task) pair.

        Args:
            interface: The adapter interface name.
            task: The PromptTask value string.

        Returns:
            List of instruction factories, or empty list if none registered.
        """
        return cls._registry.get((interface, task), [])

    @classmethod
    def clear(cls) -> None:
        """Clear all registered instructions. Primarily for testing."""
        cls._registry.clear()
