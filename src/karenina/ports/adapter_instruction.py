"""Adapter instruction protocol and registry.

Adapter instructions allow each LLM backend to inject adapter-specific
text into prompts. For example, a LangChain adapter might add JSON schema
formatting instructions, while a Claude adapter might add extraction directives.

Instructions are append-only: each adapter provides text to append to the
system and/or user prompts. The PromptAssembler appends adapter text first,
then user instructions, treating all instruction sources uniformly.

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
    """Protocol for adapter-specific prompt additions.

    An adapter instruction provides text to append to the system and/or
    user prompts. Instructions are appended in registration order.
    """

    @property
    def system_addition(self) -> str:  # noqa: vulture
        """Text to append to the system prompt (empty string for no addition)."""
        ...

    @property
    def user_addition(self) -> str:  # noqa: vulture
        """Text to append to the user prompt (empty string for no addition)."""
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
