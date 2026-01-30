"""Port capabilities declaration.

Declares what prompt features an adapter supports, used by PromptAssembler
to decide message formatting (e.g., prepend system to user text if system
prompts are not supported by the adapter).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PortCapabilities:
    """Capabilities supported by an LLM/Parser adapter.

    Attributes:
        supports_system_prompt: Whether the adapter supports separate system messages.
            If False, system text is prepended to user text.
        supports_structured_output: Whether the adapter supports structured output
            (e.g., JSON schema enforcement).
    """

    supports_system_prompt: bool = True
    supports_structured_output: bool = False
