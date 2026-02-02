"""Prompts for Claude Agent SDK adapter operations.

Note: Parser prompt construction is now centralized in
benchmark/verification/prompts/parsing/instructions.py (TemplatePromptBuilder).
Adapter-specific modifications are applied via AdapterInstructionRegistry.

Adapter instruction registration is triggered by ``registration.py``, not here,
to avoid side-effects when importing prompts for non-registration purposes.
"""

__all__: list[str] = []
