"""Prompt builder for dynamic rubric concept presence checking."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from karenina.schemas.entities.rubric import (
        AgenticRubricTrait,
        CallableRubricTrait,
        LLMRubricTrait,
        MetricRubricTrait,
        RegexRubricTrait,
    )

    AnyTrait: TypeAlias = (
        LLMRubricTrait | RegexRubricTrait | CallableRubricTrait | MetricRubricTrait | AgenticRubricTrait
    )

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a concept presence detector. Your task is to determine which \
evaluation concepts are meaningfully present in a response.

## What "present" means

A concept is PRESENT if the response engages with it substantively: \
discusses it, argues about it, provides information related to it, or \
takes a position on it.

A concept is ABSENT if it is not mentioned at all, or is mentioned only \
in passing without substantive engagement (e.g., a single word in a \
list, a parenthetical aside).

## Decision rules

- When a concept references multiple aspects (e.g., "cites sources and \
explains their relevance"), consider it present if ANY aspect appears \
substantively.
- Err toward PRESENT when uncertain. False negatives (skipping a \
relevant evaluation) are worse than false positives (running an \
evaluation on a marginally present concept).
- Evaluate each concept independently.\
"""


def _resolve_concept_text(trait: AnyTrait) -> str:
    """Return summary if set, else description.

    Args:
        trait: Any rubric trait with summary and description fields.

    Returns:
        The summary string if not None, otherwise the description string.
    """
    if trait.summary is not None:
        return trait.summary
    # DynamicRubric.validate_concept_text guarantees at least one is set
    assert trait.description is not None
    return trait.description


class PresenceCheckPromptBuilder:
    """Builds prompts for dynamic rubric concept presence checking.

    Constructs a system prompt explaining presence detection rules, a user
    prompt listing trait concepts alongside the response text, and an example
    JSON showing the expected output structure.

    Example:
        builder = PresenceCheckPromptBuilder()
        system = builder.build_system_prompt()
        user = builder.build_user_prompt(traits, response_text)
        example = builder.build_example_json(traits)
    """

    def build_system_prompt(self) -> str:
        """Build system prompt for concept presence detection.

        Returns:
            System prompt explaining presence detection rules.
        """
        return _SYSTEM_PROMPT

    def build_user_prompt(self, traits: list[AnyTrait], response_text: str) -> str:
        """Build user prompt listing concepts and the response to analyze.

        Args:
            traits: List of rubric traits whose concepts to check.
            response_text: The LLM response to analyze for concept presence.

        Returns:
            Formatted user prompt with concept list and response text.
        """
        concept_lines = []
        for trait in traits:
            text = _resolve_concept_text(trait)
            concept_lines.append(f"- **{trait.name}**: {text}")
        concept_list = "\n".join(concept_lines)
        return f"## Concepts to check\n\n{concept_list}\n\n## Response to analyze\n\n{response_text}"

    def build_example_json(self, traits: list[AnyTrait]) -> str:
        """Build example JSON showing the expected output structure.

        Used by evaluator callers to pass as instruction_context for adapter
        instructions to include in format-specific prompt sections.

        Args:
            traits: List of rubric traits to build the example for.

        Returns:
            JSON string with alternating present values for illustration.
        """
        items = []
        for i, trait in enumerate(traits):
            items.append({"trait_name": trait.name, "present": i % 2 == 0})
        return json.dumps({"results": items}, indent=2)
