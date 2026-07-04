"""Tests for PresenceCheckPromptBuilder and RUBRIC_DYNAMIC_PRESENCE_CHECK enum value.

Covers:
- PromptTask.RUBRIC_DYNAMIC_PRESENCE_CHECK enum value exists
- System prompt content
- User prompt lists concepts with correct formatting
- Summary is preferred over description for concept text
- Falls back to description when summary is None
- Example JSON format matches expected structure
"""

import json

import pytest

from karenina.benchmark.verification.prompts.task_types import PromptTask
from karenina.schemas.entities import LLMRubricTrait, RegexRubricTrait

# =============================================================================
# Helpers
# =============================================================================


def _make_llm_trait(
    name: str,
    summary: str | None = None,
    description: str | None = None,
) -> LLMRubricTrait:
    """Create a minimal LLMRubricTrait for testing."""
    return LLMRubricTrait(
        name=name,
        kind="boolean",
        higher_is_better=True,
        summary=summary,
        description=description,
    )


def _make_regex_trait(
    name: str,
    summary: str | None = None,
    description: str | None = None,
) -> RegexRubricTrait:
    """Create a minimal RegexRubricTrait for testing."""
    return RegexRubricTrait(
        name=name,
        pattern=r"\btest\b",
        higher_is_better=True,
        summary=summary,
        description=description,
    )


# =============================================================================
# PromptTask Enum Tests
# =============================================================================


@pytest.mark.unit
class TestPromptTaskEnum:
    """Tests for PromptTask.RUBRIC_DYNAMIC_PRESENCE_CHECK."""

    def test_enum_value_exists(self) -> None:
        """RUBRIC_DYNAMIC_PRESENCE_CHECK is a valid PromptTask member."""
        assert hasattr(PromptTask, "RUBRIC_DYNAMIC_PRESENCE_CHECK")

    def test_enum_value_string(self) -> None:
        """RUBRIC_DYNAMIC_PRESENCE_CHECK has the expected string value."""
        assert PromptTask.RUBRIC_DYNAMIC_PRESENCE_CHECK.value == "rubric_dynamic_presence_check"

    def test_enum_lookup_by_value(self) -> None:
        """Can look up the enum member by its string value."""
        member = PromptTask("rubric_dynamic_presence_check")
        assert member is PromptTask.RUBRIC_DYNAMIC_PRESENCE_CHECK


# =============================================================================
# PresenceCheckPromptBuilder Tests
# =============================================================================


@pytest.mark.unit
class TestPresenceCheckSystemPrompt:
    """Tests for PresenceCheckPromptBuilder.build_system_prompt."""

    def test_returns_string(self) -> None:
        """build_system_prompt returns a non-empty string."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        result = builder.build_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mentions_presence_detection(self) -> None:
        """System prompt explains the concept of presence detection."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        result = builder.build_system_prompt()
        assert "present" in result.lower()
        assert "absent" in result.lower()

    def test_mentions_decision_rules(self) -> None:
        """System prompt includes decision rules."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        result = builder.build_system_prompt()
        assert "PRESENT" in result
        assert "ABSENT" in result


@pytest.mark.unit
class TestPresenceCheckUserPrompt:
    """Tests for PresenceCheckPromptBuilder.build_user_prompt."""

    def test_lists_concept_names(self) -> None:
        """User prompt includes all trait names as concept labels."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        traits = [
            _make_llm_trait("safety", summary="Response safety"),
            _make_llm_trait("clarity", summary="Response clarity"),
        ]
        result = builder.build_user_prompt(traits, "Some response text.")
        assert "**safety**" in result
        assert "**clarity**" in result

    def test_includes_response_text(self) -> None:
        """User prompt includes the response text to analyze."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        traits = [_make_llm_trait("safety", summary="Safety check")]
        result = builder.build_user_prompt(traits, "The drug targets BCL2.")
        assert "The drug targets BCL2." in result

    def test_prefers_summary_over_description(self) -> None:
        """When both summary and description are set, concept text uses summary."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        traits = [
            _make_llm_trait(
                "safety",
                summary="Short safety label",
                description="Verbose safety description with many details",
            ),
        ]
        result = builder.build_user_prompt(traits, "Response text.")
        assert "Short safety label" in result
        assert "Verbose safety description" not in result

    def test_falls_back_to_description(self) -> None:
        """When summary is None, concept text falls back to description."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        trait = _make_llm_trait("safety", description="Detailed safety check")
        result = builder.build_user_prompt([trait], "Response text.")
        assert "Detailed safety check" in result

    def test_mixed_trait_types(self) -> None:
        """User prompt works with a mix of LLM and Regex traits."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        traits = [
            _make_llm_trait("safety", summary="Safety assessment"),
            _make_regex_trait("citation", summary="Citation format"),
        ]
        result = builder.build_user_prompt(traits, "Some response.")
        assert "**safety**" in result
        assert "**citation**" in result
        assert "Safety assessment" in result
        assert "Citation format" in result

    def test_has_concepts_section(self) -> None:
        """User prompt has a 'Concepts to check' section header."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        traits = [_make_llm_trait("safety", summary="Safety")]
        result = builder.build_user_prompt(traits, "Response.")
        assert "## Concepts to check" in result

    def test_has_response_section(self) -> None:
        """User prompt has a 'Response to analyze' section header."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        traits = [_make_llm_trait("safety", summary="Safety")]
        result = builder.build_user_prompt(traits, "My response.")
        assert "## Response to analyze" in result


@pytest.mark.unit
class TestPresenceCheckExampleJson:
    """Tests for PresenceCheckPromptBuilder.build_example_json."""

    def test_valid_json(self) -> None:
        """build_example_json returns valid JSON."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        traits = [
            _make_llm_trait("safety", summary="Safety"),
            _make_llm_trait("clarity", summary="Clarity"),
        ]
        result = builder.build_example_json(traits)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_has_results_key(self) -> None:
        """Example JSON has a 'results' top-level key."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        traits = [_make_llm_trait("safety", summary="Safety")]
        parsed = json.loads(builder.build_example_json(traits))
        assert "results" in parsed

    def test_items_have_trait_name_and_present(self) -> None:
        """Each item in results has trait_name and present keys."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        traits = [
            _make_llm_trait("safety", summary="Safety"),
            _make_llm_trait("clarity", summary="Clarity"),
        ]
        parsed = json.loads(builder.build_example_json(traits))
        for item in parsed["results"]:
            assert "trait_name" in item
            assert "present" in item
            assert isinstance(item["present"], bool)

    def test_trait_names_match_input(self) -> None:
        """Example JSON trait names match the provided traits."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        traits = [
            _make_llm_trait("safety", summary="Safety"),
            _make_regex_trait("citation", summary="Citations"),
        ]
        parsed = json.loads(builder.build_example_json(traits))
        names = [item["trait_name"] for item in parsed["results"]]
        assert "safety" in names
        assert "citation" in names

    def test_alternating_present_values(self) -> None:
        """Example JSON alternates present values (True, False, True, ...)."""
        from karenina.benchmark.verification.prompts.rubric.presence_check import (
            PresenceCheckPromptBuilder,
        )

        builder = PresenceCheckPromptBuilder()
        traits = [
            _make_llm_trait("a", summary="A"),
            _make_llm_trait("b", summary="B"),
            _make_llm_trait("c", summary="C"),
        ]
        parsed = json.loads(builder.build_example_json(traits))
        present_values = [item["present"] for item in parsed["results"]]
        assert present_values == [True, False, True]
