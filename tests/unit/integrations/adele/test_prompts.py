"""Unit tests for ADeLe classification prompts."""

from karenina.integrations.adele.prompts import (
    SYSTEM_PROMPT_BATCH,
    SYSTEM_PROMPT_SINGLE_TRAIT,
    USER_PROMPT_BATCH_TEMPLATE,
    USER_PROMPT_SINGLE_TRAIT_TEMPLATE,
)
from karenina.integrations.adele.traits import get_adele_trait


class TestSystemPrompts:
    """Tests for system prompt constants."""

    def test_single_trait_prompt_exists(self) -> None:
        """System prompt for single trait should be non-empty."""
        assert SYSTEM_PROMPT_SINGLE_TRAIT
        assert len(SYSTEM_PROMPT_SINGLE_TRAIT) > 100

    def test_batch_prompt_exists(self) -> None:
        """System prompt for batch should be non-empty."""
        assert SYSTEM_PROMPT_BATCH
        assert len(SYSTEM_PROMPT_BATCH) > 100

    def test_single_trait_prompt_contains_key_instructions(self) -> None:
        """Single trait prompt should contain key instructions."""
        prompt = SYSTEM_PROMPT_SINGLE_TRAIT
        assert "ADeLe" in prompt
        assert "SINGLE dimension" in prompt
        assert "JSON" in prompt
        assert "CRITICAL REQUIREMENTS" in prompt
        assert "Exact Class Name" in prompt

    def test_batch_prompt_contains_key_instructions(self) -> None:
        """Batch prompt should contain key instructions."""
        prompt = SYSTEM_PROMPT_BATCH
        assert "ADeLe" in prompt
        assert "multiple dimensions" in prompt
        assert "JSON" in prompt
        assert "CRITICAL REQUIREMENTS" in prompt
        assert "All Traits Required" in prompt

    def test_prompts_are_different(self) -> None:
        """Single and batch prompts should be different."""
        assert SYSTEM_PROMPT_SINGLE_TRAIT != SYSTEM_PROMPT_BATCH

    def test_single_trait_no_format_placeholders(self) -> None:
        """Single trait system prompt should have no format placeholders."""
        # Should not raise KeyError when formatting with empty dict
        # This verifies there are no {placeholder} patterns
        assert "{" not in SYSTEM_PROMPT_SINGLE_TRAIT

    def test_batch_no_format_placeholders(self) -> None:
        """Batch system prompt should have no format placeholders."""
        assert "{" not in SYSTEM_PROMPT_BATCH


class TestUserPromptTemplates:
    """Tests for user prompt templates."""

    def test_single_trait_template_exists(self) -> None:
        """Single trait template should be non-empty."""
        assert USER_PROMPT_SINGLE_TRAIT_TEMPLATE
        assert len(USER_PROMPT_SINGLE_TRAIT_TEMPLATE) > 100

    def test_batch_template_exists(self) -> None:
        """Batch template should be non-empty."""
        assert USER_PROMPT_BATCH_TEMPLATE
        assert len(USER_PROMPT_BATCH_TEMPLATE) > 100

    def test_single_trait_template_has_required_placeholders(self) -> None:
        """Single trait template should have all required placeholders."""
        template = USER_PROMPT_SINGLE_TRAIT_TEMPLATE
        required_placeholders = [
            "{trait_name}",
            "{question_text}",
            "{trait_description}",
            "{class_names}",
            "{class_details}",
            "{json_schema}",
            "{example_json}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in template, f"Missing placeholder: {placeholder}"

    def test_batch_template_has_required_placeholders(self) -> None:
        """Batch template should have all required placeholders."""
        template = USER_PROMPT_BATCH_TEMPLATE
        required_placeholders = [
            "{question_text}",
            "{traits_description}",
            "{json_schema}",
            "{example_json}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in template, f"Missing placeholder: {placeholder}"

    def test_single_trait_template_formats_correctly(self) -> None:
        """Single trait template should format without errors."""
        result = USER_PROMPT_SINGLE_TRAIT_TEMPLATE.format(
            trait_name="attention_and_scan",
            question_text="What is 2+2?",
            trait_description="Tests attention span",
            class_names="none, very_low, low",
            class_details="- none: No attention\n- very_low: Minimal",
            json_schema='{"type": "object"}',
            example_json='{"classification": "none"}',
        )
        assert "attention_and_scan" in result
        assert "What is 2+2?" in result
        assert "Tests attention span" in result

    def test_batch_template_formats_correctly(self) -> None:
        """Batch template should format without errors."""
        result = USER_PROMPT_BATCH_TEMPLATE.format(
            question_text="What is the capital of France?",
            traits_description="- attention: Tests focus\n- volume: Tests content",
            json_schema='{"type": "object"}',
            example_json='{"classifications": {"attention": "low"}}',
        )
        assert "What is the capital of France?" in result
        assert "attention: Tests focus" in result

    def test_templates_are_different(self) -> None:
        """Single and batch templates should be different."""
        assert USER_PROMPT_SINGLE_TRAIT_TEMPLATE != USER_PROMPT_BATCH_TEMPLATE


class TestPromptIntegrationWithTraits:
    """Tests for prompts working with real ADeLe traits."""

    def test_single_trait_with_real_trait(self) -> None:
        """Template should work with a real ADeLe trait."""
        trait = get_adele_trait("attention_and_scan")

        class_names = list(trait.classes.keys())
        class_details = [f"  - {name}: desc" for name in class_names]

        result = USER_PROMPT_SINGLE_TRAIT_TEMPLATE.format(
            trait_name=trait.name,
            question_text="Sample question",
            trait_description=trait.description or "No description",
            class_names=", ".join(class_names),
            class_details="\n".join(class_details),
            json_schema="{}",
            example_json="{}",
        )

        assert trait.name in result
        assert "Sample question" in result
        # Should contain all class names
        for class_name in class_names:
            assert class_name in result

    def test_batch_with_multiple_traits(self) -> None:
        """Template should work with multiple ADeLe traits."""
        traits = [get_adele_trait("attention_and_scan"), get_adele_trait("volume")]

        traits_desc = []
        for trait in traits:
            traits_desc.append(f"- {trait.name}: {trait.description or 'desc'}")

        result = USER_PROMPT_BATCH_TEMPLATE.format(
            question_text="Complex question here",
            traits_description="\n".join(traits_desc),
            json_schema="{}",
            example_json="{}",
        )

        assert "Complex question here" in result
        for trait in traits:
            assert trait.name in result


class TestPromptClassifierIntegration:
    """Tests verifying prompts work correctly via QuestionClassifier."""

    def test_classifier_builds_single_trait_prompt(self) -> None:
        """Classifier should build single trait prompt using template."""
        from karenina.integrations.adele.classifier import QuestionClassifier

        classifier = QuestionClassifier()
        trait = get_adele_trait("volume")

        system_prompt = classifier._build_system_prompt_single_trait()
        user_prompt = classifier._build_user_prompt_single_trait("Test question?", trait)

        # System prompt should match constant
        assert system_prompt == SYSTEM_PROMPT_SINGLE_TRAIT

        # User prompt should contain formatted values
        assert "volume" in user_prompt
        assert "Test question?" in user_prompt
        assert "classification" in user_prompt  # From example JSON

    def test_classifier_builds_batch_prompt(self) -> None:
        """Classifier should build batch prompt using template."""
        from karenina.integrations.adele.classifier import QuestionClassifier

        classifier = QuestionClassifier()
        traits = [get_adele_trait("volume"), get_adele_trait("atypicality")]

        system_prompt = classifier._build_system_prompt()
        user_prompt = classifier._build_user_prompt("Batch test?", traits)

        # System prompt should match constant
        assert system_prompt == SYSTEM_PROMPT_BATCH

        # User prompt should contain all trait names
        assert "Batch test?" in user_prompt
        assert "volume" in user_prompt
        assert "atypicality" in user_prompt
        assert "classifications" in user_prompt  # From example JSON

    def test_classifier_prompt_contains_json_schema(self) -> None:
        """User prompts should contain valid JSON schema."""
        from karenina.integrations.adele.classifier import QuestionClassifier

        classifier = QuestionClassifier()
        trait = get_adele_trait("attention_and_scan")

        user_prompt = classifier._build_user_prompt_single_trait("Q?", trait)

        # Extract and validate JSON schema is parseable
        assert "JSON SCHEMA" in user_prompt
        # The schema should be valid JSON (embedded in the prompt)
        assert '"type"' in user_prompt or '"properties"' in user_prompt
