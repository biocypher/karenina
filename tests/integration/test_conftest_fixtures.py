"""Tests to verify integration conftest fixtures work correctly.

This module validates that all integration fixtures can be loaded and used
without errors. It serves as a smoke test for the integration test infrastructure.
"""

import pytest

from karenina.schemas.domain import BaseAnswer, Rubric

# =============================================================================
# Model Config Fixture Tests
# =============================================================================


@pytest.mark.integration
class TestModelConfigFixtures:
    """Test model configuration fixtures."""

    def test_parsing_model_config_has_required_fields(self, parsing_model_config):
        """Verify parsing model config has all required fields."""
        assert parsing_model_config.id is not None
        assert parsing_model_config.model_name is not None
        assert parsing_model_config.model_provider is not None
        assert parsing_model_config.interface == "langchain"
        assert parsing_model_config.temperature == 0.0


# =============================================================================
# Evaluator Fixture Tests
# =============================================================================


@pytest.mark.integration
class TestEvaluatorFixtures:
    """Test evaluator fixtures."""

    def test_template_evaluator_created(self, template_evaluator):
        """Verify template evaluator is created with fixture-backed LLM."""
        assert template_evaluator is not None
        assert template_evaluator.llm is not None
        assert template_evaluator.model_config is not None

    def test_rubric_evaluator_created(self, rubric_evaluator):
        """Verify rubric evaluator is created with fixture-backed LLM."""
        assert rubric_evaluator is not None
        assert rubric_evaluator.llm is not None
        assert rubric_evaluator.model_config is not None
        assert rubric_evaluator.evaluation_strategy == "batch"


# =============================================================================
# Trace Fixture Tests
# =============================================================================


@pytest.mark.integration
class TestTraceFixtures:
    """Test trace (sample response) fixtures."""

    def test_trace_with_citations_has_citation_markers(self, trace_with_citations):
        """Verify trace with citations contains citation patterns."""
        assert "[1]" in trace_with_citations
        assert "[2]" in trace_with_citations
        assert "BCL2" in trace_with_citations

    def test_trace_without_citations_has_no_markers(self, trace_without_citations):
        """Verify trace without citations has no citation patterns."""
        import re

        assert re.search(r"\[\d+\]", trace_without_citations) is None
        assert "BCL2" in trace_without_citations

    def test_trace_with_abstention_indicates_refusal(self, trace_with_abstention):
        """Verify abstention trace indicates model refusal."""
        assert "cannot" in trace_with_abstention.lower()
        assert "consult" in trace_with_abstention.lower()

    def test_trace_with_hedging_provides_answer(self, trace_with_hedging):
        """Verify hedging trace still contains substantive answer."""
        assert "cannot be completely certain" in trace_with_hedging.lower()
        assert "BCL2" in trace_with_hedging


# =============================================================================
# Answer Template Fixture Tests
# =============================================================================


@pytest.mark.integration
class TestAnswerTemplateFixtures:
    """Test answer template fixtures."""

    def test_simple_answer_is_base_answer(self, simple_answer):
        """Verify simple answer inherits from BaseAnswer."""
        assert issubclass(simple_answer, BaseAnswer)
        assert simple_answer.__name__ == "Answer"

    def test_multi_field_answer_is_base_answer(self, multi_field_answer):
        """Verify multi-field answer inherits from BaseAnswer."""
        assert issubclass(multi_field_answer, BaseAnswer)
        assert multi_field_answer.__name__ == "Answer"

    def test_answer_with_correct_dict_is_base_answer(self, answer_with_correct_dict):
        """Verify answer with correct dict inherits from BaseAnswer."""
        assert issubclass(answer_with_correct_dict, BaseAnswer)

    def test_answer_templates_dict_has_all_templates(self, answer_templates):
        """Verify answer templates dictionary contains all template types."""
        assert "simple_extraction" in answer_templates
        assert "multi_field" in answer_templates
        assert "with_correct_dict" in answer_templates
        assert len(answer_templates) == 3

    def test_simple_answer_can_instantiate(self, simple_answer):
        """Verify simple answer can be instantiated with defaults."""
        instance = simple_answer(value="test")
        assert instance.value == "test"
        assert hasattr(instance, "verify")

    def test_simple_answer_verify_method_works(self, simple_answer):
        """Verify simple answer verify method returns boolean."""
        # Correct value
        correct_instance = simple_answer(value="42")
        assert correct_instance.verify() is True

        # Incorrect value
        incorrect_instance = simple_answer(value="wrong")
        assert incorrect_instance.verify() is False


# =============================================================================
# Rubric Fixture Tests
# =============================================================================


@pytest.mark.integration
class TestRubricFixtures:
    """Test rubric fixtures."""

    def test_boolean_rubric_has_single_trait(self, boolean_rubric):
        """Verify boolean rubric has one LLM trait."""
        assert isinstance(boolean_rubric, Rubric)
        assert len(boolean_rubric.llm_traits) == 1
        assert boolean_rubric.llm_traits[0].kind == "boolean"
        assert boolean_rubric.llm_traits[0].name == "clarity"

    def test_scored_rubric_has_numeric_trait(self, scored_rubric):
        """Verify scored rubric has numeric trait with range."""
        assert isinstance(scored_rubric, Rubric)
        assert len(scored_rubric.llm_traits) == 1
        trait = scored_rubric.llm_traits[0]
        assert trait.kind == "score"
        assert trait.min_score == 1
        assert trait.max_score == 5

    def test_multi_trait_rubric_has_mixed_types(self, multi_trait_rubric):
        """Verify multi-trait rubric has LLM and regex traits."""
        assert isinstance(multi_trait_rubric, Rubric)
        assert len(multi_trait_rubric.llm_traits) == 2
        assert len(multi_trait_rubric.regex_traits) == 1

    def test_citation_regex_rubric_is_deterministic(self, citation_regex_rubric):
        """Verify citation rubric has only regex traits (no LLM needed)."""
        assert isinstance(citation_regex_rubric, Rubric)
        assert len(citation_regex_rubric.llm_traits) == 0
        assert len(citation_regex_rubric.regex_traits) == 2


# =============================================================================
# Checkpoint Fixture Tests
# =============================================================================


@pytest.mark.integration
class TestCheckpointFixtures:
    """Test checkpoint/benchmark fixtures."""

    def test_minimal_benchmark_loads(self, minimal_benchmark):
        """Verify minimal benchmark loads successfully."""
        assert minimal_benchmark is not None
        assert len(minimal_benchmark) == 1  # 1 question

    def test_multi_question_benchmark_loads(self, multi_question_benchmark):
        """Verify multi-question benchmark loads successfully."""
        assert multi_question_benchmark is not None
        assert len(multi_question_benchmark) == 5  # 5 questions

    def test_benchmark_with_results_loads(self, benchmark_with_results):
        """Verify benchmark with results loads successfully."""
        assert benchmark_with_results is not None
        # Should have questions and potentially pre-existing rubric configuration
        assert len(benchmark_with_results) >= 1


# =============================================================================
# Integration Test Helpers
# =============================================================================


@pytest.mark.integration
class TestIntegrationHelpers:
    """Test integration test helper fixtures."""

    def test_fixtures_dir_exists(self, fixtures_dir):
        """Verify fixtures directory exists (inherited from root conftest)."""
        assert fixtures_dir.exists()
        assert fixtures_dir.is_dir()

    def test_checkpoint_fixtures_dir_exists(self, checkpoint_fixtures_dir):
        """Verify checkpoint fixtures directory exists."""
        assert checkpoint_fixtures_dir.exists()
        assert checkpoint_fixtures_dir.is_dir()
