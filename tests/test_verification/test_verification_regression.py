"""Regression tests for verification pipeline refactor.

These tests compare the new stage-based pipeline against the legacy
monolithic implementation to ensure 100% behavioral equivalence.

Test Strategy:
- Mock LLM calls for deterministic results
- Run both implementations with identical inputs
- Compare all VerificationResult fields
- Test various configuration combinations
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from karenina.benchmark.models import ModelConfig, VerificationResult
from karenina.benchmark.verification.runner import run_single_model_verification
from karenina.benchmark.verification.runner_legacy import (
    run_single_model_verification as run_single_model_verification_LEGACY,
)
from karenina.schemas.rubric_class import Rubric, RubricTrait


def compare_verification_results(
    new_result: VerificationResult,
    legacy_result: VerificationResult,
    check_timestamps: bool = False,
) -> None:
    """Compare two VerificationResult objects for equivalence.

    Args:
        new_result: Result from new stage-based implementation
        legacy_result: Result from legacy monolithic implementation
        check_timestamps: Whether to check timestamp fields (usually False for tests)
    """
    # Core verification fields
    assert new_result.question_id == legacy_result.question_id, "question_id mismatch"
    assert new_result.template_id == legacy_result.template_id, "template_id mismatch"
    assert new_result.completed_without_errors == legacy_result.completed_without_errors, (
        "completed_without_errors mismatch"
    )
    assert new_result.error == legacy_result.error, "error mismatch"
    assert new_result.verify_result == legacy_result.verify_result, "verify_result mismatch"

    # Response fields
    assert new_result.question_text == legacy_result.question_text, "question_text mismatch"
    assert new_result.raw_llm_response == legacy_result.raw_llm_response, "raw_llm_response mismatch"
    assert new_result.parsed_gt_response == legacy_result.parsed_gt_response, "parsed_gt_response mismatch"

    # Parsed LLM response comparison (dict comparison)
    if new_result.parsed_llm_response is not None and legacy_result.parsed_llm_response is not None:
        assert new_result.parsed_llm_response == legacy_result.parsed_llm_response, (
            f"parsed_llm_response mismatch: {new_result.parsed_llm_response} != {legacy_result.parsed_llm_response}"
        )

    # Model configuration
    assert new_result.answering_model == legacy_result.answering_model, "answering_model mismatch"
    assert new_result.parsing_model == legacy_result.parsing_model, "parsing_model mismatch"
    assert new_result.answering_system_prompt == legacy_result.answering_system_prompt, (
        "answering_system_prompt mismatch"
    )
    assert new_result.parsing_system_prompt == legacy_result.parsing_system_prompt, "parsing_system_prompt mismatch"

    # Metadata fields
    assert new_result.run_name == legacy_result.run_name, "run_name mismatch"
    assert new_result.job_id == legacy_result.job_id, "job_id mismatch"
    assert new_result.keywords == legacy_result.keywords, "keywords mismatch"

    # Timing (execution_time can vary slightly, check it exists)
    if check_timestamps:
        assert new_result.execution_time is not None, "execution_time missing"
        assert new_result.timestamp is not None, "timestamp missing"

    # Embedding check fields
    assert new_result.embedding_check_performed == legacy_result.embedding_check_performed, (
        "embedding_check_performed mismatch"
    )
    assert new_result.embedding_similarity_score == legacy_result.embedding_similarity_score, (
        "embedding_similarity_score mismatch"
    )
    assert new_result.embedding_override_applied == legacy_result.embedding_override_applied, (
        "embedding_override_applied mismatch"
    )
    assert new_result.embedding_model_used == legacy_result.embedding_model_used, "embedding_model_used mismatch"

    # Regex validation fields
    assert new_result.regex_validations_performed == legacy_result.regex_validations_performed, (
        "regex_validations_performed mismatch"
    )
    assert new_result.regex_validation_results == legacy_result.regex_validation_results, (
        "regex_validation_results mismatch"
    )
    assert new_result.regex_validation_details == legacy_result.regex_validation_details, (
        "regex_validation_details mismatch"
    )
    assert new_result.regex_overall_success == legacy_result.regex_overall_success, "regex_overall_success mismatch"
    assert new_result.regex_extraction_results == legacy_result.regex_extraction_results, (
        "regex_extraction_results mismatch"
    )

    # Abstention fields
    assert new_result.abstention_check_performed == legacy_result.abstention_check_performed, (
        "abstention_check_performed mismatch"
    )
    assert new_result.abstention_detected == legacy_result.abstention_detected, "abstention_detected mismatch"
    assert new_result.abstention_override_applied == legacy_result.abstention_override_applied, (
        "abstention_override_applied mismatch"
    )
    assert new_result.abstention_reasoning == legacy_result.abstention_reasoning, "abstention_reasoning mismatch"

    # Deep-judgment fields
    assert new_result.deep_judgment_enabled == legacy_result.deep_judgment_enabled, "deep_judgment_enabled mismatch"
    assert new_result.deep_judgment_performed == legacy_result.deep_judgment_performed, (
        "deep_judgment_performed mismatch"
    )
    assert new_result.extracted_excerpts == legacy_result.extracted_excerpts, "extracted_excerpts mismatch"
    assert new_result.attribute_reasoning == legacy_result.attribute_reasoning, "attribute_reasoning mismatch"
    assert new_result.deep_judgment_stages_completed == legacy_result.deep_judgment_stages_completed, (
        "deep_judgment_stages_completed mismatch"
    )
    assert new_result.deep_judgment_model_calls == legacy_result.deep_judgment_model_calls, (
        "deep_judgment_model_calls mismatch"
    )
    assert new_result.deep_judgment_excerpt_retry_count == legacy_result.deep_judgment_excerpt_retry_count, (
        "deep_judgment_excerpt_retry_count mismatch"
    )
    assert new_result.attributes_without_excerpts == legacy_result.attributes_without_excerpts, (
        "attributes_without_excerpts mismatch"
    )
    assert new_result.deep_judgment_search_enabled == legacy_result.deep_judgment_search_enabled, (
        "deep_judgment_search_enabled mismatch"
    )
    assert new_result.hallucination_risk_assessment == legacy_result.hallucination_risk_assessment, (
        "hallucination_risk_assessment mismatch"
    )

    # Rubric fields
    assert new_result.verify_rubric == legacy_result.verify_rubric, "verify_rubric mismatch"
    assert new_result.evaluation_rubric == legacy_result.evaluation_rubric, "evaluation_rubric mismatch"

    # Metric trait fields
    assert new_result.metric_trait_confusion_lists == legacy_result.metric_trait_confusion_lists, (
        "metric_trait_confusion_lists mismatch"
    )
    assert new_result.metric_trait_metrics == legacy_result.metric_trait_metrics, "metric_trait_metrics mismatch"

    # Other fields
    assert new_result.recursion_limit_reached == legacy_result.recursion_limit_reached, (
        "recursion_limit_reached mismatch"
    )
    assert new_result.answering_mcp_servers == legacy_result.answering_mcp_servers, "answering_mcp_servers mismatch"
    assert new_result.answering_replicate == legacy_result.answering_replicate, "answering_replicate mismatch"
    assert new_result.parsing_replicate == legacy_result.parsing_replicate, "parsing_replicate mismatch"


class TestBasicRegression:
    """Basic regression tests for common verification scenarios."""

    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_basic_template_verification_equivalence(
        self, mock_parse_llm: Mock, mock_generate_llm: Mock, mock_runner_legacy_llm: Mock
    ) -> None:
        """Test basic template verification produces identical results."""
        # Setup models
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLM responses - init_chat_model_unified is called with model="gpt-4o-mini", provider="openai"
        # We'll return a mock LLM that gives different responses based on which invocation
        # Track calls with modulo to alternate between answering and parsing for each run
        call_count = [0]

        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()

            # Alternate between answering (even calls) and parsing (odd calls)
            if call_count[0] % 2 == 0:
                # Answering model returns natural language
                mock_response = Mock()
                mock_response.content = "The answer is 42"
                mock_llm.invoke.return_value = mock_response
            else:
                # Parsing model returns JSON
                mock_response = Mock()
                mock_response.content = '{"result": 42}'
                mock_llm.invoke.return_value = mock_response

            call_count[0] += 1
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        # Template
        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        # Common parameters
        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": ["test"],
            "run_name": "test_run",
            "few_shot_examples": None,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Verify key fields
        assert new_result.verify_result is True
        assert new_result.completed_without_errors is True

    @patch("karenina.benchmark.verification.rubric_evaluator.RubricEvaluator")
    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_template_with_rubric_equivalence(
        self,
        mock_parse_llm: Mock,
        mock_generate_llm: Mock,
        mock_runner_legacy_llm: Mock,
        mock_evaluator_class: Mock,
    ) -> None:
        """Test template + rubric evaluation produces identical results."""
        # Setup models
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLM responses
        call_count = [0]

        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()

            # Alternate between answering (even calls) and parsing (odd calls)
            if call_count[0] % 2 == 0:
                # Answering model returns natural language
                mock_response = Mock()
                mock_response.content = "The answer is 42 and it's a great answer!"
                mock_llm.invoke.return_value = mock_response
            else:
                # Parsing model returns JSON
                mock_response = Mock()
                mock_response.content = '{"result": 42}'
                mock_llm.invoke.return_value = mock_response

            call_count[0] += 1
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        # Mock rubric evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_traits.return_value = (
            {"Clarity": 8},  # trait_result
            {"Clarity": "Very clear answer"},  # trait_reasoning
        )
        mock_evaluator_class.return_value = mock_evaluator

        # Setup rubric
        rubric = Rubric(
            traits=[
                RubricTrait(
                    name="Clarity",
                    description="Is the answer clear?",
                    kind="score",
                    min_score=1,
                    max_score=10,
                )
            ]
        )

        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "rubric": rubric,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": None,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Verify rubric was evaluated
        assert new_result.verify_rubric is not None
        assert "Clarity" in new_result.verify_rubric

    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_template_verification_failure_equivalence(
        self, mock_parse_llm: Mock, mock_generate_llm: Mock, mock_runner_legacy_llm: Mock
    ) -> None:
        """Test that verification failures produce identical results."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLM responses - wrong answer
        call_count = [0]

        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()

            # Alternate between answering (even calls) and parsing (odd calls)
            if call_count[0] % 2 == 0:
                # Answering model returns natural language
                mock_response = Mock()
                mock_response.content = "The answer is 99"
                mock_llm.invoke.return_value = mock_response
            else:
                # Parsing model returns JSON
                mock_response = Mock()
                mock_response.content = '{"result": 99}'
                mock_llm.invoke.return_value = mock_response

            call_count[0] += 1
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": None,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Verify failure is captured correctly
        assert new_result.verify_result is False
        assert new_result.completed_without_errors is True  # Pipeline succeeded, answer was wrong

    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_error_case_equivalence(
        self, mock_parse_llm: Mock, mock_generate_llm: Mock, mock_runner_legacy_llm: Mock
    ) -> None:
        """Test that pipeline errors produce identical results."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLMs (won't be called due to template error)
        # Still need mocks to prevent any potential calls
        mock_llm = Mock()
        mock_parse_llm.return_value = mock_llm
        mock_generate_llm.return_value = mock_llm
        mock_runner_legacy_llm.return_value = mock_llm

        # Invalid template (syntax error)
        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer"
    # Missing closing parenthesis
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": None,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Verify error is captured
        assert new_result.completed_without_errors is False
        assert new_result.error is not None
        assert "Template validation failed" in new_result.error or "syntax" in new_result.error.lower()

    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_few_shot_prompting_equivalence(
        self, mock_parse_llm: Mock, mock_generate_llm: Mock, mock_runner_legacy_llm: Mock
    ) -> None:
        """Test that few-shot prompting produces identical results."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLM responses
        call_count = [0]

        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()

            # Alternate between answering (even calls) and parsing (odd calls)
            if call_count[0] % 2 == 0:
                # Answering model returns natural language
                mock_response = Mock()
                mock_response.content = "The sum is 7"
                mock_llm.invoke.return_value = mock_response
            else:
                # Parsing model returns JSON
                mock_response = Mock()
                mock_response.content = '{"result": 7}'
                mock_llm.invoke.return_value = mock_response

            call_count[0] += 1
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The sum")

    def verify(self) -> bool:
        return self.result == 7
"""

        few_shot_examples = [
            {"question": "What is 1 + 1?", "answer": "The sum is 2"},
            {"question": "What is 2 + 2?", "answer": "The sum is 4"},
        ]

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is 3 + 4?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": few_shot_examples,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Verify success
        assert new_result.verify_result is True
        assert new_result.completed_without_errors is True


class TestAdvancedRegression:
    """Advanced regression tests for complex features."""

    @patch("karenina.benchmark.verification.runner_legacy.detect_abstention")
    @patch("karenina.benchmark.verification.stages.abstention_check.detect_abstention")
    @patch("karenina.benchmark.verification.runner_legacy.deep_judgment_parse")
    @patch("karenina.benchmark.verification.stages.parse_template.deep_judgment_parse")
    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_deep_judgment_parsing_equivalence(
        self,
        mock_parse_llm: Mock,
        mock_generate_llm: Mock,
        mock_runner_legacy_llm: Mock,
        mock_dj_stage: Mock,
        mock_dj_runner: Mock,
        mock_abstention_stage: Mock,
        mock_abstention_runner: Mock,
    ) -> None:
        """Test deep-judgment parsing produces identical results."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLM responses
        call_count = [0]

        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()

            # Alternate between answering (even calls) and parsing (odd calls)
            if call_count[0] % 2 == 0:
                # Answering model returns natural language with details
                mock_response = Mock()
                mock_response.content = (
                    "The answer is 42, which is the ultimate answer to life, the universe, and everything."
                )
                mock_llm.invoke.return_value = mock_response
            else:
                # Parsing model returns JSON (used if deep_judgment_parse isn't mocked properly)
                mock_response = Mock()
                mock_response.content = '{"result": 42}'
                mock_llm.invoke.return_value = mock_response

            call_count[0] += 1
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        # Mock abstention detection (required by DeepJudgmentAutoFail stage)
        # Returns: (abstention_detected, check_performed, reasoning)
        mock_abstention_result = (False, True, None)
        mock_abstention_stage.return_value = mock_abstention_result
        mock_abstention_runner.return_value = mock_abstention_result

        # Mock deep_judgment_parse to return the 4-tuple
        # Returns: (parsed_answer, excerpts, reasoning, metadata)
        from pydantic import Field

        from karenina.schemas.answer_class import BaseAnswer

        # Create a mock parsed answer
        class TestAnswer(BaseAnswer):
            result: int = Field(description="The answer")

            def verify(self) -> bool:
                return self.result == 42

        mock_parsed_answer = TestAnswer(result=42)
        mock_excerpts = {"result": [{"text": "The answer is 42", "confidence": "high", "similarity_score": 0.95}]}
        mock_reasoning = {"result": "The excerpt explicitly states 42 as the answer"}
        mock_metadata = {
            "stages_completed": ["excerpt_extraction", "reasoning_generation", "parameter_parsing"],
            "model_calls": 3,
            "excerpt_retry_count": 0,
            "attributes_without_excerpts": None,
            "hallucination_risk": None,
        }

        mock_dj_result = (mock_parsed_answer, mock_excerpts, mock_reasoning, mock_metadata)
        mock_dj_stage.return_value = mock_dj_result
        mock_dj_runner.return_value = mock_dj_result

        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": None,
            "abstention_enabled": True,  # Required by DeepJudgmentAutoFail stage
            "deep_judgment_enabled": True,
            "deep_judgment_max_excerpts_per_attribute": 3,
            "deep_judgment_fuzzy_match_threshold": 0.80,
            "deep_judgment_excerpt_retry_attempts": 2,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Verify deep judgment was performed
        assert new_result.deep_judgment_enabled is True
        assert new_result.deep_judgment_performed is True
        assert new_result.extracted_excerpts is not None
        assert "result" in new_result.extracted_excerpts
        assert new_result.attribute_reasoning is not None
        assert "result" in new_result.attribute_reasoning
        assert new_result.deep_judgment_stages_completed == [
            "excerpt_extraction",
            "reasoning_generation",
            "parameter_parsing",
        ]
        assert new_result.deep_judgment_model_calls == 3
        assert new_result.verify_result is True

    @patch("karenina.benchmark.verification.runner_legacy.detect_abstention")
    @patch("karenina.benchmark.verification.stages.abstention_check.detect_abstention")
    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_abstention_detection_equivalence(
        self,
        mock_parse_llm: Mock,
        mock_generate_llm: Mock,
        mock_runner_legacy_llm: Mock,
        mock_abstention_stage: Mock,
        mock_abstention_runner: Mock,
    ) -> None:
        """Test abstention detection produces identical results."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLM responses - answering model refuses
        call_count = [0]

        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()

            # Alternate between answering (even calls) and parsing (odd calls)
            if call_count[0] % 2 == 0:
                # Answering model refuses to answer
                mock_response = Mock()
                mock_response.content = "I cannot answer this question as it goes against my guidelines."
                mock_llm.invoke.return_value = mock_response
            else:
                # Parsing model returns JSON (shouldn't be used much due to abstention)
                mock_response = Mock()
                mock_response.content = '{"result": 0}'
                mock_llm.invoke.return_value = mock_response

            call_count[0] += 1
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        # Mock abstention detection to return detected
        # Returns: (abstention_detected, check_performed, reasoning)
        mock_abstention_result = (True, True, "Response contains explicit refusal pattern")
        mock_abstention_stage.return_value = mock_abstention_result
        mock_abstention_runner.return_value = mock_abstention_result

        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": None,
            "abstention_enabled": True,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Verify abstention was detected
        assert new_result.abstention_check_performed is True
        assert new_result.abstention_detected is True
        assert new_result.abstention_override_applied is True
        assert new_result.abstention_reasoning == "Response contains explicit refusal pattern"
        # Verification should fail due to abstention
        assert new_result.verify_result is False

    @patch("karenina.benchmark.verification.runner_legacy.perform_embedding_check")
    @patch("karenina.benchmark.verification.stages.embedding_check.perform_embedding_check")
    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_embedding_check_equivalence(
        self,
        mock_parse_llm: Mock,
        mock_generate_llm: Mock,
        mock_runner_legacy_llm: Mock,
        mock_embedding_stage: Mock,
        mock_embedding_runner: Mock,
    ) -> None:
        """Test embedding check produces identical results."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLM responses - wrong answer for field verification to fail
        call_count = [0]

        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()

            # Alternate between answering (even calls) and parsing (odd calls)
            if call_count[0] % 2 == 0:
                # Answering model returns similar but wrong answer
                mock_response = Mock()
                mock_response.content = "The answer is 43"  # Wrong value
                mock_llm.invoke.return_value = mock_response
            else:
                # Parsing model returns JSON with wrong value
                mock_response = Mock()
                mock_response.content = '{"result": 43}'  # Wrong value
                mock_llm.invoke.return_value = mock_response

            call_count[0] += 1
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        # Mock embedding check to return override success
        # Returns: (should_override, similarity_score, model_name, check_performed)
        mock_embedding_result = (True, 0.92, "all-MiniLM-L6-v2", True)
        mock_embedding_stage.return_value = mock_embedding_result
        mock_embedding_runner.return_value = mock_embedding_result

        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": None,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Verify embedding check was performed and override applied
        assert new_result.embedding_check_performed is True
        assert new_result.embedding_override_applied is True
        assert new_result.embedding_similarity_score == 0.92
        assert new_result.embedding_model_used == "all-MiniLM-L6-v2"
        # Field verification should fail, but embedding override makes overall pass
        assert new_result.verify_result is True

    def test_metric_trait_evaluation_equivalence(self) -> None:
        """Test metric trait evaluation produces identical results.

        Note: Skipped because properly mocking the RubricEvaluator is complex.
        Metric traits are tested in test_all_features_combined_equivalence.
        """
        pytest.skip("Metric trait evaluation is tested in combined features test")

    @patch("karenina.benchmark.verification.rubric_evaluator.RubricEvaluator")
    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_all_features_combined_equivalence(
        self,
        mock_parse_llm: Mock,
        mock_generate_llm: Mock,
        mock_runner_legacy_llm: Mock,
        mock_evaluator_class: Mock,
    ) -> None:
        """Test all features together produce identical results."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLM responses
        call_count = [0]

        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()

            # Alternate between answering (even calls) and parsing (odd calls)
            if call_count[0] % 2 == 0:
                # Answering model returns comprehensive natural language
                mock_response = Mock()
                mock_response.content = "The answer is 42, which is a clear and well-explained solution."
                mock_llm.invoke.return_value = mock_response
            else:
                # Parsing model returns JSON
                mock_response = Mock()
                mock_response.content = '{"result": 42}'
                mock_llm.invoke.return_value = mock_response

            call_count[0] += 1
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        # Mock rubric evaluator for both trait types
        mock_evaluator = Mock()
        # Regular traits
        mock_evaluator.evaluate_traits.return_value = (
            {"Clarity": 9, "Completeness": 8},  # trait_result
            {"Clarity": "Very clear answer", "Completeness": "Complete solution"},  # trait_reasoning
        )
        # Metric traits
        mock_confusion_lists = {"Accuracy": {"tp": ["Correct value"], "tn": [], "fp": [], "fn": []}}
        mock_metrics = {"Accuracy": {"precision": 1.0, "recall": 1.0, "f1": 1.0}}
        mock_evaluator.evaluate_metric_traits.return_value = (mock_confusion_lists, mock_metrics)
        mock_evaluator_class.return_value = mock_evaluator

        # Setup comprehensive rubric
        from karenina.schemas.rubric_class import MetricRubricTrait

        rubric = Rubric(
            traits=[
                RubricTrait(
                    name="Clarity",
                    description="Is the answer clear?",
                    kind="score",
                    min_score=1,
                    max_score=10,
                ),
                RubricTrait(
                    name="Completeness",
                    description="Is the answer complete?",
                    kind="score",
                    min_score=1,
                    max_score=10,
                ),
            ],
            metric_traits=[
                MetricRubricTrait(
                    name="Accuracy",
                    description="Evaluate accuracy",
                    evaluation_mode="tp_only",
                    metrics=["precision", "recall", "f1"],
                    tp_instructions=["Correct value"],
                )
            ],
        )

        # Few-shot examples
        few_shot_examples = [
            {"question": "What is 1 + 1?", "answer": "The answer is 2"},
            {"question": "What is 2 + 2?", "answer": "The answer is 4"},
        ]

        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer to life, the universe, and everything?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "rubric": rubric,
            "keywords": ["ultimate", "answer"],
            "run_name": "comprehensive_test",
            "few_shot_examples": few_shot_examples,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Verify all features were used
        assert new_result.verify_result is True
        assert new_result.completed_without_errors is True
        # Rubric regular traits
        assert new_result.verify_rubric is not None
        assert "Clarity" in new_result.verify_rubric
        assert "Completeness" in new_result.verify_rubric
        # Rubric metric traits
        assert new_result.metric_trait_confusion_lists is not None
        assert "Accuracy" in new_result.metric_trait_confusion_lists
        assert new_result.metric_trait_metrics is not None
        assert "Accuracy" in new_result.metric_trait_metrics
        # Keywords
        assert new_result.keywords == ["ultimate", "answer"]
        # Run metadata
        assert new_result.run_name == "comprehensive_test"


class TestEdgeCaseRegression:
    """Edge case regression tests."""

    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_empty_response_equivalence(
        self, mock_parse_llm: Mock, mock_generate_llm: Mock, mock_runner_legacy_llm: Mock
    ) -> None:
        """Test empty LLM responses produce identical results."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock empty responses - both answering and parsing return empty
        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = ""
            mock_llm.invoke.return_value = mock_response
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": None,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Should have parsing error
        assert new_result.completed_without_errors is False
        assert new_result.error is not None

    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_parsing_failure_equivalence(
        self, mock_parse_llm: Mock, mock_generate_llm: Mock, mock_runner_legacy_llm: Mock
    ) -> None:
        """Test parsing failures produce identical results."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLM responses - parsing returns malformed JSON
        call_count = [0]

        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()

            # Alternate between answering (even calls) and parsing (odd calls)
            if call_count[0] % 2 == 0:
                # Answering model returns natural language
                mock_response = Mock()
                mock_response.content = "The answer is 42"
                mock_llm.invoke.return_value = mock_response
            else:
                # Parsing model returns completely invalid JSON
                mock_response = Mock()
                mock_response.content = "This is not JSON at all!"  # Completely invalid
                mock_llm.invoke.return_value = mock_response

            call_count[0] += 1
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": None,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Should have parsing error
        assert new_result.completed_without_errors is False
        assert new_result.error is not None
        assert "Parsing failed" in new_result.error or "parsing" in new_result.error.lower()

    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_invalid_template_equivalence(
        self, mock_parse_llm: Mock, mock_generate_llm: Mock, mock_runner_legacy_llm: Mock
    ) -> None:
        """Test invalid templates produce identical results."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLMs (won't be called due to template validation error)
        mock_llm = Mock()
        mock_parse_llm.return_value = mock_llm
        mock_generate_llm.return_value = mock_llm
        mock_runner_legacy_llm.return_value = mock_llm

        # Template with wrong base class (not BaseAnswer)
        template_code = """class Answer:  # Missing BaseAnswer inheritance!
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": None,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Should have template validation error
        assert new_result.completed_without_errors is False
        assert new_result.error is not None
        # Both should have the same error about invalid template
        assert new_result.error == legacy_result.error

    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_recursion_limit_equivalence(
        self, mock_parse_llm: Mock, mock_generate_llm: Mock, mock_runner_legacy_llm: Mock
    ) -> None:
        """Test recursion limit handling produces identical results."""
        answering_model = ModelConfig(
            id="test-answering",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant.",
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLM responses - answering model hits recursion limit
        call_count = [0]

        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()

            # Alternate between answering (even calls) and parsing (odd calls)
            if call_count[0] % 2 == 0:
                # Answering model raises GraphRecursionError
                def raise_recursion_error(*args, **kwargs):  # noqa: ARG001
                    raise Exception("GraphRecursionError: recursion_limit exceeded")

                mock_llm.invoke.side_effect = raise_recursion_error
            else:
                # Parsing model returns JSON (won't be reached due to recursion error)
                mock_response = Mock()
                mock_response.content = '{"result": 0}'
                mock_llm.invoke.return_value = mock_response

            call_count[0] += 1
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": None,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Should have recursion limit reached
        assert new_result.recursion_limit_reached is True
        assert legacy_result.recursion_limit_reached is True
        # Both should have completed (recursion limit is handled gracefully)
        assert new_result.completed_without_errors is True
        assert legacy_result.completed_without_errors is True

    @patch("karenina.benchmark.verification.runner_legacy.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.generate_answer.init_chat_model_unified")
    @patch("karenina.benchmark.verification.stages.parse_template.init_chat_model_unified")
    def test_mcp_agent_equivalence(
        self, mock_parse_llm: Mock, mock_generate_llm: Mock, mock_runner_legacy_llm: Mock
    ) -> None:
        """Test MCP agent calls produce identical results."""
        # Create answering model with MCP configuration
        answering_model = ModelConfig(
            id="test-answering-mcp",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="You are a helpful assistant with tools.",
            mcp_urls_dict={"search": "http://localhost:8000"},  # Mock MCP server
            mcp_tool_filter=["search_web"],
        )
        parsing_model = ModelConfig(
            id="test-parsing",
            model_provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
            interface="langchain",
            system_prompt="Parse the response.",
        )

        # Mock LLM responses - MCP agent returns structured response
        call_count = [0]

        def mock_llm_side_effect(model, provider=None, **kwargs):  # noqa: ARG001
            mock_llm = Mock()

            # Alternate between answering (even calls) and parsing (odd calls)
            if call_count[0] % 2 == 0:
                # MCP agent returns message with tool calls
                mock_response = Mock()
                mock_response.content = "The answer is 42 (found using search tools)"
                # Simulate agent response structure
                mock_response.tool_calls = []

                # For MCP agents, we need to support async invocation
                mock_llm.ainvoke = AsyncMock(return_value={"messages": [mock_response]})
                mock_llm.invoke.return_value = mock_response
            else:
                # Parsing model returns JSON
                mock_response = Mock()
                mock_response.content = '{"result": 42}'
                mock_llm.invoke.return_value = mock_response

            call_count[0] += 1
            return mock_llm

        # Apply the same side_effect to all three patches
        mock_parse_llm.side_effect = mock_llm_side_effect
        mock_generate_llm.side_effect = mock_llm_side_effect
        mock_runner_legacy_llm.side_effect = mock_llm_side_effect

        template_code = """class Answer(BaseAnswer):
    result: int = Field(description="The answer")

    def verify(self) -> bool:
        return self.result == 42
"""

        kwargs: dict[str, Any] = {
            "question_id": "q1",
            "question_text": "What is the answer?",
            "template_code": template_code,
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "keywords": None,
            "run_name": None,
            "few_shot_examples": None,
        }

        # Run both implementations
        new_result = run_single_model_verification(**kwargs)
        legacy_result = run_single_model_verification_LEGACY(**kwargs)

        # Compare results
        compare_verification_results(new_result, legacy_result)

        # Should have completed successfully with MCP agent
        assert new_result.completed_without_errors is True
        assert new_result.verify_result is True
        # answering_mcp_servers stores list of server names
        assert new_result.answering_mcp_servers == ["search"]
