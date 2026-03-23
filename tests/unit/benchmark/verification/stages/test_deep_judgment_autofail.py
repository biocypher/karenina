"""Unit tests for DeepJudgmentAutoFailStage reasoning-only skip behavior.

Tests cover:
- Stage skips when reasoning-only mode was used (no excerpts to check)
- Stage still runs when reasoning-only flag is not set but excerpts are missing
- Stage skips when deep judgment was not performed (existing behavior preserved)

Also tests for Task 8: the parse template stage wiring of reasoning-only flag:
- deep_judgment_config includes reasoning_only from context
- DEEP_JUDGMENT_REASONING_ONLY artifact is set when reasoning-only metadata is present
"""

import pytest
from pydantic import Field

from karenina.benchmark.verification.stages.core.base import (
    ArtifactKeys,
    VerificationContext,
)
from karenina.benchmark.verification.stages.pipeline.deep_judgment_autofail import (
    DeepJudgmentAutoFailStage,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import BaseAnswer

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_model_config() -> ModelConfig:
    """Return a minimal ModelConfig for testing."""
    return ModelConfig(
        id="test-model",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
    )


@pytest.fixture
def sample_answer_class():
    """Return a sample Answer class for testing."""

    class Answer(BaseAnswer):
        drug_target: str = Field(description="The protein target")

        def verify(self) -> bool:
            return self.drug_target.lower() == "bcl-2"

    return Answer


@pytest.fixture
def minimal_context(minimal_model_config: ModelConfig) -> VerificationContext:
    """Return a minimal VerificationContext for testing."""
    return VerificationContext(
        question_id="test-question-1",
        template_id="template-hash-123",
        question_text="What is the drug target?",
        template_code="class Answer(BaseAnswer): ...",
        answering_model=minimal_model_config,
        parsing_model=minimal_model_config,
    )


@pytest.fixture
def stage() -> DeepJudgmentAutoFailStage:
    """Return a DeepJudgmentAutoFailStage instance."""
    return DeepJudgmentAutoFailStage()


# =============================================================================
# Task 7: Auto-fail stage skips for reasoning-only mode
# =============================================================================


@pytest.mark.unit
class TestDeepJudgmentAutoFailReasoningOnlySkip:
    """Tests that DeepJudgmentAutoFailStage skips when reasoning-only mode was used."""

    def test_should_not_run_when_reasoning_only_artifact_set(
        self, stage: DeepJudgmentAutoFailStage, minimal_context: VerificationContext
    ):
        """Stage must not run when reasoning-only flag is set in artifacts.

        Reasoning-only mode skips excerpt extraction, so there are no excerpts
        to validate. The auto-fail stage should recognize this and skip.
        """
        minimal_context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, True)
        minimal_context.set_artifact(ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS, ["drug_target"])
        minimal_context.set_artifact(ArtifactKeys.VERIFY_RESULT, True)
        minimal_context.set_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT, True)
        minimal_context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_REASONING_ONLY, True)

        assert stage.should_run(minimal_context) is False

    def test_should_run_when_reasoning_only_not_set_and_excerpts_missing(
        self, stage: DeepJudgmentAutoFailStage, minimal_context: VerificationContext
    ):
        """Stage must still run when reasoning-only is not set but excerpts are missing.

        This preserves the existing behavior: attributes without excerpts in
        standard deep judgment mode should trigger the auto-fail.
        """
        minimal_context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, True)
        minimal_context.set_artifact(ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS, ["drug_target"])
        minimal_context.set_artifact(ArtifactKeys.VERIFY_RESULT, True)
        minimal_context.set_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT, True)
        # No DEEP_JUDGMENT_REASONING_ONLY artifact set

        assert stage.should_run(minimal_context) is True

    def test_should_run_when_reasoning_only_explicitly_false(
        self, stage: DeepJudgmentAutoFailStage, minimal_context: VerificationContext
    ):
        """Stage must still run when reasoning-only is explicitly False."""
        minimal_context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, True)
        minimal_context.set_artifact(ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS, ["drug_target"])
        minimal_context.set_artifact(ArtifactKeys.VERIFY_RESULT, True)
        minimal_context.set_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT, True)
        minimal_context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_REASONING_ONLY, False)

        assert stage.should_run(minimal_context) is True

    def test_should_not_run_when_deep_judgment_not_performed(
        self, stage: DeepJudgmentAutoFailStage, minimal_context: VerificationContext
    ):
        """Existing behavior: stage must not run when deep judgment was not performed."""
        minimal_context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, False)
        minimal_context.set_artifact(ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS, [])
        minimal_context.set_artifact(ArtifactKeys.VERIFY_RESULT, True)
        minimal_context.set_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT, True)

        assert stage.should_run(minimal_context) is False

    def test_should_not_run_when_no_attributes_without_excerpts(
        self, stage: DeepJudgmentAutoFailStage, minimal_context: VerificationContext
    ):
        """Existing behavior: stage must not run when all attributes have excerpts."""
        minimal_context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, True)
        minimal_context.set_artifact(ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS, [])
        minimal_context.set_artifact(ArtifactKeys.VERIFY_RESULT, True)
        minimal_context.set_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT, True)

        assert stage.should_run(minimal_context) is False

    def test_execute_still_works_when_not_reasoning_only(
        self, stage: DeepJudgmentAutoFailStage, minimal_context: VerificationContext
    ):
        """Execute must auto-fail when reasoning-only is not active and excerpts missing."""
        minimal_context.set_artifact(ArtifactKeys.DEEP_JUDGMENT_PERFORMED, True)
        minimal_context.set_artifact(ArtifactKeys.ATTRIBUTES_WITHOUT_EXCERPTS, ["drug_target"])
        minimal_context.set_artifact(ArtifactKeys.VERIFY_RESULT, True)
        minimal_context.set_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT, True)

        stage.execute(minimal_context)

        assert minimal_context.get_artifact(ArtifactKeys.VERIFY_RESULT) is False
        assert minimal_context.get_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT) is False


# =============================================================================
# Task 8: Parse template stage wires reasoning-only through deep_judgment_config
# =============================================================================


@pytest.mark.unit
class TestParseTemplateReasoningOnlyWiring:
    """Tests that ParseTemplateStage passes reasoning_only through to DJ config
    and stores the reasoning-only artifact after parse completes.
    """

    def test_deep_judgment_config_includes_reasoning_only_when_enabled(
        self, minimal_context: VerificationContext, sample_answer_class
    ):
        """When context has deep_judgment_reasoning_only=True, the config dict
        passed to evaluator.parse_response must include reasoning_only=True.
        """
        from unittest.mock import MagicMock, patch

        from karenina.benchmark.verification.evaluators.template.results import ParseResult
        from karenina.benchmark.verification.stages.pipeline.parse_template import (
            ParseTemplateStage,
        )

        minimal_context.deep_judgment_enabled = True
        minimal_context.deep_judgment_reasoning_only = True
        minimal_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "BCL-2 is the target.")
        minimal_context.set_artifact(ArtifactKeys.ANSWER, sample_answer_class)
        minimal_context.set_artifact(ArtifactKeys.RAW_ANSWER, sample_answer_class)
        minimal_context.set_artifact(ArtifactKeys.USAGE_TRACKER, MagicMock())

        mock_evaluator = MagicMock()
        success_result = ParseResult(
            success=True,
            parsed_answer=MagicMock(),
            deep_judgment_performed=True,
        )
        mock_evaluator.parse_response.return_value = success_result
        mock_evaluator.model_str = "test-model"

        stage = ParseTemplateStage()

        with patch(
            "karenina.benchmark.verification.stages.pipeline.parse_template.TemplateEvaluator",
            return_value=mock_evaluator,
        ):
            stage.execute(minimal_context)

        # Verify parse_response was called with deep_judgment_config containing reasoning_only
        call_kwargs = mock_evaluator.parse_response.call_args
        dj_config = call_kwargs.kwargs.get("deep_judgment_config", {})
        assert "reasoning_only" in dj_config
        assert dj_config["reasoning_only"] is True

    def test_deep_judgment_config_has_reasoning_only_false_when_disabled(
        self, minimal_context: VerificationContext, sample_answer_class
    ):
        """When context has deep_judgment_reasoning_only=False, config must include
        reasoning_only=False.
        """
        from unittest.mock import MagicMock, patch

        from karenina.benchmark.verification.evaluators.template.results import ParseResult
        from karenina.benchmark.verification.stages.pipeline.parse_template import (
            ParseTemplateStage,
        )

        minimal_context.deep_judgment_enabled = True
        minimal_context.deep_judgment_reasoning_only = False
        minimal_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "BCL-2 is the target.")
        minimal_context.set_artifact(ArtifactKeys.ANSWER, sample_answer_class)
        minimal_context.set_artifact(ArtifactKeys.RAW_ANSWER, sample_answer_class)
        minimal_context.set_artifact(ArtifactKeys.USAGE_TRACKER, MagicMock())

        mock_evaluator = MagicMock()
        success_result = ParseResult(
            success=True,
            parsed_answer=MagicMock(),
            deep_judgment_performed=True,
        )
        mock_evaluator.parse_response.return_value = success_result
        mock_evaluator.model_str = "test-model"

        stage = ParseTemplateStage()

        with patch(
            "karenina.benchmark.verification.stages.pipeline.parse_template.TemplateEvaluator",
            return_value=mock_evaluator,
        ):
            stage.execute(minimal_context)

        call_kwargs = mock_evaluator.parse_response.call_args
        dj_config = call_kwargs.kwargs.get("deep_judgment_config", {})
        assert "reasoning_only" in dj_config
        assert dj_config["reasoning_only"] is False

    def test_reasoning_only_artifact_stored_when_metadata_indicates_reasoning_only(
        self, minimal_context: VerificationContext, sample_answer_class
    ):
        """When dj_metadata indicates reasoning_only, the artifact must be stored."""
        from unittest.mock import MagicMock, patch

        from karenina.benchmark.verification.evaluators.template.results import ParseResult
        from karenina.benchmark.verification.stages.pipeline.parse_template import (
            ParseTemplateStage,
        )

        minimal_context.deep_judgment_enabled = True
        minimal_context.deep_judgment_reasoning_only = True
        minimal_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "BCL-2 is the target.")
        minimal_context.set_artifact(ArtifactKeys.ANSWER, sample_answer_class)
        minimal_context.set_artifact(ArtifactKeys.RAW_ANSWER, sample_answer_class)
        minimal_context.set_artifact(ArtifactKeys.USAGE_TRACKER, MagicMock())

        # The parse result simulates reasoning-only metadata
        # The stage reads deep_judgment_stages_completed from the result
        # and we need the stage to store the reasoning-only artifact
        mock_evaluator = MagicMock()
        success_result = ParseResult(
            success=True,
            parsed_answer=MagicMock(),
            deep_judgment_performed=True,
            deep_judgment_stages_completed=["reasoning", "parameters"],
            attributes_without_excerpts=[],
            extracted_excerpts={},
            attribute_reasoning={"drug_target": "Some reasoning"},
        )
        mock_evaluator.parse_response.return_value = success_result
        mock_evaluator.model_str = "test-model"

        stage = ParseTemplateStage()

        with patch(
            "karenina.benchmark.verification.stages.pipeline.parse_template.TemplateEvaluator",
            return_value=mock_evaluator,
        ):
            stage.execute(minimal_context)

        assert minimal_context.get_artifact(ArtifactKeys.DEEP_JUDGMENT_REASONING_ONLY) is True

    def test_reasoning_only_artifact_not_stored_when_standard_deep_judgment(
        self, minimal_context: VerificationContext, sample_answer_class
    ):
        """When standard deep judgment is used, the reasoning-only artifact should not be True."""
        from unittest.mock import MagicMock, patch

        from karenina.benchmark.verification.evaluators.template.results import ParseResult
        from karenina.benchmark.verification.stages.pipeline.parse_template import (
            ParseTemplateStage,
        )

        minimal_context.deep_judgment_enabled = True
        minimal_context.deep_judgment_reasoning_only = False
        minimal_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "BCL-2 is the target.")
        minimal_context.set_artifact(ArtifactKeys.ANSWER, sample_answer_class)
        minimal_context.set_artifact(ArtifactKeys.RAW_ANSWER, sample_answer_class)
        minimal_context.set_artifact(ArtifactKeys.USAGE_TRACKER, MagicMock())

        mock_evaluator = MagicMock()
        success_result = ParseResult(
            success=True,
            parsed_answer=MagicMock(),
            deep_judgment_performed=True,
            deep_judgment_stages_completed=["excerpts", "reasoning", "parameters"],
            attributes_without_excerpts=[],
        )
        mock_evaluator.parse_response.return_value = success_result
        mock_evaluator.model_str = "test-model"

        stage = ParseTemplateStage()

        with patch(
            "karenina.benchmark.verification.stages.pipeline.parse_template.TemplateEvaluator",
            return_value=mock_evaluator,
        ):
            stage.execute(minimal_context)

        # Should not be True (either absent or False)
        assert minimal_context.get_artifact(ArtifactKeys.DEEP_JUDGMENT_REASONING_ONLY) is not True


# =============================================================================
# Task 8: TemplateEvaluator passes reasoning_only to VerificationConfig
# =============================================================================


@pytest.mark.unit
class TestTemplateEvaluatorReasoningOnlyConfig:
    """Tests that TemplateEvaluator passes reasoning_only to the DJ VerificationConfig."""

    def test_reasoning_only_passed_to_dj_config(self):
        """When deep_judgment_config includes reasoning_only=True, the
        VerificationConfig created for DJ must have deep_judgment_reasoning_only=True.
        """
        from unittest.mock import MagicMock, patch

        from karenina.benchmark.verification.evaluators.template.evaluator import TemplateEvaluator

        model_config = ModelConfig(
            id="test-model",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            temperature=0.0,
        )

        class TestAnswer(BaseAnswer):
            drug_target: str = Field(default="", description="The protein target")

        with (
            patch("karenina.benchmark.verification.evaluators.template.evaluator.get_llm") as mock_get_llm,
            patch("karenina.benchmark.verification.evaluators.template.evaluator.get_parser") as mock_get_parser,
        ):
            mock_get_llm.return_value = MagicMock()
            mock_parser = MagicMock()
            mock_parser.capabilities = MagicMock()
            mock_get_parser.return_value = mock_parser

            evaluator = TemplateEvaluator(
                model_config=model_config,
                answer_class=TestAnswer,
            )

            # Patch deep_judgment_parse at the import location inside the method
            with patch(
                "karenina.benchmark.verification.evaluators.template.deep_judgment.deep_judgment_parse"
            ) as mock_dj:
                mock_dj.return_value = (
                    TestAnswer(drug_target="BCL-2"),
                    {},
                    {"drug_target": "reasoning"},
                    {"stages_completed": ["reasoning", "parameters"], "reasoning_only": True, "model_calls": 2},
                )

                evaluator.parse_response(
                    raw_response="BCL-2 is the target.",
                    question_text="What is the drug target?",
                    deep_judgment_enabled=True,
                    deep_judgment_config={"reasoning_only": True},
                )

                # Verify that deep_judgment_parse was called with a config that has reasoning_only
                assert mock_dj.called
                call_kwargs = mock_dj.call_args.kwargs
                config = call_kwargs.get("config")
                assert config is not None
                assert config.deep_judgment_reasoning_only is True

    def test_reasoning_only_false_when_not_in_config(self):
        """When deep_judgment_config does not include reasoning_only, the config
        must default to deep_judgment_reasoning_only=False.
        """
        from unittest.mock import MagicMock, patch

        from karenina.benchmark.verification.evaluators.template.evaluator import TemplateEvaluator

        model_config = ModelConfig(
            id="test-model",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            temperature=0.0,
        )

        class TestAnswer(BaseAnswer):
            drug_target: str = Field(default="", description="The protein target")

        with (
            patch("karenina.benchmark.verification.evaluators.template.evaluator.get_llm") as mock_get_llm,
            patch("karenina.benchmark.verification.evaluators.template.evaluator.get_parser") as mock_get_parser,
        ):
            mock_get_llm.return_value = MagicMock()
            mock_parser = MagicMock()
            mock_parser.capabilities = MagicMock()
            mock_get_parser.return_value = mock_parser

            evaluator = TemplateEvaluator(
                model_config=model_config,
                answer_class=TestAnswer,
            )

            with patch(
                "karenina.benchmark.verification.evaluators.template.deep_judgment.deep_judgment_parse"
            ) as mock_dj:
                mock_dj.return_value = (
                    TestAnswer(drug_target="BCL-2"),
                    {"drug_target": [{"text": "BCL-2", "confidence": "high"}]},
                    {"drug_target": "reasoning"},
                    {"stages_completed": ["excerpts", "reasoning", "parameters"], "model_calls": 3},
                )

                evaluator.parse_response(
                    raw_response="BCL-2 is the target.",
                    question_text="What is the drug target?",
                    deep_judgment_enabled=True,
                    deep_judgment_config={},
                )

                config = mock_dj.call_args.kwargs.get("config")
                assert config is not None
                assert config.deep_judgment_reasoning_only is False
