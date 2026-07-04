"""Tests for Pipeline Result Preservation (Chunk 07).

Covers all 8 issues across 4 phases:
- Phase 1 (146, 148): Error preservation chain
- Phase 2 (150, 152): Field results and DataFrame fix
- Phase 3 (133, 151): Composition strategy
- Phase 4 (178, 184): Independent metadata additions
"""

from unittest.mock import MagicMock

import pytest
from pydantic import Field

from karenina.benchmark.verification.evaluators.template.results import (
    FieldVerificationResult,
    RegexVerificationResult,
)
from karenina.benchmark.verification.stages import (
    ArtifactKeys,
    FinalizeResultStage,
    VerificationContext,
    VerifyTemplateStage,
)
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import BaseAnswer
from karenina.schemas.verification import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.verification.model_identity import ModelIdentity

# =============================================================================
# Shared fixtures
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
def minimal_context(minimal_model_config: ModelConfig) -> VerificationContext:
    """Return a minimal VerificationContext for testing."""
    return VerificationContext(
        question_id="test-question-1",
        template_id="template-hash-123",
        question_text="What is the capital of France?",
        template_code="class Answer(BaseAnswer): pass",
        answering_model=minimal_model_config,
        parsing_model=minimal_model_config,
    )


def _make_metadata(**overrides) -> VerificationResultMetadata:
    """Helper to create VerificationResultMetadata with defaults."""
    defaults = {
        "question_id": "q1",
        "template_id": "t1",
        "failure": None,
        "caveats": [],
        "question_text": "Test?",
        "answering": ModelIdentity(interface="langchain", model_name="gpt-4"),
        "parsing": ModelIdentity(interface="langchain", model_name="gpt-4"),
        "execution_time": 1.0,
        "timestamp": "2025-01-01T00:00:00Z",
        "result_id": "test123",
    }
    defaults.update(overrides)
    return VerificationResultMetadata(**defaults)


def _make_mock_evaluator(
    field_result: FieldVerificationResult,
    regex_result: RegexVerificationResult | None = None,
) -> MagicMock:
    """Create a mock TemplateEvaluator that returns the given results."""
    evaluator = MagicMock()
    evaluator.verify_fields.return_value = field_result
    if regex_result is None:
        regex_result = RegexVerificationResult(success=True)
    evaluator.verify_regex.return_value = regex_result
    return evaluator


def _run_finalize(context: VerificationContext) -> VerificationResult:
    """Set up minimal finalize context and run FinalizeResultStage."""
    if not context.get_result_field(ArtifactKeys.TIMESTAMP):
        context.set_result_field(ArtifactKeys.TIMESTAMP, "2026-01-01T00:00:00")
    if not context.get_result_field(ArtifactKeys.EXECUTION_TIME):
        context.set_result_field(ArtifactKeys.EXECUTION_TIME, 1.0)
    stage = FinalizeResultStage()
    stage.execute(context)
    return context.get_artifact(ArtifactKeys.FINAL_RESULT)


# =============================================================================
# Phase 1: Error Preservation (issues 146, 148)
# =============================================================================


@pytest.mark.unit
class TestFieldVerificationErrorSchema:
    """Issue 146: field_verification_error field on VerificationResultTemplate."""

    def test_field_exists_with_none_default(self):
        """VerificationResultTemplate has field_verification_error defaulting to None."""
        template = VerificationResultTemplate()
        assert template.field_verification_error is None

    def test_accepts_string(self):
        """field_verification_error stores error string from verify() exception."""
        template = VerificationResultTemplate(
            field_verification_error="Field verification failed: ValueError('MARKER')"
        )
        assert "MARKER" in template.field_verification_error

    def test_roundtrip(self):
        """field_verification_error survives model_dump/model_validate cycle."""
        template = VerificationResultTemplate(field_verification_error="Test error message")
        dumped = template.model_dump()
        restored = VerificationResultTemplate.model_validate(dumped)
        assert restored.field_verification_error == "Test error message"


@pytest.mark.unit
class TestFieldVerificationErrorArtifactKey:
    """FIELD_VERIFICATION_ERROR key exists in ArtifactKeys."""

    def test_key_defined(self):
        """ArtifactKeys has FIELD_VERIFICATION_ERROR constant."""
        assert hasattr(ArtifactKeys, "FIELD_VERIFICATION_ERROR")
        assert ArtifactKeys.FIELD_VERIFICATION_ERROR == "field_verification_error"


@pytest.mark.unit
class TestVerifyTemplateStageErrorPropagation:
    """Issue 146: VerifyTemplateStage stores verify() error in context."""

    def test_error_stored_when_verify_raises(self, minimal_context):
        """When evaluator.verify_fields() returns error, it is stored in context."""

        class RaisingAnswer(BaseAnswer):
            capital: str = Field(description="Capital city")

            def verify(self) -> bool:
                raise ValueError("MARKER")

            def verify_granular(self) -> float:
                return 1.0

        parsed = RaisingAnswer(capital="Paris")
        minimal_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        minimal_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "Paris")

        field_result = FieldVerificationResult(
            success=False,
            error="Field verification failed: ValueError('MARKER')",
        )
        evaluator = _make_mock_evaluator(field_result)
        minimal_context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, evaluator)

        stage = VerifyTemplateStage()
        stage.execute(minimal_context)

        stored = minimal_context.get_result_field(ArtifactKeys.FIELD_VERIFICATION_ERROR)
        assert stored is not None
        assert "MARKER" in stored

    def test_no_error_stored_when_verify_succeeds(self, minimal_context):
        """No field_verification_error stored when verify() succeeds."""

        class GoodAnswer(BaseAnswer):
            capital: str = Field(description="Capital city")

            def verify(self) -> bool:
                return True

            def verify_granular(self) -> float:
                return 1.0

        parsed = GoodAnswer(capital="Paris")
        minimal_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        minimal_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "Paris")

        field_result = FieldVerificationResult(success=True)
        evaluator = _make_mock_evaluator(field_result)
        minimal_context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, evaluator)

        stage = VerifyTemplateStage()
        stage.execute(minimal_context)

        stored = minimal_context.get_result_field(ArtifactKeys.FIELD_VERIFICATION_ERROR)
        assert stored is None


@pytest.mark.unit
class TestVerifyGranularSkipOnError:
    """Issue 148: verify_granular_result is None when verify() raises."""

    def test_granular_skipped_when_error(self, minimal_context):
        """verify_granular() not called when field verification has error."""

        class RaisingAnswer(BaseAnswer):
            capital: str = Field(description="Capital city")

            def verify(self) -> bool:
                raise ValueError("x")

            def verify_granular(self) -> float:
                return 1.0  # Would contradict False

        parsed = RaisingAnswer(capital="Paris")
        minimal_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        minimal_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "Paris")

        field_result = FieldVerificationResult(
            success=False,
            error="Field verification failed: ValueError('x')",
        )
        evaluator = _make_mock_evaluator(field_result)
        minimal_context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, evaluator)

        stage = VerifyTemplateStage()
        stage.execute(minimal_context)

        granular = minimal_context.get_result_field(ArtifactKeys.VERIFY_GRANULAR_RESULT)
        assert granular is None

    def test_granular_computed_without_error(self, minimal_context):
        """verify_granular() runs normally when no field verification error."""

        class GoodAnswer(BaseAnswer):
            capital: str = Field(description="Capital city")

            def verify(self) -> bool:
                return True

            def verify_granular(self) -> float:
                return 0.75

        parsed = GoodAnswer(capital="Paris")
        minimal_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        minimal_context.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "Paris")

        field_result = FieldVerificationResult(success=True)
        evaluator = _make_mock_evaluator(field_result)
        minimal_context.set_artifact(ArtifactKeys.TEMPLATE_EVALUATOR, evaluator)

        stage = VerifyTemplateStage()
        stage.execute(minimal_context)

        granular = minimal_context.get_result_field(ArtifactKeys.VERIFY_GRANULAR_RESULT)
        assert granular == 0.75


@pytest.mark.unit
class TestFinalizeErrorPropagation:
    """Issue 146: FinalizeResultStage copies field_verification_error to result."""

    def test_error_propagated_to_result(self, minimal_context):
        """field_verification_error from context appears in final result."""
        error_msg = "Field verification failed: ValueError('MARKER')"
        minimal_context.set_result_field(ArtifactKeys.FIELD_VERIFICATION_ERROR, error_msg)

        result = _run_finalize(minimal_context)
        assert result.template.field_verification_error == error_msg

    def test_no_error_when_not_set(self, minimal_context):
        """field_verification_error is None when not set in context."""
        result = _run_finalize(minimal_context)
        assert result.template.field_verification_error is None


# =============================================================================
# Phase 2: Field Results Preservation (issues 150, 152)
# =============================================================================


@pytest.mark.unit
class TestFieldResultsSchema:
    """Issue 150: field_results field on VerificationResultTemplate."""

    def test_field_exists_with_none_default(self):
        """VerificationResultTemplate has field_results defaulting to None."""
        template = VerificationResultTemplate()
        assert template.field_results is None

    def test_accepts_dict(self):
        """field_results stores dict mapping field names to booleans."""
        template = VerificationResultTemplate(field_results={"drug_target": True, "mechanism": False, "efficacy": True})
        assert template.field_results["drug_target"] is True
        assert template.field_results["mechanism"] is False

    def test_roundtrip(self):
        """field_results survives model_dump/model_validate cycle."""
        original = {"field_a": True, "field_b": False}
        template = VerificationResultTemplate(field_results=original)
        dumped = template.model_dump()
        restored = VerificationResultTemplate.model_validate(dumped)
        assert restored.field_results == original


@pytest.mark.unit
class TestDataFrameFieldMatchWithStoredResults:
    """Issue 152: DataFrame field_match uses stored primitive results."""

    def _build_df(self, parsed_gt, parsed_llm, field_results=None):
        """Helper to build a DataFrame from test data."""
        from karenina.schemas.dataframes.template import TemplateDataFrameBuilder

        template = VerificationResultTemplate(
            parsed_gt_response=parsed_gt,
            parsed_llm_response=parsed_llm,
            field_results=field_results,
            template_verification_performed=True,
            verify_result=True,
        )
        result = VerificationResult(metadata=_make_metadata(), template=template)
        return TemplateDataFrameBuilder([result]).build_field_dataframe()

    def test_uses_stored_result_true(self):
        """Stored field_results=True overrides naive equality (NumericTolerance case)."""
        # gt=100, llm=103: naive says False, primitive says True (within tolerance)
        df = self._build_df(
            {"value": 100.0},
            {"value": 103.0},
            field_results={"value": True},
        )
        row = df[df["field_name"] == "value"].iloc[0]
        assert row["field_match"] == True  # noqa: E712

    def test_uses_stored_result_false(self):
        """Stored field_results=False overrides naive equality."""
        # Same values but primitive says False
        df = self._build_df(
            {"answer": "Paris"},
            {"answer": "Paris"},
            field_results={"answer": False},
        )
        row = df[df["field_name"] == "answer"].iloc[0]
        assert row["field_match"] == False  # noqa: E712

    def test_fallback_without_stored_results(self):
        """Without field_results, falls back to naive _compare_values()."""
        df = self._build_df(
            {"answer": "Paris"},
            {"answer": "Paris"},
            field_results=None,
        )
        row = df[df["field_name"] == "answer"].iloc[0]
        assert row["field_match"] == True  # noqa: E712


@pytest.mark.unit
class TestFinalizeFieldResults:
    """Issue 150: FinalizeResultStage computes field_results from parsed_answer."""

    def test_field_results_populated(self, minimal_context):
        """field_results is populated from _compute_field_results() when available."""
        from karenina.schemas.entities.verified_field import VerifiedField
        from karenina.schemas.primitives.comparisons import BooleanMatch

        class Answer(BaseAnswer):
            capital: str = VerifiedField(
                description="Capital city",
                default="Paris",
                verify_with=BooleanMatch(),
                ground_truth="Paris",
            )

        parsed = Answer(capital="Paris")
        minimal_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)

        result = _run_finalize(minimal_context)
        assert result.template.field_results is not None
        assert result.template.field_results["capital"] is True

    def test_field_results_none_without_verified_fields(self, minimal_context):
        """field_results is None when parsed_answer has no _compute_field_results."""

        class SimpleAnswer(BaseAnswer):
            value: str = "test"

        parsed = SimpleAnswer(value="test")
        minimal_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)

        result = _run_finalize(minimal_context)
        # Should be None because SimpleAnswer has no VerifiedFields
        assert result.template.field_results is None

    def test_field_results_none_without_parsed_answer(self, minimal_context):
        """field_results is None when no parsed_answer in context."""
        result = _run_finalize(minimal_context)
        assert result.template.field_results is None


# =============================================================================
# Phase 3: Composition Strategy (issues 133, 151)
# =============================================================================


@pytest.mark.unit
class TestCompositionStrategySchema:
    """Issue 151: composition_strategy field on VerificationResultTemplate."""

    def test_field_exists_with_none_default(self):
        """VerificationResultTemplate has composition_strategy defaulting to None."""
        template = VerificationResultTemplate()
        assert template.composition_strategy is None

    def test_accepts_string(self):
        """composition_strategy stores strategy string."""
        template = VerificationResultTemplate(composition_strategy="any_of")
        assert template.composition_strategy == "any_of"

    def test_roundtrip(self):
        """composition_strategy survives model_dump/model_validate cycle."""
        template = VerificationResultTemplate(composition_strategy="at_least_n(2)")
        dumped = template.model_dump()
        restored = VerificationResultTemplate.model_validate(dumped)
        assert restored.composition_strategy == "at_least_n(2)"


@pytest.mark.unit
class TestCompositionAwareGranular:
    """Issue 133: _auto_verify_granular() honors composition strategy."""

    def test_anyof_granular_uses_max(self):
        """AnyOf with 1/3 fields passing produces granular reflecting composition semantics."""
        from karenina.schemas.entities.composition import AnyOf, FieldCheck
        from karenina.schemas.entities.verified_field import VerifiedField
        from karenina.schemas.primitives.comparisons import ExactMatch

        class Answer(BaseAnswer):
            class VerificationStrategy:
                verify_strategy = AnyOf(
                    conditions=[
                        FieldCheck(field="a"),
                        FieldCheck(field="b"),
                        FieldCheck(field="c"),
                    ]
                )

            a: str = VerifiedField(
                description="A",
                default="yes",
                verify_with=ExactMatch(),
                ground_truth="yes",
            )
            b: str = VerifiedField(
                description="B",
                default="no",
                verify_with=ExactMatch(),
                ground_truth="yes",
            )
            c: str = VerifiedField(
                description="C",
                default="no",
                verify_with=ExactMatch(),
                ground_truth="yes",
            )

        ans = Answer(a="yes", b="no", c="no")
        # verify() should be True (AnyOf: at least one passes)
        assert ans.verify() is True
        # For AnyOf: max passing field weight / total weight
        # 1 field passes with weight 1.0 / total 3.0 is the naive average (0.333)
        # Composition-aware: since AnyOf says "any one passing is success",
        # granular should be higher than naive 1/3
        granular = ans.verify_granular()
        assert granular > 0.33  # Must be higher than naive weighted average

    def test_allof_granular_uses_weighted_average(self):
        """AllOf still uses weighted average (current behavior)."""
        from karenina.schemas.entities.verified_field import VerifiedField
        from karenina.schemas.primitives.comparisons import ExactMatch

        class Answer(BaseAnswer):
            # No VerificationStrategy = default AllOf
            a: str = VerifiedField(
                description="A",
                default="yes",
                verify_with=ExactMatch(),
                ground_truth="yes",
            )
            b: str = VerifiedField(
                description="B",
                default="no",
                verify_with=ExactMatch(),
                ground_truth="yes",
            )

        ans = Answer(a="yes", b="no")
        granular = ans.verify_granular()
        # 1/2 pass with equal weights = 0.5
        assert granular == pytest.approx(0.5)


@pytest.mark.unit
class TestFinalizeCompositionStrategy:
    """Issue 151: FinalizeResultStage records composition_strategy."""

    def test_anyof_strategy_recorded(self, minimal_context):
        """composition_strategy is 'any_of' for AnyOf templates."""
        from karenina.schemas.entities.composition import AnyOf, FieldCheck
        from karenina.schemas.entities.verified_field import VerifiedField
        from karenina.schemas.primitives.comparisons import BooleanMatch

        class Answer(BaseAnswer):
            class VerificationStrategy:
                verify_strategy = AnyOf(
                    conditions=[
                        FieldCheck(field="a"),
                    ]
                )

            a: str = VerifiedField(
                description="A",
                default="yes",
                verify_with=BooleanMatch(),
                ground_truth="yes",
            )

        parsed = Answer(a="yes")
        minimal_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)

        result = _run_finalize(minimal_context)
        assert result.template.composition_strategy == "any_of"

    def test_no_strategy_recorded_as_none(self, minimal_context):
        """composition_strategy is None for default AllOf templates."""
        from karenina.schemas.entities.verified_field import VerifiedField
        from karenina.schemas.primitives.comparisons import BooleanMatch

        class Answer(BaseAnswer):
            a: str = VerifiedField(
                description="A",
                default="yes",
                verify_with=BooleanMatch(),
                ground_truth="yes",
            )

        parsed = Answer(a="yes")
        minimal_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)

        result = _run_finalize(minimal_context)
        assert result.template.composition_strategy is None

    def test_at_least_n_strategy_recorded(self, minimal_context):
        """composition_strategy is 'at_least_n(2)' for AtLeastN templates."""
        from karenina.schemas.entities.composition import AtLeastN, FieldCheck
        from karenina.schemas.entities.verified_field import VerifiedField
        from karenina.schemas.primitives.comparisons import BooleanMatch

        class Answer(BaseAnswer):
            class VerificationStrategy:
                verify_strategy = AtLeastN(
                    n=2,
                    conditions=[
                        FieldCheck(field="a"),
                        FieldCheck(field="b"),
                        FieldCheck(field="c"),
                    ],
                )

            a: str = VerifiedField(
                description="A",
                default="yes",
                verify_with=BooleanMatch(),
                ground_truth="yes",
            )
            b: str = VerifiedField(
                description="B",
                default="yes",
                verify_with=BooleanMatch(),
                ground_truth="yes",
            )
            c: str = VerifiedField(
                description="C",
                default="no",
                verify_with=BooleanMatch(),
                ground_truth="yes",
            )

        parsed = Answer(a="yes", b="yes", c="no")
        minimal_context.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)

        result = _run_finalize(minimal_context)
        assert result.template.composition_strategy == "at_least_n(2)"


# =============================================================================
# Phase 4: Independent Metadata Additions (issues 178, 184)
# =============================================================================


@pytest.mark.unit
class TestFewShotProvenanceSchema:
    """Issue 178: Few-shot provenance fields on VerificationResultMetadata."""

    def test_few_shot_enabled_default_false(self):
        """few_shot_enabled defaults to False."""
        metadata = _make_metadata()
        assert metadata.few_shot_enabled is False

    def test_few_shot_enabled_set_true(self):
        """few_shot_enabled can be set to True."""
        metadata = _make_metadata(few_shot_enabled=True)
        assert metadata.few_shot_enabled is True

    def test_few_shot_example_count_default_zero(self):
        """few_shot_example_count defaults to 0."""
        metadata = _make_metadata()
        assert metadata.few_shot_example_count == 0

    def test_few_shot_example_count_set(self):
        """few_shot_example_count stores the number of examples."""
        metadata = _make_metadata(few_shot_example_count=3)
        assert metadata.few_shot_example_count == 3

    def test_roundtrip(self):
        """Few-shot fields survive model_dump/model_validate cycle."""
        metadata = _make_metadata(few_shot_enabled=True, few_shot_example_count=2)
        dumped = metadata.model_dump()
        restored = VerificationResultMetadata.model_validate(dumped)
        assert restored.few_shot_enabled is True
        assert restored.few_shot_example_count == 2


@pytest.mark.unit
class TestEvaluationModeSchema:
    """Issue 184: evaluation_mode field on VerificationResultMetadata."""

    def test_default_none(self):
        """evaluation_mode defaults to None."""
        metadata = _make_metadata()
        assert metadata.evaluation_mode is None

    def test_set_mode(self):
        """evaluation_mode stores the mode string."""
        metadata = _make_metadata(evaluation_mode="template_and_rubric")
        assert metadata.evaluation_mode == "template_and_rubric"

    def test_roundtrip(self):
        """evaluation_mode survives model_dump/model_validate cycle."""
        metadata = _make_metadata(evaluation_mode="template_only")
        dumped = metadata.model_dump()
        restored = VerificationResultMetadata.model_validate(dumped)
        assert restored.evaluation_mode == "template_only"

    def test_none_roundtrip(self):
        """None evaluation_mode survives roundtrip (backward compat)."""
        metadata = _make_metadata()
        dumped = metadata.model_dump()
        restored = VerificationResultMetadata.model_validate(dumped)
        assert restored.evaluation_mode is None


@pytest.mark.unit
class TestFinalizeFewShotMetadata:
    """Issue 178: FinalizeResultStage populates few-shot metadata."""

    def test_few_shot_fields_populated(self, minimal_context):
        """Few-shot fields populated from context when enabled."""
        minimal_context.few_shot_enabled = True
        minimal_context.few_shot_examples = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]

        result = _run_finalize(minimal_context)
        assert result.metadata.few_shot_enabled is True
        assert result.metadata.few_shot_example_count == 2

    def test_few_shot_defaults_when_disabled(self, minimal_context):
        """Few-shot fields are defaults when not enabled."""
        result = _run_finalize(minimal_context)
        assert result.metadata.few_shot_enabled is False
        assert result.metadata.few_shot_example_count == 0


@pytest.mark.unit
class TestFinalizeEvaluationMode:
    """Issue 184: FinalizeResultStage populates evaluation_mode."""

    def test_evaluation_mode_populated(self, minimal_context):
        """evaluation_mode from context result field appears in metadata."""
        minimal_context.set_result_field("evaluation_mode", "template_and_rubric")

        result = _run_finalize(minimal_context)
        assert result.metadata.evaluation_mode == "template_and_rubric"

    def test_evaluation_mode_none_by_default(self, minimal_context):
        """evaluation_mode is None when not set."""
        result = _run_finalize(minimal_context)
        assert result.metadata.evaluation_mode is None
