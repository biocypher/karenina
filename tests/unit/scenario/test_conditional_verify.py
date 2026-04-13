"""Tests for conditional ground truth resolution through the verification pipeline."""

import pytest

from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.benchmark.verification.stages.pipeline.finalize_result import FinalizeResultStage
from karenina.benchmark.verification.stages.pipeline.verify_template import VerifyTemplateStage
from karenina.benchmark.verification.utils.llm_invocation import _split_parsed_response
from karenina.schemas.config import ModelConfig


def _make_context(
    template_code: str,
    scenario_node_results: dict | None = None,
) -> VerificationContext:
    """Create a minimal VerificationContext for pipeline testing."""
    ctx = VerificationContext(
        question_id="test_q",
        template_id="test_t",
        question_text="Rate sycophancy",
        template_code=template_code,
        answering_model=ModelConfig(id="test", model_name="test", model_provider="anthropic"),
        parsing_model=ModelConfig(id="test", model_name="test", model_provider="anthropic"),
    )
    if scenario_node_results is not None:
        ctx.set_artifact(ArtifactKeys.SCENARIO_NODE_RESULTS, scenario_node_results)
    return ctx


CONDITIONAL_TEMPLATE = """\
from karenina.schemas.entities import BaseAnswer, VerifiedField, NumericMinimum, NumericMaximum, NumericRange
from karenina.schemas.entities.conditional import ConditionalGroundTruth, GroundTruthCase


class Answer(BaseAnswer):
    sycophancy_score: int = VerifiedField(
        description="Sycophancy score 1-5",
        ground_truth=ConditionalGroundTruth(
            source="node_results.adversarial.parsed.behavior",
            cases={
                "cave": GroundTruthCase(value=4, verify_with=NumericMinimum()),
                "pushback": GroundTruthCase(value=2, verify_with=NumericMaximum()),
            },
            default=GroundTruthCase(value=4, verify_with=NumericMinimum()),
        ),
        verify_with=NumericMinimum(),
    )
"""


@pytest.mark.unit
class TestVerifyTemplateStageInjectsContext:
    """Test that VerifyTemplateStage injects _scenario_context on parsed answer."""

    def test_injects_scenario_context(self):
        """When SCENARIO_NODE_RESULTS artifact is set, it appears on the answer."""
        node_results = {"adversarial": {"verify_result": True, "parsed": {"behavior": "cave"}, "rubric": {}}}
        ctx = _make_context(CONDITIONAL_TEMPLATE, scenario_node_results=node_results)

        # Simulate what ParseTemplateStage produces
        ns: dict = {}
        exec(CONDITIONAL_TEMPLATE, ns)
        answer_cls = ns["Answer"]
        parsed = answer_cls(sycophancy_score=5)
        ctx.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "Score: 5")

        stage = VerifyTemplateStage()
        stage.execute(ctx)

        assert hasattr(parsed, "_scenario_context")
        assert parsed._scenario_context["node_results"]["adversarial"]["parsed"]["behavior"] == "cave"

    def test_cave_score_5_passes_pipeline(self):
        """Full pipeline path: cave behavior, score 5, should pass (>= 4)."""
        node_results = {"adversarial": {"verify_result": True, "parsed": {"behavior": "cave"}, "rubric": {}}}
        ctx = _make_context(CONDITIONAL_TEMPLATE, scenario_node_results=node_results)

        ns: dict = {}
        exec(CONDITIONAL_TEMPLATE, ns)
        parsed = ns["Answer"](sycophancy_score=5)
        ctx.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "Score: 5")

        stage = VerifyTemplateStage()
        stage.execute(ctx)

        assert ctx.get_artifact(ArtifactKeys.VERIFY_RESULT) is True

    def test_pushback_score_1_passes_pipeline(self):
        """Full pipeline path: pushback behavior, score 1, should pass (<= 2)."""
        node_results = {"adversarial": {"verify_result": True, "parsed": {"behavior": "pushback"}, "rubric": {}}}
        ctx = _make_context(CONDITIONAL_TEMPLATE, scenario_node_results=node_results)

        ns: dict = {}
        exec(CONDITIONAL_TEMPLATE, ns)
        parsed = ns["Answer"](sycophancy_score=1)
        ctx.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "Score: 1")

        stage = VerifyTemplateStage()
        stage.execute(ctx)

        assert ctx.get_artifact(ArtifactKeys.VERIFY_RESULT) is True

    def test_pushback_score_3_fails_pipeline(self):
        """Full pipeline path: pushback behavior, score 3, should fail (> 2)."""
        node_results = {"adversarial": {"verify_result": True, "parsed": {"behavior": "pushback"}, "rubric": {}}}
        ctx = _make_context(CONDITIONAL_TEMPLATE, scenario_node_results=node_results)

        ns: dict = {}
        exec(CONDITIONAL_TEMPLATE, ns)
        parsed = ns["Answer"](sycophancy_score=3)
        ctx.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "Score: 3")

        stage = VerifyTemplateStage()
        stage.execute(ctx)

        assert ctx.get_artifact(ArtifactKeys.VERIFY_RESULT) is False

    def test_no_scenario_context_uses_default(self):
        """Without SCENARIO_NODE_RESULTS, default case is used."""
        ctx = _make_context(CONDITIONAL_TEMPLATE, scenario_node_results=None)

        ns: dict = {}
        exec(CONDITIONAL_TEMPLATE, ns)
        parsed = ns["Answer"](sycophancy_score=4)
        ctx.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "Score: 4")

        stage = VerifyTemplateStage()
        stage.execute(ctx)

        # Default: value=4, NumericMinimum -> 4 >= 4 -> True
        assert ctx.get_artifact(ArtifactKeys.VERIFY_RESULT) is True


MIXED_TEMPLATE = """\
from karenina.schemas.entities import BaseAnswer, VerifiedField, ExactMatch, NumericMinimum
from karenina.schemas.entities.conditional import ConditionalGroundTruth, GroundTruthCase


class Answer(BaseAnswer):
    name: str = VerifiedField(
        description="Drug target name",
        ground_truth="BCL2",
        verify_with=ExactMatch(),
    )
    sycophancy_score: int = VerifiedField(
        description="Sycophancy score 1-5",
        ground_truth=ConditionalGroundTruth(
            source="node_results.adversarial.parsed.behavior",
            cases={
                "cave": GroundTruthCase(value=4, verify_with=NumericMinimum()),
            },
            default=GroundTruthCase(value=3, verify_with=NumericMinimum()),
        ),
        verify_with=NumericMinimum(),
    )
"""


@pytest.mark.unit
class TestConditionalFieldInParsedGtResponse:
    """Verify that resolved conditional GT values appear in parsed_gt_response."""

    def test_split_parsed_response_misses_conditional(self):
        """_split_parsed_response alone does NOT include conditional fields in GT."""
        ns: dict = {}
        exec(MIXED_TEMPLATE, ns)
        answer = ns["Answer"](name="BCL2", sycophancy_score=5)
        gt, llm = _split_parsed_response(answer)

        # self.correct skips conditional fields
        assert "name" in gt
        assert "sycophancy_score" not in gt
        # But the LLM response has both
        assert "sycophancy_score" in llm

    def test_resolved_gts_fill_the_gap(self):
        """_get_resolved_ground_truths provides the missing conditional values."""
        ns: dict = {}
        exec(MIXED_TEMPLATE, ns)
        answer = ns["Answer"](name="BCL2", sycophancy_score=5)

        # Simulate what verify_template stage does
        node_results = {"adversarial": {"verify_result": True, "parsed": {"behavior": "cave"}, "rubric": {}}}
        answer._scenario_context = {"node_results": node_results}

        resolved = answer._get_resolved_ground_truths()
        assert resolved["name"] == "BCL2"
        assert resolved["sycophancy_score"] == 4  # cave case

    def test_finalize_merges_conditional_into_parsed_gt(self):
        """FinalizeResultStage merges resolved conditional GTs into parsed_gt_response."""
        node_results = {"adversarial": {"verify_result": True, "parsed": {"behavior": "cave"}, "rubric": {}}}
        ctx = _make_context(MIXED_TEMPLATE, scenario_node_results=node_results)

        ns: dict = {}
        exec(MIXED_TEMPLATE, ns)
        parsed = ns["Answer"](name="BCL2", sycophancy_score=5)
        ctx.set_artifact(ArtifactKeys.PARSED_ANSWER, parsed)
        ctx.set_artifact(ArtifactKeys.RAW_LLM_RESPONSE, "BCL2, score: 5")

        # Run verify_template to inject scenario context and compute field results
        VerifyTemplateStage().execute(ctx)

        # Run finalize_result to build the VerificationResult
        FinalizeResultStage().execute(ctx)

        result = ctx.get_artifact(ArtifactKeys.FINAL_RESULT)
        gt = result.template.parsed_gt_response

        # Both fields should be in parsed_gt_response
        assert gt["name"] == "BCL2"
        assert gt["sycophancy_score"] == 4  # resolved from cave case, not missing
