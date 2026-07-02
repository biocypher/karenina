"""Tests that downstream stages skip when RESPONSE_TIMEOUT_PARTIAL is True."""

from unittest.mock import MagicMock

import pytest

from karenina.benchmark.verification.stages.core.base import ArtifactKeys
from karenina.benchmark.verification.stages.pipeline.abstention_check import AbstentionCheckStage
from karenina.benchmark.verification.stages.pipeline.agentic_parse_template import AgenticParseTemplateStage
from karenina.benchmark.verification.stages.pipeline.parse_template import ParseTemplateStage
from karenina.benchmark.verification.stages.pipeline.sufficiency_check import SufficiencyCheckStage
from karenina.benchmark.verification.stages.pipeline.verify_template import VerifyTemplateStage

GUARDED_STAGES = [
    AbstentionCheckStage,
    SufficiencyCheckStage,
    ParseTemplateStage,
    AgenticParseTemplateStage,
    VerifyTemplateStage,
]

PARTIAL_SCORING_STAGES = [
    ParseTemplateStage,
    AgenticParseTemplateStage,
    VerifyTemplateStage,
]


@pytest.fixture
def mock_stage_context():
    """Minimal mock of VerificationContext for should_run testing.

    The mock must satisfy BaseVerificationStage.should_run(), which checks
    ``not context.error``. Setting error to None makes that check pass,
    allowing the stage's own guards (including the timeout guard) to run.
    """
    ctx = MagicMock()
    ctx.error = None
    ctx.get_artifact = MagicMock(return_value=None)
    ctx.get_result_field = MagicMock(return_value=None)
    ctx.has_artifact = MagicMock(return_value=True)
    # Stage-specific feature flags checked after the timeout guard
    ctx.abstention_enabled = True
    ctx.sufficiency_enabled = True
    ctx.agentic_parsing = True
    ctx.allow_partial_trace_scoring = False
    ctx.can_score_partial_timeout = MagicMock(return_value=False)
    return ctx


@pytest.mark.unit
class TestTimeoutPartialGuards:
    """Verify that RESPONSE_TIMEOUT_PARTIAL causes stages to skip."""

    @pytest.mark.parametrize("stage_cls", GUARDED_STAGES, ids=lambda c: c.__name__)
    def test_skips_when_partial_true(self, stage_cls, mock_stage_context):
        """Stage should_run returns False when response was truncated."""

        def get_artifact_side_effect(key, default=None):
            if key == ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL:
                return True
            return default

        mock_stage_context.get_artifact = MagicMock(side_effect=get_artifact_side_effect)

        stage = stage_cls()
        result = stage.should_run(mock_stage_context)
        assert result is False

    @pytest.mark.parametrize("stage_cls", GUARDED_STAGES, ids=lambda c: c.__name__)
    def test_does_not_skip_when_partial_false(self, stage_cls, mock_stage_context):
        """A False RESPONSE_TIMEOUT_PARTIAL must never trigger the timeout guard.

        The timeout guard is the *only* should_run clause that keys on
        RESPONSE_TIMEOUT_PARTIAL; when that artifact is False the guard is a
        no-op and should_run's verdict must come from the remaining
        feature-flag clauses. To prove the guard itself didn't force a skip,
        we compare against a baseline context where the artifact is absent
        (defaults to None): the two verdicts must be identical. A regression
        that, e.g., flipped the guard to skip on ``is not True`` would make
        the two diverge.
        """
        # Baseline: artifact absent (get_artifact returns the MagicMock default
        # for the key, which is falsy in the same way False is for the guard).
        baseline_ctx = MagicMock()
        baseline_ctx.error = None
        baseline_ctx.get_artifact = MagicMock(return_value=None)
        baseline_ctx.get_result_field = MagicMock(return_value=None)
        baseline_ctx.has_artifact = MagicMock(return_value=True)
        baseline_ctx.abstention_enabled = True
        baseline_ctx.sufficiency_enabled = True
        baseline_ctx.agentic_parsing = True
        baseline_ctx.allow_partial_trace_scoring = False
        baseline_ctx.can_score_partial_timeout = MagicMock(return_value=False)
        baseline_verdict = stage_cls().should_run(baseline_ctx)

        # Mutated: RESPONSE_TIMEOUT_PARTIAL explicitly False.
        def get_artifact_side_effect(key, default=None):
            if key == ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL:
                return False
            return default

        mock_stage_context.get_artifact = MagicMock(side_effect=get_artifact_side_effect)
        partial_false_verdict = stage_cls().should_run(mock_stage_context)

        assert partial_false_verdict == baseline_verdict, (
            f"{stage_cls.__name__}: should_run verdict changed when "
            "RESPONSE_TIMEOUT_PARTIAL went from absent to False; the timeout "
            "guard must be a no-op in that case."
        )

    @pytest.mark.parametrize("stage_cls", PARTIAL_SCORING_STAGES, ids=lambda c: c.__name__)
    def test_partial_timeout_scoring_override_allows_template_stages(self, stage_cls, mock_stage_context):
        """Template parsing/scoring stages can opt into partial timeout scoring."""

        def get_artifact_side_effect(key, default=None):
            if key == ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL:
                return True
            return default

        mock_stage_context.error = "Agent timed out with partial trace"
        mock_stage_context.allow_partial_trace_scoring = True
        mock_stage_context.can_score_partial_timeout = MagicMock(return_value=True)
        mock_stage_context.get_artifact = MagicMock(side_effect=get_artifact_side_effect)

        stage = stage_cls()
        result = stage.should_run(mock_stage_context)
        assert result is True
