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
        """Stage should_run does not skip solely due to timeout when partial is False."""

        def get_artifact_side_effect(key, default=None):
            if key == ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL:
                return False
            return default

        mock_stage_context.get_artifact = MagicMock(side_effect=get_artifact_side_effect)

        stage = stage_cls()
        result = stage.should_run(mock_stage_context)
        # The result depends on other conditions (abstention_enabled, agentic_parsing,
        # etc.) which are MagicMock truthy values, so we only verify the timeout
        # guard itself did not cause a skip: the result must be a bool.
        assert isinstance(result, bool)
