"""Tests covering KareninaAdapter.evaluate reading metadata.failure for trajectories.

The gepa adapter's evaluate() method builds KareninaTrajectory instances that
carry a ``parsing_error`` string. That string was migrated off
``metadata.error`` onto ``metadata.failure.reason`` (or ``None`` when the run
passed). This test exercises both branches with mocked benchmark/config stubs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from karenina.integrations.gepa.config import ObjectiveConfig, OptimizationTarget, TraitSelectionMode
from karenina.integrations.gepa.data_types import KareninaDataInst
from karenina.schemas.results.failure import Failure, FailureCategory


def _build_stub_result(question_id: str, answering_model: str, failure: Failure | None):
    """Build a minimal VerificationResult double used by KareninaAdapter.evaluate."""
    result = MagicMock()
    result.metadata.question_id = question_id
    result.metadata.answering_model = answering_model
    result.metadata.failure = failure
    # Template carries the raw_llm_response and verify_result the adapter reads.
    result.template.raw_llm_response = "the response text"
    result.template.verify_result = failure is None
    # No rubric evaluation in this test path.
    result.rubric = None
    result.rubric_evaluation_performed = False
    return result


def _build_adapter_with_stub_benchmark(results_iter):
    """Build KareninaAdapter wrapping stubbed benchmark + config."""
    from karenina.integrations.gepa.adapter import KareninaAdapter

    result_set = MagicMock()
    result_set.__iter__.side_effect = lambda: iter(results_iter)

    benchmark = MagicMock()
    benchmark.run_verification.return_value = result_set
    benchmark.get_global_rubric.return_value = None

    model = MagicMock()
    model.model_name = "model-a"
    model.id = "model-a"
    model.mcp_urls_dict = None
    model.mcp_tool_filter = None
    base_config = MagicMock()
    base_config.answering_models = [model]
    base_config.model_copy.return_value = base_config

    adapter = KareninaAdapter(
        benchmark=benchmark,
        base_config=base_config,
        targets=[OptimizationTarget.ANSWERING_SYSTEM_PROMPT],
        objective_config=ObjectiveConfig(include_template=True, trait_mode=TraitSelectionMode.NONE),
        auto_fetch_tool_descriptions=False,
    )
    return adapter


@pytest.mark.unit
class TestKareninaAdapterFailureMigration:
    """KareninaAdapter.evaluate reads metadata.failure.reason for parsing_error."""

    def test_trajectory_parsing_error_is_none_on_pass(self) -> None:
        result = _build_stub_result("q1", "model-a", failure=None)
        adapter = _build_adapter_with_stub_benchmark([result])

        batch = [
            KareninaDataInst(
                question_id="q1",
                question_text="what?",
                raw_answer="y",
                template_code="class Answer: pass",
            )
        ]

        evaluation = adapter.evaluate(batch, candidate={"answering_system_prompt": "be kind"}, capture_traces=True)

        assert evaluation.trajectories is not None
        assert len(evaluation.trajectories) == 1
        assert evaluation.trajectories[0].parsing_error is None

    def test_trajectory_parsing_error_reflects_failure_reason(self) -> None:
        failure = Failure(
            category=FailureCategory.PARSING,
            stage="parse_template",
            reason="unparseable response",
        )
        result = _build_stub_result("q1", "model-a", failure=failure)
        adapter = _build_adapter_with_stub_benchmark([result])

        batch = [
            KareninaDataInst(
                question_id="q1",
                question_text="what?",
                raw_answer="y",
                template_code="class Answer: pass",
            )
        ]

        evaluation = adapter.evaluate(batch, candidate={"answering_system_prompt": "be kind"}, capture_traces=True)

        assert evaluation.trajectories is not None
        assert len(evaluation.trajectories) == 1
        assert evaluation.trajectories[0].parsing_error == "unparseable response"
