"""Integration tests for agentic rubric evaluation pipeline.

Verifies that AgenticRubricTrait, Rubric, and StageOrchestrator integrate
correctly: schema roundtrip, stage ordering, registration logic for mixed
and agentic-only rubrics, and stage count in full pipeline configurations.
"""

import pytest

from karenina.benchmark.verification.stages.core.orchestrator import StageOrchestrator
from karenina.schemas.entities.rubric import AgenticRubricTrait, LLMRubricTrait, Rubric


@pytest.mark.unit
class TestAgenticRubricPipelineIntegration:
    """Test that Stage 11b integrates correctly with the full pipeline."""

    def test_orchestrator_builds_correct_stage_order(self):
        """Verify stage ordering: RubricEvaluation < AgenticRubricEvaluation < FinalizeResult."""
        rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="clarity", kind="boolean")],
            agentic_traits=[
                AgenticRubricTrait(
                    name="code_quality",
                    description="Check code quality.",
                    kind="boolean",
                )
            ],
        )
        orch = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="template_and_rubric",
        )
        names = [s.name for s in orch.stages]

        # Stage 11b after Stage 11
        assert names.index("AgenticRubricEvaluation") > names.index("RubricEvaluation")
        # Stage 11b before FinalizeResult
        assert names.index("AgenticRubricEvaluation") < names.index("FinalizeResult")

    def test_agentic_only_rubric_works(self):
        """Rubric with only agentic traits (no LLM/regex/callable/metric)."""
        rubric = Rubric(
            agentic_traits=[
                AgenticRubricTrait(
                    name="tests_pass",
                    description="Run tests and check.",
                    kind="boolean",
                )
            ],
        )
        orch = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="rubric_only",
        )
        names = [s.name for s in orch.stages]
        assert "AgenticRubricEvaluation" in names
        # Classical RubricEvaluation should NOT be registered
        assert "RubricEvaluation" not in names

    def test_mixed_rubric_has_both_stages(self):
        """Rubric with both classical and agentic traits gets both stages."""
        rubric = Rubric(
            llm_traits=[LLMRubricTrait(name="clarity", kind="boolean")],
            agentic_traits=[
                AgenticRubricTrait(
                    name="code_quality",
                    description="Check code quality.",
                    kind="boolean",
                )
            ],
        )
        orch = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="template_and_rubric",
        )
        names = [s.name for s in orch.stages]
        assert "RubricEvaluation" in names
        assert "AgenticRubricEvaluation" in names

    def test_full_pipeline_stage_count_with_agentic(self):
        """Full pipeline with agentic traits has expected number of stages."""
        rubric = Rubric(
            agentic_traits=[
                AgenticRubricTrait(
                    name="code_quality",
                    description="Check code quality.",
                    kind="boolean",
                )
            ],
        )
        orch = StageOrchestrator.from_config(
            rubric=rubric,
            evaluation_mode="template_and_rubric",
        )
        # Should have: Validate, Generate, RecursionLimit, TraceValidation,
        # Parse, Verify, Embedding, AgenticRubric, Finalize = at least 9 stages
        assert len(orch.stages) >= 9

    def test_trait_schema_roundtrip(self):
        """AgenticRubricTrait survives serialization roundtrip."""
        trait = AgenticRubricTrait(
            name="code_quality",
            description="Check code quality.",
            kind="score",
            min_score=1,
            max_score=5,
            context_mode="workspace_only",
            max_turns=10,
            timeout_seconds=60,
        )
        data = trait.model_dump()
        restored = AgenticRubricTrait.model_validate(data)
        assert restored.name == trait.name
        assert restored.kind == trait.kind
        assert restored.min_score == trait.min_score
        assert restored.max_score == trait.max_score
        assert restored.context_mode == trait.context_mode
        assert restored.max_turns == trait.max_turns
        assert restored.timeout_seconds == trait.timeout_seconds
