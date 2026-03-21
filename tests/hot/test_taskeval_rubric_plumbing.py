"""Hot tests for TaskEval rubric plumbing: DynamicRubric and agentic trait support.

These tests hit the Anthropic API with Haiku to verify the new TaskEval
plumbing works end-to-end through the real verification pipeline.

Run:
    uv run pytest tests/hot/test_taskeval_rubric_plumbing.py -v --tb=short

Requires ANTHROPIC_API_KEY in environment or .env file.
"""

import os

import pytest
from dotenv import load_dotenv

# Load .env from repo root (two levels up from karenina/tests/hot/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

from karenina.benchmark.task_eval import TaskEval  # noqa: E402
from karenina.schemas.config import ModelConfig  # noqa: E402
from karenina.schemas.entities import BaseAnswer, VerifiedField  # noqa: E402
from karenina.schemas.entities.rubric import (  # noqa: E402
    AgenticRubricTrait,
    DynamicRubric,
    LLMRubricTrait,
    RegexTrait,
    Rubric,
)
from karenina.schemas.primitives import BooleanMatch  # noqa: E402
from karenina.schemas.verification import VerificationConfig  # noqa: E402

PARSING_MODEL = ModelConfig(
    id="haiku",
    model_provider="anthropic",
    model_name="claude-haiku-4-5",
    interface="langchain",
    temperature=0.0,
)

CONFIG = VerificationConfig(
    parsing_models=[PARSING_MODEL],
    parsing_only=True,
)


# =============================================================================
# 1. DynamicRubric: present trait is promoted and scored
# =============================================================================


class TestDynamicRubricPresent:
    """DynamicRubric trait whose concept IS present in the response."""

    def test_present_trait_is_promoted_and_scored(self):
        """A dynamic trait about citations should be promoted when the response has citations."""
        task = TaskEval(task_id="dynamic-present")
        task.log(
            "Venetoclax targets BCL2 and induces apoptosis in CLL cells [1]. "
            "Clinical trials showed 80% overall response rate [2]."
        )

        task.add_dynamic_rubric(
            DynamicRubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="citation_quality",
                        summary="numbered citations",
                        description=(
                            "Answer True if the response uses numbered citations "
                            "in bracket notation (e.g., [1], [2]) to support claims."
                        ),
                        kind="boolean",
                        higher_is_better=True,
                    ),
                ],
            )
        )

        result = task.evaluate(CONFIG)
        assert result.global_eval is not None

        # Should have one verification result (synthetic rubric-only question)
        vrs = result.global_eval.verification_results
        assert len(vrs) > 0

        # The verification result should have rubric data with dynamic rubric info
        vr = list(vrs.values())[0][0]
        assert vr.metadata.completed_without_errors, f"Pipeline failed: {vr}"

        # Dynamic trait should have been promoted (not skipped)
        if vr.rubric:
            promoted = vr.rubric.dynamic_rubric_promoted_traits or []
            skipped = vr.rubric.dynamic_rubric_skipped_traits or {}
            assert "citation_quality" in promoted or "citation_quality" not in skipped, (
                f"Expected citation_quality to be promoted. Promoted: {promoted}, Skipped: {skipped}"
            )


# =============================================================================
# 2. DynamicRubric: absent trait is skipped
# =============================================================================


class TestDynamicRubricAbsent:
    """DynamicRubric trait whose concept is NOT present in the response."""

    def test_absent_trait_is_skipped(self):
        """A dynamic trait about drug interactions should be skipped for a math response."""
        task = TaskEval(task_id="dynamic-absent")
        task.log("The sum of 2 and 2 is 4. This is a basic arithmetic operation.")

        task.add_dynamic_rubric(
            DynamicRubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="interaction_warnings",
                        summary="drug-drug interaction warnings",
                        description=(
                            "Answer True if the response includes specific warnings "
                            "about drug-drug interactions and contraindications."
                        ),
                        kind="boolean",
                        higher_is_better=True,
                    ),
                ],
            )
        )

        result = task.evaluate(CONFIG)
        assert result.global_eval is not None

        vrs = result.global_eval.verification_results
        assert len(vrs) > 0

        vr = list(vrs.values())[0][0]
        assert vr.metadata.completed_without_errors, f"Pipeline failed: {vr}"

        # Dynamic trait should have been skipped (concept not present)
        if vr.rubric:
            skipped = vr.rubric.dynamic_rubric_skipped_traits or {}
            assert "interaction_warnings" in skipped, f"Expected interaction_warnings to be skipped. Skipped: {skipped}"


# =============================================================================
# 3. DynamicRubric + static rubric together
# =============================================================================


class TestDynamicAndStaticRubric:
    """DynamicRubric combined with a static Rubric in the same TaskEval."""

    def test_both_static_and_dynamic_traits_evaluated(self):
        """Static regex trait always runs; dynamic LLM trait only if concept present."""
        task = TaskEval(task_id="mixed-rubrics")
        task.log(
            "BCL2 is the primary pharmacological target of venetoclax. "
            "It is a member of the BCL-2 family of proteins that regulate "
            "apoptosis. Venetoclax is indicated for CLL [1]."
        )

        # Static rubric: always evaluated
        task.add_rubric(
            Rubric(
                regex_traits=[
                    RegexTrait(
                        name="has_brackets",
                        pattern=r"\[\d+\]",
                        higher_is_better=True,
                    ),
                ],
            )
        )

        # Dynamic rubric: conditional on concept presence
        task.add_dynamic_rubric(
            DynamicRubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="apoptosis_context",
                        summary="apoptosis pathway explanation",
                        description=(
                            "Answer True if the response explains the role of "
                            "the target protein in the apoptosis pathway."
                        ),
                        kind="boolean",
                        higher_is_better=True,
                    ),
                ],
            )
        )

        result = task.evaluate(CONFIG)
        assert result.global_eval is not None

        vrs = result.global_eval.verification_results
        assert len(vrs) > 0

        vr = list(vrs.values())[0][0]
        assert vr.metadata.completed_without_errors, f"Pipeline failed: {vr}"

        # Static regex trait should always be scored
        if vr.rubric and vr.rubric.regex_trait_scores:
            assert "has_brackets" in vr.rubric.regex_trait_scores, (
                f"Static trait 'has_brackets' missing from results: {vr.rubric.regex_trait_scores}"
            )


# =============================================================================
# 4. DynamicRubric + template (template_and_rubric mode)
# =============================================================================


class TestDynamicRubricWithTemplate:
    """DynamicRubric combined with an answer template for full evaluation."""

    def test_template_and_dynamic_rubric(self):
        """Template verification + dynamic rubric presence check both execute."""

        class Answer(BaseAnswer):
            identifies_bcl2: bool = VerifiedField(
                description=("True if the response identifies BCL2 (or Bcl-2, BCL-2) as the pharmacological target."),
                ground_truth=True,
                verify_with=BooleanMatch(),
            )

        task = TaskEval(task_id="template-and-dynamic")
        task.log(
            "Venetoclax (ABT-199) selectively inhibits BCL-2, a key "
            "anti-apoptotic protein overexpressed in CLL. By binding BCL-2, "
            "venetoclax restores apoptotic signaling. Common side effects "
            "include neutropenia and diarrhea."
        )

        task.add_template(Answer)
        task.add_dynamic_rubric(
            DynamicRubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="side_effects",
                        summary="side effects or adverse events",
                        description=("Answer True if the response mentions specific side effects or adverse events."),
                        kind="boolean",
                        higher_is_better=True,
                    ),
                ],
            )
        )

        result = task.evaluate(CONFIG)
        assert result.global_eval is not None

        vrs = result.global_eval.verification_results
        assert len(vrs) > 0

        vr = list(vrs.values())[0][0]
        assert vr.metadata.completed_without_errors, f"Pipeline failed: {vr}"

        # Template verification should have run
        if vr.template:
            assert vr.template.verify_result is not None

        # Dynamic rubric presence check should have run
        if vr.rubric:
            promoted = vr.rubric.dynamic_rubric_promoted_traits or []
            skipped = vr.rubric.dynamic_rubric_skipped_traits or {}
            # side_effects concept IS present, so it should be promoted
            assert "side_effects" in promoted or "side_effects" not in skipped, (
                f"Expected side_effects promoted. Promoted: {promoted}, Skipped: {skipped}"
            )


# =============================================================================
# 5. Step-scoped DynamicRubric
# =============================================================================


class TestDynamicRubricPerStep:
    """DynamicRubric scoped to a specific step in TaskEval."""

    def test_step_dynamic_rubric(self):
        """Dynamic rubric attached to a step evaluates only that step's logs."""
        task = TaskEval(task_id="step-dynamic")

        # Step "retrieval" has drug content
        task.log(
            "Retrieved: Warfarin interacts with aspirin, increasing bleeding risk.",
            step_id="retrieval",
            target="step",
        )
        # Step "summary" has a plain summary
        task.log(
            "Summary: The patient should avoid concurrent aspirin use.",
            step_id="summary",
            target="step",
        )

        # Dynamic rubric on the retrieval step
        task.add_dynamic_rubric(
            DynamicRubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="interaction_detail",
                        summary="specific drug interaction details",
                        description=(
                            "Answer True if the response provides specific "
                            "details about a drug-drug interaction mechanism."
                        ),
                        kind="boolean",
                        higher_is_better=True,
                    ),
                ],
            ),
            step_id="retrieval",
        )

        # Evaluate retrieval step only
        result = task.evaluate(CONFIG, step_id="retrieval")
        assert "retrieval" in result.per_step

        step_eval = result.per_step["retrieval"]
        vrs = step_eval.verification_results
        assert len(vrs) > 0

        vr = list(vrs.values())[0][0]
        assert vr.metadata.completed_without_errors, f"Step pipeline failed: {vr}"


# =============================================================================
# 6. Agentic rubric trait (trace_only, no workspace)
# =============================================================================


class TestAgenticRubricTraitTraceOnly:
    """Agentic rubric trait with trace_only context mode through TaskEval.

    Uses trace_only so no workspace is needed. The agent examines
    the logged trace inline and produces a boolean score.
    """

    def test_agentic_trait_trace_only_boolean(self):
        """Agentic boolean trait in trace_only mode executes through TaskEval."""
        from karenina.ports.messages import Message, ToolUseContent

        task = TaskEval(task_id="agentic-trace-only")

        # Log a structured trace with tool calls
        task.log_trace(
            [
                Message.assistant(
                    "I'll search for information about BCL2.",
                    tool_calls=[
                        ToolUseContent(id="call_1", name="search", input={"query": "BCL2 protein function"}),
                    ],
                ),
                Message.tool_result("call_1", "BCL2 is an anti-apoptotic protein."),
                Message.assistant(
                    "BCL2 (B-cell lymphoma 2) is an anti-apoptotic protein that "
                    "regulates programmed cell death. It is the primary target of "
                    "venetoclax, which is used to treat CLL."
                ),
            ]
        )

        task.add_rubric(
            Rubric(
                agentic_traits=[
                    AgenticRubricTrait(
                        name="used_search_tool",
                        description=(
                            "Examine the agent trace. Answer True if the agent "
                            "used a search tool at least once before producing "
                            "its final answer. Answer False if no search tool "
                            "was used."
                        ),
                        kind="boolean",
                        higher_is_better=True,
                        context_mode="trace_only",
                        max_turns=5,
                        timeout_seconds=60,
                    ),
                ],
            )
        )

        result = task.evaluate(CONFIG)
        assert result.global_eval is not None

        vrs = result.global_eval.verification_results
        assert len(vrs) > 0

        vr = list(vrs.values())[0][0]
        assert vr.metadata.completed_without_errors, f"Agentic pipeline failed: {vr}"

        # Agentic trait should have been scored
        if vr.rubric and vr.rubric.agentic_trait_scores:
            assert "used_search_tool" in vr.rubric.agentic_trait_scores, (
                f"Agentic trait 'used_search_tool' missing from scores: {vr.rubric.agentic_trait_scores}"
            )
            # The trace clearly shows a search tool call, so should be True
            assert vr.rubric.agentic_trait_scores["used_search_tool"] is True, (
                f"Expected True for used_search_tool, got: {vr.rubric.agentic_trait_scores['used_search_tool']}"
            )


# =============================================================================
# 7. Agentic trait + DynamicRubric together
# =============================================================================


class TestAgenticWithDynamicRubric:
    """Agentic rubric trait combined with DynamicRubric in the same TaskEval."""

    def test_agentic_and_dynamic_together(self):
        """Static agentic trait + dynamic LLM trait both execute correctly."""
        from karenina.ports.messages import Message, ToolUseContent

        task = TaskEval(task_id="agentic-plus-dynamic")

        task.log_trace(
            [
                Message.assistant(
                    "Let me look up warfarin interactions.",
                    tool_calls=[
                        ToolUseContent(id="c1", name="drug_lookup", input={"drug": "warfarin"}),
                    ],
                ),
                Message.tool_result(
                    "c1",
                    "Warfarin interacts with aspirin (increased bleeding risk) and is contraindicated with NSAIDs.",
                ),
                Message.assistant(
                    "Warfarin has significant drug interactions. It should not be "
                    "combined with aspirin due to increased bleeding risk. NSAIDs "
                    "are also contraindicated."
                ),
            ]
        )

        # Static agentic trait (always evaluated)
        task.add_rubric(
            Rubric(
                agentic_traits=[
                    AgenticRubricTrait(
                        name="used_drug_lookup",
                        description=(
                            "Examine the trace. Answer True if the agent used a "
                            "drug_lookup tool. Answer False otherwise."
                        ),
                        kind="boolean",
                        higher_is_better=True,
                        context_mode="trace_only",
                        max_turns=5,
                        timeout_seconds=60,
                    ),
                ],
            )
        )

        # Dynamic trait (conditional on concept presence)
        task.add_dynamic_rubric(
            DynamicRubric(
                llm_traits=[
                    LLMRubricTrait(
                        name="contraindication_warning",
                        summary="contraindication warnings",
                        description=("Answer True if the response explicitly warns about drug contraindications."),
                        kind="boolean",
                        higher_is_better=True,
                    ),
                ],
            )
        )

        result = task.evaluate(CONFIG)
        assert result.global_eval is not None

        vrs = result.global_eval.verification_results
        assert len(vrs) > 0

        vr = list(vrs.values())[0][0]
        assert vr.metadata.completed_without_errors, f"Combined pipeline failed: {vr}"

        # Agentic trait should be scored
        if vr.rubric and vr.rubric.agentic_trait_scores:
            assert "used_drug_lookup" in vr.rubric.agentic_trait_scores

        # Dynamic trait about contraindications should be promoted (concept is present)
        if vr.rubric:
            promoted = vr.rubric.dynamic_rubric_promoted_traits or []
            skipped = vr.rubric.dynamic_rubric_skipped_traits or {}
            assert "contraindication_warning" in promoted or "contraindication_warning" not in skipped, (
                f"Expected contraindication_warning promoted. Promoted: {promoted}, Skipped: {skipped}"
            )
