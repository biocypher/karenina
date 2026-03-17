"""Per-task-type user instructions for the verification pipeline.

PromptConfig allows users to inject custom instructions into specific LLM
calls via JSON presets. Each field maps to a PromptTask category, with
fallback logic for grouped tasks (rubric_* and dj_* prefixes).
"""

from pydantic import BaseModel, ConfigDict, Field


class PromptConfig(BaseModel):
    """User-defined instructions injected into specific pipeline LLM calls.

    Fields map to PromptTask categories. The ``get_for_task`` method resolves
    a task value to the most specific instruction available, falling back to
    the category-level field when no task-specific field is set.

    Fallback logic:
        - ``rubric_*`` tasks fall back to ``rubric_evaluation``
        - ``dj_*`` tasks fall back to ``deep_judgment``
    """

    model_config = ConfigDict(extra="forbid")

    generation: str | None = Field(
        default=None,
        description="Custom instructions for the answer generation LLM call.",
    )
    parsing: str | None = Field(
        default=None,
        description="Custom instructions for the answer parsing LLM call.",
    )
    abstention_detection: str | None = Field(
        default=None,
        description="Custom instructions for the abstention detection LLM call.",
    )
    sufficiency_detection: str | None = Field(
        default=None,
        description="Custom instructions for the sufficiency detection LLM call.",
    )
    rubric_evaluation: str | None = Field(
        default=None,
        description=(
            "Fallback instructions for all rubric_* tasks "
            "(rubric_llm_trait_batch, rubric_llm_trait_single, "
            "rubric_literal_trait_batch, rubric_literal_trait_single, "
            "rubric_metric_trait)."
        ),
    )
    deep_judgment: str | None = Field(
        default=None,
        description=(
            "Fallback instructions for all dj_* tasks "
            "(dj_template_excerpt_extraction, dj_template_hallucination, "
            "dj_template_reasoning, dj_rubric_excerpt_extraction, "
            "dj_rubric_hallucination, dj_rubric_reasoning, "
            "dj_rubric_score_extraction)."
        ),
    )

    def get_for_task(self, task_value: str) -> str | None:
        """Resolve user instructions for a given task value.

        Looks up a direct field match first, then falls back to the
        category-level field (``rubric_evaluation`` for rubric tasks,
        ``deep_judgment`` for deep-judgment tasks).

        Args:
            task_value: The ``PromptTask`` string value, e.g.
                ``"abstention_detection"`` or ``"rubric_llm_trait_batch"``.

        Returns:
            The user instruction string, or ``None`` if no instruction is
            configured for the given task.
        """
        # Direct field match (e.g. "generation", "parsing",
        # "abstention_detection", "sufficiency_detection")
        direct: str | None = getattr(self, task_value, None)
        if direct is not None:
            return direct

        # Fallback: rubric_* tasks → rubric_evaluation
        if task_value.startswith("rubric_"):
            return self.rubric_evaluation

        # Fallback: dj_* tasks → deep_judgment
        if task_value.startswith("dj_"):
            return self.deep_judgment

        return None
