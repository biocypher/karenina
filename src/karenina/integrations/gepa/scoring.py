"""Score computation utilities for GEPA-Karenina integration.

Provides functions to compute multi-objective scores from karenina's
VerificationResult for Pareto optimization in GEPA.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karenina.integrations.gepa.config import ObjectiveConfig
    from karenina.schemas.workflow.verification.result import VerificationResult


def compute_objective_scores(
    result: "VerificationResult",
    model_name: str,
    config: "ObjectiveConfig",
    trait_max_scores: dict[str, int] | None = None,
) -> dict[str, float]:
    """Compute per-dimension objective scores for Pareto optimization.

    Produces compound keys in format 'model:dimension' for multi-objective
    optimization across both models and evaluation dimensions.

    Args:
        result: VerificationResult from karenina verification
        model_name: Name of the model that produced this result
        config: ObjectiveConfig controlling which dimensions to include
        trait_max_scores: Optional dict mapping trait names to their max_score.
            Used for proper normalization of score-based traits. If not provided,
            defaults to 5 for backwards compatibility.

    Returns:
        Dict mapping 'model:dimension' to float scores (0.0-1.0)

    Example:
        >>> scores = compute_objective_scores(result, "claude-haiku", config)
        >>> # Returns: {"claude-haiku:template": 1.0, "claude-haiku:clarity": 0.8}
    """
    from karenina.integrations.gepa.config import TraitSelectionMode

    objectives: dict[str, float] = {}
    trait_max_scores = trait_max_scores or {}

    # Template objective
    if config.include_template:
        template_score = 0.0
        if result.template and result.template.verify_result is not None:
            template_score = 1.0 if result.template.verify_result else 0.0
        objectives[f"{model_name}:template"] = template_score

    # Rubric trait objectives
    if (
        config.trait_mode != TraitSelectionMode.NONE
        and result.rubric
        and result.rubric.rubric_evaluation_performed
    ):
        all_scores = result.rubric.get_all_trait_scores()

        for trait_name, trait_result in all_scores.items():
            if not config.should_include_trait(trait_name):
                continue

            if isinstance(trait_result, bool):
                objectives[f"{model_name}:{trait_name}"] = 1.0 if trait_result else 0.0

            elif isinstance(trait_result, int | float):
                # Normalize using trait's max_score, defaulting to 5 for backwards compatibility
                max_score = trait_max_scores.get(trait_name, 5)
                objectives[f"{model_name}:{trait_name}"] = float(trait_result) / float(max_score)

            elif isinstance(trait_result, dict):
                # Metric trait - expand based on metric_config
                enabled_metrics = config.metric_config.get_enabled_metrics()
                for metric_name in enabled_metrics:
                    if metric_name in trait_result:
                        key = f"{model_name}:{trait_name}_{metric_name}"
                        objectives[key] = float(trait_result[metric_name])

    return objectives


def extract_failed_fields(result: "VerificationResult") -> list[str]:
    """Extract list of template fields that failed verification.

    Args:
        result: VerificationResult to analyze

    Returns:
        List of field names that failed verification
    """
    failed: list[str] = []

    if not result.template:
        return failed

    # Check field results if available
    if hasattr(result.template, "field_results") and result.template.field_results:
        for field_name, field_result in result.template.field_results.items():
            if (
                isinstance(field_result, bool)
                and not field_result
                or isinstance(field_result, dict)
                and not field_result.get("passed", True)
            ):
                failed.append(field_name)

    return failed


def compute_improvement(
    baseline_score: float,
    optimized_score: float,
) -> float:
    """Compute relative improvement from baseline to optimized score.

    Args:
        baseline_score: Score before optimization
        optimized_score: Score after optimization

    Returns:
        Relative improvement as a fraction (e.g., 0.15 = 15% improvement).
        Returns optimized_score if baseline is 0 to avoid division by zero.
    """
    if baseline_score == 0.0:
        return optimized_score
    return (optimized_score - baseline_score) / baseline_score
