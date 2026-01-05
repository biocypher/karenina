"""Score computation utilities for GEPA-Karenina integration.

Provides functions to compute scores from karenina's VerificationResult
for use as GEPA's optimization metric.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karenina.schemas.workflow.verification.result import VerificationResult


def compute_single_score(
    result: "VerificationResult",
    template_weight: float = 0.7,
    rubric_weight: float = 0.3,
) -> float:
    """Compute a single score from one verification result.

    Args:
        result: VerificationResult from karenina verification
        template_weight: Weight for template pass/fail (0.0-1.0)
        rubric_weight: Weight for rubric scores (0.0-1.0)

    Returns:
        Float score between 0.0 and 1.0
    """
    template_score = 0.0
    rubric_score = 0.0
    has_rubric = False

    # Template score (binary)
    if result.template and result.template.verify_result is not None:
        template_score = 1.0 if result.template.verify_result else 0.0

    # Rubric scores (averaged)
    if result.rubric and result.rubric.rubric_evaluation_performed:
        trait_scores: list[float] = []
        # Use get_all_trait_scores() which aggregates all trait types
        all_scores = result.rubric.get_all_trait_scores()
        for trait_result in all_scores.values():
            if isinstance(trait_result, bool):
                trait_scores.append(1.0 if trait_result else 0.0)
            elif isinstance(trait_result, int | float):
                # Normalize to 0-1 range (assuming 1-5 scale)
                trait_scores.append(float(trait_result) / 5.0)
            elif isinstance(trait_result, dict):
                # Metric trait - use F1 score if available, else average all metrics
                if "f1" in trait_result:
                    trait_scores.append(float(trait_result["f1"]))
                else:
                    metric_values = [v for v in trait_result.values() if isinstance(v, int | float)]
                    if metric_values:
                        trait_scores.append(sum(metric_values) / len(metric_values))
        if trait_scores:
            rubric_score = sum(trait_scores) / len(trait_scores)
            has_rubric = True

    # Weighted combination
    if has_rubric:
        return template_weight * template_score + rubric_weight * rubric_score
    else:
        return template_score


def compute_weighted_score(
    results: dict[str, "VerificationResult"],
    template_weight: float = 0.7,
    rubric_weight: float = 0.3,
    rubric_trait_weights: dict[str, float] | None = None,  # noqa: ARG001
) -> float:
    """Compute a single aggregated score from multiple verification results.

    Args:
        results: Dict mapping result keys to VerificationResult objects
        template_weight: Weight for template verification result
        rubric_weight: Weight for rubric scores
        rubric_trait_weights: Optional custom weights per rubric trait

    Returns:
        Float score between 0.0 and 1.0
    """
    if not results:
        return 0.0

    scores: list[float] = []
    for result in results.values():
        scores.append(compute_single_score(result, template_weight, rubric_weight))

    return sum(scores) / len(scores) if scores else 0.0


def compute_multi_model_score(
    results_by_model: dict[str, list["VerificationResult"]],
    template_weight: float = 0.7,
    rubric_weight: float = 0.3,
) -> tuple[float, dict[str, float]]:
    """Compute scores across multiple models with per-model breakdown.

    This function:
    1. Computes per-model average scores
    2. Averages across all models for final score

    Args:
        results_by_model: Dict mapping model names to lists of results
        template_weight: Weight for template verification result
        rubric_weight: Weight for rubric scores

    Returns:
        Tuple of (overall_score, per_model_scores)
    """
    if not results_by_model:
        return 0.0, {}

    model_scores: dict[str, float] = {}

    for model_name, model_results in results_by_model.items():
        if not model_results:
            model_scores[model_name] = 0.0
            continue

        scores: list[float] = []
        for result in model_results:
            scores.append(compute_single_score(result, template_weight, rubric_weight))

        model_scores[model_name] = sum(scores) / len(scores) if scores else 0.0

    # Average across models
    overall = sum(model_scores.values()) / len(model_scores) if model_scores else 0.0

    return overall, model_scores


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
        Returns 0.0 if baseline is 0 to avoid division by zero.
    """
    if baseline_score == 0.0:
        return optimized_score  # If baseline is 0, return absolute score
    return (optimized_score - baseline_score) / baseline_score
