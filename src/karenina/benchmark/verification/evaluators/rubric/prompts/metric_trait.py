"""Prompt construction for metric trait evaluation using confusion matrix analysis.

Backwards-compatibility re-export. Canonical location:
karenina.benchmark.verification.prompts.rubric.metric_trait

This module provides the MetricTraitPromptBuilder class for constructing prompts
used in metric trait evaluation, which computes precision, recall, F1, specificity,
and accuracy by categorizing answer content into TP/TN/FP/FN buckets.

Two evaluation modes are supported:
- tp_only: Only TP instructions provided; computes precision, recall, F1
- full_matrix: Both TP and TN instructions; computes all metrics
"""

from karenina.benchmark.verification.prompts.rubric.metric_trait import MetricTraitPromptBuilder

__all__ = ["MetricTraitPromptBuilder"]
