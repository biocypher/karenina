"""Result container models for verification outputs.

This module contains models for storing and aggregating verification results:
- TemplateResults, RubricResults, JudgmentResults: Specialized result containers
- VerificationResultSet: Top-level container for all results
- Aggregation utilities for computing statistics
"""

from .aggregation import (
    AggregatorRegistry,
    CountAggregator,
    FirstAggregator,
    MajorityVoteAggregator,
    MeanAggregator,
    MedianAggregator,
    ModeAggregator,
    ResultAggregator,
    create_default_registry,
)
from .judgment import JudgmentResults
from .rubric import RubricResults
from .rubric_judgment import RubricJudgmentResults
from .template import TemplateResults
from .verification_result_set import VerificationResultSet

__all__ = [
    # Aggregation
    "ResultAggregator",
    "AggregatorRegistry",
    "MeanAggregator",
    "MedianAggregator",
    "ModeAggregator",
    "MajorityVoteAggregator",
    "FirstAggregator",
    "CountAggregator",
    "create_default_registry",
    # Result containers
    "TemplateResults",
    "RubricResults",
    "JudgmentResults",
    "RubricJudgmentResults",
    "VerificationResultSet",
]
