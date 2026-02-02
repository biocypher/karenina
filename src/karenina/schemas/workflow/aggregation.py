"""Aggregation interface and registry for verification results.

DEPRECATED: Import from `karenina.schemas.results` instead.
"""

# Re-export from new location for backward compatibility
from ..results.aggregation import (
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

__all__ = [
    "ResultAggregator",
    "AggregatorRegistry",
    "MeanAggregator",
    "MedianAggregator",
    "ModeAggregator",
    "MajorityVoteAggregator",
    "FirstAggregator",
    "CountAggregator",
    "create_default_registry",
]
