"""
Aggregation interface and registry for verification results.

This module provides an extensible framework for aggregating verification results
across different axes (question, model, replicate, trait, etc.) using various
strategies (mean, median, mode, custom logic).
"""

import statistics
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from .verification import VerificationResult

T = TypeVar("T")


class ResultAggregator(Protocol):
    """
    Protocol defining the interface for result aggregators.

    Aggregators implement specific strategies for combining multiple values
    into a single aggregate value. They can handle different data types and
    aggregation logic.
    """

    def aggregate(self, values: list[Any], **kwargs: Any) -> Any:
        """
        Aggregate a list of values into a single result.

        Args:
            values: List of values to aggregate
            **kwargs: Additional aggregator-specific parameters

        Returns:
            Aggregated result (type depends on aggregator implementation)
        """
        ...


class AggregatorRegistry:
    """
    Registry for managing and accessing result aggregators.

    Provides a central place to register custom aggregators and retrieve
    them by name. Each specialized result class can maintain its own registry.
    """

    def __init__(self) -> None:
        """Initialize empty aggregator registry."""
        self._aggregators: dict[str, ResultAggregator] = {}

    def register(self, name: str, aggregator: ResultAggregator) -> None:
        """
        Register a new aggregator.

        Args:
            name: Unique name for the aggregator
            aggregator: Aggregator instance implementing ResultAggregator protocol

        Raises:
            ValueError: If aggregator name already exists
        """
        if name in self._aggregators:
            raise ValueError(f"Aggregator '{name}' is already registered")
        self._aggregators[name] = aggregator

    def get(self, name: str) -> ResultAggregator:
        """
        Retrieve an aggregator by name.

        Args:
            name: Name of the aggregator to retrieve

        Returns:
            The registered aggregator

        Raises:
            KeyError: If aggregator name not found
        """
        if name not in self._aggregators:
            raise KeyError(f"Aggregator '{name}' not found. Available: {list(self._aggregators.keys())}")
        return self._aggregators[name]

    def list_aggregators(self) -> list[str]:
        """
        List all registered aggregator names.

        Returns:
            List of aggregator names
        """
        return list(self._aggregators.keys())

    def __contains__(self, name: str) -> bool:
        """Check if aggregator is registered."""
        return name in self._aggregators


# ============================================================================
# GroupBy Strategy Pattern
# ============================================================================


class GroupByStrategy(Protocol):
    """
    Protocol defining the interface for grouping strategies.

    Grouping strategies determine how to organize results into groups
    for aggregation. They take a result and return a group key.
    """

    def get_group_key(self, result: "VerificationResult") -> str:
        """
        Extract group key from a verification result.

        Args:
            result: VerificationResult to extract key from

        Returns:
            Group key as a string
        """
        ...


class GroupByRegistry:
    """
    Registry for managing and accessing grouping strategies.

    Provides a central place to register custom grouping strategies and
    retrieve them by name.
    """

    def __init__(self) -> None:
        """Initialize empty groupby registry."""
        self._strategies: dict[str, GroupByStrategy] = {}

    def register(self, name: str, strategy: GroupByStrategy) -> None:
        """
        Register a new grouping strategy.

        Args:
            name: Unique name for the strategy
            strategy: Strategy instance implementing GroupByStrategy protocol

        Raises:
            ValueError: If strategy name already exists
        """
        if name in self._strategies:
            raise ValueError(f"GroupBy strategy '{name}' is already registered")
        self._strategies[name] = strategy

    def get(self, name: str) -> GroupByStrategy:
        """
        Retrieve a grouping strategy by name.

        Args:
            name: Name of the strategy to retrieve

        Returns:
            The registered strategy

        Raises:
            KeyError: If strategy name not found
        """
        if name not in self._strategies:
            raise KeyError(f"GroupBy strategy '{name}' not found. Available: {list(self._strategies.keys())}")
        return self._strategies[name]

    def list_strategies(self) -> list[str]:
        """
        List all registered strategy names.

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())

    def __contains__(self, name: str) -> bool:
        """Check if strategy is registered."""
        return name in self._strategies


# ============================================================================
# Built-in GroupBy Strategies
# ============================================================================


class ByQuestionStrategy:
    """Group results by question ID."""

    def get_group_key(self, result: "VerificationResult") -> str:
        """Return question ID as group key."""
        return result.metadata.question_id


class ByModelStrategy:
    """Group results by answering model."""

    def get_group_key(self, result: "VerificationResult") -> str:
        """Return answering model name as group key."""
        return result.metadata.answering_model


class ByParsingModelStrategy:
    """Group results by parsing model."""

    def get_group_key(self, result: "VerificationResult") -> str:
        """Return parsing model name as group key."""
        return result.metadata.parsing_model


class ByReplicateStrategy:
    """Group results by replicate number."""

    def get_group_key(self, result: "VerificationResult") -> str:
        """Return replicate number as group key."""
        return str(result.metadata.answering_replicate or 0)


class ByRunNameStrategy:
    """Group results by run name."""

    def get_group_key(self, result: "VerificationResult") -> str:
        """Return run name as group key."""
        return result.metadata.run_name or "default"


class ByModelPairStrategy:
    """Group results by (answering_model, parsing_model) pair."""

    def get_group_key(self, result: "VerificationResult") -> str:
        """Return model pair as group key."""
        return f"{result.metadata.answering_model}_{result.metadata.parsing_model}"


# ============================================================================
# Built-in Aggregators
# ============================================================================


class MeanAggregator:
    """
    Aggregate numeric values using arithmetic mean.

    Filters out None values before computing mean.
    """

    def aggregate(self, values: list[float | int | None], **_kwargs: Any) -> float | None:
        """
        Compute arithmetic mean of numeric values.

        Args:
            values: List of numeric values (None values are filtered out)

        Returns:
            Mean value, or None if no valid values
        """
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return None
        return statistics.mean(valid_values)


class MedianAggregator:
    """
    Aggregate numeric values using median.

    Filters out None values before computing median.
    """

    def aggregate(self, values: list[float | int | None], **_kwargs: Any) -> float | None:
        """
        Compute median of numeric values.

        Args:
            values: List of numeric values (None values are filtered out)

        Returns:
            Median value, or None if no valid values
        """
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return None
        return statistics.median(valid_values)


class ModeAggregator:
    """
    Aggregate values using mode (most common value).

    Works with any hashable type. Filters out None values.
    """

    def aggregate(self, values: list[Any], **_kwargs: Any) -> Any:
        """
        Find most common value.

        Args:
            values: List of hashable values (None values are filtered out)

        Returns:
            Most common value, or None if no valid values

        Raises:
            statistics.StatisticsError: If there's no unique mode
        """
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return None
        return statistics.mode(valid_values)


class MajorityVoteAggregator:
    """
    Aggregate boolean values using majority vote.

    Returns True if more than 50% of values are True.
    Filters out None values before computing.
    """

    def aggregate(self, values: list[bool | None], **kwargs: Any) -> bool | None:
        """
        Compute majority vote for boolean values.

        Args:
            values: List of boolean values (None values are filtered out)
            **kwargs: Optional parameters:
                - threshold: Fraction of True votes needed (default: 0.5)

        Returns:
            True if majority voted True, False otherwise, or None if no valid values
        """
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return None

        threshold: float = kwargs.get("threshold", 0.5)
        true_count = sum(1 for v in valid_values if v)
        return (true_count / len(valid_values)) > threshold


class ListAggregator:
    """
    Aggregate by collecting all values into a list.

    This is useful for preserving individual values while still
    providing an aggregated view. Filters out None values.
    """

    def aggregate(self, values: list[Any], **kwargs: Any) -> list[Any]:
        """
        Collect all values into a list.

        Args:
            values: List of values to collect
            **kwargs: Optional parameters:
                - include_none: Whether to include None values (default: False)

        Returns:
            List of all values
        """
        include_none: bool = kwargs.get("include_none", False)
        if include_none:
            return list(values)
        return [v for v in values if v is not None]


class FirstAggregator:
    """
    Aggregate by returning the first non-None value.

    Useful for metadata fields that should be consistent across results.
    """

    def aggregate(self, values: list[Any], **_kwargs: Any) -> Any:
        """
        Return first non-None value.

        Args:
            values: List of values

        Returns:
            First non-None value, or None if all are None
        """
        for v in values:
            if v is not None:
                return v
        return None


class CountAggregator:
    """
    Aggregate by counting occurrences of each unique value.

    Returns a dictionary mapping values to their counts.
    """

    def aggregate(self, values: list[Any], **kwargs: Any) -> dict[Any, int]:
        """
        Count occurrences of each value.

        Args:
            values: List of hashable values
            **kwargs: Optional parameters:
                - include_none: Whether to count None values (default: False)

        Returns:
            Dictionary mapping values to counts
        """
        include_none = kwargs.get("include_none", False)
        if not include_none:
            values = [v for v in values if v is not None]

        counts: dict[Any, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return counts


# ============================================================================
# Default Registry Factory
# ============================================================================


def create_default_registry() -> AggregatorRegistry:
    """
    Create a registry with all built-in aggregators registered.

    Returns:
        AggregatorRegistry with standard aggregators:
        - 'mean': Arithmetic mean for numeric values
        - 'median': Median for numeric values
        - 'mode': Most common value
        - 'majority_vote': Majority vote for boolean values
        - 'list': Collect all values into a list
        - 'first': First non-None value
        - 'count': Count occurrences of each value
    """
    registry = AggregatorRegistry()
    registry.register("mean", MeanAggregator())
    registry.register("median", MedianAggregator())
    registry.register("mode", ModeAggregator())
    registry.register("majority_vote", MajorityVoteAggregator())
    registry.register("list", ListAggregator())
    registry.register("first", FirstAggregator())
    registry.register("count", CountAggregator())
    return registry


def create_default_groupby_registry() -> GroupByRegistry:
    """
    Create a registry with all built-in grouping strategies registered.

    Returns:
        GroupByRegistry with standard strategies:
        - 'question': Group by question ID
        - 'model': Group by answering model
        - 'parsing_model': Group by parsing model
        - 'replicate': Group by replicate number
        - 'run_name': Group by run name
        - 'model_pair': Group by (answering_model, parsing_model) pair
    """
    registry = GroupByRegistry()
    registry.register("question", ByQuestionStrategy())
    registry.register("model", ByModelStrategy())
    registry.register("parsing_model", ByParsingModelStrategy())
    registry.register("replicate", ByReplicateStrategy())
    registry.register("run_name", ByRunNameStrategy())
    registry.register("model_pair", ByModelPairStrategy())
    return registry
