"""
Aggregation interface and registry for verification results.

This module provides an extensible framework for aggregating verification results
using pandas-compatible aggregator functions. Aggregators work seamlessly with
pandas Series and DataFrame groupby operations.
"""

from typing import Any, Protocol

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment,unused-ignore]


class ResultAggregator(Protocol):
    """
    Protocol defining the interface for result aggregators.

    Aggregators implement specific strategies for combining multiple values
    into a single aggregate value. They work with pandas Series for seamless
    integration with DataFrame groupby operations.
    """

    def aggregate(self, series: Any, **kwargs: Any) -> Any:
        """
        Aggregate a pandas Series into a single result.

        Args:
            series: pandas Series of values to aggregate
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
# Built-in Aggregators (Pandas-compatible)
# ============================================================================


class MeanAggregator:
    """
    Aggregate numeric values using arithmetic mean.

    Works with pandas Series. Automatically handles NaN values.
    """

    def aggregate(self, series: Any, **_kwargs: Any) -> float | None:
        """
        Compute arithmetic mean of numeric values.

        Args:
            series: pandas Series of numeric values

        Returns:
            Mean value, or None if no valid values
        """
        if pd is None:
            raise ImportError("pandas is required for aggregation")

        result = series.mean()
        return None if pd.isna(result) else float(result)


class MedianAggregator:
    """
    Aggregate numeric values using median.

    Works with pandas Series. Automatically handles NaN values.
    """

    def aggregate(self, series: Any, **_kwargs: Any) -> float | None:
        """
        Compute median of numeric values.

        Args:
            series: pandas Series of numeric values

        Returns:
            Median value, or None if no valid values
        """
        if pd is None:
            raise ImportError("pandas is required for aggregation")

        result = series.median()
        return None if pd.isna(result) else float(result)


class ModeAggregator:
    """
    Aggregate values using mode (most common value).

    Works with pandas Series. Handles NaN values.
    """

    def aggregate(self, series: Any, **_kwargs: Any) -> Any:
        """
        Find most common value.

        Args:
            series: pandas Series of values

        Returns:
            Most common value, or None if no valid values
        """
        if pd is None:
            raise ImportError("pandas is required for aggregation")

        mode_result = series.mode()
        if len(mode_result) == 0:
            return None
        return mode_result.iloc[0]


class MajorityVoteAggregator:
    """
    Aggregate boolean values using majority vote.

    Returns True if more than 50% of values are True.
    Works with pandas Series.
    """

    def aggregate(self, series: Any, **kwargs: Any) -> bool | None:
        """
        Compute majority vote for boolean values.

        Args:
            series: pandas Series of boolean values
            **kwargs: Optional parameters:
                - threshold: Fraction of True votes needed (default: 0.5)

        Returns:
            True if majority voted True, False otherwise, or None if no valid values
        """
        if pd is None:
            raise ImportError("pandas is required for aggregation")

        # Drop NaN values
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return None

        threshold: float = kwargs.get("threshold", 0.5)
        true_fraction = clean_series.sum() / len(clean_series)
        return bool(true_fraction > threshold)


class FirstAggregator:
    """
    Aggregate by returning the first non-null value.

    Useful for metadata fields that should be consistent across results.
    Works with pandas Series.
    """

    def aggregate(self, series: Any, **_kwargs: Any) -> Any:
        """
        Return first non-null value.

        Args:
            series: pandas Series of values

        Returns:
            First non-null value, or None if all are null
        """
        if pd is None:
            raise ImportError("pandas is required for aggregation")

        clean_series = series.dropna()
        if len(clean_series) == 0:
            return None
        return clean_series.iloc[0]


class CountAggregator:
    """
    Aggregate by counting value occurrences.

    Works with pandas Series.
    """

    def aggregate(self, series: Any, include_none: bool = False, **_kwargs: Any) -> dict[Any, int]:
        """
        Count occurrences of each unique value.

        Args:
            series: pandas Series of values
            include_none: Whether to include None values in counts

        Returns:
            Dictionary mapping values to their occurrence counts
        """
        if pd is None:
            raise ImportError("pandas is required for aggregation")

        if not include_none:
            series = series.dropna()

        value_counts = series.value_counts(dropna=not include_none)
        # Convert NaN keys to None and ensure consistent types
        result = {}
        for k, v in value_counts.items():
            # Replace pandas NaN with None
            key = None if pd.isna(k) else k
            result[key] = int(v)
        return result


# ============================================================================
# Default Registry Factory
# ============================================================================


def create_default_registry() -> AggregatorRegistry:
    """
    Create a registry with all built-in aggregators registered.

    All aggregators are pandas-compatible and work seamlessly with
    DataFrame groupby operations.

    Returns:
        AggregatorRegistry with standard aggregators:
        - 'mean': Arithmetic mean for numeric values
        - 'median': Median for numeric values
        - 'mode': Most common value
        - 'majority_vote': Majority vote for boolean values
        - 'first': First non-null value
        - 'count': Count value occurrences
    """
    registry = AggregatorRegistry()
    registry.register("mean", MeanAggregator())
    registry.register("median", MedianAggregator())
    registry.register("mode", ModeAggregator())
    registry.register("majority_vote", MajorityVoteAggregator())
    registry.register("first", FirstAggregator())
    registry.register("count", CountAggregator())
    return registry
