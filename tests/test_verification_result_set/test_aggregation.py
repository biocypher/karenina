"""Tests for the aggregation framework."""

import pandas as pd
import pytest

from karenina.schemas.workflow.aggregation import (
    AggregatorRegistry,
    CountAggregator,
    FirstAggregator,
    MajorityVoteAggregator,
    MeanAggregator,
    MedianAggregator,
    ModeAggregator,
    create_default_registry,
)


class TestAggregators:
    """Test suite for built-in aggregators."""

    def test_mean_aggregator(self):
        """Test MeanAggregator."""
        aggregator = MeanAggregator()

        # Test with valid values
        assert aggregator.aggregate(pd.Series([1, 2, 3, 4, 5])) == 3.0
        assert aggregator.aggregate(pd.Series([10, 20, 30])) == 20.0

        # Test with None values (should be filtered out)
        assert aggregator.aggregate(pd.Series([1, None, 3, None, 5])) == 3.0

        # Test with all None values
        assert aggregator.aggregate(pd.Series([None, None, None])) is None

        # Test with empty Series
        assert aggregator.aggregate(pd.Series([])) is None

    def test_median_aggregator(self):
        """Test MedianAggregator."""
        aggregator = MedianAggregator()

        # Test with odd number of values
        assert aggregator.aggregate(pd.Series([1, 2, 3, 4, 5])) == 3

        # Test with even number of values
        assert aggregator.aggregate(pd.Series([1, 2, 3, 4])) == 2.5

        # Test with None values (should be filtered out)
        assert aggregator.aggregate(pd.Series([1, None, 3, None, 5])) == 3

        # Test with all None values
        assert aggregator.aggregate(pd.Series([None, None])) is None

    def test_mode_aggregator(self):
        """Test ModeAggregator."""
        aggregator = ModeAggregator()

        # Test with clear mode
        assert aggregator.aggregate(pd.Series([1, 2, 2, 3, 2])) == 2
        assert aggregator.aggregate(pd.Series(["a", "b", "b", "c"])) == "b"

        # Test with None values (should be filtered out)
        assert aggregator.aggregate(pd.Series([1, None, 2, 2, None])) == 2

        # Test with no unique mode (returns first mode in Python 3.11+)
        result = aggregator.aggregate(pd.Series([1, 2]))
        assert result in [1, 2]  # Either is valid as they're both modes

    def test_majority_vote_aggregator(self):
        """Test MajorityVoteAggregator."""
        aggregator = MajorityVoteAggregator()

        # Test with majority True
        assert aggregator.aggregate(pd.Series([True, True, False])) is True

        # Test with majority False
        assert aggregator.aggregate(pd.Series([False, False, True])) is False

        # Test exactly 50% (should return False with default threshold)
        assert aggregator.aggregate(pd.Series([True, False])) is False

        # Test with custom threshold
        assert aggregator.aggregate(pd.Series([True, False]), threshold=0.4) is True

        # Test with None values (should be filtered out)
        assert aggregator.aggregate(pd.Series([True, None, True, False])) is True

        # Test with all None
        assert aggregator.aggregate(pd.Series([None, None])) is None

    def test_first_aggregator(self):
        """Test FirstAggregator."""
        aggregator = FirstAggregator()

        # Test with values
        assert aggregator.aggregate(pd.Series([1, 2, 3])) == 1
        assert aggregator.aggregate(pd.Series(["a", "b", "c"])) == "a"

        # Test with leading None
        assert aggregator.aggregate(pd.Series([None, 2, 3])) == 2

        # Test with all None
        assert aggregator.aggregate(pd.Series([None, None])) is None

    def test_count_aggregator(self):
        """Test CountAggregator."""
        aggregator = CountAggregator()

        # Test basic counting
        result = aggregator.aggregate(pd.Series([1, 2, 2, 3, 2]))
        assert result == {1: 1, 2: 3, 3: 1}

        # Test with strings
        result = aggregator.aggregate(pd.Series(["a", "b", "a", "c"]))
        assert result == {"a": 2, "b": 1, "c": 1}

        # Test with None values (default: filter out)
        result = aggregator.aggregate(pd.Series([1, None, 2, 2, None]))
        assert result == {1: 1, 2: 2}

        # Test with include_none=True
        result = aggregator.aggregate(pd.Series([1, None, 2, None]), include_none=True)
        assert result == {1: 1, None: 2, 2: 1}


class TestAggregatorRegistry:
    """Test suite for AggregatorRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving aggregators."""
        registry = AggregatorRegistry()

        # Register an aggregator
        mean_agg = MeanAggregator()
        registry.register("mean", mean_agg)

        # Retrieve it
        retrieved = registry.get("mean")
        assert retrieved is mean_agg

    def test_register_duplicate(self):
        """Test that registering duplicate names raises error."""
        registry = AggregatorRegistry()
        registry.register("mean", MeanAggregator())

        with pytest.raises(ValueError, match="already registered"):
            registry.register("mean", MedianAggregator())

    def test_get_nonexistent(self):
        """Test that getting non-existent aggregator raises error."""
        registry = AggregatorRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_aggregators(self):
        """Test listing aggregators."""
        registry = AggregatorRegistry()
        registry.register("mean", MeanAggregator())
        registry.register("median", MedianAggregator())

        aggregators = registry.list_aggregators()
        assert "mean" in aggregators
        assert "median" in aggregators
        assert len(aggregators) == 2

    def test_contains(self):
        """Test __contains__ operator."""
        registry = AggregatorRegistry()
        registry.register("mean", MeanAggregator())

        assert "mean" in registry
        assert "median" not in registry


class TestDefaultRegistries:
    """Test suite for default registry factories."""

    def test_create_default_aggregator_registry(self):
        """Test that default aggregator registry has all built-in aggregators."""
        registry = create_default_registry()

        # Check all built-in aggregators are registered
        assert "mean" in registry
        assert "median" in registry
        assert "mode" in registry
        assert "majority_vote" in registry
        assert "first" in registry
        assert "count" in registry

        # Verify they work
        # Note: aggregators now work with pandas Series
        import pandas as pd

        mean_agg = registry.get("mean")
        assert mean_agg.aggregate(pd.Series([1, 2, 3])) == 2.0


class TestCustomAggregator:
    """Test that custom aggregators can be created and registered."""

    def test_custom_aggregator(self):
        """Test creating and using a custom aggregator."""

        # Define a custom aggregator
        class CustomAggregator:
            def aggregate(self, values, **_kwargs):
                # Sum of squares
                return sum(v**2 for v in values if v is not None)

        # Register it
        registry = AggregatorRegistry()
        registry.register("sum_of_squares", CustomAggregator())

        # Use it
        aggregator = registry.get("sum_of_squares")
        # Note: aggregators now work with pandas Series
        import pandas as pd

        assert aggregator.aggregate(pd.Series([1, 2, 3])) == 14  # 1 + 4 + 9
