"""Tests for the aggregation framework."""

import statistics

import pytest

from karenina.schemas.workflow import VerificationResult, VerificationResultMetadata
from karenina.schemas.workflow.aggregation import (
    AggregatorRegistry,
    CountAggregator,
    FirstAggregator,
    GroupByRegistry,
    ListAggregator,
    MajorityVoteAggregator,
    MeanAggregator,
    MedianAggregator,
    ModeAggregator,
    create_default_groupby_registry,
    create_default_registry,
)


class TestAggregators:
    """Test suite for built-in aggregators."""

    def test_mean_aggregator(self):
        """Test MeanAggregator."""
        aggregator = MeanAggregator()

        # Test with valid values
        assert aggregator.aggregate([1, 2, 3, 4, 5]) == 3.0
        assert aggregator.aggregate([10, 20, 30]) == 20.0

        # Test with None values (should be filtered out)
        assert aggregator.aggregate([1, None, 3, None, 5]) == 3.0

        # Test with all None values
        assert aggregator.aggregate([None, None, None]) is None

        # Test with empty list
        assert aggregator.aggregate([]) is None

    def test_median_aggregator(self):
        """Test MedianAggregator."""
        aggregator = MedianAggregator()

        # Test with odd number of values
        assert aggregator.aggregate([1, 2, 3, 4, 5]) == 3

        # Test with even number of values
        assert aggregator.aggregate([1, 2, 3, 4]) == 2.5

        # Test with None values (should be filtered out)
        assert aggregator.aggregate([1, None, 3, None, 5]) == 3

        # Test with all None values
        assert aggregator.aggregate([None, None]) is None

    def test_mode_aggregator(self):
        """Test ModeAggregator."""
        aggregator = ModeAggregator()

        # Test with clear mode
        assert aggregator.aggregate([1, 2, 2, 3, 2]) == 2
        assert aggregator.aggregate(["a", "b", "b", "c"]) == "b"

        # Test with None values (should be filtered out)
        assert aggregator.aggregate([1, None, 2, 2, None]) == 2

        # Test with no unique mode (should raise error)
        with pytest.raises(statistics.StatisticsError):
            aggregator.aggregate([1, 2])

    def test_majority_vote_aggregator(self):
        """Test MajorityVoteAggregator."""
        aggregator = MajorityVoteAggregator()

        # Test with majority True
        assert aggregator.aggregate([True, True, False]) is True

        # Test with majority False
        assert aggregator.aggregate([False, False, True]) is False

        # Test exactly 50% (should return False with default threshold)
        assert aggregator.aggregate([True, False]) is False

        # Test with custom threshold
        assert aggregator.aggregate([True, False], threshold=0.4) is True

        # Test with None values (should be filtered out)
        assert aggregator.aggregate([True, None, True, False]) is True

        # Test with all None
        assert aggregator.aggregate([None, None]) is None

    def test_list_aggregator(self):
        """Test ListAggregator."""
        aggregator = ListAggregator()

        # Test basic collection
        assert aggregator.aggregate([1, 2, 3]) == [1, 2, 3]

        # Test with None values (default: filter out)
        assert aggregator.aggregate([1, None, 3]) == [1, 3]

        # Test with include_none=True
        assert aggregator.aggregate([1, None, 3], include_none=True) == [1, None, 3]

    def test_first_aggregator(self):
        """Test FirstAggregator."""
        aggregator = FirstAggregator()

        # Test with values
        assert aggregator.aggregate([1, 2, 3]) == 1
        assert aggregator.aggregate(["a", "b", "c"]) == "a"

        # Test with leading None
        assert aggregator.aggregate([None, 2, 3]) == 2

        # Test with all None
        assert aggregator.aggregate([None, None]) is None

    def test_count_aggregator(self):
        """Test CountAggregator."""
        aggregator = CountAggregator()

        # Test basic counting
        result = aggregator.aggregate([1, 2, 2, 3, 2])
        assert result == {1: 1, 2: 3, 3: 1}

        # Test with strings
        result = aggregator.aggregate(["a", "b", "a", "c"])
        assert result == {"a": 2, "b": 1, "c": 1}

        # Test with None values (default: filter out)
        result = aggregator.aggregate([1, None, 2, 2, None])
        assert result == {1: 1, 2: 2}

        # Test with include_none=True
        result = aggregator.aggregate([1, None, 2, None], include_none=True)
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


class TestGroupByStrategies:
    """Test suite for built-in grouping strategies."""

    def create_sample_result(self, question_id: str, model: str, replicate: int = 1):
        """Helper to create a sample verification result."""
        return VerificationResult(
            metadata=VerificationResultMetadata(
                question_id=question_id,
                template_id="test_template",
                question_text="Test question",
                answering_model=model,
                parsing_model="parsing_model",
                answering_replicate=replicate,
                completed_without_errors=True,
                execution_time=1.0,
                timestamp="2023-01-01T00:00:00",
                run_name="test_run",
            )
        )

    def test_by_question_strategy(self):
        """Test ByQuestionStrategy."""
        from karenina.schemas.workflow.aggregation import ByQuestionStrategy

        strategy = ByQuestionStrategy()
        result = self.create_sample_result("q1", "model1")
        assert strategy.get_group_key(result) == "q1"

    def test_by_model_strategy(self):
        """Test ByModelStrategy."""
        from karenina.schemas.workflow.aggregation import ByModelStrategy

        strategy = ByModelStrategy()
        result = self.create_sample_result("q1", "model1")
        assert strategy.get_group_key(result) == "model1"

    def test_by_parsing_model_strategy(self):
        """Test ByParsingModelStrategy."""
        from karenina.schemas.workflow.aggregation import ByParsingModelStrategy

        strategy = ByParsingModelStrategy()
        result = self.create_sample_result("q1", "model1")
        assert strategy.get_group_key(result) == "parsing_model"

    def test_by_replicate_strategy(self):
        """Test ByReplicateStrategy."""
        from karenina.schemas.workflow.aggregation import ByReplicateStrategy

        strategy = ByReplicateStrategy()
        result = self.create_sample_result("q1", "model1", replicate=3)
        assert strategy.get_group_key(result) == "3"

    def test_by_run_name_strategy(self):
        """Test ByRunNameStrategy."""
        from karenina.schemas.workflow.aggregation import ByRunNameStrategy

        strategy = ByRunNameStrategy()
        result = self.create_sample_result("q1", "model1")
        assert strategy.get_group_key(result) == "test_run"

    def test_by_model_pair_strategy(self):
        """Test ByModelPairStrategy."""
        from karenina.schemas.workflow.aggregation import ByModelPairStrategy

        strategy = ByModelPairStrategy()
        result = self.create_sample_result("q1", "model1")
        assert strategy.get_group_key(result) == "model1_parsing_model"


class TestGroupByRegistry:
    """Test suite for GroupByRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving strategies."""
        from karenina.schemas.workflow.aggregation import ByQuestionStrategy

        registry = GroupByRegistry()
        strategy = ByQuestionStrategy()
        registry.register("question", strategy)

        retrieved = registry.get("question")
        assert retrieved is strategy

    def test_register_duplicate(self):
        """Test that registering duplicate names raises error."""
        from karenina.schemas.workflow.aggregation import ByModelStrategy, ByQuestionStrategy

        registry = GroupByRegistry()
        registry.register("test", ByQuestionStrategy())

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", ByModelStrategy())

    def test_get_nonexistent(self):
        """Test that getting non-existent strategy raises error."""
        registry = GroupByRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_strategies(self):
        """Test listing strategies."""
        from karenina.schemas.workflow.aggregation import ByModelStrategy, ByQuestionStrategy

        registry = GroupByRegistry()
        registry.register("question", ByQuestionStrategy())
        registry.register("model", ByModelStrategy())

        strategies = registry.list_strategies()
        assert "question" in strategies
        assert "model" in strategies
        assert len(strategies) == 2


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
        assert "list" in registry
        assert "first" in registry
        assert "count" in registry

        # Verify they work
        mean_agg = registry.get("mean")
        assert mean_agg.aggregate([1, 2, 3]) == 2.0

    def test_create_default_groupby_registry(self):
        """Test that default groupby registry has all built-in strategies."""
        registry = create_default_groupby_registry()

        # Check all built-in strategies are registered
        assert "question" in registry
        assert "model" in registry
        assert "parsing_model" in registry
        assert "replicate" in registry
        assert "run_name" in registry
        assert "model_pair" in registry


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
        assert aggregator.aggregate([1, 2, 3]) == 14  # 1 + 4 + 9


class TestCustomGroupByStrategy:
    """Test that custom grouping strategies can be created and registered."""

    def test_custom_strategy(self):
        """Test creating and using a custom grouping strategy."""

        # Define a custom strategy
        class ByKeywordStrategy:
            def get_group_key(self, result):
                keywords = result.metadata.keywords or []
                return keywords[0] if keywords else "no_keyword"

        # Create test result with keywords
        result = VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="q1",
                template_id="test",
                question_text="Test",
                answering_model="model1",
                parsing_model="model2",
                completed_without_errors=True,
                execution_time=1.0,
                timestamp="2023-01-01T00:00:00",
                keywords=["keyword1", "keyword2"],
            )
        )

        # Register and use it
        registry = GroupByRegistry()
        registry.register("keyword", ByKeywordStrategy())

        strategy = registry.get("keyword")
        assert strategy.get_group_key(result) == "keyword1"
