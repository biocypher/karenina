"""Tests for JudgmentResults class."""

from typing import Any

from karenina.schemas.workflow import (
    VerificationResult,
    VerificationResultDeepJudgment,
    VerificationResultMetadata,
)
from karenina.schemas.workflow.judgment_results import JudgmentResults


class TestJudgmentResultsDataAccess:
    """Test suite for basic data access methods."""

    def create_sample_result(
        self,
        question_id: str = "q1",
        model: str = "model1",
        excerpts: dict[str, list[dict[str, Any]]] | None = None,
        reasoning: dict[str, str] | None = None,
        hallucination_risks: dict[str, str] | None = None,
        attributes_without_excerpts: list[str] | None = None,
        model_calls: int = 0,
        search_enabled: bool = False,
    ):
        """Helper to create a sample verification result with deep judgment data."""
        judgment_data = None
        if any([excerpts, reasoning, hallucination_risks, attributes_without_excerpts, model_calls > 0]):
            judgment_data = VerificationResultDeepJudgment(
                deep_judgment_enabled=True,
                deep_judgment_performed=True,
                extracted_excerpts=excerpts,
                attribute_reasoning=reasoning,
                hallucination_risk_assessment=hallucination_risks,
                attributes_without_excerpts=attributes_without_excerpts,
                deep_judgment_model_calls=model_calls,
                deep_judgment_search_enabled=search_enabled,
            )

        return VerificationResult(
            metadata=VerificationResultMetadata(
                question_id=question_id,
                template_id="template1",
                question_text="Test question",
                answering_model=model,
                parsing_model="parsing1",
                completed_without_errors=True,
                execution_time=1.0,
                timestamp="2023-01-01T00:00:00",
            ),
            deep_judgment=judgment_data,
        )

    def test_get_results_with_judgment(self):
        """Test filtering results that have deep judgment data."""
        with_judgment = self.create_sample_result(excerpts={"attr1": [{"text": "excerpt1"}]})
        without_judgment = self.create_sample_result()

        judgment_results = JudgmentResults(results=[with_judgment, without_judgment])

        filtered = judgment_results.get_results_with_judgment()
        assert len(filtered) == 1
        assert filtered[0] is with_judgment

    def test_get_extracted_excerpts(self):
        """Test retrieving extracted excerpts."""
        result1 = self.create_sample_result(
            question_id="q1",
            excerpts={
                "attr1": [{"text": "excerpt1", "confidence": "high"}],
                "attr2": [{"text": "excerpt2", "confidence": "medium"}],
            },
        )
        result2 = self.create_sample_result(
            question_id="q2", excerpts={"attr1": [{"text": "excerpt3", "confidence": "low"}]}
        )

        judgment_results = JudgmentResults(results=[result1, result2])

        # Get all excerpts
        all_excerpts = judgment_results.get_extracted_excerpts()
        assert len(all_excerpts) == 2

        # Filter by question
        q1_excerpts = judgment_results.get_extracted_excerpts(question_id="q1")
        assert len(q1_excerpts) == 1
        assert "attr1" in list(q1_excerpts.values())[0]
        assert "attr2" in list(q1_excerpts.values())[0]

        # Filter by attribute
        attr1_excerpts = judgment_results.get_extracted_excerpts(attribute_name="attr1")
        assert len(attr1_excerpts) == 2
        for excerpts in attr1_excerpts.values():
            assert "attr1" in excerpts
            assert "attr2" not in excerpts

    def test_get_attribute_reasoning(self):
        """Test retrieving attribute reasoning traces."""
        result1 = self.create_sample_result(question_id="q1", reasoning={"attr1": "reasoning1", "attr2": "reasoning2"})
        result2 = self.create_sample_result(question_id="q2", reasoning={"attr1": "reasoning3"})

        judgment_results = JudgmentResults(results=[result1, result2])

        # Get all reasoning
        all_reasoning = judgment_results.get_attribute_reasoning()
        assert len(all_reasoning) == 2

        # Filter by question
        q1_reasoning = judgment_results.get_attribute_reasoning(question_id="q1")
        assert len(q1_reasoning) == 1

        # Filter by attribute
        attr1_reasoning = judgment_results.get_attribute_reasoning(attribute_name="attr1")
        assert len(attr1_reasoning) == 2

    def test_get_hallucination_risks(self):
        """Test retrieving hallucination risk assessments."""
        result1 = self.create_sample_result(
            question_id="q1",
            hallucination_risks={"attr1": "low", "attr2": "medium"},
            search_enabled=True,
        )
        result2 = self.create_sample_result(
            question_id="q2", hallucination_risks={"attr1": "high"}, search_enabled=True
        )

        judgment_results = JudgmentResults(results=[result1, result2])

        # Get all risks
        all_risks = judgment_results.get_hallucination_risks()
        assert len(all_risks) == 2

        # Filter by attribute
        attr1_risks = judgment_results.get_hallucination_risks(attribute_name="attr1")
        assert len(attr1_risks) == 2

    def test_get_attributes_without_excerpts(self):
        """Test retrieving attributes without excerpts."""
        result1 = self.create_sample_result(question_id="q1", attributes_without_excerpts=["attr1", "attr2"])
        result2 = self.create_sample_result(question_id="q2", attributes_without_excerpts=["attr3"])

        judgment_results = JudgmentResults(results=[result1, result2])

        without_excerpts = judgment_results.get_attributes_without_excerpts()
        assert len(without_excerpts) == 2

    def test_get_processing_metrics(self):
        """Test retrieving processing metrics."""
        result1 = self.create_sample_result(question_id="q1", model_calls=5)
        result2 = self.create_sample_result(question_id="q2", model_calls=3)

        judgment_results = JudgmentResults(results=[result1, result2])

        metrics = judgment_results.get_processing_metrics()
        assert len(metrics) == 2


class TestJudgmentResultsAggregation:
    """Test suite for aggregation methods."""

    def create_sample_result(
        self,
        question_id: str = "q1",
        model: str = "model1",
        replicate: int = 1,
        excerpts: dict[str, list[dict[str, Any]]] | None = None,
        hallucination_risks: dict[str, str] | None = None,
        model_calls: int = 0,
    ):
        """Helper to create a sample verification result."""
        judgment_data = VerificationResultDeepJudgment(
            deep_judgment_enabled=True,
            deep_judgment_performed=True,
            extracted_excerpts=excerpts or {},
            hallucination_risk_assessment=hallucination_risks,
            deep_judgment_model_calls=model_calls,
        )

        return VerificationResult(
            metadata=VerificationResultMetadata(
                question_id=question_id,
                template_id="template1",
                question_text="Test question",
                answering_model=model,
                parsing_model="parsing1",
                answering_replicate=replicate,
                completed_without_errors=True,
                execution_time=1.0,
                timestamp="2023-01-01T00:00:00",
            ),
            deep_judgment=judgment_data,
        )

    def test_aggregate_excerpt_counts_by_question(self):
        """Test aggregating excerpt counts grouped by question."""
        result1 = self.create_sample_result(
            question_id="q1",
            replicate=1,
            excerpts={"attr1": [{"text": "e1"}, {"text": "e2"}], "attr2": [{"text": "e3"}]},
        )
        result2 = self.create_sample_result(
            question_id="q1",
            replicate=2,
            excerpts={"attr1": [{"text": "e4"}], "attr2": [{"text": "e5"}, {"text": "e6"}]},
        )
        result3 = self.create_sample_result(question_id="q2", replicate=1, excerpts={"attr1": [{"text": "e7"}]})

        judgment_results = JudgmentResults(results=[result1, result2, result3])

        # Aggregate by question using mean
        aggregated = judgment_results.aggregate_excerpt_counts(strategy="mean", by="question")

        assert "q1" in aggregated
        assert "q2" in aggregated
        # q1: attr1 has (2+1)/2 = 1.5, attr2 has (1+2)/2 = 1.5
        assert aggregated["q1"]["attr1"] == 1.5
        assert aggregated["q1"]["attr2"] == 1.5
        # q2: attr1 has 1
        assert aggregated["q2"]["attr1"] == 1.0

    def test_aggregate_excerpt_counts_by_model(self):
        """Test aggregating excerpt counts grouped by model."""
        result1 = self.create_sample_result(model="m1", excerpts={"attr1": [{"text": "e1"}, {"text": "e2"}]})
        result2 = self.create_sample_result(model="m1", excerpts={"attr1": [{"text": "e3"}]})
        result3 = self.create_sample_result(
            model="m2", excerpts={"attr1": [{"text": "e4"}, {"text": "e5"}, {"text": "e6"}]}
        )

        judgment_results = JudgmentResults(results=[result1, result2, result3])

        aggregated = judgment_results.aggregate_excerpt_counts(strategy="mean", by="model")

        assert "m1" in aggregated
        assert "m2" in aggregated
        assert aggregated["m1"]["attr1"] == 1.5  # (2+1)/2
        assert aggregated["m2"]["attr1"] == 3.0  # 3/1

    def test_aggregate_hallucination_risk_distribution(self):
        """Test aggregating hallucination risk distributions."""
        result1 = self.create_sample_result(
            question_id="q1",
            replicate=1,
            hallucination_risks={"attr1": "low", "attr2": "medium"},
        )
        result2 = self.create_sample_result(
            question_id="q1",
            replicate=2,
            hallucination_risks={"attr1": "high", "attr2": "low"},
        )

        judgment_results = JudgmentResults(results=[result1, result2])

        # Aggregate by question - should get distribution of risk levels
        aggregated = judgment_results.aggregate_hallucination_risk_distribution(by="question")

        assert "q1" in aggregated
        # For attr1: 1 low, 1 high -> {"low": 0.5, "high": 0.5}
        # For attr2: 1 medium, 1 low -> {"medium": 0.5, "low": 0.5}
        assert "attr1" in aggregated["q1"]
        assert "attr2" in aggregated["q1"]

    def test_aggregate_model_calls(self):
        """Test aggregating model call counts."""
        result1 = self.create_sample_result(question_id="q1", replicate=1, model_calls=5)
        result2 = self.create_sample_result(question_id="q1", replicate=2, model_calls=3)
        result3 = self.create_sample_result(question_id="q2", replicate=1, model_calls=7)

        judgment_results = JudgmentResults(results=[result1, result2, result3])

        # Aggregate by question using mean
        aggregated = judgment_results.aggregate_model_calls(strategy="mean", by="question")

        assert aggregated["q1"] == 4.0  # (5+3)/2
        assert aggregated["q2"] == 7.0  # 7/1


class TestJudgmentResultsExtensibility:
    """Test suite for extensibility features."""

    def create_sample_result(self, question_id: str = "q1", excerpts: dict[str, list[dict[str, Any]]] | None = None):
        """Helper to create a sample verification result."""
        judgment_data = VerificationResultDeepJudgment(
            deep_judgment_enabled=True,
            deep_judgment_performed=True,
            extracted_excerpts=excerpts or {},
        )

        return VerificationResult(
            metadata=VerificationResultMetadata(
                question_id=question_id,
                template_id="template1",
                question_text="Test question",
                answering_model="model1",
                parsing_model="parsing1",
                completed_without_errors=True,
                execution_time=1.0,
                timestamp="2023-01-01T00:00:00",
            ),
            deep_judgment=judgment_data,
        )

    def test_register_custom_aggregator(self):
        """Test registering and listing custom aggregators."""
        judgment_results = JudgmentResults(results=[])

        # Register custom aggregator
        class CustomAggregator:
            def aggregate(self, values, **_kwargs):
                return sum(v for v in values if v is not None)

        judgment_results.register_aggregator("custom_sum", CustomAggregator())

        # Verify it's in the list
        aggregators = judgment_results.list_aggregators()
        assert "custom_sum" in aggregators
        assert "mean" in aggregators  # Built-in still exists

    def test_list_aggregators(self):
        """Test listing available aggregators."""
        judgment_results = JudgmentResults(results=[])

        aggregators = judgment_results.list_aggregators()
        assert "mean" in aggregators
        assert "median" in aggregators

    def test_register_custom_groupby_strategy(self):
        """Test registering a custom groupby strategy."""
        result1 = self.create_sample_result(question_id="q1", excerpts={"attr1": [{"text": "e1"}]})
        result2 = self.create_sample_result(question_id="q2", excerpts={"attr1": [{"text": "e2"}]})

        judgment_results = JudgmentResults(results=[result1, result2])

        # Register a constant grouping strategy
        class AllStrategy:
            def get_group_key(self, _result):
                return "all"

        judgment_results.register_groupby_strategy("all", AllStrategy())

        # Use the custom strategy
        aggregated = judgment_results.aggregate_excerpt_counts(strategy="mean", by="all")
        assert "all" in aggregated

    def test_list_groupby_strategies(self):
        """Test listing available groupby strategies."""
        judgment_results = JudgmentResults(results=[])

        strategies = judgment_results.list_groupby_strategies()
        assert "question" in strategies
        assert "model" in strategies


class TestJudgmentResultsFiltering:
    """Test suite for filtering and grouping methods."""

    def create_sample_result(
        self, question_id: str = "q1", model: str = "model1", excerpts: dict[str, list[dict[str, Any]]] | None = None
    ):
        """Helper to create a sample verification result."""
        judgment_data = VerificationResultDeepJudgment(
            deep_judgment_enabled=True,
            deep_judgment_performed=True,
            extracted_excerpts=excerpts or {},
        )

        return VerificationResult(
            metadata=VerificationResultMetadata(
                question_id=question_id,
                template_id="template1",
                question_text="Test question",
                answering_model=model,
                parsing_model="parsing1",
                completed_without_errors=True,
                execution_time=1.0,
                timestamp="2023-01-01T00:00:00",
            ),
            deep_judgment=judgment_data,
        )

    def test_filter_by_question(self):
        """Test filtering results by question ID."""
        result1 = self.create_sample_result(question_id="q1", excerpts={"attr1": [{"text": "e1"}]})
        result2 = self.create_sample_result(question_id="q2", excerpts={"attr1": [{"text": "e2"}]})

        judgment_results = JudgmentResults(results=[result1, result2])
        filtered = judgment_results.filter(question_ids=["q1"])

        assert len(filtered) == 1
        assert filtered.results[0].metadata.question_id == "q1"

    def test_filter_by_model(self):
        """Test filtering results by answering model."""
        result1 = self.create_sample_result(model="m1", excerpts={"attr1": [{"text": "e1"}]})
        result2 = self.create_sample_result(model="m2", excerpts={"attr1": [{"text": "e2"}]})

        judgment_results = JudgmentResults(results=[result1, result2])
        filtered = judgment_results.filter(answering_models=["m1"])

        assert len(filtered) == 1
        assert filtered.results[0].metadata.answering_model == "m1"

    def test_filter_multiple_criteria(self):
        """Test filtering with multiple criteria."""
        result1 = self.create_sample_result(question_id="q1", model="m1", excerpts={"attr1": [{"text": "e1"}]})
        result2 = self.create_sample_result(question_id="q1", model="m2", excerpts={"attr1": [{"text": "e2"}]})
        result3 = self.create_sample_result(question_id="q2", model="m1", excerpts={"attr1": [{"text": "e3"}]})

        judgment_results = JudgmentResults(results=[result1, result2, result3])
        filtered = judgment_results.filter(question_ids=["q1"], answering_models=["m1"])

        assert len(filtered) == 1
        assert filtered.results[0] is result1

    def test_group_by_question(self):
        """Test grouping results by question."""
        result1 = self.create_sample_result(question_id="q1", excerpts={"attr1": [{"text": "e1"}]})
        result2 = self.create_sample_result(question_id="q1", excerpts={"attr1": [{"text": "e2"}]})
        result3 = self.create_sample_result(question_id="q2", excerpts={"attr1": [{"text": "e3"}]})

        judgment_results = JudgmentResults(results=[result1, result2, result3])
        grouped = judgment_results.group_by_question()

        assert "q1" in grouped
        assert "q2" in grouped
        assert len(grouped["q1"]) == 2
        assert len(grouped["q2"]) == 1

    def test_group_by_model(self):
        """Test grouping results by model."""
        result1 = self.create_sample_result(model="m1", excerpts={"attr1": [{"text": "e1"}]})
        result2 = self.create_sample_result(model="m1", excerpts={"attr1": [{"text": "e2"}]})
        result3 = self.create_sample_result(model="m2", excerpts={"attr1": [{"text": "e3"}]})

        judgment_results = JudgmentResults(results=[result1, result2, result3])
        grouped = judgment_results.group_by_model()

        assert "m1" in grouped
        assert "m2" in grouped
        assert len(grouped["m1"]) == 2
        assert len(grouped["m2"]) == 1


class TestJudgmentResultsSummary:
    """Test suite for summary statistics."""

    def create_sample_result(
        self,
        excerpts: dict[str, list[dict[str, Any]]] | None = None,
        model_calls: int = 0,
        search_enabled: bool = False,
    ):
        """Helper to create a sample verification result."""
        judgment_data = VerificationResultDeepJudgment(
            deep_judgment_enabled=True,
            deep_judgment_performed=True,
            extracted_excerpts=excerpts or {},
            deep_judgment_model_calls=model_calls,
            deep_judgment_search_enabled=search_enabled,
        )

        return VerificationResult(
            metadata=VerificationResultMetadata(
                question_id="q1",
                template_id="template1",
                question_text="Test question",
                answering_model="model1",
                parsing_model="parsing1",
                completed_without_errors=True,
                execution_time=1.0,
                timestamp="2023-01-01T00:00:00",
            ),
            deep_judgment=judgment_data,
        )

    def test_get_judgment_summary(self):
        """Test getting summary statistics for judgment results."""
        result1 = self.create_sample_result(
            excerpts={"attr1": [{"text": "e1"}, {"text": "e2"}]}, model_calls=5, search_enabled=True
        )
        result2 = self.create_sample_result(excerpts={"attr1": [{"text": "e3"}]}, model_calls=3)

        judgment_results = JudgmentResults(results=[result1, result2])
        summary = judgment_results.get_judgment_summary()

        assert "num_results" in summary
        assert summary["num_results"] == 2
        assert "attributes" in summary
        assert "attr1" in summary["attributes"]
        assert "num_with_search" in summary
        assert summary["num_with_search"] == 1
        assert "mean_model_calls" in summary
        assert summary["mean_model_calls"] == 4.0  # (5+3)/2


class TestJudgmentResultsCollectionOperations:
    """Test suite for collection-like operations."""

    def create_sample_result(self, question_id: str = "q1"):
        """Helper to create a sample verification result."""
        judgment_data = VerificationResultDeepJudgment(
            deep_judgment_enabled=True,
            deep_judgment_performed=True,
            extracted_excerpts={"attr1": [{"text": "excerpt"}]},
        )

        return VerificationResult(
            metadata=VerificationResultMetadata(
                question_id=question_id,
                template_id="template1",
                question_text="Test question",
                answering_model="model1",
                parsing_model="parsing1",
                completed_without_errors=True,
                execution_time=1.0,
                timestamp="2023-01-01T00:00:00",
            ),
            deep_judgment=judgment_data,
        )

    def test_len(self):
        """Test __len__ method."""
        result1 = self.create_sample_result()
        result2 = self.create_sample_result()

        judgment_results = JudgmentResults(results=[result1, result2])
        assert len(judgment_results) == 2

    def test_iter(self):
        """Test __iter__ method."""
        result1 = self.create_sample_result(question_id="q1")
        result2 = self.create_sample_result(question_id="q2")

        judgment_results = JudgmentResults(results=[result1, result2])

        question_ids = [r.metadata.question_id for r in judgment_results]
        assert question_ids == ["q1", "q2"]

    def test_getitem(self):
        """Test __getitem__ method."""
        result1 = self.create_sample_result(question_id="q1")
        result2 = self.create_sample_result(question_id="q2")

        judgment_results = JudgmentResults(results=[result1, result2])

        assert judgment_results[0] is result1
        assert judgment_results[1] is result2
