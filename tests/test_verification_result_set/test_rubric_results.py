"""Tests for RubricResults class."""

from karenina.schemas.workflow import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultRubric,
)
from karenina.schemas.workflow.rubric_results import RubricResults


class TestRubricResultsDataAccess:
    """Test suite for basic data access methods."""

    def create_sample_result(
        self,
        question_id: str = "q1",
        model: str = "model1",
        llm_traits: dict[str, int] | None = None,
        regex_traits: dict[str, bool] | None = None,
        callable_traits: dict[str, bool | int] | None = None,
        metric_traits: dict[str, dict[str, float]] | None = None,
        confusion_matrix: dict[str, dict[str, list[str]]] | None = None,
    ):
        """Helper to create a sample verification result with rubric data."""
        rubric_eval = None
        if (
            llm_traits is not None
            or regex_traits is not None
            or callable_traits is not None
            or metric_traits is not None
        ):
            rubric_eval = VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores=llm_traits,
                regex_trait_scores=regex_traits,
                callable_trait_scores=callable_traits,
                metric_trait_scores=metric_traits,
                metric_trait_confusion_lists=confusion_matrix,
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
            rubric=rubric_eval,
        )

    def test_get_results_with_rubric(self):
        """Test filtering results that have rubric data."""
        # Create results with and without rubric
        with_rubric = self.create_sample_result(llm_traits={"trait1": 4})
        without_rubric = self.create_sample_result()

        rubric_results = RubricResults(results=[with_rubric, without_rubric])

        filtered = rubric_results.get_results_with_rubric()
        assert len(filtered) == 1
        assert filtered[0] is with_rubric

    def test_get_llm_trait_scores(self):
        """Test retrieving LLM trait scores."""
        result1 = self.create_sample_result(question_id="q1", model="m1", llm_traits={"clarity": 4, "accuracy": 5})
        result2 = self.create_sample_result(question_id="q2", model="m1", llm_traits={"clarity": 3, "accuracy": 4})

        rubric_results = RubricResults(results=[result1, result2])

        # Get all scores
        all_scores = rubric_results.get_llm_trait_scores()
        assert len(all_scores) == 2
        assert "clarity" in list(all_scores.values())[0]
        assert "accuracy" in list(all_scores.values())[0]

        # Filter by question
        q1_scores = rubric_results.get_llm_trait_scores(question_id="q1")
        assert len(q1_scores) == 1
        assert list(q1_scores.values())[0]["clarity"] == 4

        # Filter by trait name
        clarity_scores = rubric_results.get_llm_trait_scores(trait_name="clarity")
        assert len(clarity_scores) == 2
        for scores in clarity_scores.values():
            assert "clarity" in scores
            assert "accuracy" not in scores

    def test_get_regex_trait_scores(self):
        """Test retrieving regex trait scores."""
        result1 = self.create_sample_result(question_id="q1", regex_traits={"correct": True, "complete": False})
        result2 = self.create_sample_result(question_id="q2", regex_traits={"correct": False})

        rubric_results = RubricResults(results=[result1, result2])

        # Get all scores
        all_scores = rubric_results.get_regex_trait_scores()
        assert len(all_scores) == 2

        # Filter by question
        q1_scores = rubric_results.get_regex_trait_scores(question_id="q1")
        assert len(q1_scores) == 1
        assert list(q1_scores.values())[0]["correct"] is True

        # Filter by trait name
        correct_scores = rubric_results.get_regex_trait_scores(trait_name="correct")
        assert len(correct_scores) == 2

    def test_get_metric_trait_scores(self):
        """Test retrieving metric trait scores."""
        result1 = self.create_sample_result(
            question_id="q1", metric_traits={"precision": {"value": 0.95}, "recall": {"value": 0.85}}
        )
        result2 = self.create_sample_result(question_id="q2", metric_traits={"precision": {"value": 0.90}})

        rubric_results = RubricResults(results=[result1, result2])

        # Get all scores
        all_scores = rubric_results.get_metric_trait_scores()
        assert len(all_scores) == 2

        # Filter by trait name
        precision_scores = rubric_results.get_metric_trait_scores(trait_name="precision")
        assert len(precision_scores) == 2

    def test_get_all_trait_scores(self):
        """Test retrieving all trait scores together."""
        result = self.create_sample_result(
            llm_traits={"clarity": 4},
            regex_traits={"correct": True},
            metric_traits={"precision": {"value": 0.95}},
        )

        rubric_results = RubricResults(results=[result])
        all_scores = rubric_results.get_all_trait_scores()

        assert len(all_scores) == 1
        result_scores = list(all_scores.values())[0]
        assert "clarity" in result_scores
        assert "precision" in result_scores

    def test_get_confusion_matrices(self):
        """Test retrieving confusion matrices for metric traits."""
        result1 = self.create_sample_result(
            question_id="q1",
            metric_traits={"trait1": {"precision": 0.9}},
            confusion_matrix={"trait1": {"tp": ["excerpt1", "excerpt2"], "fp": ["excerpt3"], "tn": [], "fn": []}},
        )
        result2 = self.create_sample_result(
            question_id="q2",
            metric_traits={"trait1": {"precision": 0.8}},
            confusion_matrix={"trait1": {"tp": ["excerpt4"], "fp": [], "tn": ["excerpt5"], "fn": []}},
        )

        rubric_results = RubricResults(results=[result1, result2])

        matrices = rubric_results.get_confusion_matrices()
        assert len(matrices) == 2

        # Filter by trait name
        trait1_matrices = rubric_results.get_confusion_matrices(trait_name="trait1")
        assert len(trait1_matrices) == 2


class TestRubricResultsAggregation:
    """Test suite for aggregation methods."""

    def create_sample_result(
        self,
        question_id: str = "q1",
        model: str = "model1",
        replicate: int = 1,
        llm_traits: dict[str, int] | None = None,
        regex_traits: dict[str, bool] | None = None,
        callable_traits: dict[str, bool | int] | None = None,
        metric_traits: dict[str, dict[str, float]] | None = None,
    ):
        """Helper to create a sample verification result."""
        rubric_eval = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores=llm_traits,
            regex_trait_scores=regex_traits,
            callable_trait_scores=callable_traits,
            metric_trait_scores=metric_traits,
        )

        return VerificationResult(
            metadata=VerificationResultMetadata(
                question_id=question_id,
                template_id="template1",
                question_text="Test question",
                answering_model=model,
                parsing_model="parsing1",
                replicate=replicate,
                completed_without_errors=True,
                execution_time=1.0,
                timestamp="2023-01-01T00:00:00",
            ),
            rubric=rubric_eval,
        )

    def test_aggregate_llm_traits_by_question(self):
        """Test aggregating LLM traits grouped by question."""
        result1 = self.create_sample_result(question_id="q1", llm_traits={"clarity": 4})
        result2 = self.create_sample_result(question_id="q1", llm_traits={"clarity": 5})
        result3 = self.create_sample_result(question_id="q2", llm_traits={"clarity": 3})

        rubric_results = RubricResults(results=[result1, result2, result3])

        # Aggregate by question using mean
        aggregated = rubric_results.aggregate_llm_traits(strategy="mean", by="question_id")

        assert "q1" in aggregated
        assert "q2" in aggregated
        assert aggregated["q1"]["clarity"] == 4.5  # (4 + 5) / 2
        assert aggregated["q2"]["clarity"] == 3.0

    def test_aggregate_llm_traits_by_model(self):
        """Test aggregating LLM traits grouped by model."""
        result1 = self.create_sample_result(model="m1", llm_traits={"clarity": 4})
        result2 = self.create_sample_result(model="m1", llm_traits={"clarity": 5})
        result3 = self.create_sample_result(model="m2", llm_traits={"clarity": 3})

        rubric_results = RubricResults(results=[result1, result2, result3])

        # Aggregate by model
        aggregated = rubric_results.aggregate_llm_traits(strategy="mean", by="answering_model")

        assert "m1" in aggregated
        assert "m2" in aggregated
        assert aggregated["m1"]["clarity"] == 4.5
        assert aggregated["m2"]["clarity"] == 3.0

    def test_aggregate_llm_traits_median(self):
        """Test aggregating LLM traits using median strategy."""
        result1 = self.create_sample_result(llm_traits={"clarity": 2})
        result2 = self.create_sample_result(llm_traits={"clarity": 4})
        result3 = self.create_sample_result(llm_traits={"clarity": 5})

        rubric_results = RubricResults(results=[result1, result2, result3])

        aggregated = rubric_results.aggregate_llm_traits(strategy="median", by="question_id")
        assert list(aggregated.values())[0]["clarity"] == 4

    def test_aggregate_regex_traits_majority_vote(self):
        """Test aggregating regex traits using majority vote."""
        result1 = self.create_sample_result(regex_traits={"correct": True})
        result2 = self.create_sample_result(regex_traits={"correct": True})
        result3 = self.create_sample_result(regex_traits={"correct": False})

        rubric_results = RubricResults(results=[result1, result2, result3])

        aggregated = rubric_results.aggregate_regex_traits(strategy="majority_vote", by="question_id")
        assert list(aggregated.values())[0]["correct"] is True

    def test_aggregate_metric_traits_mean(self):
        """Test aggregating metric traits using mean."""
        result1 = self.create_sample_result(metric_traits={"feature_detection": {"precision": 0.9, "recall": 0.85}})
        result2 = self.create_sample_result(metric_traits={"feature_detection": {"precision": 0.8, "recall": 0.75}})

        rubric_results = RubricResults(results=[result1, result2])

        # Aggregate precision metric
        aggregated = rubric_results.aggregate_metric_traits(metric_name="precision", strategy="mean", by="question_id")
        assert abs(list(aggregated.values())[0]["feature_detection"] - 0.85) < 0.0001


class TestRubricResultsExtensibility:
    """Test suite for extensibility features."""

    def create_sample_result(self, question_id: str = "q1", llm_traits: dict[str, int] | None = None):
        """Helper to create a sample verification result."""
        rubric_eval = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores=llm_traits or {},
            manual_trait_scores={},
            metric_trait_scores={},
            confusion_matrices={},
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
            rubric=rubric_eval,
        )

    def test_register_custom_aggregator(self):
        """Test registering a custom aggregator."""
        result1 = self.create_sample_result(llm_traits={"clarity": 2})
        result2 = self.create_sample_result(llm_traits={"clarity": 4})

        rubric_results = RubricResults(results=[result1, result2])

        # Register custom max aggregator
        class MaxAggregator:
            def aggregate(self, values, **_kwargs):
                return max(v for v in values if v is not None)

        rubric_results.register_aggregator("max", MaxAggregator())

        # Use the custom aggregator
        aggregated = rubric_results.aggregate_llm_traits(strategy="max", by="question_id")
        assert list(aggregated.values())[0]["clarity"] == 4

    def test_list_aggregators(self):
        """Test listing available aggregators."""
        rubric_results = RubricResults(results=[])

        aggregators = rubric_results.list_aggregators()
        assert "mean" in aggregators
        assert "median" in aggregators
        assert "majority_vote" in aggregators


class TestRubricResultsFiltering:
    """Test suite for filtering and grouping methods."""

    def create_sample_result(
        self, question_id: str = "q1", model: str = "model1", llm_traits: dict[str, int] | None = None
    ):
        """Helper to create a sample verification result."""
        rubric_eval = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores=llm_traits or {},
            manual_trait_scores={},
            metric_trait_scores={},
            confusion_matrices={},
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
            rubric=rubric_eval,
        )

    def test_filter_by_question(self):
        """Test filtering results by question ID."""
        result1 = self.create_sample_result(question_id="q1")
        result2 = self.create_sample_result(question_id="q2")

        rubric_results = RubricResults(results=[result1, result2])
        filtered = rubric_results.filter(question_ids=["q1"])

        assert len(filtered) == 1
        assert filtered.results[0].metadata.question_id == "q1"

    def test_filter_by_model(self):
        """Test filtering results by answering model."""
        result1 = self.create_sample_result(model="m1")
        result2 = self.create_sample_result(model="m2")

        rubric_results = RubricResults(results=[result1, result2])
        filtered = rubric_results.filter(answering_models=["m1"])

        assert len(filtered) == 1
        assert filtered.results[0].metadata.answering_model == "m1"

    def test_filter_multiple_criteria(self):
        """Test filtering with multiple criteria."""
        result1 = self.create_sample_result(question_id="q1", model="m1")
        result2 = self.create_sample_result(question_id="q1", model="m2")
        result3 = self.create_sample_result(question_id="q2", model="m1")

        rubric_results = RubricResults(results=[result1, result2, result3])
        filtered = rubric_results.filter(question_ids=["q1"], answering_models=["m1"])

        assert len(filtered) == 1
        assert filtered.results[0] is result1

    def test_group_by_question(self):
        """Test grouping results by question."""
        result1 = self.create_sample_result(question_id="q1")
        result2 = self.create_sample_result(question_id="q1")
        result3 = self.create_sample_result(question_id="q2")

        rubric_results = RubricResults(results=[result1, result2, result3])
        grouped = rubric_results.group_by_question()

        assert "q1" in grouped
        assert "q2" in grouped
        assert len(grouped["q1"]) == 2
        assert len(grouped["q2"]) == 1

    def test_group_by_model(self):
        """Test grouping results by model."""
        result1 = self.create_sample_result(model="m1")
        result2 = self.create_sample_result(model="m1")
        result3 = self.create_sample_result(model="m2")

        rubric_results = RubricResults(results=[result1, result2, result3])
        grouped = rubric_results.group_by_model()

        assert "m1" in grouped
        assert "m2" in grouped
        assert len(grouped["m1"]) == 2
        assert len(grouped["m2"]) == 1


class TestRubricResultsSummary:
    """Test suite for summary statistics."""

    def create_sample_result(self, llm_traits: dict[str, int] | None = None):
        """Helper to create a sample verification result."""
        rubric_eval = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores=llm_traits or {},
            manual_trait_scores={},
            metric_trait_scores={},
            confusion_matrices={},
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
            rubric=rubric_eval,
        )

    def test_get_trait_summary(self):
        """Test getting summary statistics for traits."""
        result1 = self.create_sample_result(llm_traits={"clarity": 4, "accuracy": 5})
        result2 = self.create_sample_result(llm_traits={"clarity": 5, "accuracy": 4})

        rubric_results = RubricResults(results=[result1, result2])
        summary = rubric_results.get_trait_summary()

        assert "num_results" in summary
        assert summary["num_results"] == 2
        assert "llm_traits" in summary
        # llm_traits is a list of trait names
        assert "clarity" in summary["llm_traits"]
        assert "accuracy" in summary["llm_traits"]


class TestRubricResultsCollectionOperations:
    """Test suite for collection-like operations."""

    def create_sample_result(self, question_id: str = "q1"):
        """Helper to create a sample verification result."""
        rubric_eval = VerificationResultRubric(
            rubric_evaluation_performed=True,
            llm_trait_scores={},
            manual_trait_scores={},
            metric_trait_scores={},
            confusion_matrices={},
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
            rubric=rubric_eval,
        )

    def test_len(self):
        """Test __len__ method."""
        result1 = self.create_sample_result()
        result2 = self.create_sample_result()

        rubric_results = RubricResults(results=[result1, result2])
        assert len(rubric_results) == 2

    def test_iter(self):
        """Test __iter__ method."""
        result1 = self.create_sample_result(question_id="q1")
        result2 = self.create_sample_result(question_id="q2")

        rubric_results = RubricResults(results=[result1, result2])

        question_ids = [r.metadata.question_id for r in rubric_results]
        assert question_ids == ["q1", "q2"]

    def test_getitem(self):
        """Test __getitem__ method."""
        result1 = self.create_sample_result(question_id="q1")
        result2 = self.create_sample_result(question_id="q2")

        rubric_results = RubricResults(results=[result1, result2])

        assert rubric_results[0] is result1
        assert rubric_results[1] is result2
