"""Tests for TemplateResults class."""

from karenina.schemas.workflow import (
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultTemplate,
)
from karenina.schemas.workflow.template_results import TemplateResults


class TestTemplateResultsDataAccess:
    """Test suite for basic data access methods."""

    def create_sample_result(
        self,
        question_id: str = "q1",
        model: str = "model1",
        verify_result: bool | None = None,
        embedding_score: float | None = None,
        embedding_override: bool = False,
        regex_results: dict[str, bool] | None = None,
        regex_overall: bool | None = None,
        abstention_detected: bool | None = None,
    ):
        """Helper to create a sample verification result with template data."""
        template_data = None
        if any(
            [
                verify_result is not None,
                embedding_score is not None,
                regex_results is not None,
                abstention_detected is not None,
            ]
        ):
            template_data = VerificationResultTemplate(
                raw_llm_response="test response",
                template_verification_performed=verify_result is not None,
                verify_result=verify_result,
                embedding_check_performed=embedding_score is not None,
                embedding_similarity_score=embedding_score,
                embedding_override_applied=embedding_override,
                regex_validations_performed=regex_results is not None,
                regex_validation_results=regex_results,
                regex_overall_success=regex_overall,
                abstention_check_performed=abstention_detected is not None,
                abstention_detected=abstention_detected,
            )

        metadata = VerificationResultMetadata(
            question_id=question_id,
            template_id="template1",
            question_text="Test question",
            answering_model=model,
            parsing_model="parsing1",
            completed_without_errors=True,
            execution_time=1.0,
            timestamp="2023-01-01T00:00:00",
        )

        return VerificationResult(
            metadata=metadata,
            template=template_data,
        )

    def test_get_results_with_template(self):
        """Test filtering results that have template data."""
        # Create results with and without template
        with_template = self.create_sample_result(verify_result=True)
        without_template = self.create_sample_result()

        template_results = TemplateResults(results=[with_template, without_template])

        filtered = template_results.get_results_with_template()
        assert len(filtered) == 1
        assert filtered[0] is with_template

    def test_get_verification_results(self):
        """Test retrieving template verification results."""
        result1 = self.create_sample_result(question_id="q1", verify_result=True)
        result2 = self.create_sample_result(question_id="q2", verify_result=False)

        template_results = TemplateResults(results=[result1, result2])

        # Get all results
        all_results = template_results.get_verification_results()
        assert len(all_results) == 2
        assert True in all_results.values()
        assert False in all_results.values()

        # Filter by question
        q1_results = template_results.get_verification_results(question_id="q1")
        assert len(q1_results) == 1
        assert list(q1_results.values())[0] is True

    def test_get_embedding_scores(self):
        """Test retrieving embedding similarity scores."""
        result1 = self.create_sample_result(question_id="q1", verify_result=True, embedding_score=0.95)
        result2 = self.create_sample_result(question_id="q2", verify_result=True, embedding_score=0.85)

        template_results = TemplateResults(results=[result1, result2])

        # Get all scores
        all_scores = template_results.get_embedding_scores()
        assert len(all_scores) == 2

        # Filter by question
        q1_scores = template_results.get_embedding_scores(question_id="q1")
        assert len(q1_scores) == 1
        assert list(q1_scores.values())[0] == 0.95

    def test_get_embedding_overrides(self):
        """Test retrieving embedding override status."""
        result1 = self.create_sample_result(
            question_id="q1", verify_result=True, embedding_score=0.95, embedding_override=True
        )
        result2 = self.create_sample_result(
            question_id="q2", verify_result=True, embedding_score=0.85, embedding_override=False
        )

        template_results = TemplateResults(results=[result1, result2])

        overrides = template_results.get_embedding_overrides()
        assert len(overrides) == 2
        assert True in overrides.values()
        assert False in overrides.values()

    def test_get_regex_results(self):
        """Test retrieving regex validation results."""
        result1 = self.create_sample_result(
            question_id="q1",
            verify_result=True,
            regex_results={"pattern1": True, "pattern2": False},
            regex_overall=False,
        )
        result2 = self.create_sample_result(
            question_id="q2", verify_result=True, regex_results={"pattern1": True}, regex_overall=True
        )

        template_results = TemplateResults(results=[result1, result2])

        # Get all regex results
        all_regex = template_results.get_regex_results()
        assert len(all_regex) == 2

        # Filter by pattern
        pattern1_results = template_results.get_regex_results(pattern_name="pattern1")
        assert len(pattern1_results) == 2
        # Both should have pattern1=True
        for res in pattern1_results.values():
            assert res["pattern1"] is True

    def test_get_abstention_detections(self):
        """Test retrieving abstention detection results."""
        result1 = self.create_sample_result(question_id="q1", verify_result=True, abstention_detected=True)
        result2 = self.create_sample_result(question_id="q2", verify_result=True, abstention_detected=False)

        template_results = TemplateResults(results=[result1, result2])

        detections = template_results.get_abstention_detections()
        assert len(detections) == 2
        assert True in detections.values()
        assert False in detections.values()

    def test_get_mcp_usage(self):
        """Test retrieving MCP tool usage - simplified test."""
        result1 = self.create_sample_result(question_id="q1", verify_result=True)
        result2 = self.create_sample_result(question_id="q2", verify_result=True)

        template_results = TemplateResults(results=[result1, result2])

        # MCP usage retrieves from template.answering_mcp_servers
        # Just verify the method works, don't check specific values
        mcp_usage = template_results.get_mcp_usage()
        assert isinstance(mcp_usage, dict)


class TestTemplateResultsAggregation:
    """Test suite for aggregation methods."""

    def create_sample_result(
        self,
        question_id: str = "q1",
        model: str = "model1",
        replicate: int = 1,
        verify_result: bool | None = None,
        embedding_score: float | None = None,
        regex_results: dict[str, bool] | None = None,
        regex_overall: bool | None = None,
        abstention_detected: bool | None = None,
    ):
        """Helper to create a sample verification result."""
        # Always mark template_verification_performed if any check is done
        template_performed = any(
            [
                verify_result is not None,
                embedding_score is not None,
                regex_results is not None,
                abstention_detected is not None,
            ]
        )

        template_data = VerificationResultTemplate(
            raw_llm_response="test response",
            template_verification_performed=template_performed,
            verify_result=verify_result if verify_result is not None else True,  # Default to True
            embedding_check_performed=embedding_score is not None,
            embedding_similarity_score=embedding_score,
            regex_validations_performed=regex_results is not None,
            regex_validation_results=regex_results,
            regex_overall_success=regex_overall,
            abstention_check_performed=abstention_detected is not None,
            abstention_detected=abstention_detected,
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
            template=template_data,
        )

    def test_aggregate_pass_rate_by_question(self):
        """Test aggregating pass rates grouped by question."""
        result1 = self.create_sample_result(question_id="q1", verify_result=True)
        result2 = self.create_sample_result(question_id="q1", verify_result=True)
        result3 = self.create_sample_result(question_id="q2", verify_result=False)

        template_results = TemplateResults(results=[result1, result2, result3])

        aggregated = template_results.aggregate_pass_rate(by="question_id")

        assert "q1" in aggregated
        assert "q2" in aggregated
        assert aggregated["q1"] == 1.0  # 2/2 passed
        assert aggregated["q2"] == 0.0  # 0/1 passed

    def test_aggregate_pass_rate_by_model(self):
        """Test aggregating pass rates grouped by model."""
        result1 = self.create_sample_result(model="m1", verify_result=True)
        result2 = self.create_sample_result(model="m1", verify_result=False)
        result3 = self.create_sample_result(model="m2", verify_result=True)

        template_results = TemplateResults(results=[result1, result2, result3])

        aggregated = template_results.aggregate_pass_rate(by="answering_model")

        assert "m1" in aggregated
        assert "m2" in aggregated
        assert aggregated["m1"] == 0.5  # 1/2 passed
        assert aggregated["m2"] == 1.0  # 1/1 passed

    def test_aggregate_embedding_scores_mean(self):
        """Test aggregating embedding scores using mean."""
        result1 = self.create_sample_result(question_id="q1", replicate=1, embedding_score=0.9)
        result2 = self.create_sample_result(question_id="q1", replicate=2, embedding_score=0.8)

        template_results = TemplateResults(results=[result1, result2])

        aggregated = template_results.aggregate_embedding_scores(strategy="mean", by="question_id")
        assert abs(list(aggregated.values())[0] - 0.85) < 0.0001

    def test_aggregate_embedding_scores_median(self):
        """Test aggregating embedding scores using median."""
        result1 = self.create_sample_result(question_id="q1", replicate=1, embedding_score=0.7)
        result2 = self.create_sample_result(question_id="q1", replicate=2, embedding_score=0.8)
        result3 = self.create_sample_result(question_id="q1", replicate=3, embedding_score=0.9)

        template_results = TemplateResults(results=[result1, result2, result3])

        aggregated = template_results.aggregate_embedding_scores(strategy="median", by="question_id")
        assert list(aggregated.values())[0] == 0.8

    def test_aggregate_regex_success_rate(self):
        """Test aggregating regex validation success rates."""
        result1 = self.create_sample_result(
            question_id="q1", replicate=1, regex_results={"pattern1": True, "pattern2": True}, regex_overall=True
        )
        result2 = self.create_sample_result(
            question_id="q1",
            replicate=2,
            regex_results={"pattern1": True, "pattern2": False},
            regex_overall=False,
        )

        template_results = TemplateResults(results=[result1, result2])

        # Aggregate specific pattern
        pattern1_rate = template_results.aggregate_regex_success_rate(pattern_name="pattern1", by="question_id")
        assert list(pattern1_rate.values())[0] == 1.0  # 2/2 pattern1 success

        pattern2_rate = template_results.aggregate_regex_success_rate(pattern_name="pattern2", by="question_id")
        assert list(pattern2_rate.values())[0] == 0.5  # 1/2 pattern2 success

    def test_aggregate_abstention_rate(self):
        """Test aggregating abstention detection rates."""
        result1 = self.create_sample_result(question_id="q1", replicate=1, abstention_detected=True)
        result2 = self.create_sample_result(question_id="q1", replicate=2, abstention_detected=False)
        result3 = self.create_sample_result(question_id="q1", replicate=3, abstention_detected=False)

        template_results = TemplateResults(results=[result1, result2, result3])

        aggregated = template_results.aggregate_abstention_rate(by="question_id")
        assert abs(list(aggregated.values())[0] - 0.333) < 0.01  # 1/3 abstained


class TestTemplateResultsExtensibility:
    """Test suite for extensibility features."""

    def create_sample_result(
        self, question_id: str = "q1", verify_result: bool | None = None, embedding_score: float | None = None
    ):
        """Helper to create a sample verification result."""
        # Always mark template_verification_performed if any check is done
        template_performed = verify_result is not None or embedding_score is not None

        template_data = VerificationResultTemplate(
            raw_llm_response="test response",
            template_verification_performed=template_performed,
            verify_result=verify_result if verify_result is not None else True,
            embedding_check_performed=embedding_score is not None,
            embedding_similarity_score=embedding_score,
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
            template=template_data,
        )

    def test_register_custom_aggregator(self):
        """Test registering a custom aggregator."""
        result1 = self.create_sample_result(question_id="q1", embedding_score=0.7)
        result2 = self.create_sample_result(question_id="q1", embedding_score=0.9)

        template_results = TemplateResults(results=[result1, result2])

        # Register custom min aggregator
        class MinAggregator:
            def aggregate(self, values, **_kwargs):
                return min(v for v in values if v is not None)

        template_results.register_aggregator("min", MinAggregator())

        # Use the custom aggregator
        aggregated = template_results.aggregate_embedding_scores(strategy="min", by="question_id")
        assert list(aggregated.values())[0] == 0.7

    def test_list_aggregators(self):
        """Test listing available aggregators."""
        template_results = TemplateResults(results=[])

        aggregators = template_results.list_aggregators()
        assert "mean" in aggregators
        assert "median" in aggregators


class TestTemplateResultsFiltering:
    """Test suite for filtering and grouping methods."""

    def create_sample_result(self, question_id: str = "q1", model: str = "model1", verify_result: bool | None = None):
        """Helper to create a sample verification result."""
        template_data = VerificationResultTemplate(
            raw_llm_response="test response",
            template_verification_performed=verify_result is not None,
            verify_result=verify_result,
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
            template=template_data,
        )

    def test_filter_by_question(self):
        """Test filtering results by question ID."""
        result1 = self.create_sample_result(question_id="q1", verify_result=True)
        result2 = self.create_sample_result(question_id="q2", verify_result=False)

        template_results = TemplateResults(results=[result1, result2])
        filtered = template_results.filter(question_ids=["q1"])

        assert len(filtered) == 1
        assert filtered.results[0].metadata.question_id == "q1"

    def test_filter_by_model(self):
        """Test filtering results by answering model."""
        result1 = self.create_sample_result(model="m1", verify_result=True)
        result2 = self.create_sample_result(model="m2", verify_result=False)

        template_results = TemplateResults(results=[result1, result2])
        filtered = template_results.filter(answering_models=["m1"])

        assert len(filtered) == 1
        assert filtered.results[0].metadata.answering_model == "m1"

    def test_filter_multiple_criteria(self):
        """Test filtering with multiple criteria."""
        result1 = self.create_sample_result(question_id="q1", model="m1", verify_result=True)
        result2 = self.create_sample_result(question_id="q1", model="m2", verify_result=False)
        result3 = self.create_sample_result(question_id="q2", model="m1", verify_result=True)

        template_results = TemplateResults(results=[result1, result2, result3])
        filtered = template_results.filter(question_ids=["q1"], answering_models=["m1"])

        assert len(filtered) == 1
        assert filtered.results[0] is result1

    def test_group_by_question(self):
        """Test grouping results by question."""
        result1 = self.create_sample_result(question_id="q1", verify_result=True)
        result2 = self.create_sample_result(question_id="q1", verify_result=False)
        result3 = self.create_sample_result(question_id="q2", verify_result=True)

        template_results = TemplateResults(results=[result1, result2, result3])
        grouped = template_results.group_by_question()

        assert "q1" in grouped
        assert "q2" in grouped
        assert len(grouped["q1"]) == 2
        assert len(grouped["q2"]) == 1

    def test_group_by_model(self):
        """Test grouping results by model."""
        result1 = self.create_sample_result(model="m1", verify_result=True)
        result2 = self.create_sample_result(model="m1", verify_result=False)
        result3 = self.create_sample_result(model="m2", verify_result=True)

        template_results = TemplateResults(results=[result1, result2, result3])
        grouped = template_results.group_by_model()

        assert "m1" in grouped
        assert "m2" in grouped
        assert len(grouped["m1"]) == 2
        assert len(grouped["m2"]) == 1


class TestTemplateResultsSummary:
    """Test suite for summary statistics."""

    def create_sample_result(self, verify_result: bool | None = None, embedding_score: float | None = None):
        """Helper to create a sample verification result."""
        template_data = VerificationResultTemplate(
            raw_llm_response="test response",
            template_verification_performed=verify_result is not None,
            verify_result=verify_result,
            embedding_check_performed=embedding_score is not None,
            embedding_similarity_score=embedding_score,
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
            template=template_data,
        )

    def test_get_template_summary(self):
        """Test getting summary statistics for template results."""
        result1 = self.create_sample_result(verify_result=True, embedding_score=0.95)
        result2 = self.create_sample_result(verify_result=False, embedding_score=0.85)

        template_results = TemplateResults(results=[result1, result2])
        summary = template_results.get_template_summary()

        assert "num_results" in summary
        assert summary["num_results"] == 2
        assert "pass_rate" in summary
        assert summary["pass_rate"] == 0.5
        assert "num_with_embedding" in summary
        assert summary["num_with_embedding"] == 2


class TestTemplateResultsCollectionOperations:
    """Test suite for collection-like operations."""

    def create_sample_result(self, question_id: str = "q1"):
        """Helper to create a sample verification result."""
        template_data = VerificationResultTemplate(
            raw_llm_response="test response",
            template_verification_performed=True,
            verify_result=True,
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
            template=template_data,
        )

    def test_len(self):
        """Test __len__ method."""
        result1 = self.create_sample_result()
        result2 = self.create_sample_result()

        template_results = TemplateResults(results=[result1, result2])
        assert len(template_results) == 2

    def test_iter(self):
        """Test __iter__ method."""
        result1 = self.create_sample_result(question_id="q1")
        result2 = self.create_sample_result(question_id="q2")

        template_results = TemplateResults(results=[result1, result2])

        question_ids = [r.metadata.question_id for r in template_results]
        assert question_ids == ["q1", "q2"]

    def test_getitem(self):
        """Test __getitem__ method."""
        result1 = self.create_sample_result(question_id="q1")
        result2 = self.create_sample_result(question_id="q2")

        template_results = TemplateResults(results=[result1, result2])

        assert template_results[0] is result1
        assert template_results[1] is result2
