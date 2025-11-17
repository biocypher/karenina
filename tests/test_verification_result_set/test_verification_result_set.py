"""
Unit tests for VerificationResultSet class.

This test module covers:
- Accessor methods (get_rubrics_results, get_template_results, get_judgment_results)
- Filtering functionality
- Grouping operations
- Summary and utility methods
- Collection operations
- Legacy dict conversion
"""

from karenina.schemas.workflow import (
    VerificationResult,
    VerificationResultDeepJudgment,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultSet,
    VerificationResultTemplate,
)
from karenina.schemas.workflow.judgment_results import JudgmentResults
from karenina.schemas.workflow.rubric_results import RubricResults
from karenina.schemas.workflow.template_results import TemplateResults


class TestHelpers:
    """Helper methods for creating test data."""

    @staticmethod
    def create_sample_result(
        question_id: str = "q1",
        answering_model: str = "model1",
        parsing_model: str = "parser1",
        answering_replicate: int = 1,
        parsing_replicate: int = 1,
        completed: bool = True,
        has_template: bool = True,
        has_rubric: bool = True,
        has_judgment: bool = True,
        verify_result: bool = True,
        llm_scores: dict[str, int] | None = None,
        excerpts: dict[str, list[dict]] | None = None,
        timestamp: str | None = None,
    ) -> VerificationResult:
        """Create a sample VerificationResult with configurable data."""
        # Create metadata
        metadata = VerificationResultMetadata(
            question_id=question_id,
            template_id="test_template_id",
            question_text=f"Question text for {question_id}",
            answering_model=answering_model,
            parsing_model=parsing_model,
            answering_replicate=answering_replicate,
            parsing_replicate=parsing_replicate,
            completed_without_errors=completed,
            execution_time=1.5,
            timestamp=timestamp or "2024-01-01T00:00:00Z",
        )

        # Create template data
        template = None
        if has_template:
            template = VerificationResultTemplate(
                raw_llm_response="test response",
                template_verification_performed=True,
                verify_result=verify_result,
            )

        # Create rubric data
        rubric = None
        if has_rubric:
            rubric = VerificationResultRubric(
                rubric_evaluation_performed=True,
                llm_trait_scores=llm_scores or {"trait1": 5, "trait2": 4},
            )

        # Create judgment data
        deep_judgment = None
        if has_judgment:
            deep_judgment = VerificationResultDeepJudgment(
                deep_judgment_enabled=True,
                deep_judgment_performed=True,
                extracted_excerpts=excerpts or {"attr1": [{"text": "excerpt"}]},
            )

        return VerificationResult(
            metadata=metadata,
            template=template,
            rubric=rubric,
            deep_judgment=deep_judgment,
        )


class TestVerificationResultSetAccessors:
    """Test accessor methods for specialized result views."""

    def test_get_rubrics_results(self):
        """Test get_rubrics_results returns RubricResults instance."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", llm_scores={"trait1": 5}),
            TestHelpers.create_sample_result(question_id="q2", llm_scores={"trait1": 4}),
        ]
        result_set = VerificationResultSet(results=results)

        rubric_results = result_set.get_rubrics_results()

        assert isinstance(rubric_results, RubricResults)
        assert len(rubric_results) == 2

    def test_get_template_results(self):
        """Test get_template_results returns TemplateResults instance."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", verify_result=True),
            TestHelpers.create_sample_result(question_id="q2", verify_result=False),
        ]
        result_set = VerificationResultSet(results=results)

        template_results = result_set.get_template_results()

        assert isinstance(template_results, TemplateResults)
        assert len(template_results) == 2

    def test_get_judgment_results(self):
        """Test get_judgment_results returns JudgmentResults instance."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", excerpts={"attr1": [{"text": "excerpt1"}]}),
            TestHelpers.create_sample_result(question_id="q2", excerpts={"attr2": [{"text": "excerpt2"}]}),
        ]
        result_set = VerificationResultSet(results=results)

        judgment_results = result_set.get_judgment_results()

        assert isinstance(judgment_results, JudgmentResults)
        assert len(judgment_results) == 2

    def test_accessor_methods_preserve_filters(self):
        """Test that accessor methods work on filtered result sets."""
        results = [
            TestHelpers.create_sample_result(question_id="q1"),
            TestHelpers.create_sample_result(question_id="q2"),
            TestHelpers.create_sample_result(question_id="q3"),
        ]
        result_set = VerificationResultSet(results=results)

        # Filter then access
        filtered = result_set.filter(question_ids=["q1", "q2"])
        rubric_results = filtered.get_rubrics_results()

        assert len(rubric_results) == 2


class TestVerificationResultSetFiltering:
    """Test filtering operations."""

    def test_filter_by_question_ids(self):
        """Test filtering by question IDs."""
        results = [
            TestHelpers.create_sample_result(question_id="q1"),
            TestHelpers.create_sample_result(question_id="q2"),
            TestHelpers.create_sample_result(question_id="q3"),
        ]
        result_set = VerificationResultSet(results=results)

        filtered = result_set.filter(question_ids=["q1", "q2"])

        assert len(filtered) == 2
        assert all(r.metadata.question_id in ["q1", "q2"] for r in filtered.results)

    def test_filter_by_answering_models(self):
        """Test filtering by answering models."""
        results = [
            TestHelpers.create_sample_result(answering_model="gpt-4"),
            TestHelpers.create_sample_result(answering_model="claude-3"),
            TestHelpers.create_sample_result(answering_model="gpt-4"),
        ]
        result_set = VerificationResultSet(results=results)

        filtered = result_set.filter(answering_models=["gpt-4"])

        assert len(filtered) == 2
        assert all(r.metadata.answering_model == "gpt-4" for r in filtered.results)

    def test_filter_by_parsing_models(self):
        """Test filtering by parsing models."""
        results = [
            TestHelpers.create_sample_result(parsing_model="parser1"),
            TestHelpers.create_sample_result(parsing_model="parser2"),
            TestHelpers.create_sample_result(parsing_model="parser1"),
        ]
        result_set = VerificationResultSet(results=results)

        filtered = result_set.filter(parsing_models=["parser1"])

        assert len(filtered) == 2
        assert all(r.metadata.parsing_model == "parser1" for r in filtered.results)

    def test_filter_by_replicates(self):
        """Test filtering by replicate numbers."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", answering_replicate=1, parsing_replicate=1),
            TestHelpers.create_sample_result(question_id="q2", answering_replicate=2, parsing_replicate=2),
            TestHelpers.create_sample_result(question_id="q3", answering_replicate=3, parsing_replicate=3),
        ]
        result_set = VerificationResultSet(results=results)

        filtered = result_set.filter(replicates=[1, 2])

        assert len(filtered) == 2
        assert all(
            r.metadata.answering_replicate in [1, 2] or r.metadata.parsing_replicate in [1, 2] for r in filtered.results
        )

    def test_filter_completed_only(self):
        """Test filtering for only completed results."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", completed=True),
            TestHelpers.create_sample_result(question_id="q2", completed=False),
            TestHelpers.create_sample_result(question_id="q3", completed=True),
        ]
        result_set = VerificationResultSet(results=results)

        filtered = result_set.filter(completed_only=True)

        assert len(filtered) == 2
        assert all(r.metadata.completed_without_errors for r in filtered.results)

    def test_filter_has_template(self):
        """Test filtering for results with template verification."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", has_template=True),
            TestHelpers.create_sample_result(question_id="q2", has_template=False),
            TestHelpers.create_sample_result(question_id="q3", has_template=True),
        ]
        result_set = VerificationResultSet(results=results)

        filtered = result_set.filter(has_template=True)

        assert len(filtered) == 2
        assert all(r.template and r.template.template_verification_performed for r in filtered.results)

    def test_filter_has_rubric(self):
        """Test filtering for results with rubric evaluation."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", has_rubric=True),
            TestHelpers.create_sample_result(question_id="q2", has_rubric=False),
            TestHelpers.create_sample_result(question_id="q3", has_rubric=True),
        ]
        result_set = VerificationResultSet(results=results)

        filtered = result_set.filter(has_rubric=True)

        assert len(filtered) == 2
        assert all(r.rubric and r.rubric.rubric_evaluation_performed for r in filtered.results)

    def test_filter_has_judgment(self):
        """Test filtering for results with deep judgment."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", has_judgment=True),
            TestHelpers.create_sample_result(question_id="q2", has_judgment=False),
            TestHelpers.create_sample_result(question_id="q3", has_judgment=True),
        ]
        result_set = VerificationResultSet(results=results)

        filtered = result_set.filter(has_judgment=True)

        assert len(filtered) == 2
        assert all(r.deep_judgment and r.deep_judgment.deep_judgment_performed for r in filtered.results)

    def test_filter_multiple_criteria(self):
        """Test combining multiple filter criteria."""
        results = [
            TestHelpers.create_sample_result(
                question_id="q1", answering_model="gpt-4", completed=True, has_rubric=True
            ),
            TestHelpers.create_sample_result(
                question_id="q2", answering_model="claude-3", completed=True, has_rubric=True
            ),
            TestHelpers.create_sample_result(
                question_id="q1", answering_model="gpt-4", completed=False, has_rubric=True
            ),
            TestHelpers.create_sample_result(
                question_id="q1", answering_model="gpt-4", completed=True, has_rubric=False
            ),
        ]
        result_set = VerificationResultSet(results=results)

        filtered = result_set.filter(
            question_ids=["q1"], answering_models=["gpt-4"], completed_only=True, has_rubric=True
        )

        assert len(filtered) == 1
        result = filtered.results[0]
        assert result.metadata.question_id == "q1"
        assert result.metadata.answering_model == "gpt-4"
        assert result.metadata.completed_without_errors is True
        assert result.rubric and result.rubric.rubric_evaluation_performed

    def test_filter_returns_new_result_set(self):
        """Test that filter returns a new VerificationResultSet instance."""
        results = [TestHelpers.create_sample_result(question_id="q1")]
        result_set = VerificationResultSet(results=results)

        filtered = result_set.filter(question_ids=["q1"])

        assert isinstance(filtered, VerificationResultSet)
        assert filtered is not result_set


class TestVerificationResultSetGrouping:
    """Test grouping operations."""

    def test_group_by_question(self):
        """Test grouping results by question ID."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", answering_model="model1"),
            TestHelpers.create_sample_result(question_id="q1", answering_model="model2"),
            TestHelpers.create_sample_result(question_id="q2", answering_model="model1"),
            TestHelpers.create_sample_result(question_id="q2", answering_model="model2"),
        ]
        result_set = VerificationResultSet(results=results)

        grouped = result_set.group_by_question()

        assert len(grouped) == 2
        assert "q1" in grouped
        assert "q2" in grouped
        assert len(grouped["q1"]) == 2
        assert len(grouped["q2"]) == 2
        assert all(isinstance(rs, VerificationResultSet) for rs in grouped.values())

    def test_group_by_model(self):
        """Test grouping results by answering model."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", answering_model="gpt-4"),
            TestHelpers.create_sample_result(question_id="q2", answering_model="gpt-4"),
            TestHelpers.create_sample_result(question_id="q1", answering_model="claude-3"),
            TestHelpers.create_sample_result(question_id="q2", answering_model="claude-3"),
        ]
        result_set = VerificationResultSet(results=results)

        grouped = result_set.group_by_model()

        assert len(grouped) == 2
        assert "gpt-4" in grouped
        assert "claude-3" in grouped
        assert len(grouped["gpt-4"]) == 2
        assert len(grouped["claude-3"]) == 2
        assert all(isinstance(rs, VerificationResultSet) for rs in grouped.values())

    def test_group_by_replicate(self):
        """Test grouping results by replicate number."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", answering_replicate=1),
            TestHelpers.create_sample_result(question_id="q2", answering_replicate=1),
            TestHelpers.create_sample_result(question_id="q1", answering_replicate=2),
            TestHelpers.create_sample_result(question_id="q2", answering_replicate=2),
        ]
        result_set = VerificationResultSet(results=results)

        grouped = result_set.group_by_replicate()

        assert len(grouped) == 2
        assert 1 in grouped
        assert 2 in grouped
        assert len(grouped[1]) == 2
        assert len(grouped[2]) == 2
        assert all(isinstance(rs, VerificationResultSet) for rs in grouped.values())

    def test_group_by_replicate_with_none(self):
        """Test grouping by replicate when some results have None replicate."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", answering_replicate=1),
            TestHelpers.create_sample_result(question_id="q2", answering_replicate=None),
        ]
        # Manually set answering_replicate to None for the second result
        results[1].metadata.answering_replicate = None

        result_set = VerificationResultSet(results=results)

        grouped = result_set.group_by_replicate()

        # Results with None replicate are grouped under 0
        assert 0 in grouped
        assert 1 in grouped
        assert len(grouped[0]) == 1
        assert len(grouped[1]) == 1

    def test_grouping_preserves_result_data(self):
        """Test that grouping preserves all result data."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", answering_model="model1", llm_scores={"trait1": 5}),
            TestHelpers.create_sample_result(question_id="q1", answering_model="model2", llm_scores={"trait1": 4}),
        ]
        result_set = VerificationResultSet(results=results)

        grouped = result_set.group_by_question()

        # Check that rubric data is preserved
        rubric_results = grouped["q1"].get_rubrics_results()
        scores = rubric_results.get_llm_trait_scores(trait_name="trait1")
        assert len(scores) == 2


class TestVerificationResultSetSummary:
    """Test summary and utility methods."""

    def test_get_summary(self):
        """Test get_summary returns correct statistics."""
        results = [
            TestHelpers.create_sample_result(
                question_id="q1",
                answering_model="model1",
                answering_replicate=1,
                completed=True,
                has_template=True,
                has_rubric=True,
                has_judgment=True,
            ),
            TestHelpers.create_sample_result(
                question_id="q2",
                answering_model="model2",
                answering_replicate=2,
                completed=False,
                has_template=False,
                has_rubric=True,
                has_judgment=False,
            ),
            TestHelpers.create_sample_result(
                question_id="q1",
                answering_model="model1",
                answering_replicate=2,
                completed=True,
                has_template=True,
                has_rubric=False,
                has_judgment=True,
            ),
        ]
        result_set = VerificationResultSet(results=results)

        summary = result_set.get_summary()

        assert summary["num_results"] == 3
        assert summary["num_completed"] == 2
        assert summary["num_with_template"] == 2
        assert summary["num_with_rubric"] == 2
        assert summary["num_with_judgment"] == 2
        assert summary["num_questions"] == 2
        assert summary["num_models"] == 2
        assert summary["num_replicates"] == 2

    def test_get_question_ids(self):
        """Test get_question_ids returns unique sorted question IDs."""
        results = [
            TestHelpers.create_sample_result(question_id="q3"),
            TestHelpers.create_sample_result(question_id="q1"),
            TestHelpers.create_sample_result(question_id="q2"),
            TestHelpers.create_sample_result(question_id="q1"),  # Duplicate
        ]
        result_set = VerificationResultSet(results=results)

        question_ids = result_set.get_question_ids()

        assert question_ids == ["q1", "q2", "q3"]

    def test_get_model_names(self):
        """Test get_model_names returns unique sorted model names."""
        results = [
            TestHelpers.create_sample_result(answering_model="gpt-4"),
            TestHelpers.create_sample_result(answering_model="claude-3"),
            TestHelpers.create_sample_result(answering_model="gpt-4"),  # Duplicate
            TestHelpers.create_sample_result(answering_model="gemini"),
        ]
        result_set = VerificationResultSet(results=results)

        model_names = result_set.get_model_names()

        assert model_names == ["claude-3", "gemini", "gpt-4"]

    def test_get_summary_empty_result_set(self):
        """Test get_summary on empty result set."""
        result_set = VerificationResultSet(results=[])

        summary = result_set.get_summary()

        assert summary["num_results"] == 0
        assert summary["num_completed"] == 0
        assert summary["num_with_template"] == 0
        assert summary["num_with_rubric"] == 0
        assert summary["num_with_judgment"] == 0
        assert summary["num_questions"] == 0
        assert summary["num_models"] == 0
        assert summary["num_replicates"] == 0


class TestVerificationResultSetCollectionOps:
    """Test collection operations (__len__, __iter__, __getitem__, __repr__)."""

    def test_len(self):
        """Test __len__ returns correct count."""
        results = [
            TestHelpers.create_sample_result(question_id="q1"),
            TestHelpers.create_sample_result(question_id="q2"),
            TestHelpers.create_sample_result(question_id="q3"),
        ]
        result_set = VerificationResultSet(results=results)

        assert len(result_set) == 3

    def test_iter(self):
        """Test __iter__ allows iteration over results."""
        results = [
            TestHelpers.create_sample_result(question_id="q1"),
            TestHelpers.create_sample_result(question_id="q2"),
        ]
        result_set = VerificationResultSet(results=results)

        iterated = list(result_set)

        assert len(iterated) == 2
        assert all(isinstance(r, VerificationResult) for r in iterated)
        assert iterated[0].metadata.question_id == "q1"
        assert iterated[1].metadata.question_id == "q2"

    def test_getitem_positive_index(self):
        """Test __getitem__ with positive index."""
        results = [
            TestHelpers.create_sample_result(question_id="q1"),
            TestHelpers.create_sample_result(question_id="q2"),
            TestHelpers.create_sample_result(question_id="q3"),
        ]
        result_set = VerificationResultSet(results=results)

        assert result_set[0].metadata.question_id == "q1"
        assert result_set[1].metadata.question_id == "q2"
        assert result_set[2].metadata.question_id == "q3"

    def test_getitem_negative_index(self):
        """Test __getitem__ with negative index."""
        results = [
            TestHelpers.create_sample_result(question_id="q1"),
            TestHelpers.create_sample_result(question_id="q2"),
            TestHelpers.create_sample_result(question_id="q3"),
        ]
        result_set = VerificationResultSet(results=results)

        assert result_set[-1].metadata.question_id == "q3"
        assert result_set[-2].metadata.question_id == "q2"
        assert result_set[-3].metadata.question_id == "q1"

    def test_repr(self):
        """Test __repr__ provides useful string representation."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", answering_model="model1"),
            TestHelpers.create_sample_result(question_id="q2", answering_model="model2"),
            TestHelpers.create_sample_result(question_id="q1", answering_model="model1"),
        ]
        result_set = VerificationResultSet(results=results)

        repr_str = repr(result_set)

        assert "VerificationResultSet" in repr_str
        assert "results=3" in repr_str
        assert "questions=2" in repr_str
        assert "models=2" in repr_str


class TestVerificationResultSetLegacy:
    """Test legacy compatibility methods."""

    def test_to_legacy_dict_basic(self):
        """Test to_legacy_dict creates proper legacy format keys."""
        results = [
            TestHelpers.create_sample_result(
                question_id="q1",
                answering_model="gpt-4",
                parsing_model="parser1",
                answering_replicate=1,
                timestamp="2024-01-01T12:00:00Z",
            ),
        ]
        result_set = VerificationResultSet(results=results)

        legacy_dict = result_set.to_legacy_dict()

        # Should have one key
        assert len(legacy_dict) == 1

        # Key should be in format: question_answering_parsing_repN_timestamp
        key = list(legacy_dict.keys())[0]
        assert key.startswith("q1_gpt-4_parser1_rep1_")

        # Value should be the VerificationResult
        assert isinstance(legacy_dict[key], VerificationResult)
        assert legacy_dict[key].metadata.question_id == "q1"

    def test_to_legacy_dict_multiple_results(self):
        """Test to_legacy_dict with multiple results."""
        results = [
            TestHelpers.create_sample_result(
                question_id="q1",
                answering_model="model1",
                parsing_model="parser1",
                answering_replicate=1,
            ),
            TestHelpers.create_sample_result(
                question_id="q2",
                answering_model="model2",
                parsing_model="parser2",
                answering_replicate=2,
            ),
        ]
        result_set = VerificationResultSet(results=results)

        legacy_dict = result_set.to_legacy_dict()

        assert len(legacy_dict) == 2

        # Check that keys contain the right components
        keys = list(legacy_dict.keys())
        assert any("q1" in key and "model1" in key and "parser1" in key for key in keys)
        assert any("q2" in key and "model2" in key and "parser2" in key for key in keys)

    def test_to_legacy_dict_without_replicate(self):
        """Test to_legacy_dict when result has no replicate number."""
        results = [
            TestHelpers.create_sample_result(
                question_id="q1",
                answering_model="model1",
                parsing_model="parser1",
                answering_replicate=None,
            ),
        ]
        # Manually set replicate to None
        results[0].metadata.answering_replicate = None

        result_set = VerificationResultSet(results=results)

        legacy_dict = result_set.to_legacy_dict()

        # Should still work, just without "repN" in the key
        assert len(legacy_dict) == 1
        key = list(legacy_dict.keys())[0]
        assert "q1_model1_parser1_" in key
        assert "rep" not in key

    def test_to_legacy_dict_preserves_result_data(self):
        """Test that to_legacy_dict preserves all result data."""
        results = [
            TestHelpers.create_sample_result(
                question_id="q1",
                llm_scores={"trait1": 5},
                verify_result=True,
                excerpts={"attr1": [{"text": "excerpt"}]},
            ),
        ]
        result_set = VerificationResultSet(results=results)

        legacy_dict = result_set.to_legacy_dict()

        # Get the single result
        result = list(legacy_dict.values())[0]

        # Verify all data is preserved
        assert result.rubric.llm_trait_scores["trait1"] == 5
        assert result.template.verify_result is True
        assert result.deep_judgment.extracted_excerpts["attr1"][0]["text"] == "excerpt"


class TestVerificationResultSetIntegration:
    """Integration tests combining multiple operations."""

    def test_filter_then_group(self):
        """Test filtering followed by grouping."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", answering_model="gpt-4", completed=True),
            TestHelpers.create_sample_result(question_id="q2", answering_model="gpt-4", completed=False),
            TestHelpers.create_sample_result(question_id="q1", answering_model="claude-3", completed=True),
            TestHelpers.create_sample_result(question_id="q2", answering_model="claude-3", completed=True),
        ]
        result_set = VerificationResultSet(results=results)

        # Filter to completed only, then group by question
        filtered = result_set.filter(completed_only=True)
        grouped = filtered.group_by_question()

        assert len(grouped) == 2
        assert len(grouped["q1"]) == 2
        assert len(grouped["q2"]) == 1

    def test_group_then_access_specialized_results(self):
        """Test grouping followed by accessing specialized result views."""
        results = [
            TestHelpers.create_sample_result(question_id="q1", llm_scores={"trait1": 5}),
            TestHelpers.create_sample_result(question_id="q1", llm_scores={"trait1": 4}),
            TestHelpers.create_sample_result(question_id="q2", llm_scores={"trait1": 3}),
        ]
        result_set = VerificationResultSet(results=results)

        # Group by question
        grouped = result_set.group_by_question()

        # Access rubric results for each group
        q1_rubrics = grouped["q1"].get_rubrics_results()
        q2_rubrics = grouped["q2"].get_rubrics_results()

        assert len(q1_rubrics) == 2
        assert len(q2_rubrics) == 1

    def test_complex_workflow(self):
        """Test a complex workflow combining multiple operations."""
        results = [
            TestHelpers.create_sample_result(
                question_id="q1",
                answering_model="gpt-4",
                answering_replicate=1,
                llm_scores={"trait1": 5},
                verify_result=True,
            ),
            TestHelpers.create_sample_result(
                question_id="q1",
                answering_model="gpt-4",
                answering_replicate=2,
                llm_scores={"trait1": 4},
                verify_result=True,
            ),
            TestHelpers.create_sample_result(
                question_id="q1",
                answering_model="claude-3",
                answering_replicate=1,
                llm_scores={"trait1": 3},
                verify_result=False,
            ),
            TestHelpers.create_sample_result(
                question_id="q2",
                answering_model="gpt-4",
                answering_replicate=1,
                llm_scores={"trait1": 5},
                verify_result=True,
            ),
        ]
        result_set = VerificationResultSet(results=results)

        # Filter to q1 only
        q1_results = result_set.filter(question_ids=["q1"])
        assert len(q1_results) == 3

        # Group by model
        by_model = q1_results.group_by_model()
        assert len(by_model) == 2

        # Get template results for gpt-4
        gpt4_templates = by_model["gpt-4"].get_template_results()
        verification_results = gpt4_templates.get_verification_results()
        assert len(verification_results) == 2
        assert all(v for v in verification_results.values())

        # Get summary for claude-3
        claude_summary = by_model["claude-3"].get_summary()
        assert claude_summary["num_results"] == 1
