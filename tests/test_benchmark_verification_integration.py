"""Tests for the enhanced verification integration in the Benchmark class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from karenina.benchmark.benchmark import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig, VerificationResult
from karenina.schemas.domain import RubricTrait


@pytest.fixture
def sample_benchmark() -> None:
    """Create a sample benchmark for testing."""
    benchmark = Benchmark.create(
        name="Test Benchmark",
        description="A test benchmark for verification integration",
        version="1.0.0",
        creator="Test Suite",
    )

    # Add some test questions with templates
    q1_id = benchmark.add_question(
        question="What is 2 + 2?",
        raw_answer="4",
        answer_template="""class Answer(BaseAnswer):
    result: int = Field(description="The arithmetic result")
    correct: dict = Field(description="Correct answer")

    def verify(self) -> bool:
        return self.result == 4
""",
        finished=True,
    )

    q2_id = benchmark.add_question(
        question="What is the capital of France?",
        raw_answer="Paris",
        answer_template="""class Answer(BaseAnswer):
    capital: str = Field(description="The capital city")
    correct: dict = Field(description="Correct answer")

    def verify(self) -> bool:
        return self.capital.lower() == "paris"
""",
        finished=True,
    )

    q3_id = benchmark.add_question(
        question="Unfinished question",
        raw_answer="Some answer",
        finished=False,
    )

    return benchmark, [q1_id, q2_id, q3_id]


@pytest.fixture
def sample_config() -> None:
    """Create a sample verification configuration."""
    answering_model = ModelConfig(
        id="test-answering",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.1,
        interface="langchain",
        system_prompt="You are a helpful assistant.",
    )

    parsing_model = ModelConfig(
        id="test-parsing",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.0,
        interface="langchain",
        system_prompt="Parse the response according to the template.",
    )

    return VerificationConfig(
        answering_models=[answering_model],
        parsing_models=[parsing_model],
        replicate_count=1,
    )


class TestBenchmarkVerificationIntegration:
    """Test suite for benchmark verification integration."""

    def test_template_validation_integration(self, sample_benchmark) -> None:
        """Test that template validation is properly integrated."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        # Test valid template
        valid_template = """class Answer(BaseAnswer):
    response: str = Field(description="A response")
    correct: dict = Field(description="Correct answer")

    def verify(self) -> bool:
        return True
"""
        # Should not raise an exception
        benchmark.add_answer_template(q3_id, valid_template)

        # Test invalid template (missing verify method)
        invalid_template = """class Answer(BaseAnswer):
    response: str = Field(description="A response")
    correct: dict = Field(description="Correct answer")
"""
        with pytest.raises(ValueError, match="Invalid template"):
            benchmark.add_answer_template(q3_id, invalid_template)

    def test_benchmark_validation_with_real_templates(self, sample_benchmark) -> None:
        """Test benchmark validation using real template validation."""
        benchmark, _ = sample_benchmark

        # Should validate successfully
        is_valid, error_msg = benchmark.validate()
        assert is_valid
        assert error_msg == "Benchmark is valid"

    @patch("karenina.benchmark.core.verification_manager.run_question_verification")
    def test_run_verification_basic(self, mock_run_verification, sample_benchmark, sample_config) -> None:
        """Test basic run_verification functionality."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        # Mock the verification results
        mock_verification_result = VerificationResult(
            question_id=q1_id,
            template_id="no_template",
            success=True,
            question_text="What is 2 + 2?",
            raw_llm_response="The answer is 4",
            answering_model="openai/gpt-4.1-mini",
            parsing_model="openai/gpt-4.1-mini",
            execution_time=1.5,
            timestamp="2023-01-01T00:00:00",
        )

        mock_run_verification.return_value = {f"{q1_id}_test": mock_verification_result}

        # Run verification on specific question
        results = benchmark.run_verification(
            config=sample_config,
            question_ids=[q1_id],
        )

        # Verify the call was made correctly
        mock_run_verification.assert_called_once()
        call_args = mock_run_verification.call_args
        assert call_args[1]["question_id"] == q1_id
        assert call_args[1]["question_text"] == "What is 2 + 2?"
        assert call_args[1]["config"] == sample_config

        # Verify results
        assert len(results) == 1
        assert f"{q1_id}_test" in results
        assert results[f"{q1_id}_test"].success

    def test_verify_question_single(self, sample_benchmark, sample_config) -> None:
        """Test single question verification."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        with patch("karenina.benchmark.core.verification_manager.run_question_verification") as mock_verify:
            mock_result = VerificationResult(
                question_id=q1_id,
                template_id="no_template",
                success=True,
                question_text="What is 2 + 2?",
                raw_llm_response="4",
                answering_model="test",
                parsing_model="test",
                execution_time=1.0,
                timestamp="2023-01-01T00:00:00",
            )
            mock_verify.return_value = {f"{q1_id}_result": mock_result}

            results = benchmark.verify_question(q1_id, sample_config)

            assert len(results) == 1
            assert f"{q1_id}_result" in results

    def test_verify_filtered(self, sample_benchmark, sample_config) -> None:
        """Test filtered verification."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        with patch("karenina.benchmark.core.verification_manager.run_question_verification") as mock_verify:
            mock_verify.return_value = {}

            # Test filtering by finished status
            benchmark.verify_filtered(
                config=sample_config,
                finished=True,
            )

            # Should have been called twice (for q1 and q2, not q3 since it's unfinished)
            assert mock_verify.call_count == 2

    def test_verify_dry_run(self, sample_benchmark, sample_config) -> None:
        """Test dry run verification."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        # Dry run should not call actual verification
        with patch("karenina.benchmark.core.verification_manager.run_question_verification") as mock_verify:
            results = benchmark.verify_dry_run(sample_config)

            # Should not call actual verification
            mock_verify.assert_not_called()

            # Should return readiness status for finished questions
            assert q1_id in results
            assert q2_id in results
            assert q3_id not in results  # Unfinished questions not included

            # Questions with valid templates should be ready
            assert results[q1_id] is True
            assert results[q2_id] is True

    def test_rubric_merging(self, sample_benchmark) -> None:
        """Test that global and question-specific rubrics are properly merged."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        # Add global rubric
        global_trait = RubricTrait(
            name="clarity",
            description="Is the answer clear?",
            kind="boolean",
        )
        benchmark.add_global_rubric_trait(global_trait)

        # Add question-specific rubric
        question_trait = RubricTrait(
            name="accuracy",
            description="Is the answer accurate?",
            kind="score",
            min_score=1,
            max_score=5,
        )
        benchmark.add_question_rubric_trait(q1_id, question_trait)

        # Test rubric merging
        merged_rubric = benchmark._get_merged_rubric_for_question(q1_id)
        assert merged_rubric is not None
        assert len(merged_rubric.traits) == 2

        trait_names = [t.name for t in merged_rubric.traits]
        assert "clarity" in trait_names
        assert "accuracy" in trait_names

        # Test question without specific rubric (should only have global)
        merged_rubric_q2 = benchmark._get_merged_rubric_for_question(q2_id)
        assert merged_rubric_q2 is not None
        assert len(merged_rubric_q2.traits) == 1
        assert merged_rubric_q2.traits[0].name == "clarity"

    def test_verification_result_storage(self, sample_benchmark, sample_config):  # noqa -> None: ARG002
        """Test verification result storage and retrieval."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        # Create some mock results
        result1 = VerificationResult(
            question_id=q1_id,
            template_id="no_template",
            success=True,
            question_text="What is 2 + 2?",
            raw_llm_response="4",
            answering_model="test",
            parsing_model="test",
            execution_time=1.0,
            timestamp="2023-01-01T00:00:00",
        )

        result2 = VerificationResult(
            question_id=q2_id,
            template_id="no_template",
            success=False,
            error="Some error",
            question_text="What is the capital of France?",
            raw_llm_response="",
            answering_model="test",
            parsing_model="test",
            execution_time=0.5,
            timestamp="2023-01-01T00:01:00",
        )

        results = {
            f"{q1_id}_test": result1,
            f"{q2_id}_test": result2,
        }

        # Store results
        benchmark.store_verification_results(results, "test_run")

        # Retrieve all results
        retrieved_results = benchmark.get_verification_results()
        assert len(retrieved_results) == 2

        # Retrieve specific run
        run_results = benchmark.get_verification_results(run_name="test_run")
        assert len(run_results) == 2

        # Retrieve specific question
        question_results = benchmark.get_verification_results(question_ids=[q1_id])
        assert len(question_results) == 1
        assert question_results[f"{q1_id}_test"].question_id == q1_id

    def test_verification_history(self, sample_benchmark) -> None:
        """Test verification history tracking."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        # Store results for multiple runs
        result1 = VerificationResult(
            question_id=q1_id,
            template_id="no_template",
            success=True,
            question_text="What is 2 + 2?",
            raw_llm_response="4",
            answering_model="test",
            parsing_model="test",
            execution_time=1.0,
            timestamp="2023-01-01T00:00:00",
        )

        result2 = VerificationResult(
            question_id=q1_id,
            template_id="no_template",
            success=False,
            error="Different error",
            question_text="What is 2 + 2?",
            raw_llm_response="",
            answering_model="test",
            parsing_model="test",
            execution_time=0.8,
            timestamp="2023-01-01T00:02:00",
        )

        benchmark.store_verification_results({f"{q1_id}_run1": result1}, "run1")
        benchmark.store_verification_results({f"{q1_id}_run2": result2}, "run2")

        # Get history for specific question
        history = benchmark.get_verification_history(q1_id)
        assert len(history) == 2
        assert "run1" in history
        assert "run2" in history

        # Get full history
        full_history = benchmark.get_verification_history()
        assert len(full_history) == 2

    def test_clear_verification_results(self, sample_benchmark) -> None:
        """Test clearing verification results."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        # Store some results
        result1 = VerificationResult(
            question_id=q1_id,
            template_id="no_template",
            success=True,
            question_text="What is 2 + 2?",
            raw_llm_response="4",
            answering_model="test",
            parsing_model="test",
            execution_time=1.0,
            timestamp="2023-01-01T00:00:00",
        )

        result2 = VerificationResult(
            question_id=q2_id,
            template_id="no_template",
            success=True,
            question_text="What is the capital of France?",
            raw_llm_response="Paris",
            answering_model="test",
            parsing_model="test",
            execution_time=1.2,
            timestamp="2023-01-01T00:01:00",
        )

        benchmark.store_verification_results(
            {
                f"{q1_id}_test": result1,
                f"{q2_id}_test": result2,
            },
            "test_run",
        )

        # Clear specific question results
        cleared_count = benchmark.clear_verification_results(question_ids=[q1_id])
        assert cleared_count == 1

        # Verify only q2 results remain
        remaining_results = benchmark.get_verification_results()
        assert len(remaining_results) == 1
        assert f"{q2_id}_test" in remaining_results

    def test_export_verification_results(self, sample_benchmark) -> None:
        """Test exporting verification results."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        # Store some results
        result = VerificationResult(
            question_id=q1_id,
            template_id="no_template",
            success=True,
            question_text="What is 2 + 2?",
            raw_llm_response="4",
            answering_model="test",
            parsing_model="test",
            execution_time=1.0,
            timestamp="2023-01-01T00:00:00",
        )

        benchmark.store_verification_results({f"{q1_id}_test": result}, "test_run")

        # Test JSON export
        json_export = benchmark.export_verification_results(format="json")
        exported_data = json.loads(json_export)
        # Check that the result data is in the exported list
        assert any(item.get("question_id") == q1_id for item in exported_data)

        # Test CSV export
        csv_export = benchmark.export_verification_results(format="csv")
        assert "question_id" in csv_export  # Check for question_id column
        assert q1_id in csv_export  # Check that the question ID is in the CSV

    def test_verification_summary(self, sample_benchmark) -> None:
        """Test verification summary statistics."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        # Store mixed results
        success_result = VerificationResult(
            question_id=q1_id,
            template_id="no_template",
            success=True,
            question_text="What is 2 + 2?",
            raw_llm_response="4",
            answering_model="test1",
            parsing_model="test1",
            execution_time=1.0,
            timestamp="2023-01-01T00:00:00",
        )

        failure_result = VerificationResult(
            question_id=q2_id,
            template_id="no_template",
            success=False,
            error="Failed",
            question_text="What is the capital of France?",
            raw_llm_response="",
            answering_model="test2",
            parsing_model="test2",
            execution_time=0.5,
            timestamp="2023-01-01T00:01:00",
        )

        benchmark.store_verification_results(
            {
                f"{q1_id}_test": success_result,
                f"{q2_id}_test": failure_result,
            },
            "test_run",
        )

        # Get summary
        summary = benchmark.get_verification_summary("test_run")

        assert summary["total_results"] == 2
        assert summary["successful_count"] == 1
        assert summary["failed_count"] == 1
        assert summary["success_rate"] == 50.0
        assert summary["unique_questions"] == 2
        assert summary["average_execution_time"] == 0.75
        assert summary["model_combinations"] == 2

    def test_error_handling_in_verification(self, sample_benchmark, sample_config) -> None:
        """Test error handling during verification."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        # Test verification with non-existent question
        with pytest.raises(ValueError, match="Question not found"):
            benchmark.run_verification(
                config=sample_config,
                question_ids=["non_existent_id"],
            )

        # Add a question truly without any template
        q4_id = benchmark.add_question(
            question="Question without template",
            raw_answer="Some answer",
            answer_template=None,
            finished=True,
        )

        # Clear the default template that might have been set
        benchmark._questions_cache[q4_id]["answer_template"] = None

        # Test verification with question without template
        with pytest.raises(ValueError, match="has no answer template"):
            benchmark.run_verification(
                config=sample_config,
                question_ids=[q4_id],
            )

    @patch("karenina.benchmark.verification.orchestrator.run_question_verification")
    def test_verification_with_run_name_storage(self, mock_run_verification, sample_benchmark, sample_config) -> None:
        """Test that results can be stored when run_name is provided."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        mock_result = VerificationResult(
            question_id=q1_id,
            template_id="no_template",
            success=True,
            question_text="What is 2 + 2?",
            raw_llm_response="4",
            answering_model="test",
            parsing_model="test",
            execution_time=1.0,
            timestamp="2023-01-01T00:00:00",
            embedding_check_performed=False,
            regex_validations_performed=False,
            recursion_limit_reached=False,
        )

        expected_key = f"{q1_id}_test-answering_test-parsing"
        mock_run_verification.return_value = {expected_key: mock_result}

        # Run verification with run_name
        results = benchmark.run_verification(
            config=sample_config,
            question_ids=[q1_id],
            run_name="auto_store_test",
        )

        # Explicitly store the results
        benchmark.store_verification_results(results, "auto_store_test")

        # Results should be stored
        stored_results = benchmark.get_verification_results(run_name="auto_store_test")
        assert len(stored_results) == 1
        assert expected_key in stored_results

    def test_benchmark_persistence_with_verification_results(self, sample_benchmark) -> None:
        """Test that verification results do NOT persist when saving/loading benchmark (in-memory only)."""
        benchmark, (q1_id, q2_id, q3_id) = sample_benchmark

        # Store some verification results
        result = VerificationResult(
            question_id=q1_id,
            template_id="no_template",
            success=True,
            question_text="What is 2 + 2?",
            raw_llm_response="4",
            answering_model="test",
            parsing_model="test",
            execution_time=1.0,
            timestamp="2023-01-01T00:00:00",
            embedding_check_performed=False,
            regex_validations_performed=False,
            recursion_limit_reached=False,
        )

        result_key = f"{q1_id}_test"
        benchmark.store_verification_results({result_key: result}, "persistent_test")

        # Verify results are accessible before saving
        stored_results = benchmark.get_verification_results(run_name="persistent_test")
        assert len(stored_results) == 1
        assert result_key in stored_results

        # Save and reload benchmark
        with tempfile.NamedTemporaryFile(suffix=".jsonld", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            benchmark.save(tmp_path)
            reloaded_benchmark = Benchmark.load(tmp_path)

            # Verification results should NOT persist (in-memory only)
            stored_results = reloaded_benchmark.get_verification_results(run_name="persistent_test")
            assert len(stored_results) == 0  # Results don't persist across save/load

        finally:
            tmp_path.unlink(missing_ok=True)
