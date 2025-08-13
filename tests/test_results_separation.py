"""Tests for verification result separation from checkpoint storage."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from karenina.benchmark import Benchmark
from karenina.benchmark.models import VerificationResult


class TestResultsSeparation:
    """Test that verification results are kept separate from checkpoint storage."""

    @pytest.fixture
    def sample_benchmark(self):
        """Create a sample benchmark with questions."""
        benchmark = Benchmark(
            name="Test Benchmark", description="Test benchmark for results separation", version="1.0.0"
        )

        # Add some test questions
        benchmark.add_question(
            question="What is 2+2?",
            raw_answer="Four",
            answer_template="class Answer(BaseModel): result: int",
            finished=True,
        )
        benchmark.add_question(
            question="What is the capital of France?",
            raw_answer="Paris",
            answer_template="class Answer(BaseModel): city: str",
            finished=True,
        )

        return benchmark

    @pytest.fixture
    def sample_verification_results(self):
        """Create sample verification results."""
        return {
            "q1_result1": VerificationResult(
                question_id="test_q1",
                success=True,
                question_text="What is 2+2?",
                raw_llm_response="4",
                answering_model="gpt-4",
                parsing_model="gpt-4",
                execution_time=1.5,
                timestamp=datetime.now().isoformat(),
                run_name="test_run",
            ),
            "q2_result1": VerificationResult(
                question_id="test_q2",
                success=True,
                question_text="What is the capital of France?",
                raw_llm_response="Paris",
                answering_model="gpt-4",
                parsing_model="gpt-4",
                execution_time=2.0,
                timestamp=datetime.now().isoformat(),
                run_name="test_run",
            ),
        }

    def test_verification_results_not_saved_to_checkpoint(self, sample_benchmark, sample_verification_results):
        """Test that verification results are not included when saving checkpoint."""
        # Store some verification results
        sample_benchmark.store_verification_results(sample_verification_results, "test_run")

        # Verify results are accessible
        retrieved_results = sample_benchmark.get_verification_results(run_name="test_run")
        assert len(retrieved_results) == 2

        # Save benchmark to file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonld", delete=False) as f:
            checkpoint_path = Path(f.name)

        try:
            sample_benchmark.save(checkpoint_path)

            # Load the saved file and verify no verification results are included
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            # Check that no verification_results_ properties exist
            if "additionalProperty" in checkpoint_data:
                for prop in checkpoint_data["additionalProperty"]:
                    if isinstance(prop, dict) and "name" in prop:
                        assert not prop["name"].startswith("verification_results_"), (
                            f"Found verification result property in checkpoint: {prop['name']}"
                        )

        finally:
            # Clean up
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    def test_results_persist_in_memory_after_save(self, sample_benchmark, sample_verification_results):
        """Test that results remain accessible in memory even after saving checkpoint."""
        # Store verification results
        sample_benchmark.store_verification_results(sample_verification_results, "test_run")

        # Save benchmark
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonld", delete=False) as f:
            checkpoint_path = Path(f.name)

        try:
            sample_benchmark.save(checkpoint_path)

            # Results should still be accessible in memory
            retrieved_results = sample_benchmark.get_verification_results(run_name="test_run")
            assert len(retrieved_results) == 2

            # Verify result details
            result = list(retrieved_results.values())[0]
            assert result.success is True
            assert result.answering_model == "gpt-4"

        finally:
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    def test_load_checkpoint_without_results(self, sample_benchmark):
        """Test that loading a checkpoint works normally when no results were saved."""
        # Save benchmark without any results
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonld", delete=False) as f:
            checkpoint_path = Path(f.name)

        try:
            sample_benchmark.save(checkpoint_path)

            # Load benchmark from file
            loaded_benchmark = Benchmark.load(checkpoint_path)

            # Verify basic properties
            assert loaded_benchmark.name == "Test Benchmark"
            assert loaded_benchmark.question_count == 2

            # Verify no results are present
            assert len(loaded_benchmark.get_verification_results()) == 0
            assert len(loaded_benchmark.get_all_run_names()) == 0

        finally:
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    def test_export_results_to_file_json(self, sample_benchmark, sample_verification_results):
        """Test exporting results to a JSON file."""
        # Store verification results
        sample_benchmark.store_verification_results(sample_verification_results, "test_run")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            results_path = Path(f.name)

        try:
            # Export results to file
            sample_benchmark.export_verification_results_to_file(results_path, run_name="test_run")

            # Verify file was created and contains correct data
            assert results_path.exists()

            with open(results_path) as f:
                exported_data = json.load(f)

            # New format should be an array with row_index
            assert isinstance(exported_data, list)
            assert len(exported_data) == 2

            # Verify result structure
            result = exported_data[0]  # First item in array
            assert "row_index" in result
            assert result["row_index"] == 1
            assert result["success"] is True
            assert result["question_text"] == "What is 2+2?"
            assert result["answering_model"] == "gpt-4"

        finally:
            if results_path.exists():
                results_path.unlink()

    def test_export_results_to_file_csv(self, sample_benchmark, sample_verification_results):
        """Test exporting results to a CSV file."""
        # Store verification results
        sample_benchmark.store_verification_results(sample_verification_results, "test_run")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            results_path = Path(f.name)

        try:
            # Export results to file
            sample_benchmark.export_verification_results_to_file(results_path, run_name="test_run")

            # Verify file was created
            assert results_path.exists()

            # Read and verify CSV content
            with open(results_path) as f:
                csv_content = f.read()

            # Should contain frontend-format headers and data
            assert "row_index" in csv_content
            assert "success" in csv_content
            assert "question_text" in csv_content
            assert "rubric_summary" in csv_content
            # Row data should start with index 1, 2, etc.
            assert "1," in csv_content or "1\t" in csv_content
            assert "2," in csv_content or "2\t" in csv_content

        finally:
            if results_path.exists():
                results_path.unlink()

    def test_load_results_from_file_json(self, sample_benchmark, sample_verification_results):
        """Test loading results from a JSON file."""
        # First export results to a file
        sample_benchmark.store_verification_results(sample_verification_results, "original_run")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            results_path = Path(f.name)

        try:
            sample_benchmark.export_verification_results_to_file(results_path, run_name="original_run")

            # Clear in-memory results
            sample_benchmark.clear_verification_results()
            assert len(sample_benchmark.get_verification_results()) == 0

            # Load results from file
            loaded_results = sample_benchmark.load_verification_results_from_file(results_path, "loaded_run")

            # Verify loaded results
            assert len(loaded_results) == 2

            # Keys are now generated from question_id and row_index
            result_keys = list(loaded_results.keys())
            assert len(result_keys) == 2

            # Verify they're now accessible through normal methods
            retrieved_results = sample_benchmark.get_verification_results(run_name="loaded_run")
            assert len(retrieved_results) == 2

            # Verify result details - get first result
            first_result = list(loaded_results.values())[0]
            assert first_result.success is True
            assert first_result.question_text in ["What is 2+2?", "What is the capital of France?"]

        finally:
            if results_path.exists():
                results_path.unlink()

    def test_separate_export_import_workflow(self, sample_benchmark, sample_verification_results):
        """Test the complete workflow of separating results from checkpoints."""
        # 1. Store results and save checkpoint
        sample_benchmark.store_verification_results(sample_verification_results, "test_run")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonld", delete=False) as f:
            checkpoint_path = Path(f.name)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            results_path = Path(f.name)

        try:
            # Save checkpoint and export results separately
            sample_benchmark.save(checkpoint_path)
            sample_benchmark.export_verification_results_to_file(results_path, run_name="test_run")

            # 2. Load checkpoint in new benchmark instance
            new_benchmark = Benchmark.load(checkpoint_path)

            # Verify checkpoint data is intact but no results
            assert new_benchmark.name == "Test Benchmark"
            assert new_benchmark.question_count == 2
            assert len(new_benchmark.get_verification_results()) == 0

            # 3. Load results separately
            loaded_results = new_benchmark.load_verification_results_from_file(results_path, "imported_run")

            # 4. Verify results are now available
            assert len(loaded_results) == 2
            retrieved_results = new_benchmark.get_verification_results(run_name="imported_run")
            assert len(retrieved_results) == 2

            # Verify result content matches original
            result = list(retrieved_results.values())[0]
            original_result = list(sample_verification_results.values())[0]
            assert result.question_text == original_result.question_text
            assert result.success == original_result.success

        finally:
            for path in [checkpoint_path, results_path]:
                if path.exists():
                    path.unlink()
