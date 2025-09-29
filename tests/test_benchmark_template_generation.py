"""Tests for benchmark template generation functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.benchmark import Benchmark


@pytest.fixture
def sample_benchmark():
    """Create a sample benchmark with test questions."""
    benchmark = Benchmark.create("Test Benchmark", "A test benchmark for template generation")

    # Add test questions
    benchmark.add_question(question="What is Python?", raw_answer="Python is a programming language", question_id="q1")
    benchmark.add_question(
        question="What is machine learning?",
        raw_answer="Machine learning is a subset of artificial intelligence",
        question_id="q2",
    )
    benchmark.add_question(
        question="What is a database?",
        raw_answer="A database is a collection of organized information",
        question_id="q3",
    )

    return benchmark


@pytest.fixture
def mock_llm_response():
    """Mock LLM response with code blocks."""
    return """Here's your answer template:

```python
from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    response: str = Field(description="The answer response")

    def verify(self) -> bool:
        return len(self.response) > 0
```

This template should work well for your question."""


@pytest.fixture
def mock_code_blocks():
    """Mock extracted code blocks."""
    return """from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    response: str = Field(description="The answer response")

    def verify(self) -> bool:
        return len(self.response) > 0"""


@pytest.fixture
def valid_template_code():
    """Valid template code for testing."""
    return """from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    existing_response: str = Field(description="Existing template response")

    def verify(self) -> bool:
        return len(self.existing_response) > 0"""


class TestSingleQuestionTemplateGeneration:
    """Test single question template generation."""

    @patch("karenina.benchmark.benchmark.generate_answer_template")
    @patch("karenina.benchmark.benchmark.extract_and_combine_codeblocks")
    def test_generate_template_for_question_success(
        self, mock_extract, mock_generate, sample_benchmark, mock_llm_response, mock_code_blocks
    ):
        """Test successful template generation for a single question."""
        # Setup mocks
        mock_generate.return_value = mock_llm_response
        mock_extract.return_value = mock_code_blocks

        # Generate template
        result = sample_benchmark.generate_template_for_question("q1")

        # Verify the result
        assert result["success"] is True
        assert result["template_code"] == mock_code_blocks
        assert result["error"] is None
        assert result["raw_response"] == mock_llm_response
        assert result["skipped"] is False

        # Verify the template was added to the benchmark
        assert sample_benchmark.has_template("q1")
        assert sample_benchmark.get_template("q1") == mock_code_blocks

        # Verify the generator was called with correct parameters
        mock_generate.assert_called_once_with(
            question="What is Python?",
            raw_answer="Python is a programming language",
            model="gpt-4.1-mini",
            model_provider="openai",
            temperature=0,
            custom_system_prompt=None,
            interface="langchain",
        )

    @patch("karenina.benchmark.benchmark.generate_answer_template")
    @patch("karenina.benchmark.benchmark.extract_and_combine_codeblocks")
    def test_generate_template_for_question_no_code_blocks(
        self, mock_extract, mock_generate, sample_benchmark, mock_llm_response
    ):
        """Test template generation when no code blocks are found."""
        # Setup mocks
        mock_generate.return_value = mock_llm_response
        mock_extract.return_value = ""  # No code blocks found

        # Generate template
        result = sample_benchmark.generate_template_for_question("q1")

        # Verify the result fails when no code blocks found
        assert result["success"] is False
        assert result["template_code"] == ""
        assert "No valid code blocks found" in result["error"]
        assert result["skipped"] is False

    def test_generate_template_for_question_nonexistent(self, sample_benchmark):
        """Test template generation for nonexistent question."""
        with pytest.raises(ValueError, match="Question not found: nonexistent"):
            sample_benchmark.generate_template_for_question("nonexistent")

    @patch("karenina.benchmark.benchmark.generate_answer_template")
    def test_generate_template_for_question_with_existing_template(
        self, mock_generate, sample_benchmark, valid_template_code
    ):
        """Test template generation when template already exists."""
        # Add a template to the question first
        sample_benchmark.add_answer_template("q1", valid_template_code)

        # Try to generate template without force_regenerate
        result = sample_benchmark.generate_template_for_question("q1")

        # Should return existing template and not call LLM
        assert result["success"] is True
        assert result["template_code"] == valid_template_code
        assert "already exists" in result["error"]
        assert result["skipped"] is True
        mock_generate.assert_not_called()

    @patch("karenina.benchmark.benchmark.generate_answer_template")
    @patch("karenina.benchmark.benchmark.extract_and_combine_codeblocks")
    def test_generate_template_for_question_force_regenerate(
        self, mock_extract, mock_generate, sample_benchmark, mock_llm_response, mock_code_blocks, valid_template_code
    ):
        """Test template generation with force_regenerate=True."""
        # Add a template to the question first
        sample_benchmark.add_answer_template("q1", valid_template_code)

        # Setup mocks
        mock_generate.return_value = mock_llm_response
        mock_extract.return_value = mock_code_blocks

        # Generate template with force_regenerate
        result = sample_benchmark.generate_template_for_question("q1", force_regenerate=True)

        # Should generate new template
        assert result["success"] is True
        assert result["template_code"] == mock_code_blocks
        assert result["skipped"] is False
        mock_generate.assert_called_once()

    @patch("karenina.benchmark.benchmark.generate_answer_template")
    def test_generate_template_for_question_llm_error(self, mock_generate, sample_benchmark):
        """Test template generation when LLM call fails."""
        # Setup mock to raise exception
        mock_generate.side_effect = Exception("LLM API error")

        # Generate template
        result = sample_benchmark.generate_template_for_question("q1")

        # Verify error handling
        assert result["success"] is False
        assert result["template_code"] == ""
        assert "LLM API error" in result["error"]
        assert result["skipped"] is False

    @patch("karenina.benchmark.benchmark.generate_answer_template")
    @patch("karenina.benchmark.benchmark.extract_and_combine_codeblocks")
    def test_generate_template_for_question_custom_parameters(
        self, mock_extract, mock_generate, sample_benchmark, mock_llm_response, mock_code_blocks
    ):
        """Test template generation with custom parameters."""
        # Setup mocks
        mock_generate.return_value = mock_llm_response
        mock_extract.return_value = mock_code_blocks

        # Generate template with custom parameters
        sample_benchmark.generate_template_for_question(
            question_id="q1",
            model="gpt-4.1-mini",
            model_provider="openai",
            temperature=0.7,
            custom_system_prompt="Custom prompt",
            interface="openrouter",
        )

        # Verify custom parameters were passed
        mock_generate.assert_called_once_with(
            question="What is Python?",
            raw_answer="Python is a programming language",
            model="gpt-4.1-mini",
            model_provider="openai",
            temperature=0.7,
            custom_system_prompt="Custom prompt",
            interface="openrouter",
        )


class TestBatchTemplateGeneration:
    """Test batch template generation."""

    @patch("karenina.benchmark.benchmark.generate_answer_template")
    @patch("karenina.benchmark.benchmark.extract_and_combine_codeblocks")
    def test_generate_templates_success(
        self, mock_extract, mock_generate, sample_benchmark, mock_llm_response, mock_code_blocks
    ):
        """Test successful batch template generation."""
        # Setup mocks
        mock_generate.return_value = mock_llm_response
        mock_extract.return_value = mock_code_blocks

        # Generate templates for multiple questions
        results = sample_benchmark.generate_templates(["q1", "q2"])

        # Verify results
        assert len(results) == 2
        assert results["q1"]["success"] is True
        assert results["q2"]["success"] is True

        # Verify templates were added
        assert sample_benchmark.has_template("q1")
        assert sample_benchmark.has_template("q2")

        # Verify generator was called twice
        assert mock_generate.call_count == 2

    def test_generate_templates_invalid_question_ids(self, sample_benchmark):
        """Test batch generation with invalid question IDs."""
        with pytest.raises(ValueError, match="Questions not found: \\['nonexistent'\\]"):
            sample_benchmark.generate_templates(["q1", "nonexistent", "q2"])

    @patch("karenina.benchmark.benchmark.generate_answer_template")
    @patch("karenina.benchmark.benchmark.extract_and_combine_codeblocks")
    def test_generate_templates_with_progress_callback(
        self, mock_extract, mock_generate, sample_benchmark, mock_llm_response, mock_code_blocks
    ):
        """Test batch generation with progress callback."""
        # Setup mocks
        mock_generate.return_value = mock_llm_response
        mock_extract.return_value = mock_code_blocks

        # Create mock progress callback
        progress_callback = MagicMock()

        # Generate templates with progress callback
        sample_benchmark.generate_templates(["q1", "q2"], progress_callback=progress_callback)

        # Verify progress callback was called
        assert progress_callback.call_count >= 3  # At least once per question + final

        # Check the final call
        final_call = progress_callback.call_args_list[-1]
        assert final_call[0][0] == 100.0  # 100% progress
        assert "completed" in final_call[0][1].lower()

    @patch("karenina.benchmark.benchmark.generate_answer_template")
    def test_generate_templates_mixed_results(self, mock_generate, sample_benchmark, mock_code_blocks):
        """Test batch generation with mixed success/failure results."""

        # Setup mock to fail on second question
        def mock_generate_side_effect(*args, **kwargs):
            if "machine learning" in kwargs.get("question", ""):
                raise Exception("API error")
            return "mock template response"

        mock_generate.side_effect = mock_generate_side_effect

        with patch("karenina.benchmark.benchmark.extract_and_combine_codeblocks") as mock_extract:
            mock_extract.return_value = mock_code_blocks

            # Generate templates
            results = sample_benchmark.generate_templates(["q1", "q2"])

            # Verify mixed results
            assert results["q1"]["success"] is True
            assert results["q2"]["success"] is False
            assert "API error" in results["q2"]["error"]


class TestGenerateAllTemplates:
    """Test generate all templates functionality."""

    @patch("karenina.benchmark.benchmark.generate_answer_template")
    @patch("karenina.benchmark.benchmark.extract_and_combine_codeblocks")
    def test_generate_all_templates_only_missing(
        self, mock_extract, mock_generate, sample_benchmark, mock_llm_response, mock_code_blocks, valid_template_code
    ):
        """Test generating templates only for questions without templates."""
        # Setup mocks
        mock_generate.return_value = mock_llm_response
        mock_extract.return_value = mock_code_blocks

        # Add template to first question
        sample_benchmark.add_answer_template("q1", valid_template_code)

        # Generate all templates (only missing)
        results = sample_benchmark.generate_all_templates(only_missing=True)

        # Should only generate for q2 and q3 (q1 already has template)
        assert len(results) == 2
        assert "q1" not in results
        assert "q2" in results
        assert "q3" in results

    @patch("karenina.benchmark.benchmark.generate_answer_template")
    @patch("karenina.benchmark.benchmark.extract_and_combine_codeblocks")
    def test_generate_all_templates_force_regenerate(
        self, mock_extract, mock_generate, sample_benchmark, mock_llm_response, mock_code_blocks, valid_template_code
    ):
        """Test generating templates for all questions with force_regenerate."""
        # Setup mocks
        mock_generate.return_value = mock_llm_response
        mock_extract.return_value = mock_code_blocks

        # Add template to first question
        sample_benchmark.add_answer_template("q1", valid_template_code)

        # Generate all templates with force_regenerate
        results = sample_benchmark.generate_all_templates(force_regenerate=True, only_missing=False)

        # Should generate for all questions
        assert len(results) == 3
        assert all(qid in results for qid in ["q1", "q2", "q3"])

    def test_generate_all_templates_empty_benchmark(self):
        """Test generating templates on empty benchmark."""
        benchmark = Benchmark.create("Empty Benchmark")

        results = benchmark.generate_all_templates()

        assert len(results) == 0


class TestTemplateExportImport:
    """Test template export and import functionality."""

    def test_export_generated_templates(self, sample_benchmark, valid_template_code):
        """Test exporting templates to JSON file."""
        # Add templates to some questions
        sample_benchmark.add_answer_template("q1", valid_template_code)
        sample_benchmark.add_answer_template("q2", valid_template_code)

        # Export templates
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            sample_benchmark.export_generated_templates(tmp_path)

            # Verify file was created and contains expected data
            assert tmp_path.exists()

            with tmp_path.open() as f:
                exported_data = json.load(f)

            assert len(exported_data) == 2
            assert exported_data["q1"] == valid_template_code
            assert exported_data["q2"] == valid_template_code
            assert "q3" not in exported_data  # q3 has no template

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_export_generated_templates_no_templates(self, sample_benchmark):
        """Test exporting when no templates exist."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            sample_benchmark.export_generated_templates(tmp_path)

            with tmp_path.open() as f:
                exported_data = json.load(f)

            assert len(exported_data) == 0

        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @patch("karenina.benchmark.benchmark.load_answer_templates_from_json")
    def test_import_generated_templates_success(self, mock_load, sample_benchmark, valid_template_code):
        """Test successful template import."""
        # Setup mock return value
        mock_templates = {}
        mock_code_blocks = {"q1": valid_template_code, "q2": valid_template_code}
        mock_load.return_value = (mock_templates, mock_code_blocks)

        # Import templates
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
            tmp_path = Path(tmp_file.name)
            results = sample_benchmark.import_generated_templates(tmp_path)

        # Verify results
        assert results["q1"] is True
        assert results["q2"] is True

        # Verify templates were added
        assert sample_benchmark.has_template("q1")
        assert sample_benchmark.has_template("q2")

    @patch("karenina.benchmark.benchmark.load_answer_templates_from_json")
    def test_import_generated_templates_with_existing(self, mock_load, sample_benchmark, valid_template_code):
        """Test import when some templates already exist."""
        # Add existing template
        sample_benchmark.add_answer_template("q1", valid_template_code)

        # Setup mock return value
        mock_templates = {}
        # Use different template code for import
        import_template = """from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    imported_field: str = Field(description="Imported template field")

    def verify(self) -> bool:
        return len(self.imported_field) > 0"""

        mock_code_blocks = {"q1": import_template, "q2": import_template}
        mock_load.return_value = (mock_templates, mock_code_blocks)

        # Import templates without force_overwrite
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
            tmp_path = Path(tmp_file.name)
            results = sample_benchmark.import_generated_templates(tmp_path)

        # q1 should be skipped, q2 should be imported
        assert results["q1"] is False  # Skipped due to existing template
        assert results["q2"] is True  # Successfully imported

        # Verify original template wasn't overwritten
        assert sample_benchmark.get_template("q1") == valid_template_code

    @patch("karenina.benchmark.benchmark.load_answer_templates_from_json")
    def test_import_generated_templates_force_overwrite(self, mock_load, sample_benchmark, valid_template_code):
        """Test import with force_overwrite=True."""
        # Add existing template
        sample_benchmark.add_answer_template("q1", valid_template_code)

        # Setup mock return value
        import_template = """from karenina.schemas.answer_class import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    overwrite_field: str = Field(description="Overwritten template field")

    def verify(self) -> bool:
        return len(self.overwrite_field) > 0"""

        mock_templates = {}
        mock_code_blocks = {"q1": import_template}
        mock_load.return_value = (mock_templates, mock_code_blocks)

        # Import templates with force_overwrite
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
            tmp_path = Path(tmp_file.name)
            results = sample_benchmark.import_generated_templates(tmp_path, force_overwrite=True)

        # Should overwrite existing template
        assert results["q1"] is True
        assert sample_benchmark.get_template("q1") == import_template

    @patch("karenina.benchmark.benchmark.load_answer_templates_from_json")
    def test_import_generated_templates_nonexistent_questions(self, mock_load, sample_benchmark, valid_template_code):
        """Test import with templates for non-existent questions."""
        # Setup mock return value with non-existent question
        mock_templates = {}
        mock_code_blocks = {"q1": valid_template_code, "nonexistent": valid_template_code}
        mock_load.return_value = (mock_templates, mock_code_blocks)

        # Import templates
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
            tmp_path = Path(tmp_file.name)
            results = sample_benchmark.import_generated_templates(tmp_path)

        # Existing question should succeed, nonexistent should fail
        assert results["q1"] is True
        assert results["nonexistent"] is False

    @patch("karenina.benchmark.benchmark.load_answer_templates_from_json")
    def test_import_generated_templates_load_error(self, mock_load, sample_benchmark):
        """Test import when loading fails."""
        # Setup mock to return invalid format
        mock_load.return_value = {}  # Not a tuple

        # Import should raise ValueError
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
            tmp_path = Path(tmp_file.name)
            with pytest.raises(ValueError, match="Unable to load code blocks"):
                sample_benchmark.import_generated_templates(tmp_path)


class TestTemplateGenerationIntegration:
    """Integration tests for template generation functionality."""

    def test_full_workflow_generate_export_import(self, sample_benchmark, mock_code_blocks):
        """Test complete workflow: generate -> export -> import to new benchmark."""
        # Step 1: Generate templates (mock the LLM calls)
        with (
            patch("karenina.benchmark.benchmark.generate_answer_template") as mock_generate,
            patch("karenina.benchmark.benchmark.extract_and_combine_codeblocks") as mock_extract,
        ):
            mock_generate.return_value = "mock llm response"
            mock_extract.return_value = mock_code_blocks

            # Generate templates for all questions
            results = sample_benchmark.generate_all_templates()

            # Verify generation worked
            assert len(results) == 3
            assert all(sample_benchmark.has_template(qid) for qid in ["q1", "q2", "q3"])

        # Step 2: Export templates
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_file:
            export_path = Path(tmp_file.name)

        try:
            sample_benchmark.export_generated_templates(export_path)

            # Step 3: Create new benchmark and import
            new_benchmark = Benchmark.create("Import Test Benchmark")

            # Add same questions to new benchmark
            new_benchmark.add_question(
                question="What is Python?", raw_answer="Python is a programming language", question_id="q1"
            )
            new_benchmark.add_question(
                question="What is machine learning?",
                raw_answer="Machine learning is a subset of artificial intelligence",
                question_id="q2",
            )

            # Import templates
            with patch("karenina.benchmark.benchmark.load_answer_templates_from_json") as mock_load:
                # Simulate the loaded data
                import_code_blocks = {
                    "q1": mock_code_blocks,
                    "q2": mock_code_blocks,
                    "q3": mock_code_blocks,  # This should be skipped (question doesn't exist)
                }
                mock_load.return_value = ({}, import_code_blocks)

                import_results = new_benchmark.import_generated_templates(export_path)

                # Verify import results
                assert import_results["q1"] is True
                assert import_results["q2"] is True
                assert import_results["q3"] is False  # Question doesn't exist

                # Verify templates were imported
                assert new_benchmark.has_template("q1")
                assert new_benchmark.has_template("q2")

        finally:
            if export_path.exists():
                export_path.unlink()
