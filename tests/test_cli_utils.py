"""
Unit tests for CLI utility functions.

Tests parse_question_indices, validate_output_path, filter functions, etc.
"""

from pathlib import Path

import pytest

from karenina.cli.utils import (
    create_export_job,
    filter_templates_by_ids,
    filter_templates_by_indices,
    get_preset_path,
    list_presets,
    parse_question_indices,
    validate_output_path,
)
from karenina.schemas import FinishedTemplate, ModelConfig, VerificationConfig, VerificationResult
from karenina.schemas.workflow.verification import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


class TestParseQuestionIndices:
    """Test parse_question_indices function."""

    def test_single_index(self):
        """Test parsing a single index."""
        result = parse_question_indices("5", 10)
        assert result == [5]

    def test_multiple_indices(self):
        """Test parsing multiple comma-separated indices."""
        result = parse_question_indices("0,2,5", 10)
        assert result == [0, 2, 5]

    def test_range(self):
        """Test parsing a range."""
        result = parse_question_indices("2-5", 10)
        assert result == [2, 3, 4, 5]

    def test_mixed(self):
        """Test parsing mixed indices and ranges."""
        result = parse_question_indices("0,2-4,7", 10)
        assert result == [0, 2, 3, 4, 7]

    def test_duplicates_removed(self):
        """Test that duplicate indices are removed."""
        result = parse_question_indices("0,0,1,1", 10)
        assert result == [0, 1]

    def test_sorted_output(self):
        """Test that output is sorted."""
        result = parse_question_indices("5,1,3", 10)
        assert result == [1, 3, 5]

    def test_whitespace_handling(self):
        """Test handling of whitespace."""
        result = parse_question_indices(" 0 , 2 - 4 , 7 ", 10)
        assert result == [0, 2, 3, 4, 7]

    def test_out_of_range_single(self):
        """Test error on out-of-range single index."""
        with pytest.raises(ValueError, match="Index out of range"):
            parse_question_indices("10", 10)

    def test_out_of_range_in_range(self):
        """Test error on out-of-range in range."""
        with pytest.raises(ValueError, match="Index out of range"):
            parse_question_indices("8-12", 10)

    def test_negative_index(self):
        """Test error on negative index."""
        with pytest.raises(ValueError, match="Negative|Invalid"):
            parse_question_indices("-1", 10)

    def test_negative_range(self):
        """Test error on negative range."""
        with pytest.raises(ValueError, match="Negative|Invalid"):
            parse_question_indices("-5--1", 10)

    def test_invalid_range_reversed(self):
        """Test error on reversed range."""
        with pytest.raises(ValueError, match="Invalid range"):
            parse_question_indices("5-2", 10)

    def test_invalid_format(self):
        """Test error on invalid format."""
        with pytest.raises(ValueError, match="Invalid"):
            parse_question_indices("abc", 10)

    def test_empty_parts_ignored(self):
        """Test that empty parts (e.g., trailing comma) are ignored."""
        result = parse_question_indices("0,1,", 10)
        assert result == [0, 1]


class TestValidateOutputPath:
    """Test validate_output_path function."""

    def test_json_extension(self, tmp_path):
        """Test JSON extension validation."""
        output_path = tmp_path / "results.json"
        result = validate_output_path(output_path)
        assert result == "json"

    def test_csv_extension(self, tmp_path):
        """Test CSV extension validation."""
        output_path = tmp_path / "results.csv"
        result = validate_output_path(output_path)
        assert result == "csv"

    def test_uppercase_extension(self, tmp_path):
        """Test uppercase extension is handled."""
        output_path = tmp_path / "results.JSON"
        result = validate_output_path(output_path)
        assert result == "json"

    def test_invalid_extension(self, tmp_path):
        """Test error on invalid extension."""
        output_path = tmp_path / "results.txt"
        with pytest.raises(ValueError, match="Invalid output format"):
            validate_output_path(output_path)

    def test_missing_parent_directory(self):
        """Test error on missing parent directory."""
        output_path = Path("/nonexistent/directory/results.json")
        with pytest.raises(ValueError, match="Parent directory does not exist"):
            validate_output_path(output_path)


class TestFilterTemplates:
    """Test template filtering functions."""

    @pytest.fixture
    def mock_templates(self):
        """Create mock FinishedTemplate objects."""
        from datetime import datetime

        templates = []
        for i in range(5):
            template = FinishedTemplate(
                question_id=f"q{i}",
                question_text=f"Question {i}",
                question_preview=f"Question {i}",
                expected_answer="test",
                test_code="assert True",
                template_code="def test(): pass",
                template_id=f"template-{i}",
                last_modified=datetime.now().isoformat(),
            )
            templates.append(template)
        return templates

    def test_filter_by_indices(self, mock_templates):
        """Test filtering templates by indices."""
        result = filter_templates_by_indices(mock_templates, [0, 2, 4])
        assert len(result) == 3
        assert result[0].question_id == "q0"
        assert result[1].question_id == "q2"
        assert result[2].question_id == "q4"

    def test_filter_by_indices_empty(self, mock_templates):
        """Test filtering with empty indices list."""
        result = filter_templates_by_indices(mock_templates, [])
        assert len(result) == 0

    def test_filter_by_ids(self, mock_templates):
        """Test filtering templates by question IDs."""
        result = filter_templates_by_ids(mock_templates, ["q1", "q3"])
        assert len(result) == 2
        assert result[0].question_id == "q1"
        assert result[1].question_id == "q3"

    def test_filter_by_ids_no_match(self, mock_templates):
        """Test filtering with non-matching IDs."""
        result = filter_templates_by_ids(mock_templates, ["nonexistent"])
        assert len(result) == 0


class TestPresetUtilities:
    """Test preset-related utility functions."""

    @pytest.fixture
    def temp_presets_dir(self, tmp_path):
        """Create temporary presets directory with test presets."""
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()

        # Create test preset files with proper wrapper format
        preset1 = {
            "id": "preset-1",
            "name": "test-preset-1",
            "description": "Test preset 1",
            "config": {
                "answering_models": [
                    {
                        "id": "answering-1",
                        "model_name": "gpt-4.1-mini",
                        "model_provider": "openai",
                        "interface": "langchain",
                        "temperature": 0.1,
                    }
                ],
                "parsing_models": [
                    {
                        "id": "parsing-1",
                        "model_name": "gpt-4.1-mini",
                        "model_provider": "openai",
                        "interface": "langchain",
                        "temperature": 0.1,
                    }
                ],
                "replicate_count": 1,
            },
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

        preset2 = {
            "id": "preset-2",
            "name": "test-preset-2",
            "description": "Test preset 2",
            "config": {
                "answering_models": [
                    {
                        "id": "answering-1",
                        "model_name": "gpt-4o",
                        "model_provider": "openai",
                        "interface": "langchain",
                        "temperature": 0.1,
                    }
                ],
                "parsing_models": [
                    {
                        "id": "parsing-1",
                        "model_name": "gpt-4o",
                        "model_provider": "openai",
                        "interface": "langchain",
                        "temperature": 0.1,
                    }
                ],
                "replicate_count": 2,
            },
            "created_at": "2024-01-02T00:00:00",
            "updated_at": "2024-01-02T00:00:00",
        }

        import json

        (presets_dir / "test-preset-1.json").write_text(json.dumps(preset1, indent=2))
        (presets_dir / "test-preset-2.json").write_text(json.dumps(preset2, indent=2))

        # Create an invalid file to test error handling
        (presets_dir / "invalid.json").write_text("not valid json")

        return presets_dir

    def test_list_presets(self, temp_presets_dir):
        """Test listing presets."""
        presets = list_presets(temp_presets_dir)

        # Should have 2 valid presets (invalid.json is skipped silently)
        assert len(presets) >= 2
        preset_names = [p["name"] for p in presets]
        assert "test-preset-1" in preset_names
        assert "test-preset-2" in preset_names
        assert "filepath" in presets[0]
        assert "modified" in presets[0]

    def test_list_presets_sorted(self, temp_presets_dir):
        """Test that presets are sorted by name."""
        presets = list_presets(temp_presets_dir)
        names = [p["name"] for p in presets]
        assert names == sorted(names)

    def test_list_presets_empty_directory(self, tmp_path):
        """Test listing from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        presets = list_presets(empty_dir)
        assert presets == []

    def test_list_presets_nonexistent_directory(self, tmp_path):
        """Test listing from nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"
        presets = list_presets(nonexistent)
        assert presets == []

    def test_get_preset_path_by_name(self, temp_presets_dir):
        """Test resolving preset by name."""
        path = get_preset_path("test-preset-1", temp_presets_dir)
        assert path.exists()
        assert path.stem == "test-preset-1"

    def test_get_preset_path_by_name_with_extension(self, temp_presets_dir):
        """Test resolving preset by name with .json extension."""
        path = get_preset_path("test-preset-1.json", temp_presets_dir)
        assert path.exists()
        assert path.stem == "test-preset-1"

    def test_get_preset_path_by_absolute_path(self, temp_presets_dir):
        """Test resolving preset by absolute path."""
        absolute_path = temp_presets_dir / "test-preset-1.json"
        path = get_preset_path(str(absolute_path), temp_presets_dir)
        assert path == absolute_path.resolve()

    def test_get_preset_path_not_found(self, temp_presets_dir):
        """Test error when preset not found."""
        with pytest.raises(FileNotFoundError, match="Preset 'nonexistent' not found"):
            get_preset_path("nonexistent", temp_presets_dir)

    def test_get_preset_path_directory_not_found(self, tmp_path):
        """Test error when presets directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="Presets directory not found"):
            get_preset_path("test", nonexistent)


class TestCreateExportJob:
    """Test create_export_job function."""

    @pytest.fixture
    def mock_config(self):
        """Create mock VerificationConfig."""
        return VerificationConfig(
            answering_models=[
                ModelConfig(
                    id="answering-1",
                    model_name="gpt-4.1-mini",
                    model_provider="openai",
                    interface="langchain",
                    temperature=0.1,
                )
            ],
            parsing_models=[
                ModelConfig(
                    id="parsing-1",
                    model_name="gpt-4.1-mini",
                    model_provider="openai",
                    interface="langchain",
                    temperature=0.1,
                )
            ],
            replicate_count=1,
        )

    @pytest.fixture
    def mock_results(self):
        """Create mock verification results."""
        from datetime import datetime

        return {
            "result-1": VerificationResult(
                metadata=VerificationResultMetadata(
                    verification_id="result-1",
                    question_id="q1",
                    template_id="t1",
                    completed_without_errors=True,
                    question_text="Question 1",
                    answering_model="openai/gpt-4.1-mini",
                    parsing_model="openai/gpt-4.1-mini",
                    answering_model_id="answering-1",
                    parsing_model_id="parsing-1",
                    replicate_number=0,
                    execution_time=1.0,
                    timestamp=datetime.now().isoformat(),
                ),
                template=VerificationResultTemplate(
                    raw_llm_response="response 1",
                    template_verification_performed=True,
                    verify_result=True,
                ),
                rubric=VerificationResultRubric(
                    rubric_evaluation_performed=False,
                ),
            ),
            "result-2": VerificationResult(
                metadata=VerificationResultMetadata(
                    verification_id="result-2",
                    question_id="q2",
                    template_id="t2",
                    completed_without_errors=True,
                    question_text="Question 2",
                    answering_model="openai/gpt-4.1-mini",
                    parsing_model="openai/gpt-4.1-mini",
                    answering_model_id="answering-1",
                    parsing_model_id="parsing-1",
                    replicate_number=0,
                    execution_time=1.0,
                    timestamp=datetime.now().isoformat(),
                ),
                template=VerificationResultTemplate(
                    raw_llm_response="response 2",
                    template_verification_performed=True,
                    verify_result=False,
                ),
                rubric=VerificationResultRubric(
                    rubric_evaluation_performed=False,
                ),
            ),
            "result-3": VerificationResult(
                metadata=VerificationResultMetadata(
                    verification_id="result-3",
                    question_id="q3",
                    template_id="t3",
                    completed_without_errors=False,
                    question_text="Question 3",
                    answering_model="openai/gpt-4.1-mini",
                    parsing_model="openai/gpt-4.1-mini",
                    answering_model_id="answering-1",
                    parsing_model_id="parsing-1",
                    replicate_number=0,
                    execution_time=1.0,
                    timestamp=datetime.now().isoformat(),
                ),
                template=VerificationResultTemplate(
                    raw_llm_response="response 3",
                    template_verification_performed=True,
                    verify_result=False,
                ),
                rubric=VerificationResultRubric(
                    rubric_evaluation_performed=False,
                ),
            ),
        }

    def test_create_export_job_basic(self, mock_results, mock_config):
        """Test creating export job."""
        start_time = 1000.0
        end_time = 1100.0

        job = create_export_job(
            results=mock_results,
            config=mock_config,
            run_name="test-run",
            start_time=start_time,
            end_time=end_time,
        )

        assert job.run_name == "test-run"
        assert job.total_questions == 3
        assert job.successful_count == 2  # 2 completed without errors
        assert job.failed_count == 1
        assert job.start_time == start_time
        assert job.end_time == end_time
        assert job.status == "completed"
        assert job.config == mock_config

    def test_create_export_job_default_run_name(self, mock_results, mock_config):
        """Test creating export job with empty run_name falls back to cli-verification."""
        job = create_export_job(
            results=mock_results,
            config=mock_config,
            run_name="",
            start_time=1000.0,
            end_time=1100.0,
        )

        assert job.run_name == "cli-verification"
