"""Unit tests for CLI utility functions.

Tests cover:
- _get_presets_directory() - presets directory resolution
- list_presets() - listing available presets
- get_preset_path() - preset name/path resolution
- parse_question_indices() - parsing question index strings
- validate_output_path() - output path validation
- filter_templates_by_indices() - filtering templates by index
- filter_templates_by_ids() - filtering templates by ID
- create_export_job() - creating VerificationJob for export
- get_traces_path() - trace file path resolution
- load_manual_traces_from_file() - loading manual traces

All tests use temp directories and avoid external dependencies.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from karenina.cli.utils import (
    _get_presets_directory,
    create_export_job,
    filter_templates_by_ids,
    filter_templates_by_indices,
    get_preset_path,
    get_traces_path,
    list_presets,
    load_manual_traces_from_file,
    parse_question_indices,
    validate_output_path,
)
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.workflow.verification.api_models import FinishedTemplate
from karenina.schemas.workflow.verification.config import VerificationConfig
from karenina.schemas.workflow.verification.result import VerificationResult
from karenina.schemas.workflow.verification.result_components import VerificationResultMetadata
from karenina.schemas.workflow.verification_result_set import VerificationResultSet


# Helper function to create minimal FinishedTemplate
def _make_template(question_id: str) -> FinishedTemplate:
    return FinishedTemplate(
        question_id=question_id,
        question_text=f"Question {question_id}",
        question_preview=f"Q{question_id}",
        template_code=f"# Template for {question_id}",
        last_modified="2025-01-11T00:00:00",
    )


# Helper function to create minimal VerificationResult
def _make_result(question_id: str, completed: bool = True) -> VerificationResult:
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="test-template",
            completed_without_errors=completed,
            question_text=f"Question {question_id}",
            answering=ModelIdentity(interface="langchain", model_name="gpt-4"),
            parsing=ModelIdentity(interface="langchain", model_name="gpt-4"),
            execution_time=1.0,
            timestamp="2025-01-11T00:00:00",
            result_id="1234567890123456",
        )
    )


# =============================================================================
# _get_presets_directory Tests
# =============================================================================


@pytest.mark.unit
def test_get_presets_directory_default() -> None:
    """Test default presets directory."""
    with patch.dict("os.environ", {}, clear=True):
        result = _get_presets_directory()
    assert result == Path("presets")


@pytest.mark.unit
def test_get_presets_directory_explicit() -> None:
    """Test explicit presets directory."""
    custom = Path("/custom/presets")
    result = _get_presets_directory(custom)
    assert result == custom


@pytest.mark.unit
def test_get_presets_directory_from_env() -> None:
    """Test presets directory from environment variable."""
    with patch.dict("os.environ", {"KARENINA_PRESETS_DIR": "/env/presets"}):
        result = _get_presets_directory()
    assert result == Path("/env/presets")


@pytest.mark.unit
def test_get_presets_directory_explicit_overrides_env() -> None:
    """Test explicit path overrides environment variable."""
    with patch.dict("os.environ", {"KARENINA_PRESETS_DIR": "/env/presets"}):
        result = _get_presets_directory(Path("/explicit/presets"))
    assert result == Path("/explicit/presets")


# =============================================================================
# list_presets Tests
# =============================================================================


@pytest.mark.unit
def test_list_presets_empty_directory(tmp_path: Path) -> None:
    """Test listing presets from empty directory."""
    result = list_presets(tmp_path)
    assert result == []


@pytest.mark.unit
def test_list_presets_nonexistent_directory() -> None:
    """Test listing presets from nonexistent directory."""
    result = list_presets(Path("/nonexistent/presets"))
    assert result == []


@pytest.mark.unit
def test_list_presets_single_file(tmp_path: Path) -> None:
    """Test listing presets with single preset file."""
    preset_file = tmp_path / "default.json"
    preset_file.write_text("{}")

    result = list_presets(tmp_path)
    assert len(result) == 1
    assert result[0]["name"] == "default"
    assert "filepath" in result[0]
    assert "modified" in result[0]


@pytest.mark.unit
def test_list_presets_multiple_files_sorted(tmp_path: Path) -> None:
    """Test listing presets with multiple files (sorted by name)."""
    (tmp_path / "zebra.json").write_text("{}")
    (tmp_path / "alpha.json").write_text("{}")
    (tmp_path / "beta.json").write_text("{}")

    result = list_presets(tmp_path)
    assert len(result) == 3
    assert result[0]["name"] == "alpha"
    assert result[1]["name"] == "beta"
    assert result[2]["name"] == "zebra"


@pytest.mark.unit
def test_list_presets_ignores_non_json_files(tmp_path: Path) -> None:
    """Test listing presets ignores non-JSON files."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "readme.txt").write_text("text")
    (tmp_path / "script.sh").write_text("#!/bin/bash")

    result = list_presets(tmp_path)
    assert len(result) == 1
    assert result[0]["name"] == "config"


@pytest.mark.unit
def test_list_presets_skips_invalid_files(tmp_path: Path) -> None:
    """Test listing presets skips files that raise exceptions."""
    (tmp_path / "valid.json").write_text("{}")
    # Create a file that will cause getmtime to fail (symlink to nowhere)
    invalid = tmp_path / "invalid.json"
    from contextlib import suppress

    with suppress(OSError):
        invalid.symlink_to("/nonexistent/target")

    result = list_presets(tmp_path)
    # Should at least have the valid file
    assert any(p["name"] == "valid" for p in result)


# =============================================================================
# get_preset_path Tests
# =============================================================================


@pytest.mark.unit
def test_get_preset_path_direct_absolute_path(tmp_path: Path) -> None:
    """Test get_preset_path with absolute path that exists."""
    preset_file = tmp_path / "custom.json"
    preset_file.write_text("{}")

    result = get_preset_path(str(preset_file))
    assert result == preset_file.resolve()


@pytest.mark.unit
def test_get_preset_path_direct_relative_path(tmp_path: Path) -> None:
    """Test get_preset_path with relative path that exists."""
    # Create a temporary file in the actual working directory
    import tempfile

    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=".json", text=True)
    try:
        # Write some content
        with open(fd, "w") as f:
            f.write("{}")

        # Test that direct path works
        result = get_preset_path(temp_path)
        assert result == Path(temp_path).resolve()
    finally:
        # Clean up
        if Path(temp_path).exists():
            Path(temp_path).unlink()


@pytest.mark.unit
def test_get_preset_path_by_name_in_directory(tmp_path: Path) -> None:
    """Test get_preset_path finds preset by name in directory."""
    preset_file = tmp_path / "default.json"
    preset_file.write_text("{}")

    result = get_preset_path("default", tmp_path)
    assert result == preset_file.resolve()


@pytest.mark.unit
def test_get_preset_path_by_name_with_json_extension(tmp_path: Path) -> None:
    """Test get_preset_path with name including .json extension."""
    preset_file = tmp_path / "default.json"
    preset_file.write_text("{}")

    result = get_preset_path("default.json", tmp_path)
    assert result == preset_file.resolve()


@pytest.mark.unit
def test_get_preset_path_not_found(tmp_path: Path) -> None:
    """Test get_preset_path raises when preset not found."""
    with pytest.raises(FileNotFoundError, match="Preset 'missing' not found"):
        get_preset_path("missing", tmp_path)


@pytest.mark.unit
def test_get_preset_path_directory_not_found() -> None:
    """Test get_preset_path raises when presets directory doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Presets directory not found"):
        get_preset_path("missing", Path("/nonexistent/presets"))


# =============================================================================
# parse_question_indices Tests
# =============================================================================


@pytest.mark.unit
def test_parse_question_indices_single() -> None:
    """Test parsing single index."""
    result = parse_question_indices("5", total_questions=10)
    assert result == [5]


@pytest.mark.unit
def test_parse_question_indices_multiple() -> None:
    """Test parsing multiple indices."""
    result = parse_question_indices("0,2,5", total_questions=10)
    assert result == [0, 2, 5]


@pytest.mark.unit
def test_parse_question_indices_range() -> None:
    """Test parsing index range."""
    result = parse_question_indices("0-5", total_questions=10)
    assert result == [0, 1, 2, 3, 4, 5]


@pytest.mark.unit
def test_parse_question_indices_mixed() -> None:
    """Test parsing mixed indices and ranges."""
    result = parse_question_indices("0,2,5-7,9", total_questions=15)
    assert result == [0, 2, 5, 6, 7, 9]


@pytest.mark.unit
def test_parse_question_indices_unique_and_sorted() -> None:
    """Test results are unique and sorted."""
    result = parse_question_indices("5,2,5,0-2", total_questions=10)
    assert result == [0, 1, 2, 5]


@pytest.mark.unit
def test_parse_question_indices_ignores_empty_parts() -> None:
    """Test empty parts are ignored."""
    result = parse_question_indices("0,2,,5", total_questions=10)
    assert result == [0, 2, 5]


@pytest.mark.unit
def test_parse_question_indices_handles_whitespace() -> None:
    """Test whitespace is handled correctly."""
    result = parse_question_indices(" 0 , 2 - 5 , 7 ", total_questions=10)
    assert result == [0, 2, 3, 4, 5, 7]


@pytest.mark.unit
def test_parse_question_indices_negative_index_raises() -> None:
    """Test negative index raises error."""
    with pytest.raises(ValueError, match="Invalid range format"):
        parse_question_indices("-1", total_questions=10)


@pytest.mark.unit
def test_parse_question_indices_negative_in_range_raises() -> None:
    """Test negative value in range raises error."""
    with pytest.raises(ValueError, match="Invalid range format"):
        parse_question_indices("-5-0", total_questions=10)


@pytest.mark.unit
def test_parse_question_indices_range_start_gt_end_raises() -> None:
    """Test range with start > end raises error."""
    with pytest.raises(ValueError, match="start > end"):
        parse_question_indices("5-2", total_questions=10)


@pytest.mark.unit
def test_parse_question_indices_out_of_range_raises() -> None:
    """Test index out of range raises error."""
    with pytest.raises(ValueError, match="Index out of range"):
        parse_question_indices("15", total_questions=10)


@pytest.mark.unit
def test_parse_question_indices_range_out_of_range_raises() -> None:
    """Test range out of bounds raises error."""
    with pytest.raises(ValueError, match="Index out of range"):
        parse_question_indices("5-15", total_questions=10)


@pytest.mark.unit
def test_parse_question_indices_invalid_format_raises() -> None:
    """Test invalid format raises error."""
    with pytest.raises(ValueError, match="Invalid index"):
        parse_question_indices("abc", total_questions=10)


@pytest.mark.unit
def test_parse_question_indices_invalid_range_format_raises() -> None:
    """Test invalid range format raises error."""
    with pytest.raises(ValueError, match="Invalid range format"):
        parse_question_indices("0-abc", total_questions=10)


@pytest.mark.unit
def test_parse_question_indices_empty_string() -> None:
    """Test empty string returns empty list."""
    result = parse_question_indices("", total_questions=10)
    assert result == []


@pytest.mark.unit
def test_parse_question_indices_only_commas() -> None:
    """Test string with only commas returns empty list."""
    result = parse_question_indices(",,,", total_questions=10)
    assert result == []


# =============================================================================
# validate_output_path Tests
# =============================================================================


@pytest.mark.unit
def test_validate_output_path_json(tmp_path: Path) -> None:
    """Test validating JSON output path."""
    output_path = tmp_path / "results.json"
    result = validate_output_path(output_path)
    assert result == "json"


@pytest.mark.unit
def test_validate_output_path_csv(tmp_path: Path) -> None:
    """Test validating CSV output path."""
    output_path = tmp_path / "results.csv"
    result = validate_output_path(output_path)
    assert result == "csv"


@pytest.mark.unit
def test_validate_output_path_uppercase_extension(tmp_path: Path) -> None:
    """Test validating uppercase extension."""
    output_path = tmp_path / "results.JSON"
    result = validate_output_path(output_path)
    assert result == "json"


@pytest.mark.unit
def test_validate_output_path_invalid_extension_raises(tmp_path: Path) -> None:
    """Test invalid extension raises error."""
    output_path = tmp_path / "results.txt"
    with pytest.raises(ValueError, match="Invalid output format"):
        validate_output_path(output_path)


@pytest.mark.unit
def test_validate_output_path_missing_parent_raises() -> None:
    """Test missing parent directory raises error."""
    output_path = Path("/nonexistent/dir/results.json")
    with pytest.raises(ValueError, match="Parent directory does not exist"):
        validate_output_path(output_path)


# =============================================================================
# filter_templates_by_indices Tests
# =============================================================================


@pytest.mark.unit
def test_filter_templates_by_indices_empty() -> None:
    """Test filtering with empty indices returns empty list."""
    templates = [
        _make_template("q-1"),
        _make_template("q-2"),
    ]
    result = filter_templates_by_indices(templates, [])
    assert result == []


@pytest.mark.unit
def test_filter_templates_by_indices_single() -> None:
    """Test filtering by single index."""
    templates = [
        _make_template("q-1"),
        _make_template("q-2"),
        _make_template("q-3"),
    ]
    result = filter_templates_by_indices(templates, [1])
    assert len(result) == 1
    assert result[0].question_id == "q-2"


@pytest.mark.unit
def test_filter_templates_by_indices_multiple() -> None:
    """Test filtering by multiple indices."""
    templates = [
        _make_template("q-1"),
        _make_template("q-2"),
        _make_template("q-3"),
        _make_template("q-4"),
    ]
    result = filter_templates_by_indices(templates, [0, 2, 3])
    assert len(result) == 3
    assert [t.question_id for t in result] == ["q-1", "q-3", "q-4"]


@pytest.mark.unit
def test_filter_templates_by_indices_preserves_order() -> None:
    """Test filtering preserves original order."""
    templates = [
        _make_template("q-1"),
        _make_template("q-2"),
        _make_template("q-3"),
    ]
    result = filter_templates_by_indices(templates, [2, 0])
    assert [t.question_id for t in result] == ["q-1", "q-3"]


# =============================================================================
# filter_templates_by_ids Tests
# =============================================================================


@pytest.mark.unit
def test_filter_templates_by_ids_empty() -> None:
    """Test filtering with empty IDs returns empty list."""
    templates = [
        _make_template("q-1"),
        _make_template("q-2"),
    ]
    result = filter_templates_by_ids(templates, [])
    assert result == []


@pytest.mark.unit
def test_filter_templates_by_ids_single() -> None:
    """Test filtering by single ID."""
    templates = [
        _make_template("q-1"),
        _make_template("q-2"),
        _make_template("q-3"),
    ]
    result = filter_templates_by_ids(templates, ["q-2"])
    assert len(result) == 1
    assert result[0].question_id == "q-2"


@pytest.mark.unit
def test_filter_templates_by_ids_multiple() -> None:
    """Test filtering by multiple IDs."""
    templates = [
        _make_template("q-1"),
        _make_template("q-2"),
        _make_template("q-3"),
        _make_template("q-4"),
    ]
    result = filter_templates_by_ids(templates, ["q-1", "q-3", "q-4"])
    assert len(result) == 3
    assert [t.question_id for t in result] == ["q-1", "q-3", "q-4"]


@pytest.mark.unit
def test_filter_templates_by_ids_nonexistent() -> None:
    """Test filtering with non-existent IDs returns subset."""
    templates = [
        _make_template("q-1"),
        _make_template("q-2"),
    ]
    result = filter_templates_by_ids(templates, ["q-1", "q-99"])
    assert len(result) == 1
    assert result[0].question_id == "q-1"


@pytest.mark.unit
def test_filter_templates_by_ids_preserves_order() -> None:
    """Test filtering preserves original order."""
    templates = [
        _make_template("q-1"),
        _make_template("q-2"),
        _make_template("q-3"),
    ]
    result = filter_templates_by_ids(templates, ["q-3", "q-1"])
    assert [t.question_id for t in result] == ["q-1", "q-3"]


# =============================================================================
# create_export_job Tests
# =============================================================================


@pytest.mark.unit
def test_create_export_job_basic() -> None:
    """Test creating basic export job."""
    from karenina.schemas.workflow.models import ModelConfig

    result_set = VerificationResultSet(
        results=[
            _make_result("q-1", completed=True),
            _make_result("q-2", completed=True),
        ]
    )
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[
            ModelConfig(
                id="answering",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
    )

    job = create_export_job(
        result_set=result_set,
        config=config,
        run_name="test-run",
        start_time=1000.0,
        end_time=2000.0,
    )

    assert job.run_name == "test-run"
    assert job.status == "completed"
    assert job.total_questions == 2
    assert job.successful_count == 2
    assert job.failed_count == 0
    assert job.start_time == 1000.0
    assert job.end_time == 2000.0
    assert job.config == config


@pytest.mark.unit
def test_create_export_job_with_failures() -> None:
    """Test creating export job with some failures."""
    from karenina.schemas.workflow.models import ModelConfig

    result_set = VerificationResultSet(
        results=[
            _make_result("q-1", completed=True),
            _make_result("q-2", completed=False),
            _make_result("q-3", completed=False),
        ]
    )
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[
            ModelConfig(
                id="answering",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
    )

    job = create_export_job(
        result_set=result_set,
        config=config,
        run_name="test",
        start_time=0.0,
        end_time=1.0,
    )

    assert job.total_questions == 3
    assert job.successful_count == 1
    assert job.failed_count == 2


@pytest.mark.unit
def test_create_export_job_default_run_name() -> None:
    """Test creating export job with empty run name uses default."""
    from karenina.schemas.workflow.models import ModelConfig

    result_set = VerificationResultSet(
        results=[
            _make_result("q-1", completed=True),
        ]
    )
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[
            ModelConfig(
                id="answering",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
    )

    job = create_export_job(
        result_set=result_set,
        config=config,
        run_name="",
        start_time=0.0,
        end_time=1.0,
    )

    # Empty run_name defaults to "cli-verification"
    assert job.run_name == "cli-verification"


@pytest.mark.unit
def test_create_export_job_generates_uuid() -> None:
    """Test creating export job generates UUID."""
    from karenina.schemas.workflow.models import ModelConfig

    result_set = VerificationResultSet(
        results=[
            _make_result("q-1", completed=True),
        ]
    )
    config = VerificationConfig(
        parsing_models=[
            ModelConfig(
                id="parsing",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        answering_models=[
            ModelConfig(
                id="answering",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
    )

    job1 = create_export_job(result_set, config, "", 0.0, 1.0)
    job2 = create_export_job(result_set, config, "", 0.0, 1.0)

    assert job1.job_id != job2.job_id


# =============================================================================
# get_traces_path Tests
# =============================================================================


@pytest.mark.unit
def test_get_traces_path_absolute_path() -> None:
    """Test getting traces path with absolute path."""
    trace_file = Path("/absolute/path/traces.json")
    with patch("pathlib.Path.exists", return_value=True):
        result = get_traces_path(trace_file)
    assert result == trace_file.resolve()


@pytest.mark.unit
def test_get_traces_path_relative_path_exists() -> None:
    """Test getting traces path with relative path that exists."""
    with patch("pathlib.Path.exists", return_value=True):
        result = get_traces_path("traces.json")
    assert result == Path("traces.json").resolve()


@pytest.mark.unit
def test_get_traces_path_finds_in_traces_directory(tmp_path: Path) -> None:
    """Test getting traces path finds file in traces/ directory."""
    trace_file = tmp_path / "traces" / "my_traces.json"
    trace_file.parent.mkdir(parents=True)
    trace_file.write_text("{}")

    with patch("pathlib.Path.cwd", return_value=tmp_path):
        result = get_traces_path("my_traces.json")

    assert result == trace_file.resolve()


@pytest.mark.unit
def test_get_traces_path_not_found(tmp_path: Path) -> None:
    """Test getting traces path raises when file not found."""
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()

    with (
        patch("pathlib.Path.cwd", return_value=tmp_path),
        pytest.raises(FileNotFoundError, match="Trace file not found"),
    ):
        get_traces_path("missing.json")


@pytest.mark.unit
def test_get_traces_path_priority_to_direct_path(tmp_path: Path) -> None:
    """Test direct path takes priority over traces/ directory."""
    # Create direct file in tmp_path
    direct_file = tmp_path / "traces.json"
    direct_file.write_text("direct")

    # Create traces subdirectory with same name file
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    traces_file = traces_dir / "traces.json"
    traces_file.write_text("traces_dir")

    # When looking for "traces.json", should find the direct file first
    # since Path("traces.json").exists() checks the current working directory
    result = get_traces_path(str(direct_file))

    assert result == direct_file.resolve()


# =============================================================================
# load_manual_traces_from_file Tests
# =============================================================================


@pytest.mark.unit
def test_load_manual_traces_from_file_valid(tmp_path: Path) -> None:
    """Test loading valid manual traces file."""
    trace_file = tmp_path / "traces.json"
    traces_data = {
        "936dbc8755f623c951d96ea2b03e13bc": "Answer for question 1",
        "8f2e2b1e4d5c6a7b8c9d0e1f2a3b4c5d": "Answer for question 2",
    }
    trace_file.write_text(json.dumps(traces_data))

    mock_benchmark = MagicMock()

    with (
        patch("karenina.adapters.manual.load_manual_traces"),
        patch("karenina.adapters.manual.ManualTraces", return_value="mock_manual_traces") as MockManualTraces,
    ):
        result = load_manual_traces_from_file(trace_file, mock_benchmark)
        assert result == "mock_manual_traces"
        MockManualTraces.assert_called_once_with(mock_benchmark)


@pytest.mark.unit
def test_load_manual_traces_from_file_not_found() -> None:
    """Test loading non-existent file raises error."""
    mock_benchmark = MagicMock()
    with pytest.raises(FileNotFoundError, match="Manual traces file not found"):
        load_manual_traces_from_file(Path("/nonexistent/traces.json"), mock_benchmark)


@pytest.mark.unit
def test_load_manual_traces_from_file_invalid_json(tmp_path: Path) -> None:
    """Test loading invalid JSON raises error."""
    trace_file = tmp_path / "traces.json"
    trace_file.write_text("not valid json")

    mock_benchmark = MagicMock()
    with pytest.raises(json.JSONDecodeError):
        load_manual_traces_from_file(trace_file, mock_benchmark)


@pytest.mark.unit
def test_load_manual_traces_from_file_not_a_dict(tmp_path: Path) -> None:
    """Test loading non-dict JSON raises error."""
    trace_file = tmp_path / "traces.json"
    trace_file.write_text('["list", "not", "dict"]')

    mock_benchmark = MagicMock()
    with pytest.raises(ValueError, match="expected JSON object"):
        load_manual_traces_from_file(trace_file, mock_benchmark)
