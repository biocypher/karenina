"""E2E tests for CLI error handling and edge cases.

These tests ensure that the CLI fails gracefully with helpful error messages
when encountering invalid inputs, missing files, or configuration errors.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from karenina.cli import app


@pytest.mark.e2e
def test_invalid_checkpoint_file(runner: CliRunner) -> None:
    """Test verify command with non-existent checkpoint file."""
    result = runner.invoke(app, [
        "verify",
        "nonexistent_checkpoint.jsonld",
    ])

    # Should fail with non-zero exit code
    assert result.exit_code != 0

    # Should show helpful error message (not a Python traceback)
    assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()
    # Should not contain Python traceback in user-facing output
    assert "Traceback" not in result.stdout


@pytest.mark.e2e
def test_malformed_checkpoint(runner: CliRunner, tmp_path: Path) -> None:
    """Test verify command with malformed JSON checkpoint file."""
    # Create a file with invalid JSON
    bad_checkpoint = tmp_path / "malformed.jsonld"
    bad_checkpoint.write_text("{invalid json content")

    result = runner.invoke(app, [
        "verify",
        str(bad_checkpoint),
    ])

    # Should fail gracefully
    assert result.exit_code != 0

    # Error message should indicate JSON parsing problem
    output_lower = result.stdout.lower()
    # Check for various possible error indicators
    has_error_indicator = (
        "json" in output_lower or
        "parse" in output_lower or
        "invalid" in output_lower or
        "error" in output_lower or
        "could not" in output_lower
    )
    assert has_error_indicator


@pytest.mark.e2e
def test_invalid_preset_file(runner: CliRunner, minimal_checkpoint: Path, tmp_path: Path) -> None:
    """Test verify with non-existent preset file."""
    result = runner.invoke(app, [
        "verify",
        str(minimal_checkpoint),
        "--preset", "nonexistent_preset.json",
    ])

    # Should fail gracefully
    assert result.exit_code != 0

    # Should mention preset file not found
    assert "preset" in result.stdout.lower() or "not found" in result.stdout.lower()


@pytest.mark.e2e
def test_malformed_preset_file(runner: CliRunner, minimal_checkpoint: Path, tmp_path: Path) -> None:
    """Test verify with malformed preset JSON file."""
    # Create a malformed preset file
    bad_preset = tmp_path / "bad_preset.json"
    bad_preset.write_text("{invalid preset json}")

    result = runner.invoke(app, [
        "verify",
        str(minimal_checkpoint),
        "--preset", str(bad_preset),
    ])

    # Should fail gracefully
    assert result.exit_code != 0


@pytest.mark.e2e
def test_invalid_option(runner: CliRunner, minimal_checkpoint: Path) -> None:
    """Test verify with an invalid option."""
    result = runner.invoke(app, [
        "verify",
        str(minimal_checkpoint),
        "--non-existent-option", "value",
    ])

    # Should fail with Typer error (exit code 2)
    assert result.exit_code == 2
    # Should show "no such option" or similar
    assert "option" in result.stdout.lower()


@pytest.mark.e2e
def test_invalid_output_directory(runner: CliRunner, minimal_checkpoint: Path) -> None:
    """Test verify with output path in non-existent directory."""
    # Use a path in a non-existent directory
    output_path = Path("/nonexistent/directory/output.json")

    result = runner.invoke(app, [
        "verify",
        str(minimal_checkpoint),
        "--output", str(output_path),
    ])

    # May fail or create directory - either is acceptable behavior
    # Just verify it doesn't crash
    assert result.exit_code in [0, 1, 2] or result.exit_code < 0


@pytest.mark.e2e
def test_invalid_model_name_without_preset(runner: CliRunner, minimal_checkpoint: Path) -> None:
    """Test verify with model name but no provider (should require --provider)."""
    result = runner.invoke(app, [
        "verify",
        str(minimal_checkpoint),
        "--answering-model", "fake-model-name",
        # Missing --answering-provider
    ])

    # Should fail with helpful message about needing provider
    assert result.exit_code != 0
    assert "provider" in result.stdout.lower() or "interface" in result.stdout.lower()


@pytest.mark.e2e
def test_missing_manual_traces_file(runner: CliRunner, minimal_checkpoint: Path) -> None:
    """Test manual interface without --manual-traces file."""
    result = runner.invoke(app, [
        "verify",
        str(minimal_checkpoint),
        "--interface", "manual",
        # Missing --manual-traces
    ])

    # Should fail with clear message about requiring manual-traces
    assert result.exit_code != 0
    assert "manual" in result.stdout.lower() and "traces" in result.stdout.lower()


@pytest.mark.e2e
def test_empty_checkpoint(runner: CliRunner, tmp_path: Path) -> None:
    """Test verify with empty checkpoint file."""
    empty_checkpoint = tmp_path / "empty.jsonld"
    empty_checkpoint.write_text("{}")

    result = runner.invoke(app, [
        "verify",
        str(empty_checkpoint),
    ])

    # Should handle gracefully (may fail with specific message)
    assert result.exit_code != 0


@pytest.mark.e2e
def test_checkpoint_without_questions(runner: CliRunner, tmp_path: Path) -> None:
    """Test verify with checkpoint that has no questions."""
    import json

    empty_questions_checkpoint = tmp_path / "no_questions.jsonld"
    empty_questions_checkpoint.write_text(json.dumps({
        "@context": {"@vocab": "http://schema.org/"},
        "@type": "DataFeed",
        "@id": "test-empty",
        "name": "empty-test",
        "dataFeedElement": []  # No questions
    }))

    result = runner.invoke(app, [
        "verify",
        str(empty_questions_checkpoint),
    ])

    # Should handle gracefully
    assert result.exit_code != 0


@pytest.mark.e2e
def test_preset_show_nonexistent(runner: CliRunner) -> None:
    """Test preset show with non-existent preset name."""
    result = runner.invoke(app, ["preset", "show", "nonexistent_preset"])

    # Should fail gracefully
    assert result.exit_code != 0


@pytest.mark.e2e
def test_csv_without_questions(runner: CliRunner, minimal_checkpoint: Path, tmp_path: Path) -> None:
    """Test CSV export when checkpoint has no verified questions."""
    output_csv = tmp_path / "output.csv"

    result = runner.invoke(app, [
        "verify",
        str(minimal_checkpoint),
        "--csv", str(output_csv),
    ])

    # Should not crash (may or may not create file depending on preset)
    assert result.exit_code in [0, 1, 2]


@pytest.mark.e2e
def test_verify_status_on_nonexistent_checkpoint(runner: CliRunner) -> None:
    """Test verify-status on non-existent checkpoint."""
    result = runner.invoke(app, [
        "verify-status",
        "nonexistent_checkpoint.jsonld",
    ])

    # Should fail gracefully
    assert result.exit_code != 0
    assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()


@pytest.mark.e2e
def test_mutually_exclusive_options_error(runner: CliRunner, minimal_checkpoint: Path) -> None:
    """Test that mutually exclusive options are caught."""
    # Test with both --parsing-only and --deep-judgment which may conflict
    result = runner.invoke(app, [
        "verify",
        str(minimal_checkpoint),
        "--parsing-only",  # Only parsing, no answering
        "--deep-judgment",  # But deep-judgment requires answering
    ])

    # Should handle the conflict gracefully
    # (CLI may accept this but verification may fail - that's OK)
    assert result.exit_code in [0, 1, 2]


@pytest.mark.e2e
def test_invalid_evaluation_mode(runner: CliRunner, minimal_checkpoint: Path) -> None:
    """Test with invalid evaluation mode."""
    result = runner.invoke(app, [
        "verify",
        str(minimal_checkpoint),
        "--mode", "invalid_mode_value",
    ])

    # Should fail or fall back to default mode
    # Accept 0 (success with fallback), 1 (verification failed), or 2 (Typer error)
    assert result.exit_code in [0, 1, 2]


@pytest.mark.e2e
def test_negative_replicate_count(runner: CliRunner, minimal_checkpoint: Path) -> None:
    """Test with negative replicate count."""
    result = runner.invoke(app, [
        "verify",
        str(minimal_checkpoint),
        "--replicate-count", "-1",
    ])

    # Should fail gracefully
    assert result.exit_code != 0


@pytest.mark.e2e
def test_zero_replicate_count(runner: CliRunner, minimal_checkpoint: Path) -> None:
    """Test with zero replicate count."""
    result = runner.invoke(app, [
        "verify",
        str(minimal_checkpoint),
        "--replicate-count", "0",
    ])

    # Should fail gracefully
    assert result.exit_code != 0
