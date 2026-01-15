"""E2E tests for preset management commands - Edge Cases.

These tests cover edge cases and advanced scenarios for preset commands
that are NOT covered by the integration tests in tests/integration/cli/test_preset_command.py.

Core functionality (basic list, show, delete operations) is tested in the
integration tests. This file focuses on:
- Non-existent directories
- JSON extension handling
- Invalid JSON handling
- Detailed summary field verification
- Delete by full path
- Alphabetical sorting

See tests/integration/cli/test_preset_command.py for core preset CLI tests.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from karenina.cli import app


def _make_valid_preset(**kwargs) -> dict:
    """Return a valid preset file structure with config wrapper.

    Preset files require a top-level "config" key containing the actual
    VerificationConfig data.

    For testing, we use parsing_only=True with langchain interface to
    avoid requiring manual_traces or API keys.

    Note: Model config uses 'model_provider' and 'model_name' as keys.
    """
    default_config = {
        "parsing_models": [
            {
                "id": "parsing-1",
                "model_provider": "anthropic",
                "model_name": "claude-haiku-4-5",
                "interface": "langchain",
                "temperature": 0.0,
            }
        ],
        "answering_models": [],
        "parsing_only": True,
        "replicate_count": 1,
        "rubric_enabled": False,
        "evaluation_mode": "template_only",
        "async_enabled": False,
    }
    default_config.update(kwargs)
    return {"name": "test-preset", "config": default_config}


# =============================================================================
# Edge Case: Non-existent Directories
# =============================================================================


@pytest.mark.e2e
def test_preset_list_nonexistent_directory(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset list with non-existent presets directory.

    Unlike empty directory (tested in integration), this tests when the
    directory itself doesn't exist at all.
    """
    nonexistent = tmp_path / "nonexistent_presets"

    result = runner.invoke(app, ["preset", "list"], env={"KARENINA_PRESETS_DIR": str(nonexistent)})

    # Should succeed but show no presets (dir doesn't exist)
    assert result.exit_code == 0
    output_lower = result.stdout.lower()
    assert "no preset" in output_lower or "0 preset" in output_lower


# =============================================================================
# Edge Case: JSON Extension Handling
# =============================================================================


@pytest.mark.e2e
def test_preset_show_with_json_extension(runner: CliRunner, tmp_presets_dir: Path) -> None:
    """Test preset show with .json extension included.

    Verifies that users can specify "default.json" or just "default"
    and get the same result.
    """
    result = runner.invoke(app, ["preset", "show", "default.json"], env={"KARENINA_PRESETS_DIR": str(tmp_presets_dir)})

    # Should work the same as without extension
    assert result.exit_code == 0
    assert "default" in result.stdout.lower() or "preset" in result.stdout.lower()


# =============================================================================
# Edge Case: Invalid JSON Handling
# =============================================================================


@pytest.mark.e2e
def test_preset_show_invalid_json(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset show with malformed preset JSON file.

    Verifies graceful error handling when a .json file contains
    invalid JSON syntax.
    """
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()

    # Create a malformed preset file
    bad_preset = presets_dir / "bad.json"
    bad_preset.write_text("{invalid json content")

    result = runner.invoke(app, ["preset", "show", "bad"], env={"KARENINA_PRESETS_DIR": str(presets_dir)})

    # Should fail gracefully
    assert result.exit_code != 0
    output_lower = result.stdout.lower()
    # Should indicate error (json, parse, or error)
    has_error_indicator = (
        "json" in output_lower or "parse" in output_lower or "invalid" in output_lower or "error" in output_lower
    )
    assert has_error_indicator


# =============================================================================
# Edge Case: Detailed Summary Fields
# =============================================================================


@pytest.mark.e2e
def test_preset_show_displays_summary_fields(runner: CliRunner, tmp_path: Path) -> None:
    """Test that preset show displays all summary fields with complex config.

    This tests a more complex preset configuration with multiple features
    enabled, verifying all summary fields are displayed correctly.
    """
    import json

    preset_file = tmp_path / "detailed.json"
    preset_data = _make_valid_preset(
        name="detailed",
        parsing_models=[
            {
                "id": "parsing-1",
                "model_provider": "anthropic",
                "model_name": "claude-haiku-4-5",
                "interface": "langchain",
            }
        ],
        answering_models=[
            {
                "id": "answering-1",
                "model_provider": "anthropic",
                "model_name": "claude-haiku-4-5",
                "interface": "langchain",
            }
        ],
        replicate_count=5,
        rubric_enabled=True,
        evaluation_mode="template_and_rubric",  # Required when rubric_enabled=True
        abstention_enabled=True,
        embedding_check_enabled=True,
        deep_judgment_enabled=False,
        async_enabled=True,
    )
    with preset_file.open("w") as f:
        json.dump(preset_data, f)

    result = runner.invoke(app, ["preset", "show", str(preset_file)])

    # Should display all summary fields
    assert result.exit_code == 0
    output_lower = result.stdout.lower()
    # Check for key summary labels
    assert "answering" in output_lower
    assert "parsing" in output_lower
    assert "replicate" in output_lower
    assert "rubric" in output_lower
    # Check displayed values
    assert "5" in result.stdout  # replicate_count
    # Enabled flags should show up
    assert "enabled" in output_lower or "true" in output_lower


# =============================================================================
# Edge Case: Delete by Full Path
# =============================================================================


@pytest.mark.e2e
def test_preset_delete_by_full_path(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset delete using full file path.

    Unlike delete by name (tested in integration), this tests deleting
    a preset file that is outside the standard presets directory.
    """
    import json

    # Create a preset outside of standard presets directory with valid structure
    preset_file = tmp_path / "custom_preset.json"
    with preset_file.open("w") as f:
        json.dump(_make_valid_preset(name="custom-preset"), f)

    # Invoke delete with full path
    result = runner.invoke(app, ["preset", "delete", str(preset_file)], input="y\n")

    # Should succeed
    assert result.exit_code == 0
    # File should be deleted
    assert not preset_file.exists()


# =============================================================================
# Edge Case: Alphabetical Sorting
# =============================================================================


@pytest.mark.e2e
def test_preset_list_sorts_alphabetically(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset list outputs presets in alphabetical order.

    Verifies the output order is alphabetically sorted regardless of
    filesystem ordering.
    """
    import json

    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()

    # Create presets in non-alphabetical order
    for name in ["zebra", "alpha", "beta", "gamma"]:
        with (presets_dir / f"{name}.json").open("w") as f:
            json.dump(_make_valid_preset(name=name), f)

    result = runner.invoke(app, ["preset", "list"], env={"KARENINA_PRESETS_DIR": str(presets_dir)})

    # Should succeed
    assert result.exit_code == 0
    # Find positions of preset names in output
    alpha_pos = result.stdout.find("alpha")
    beta_pos = result.stdout.find("beta")
    gamma_pos = result.stdout.find("gamma")
    zebra_pos = result.stdout.find("zebra")

    # Verify alphabetical order
    assert alpha_pos < beta_pos < gamma_pos < zebra_pos
