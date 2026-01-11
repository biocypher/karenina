"""E2E tests for preset management commands.

These tests ensure that the preset commands (list, show, delete) work correctly
with various inputs and edge cases.
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


@pytest.mark.e2e
def test_preset_list_empty_presets_dir(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset list with empty presets directory."""
    # Create empty presets directory
    empty_presets = tmp_path / "presets"
    empty_presets.mkdir()

    result = runner.invoke(app, ["preset", "list"], env={"KARENINA_PRESETS_DIR": str(empty_presets)})

    # Should succeed but show "no presets" message
    assert result.exit_code == 0
    output_lower = result.stdout.lower()
    assert "no preset" in output_lower or "0 preset" in output_lower


@pytest.mark.e2e
def test_preset_list_with_multiple_presets(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset list displays multiple presets in a table."""
    import json

    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()

    # Create multiple preset files with valid structure
    with (presets_dir / "default.json").open("w") as f:
        json.dump(_make_valid_preset(name="default"), f)
    with (presets_dir / "fast.json").open("w") as f:
        json.dump(_make_valid_preset(name="fast"), f)
    with (presets_dir / "thorough.json").open("w") as f:
        json.dump(_make_valid_preset(name="thorough"), f)

    result = runner.invoke(app, ["preset", "list"], env={"KARENINA_PRESETS_DIR": str(presets_dir)})

    # Should show table with all presets
    assert result.exit_code == 0
    # All preset names should appear in output
    assert "default" in result.stdout
    assert "fast" in result.stdout
    assert "thorough" in result.stdout
    # Should show total count
    assert "3" in result.stdout


@pytest.mark.e2e
def test_preset_list_nonexistent_directory(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset list with non-existent presets directory."""
    nonexistent = tmp_path / "nonexistent_presets"

    result = runner.invoke(app, ["preset", "list"], env={"KARENINA_PRESETS_DIR": str(nonexistent)})

    # Should succeed but show no presets (dir doesn't exist)
    assert result.exit_code == 0
    output_lower = result.stdout.lower()
    assert "no preset" in output_lower or "0 preset" in output_lower


@pytest.mark.e2e
def test_preset_show_by_name(runner: CliRunner, tmp_presets_dir: Path) -> None:
    """Test preset show using preset name (not full path)."""
    result = runner.invoke(app, ["preset", "show", "default"], env={"KARENINA_PRESETS_DIR": str(tmp_presets_dir)})

    # Should show preset details
    assert result.exit_code == 0
    # Output should contain preset info
    output_lower = result.stdout.lower()
    assert "default" in output_lower or "preset" in output_lower
    # Should show configuration fields
    assert "replicate" in output_lower or "parsing" in output_lower


@pytest.mark.e2e
def test_preset_show_by_file_path(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset show using full file path."""
    import json

    # Create a preset file with valid structure
    preset_file = tmp_path / "my-preset.json"
    preset_data = _make_valid_preset(
        name="my-preset",
        replicate_count=3,
    )
    with preset_file.open("w") as f:
        json.dump(preset_data, f)

    result = runner.invoke(app, ["preset", "show", str(preset_file)])

    # Should show preset details
    assert result.exit_code == 0
    assert "my-preset" in result.stdout or "preset" in result.stdout.lower()
    # Should show the replicate count value
    assert "3" in result.stdout


@pytest.mark.e2e
def test_preset_show_with_json_extension(runner: CliRunner, tmp_presets_dir: Path) -> None:
    """Test preset show with .json extension included."""
    result = runner.invoke(app, ["preset", "show", "default.json"], env={"KARENINA_PRESETS_DIR": str(tmp_presets_dir)})

    # Should work the same as without extension
    assert result.exit_code == 0
    assert "default" in result.stdout.lower() or "preset" in result.stdout.lower()


@pytest.mark.e2e
def test_preset_show_nonexistent_preset(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset show with non-existent preset name."""
    result = runner.invoke(app, ["preset", "show", "nonexistent_preset"], env={"KARENINA_PRESETS_DIR": str(tmp_path)})

    # Should fail with error
    assert result.exit_code != 0
    output_lower = result.stdout.lower()
    assert "not found" in output_lower or "error" in output_lower


@pytest.mark.e2e
def test_preset_show_invalid_json(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset show with malformed preset JSON file."""
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


@pytest.mark.e2e
def test_preset_show_displays_summary_fields(runner: CliRunner, tmp_path: Path) -> None:
    """Test that preset show displays all summary fields."""
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


@pytest.mark.e2e
def test_preset_delete_with_confirmation(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset delete with user confirmation (input='y')."""
    import json

    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()

    # Create a preset to delete with valid structure
    preset_file = presets_dir / "to_delete.json"
    with preset_file.open("w") as f:
        json.dump(_make_valid_preset(name="to_delete"), f)

    # Verify file exists before deletion
    assert preset_file.exists()

    # Invoke delete with 'y' input to confirm
    result = runner.invoke(
        app, ["preset", "delete", "to_delete"], env={"KARENINA_PRESETS_DIR": str(presets_dir)}, input="y\n"
    )

    # Should succeed
    assert result.exit_code == 0
    # File should be deleted
    assert not preset_file.exists()
    # Should mention deletion
    assert "deleted" in result.stdout.lower()


@pytest.mark.e2e
def test_preset_delete_cancelled(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset delete when user cancels (input='n')."""
    import json

    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()

    # Create a preset with valid structure
    preset_file = presets_dir / "keep_me.json"
    with preset_file.open("w") as f:
        json.dump(_make_valid_preset(name="keep_me"), f)

    # Invoke delete with 'n' input to cancel
    result = runner.invoke(
        app, ["preset", "delete", "keep_me"], env={"KARENINA_PRESETS_DIR": str(presets_dir)}, input="n\n"
    )

    # Should succeed (cancellation is not an error)
    assert result.exit_code == 0
    # File should still exist
    assert preset_file.exists()
    # Should mention cancellation
    assert "cancel" in result.stdout.lower()


@pytest.mark.e2e
def test_preset_delete_by_full_path(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset delete using full file path."""
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


@pytest.mark.e2e
def test_preset_delete_nonexistent(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset delete with non-existent preset."""
    result = runner.invoke(
        app, ["preset", "delete", "nonexistent"], env={"KARENINA_PRESETS_DIR": str(tmp_path)}, input="y\n"
    )

    # Should fail with error
    assert result.exit_code != 0
    output_lower = result.stdout.lower()
    assert "not found" in output_lower or "error" in output_lower


@pytest.mark.e2e
def test_preset_list_with_env_var(runner: CliRunner, tmp_presets_dir: Path) -> None:
    """Test preset list respects KARENINA_PRESETS_DIR environment variable."""
    result = runner.invoke(app, ["preset", "list"], env={"KARENINA_PRESETS_DIR": str(tmp_presets_dir)})

    # Should find the preset from tmp_presets_dir
    assert result.exit_code == 0
    assert "default" in result.stdout


@pytest.mark.e2e
def test_preset_show_with_env_var(runner: CliRunner, tmp_presets_dir: Path) -> None:
    """Test preset show respects KARENINA_PRESETS_DIR environment variable."""
    result = runner.invoke(app, ["preset", "show", "default"], env={"KARENINA_PRESETS_DIR": str(tmp_presets_dir)})

    # Should find preset from the custom directory
    assert result.exit_code == 0
    output_lower = result.stdout.lower()
    assert "default" in output_lower or "preset" in output_lower


@pytest.mark.e2e
def test_preset_list_ignores_non_json_files(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset list ignores non-.json files in presets directory."""
    import json

    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()

    # Create valid preset with proper structure
    with (presets_dir / "valid.json").open("w") as f:
        json.dump(_make_valid_preset(name="valid"), f)

    # Create non-JSON files that should be ignored
    (presets_dir / "README.md").write_text("# Presets")
    (presets_dir / "config.txt").write_text("not a preset")
    (presets_dir / ".hidden").write_text("hidden file")

    result = runner.invoke(app, ["preset", "list"], env={"KARENINA_PRESETS_DIR": str(presets_dir)})

    # Should only show the valid preset
    assert result.exit_code == 0
    assert "valid" in result.stdout
    # Should show count of 1
    assert "1" in result.stdout
    # Non-JSON files should not appear in the table
    assert "README" not in result.stdout
    assert "config.txt" not in result.stdout


@pytest.mark.e2e
def test_preset_list_sorts_alphabetically(runner: CliRunner, tmp_path: Path) -> None:
    """Test preset list outputs presets in alphabetical order."""
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
