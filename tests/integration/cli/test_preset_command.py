"""Integration tests for CLI preset commands.

These tests verify that the preset CLI commands work correctly,
including list, show, and delete operations.

Test scenarios:
- preset list with various preset states
- preset show with valid/invalid presets
- preset delete with confirmation handling
- Error handling for missing/invalid presets
"""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from karenina.cli import app

runner = CliRunner()


# =============================================================================
# Helper Functions
# =============================================================================


def create_minimal_preset(path: Path, name: str = "test-preset") -> Path:
    """Create a minimal valid preset file."""
    preset_path = path / f"{name}.json"
    preset_data = {
        "config": {
            "answering_models": [
                {
                    "id": "answering-1",
                    "model_provider": "anthropic",
                    "model_name": "claude-haiku-4-5",
                }
            ],
            "parsing_models": [
                {
                    "id": "parsing-1",
                    "model_provider": "anthropic",
                    "model_name": "claude-haiku-4-5",
                }
            ],
            "replicate_count": 1,
        }
    }
    preset_path.write_text(json.dumps(preset_data, indent=2))
    return preset_path


# =============================================================================
# Preset List Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.cli
class TestPresetList:
    """Test preset list command."""

    def test_list_no_presets(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verify list shows message when no presets exist."""
        # Set empty preset directory
        monkeypatch.setenv("KARENINA_PRESETS_DIR", str(tmp_path))

        result = runner.invoke(app, ["preset", "list"])

        assert result.exit_code == 0
        assert "No presets found" in result.stdout

    def test_list_with_presets(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verify list shows available presets."""
        # Create preset directory with presets
        monkeypatch.setenv("KARENINA_PRESETS_DIR", str(tmp_path))

        create_minimal_preset(tmp_path, "preset-one")
        create_minimal_preset(tmp_path, "preset-two")

        result = runner.invoke(app, ["preset", "list"])

        assert result.exit_code == 0
        assert "preset-one" in result.stdout
        assert "preset-two" in result.stdout
        assert "2 preset" in result.stdout.lower()

    def test_list_ignores_non_json_files(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verify list ignores non-JSON files."""
        monkeypatch.setenv("KARENINA_PRESETS_DIR", str(tmp_path))

        create_minimal_preset(tmp_path, "valid-preset")
        (tmp_path / "readme.txt").write_text("This is not a preset")
        (tmp_path / "notes.md").write_text("# Notes")

        result = runner.invoke(app, ["preset", "list"])

        assert result.exit_code == 0
        assert "valid-preset" in result.stdout
        assert "readme" not in result.stdout.lower()
        assert "notes" not in result.stdout.lower()


# =============================================================================
# Preset Show Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.cli
class TestPresetShow:
    """Test preset show command."""

    def test_show_valid_preset_by_name(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verify show displays preset details when found by name."""
        monkeypatch.setenv("KARENINA_PRESETS_DIR", str(tmp_path))
        create_minimal_preset(tmp_path, "my-preset")

        result = runner.invoke(app, ["preset", "show", "my-preset"])

        assert result.exit_code == 0
        assert "my-preset" in result.stdout.lower()
        assert "Configuration" in result.stdout

    def test_show_valid_preset_by_path(self, tmp_path: Path) -> None:
        """Verify show displays preset details when given full path."""
        preset_path = create_minimal_preset(tmp_path, "full-path-preset")

        result = runner.invoke(app, ["preset", "show", str(preset_path)])

        assert result.exit_code == 0
        assert "full-path-preset" in result.stdout.lower()

    def test_show_nonexistent_preset(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verify show fails gracefully for missing preset."""
        monkeypatch.setenv("KARENINA_PRESETS_DIR", str(tmp_path))

        result = runner.invoke(app, ["preset", "show", "nonexistent"])

        assert result.exit_code == 1
        assert "Error" in result.stdout or "error" in result.stdout.lower()

    def test_show_displays_summary(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verify show displays configuration summary."""
        monkeypatch.setenv("KARENINA_PRESETS_DIR", str(tmp_path))
        create_minimal_preset(tmp_path, "summary-preset")

        result = runner.invoke(app, ["preset", "show", "summary-preset"])

        assert result.exit_code == 0
        assert "Summary" in result.stdout
        assert "Answering models" in result.stdout
        assert "Parsing models" in result.stdout


# =============================================================================
# Preset Delete Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.cli
class TestPresetDelete:
    """Test preset delete command."""

    def test_delete_confirmed(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verify delete removes preset when confirmed."""
        monkeypatch.setenv("KARENINA_PRESETS_DIR", str(tmp_path))
        preset_path = create_minimal_preset(tmp_path, "to-delete")

        assert preset_path.exists()

        # Simulate 'y' confirmation
        result = runner.invoke(app, ["preset", "delete", "to-delete"], input="y\n")

        assert result.exit_code == 0
        assert not preset_path.exists()
        assert "deleted" in result.stdout.lower()

    def test_delete_cancelled(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verify delete preserves preset when cancelled."""
        monkeypatch.setenv("KARENINA_PRESETS_DIR", str(tmp_path))
        preset_path = create_minimal_preset(tmp_path, "keep-me")

        # Simulate 'n' cancellation
        result = runner.invoke(app, ["preset", "delete", "keep-me"], input="n\n")

        assert result.exit_code == 0
        assert preset_path.exists()
        assert "cancelled" in result.stdout.lower()

    def test_delete_nonexistent_preset(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verify delete fails gracefully for missing preset."""
        monkeypatch.setenv("KARENINA_PRESETS_DIR", str(tmp_path))

        result = runner.invoke(app, ["preset", "delete", "nonexistent"], input="y\n")

        assert result.exit_code == 1
        assert "Error" in result.stdout or "error" in result.stdout.lower()


# =============================================================================
# Main CLI Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.cli
class TestMainCLI:
    """Test main CLI app structure."""

    def test_help_displays(self) -> None:
        """Verify main help is displayed."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "karenina" in result.stdout.lower()

    def test_verify_command_exists(self) -> None:
        """Verify verify command is registered."""
        result = runner.invoke(app, ["verify", "--help"])

        assert result.exit_code == 0
        assert "verify" in result.stdout.lower()

    def test_preset_command_exists(self) -> None:
        """Verify preset command group is registered."""
        result = runner.invoke(app, ["preset", "--help"])

        assert result.exit_code == 0
        assert "preset" in result.stdout.lower()
        assert "list" in result.stdout.lower()
        assert "show" in result.stdout.lower()
        assert "delete" in result.stdout.lower()

    def test_verify_status_command_exists(self) -> None:
        """Verify verify-status command is registered."""
        result = runner.invoke(app, ["verify-status", "--help"])

        assert result.exit_code == 0
