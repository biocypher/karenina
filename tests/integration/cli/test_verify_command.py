"""Integration tests for CLI verify command.

These tests verify that the verify command structure works correctly,
including argument parsing, option handling, and error cases.

Test scenarios:
- Verify command help and options
- Argument validation (paths, indices, models)
- Error handling for invalid inputs
- Option combinations and overrides

Note: Full verification workflows are tested in E2E tests.
These tests focus on CLI structure and argument handling.
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


def create_minimal_checkpoint(path: Path, name: str = "test") -> Path:
    """Create a minimal valid checkpoint file."""
    checkpoint_path = path / f"{name}.jsonld"
    checkpoint_data = {
        "@context": "https://schema.org",
        "@type": "DataFeed",
        "name": "Test Benchmark",
        "description": "Test benchmark for CLI tests",
        "version": "1.0.0",
        "dateCreated": "2024-01-01T00:00:00",
        "dateModified": "2024-01-01T00:00:00",
        "dataFeedElement": [
            {
                "@type": "DataFeedItem",
                "@id": "q1",
                "dateCreated": "2024-01-01T00:00:00",
                "dateModified": "2024-01-01T00:00:00",
                "item": {
                    "@type": "Question",
                    "text": "What is 2+2?",
                    "acceptedAnswer": {"@type": "Answer", "text": "4"},
                },
            }
        ],
    }
    checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))
    return checkpoint_path


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
# Verify Command Help Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.cli
class TestVerifyCommandHelp:
    """Test verify command help and documentation."""

    def test_verify_help_shows_description(self) -> None:
        """Verify help shows command description."""
        result = runner.invoke(app, ["verify", "--help"])

        assert result.exit_code == 0
        # Should mention verification or benchmark
        assert "verify" in result.stdout.lower() or "benchmark" in result.stdout.lower()

    def test_verify_help_shows_benchmark_argument(self) -> None:
        """Verify help shows benchmark path argument."""
        result = runner.invoke(app, ["verify", "--help"])

        assert result.exit_code == 0
        # Should mention benchmark or checkpoint
        assert "benchmark" in result.stdout.lower() or "checkpoint" in result.stdout.lower()

    def test_verify_help_shows_model_options(self) -> None:
        """Verify help shows model configuration options."""
        result = runner.invoke(app, ["verify", "--help"])

        assert result.exit_code == 0
        assert "model" in result.stdout.lower()

    def test_verify_help_shows_preset_option(self) -> None:
        """Verify help shows preset option."""
        result = runner.invoke(app, ["verify", "--help"])

        assert result.exit_code == 0
        assert "preset" in result.stdout.lower()

    def test_verify_help_shows_output_option(self) -> None:
        """Verify help shows output option."""
        result = runner.invoke(app, ["verify", "--help"])

        assert result.exit_code == 0
        assert "output" in result.stdout.lower()


# =============================================================================
# Verify Command Error Handling Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.cli
class TestVerifyCommandErrors:
    """Test verify command error handling."""

    def test_missing_benchmark_file(self, tmp_path: Path) -> None:
        """Verify error when benchmark file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.jsonld"
        output_path = tmp_path / "results.json"

        result = runner.invoke(app, ["verify", str(nonexistent), "--output", str(output_path)])

        # Should fail with non-zero exit code
        assert result.exit_code != 0

    def test_invalid_benchmark_file(self, tmp_path: Path) -> None:
        """Verify error when benchmark file is invalid JSON."""
        invalid_path = tmp_path / "invalid.jsonld"
        invalid_path.write_text("{ this is not valid json }")
        output_path = tmp_path / "results.json"

        result = runner.invoke(app, ["verify", str(invalid_path), "--output", str(output_path)])

        # Should fail with non-zero exit code
        assert result.exit_code != 0

    def test_invalid_preset_reference(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verify error when preset doesn't exist."""
        monkeypatch.setenv("KARENINA_PRESETS_DIR", str(tmp_path))
        checkpoint = create_minimal_checkpoint(tmp_path)
        output_path = tmp_path / "results.json"

        result = runner.invoke(
            app, ["verify", str(checkpoint), "--preset", "nonexistent", "--output", str(output_path)]
        )

        # Should fail with non-zero exit code
        assert result.exit_code != 0

    def test_invalid_output_extension(self, tmp_path: Path) -> None:
        """Verify error for unsupported output file extension."""
        checkpoint = create_minimal_checkpoint(tmp_path)
        invalid_output = tmp_path / "output.xyz"

        result = runner.invoke(app, ["verify", str(checkpoint), "--output", str(invalid_output)])

        # Should fail - only .json and .csv are supported
        assert result.exit_code != 0


# =============================================================================
# Verify Command Option Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.cli
class TestVerifyCommandOptions:
    """Test verify command option parsing."""

    def test_preset_option_accepted(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Verify preset option is accepted."""
        monkeypatch.setenv("KARENINA_PRESETS_DIR", str(tmp_path))
        checkpoint = create_minimal_checkpoint(tmp_path)
        create_minimal_preset(tmp_path, "my-preset")
        output_path = tmp_path / "results.json"

        # Just check the option is parsed (may fail later due to API key)
        result = runner.invoke(app, ["verify", str(checkpoint), "--preset", "my-preset", "--output", str(output_path)])

        # Should parse the option (may fail later for other reasons)
        # Check it doesn't fail on "unrecognized option" style errors
        assert "unrecognized" not in result.stdout.lower()

    def test_output_option_accepted(self, tmp_path: Path) -> None:
        """Verify output option is accepted."""
        checkpoint = create_minimal_checkpoint(tmp_path)
        output_path = tmp_path / "results.json"

        result = runner.invoke(app, ["verify", str(checkpoint), "--output", str(output_path)])

        # Option should be parsed (may fail later for API key)
        assert "unrecognized" not in result.stdout.lower()

    def test_replicate_count_option_accepted(self, tmp_path: Path) -> None:
        """Verify replicate-count option is accepted."""
        checkpoint = create_minimal_checkpoint(tmp_path)
        output_path = tmp_path / "results.json"

        result = runner.invoke(app, ["verify", str(checkpoint), "--replicate-count", "3", "--output", str(output_path)])

        assert "unrecognized" not in result.stdout.lower()

    def test_abstention_flag_accepted(self, tmp_path: Path) -> None:
        """Verify abstention flag is accepted."""
        checkpoint = create_minimal_checkpoint(tmp_path)
        output_path = tmp_path / "results.json"

        result = runner.invoke(app, ["verify", str(checkpoint), "--abstention", "--output", str(output_path)])

        assert "unrecognized" not in result.stdout.lower()

    def test_deep_judgment_flag_accepted(self, tmp_path: Path) -> None:
        """Verify deep-judgment flag is accepted."""
        checkpoint = create_minimal_checkpoint(tmp_path)
        output_path = tmp_path / "results.json"

        result = runner.invoke(app, ["verify", str(checkpoint), "--deep-judgment", "--output", str(output_path)])

        assert "unrecognized" not in result.stdout.lower()


# =============================================================================
# Verify Status Command Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.cli
class TestVerifyStatusCommand:
    """Test verify-status command."""

    def test_verify_status_help(self) -> None:
        """Verify verify-status help is displayed."""
        result = runner.invoke(app, ["verify-status", "--help"])

        assert result.exit_code == 0
        assert "status" in result.stdout.lower()

    def test_verify_status_nonexistent_file(self, tmp_path: Path) -> None:
        """Verify status fails gracefully for missing state file."""
        nonexistent = tmp_path / "nonexistent.state.json"

        result = runner.invoke(app, ["verify-status", str(nonexistent)])

        # Should fail with appropriate message
        assert result.exit_code != 0 or "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_verify_status_invalid_file(self, tmp_path: Path) -> None:
        """Verify status fails gracefully for invalid state file."""
        invalid_path = tmp_path / "invalid.state.json"
        invalid_path.write_text("{ invalid json }")

        result = runner.invoke(app, ["verify-status", str(invalid_path)])

        # Should fail or show error
        assert result.exit_code != 0 or "error" in result.stdout.lower()


# =============================================================================
# CLI Integration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.cli
class TestCLIIntegration:
    """Test CLI command integration."""

    def test_version_or_help_at_root(self) -> None:
        """Verify root command shows help."""
        result = runner.invoke(app)

        # Should show help or version info
        assert result.exit_code == 0
        assert "karenina" in result.stdout.lower() or "verify" in result.stdout.lower()

    def test_commands_are_discoverable(self) -> None:
        """Verify all main commands appear in help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "verify" in result.stdout.lower()
        assert "preset" in result.stdout.lower()
