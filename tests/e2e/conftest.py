"""Pytest fixtures for end-to-end CLI testing.

E2E tests invoke actual CLI entry points through Typer's CliRunner. Fixtures
provide complete environments including checkpoints, presets, and temporary
directories.

These fixtures are NOT used for unit or integration tests - those use the
shared fixtures in tests/conftest.py.
"""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from karenina import Benchmark

# =============================================================================
# CLI Runner
# =============================================================================


@pytest.fixture
def runner() -> CliRunner:
    """Return a Typer CliRunner for invoking CLI commands.

    The CliRunner captures stdout/stderr and exit codes without invoking
    a subprocess, making tests fast and reliable.

    Example:
        result = runner.invoke(app, ["verify", "checkpoint.jsonld"])
        assert result.exit_code == 0
        assert "verified" in result.stdout
    """
    return CliRunner()


# =============================================================================
# Checkpoint fixtures
# =============================================================================

# NOTE: fixtures_dir is inherited from the root conftest.py at tests/conftest.py
# Do not redefine it here - pytest will automatically use the parent fixture


@pytest.fixture
def minimal_checkpoint(fixtures_dir: Path) -> Path:
    """Return path to minimal checkpoint fixture (1 question).

    This checkpoint has a simple arithmetic question with a working template.
    Use for basic CLI verification tests.
    """
    return fixtures_dir / "checkpoints" / "minimal.jsonld"


@pytest.fixture
def checkpoint_with_results(fixtures_dir: Path) -> Path:
    """Return path to checkpoint with existing verification results.

    Use for testing status reporting, incremental verification, and
    result export functionality.
    """
    return fixtures_dir / "checkpoints" / "with_results.jsonld"


@pytest.fixture
def large_checkpoint(fixtures_dir: Path) -> Path:
    """Return path to large checkpoint fixture (5+ questions).

    Use for testing batch operations, filtering, and progress reporting.
    """
    return fixtures_dir / "checkpoints" / "multi_question.jsonld"


@pytest.fixture
def loaded_minimal_checkpoint(minimal_checkpoint: Path) -> Benchmark:
    """Return a loaded Benchmark from the minimal checkpoint fixture.

    Use when tests need to inspect the Benchmark object directly rather
    than passing a file path to the CLI.
    """
    return Benchmark.load(minimal_checkpoint)


@pytest.fixture
def loaded_large_checkpoint(large_checkpoint: Path) -> Benchmark:
    """Return a loaded Benchmark from the large checkpoint fixture."""
    return Benchmark.load(large_checkpoint)


# =============================================================================
# Preset fixtures
# =============================================================================


@pytest.fixture
def preset_dict() -> dict[str, Any]:
    """Return a minimal VerificationConfig as a dictionary.

    Use with tmp_preset_file to create valid preset JSON files.

    Includes manual interface (no API key required) for testing.
    """
    return {
        "parsing_models": [
            {
                "provider": "manual",
                "name": "manual",
                "interface": "manual",
            }
        ],
        "answering_models": [],
        "parsing_only": True,
        "replicate_count": 1,
        "rubric_enabled": False,
        "evaluation_mode": "template_only",
        "async_enabled": False,
    }


@pytest.fixture
def preset_with_manual(preset_dict: dict[str, Any]) -> dict[str, Any]:
    """Return preset config using manual interface (no API calls)."""
    return preset_dict


@pytest.fixture
def preset_with_claude() -> dict[str, Any]:
    """Return preset config for Claude (requires ANTHROPIC_API_KEY).

    Use with env_with_api_key fixture to set environment variable.
    """
    return {
        "parsing_models": [
            {
                "provider": "anthropic",
                "name": "claude-haiku-4-5",
                "interface": "langchain",
                "temperature": 0.0,
            }
        ],
        "answering_models": [
            {
                "provider": "anthropic",
                "name": "claude-haiku-4-5",
                "interface": "langchain",
                "temperature": 0.0,
            }
        ],
        "replicate_count": 1,
        "rubric_enabled": False,
        "evaluation_mode": "template_only",
        "async_enabled": False,
    }


@pytest.fixture
def tmp_preset_file(tmp_path: Path, preset_dict: dict[str, Any]) -> Path:
    """Create a temporary preset JSON file.

    Returns the path to a valid preset file using the manual interface.
    File is automatically cleaned up after the test.

    Example:
        def test_verify_with_preset(runner, tmp_preset_file, minimal_checkpoint):
            result = runner.invoke(app, [
                "verify",
                str(minimal_checkpoint),
                "--preset", str(tmp_preset_file)
            ])
            assert result.exit_code == 0
    """
    import json

    preset_path = tmp_path / "test-preset.json"
    with preset_path.open("w") as f:
        json.dump(preset_dict, f, indent=2)
    return preset_path


@pytest.fixture
def tmp_preset_file_claude(tmp_path: Path, preset_with_claude: dict[str, Any]) -> Path:
    """Create a temporary preset JSON file for Claude (requires API key)."""
    import json

    preset_path = tmp_path / "test-preset-claude.json"
    with preset_path.open("w") as f:
        json.dump(preset_with_claude, f, indent=2)
    return preset_path


@pytest.fixture
def tmp_presets_dir(tmp_path: Path) -> Path:
    """Create a temporary presets directory with a default.json preset.

    Use for testing preset discovery and listing commands.

    Example:
        def test_preset_list(runner, tmp_presets_dir):
            result = runner.invoke(app, ["preset", "list"])
            assert "default" in result.stdout
    """
    import json

    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()

    default_preset = presets_dir / "default.json"
    # Preset files require a top-level "config" wrapper
    # Use langchain interface to avoid manual_traces requirement
    preset_with_wrapper = {
        "name": "default",
        "config": {
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
        },
    }
    with default_preset.open("w") as f:
        json.dump(preset_with_wrapper, f, indent=2)

    return presets_dir


# =============================================================================
# Environment fixtures
# =============================================================================


@pytest.fixture
def env_with_api_key(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set mock API key environment variables for testing.

    This fixture uses monkeypatch to set environment variables that are
    automatically restored after the test.

    Example:
        def test_with_api_key(runner, env_with_api_key, minimal_checkpoint):
            result = runner.invoke(app, ["verify", str(minimal_checkpoint)])
            assert result.exit_code == 0
    """
    # Set mock API keys (not real values)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic-key-12345")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key-67890")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key-abcdef")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test-key-12345")

    return {
        "ANTHROPIC_API_KEY": "sk-test-anthropic-key-12345",
        "OPENAI_API_KEY": "sk-test-openai-key-67890",
        "GOOGLE_API_KEY": "test-google-key-abcdef",
        "TAVILY_API_KEY": "tavily-test-key-12345",
    }


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all KARENINA_* environment variables for testing defaults.

    Use when testing default behavior without environment influence.
    """
    for key in list(monkeypatch.os.environ):
        if key.startswith("KARENINA_"):
            monkeypatch.delenv(key, raising=False)


# =============================================================================
# Output fixtures
# =============================================================================


@pytest.fixture
def output_json(tmp_path: Path) -> Path:
    """Return a path for JSON output in tmp directory.

    File is created automatically to ensure parent directory exists.

    Example:
        def test_verify_output(runner, minimal_checkpoint, output_json):
            result = runner.invoke(app, [
                "verify", str(minimal_checkpoint),
                "--output", str(output_json)
            ])
            assert output_json.exists()
    """
    output_path = tmp_path / "output.json"
    # Create parent dir if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


@pytest.fixture
def output_csv(tmp_path: Path) -> Path:
    """Return a path for CSV output in tmp directory."""
    output_path = tmp_path / "output.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


# =============================================================================
# Trace fixtures
# =============================================================================


@pytest.fixture
def tmp_traces_file(tmp_path: Path) -> Path:
    """Create a temporary manual traces JSON file.

    Contains question_hash -> trace_string mappings.

    Example:
        def test_with_traces(runner, minimal_checkpoint, tmp_traces_file):
            result = runner.invoke(app, [
                "verify", str(minimal_checkpoint),
                "--traces", str(tmp_traces_file)
            ])
    """
    import json

    traces_path = tmp_path / "traces.json"
    # Create a minimal traces file
    traces_data = {"936dbc8755f623c951d96ea2b03e13bc": "The answer is 4."}
    with traces_path.open("w") as f:
        json.dump(traces_data, f)
    return traces_path


# =============================================================================
# Working directory fixtures
# =============================================================================


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    """Create a workspace directory with checkpoints and presets subdirs.

    Use for testing CLI commands that expect a specific directory structure.

    Returns:
        Path to workspace directory with:
        - checkpoints/ (empty, for checkpoint files)
        - presets/ (with default.json)
    """
    import json

    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()

    # Add a default preset with proper structure
    # Use langchain interface to avoid manual_traces requirement
    default_preset_config = {
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
    default_preset = {
        "name": "default",
        "config": default_preset_config,
    }
    with (presets_dir / "default.json").open("w") as f:
        json.dump(default_preset, f, indent=2)

    return tmp_path
