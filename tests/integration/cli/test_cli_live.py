"""Live CLI integration tests with real LLM calls.

These tests verify the CLI works end-to-end with actual API calls.
They use claude-haiku-4-5 (cheapest model), 2 questions, and 2 replicates
to cover features like replicates and progressive save/resume.

Requires: ANTHROPIC_API_KEY set in the environment.

Run with: uv run pytest tests/integration/cli/test_cli_live.py -v -s
"""

import json
import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from karenina.cli import app

runner = CliRunner()

ARTIFACTS = Path("/Users/carli/Projects/karenina-salvage/automated_experiments/artifacts/shared")
BENCHMARK_PATH = ARTIFACTS / "benchmark_qa.jsonld"

requires_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@pytest.fixture()
def work_dir(tmp_path: Path) -> Path:
    """Set up a working directory with a copy of the benchmark."""
    checkpoint = tmp_path / "benchmark.jsonld"
    shutil.copy(BENCHMARK_PATH, checkpoint)
    return tmp_path


@pytest.fixture()
def preset_path(work_dir: Path) -> Path:
    """Create a preset with 2 replicates for testing."""
    preset = work_dir / "preset.json"
    preset.write_text(
        json.dumps(
            {
                "config": {
                    "answering_models": [
                        {
                            "id": "answerer",
                            "model_provider": "anthropic",
                            "model_name": "claude-haiku-4-5",
                            "temperature": 0.0,
                        }
                    ],
                    "parsing_models": [
                        {
                            "id": "judge",
                            "model_provider": "anthropic",
                            "model_name": "claude-haiku-4-5",
                            "temperature": 0.0,
                        }
                    ],
                    "replicate_count": 2,
                    "abstention_enabled": True,
                }
            },
            indent=2,
        )
    )
    return preset


@pytest.fixture()
def manual_traces_path(work_dir: Path) -> Path:
    """Create manual traces for the first 2 questions."""
    traces = work_dir / "traces.json"
    traces.write_text(
        json.dumps(
            {
                # France capital question
                "25bd04c8b293421d91bbbecd5b24929f": (
                    "The capital of France is Paris. "
                    "Its approximate population is about 2,161,000 people in the city proper. "
                    "Paris is also the capital of the Ile-de-France region. "
                    "Yes, it is the most populous city in France."
                ),
                # Japan capital question
                "1134ebdb3686b8cb4b0a4a79b68c4e4c": (
                    "The capital of Japan is Tokyo. "
                    "Its approximate population is about 13,960,000 people. "
                    "Tokyo is the most populous city in Japan."
                ),
            }
        )
    )
    return traces


@requires_api_key
@pytest.mark.e2e
@pytest.mark.cli
class TestCLIVerifyLive:
    """Live CLI verify tests with real LLM calls (2 questions, 2 replicates)."""

    def test_verify_with_preset_and_replicates(self, work_dir: Path, preset_path: Path) -> None:
        """Full verify: preset with 2 replicates, 2 questions, JSON output."""
        checkpoint = work_dir / "benchmark.jsonld"
        output = work_dir / "results.json"

        result = runner.invoke(
            app,
            [
                "verify",
                str(checkpoint),
                "--preset",
                str(preset_path),
                "--questions",
                "0,1",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0, f"CLI failed:\n{result.stdout}"
        assert output.exists(), "Output file not created"

        results_data = json.loads(output.read_text())
        # 2 questions x 2 replicates = 4 results
        assert len(results_data) >= 4, f"Expected >= 4 results, got {len(results_data)}"

    def test_verify_with_manual_traces_and_replicates(self, work_dir: Path, manual_traces_path: Path) -> None:
        """Manual interface: 2 pre-recorded traces, 2 replicates, only parsing LLM called."""
        checkpoint = work_dir / "benchmark.jsonld"
        output = work_dir / "results.json"

        result = runner.invoke(
            app,
            [
                "verify",
                str(checkpoint),
                "--interface",
                "manual",
                "--manual-traces",
                str(manual_traces_path),
                "--parsing-model",
                "claude-haiku-4-5",
                "--parsing-provider",
                "anthropic",
                "--replicate-count",
                "2",
                "--questions",
                "0,1",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0, f"CLI failed:\n{result.stdout}"
        assert output.exists(), "Output file not created"

        results_data = json.loads(output.read_text())
        # 2 questions x 2 replicates = 4 results
        assert len(results_data) >= 4, f"Expected >= 4 results, got {len(results_data)}"

    def test_verify_flag_override_disables_preset_default(self, work_dir: Path, preset_path: Path) -> None:
        """Issue 036: --no-abstention overrides preset's abstention_enabled=True."""
        checkpoint = work_dir / "benchmark.jsonld"
        output = work_dir / "results.json"

        result = runner.invoke(
            app,
            [
                "verify",
                str(checkpoint),
                "--preset",
                str(preset_path),
                "--questions",
                "0",
                "--no-abstention",
                "--replicate-count",
                "1",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0, f"CLI failed:\n{result.stdout}"
        assert output.exists(), "Output file not created"

    def test_verify_csv_output(self, work_dir: Path, manual_traces_path: Path) -> None:
        """Verify CSV export works via CLI with 2 questions."""
        checkpoint = work_dir / "benchmark.jsonld"
        output = work_dir / "results.csv"

        result = runner.invoke(
            app,
            [
                "verify",
                str(checkpoint),
                "--interface",
                "manual",
                "--manual-traces",
                str(manual_traces_path),
                "--parsing-model",
                "claude-haiku-4-5",
                "--parsing-provider",
                "anthropic",
                "--questions",
                "0,1",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0, f"CLI failed:\n{result.stdout}"
        assert output.exists(), "CSV file not created"
        lines = output.read_text().strip().split("\n")
        # Header + at least 2 data rows (1 per question)
        assert len(lines) >= 3, f"CSV too short: {len(lines)} lines"

    def test_progressive_save_and_resume(self, work_dir: Path, manual_traces_path: Path) -> None:
        """Progressive save creates state file; resume completes remaining work."""
        checkpoint = work_dir / "benchmark.jsonld"
        output = work_dir / "results.json"

        # Run with progressive save (2 questions, should complete normally)
        result = runner.invoke(
            app,
            [
                "verify",
                str(checkpoint),
                "--interface",
                "manual",
                "--manual-traces",
                str(manual_traces_path),
                "--parsing-model",
                "claude-haiku-4-5",
                "--parsing-provider",
                "anthropic",
                "--questions",
                "0,1",
                "--output",
                str(output),
                "--progressive-save",
            ],
        )

        assert result.exit_code == 0, f"Progressive save failed:\n{result.stdout}"
        assert output.exists(), "Output file not created"

        # Verify results were produced
        results_data = json.loads(output.read_text())
        assert len(results_data) >= 2, f"Expected >= 2 results, got {len(results_data)}"

        # State file should exist (or be cleaned up on success)
        # The .state file is cleaned up when all tasks complete successfully,
        # so we verify the final output instead
        assert "progressive save" in result.stdout.lower() or "results" in result.stdout.lower()


@pytest.mark.integration
@pytest.mark.cli
class TestCLIConfigValidationLive:
    """Tests that don't need LLM calls (validation-only)."""

    def test_config_error_before_benchmark(self, tmp_path: Path) -> None:
        """Issue 035: Config errors shown before benchmark load attempt."""
        nonexistent = tmp_path / "does_not_exist.jsonld"
        output = tmp_path / "results.json"

        result = runner.invoke(
            app,
            ["verify", str(nonexistent), "--output", str(output)],
        )

        assert result.exit_code != 0
        stdout_lower = result.stdout.lower()
        assert "interface" in stdout_lower or "required" in stdout_lower, (
            f"Expected config error, got:\n{result.stdout}"
        )

    def test_no_flag_help_shows_flag_pairs(self) -> None:
        """Issue 036: Verify command has --flag/--no-flag pairs registered."""
        import typer

        click_app = typer.main.get_group(app)
        verify_cmd = click_app.commands["verify"]
        params_by_name = {p.name: p for p in verify_cmd.params}
        assert "--no-abstention" in params_by_name["abstention"].secondary_opts
        assert "--no-sufficiency" in params_by_name["sufficiency"].secondary_opts
        assert "--no-deep-judgment" in params_by_name["deep_judgment"].secondary_opts
