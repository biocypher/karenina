# End-to-End Test Strategy

**Parent**: [README.md](./README.md)

---

## Scope

E2E tests run complete workflows from CLI entry point to final output. They verify that all components work together in realistic scenarios.

**Approach**: Call CLI entry point directly (import and invoke `main()`) rather than subprocess — faster and easier to debug.

---

## Canonical Scenarios

| Scenario | Description | Key Assertions |
|----------|-------------|----------------|
| Full pipeline success | Verify a benchmark with all questions passing | Exit code 0, results file created |
| Mixed results | Benchmark with some passes, some failures | Correct aggregation, proper exit code |
| Resume from checkpoint | Interrupt and resume verification | No duplicate work, correct final count |
| Preset application | Apply verification preset | Config options respected |
| Error handling | Invalid inputs, missing files | Helpful error messages, non-zero exit |

---

## Test Structure

```python
"""
End-to-End Tests

These tests invoke the CLI entry point and verify complete workflows.
"""

from karenina.cli.main import main
from click.testing import CliRunner


class TestFullVerificationPipeline:
    """Complete verification workflows."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_verify_minimal_benchmark(self, runner, tmp_path, minimal_checkpoint):
        """
        Verify a minimal benchmark with one question.

        Command: karenina verify checkpoint.jsonld --output results.json
        """
        output_path = tmp_path / "results.json"

        result = runner.invoke(main, [
            "verify",
            str(minimal_checkpoint),
            "--output", str(output_path)
        ])

        assert result.exit_code == 0
        assert output_path.exists()

        with open(output_path) as f:
            results = json.load(f)

        assert "questions" in results
        assert len(results["questions"]) == 1

    def test_verify_with_preset(self, runner, tmp_path, benchmark_checkpoint, preset_file):
        """
        Verify using a preset configuration.

        Preset options should be applied to the verification run.
        """
        result = runner.invoke(main, [
            "verify",
            str(benchmark_checkpoint),
            "--preset", str(preset_file),
            "--output", str(tmp_path / "results.json")
        ])

        assert result.exit_code == 0

    def test_verify_resume_from_checkpoint(self, runner, tmp_path, large_benchmark):
        """
        Resume interrupted verification.

        Run partial verification, then resume from checkpoint.
        Should not re-process already completed questions.
        """
        checkpoint_path = tmp_path / "checkpoint.jsonld"

        # Run partial verification
        result1 = runner.invoke(main, [
            "verify",
            str(large_benchmark),
            "--checkpoint", str(checkpoint_path),
            "--max-questions", "5"
        ])

        assert result1.exit_code == 0

        # Resume
        result2 = runner.invoke(main, [
            "verify",
            str(large_benchmark),
            "--checkpoint", str(checkpoint_path),
            "--resume"
        ])

        assert result2.exit_code == 0

        # Verify all questions processed exactly once
        with open(checkpoint_path) as f:
            final = json.load(f)
        assert len(final["results"]) == 10


class TestErrorHandling:
    """Error scenarios and edge cases."""

    def test_invalid_checkpoint_error(self, runner, tmp_path):
        """
        Invalid checkpoint file → helpful error message.
        """
        invalid_path = tmp_path / "nonexistent.jsonld"

        result = runner.invoke(main, ["verify", str(invalid_path)])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_invalid_preset_error(self, runner, tmp_path, valid_checkpoint):
        """
        Invalid preset file → helpful error message.
        """
        invalid_preset = tmp_path / "bad_preset.json"
        invalid_preset.write_text("not valid json")

        result = runner.invoke(main, [
            "verify",
            str(valid_checkpoint),
            "--preset", str(invalid_preset)
        ])

        assert result.exit_code != 0

    def test_missing_api_key_error(self, runner, valid_checkpoint, monkeypatch):
        """
        Missing API key → clear error, not stack trace.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        result = runner.invoke(main, ["verify", str(valid_checkpoint)])

        assert result.exit_code != 0
        assert "api" in result.output.lower() or "key" in result.output.lower()


class TestPresetCommands:
    """Preset management commands."""

    def test_preset_list(self, runner):
        """List available presets."""
        result = runner.invoke(main, ["preset", "list"])

        assert result.exit_code == 0
        # Should show at least default preset
        assert "default" in result.output.lower() or len(result.output) > 0

    def test_preset_show(self, runner):
        """Show preset details."""
        result = runner.invoke(main, ["preset", "show", "default"])

        assert result.exit_code == 0
```

---

## Fixtures for E2E Tests

```python
# tests/e2e/conftest.py

@pytest.fixture
def minimal_checkpoint(tmp_path, fixtures_dir):
    """Single-question checkpoint for quick tests."""
    src = fixtures_dir / "checkpoints" / "minimal.jsonld"
    dst = tmp_path / "minimal.jsonld"
    dst.write_text(src.read_text())
    return dst

@pytest.fixture
def large_benchmark(tmp_path, fixtures_dir):
    """10-question benchmark for resume tests."""
    src = fixtures_dir / "checkpoints" / "complex_benchmark.jsonld"
    dst = tmp_path / "complex.jsonld"
    dst.write_text(src.read_text())
    return dst

@pytest.fixture
def preset_file(tmp_path):
    """Sample preset configuration."""
    preset = {
        "name": "test_preset",
        "model": "claude-haiku-4-5",
        "max_retries": 2
    }
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(preset))
    return path
```

---

## Directory Structure

```
tests/e2e/
├── conftest.py
├── test_full_verification_pipeline.py
├── test_batch_verification.py
└── test_cli_workflows.py
```

---

*Last updated: 2025-01-11*
