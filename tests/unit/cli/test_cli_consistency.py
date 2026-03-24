"""Tests for CLI consistency fixes (issues 035, 036, 037).

Issue 035: CLI verify validates benchmark before config, masking config errors.
Issue 036: CLI boolean feature flags are enable-only, cannot override preset defaults.
Issue 037: CLI embedding defaults conflict with env var defaults in VerificationConfig.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from karenina.cli import app
from karenina.cli.verify_config import build_config_from_cli_args
from karenina.schemas import VerificationConfig
from karenina.schemas.verification.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_THRESHOLD

runner = CliRunner()


def _create_checkpoint(path: Path) -> Path:
    """Create a minimal valid checkpoint file."""
    checkpoint_path = path / "test.jsonld"
    checkpoint_data = {
        "@context": "https://schema.org",
        "@type": "DataFeed",
        "name": "Test",
        "description": "Test",
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


def _create_preset(path: Path, **overrides: Any) -> Path:
    """Create a preset with optional config overrides."""
    preset_path = path / "preset.json"
    config = {
        "answering_models": [{"id": "answering-1", "model_provider": "anthropic", "model_name": "claude-haiku-4-5"}],
        "parsing_models": [{"id": "parsing-1", "model_provider": "anthropic", "model_name": "claude-haiku-4-5"}],
        "replicate_count": 1,
    }
    config.update(overrides)
    preset_path.write_text(json.dumps({"config": config}, indent=2))
    return preset_path


# =============================================================================
# Issue 035: Config validation should run before benchmark loading
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
class TestConfigValidationOrder:
    """Issue 035: Config errors should be reported even when benchmark path is invalid."""

    def test_config_error_shown_before_benchmark_error(self, tmp_path: Path) -> None:
        """When both config and benchmark are invalid, config error should appear.

        Currently, the benchmark load error masks config errors because
        Benchmark.load() runs first. After the fix, config validation
        should run first, so missing --interface is reported even when
        the benchmark file doesn't exist.
        """
        nonexistent_benchmark = tmp_path / "does_not_exist.jsonld"
        output_path = tmp_path / "results.json"

        result = runner.invoke(
            app,
            [
                "verify",
                str(nonexistent_benchmark),
                "--output",
                str(output_path),
                # No --interface, no --answering-model, no --parsing-model, no --preset
            ],
        )

        assert result.exit_code != 0
        # The config error should appear, not a benchmark loading error
        assert "interface" in result.stdout.lower() or "required" in result.stdout.lower()


# =============================================================================
# Issue 036: Boolean flags should support --flag/--no-flag pairs
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
class TestBooleanFlagPairs:
    """Issue 036: Boolean flags should be tri-state (None/True/False) to allow overriding preset defaults."""

    def test_no_flag_does_not_override_preset_abstention(self, tmp_path: Path) -> None:
        """When user doesn't pass --abstention or --no-abstention, preset's value should be preserved.

        Currently, the CLI passes abstention=False (the default) to from_overrides(),
        which overrides a preset that has abstention_enabled=True.
        """
        preset_path = _create_preset(tmp_path, abstention_enabled=True)
        preset_config = VerificationConfig.from_preset(preset_path)

        # Simulate CLI with no flag (should be None, not False)
        config = build_config_from_cli_args(
            answering_model=None,
            answering_provider=None,
            answering_id="answering-1",
            parsing_model=None,
            parsing_provider=None,
            parsing_id="parsing-1",
            temperature=None,
            interface=None,
            replicate_count=None,
            abstention=None,  # Should be None when user doesn't pass flag
            sufficiency=None,
            embedding_check=None,
            deep_judgment=None,
            deep_judgment_rubric_mode="disabled",
            deep_judgment_rubric_excerpts=True,
            deep_judgment_rubric_max_excerpts=3,
            deep_judgment_rubric_fuzzy_threshold=0.8,
            deep_judgment_rubric_retry_attempts=1,
            deep_judgment_rubric_search=False,
            deep_judgment_rubric_search_tool="tavily",
            deep_judgment_rubric_config=None,
            use_full_trace_for_template=False,
            use_full_trace_for_rubric=True,
            evaluation_mode="template_only",
            embedding_threshold=None,
            embedding_model=None,
            async_execution=True,
            async_workers=None,
            preset_config=preset_config,
        )

        # Preset had abstention_enabled=True; it should be preserved
        assert config.abstention_enabled is True

    def test_explicit_no_flag_overrides_preset_abstention(self, tmp_path: Path) -> None:
        """When user passes --no-abstention, preset's True should be overridden to False."""
        preset_path = _create_preset(tmp_path, abstention_enabled=True)
        preset_config = VerificationConfig.from_preset(preset_path)

        config = build_config_from_cli_args(
            answering_model=None,
            answering_provider=None,
            answering_id="answering-1",
            parsing_model=None,
            parsing_provider=None,
            parsing_id="parsing-1",
            temperature=None,
            interface=None,
            replicate_count=None,
            abstention=False,  # Explicit --no-abstention
            sufficiency=None,
            embedding_check=None,
            deep_judgment=None,
            deep_judgment_rubric_mode="disabled",
            deep_judgment_rubric_excerpts=True,
            deep_judgment_rubric_max_excerpts=3,
            deep_judgment_rubric_fuzzy_threshold=0.8,
            deep_judgment_rubric_retry_attempts=1,
            deep_judgment_rubric_search=False,
            deep_judgment_rubric_search_tool="tavily",
            deep_judgment_rubric_config=None,
            use_full_trace_for_template=False,
            use_full_trace_for_rubric=True,
            evaluation_mode="template_only",
            embedding_threshold=None,
            embedding_model=None,
            async_execution=True,
            async_workers=None,
            preset_config=preset_config,
        )

        assert config.abstention_enabled is False

    def test_no_flag_preserves_preset_for_all_boolean_flags(self, tmp_path: Path) -> None:
        """All four boolean flags should preserve preset values when not explicitly set."""
        preset_path = _create_preset(
            tmp_path,
            abstention_enabled=True,
            sufficiency_enabled=True,
            embedding_check_enabled=True,
            deep_judgment_mode="full",
        )
        preset_config = VerificationConfig.from_preset(preset_path)

        config = build_config_from_cli_args(
            answering_model=None,
            answering_provider=None,
            answering_id="answering-1",
            parsing_model=None,
            parsing_provider=None,
            parsing_id="parsing-1",
            temperature=None,
            interface=None,
            replicate_count=None,
            abstention=None,  # Not set by user
            sufficiency=None,
            embedding_check=None,
            deep_judgment=None,
            deep_judgment_rubric_mode="disabled",
            deep_judgment_rubric_excerpts=True,
            deep_judgment_rubric_max_excerpts=3,
            deep_judgment_rubric_fuzzy_threshold=0.8,
            deep_judgment_rubric_retry_attempts=1,
            deep_judgment_rubric_search=False,
            deep_judgment_rubric_search_tool="tavily",
            deep_judgment_rubric_config=None,
            use_full_trace_for_template=False,
            use_full_trace_for_rubric=True,
            evaluation_mode="template_only",
            embedding_threshold=None,
            embedding_model=None,
            async_execution=True,
            async_workers=None,
            preset_config=preset_config,
        )

        assert config.abstention_enabled is True
        assert config.sufficiency_enabled is True
        assert config.embedding_check_enabled is True
        assert config.deep_judgment_mode == "full"

    def test_cli_accepts_no_abstention_flag(self) -> None:
        """The CLI should accept --no-abstention as a valid flag."""
        result = runner.invoke(app, ["verify", "--help"])
        assert result.exit_code == 0
        assert "--no-abstention" in result.stdout

    def test_cli_accepts_no_sufficiency_flag(self) -> None:
        """The CLI should accept --no-sufficiency as a valid flag."""
        result = runner.invoke(app, ["verify", "--help"])
        assert result.exit_code == 0
        assert "--no-sufficiency" in result.stdout

    def test_cli_accepts_no_embedding_check_flag(self) -> None:
        """The CLI should accept --no-embedding-check as a valid flag."""
        result = runner.invoke(app, ["verify", "--help"])
        assert result.exit_code == 0
        # Rich console may truncate long flag names; check for the prefix
        assert "--no-embedding-che" in result.stdout

    def test_cli_accepts_no_deep_judgment_flag(self) -> None:
        """The CLI should accept --no-deep-judgment as a valid flag."""
        result = runner.invoke(app, ["verify", "--help"])
        assert result.exit_code == 0
        assert "--no-deep-judgment" in result.stdout


# =============================================================================
# Issue 037: Embedding defaults should defer to VerificationConfig
# =============================================================================


@pytest.mark.unit
@pytest.mark.cli
class TestEmbeddingDefaults:
    """Issue 037: CLI embedding defaults should not override env vars."""

    def test_env_var_embedding_threshold_not_masked_by_cli(self, monkeypatch: Any) -> None:
        """Env var EMBEDDING_CHECK_THRESHOLD should apply when CLI doesn't specify threshold.

        Currently, CLI always passes DEFAULT_EMBEDDING_THRESHOLD (0.85) to from_overrides(),
        which prevents the env var from taking effect.
        """
        monkeypatch.setenv("EMBEDDING_CHECK_THRESHOLD", "0.72")

        config = build_config_from_cli_args(
            answering_model="test-model",
            answering_provider="anthropic",
            answering_id="answering-1",
            parsing_model="test-model",
            parsing_provider="anthropic",
            parsing_id="parsing-1",
            temperature=None,
            interface="langchain",
            replicate_count=None,
            abstention=None,
            sufficiency=None,
            embedding_check=None,
            deep_judgment=None,
            deep_judgment_rubric_mode="disabled",
            deep_judgment_rubric_excerpts=True,
            deep_judgment_rubric_max_excerpts=3,
            deep_judgment_rubric_fuzzy_threshold=0.8,
            deep_judgment_rubric_retry_attempts=1,
            deep_judgment_rubric_search=False,
            deep_judgment_rubric_search_tool="tavily",
            deep_judgment_rubric_config=None,
            use_full_trace_for_template=False,
            use_full_trace_for_rubric=True,
            evaluation_mode="template_only",
            embedding_threshold=None,  # CLI didn't specify; env var should apply
            embedding_model=None,
            async_execution=True,
            async_workers=None,
        )

        assert config.embedding_check_threshold == pytest.approx(0.72)

    def test_env_var_embedding_model_not_masked_by_cli(self, monkeypatch: Any) -> None:
        """Env var EMBEDDING_CHECK_MODEL should apply when CLI doesn't specify model."""
        monkeypatch.setenv("EMBEDDING_CHECK_MODEL", "custom-embed-model")

        config = build_config_from_cli_args(
            answering_model="test-model",
            answering_provider="anthropic",
            answering_id="answering-1",
            parsing_model="test-model",
            parsing_provider="anthropic",
            parsing_id="parsing-1",
            temperature=None,
            interface="langchain",
            replicate_count=None,
            abstention=None,
            sufficiency=None,
            embedding_check=None,
            deep_judgment=None,
            deep_judgment_rubric_mode="disabled",
            deep_judgment_rubric_excerpts=True,
            deep_judgment_rubric_max_excerpts=3,
            deep_judgment_rubric_fuzzy_threshold=0.8,
            deep_judgment_rubric_retry_attempts=1,
            deep_judgment_rubric_search=False,
            deep_judgment_rubric_search_tool="tavily",
            deep_judgment_rubric_config=None,
            use_full_trace_for_template=False,
            use_full_trace_for_rubric=True,
            evaluation_mode="template_only",
            embedding_threshold=None,
            embedding_model=None,  # CLI didn't specify; env var should apply
            async_execution=True,
            async_workers=None,
        )

        assert config.embedding_check_model == "custom-embed-model"

    def test_explicit_cli_threshold_overrides_env_var(self, monkeypatch: Any) -> None:
        """When user explicitly passes --embedding-threshold, it should override the env var."""
        monkeypatch.setenv("EMBEDDING_CHECK_THRESHOLD", "0.72")

        config = build_config_from_cli_args(
            answering_model="test-model",
            answering_provider="anthropic",
            answering_id="answering-1",
            parsing_model="test-model",
            parsing_provider="anthropic",
            parsing_id="parsing-1",
            temperature=None,
            interface="langchain",
            replicate_count=None,
            abstention=None,
            sufficiency=None,
            embedding_check=None,
            deep_judgment=None,
            deep_judgment_rubric_mode="disabled",
            deep_judgment_rubric_excerpts=True,
            deep_judgment_rubric_max_excerpts=3,
            deep_judgment_rubric_fuzzy_threshold=0.8,
            deep_judgment_rubric_retry_attempts=1,
            deep_judgment_rubric_search=False,
            deep_judgment_rubric_search_tool="tavily",
            deep_judgment_rubric_config=None,
            use_full_trace_for_template=False,
            use_full_trace_for_rubric=True,
            evaluation_mode="template_only",
            embedding_threshold=0.90,  # Explicit CLI override
            embedding_model=None,
            async_execution=True,
            async_workers=None,
        )

        assert config.embedding_check_threshold == pytest.approx(0.90)

    def test_no_env_var_uses_field_default(self) -> None:
        """When neither CLI nor env var sets threshold, field default should apply."""
        config = build_config_from_cli_args(
            answering_model="test-model",
            answering_provider="anthropic",
            answering_id="answering-1",
            parsing_model="test-model",
            parsing_provider="anthropic",
            parsing_id="parsing-1",
            temperature=None,
            interface="langchain",
            replicate_count=None,
            abstention=None,
            sufficiency=None,
            embedding_check=None,
            deep_judgment=None,
            deep_judgment_rubric_mode="disabled",
            deep_judgment_rubric_excerpts=True,
            deep_judgment_rubric_max_excerpts=3,
            deep_judgment_rubric_fuzzy_threshold=0.8,
            deep_judgment_rubric_retry_attempts=1,
            deep_judgment_rubric_search=False,
            deep_judgment_rubric_search_tool="tavily",
            deep_judgment_rubric_config=None,
            use_full_trace_for_template=False,
            use_full_trace_for_rubric=True,
            evaluation_mode="template_only",
            embedding_threshold=None,
            embedding_model=None,
            async_execution=True,
            async_workers=None,
        )

        assert config.embedding_check_threshold == pytest.approx(DEFAULT_EMBEDDING_THRESHOLD)
        assert config.embedding_check_model == DEFAULT_EMBEDDING_MODEL
