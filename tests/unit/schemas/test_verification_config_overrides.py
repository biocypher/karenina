"""Unit tests for VerificationConfig.from_overrides() classmethod.

Tests cover:
- Base config with no overrides → equivalent config returned
- Scalar overrides (temperature, replicate_count, feature flags)
- None overrides don't change base values (validates cli-bugs-001 fix)
- Model config construction (answering + parsing separately)
- evaluation_mode auto-sets rubric_enabled
- Deep judgment rubric settings
- Manual traces handling
"""

from unittest.mock import patch

import pytest

from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig


def _make_base_config(**kwargs) -> VerificationConfig:
    """Create a minimal VerificationConfig for testing."""
    defaults = {
        "answering_models": [
            ModelConfig(
                id="ans-1",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                temperature=0.5,
            )
        ],
        "parsing_models": [
            ModelConfig(
                id="par-1",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                temperature=0.0,
            )
        ],
        "replicate_count": 3,
        "abstention_enabled": True,
        "embedding_check_enabled": True,
        "embedding_check_threshold": 0.9,
        "evaluation_mode": "template_and_rubric",
        "rubric_enabled": True,
    }
    defaults.update(kwargs)
    return VerificationConfig(**defaults)


@pytest.mark.unit
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
class TestFromOverridesNoBase:
    """Tests for from_overrides() without a base config.

    Note: from_overrides() without a base always needs at least parsing_model
    overrides because VerificationConfig.parsing_models is a required field
    with no default. Tests that don't need model-specific assertions use
    a base config instead.
    """

    def test_no_base_with_model_overrides(self, _mock) -> None:
        """Model name + provider + interface + id builds valid model configs."""
        config = VerificationConfig.from_overrides(
            answering_model="claude-sonnet-4-20250514",
            answering_provider="anthropic",
            answering_interface="langchain",
            answering_id="ans-1",
            parsing_model="claude-sonnet-4-20250514",
            parsing_provider="anthropic",
            parsing_interface="langchain",
            parsing_id="par-1",
        )
        assert len(config.answering_models) == 1
        model = config.answering_models[0]
        assert model.model_name == "claude-sonnet-4-20250514"
        assert model.model_provider == "anthropic"
        assert model.interface == "langchain"
        assert model.id == "ans-1"

    def test_no_base_temperature_applied_to_model(self, _mock) -> None:
        """Temperature override is applied to constructed model."""
        config = VerificationConfig.from_overrides(
            answering_model="gpt-4",
            answering_provider="openai",
            answering_id="ans-1",
            parsing_model="gpt-4",
            parsing_provider="openai",
            parsing_id="par-1",
            temperature=0.7,
        )
        assert config.answering_models[0].temperature == 0.7

    def test_no_base_default_temperature(self, _mock) -> None:
        """Without temperature override, model gets default 0.1."""
        config = VerificationConfig.from_overrides(
            answering_model="gpt-4",
            answering_provider="openai",
            answering_id="ans-1",
            parsing_model="gpt-4",
            parsing_provider="openai",
            parsing_id="par-1",
        )
        assert config.answering_models[0].temperature == 0.1

    def test_no_base_replicate_count_default(self, _mock) -> None:
        """Without base, replicate_count defaults to 1."""
        config = VerificationConfig.from_overrides(
            answering_model="gpt-4",
            answering_provider="openai",
            answering_id="a",
            parsing_model="gpt-4",
            parsing_provider="openai",
            parsing_id="p",
        )
        assert config.replicate_count == 1

    def test_no_base_replicate_count_override(self, _mock) -> None:
        """Replicate count override works without base."""
        config = VerificationConfig.from_overrides(
            answering_model="gpt-4",
            answering_provider="openai",
            answering_id="a",
            parsing_model="gpt-4",
            parsing_provider="openai",
            parsing_id="p",
            replicate_count=5,
        )
        assert config.replicate_count == 5

    def test_no_base_feature_flags(self, _mock) -> None:
        """Feature flag overrides applied correctly without base."""
        config = VerificationConfig.from_overrides(
            answering_model="gpt-4",
            answering_provider="openai",
            answering_id="a",
            parsing_model="gpt-4",
            parsing_provider="openai",
            parsing_id="p",
            abstention=True,
            sufficiency=True,
            embedding_check=True,
            deep_judgment=True,
        )
        assert config.abstention_enabled is True
        assert config.sufficiency_enabled is True
        assert config.embedding_check_enabled is True
        assert config.deep_judgment_enabled is True


@pytest.mark.unit
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
class TestFromOverridesEvaluationMode:
    """Tests for evaluation_mode and rubric_enabled interaction."""

    def test_evaluation_mode_template_and_rubric_sets_rubric(self, _mock) -> None:
        """evaluation_mode='template_and_rubric' auto-sets rubric_enabled."""
        config = VerificationConfig.from_overrides(
            _make_base_config(),
            evaluation_mode="template_and_rubric",
        )
        assert config.rubric_enabled is True

    def test_evaluation_mode_rubric_only_sets_rubric(self, _mock) -> None:
        """evaluation_mode='rubric_only' auto-sets rubric_enabled."""
        config = VerificationConfig.from_overrides(
            _make_base_config(),
            evaluation_mode="rubric_only",
        )
        assert config.rubric_enabled is True

    def test_evaluation_mode_template_only_disables_rubric(self, _mock) -> None:
        """evaluation_mode='template_only' does not set rubric_enabled."""
        config = VerificationConfig.from_overrides(
            _make_base_config(rubric_enabled=False, evaluation_mode="template_only"),
            evaluation_mode="template_only",
        )
        assert config.rubric_enabled is False


@pytest.mark.unit
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
class TestFromOverridesWithBase:
    """Tests for from_overrides() with a base config."""

    def test_base_with_no_overrides_preserves_values(self, _mock) -> None:
        """No overrides returns config equivalent to base."""
        base = _make_base_config()
        config = VerificationConfig.from_overrides(base)

        assert config.replicate_count == base.replicate_count
        assert config.abstention_enabled == base.abstention_enabled
        assert config.embedding_check_enabled == base.embedding_check_enabled
        assert config.embedding_check_threshold == base.embedding_check_threshold

    def test_none_overrides_preserve_base(self, _mock) -> None:
        """Explicitly passing None does NOT override base values.

        This is the critical test validating the cli-bugs-001 fix:
        sentinel None values must be distinguished from 'user provided a value'.
        """
        base = _make_base_config(
            replicate_count=3,
            abstention_enabled=True,
            embedding_check_threshold=0.9,
        )
        config = VerificationConfig.from_overrides(
            base,
            temperature=None,
            replicate_count=None,
            abstention=None,
            embedding_threshold=None,
        )
        assert config.replicate_count == 3
        assert config.abstention_enabled is True
        assert config.embedding_check_threshold == 0.9

    def test_temperature_override_applied_to_base_model(self, _mock) -> None:
        """Temperature override modifies the base answering model."""
        base = _make_base_config()
        assert base.answering_models[0].temperature == 0.5  # precondition

        config = VerificationConfig.from_overrides(
            base,
            answering_model=base.answering_models[0].model_name,
            answering_provider=base.answering_models[0].model_provider,
            temperature=0.9,
        )
        assert config.answering_models[0].temperature == 0.9

    def test_replicate_count_override(self, _mock) -> None:
        """Replicate count override works with base."""
        base = _make_base_config(replicate_count=3)
        config = VerificationConfig.from_overrides(base, replicate_count=10)
        assert config.replicate_count == 10

    def test_embedding_threshold_override(self, _mock) -> None:
        """Embedding threshold maps to correct field name."""
        base = _make_base_config(embedding_check_threshold=0.9)
        config = VerificationConfig.from_overrides(base, embedding_threshold=0.75)
        assert config.embedding_check_threshold == 0.75

    def test_embedding_model_override(self, _mock) -> None:
        """Embedding model maps to correct field name."""
        base = _make_base_config()
        config = VerificationConfig.from_overrides(base, embedding_model="all-mpnet-base-v2")
        assert config.embedding_check_model == "all-mpnet-base-v2"

    def test_async_settings_override(self, _mock) -> None:
        """Async execution and workers map to correct fields."""
        config = VerificationConfig.from_overrides(
            _make_base_config(),
            async_execution=False,
            async_workers=8,
        )
        assert config.async_enabled is False
        assert config.async_max_workers == 8

    def test_deep_judgment_rubric_settings(self, _mock) -> None:
        """Deep judgment rubric settings map to correct field names."""
        config = VerificationConfig.from_overrides(
            _make_base_config(),
            deep_judgment_rubric_mode="enable_all",
            deep_judgment_rubric_excerpts=False,
            deep_judgment_rubric_max_excerpts=10,
            deep_judgment_rubric_fuzzy_threshold=0.95,
            deep_judgment_rubric_retry_attempts=5,
        )
        assert config.deep_judgment_rubric_mode == "enable_all"
        assert config.deep_judgment_rubric_global_excerpts is False
        assert config.deep_judgment_rubric_max_excerpts_default == 10
        assert config.deep_judgment_rubric_fuzzy_match_threshold_default == 0.95
        assert config.deep_judgment_rubric_excerpt_retry_attempts_default == 5

    def test_trace_filtering_overrides(self, _mock) -> None:
        """Trace filtering flags are applied correctly."""
        config = VerificationConfig.from_overrides(
            _make_base_config(),
            use_full_trace_for_template=True,
            use_full_trace_for_rubric=False,
        )
        assert config.use_full_trace_for_template is True
        assert config.use_full_trace_for_rubric is False

    def test_answering_model_override_preserves_parsing(self, _mock) -> None:
        """Overriding answering model doesn't affect parsing model."""
        base = _make_base_config()
        original_parsing = base.parsing_models[0].model_name

        config = VerificationConfig.from_overrides(
            base,
            answering_model="claude-haiku-4-5",
            answering_provider="anthropic",
        )
        assert config.answering_models[0].model_name == "claude-haiku-4-5"
        assert config.parsing_models[0].model_name == original_parsing

    def test_parsing_model_override_preserves_answering(self, _mock) -> None:
        """Overriding parsing model doesn't affect answering model."""
        base = _make_base_config()
        original_answering = base.answering_models[0].model_name

        config = VerificationConfig.from_overrides(
            base,
            parsing_model="claude-haiku-4-5",
            parsing_provider="anthropic",
        )
        assert config.parsing_models[0].model_name == "claude-haiku-4-5"
        assert config.answering_models[0].model_name == original_answering


@pytest.mark.unit
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
class TestFromOverridesManualTraces:
    """Tests for manual traces handling in from_overrides()."""

    def test_manual_traces_creates_manual_interface(self, _mock) -> None:
        """Providing manual_traces without model overrides creates manual answering model."""
        base = _make_base_config()
        traces = {"q1": "some trace"}
        config = VerificationConfig.from_overrides(base, manual_traces=traces)
        assert len(config.answering_models) == 1
        assert config.answering_models[0].interface == "manual"

    def test_manual_traces_with_explicit_manual_interface(self, _mock) -> None:
        """Manual traces + explicit manual interface creates manual model."""
        base = _make_base_config()
        traces = {"q1": "trace"}
        config = VerificationConfig.from_overrides(
            base,
            answering_interface="manual",
            manual_traces=traces,
        )
        assert config.answering_models[0].interface == "manual"

    def test_parsing_model_never_uses_manual_interface(self, _mock) -> None:
        """Parsing model config never gets manual_traces."""
        base = _make_base_config()
        traces = {"q1": "trace"}
        config = VerificationConfig.from_overrides(
            base,
            manual_traces=traces,
            parsing_model="gpt-4",
            parsing_provider="openai",
        )
        assert config.parsing_models[0].interface != "manual"
