"""Unit tests for VerificationConfig.max_concurrent_requests field.

Tests cover:
- Default value is None
- Explicit integer value is accepted
- Environment variable KARENINA_MAX_CONCURRENT_LLM_REQUESTS sets the field
- Explicit value overrides environment variable
- Invalid environment variable is silently ignored (uses default)
- from_overrides() forwards the parameter
- Validation rejects values < 1
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig


def _make_parsing_only_config(**kwargs) -> VerificationConfig:
    """Create a minimal parsing-only VerificationConfig for testing."""
    defaults = {
        "parsing_models": [
            ModelConfig(
                id="gpt-4",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
                system_prompt="test",
                temperature=0.1,
            )
        ],
        "answering_models": [],
        "parsing_only": True,
    }
    defaults.update(kwargs)
    return VerificationConfig(**defaults)


def _make_base_config(**kwargs) -> VerificationConfig:
    """Create a minimal VerificationConfig with answering models for testing."""
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
    }
    defaults.update(kwargs)
    return VerificationConfig(**defaults)


# =============================================================================
# Field Default Tests
# =============================================================================


@pytest.mark.unit
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
def test_default_is_none(_mock_getenv) -> None:
    """max_concurrent_requests defaults to None (no global limit)."""
    config = _make_parsing_only_config()
    assert config.max_concurrent_requests is None


@pytest.mark.unit
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
def test_explicit_value_accepted(_mock_getenv) -> None:
    """Explicit integer value is stored correctly."""
    config = _make_parsing_only_config(max_concurrent_requests=32)
    assert config.max_concurrent_requests == 32


@pytest.mark.unit
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
def test_rejects_zero(_mock_getenv) -> None:
    """Value of 0 is rejected by ge=1 constraint."""
    with pytest.raises(ValidationError, match="max_concurrent_requests"):
        _make_parsing_only_config(max_concurrent_requests=0)


@pytest.mark.unit
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
def test_rejects_negative(_mock_getenv) -> None:
    """Negative values are rejected by ge=1 constraint."""
    with pytest.raises(ValidationError, match="max_concurrent_requests"):
        _make_parsing_only_config(max_concurrent_requests=-1)


# =============================================================================
# Environment Variable Tests
# =============================================================================


@pytest.mark.unit
def test_env_var_sets_value() -> None:
    """KARENINA_MAX_CONCURRENT_LLM_REQUESTS env var sets the field."""
    with patch.dict("os.environ", {"KARENINA_MAX_CONCURRENT_LLM_REQUESTS": "16"}):
        config = _make_parsing_only_config()
        assert config.max_concurrent_requests == 16


@pytest.mark.unit
def test_explicit_value_overrides_env_var() -> None:
    """Explicit value takes precedence over environment variable."""
    with patch.dict("os.environ", {"KARENINA_MAX_CONCURRENT_LLM_REQUESTS": "16"}):
        config = _make_parsing_only_config(max_concurrent_requests=64)
        assert config.max_concurrent_requests == 64


@pytest.mark.unit
def test_invalid_env_var_ignored() -> None:
    """Non-numeric env var is silently ignored (field stays None)."""
    with patch.dict("os.environ", {"KARENINA_MAX_CONCURRENT_LLM_REQUESTS": "not-a-number"}):
        config = _make_parsing_only_config()
        assert config.max_concurrent_requests is None


# =============================================================================
# from_overrides Tests
# =============================================================================


@pytest.mark.unit
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
def test_from_overrides_sets_value(_mock_getenv) -> None:
    """from_overrides() forwards max_concurrent_requests to config."""
    config = VerificationConfig.from_overrides(
        _make_base_config(),
        max_concurrent_requests=48,
    )
    assert config.max_concurrent_requests == 48


@pytest.mark.unit
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
def test_from_overrides_none_preserves_base(_mock_getenv) -> None:
    """from_overrides(max_concurrent_requests=None) preserves base value."""
    base = _make_base_config(max_concurrent_requests=32)
    config = VerificationConfig.from_overrides(base)
    assert config.max_concurrent_requests == 32


@pytest.mark.unit
@patch("karenina.schemas.verification.config.os.getenv", return_value=None)
def test_from_overrides_overrides_base(_mock_getenv) -> None:
    """from_overrides() explicit value overrides base config value."""
    base = _make_base_config(max_concurrent_requests=32)
    config = VerificationConfig.from_overrides(base, max_concurrent_requests=64)
    assert config.max_concurrent_requests == 64
