"""Tests for max_scenario_turn_retries config field."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig


def _make_config(**overrides) -> VerificationConfig:
    """Create a minimal VerificationConfig."""
    defaults = {
        "answering_models": [ModelConfig(id="m", model_name="m", model_provider="openai")],
        "parsing_models": [ModelConfig(id="m", model_name="m", model_provider="openai")],
    }
    defaults.update(overrides)
    with patch("karenina.schemas.verification.config.os.getenv", return_value=None):
        return VerificationConfig(**defaults)


@pytest.mark.unit
class TestMaxScenarioTurnRetries:
    def test_default_is_2(self) -> None:
        config = _make_config()
        assert config.max_scenario_turn_retries == 2

    def test_minimum_is_1(self) -> None:
        with pytest.raises(ValidationError):
            _make_config(max_scenario_turn_retries=0)

    def test_custom_value(self) -> None:
        config = _make_config(max_scenario_turn_retries=5)
        assert config.max_scenario_turn_retries == 5
