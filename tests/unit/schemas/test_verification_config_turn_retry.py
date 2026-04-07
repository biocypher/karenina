"""Tests for removal of max_scenario_turn_retries config field.

The field was replaced by retry_policy on VerificationConfig. Scenario turn
retries now derive from retry_policy.derive_sdk_max_retries().
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy


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
class TestMaxScenarioTurnRetriesRemoved:
    def test_field_no_longer_exists(self) -> None:
        """max_scenario_turn_retries is not accepted by VerificationConfig."""
        with pytest.raises(ValidationError):
            _make_config(max_scenario_turn_retries=2)

    def test_retry_policy_present(self) -> None:
        """VerificationConfig has retry_policy field instead."""
        config = _make_config()
        assert isinstance(config.retry_policy, RetryPolicy)

    def test_custom_retry_policy(self) -> None:
        """Custom RetryPolicy is accepted."""
        policy = RetryPolicy(connection=CategoryRetryConfig(max_attempts=10))
        config = _make_config(retry_policy=policy)
        assert config.retry_policy.connection.max_attempts == 10
