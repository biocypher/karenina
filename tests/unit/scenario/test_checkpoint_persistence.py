"""Tests for scenario checkpoint persistence."""

import pytest

from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.config import VerificationConfig

_TEST_MODEL = ModelConfig(id="test", model_name="test", model_provider="test")


@pytest.mark.unit
class TestScenarioTurnLimit:
    def test_default_scenario_turn_limit(self):
        config = VerificationConfig(parsing_models=[_TEST_MODEL], parsing_only=True)
        assert config.scenario_turn_limit == 20

    def test_custom_scenario_turn_limit(self):
        config = VerificationConfig(parsing_models=[_TEST_MODEL], parsing_only=True, scenario_turn_limit=5)
        assert config.scenario_turn_limit == 5
