"""Tests for TaskEval metadata coherence fixes (issues 024, 114, 115, 117, 160, 165, 166, 168, 179)."""

import pytest

from karenina.schemas.config import ModelConfig


@pytest.mark.unit
class TestTaskEvalInterface:
    """Issue 166: taskeval interface registration."""

    def test_taskeval_interface_registered(self):
        """ModelConfig with interface='taskeval' should not raise."""
        config = ModelConfig(
            id="taskeval_user_provided",
            model_name="user-provided",
            model_provider="user-provided",
            interface="taskeval",
        )
        assert config.interface == "taskeval"

    def test_taskeval_sentinel_fields(self):
        """Sentinel model has expected field values."""
        config = ModelConfig(
            id="taskeval_user_provided",
            model_name="user-provided",
            model_provider="user-provided",
            interface="taskeval",
        )
        assert config.model_name == "user-provided"
        assert config.model_provider == "user-provided"
