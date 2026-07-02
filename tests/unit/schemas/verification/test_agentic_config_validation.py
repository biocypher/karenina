"""Tests for agentic parsing validation in VerificationConfig."""

import pytest

from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig


def _sdk_model():
    return ModelConfig(
        id="test",
        model_name="test",
        interface="claude_agent_sdk",
    )


@pytest.mark.unit
class TestAgenticParsingValidation:
    def test_disabled_agentic_parsing_needs_no_special_config(self):
        config = VerificationConfig(
            parsing_models=[_sdk_model()],
            parsing_only=True,
            agentic_parsing=False,
        )
        assert config.agentic_parsing is False

    def test_agentic_parsing_accepted_without_workspace_root(self):
        """workspace_root now lives on Benchmark, not VerificationConfig."""
        config = VerificationConfig(
            parsing_models=[_sdk_model()],
            parsing_only=True,
            agentic_parsing=True,
        )
        assert config.agentic_parsing is True

    def test_agentic_parsing_rejects_tool_loop_interface(self):
        """agentic_parsing=True should reject interfaces with agent_tier='tool_loop'."""
        with pytest.raises(ValueError, match="deep_agent"):
            VerificationConfig(
                answering_models=[
                    ModelConfig(
                        id="a",
                        model_name="m",
                        model_provider="anthropic",
                    )
                ],
                parsing_models=[
                    ModelConfig(
                        id="p",
                        model_name="m",
                        model_provider="anthropic",
                        interface="langchain",  # agent_tier="tool_loop"
                    )
                ],
                agentic_parsing=True,
            )

    def test_agentic_parsing_not_supported_with_rubric_only(self):
        with pytest.raises(ValueError, match="rubric_only"):
            VerificationConfig(
                answering_models=[_sdk_model()],
                parsing_models=[_sdk_model()],
                evaluation_mode="rubric_only",
                agentic_parsing=True,
            )

    def test_dynamic_trigger_requires_agentic_parsing(self):
        with pytest.raises(ValueError, match="agentic_parsing_trigger='dynamic' requires agentic_parsing=True"):
            VerificationConfig(
                parsing_models=[_sdk_model()],
                parsing_only=True,
                agentic_parsing=False,
                agentic_parsing_trigger="dynamic",
            )

    def test_dynamic_trigger_warns_on_trace_only(self, caplog):
        config = VerificationConfig(
            parsing_models=[_sdk_model()],
            parsing_only=True,
            agentic_parsing=True,
            agentic_parsing_trigger="dynamic",
            agentic_judge_context="trace_only",
        )
        assert config.agentic_parsing_trigger == "dynamic"
        assert "workspace_only" in caplog.text or "trace_and_workspace" in caplog.text

    def test_materialize_trace_requires_trace_in_context(self):
        """materialize_trace=True with workspace_only is rejected.

        The trace file is the materialized agent trace, so the judge context
        must include the trace. workspace_only excludes it.
        """
        with pytest.raises(ValueError, match="materialize_trace=True requires"):
            VerificationConfig(
                answering_models=[_sdk_model()],
                parsing_models=[_sdk_model()],
                evaluation_mode="template_only",
                agentic_parsing=True,
                agentic_judge_context="workspace_only",
                agentic_parsing_materialize_trace=True,
            )
