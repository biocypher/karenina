"""Tests that VerificationContext and VerificationConfig carry replay fields."""

from __future__ import annotations

import pytest

from karenina.benchmark.verification.stages.core.base import ArtifactKeys
from karenina.replay import ReplayStore
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.config import VerificationConfig

_TEST_MODEL = ModelConfig(id="test", model_name="test", model_provider="test")


@pytest.mark.unit
class TestVerificationConfigReplayField:
    def test_default_is_none(self):
        config = VerificationConfig(
            parsing_models=[_TEST_MODEL],
            parsing_only=True,
        )
        assert config.replay_store is None
        assert config.replay_parse_on_hydration_mismatch == "fall_through"

    def test_accepts_replay_store_instance(self):
        store = ReplayStore()
        config = VerificationConfig(
            parsing_models=[_TEST_MODEL],
            parsing_only=True,
            replay_store=store,
        )
        assert config.replay_store is store

    def test_replay_store_excluded_from_serialization(self):
        store = ReplayStore()
        config = VerificationConfig(
            parsing_models=[_TEST_MODEL],
            parsing_only=True,
            replay_store=store,
        )
        dumped = config.model_dump(mode="python")
        assert "replay_store" not in dumped


@pytest.mark.unit
class TestVerificationContextReplayFields:
    def test_default_field_values(self):
        from dataclasses import fields as dc_fields

        from karenina.benchmark.verification.stages.core.base import VerificationContext

        field_names = {f.name for f in dc_fields(VerificationContext)}
        assert "replay_store" in field_names
        assert "replay_parse_on_hydration_mismatch" in field_names
        assert "scenario_node_visit_index" in field_names

    def test_artifact_key_replay_entry_exists(self):
        assert hasattr(ArtifactKeys, "REPLAY_ENTRY")
        assert ArtifactKeys.REPLAY_ENTRY == "replay_entry"
