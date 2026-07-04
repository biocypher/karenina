"""Unit tests for replay JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from karenina.replay import ReplayEntry, ReplayKey, ReplayStore
from karenina.replay.exceptions import ReplayPersistenceError
from karenina.replay.persistence import dump, load


def _mixed_store() -> ReplayStore:
    store = ReplayStore(miss_policy="fall_through")
    store.register(
        ReplayKey(question_id="q1", answering_model_id="gpt-5"),
        ReplayEntry(
            raw_trace="qa answer",
            parsed_answer_fields={"value": 42},
            captured_model_id="gpt-5",
            captured_at="2026-04-08T12:00:00Z",
        ),
    )
    store.register(
        ReplayKey(
            question_id="q2",
            scenario_id="syco",
            scenario_node="setup",
            answering_model_id="gpt-5",
            visit_index=0,
        ),
        ReplayEntry(
            raw_trace="scenario turn",
            trace_messages=[{"role": "assistant", "content": "scenario turn"}],
            agent_metrics={"iterations": 1, "limit_reached": False},
        ),
    )
    return store


@pytest.mark.unit
class TestReplayPersistence:
    def test_round_trip_preserves_entries_and_policy(self, tmp_path):
        original = _mixed_store()
        path = tmp_path / "replay.json"
        dump(original, path)

        reloaded = load(path)
        assert reloaded.miss_policy == "fall_through"
        assert len(reloaded.entries) == len(original.entries)

        qa_hit = reloaded.lookup(question_id="q1", answering_model_id="gpt-5")
        assert qa_hit is not None
        assert qa_hit.raw_trace == "qa answer"
        assert qa_hit.parsed_answer_fields == {"value": 42}

        scen_hit = reloaded.lookup(
            question_id="q2",
            scenario_id="syco",
            scenario_node="setup",
            answering_model_id="gpt-5",
            visit_index=0,
        )
        assert scen_hit is not None
        assert scen_hit.trace_messages == [{"role": "assistant", "content": "scenario turn"}]

    def test_persistence_format_is_version_wrapped_object_list(self, tmp_path):
        store = _mixed_store()
        path = tmp_path / "replay.json"
        dump(store, path)

        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert data["miss_policy"] == "fall_through"
        assert isinstance(data["entries"], list)
        assert "key" in data["entries"][0]
        assert "entry" in data["entries"][0]

    def test_load_rejects_unknown_version(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"version": 99, "miss_policy": "fall_through", "entries": []}))
        with pytest.raises(ReplayPersistenceError) as exc_info:
            load(path)
        assert "version" in str(exc_info.value).lower()

    def test_load_rejects_malformed_json(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not valid json")
        with pytest.raises(ReplayPersistenceError):
            load(path)

    def test_load_rejects_schema_violation(self, tmp_path):
        path = tmp_path / "bad_schema.json"
        payload = {
            "version": 1,
            "miss_policy": "fall_through",
            "entries": [{"key": {"question_id": "q"}, "entry": {"BOGUS": "x"}}],
        }
        path.write_text(json.dumps(payload))
        with pytest.raises(ReplayPersistenceError):
            load(path)

    def test_load_miss_policy_override(self, tmp_path):
        store = _mixed_store()
        path = tmp_path / "replay.json"
        dump(store, path)

        reloaded = load(path, miss_policy="strict")
        assert reloaded.miss_policy == "strict"

    def test_load_rejects_invalid_miss_policy_string(self, tmp_path):
        """A garbage miss_policy in the file must surface as ReplayPersistenceError,
        not as an unwrapped Pydantic ValidationError."""
        path = tmp_path / "bad_policy.json"
        path.write_text(json.dumps({"version": 1, "miss_policy": "garbage", "entries": []}))
        with pytest.raises(ReplayPersistenceError):
            load(path)

    def test_atomic_save_no_partial_file_on_failure(self, tmp_path, monkeypatch):
        store = _mixed_store()
        path = tmp_path / "replay.json"

        import os as os_mod

        def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
            raise OSError("simulated rename failure")

        monkeypatch.setattr(os_mod, "replace", _boom)
        with pytest.raises(OSError):
            dump(store, path)
        assert not path.exists()
        # Temp files should have been cleaned up
        leftover = list(tmp_path.glob("*.tmp*"))
        assert leftover == [], f"temp files leaked: {leftover}"

    def test_save_and_classmethod_load_convenience(self, tmp_path):
        store = _mixed_store()
        path = tmp_path / "replay.json"
        store.save(path)
        reloaded = ReplayStore.load(path)
        assert reloaded.lookup(question_id="q1", answering_model_id="gpt-5") is not None

    def test_empty_store_round_trips(self, tmp_path):
        store = ReplayStore()
        path = tmp_path / "empty.json"
        dump(store, path)
        reloaded = load(path)
        assert reloaded.entries == []


@pytest.mark.unit
class TestPersistenceReplicateField:
    def test_round_trip_with_replicate(self, tmp_path: Path):
        store = ReplayStore()
        key = ReplayKey(
            question_id="q1",
            answering_model_id="m1",
            visit_index=0,
            replicate=2,
        )
        store.register(key, ReplayEntry(raw_trace="hello"))
        store.save(tmp_path / "store.json")
        loaded = ReplayStore.load(tmp_path / "store.json")
        assert len(loaded.entries) == 1
        loaded_key, loaded_entry = loaded.entries[0]
        assert loaded_key == key
        assert loaded_key.replicate == 2
        assert loaded_entry.raw_trace == "hello"

    def test_load_legacy_fixture_without_replicate(self):
        """A version=1 file whose keys predate the replicate field must
        load, treat every key's replicate as None, and resolve lookups
        with a non-None replicate via the all-wildcard rung.
        """
        fixture = Path(__file__).parent / "fixtures" / "legacy_replay_v1.json"
        store = ReplayStore.load(fixture)
        assert len(store.entries) == 1
        key, _ = store.entries[0]
        assert key.replicate is None

        # A request with a non-None replicate should walk to the
        # (None, None, None) rung, but this legacy key has
        # answering_model_id="gpt-5" and visit_index=None, so the matching
        # rung is (model, None, None). Lookup with model=gpt-5 and any
        # replicate must hit it.
        hit = store.lookup(
            question_id="q-legacy-1",
            answering_model_id="gpt-5",
            visit_index=None,
            replicate=7,
        )
        assert hit is not None
        assert hit.raw_trace == "legacy-trace"
