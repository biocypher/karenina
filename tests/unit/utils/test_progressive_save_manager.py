"""Unit tests for ProgressiveSaveManager and related functions.

Tests cover:
- ProgressiveSaveManager initialization and file creation
- add_result increments state and saves files
- get_result_set returns correct format
- load_for_resume restores state (roundtrip)
- Format version mismatch on resume raises clear error
- inspect_state_file returns correct metadata
- is_compatible validates config/benchmark match
- finalize removes state files
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result import VerificationResult
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
)
from karenina.utils.progressive_save import (
    ProgressiveJobStatus,
    ProgressiveSaveManager,
    TaskIdentifier,
    inspect_state_file,
)


def _make_config(**kwargs) -> VerificationConfig:
    """Create a minimal VerificationConfig for testing."""
    defaults = {
        "answering_models": [
            ModelConfig(
                id="ans-1",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
            )
        ],
        "parsing_models": [
            ModelConfig(
                id="par-1",
                model_name="gpt-4",
                model_provider="openai",
                interface="langchain",
            )
        ],
    }
    defaults.update(kwargs)
    return VerificationConfig(**defaults)


def _make_result(question_id: str = "q1", replicate: int | None = 1) -> VerificationResult:
    """Create a minimal VerificationResult for testing."""
    answering = ModelIdentity(interface="langchain", model_name="gpt-4", tools=[])
    parsing = ModelIdentity(interface="langchain", model_name="gpt-4", tools=[])
    timestamp = "2026-01-01T00:00:00Z"

    metadata = VerificationResultMetadata(
        question_id=question_id,
        template_id="tmpl_abc123",
        completed_without_errors=True,
        question_text="What is 2+2?",
        answering=answering,
        parsing=parsing,
        execution_time=1.5,
        timestamp=timestamp,
        result_id=VerificationResultMetadata.compute_result_id(question_id, answering, parsing, timestamp, replicate),
        replicate=replicate,
    )
    return VerificationResult(metadata=metadata)


def _task_key_for(question_id: str = "q1", replicate: int | None = 1) -> str:
    """Compute the task key that add_result would generate for _make_result()."""
    result = _make_result(question_id=question_id, replicate=replicate)
    return TaskIdentifier.from_result(result).to_key()


@pytest.mark.unit
class TestProgressiveSaveManagerInit:
    """Tests for ProgressiveSaveManager initialization."""

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_init_sets_paths(self, _mock, tmp_path: Path) -> None:
        """Initializer derives .tmp and .state paths from output_path."""
        output = tmp_path / "results.json"
        config = _make_config()
        mgr = ProgressiveSaveManager(output, config, "/path/to/benchmark.jsonld")

        assert mgr.output_path == output
        assert mgr.tmp_path == output.with_suffix(".json.tmp")
        assert mgr.state_path == output.with_suffix(".json.state")
        assert mgr.completed_count == 0
        assert mgr.total_tasks == 0

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_initialize_creates_files(self, _mock, tmp_path: Path) -> None:
        """initialize() creates .state and .tmp files."""
        output = tmp_path / "results.json"
        config = _make_config()
        mgr = ProgressiveSaveManager(output, config, "/path/to/benchmark.jsonld")
        mgr.initialize(["task1", "task2", "task3"])

        assert mgr.state_path.exists()
        assert mgr.tmp_path.exists()
        assert mgr.total_tasks == 3
        assert mgr.completed_count == 0

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_initialize_state_file_content(self, _mock, tmp_path: Path) -> None:
        """State file contains expected structure."""
        output = tmp_path / "results.json"
        config = _make_config()
        mgr = ProgressiveSaveManager(output, config, "/bench.jsonld")
        mgr.initialize(["t1", "t2"])

        state = json.loads(mgr.state_path.read_text())
        assert state["format_version"] == "1.0"
        assert state["benchmark_path"] == "/bench.jsonld"
        assert state["task_manifest"] == ["t1", "t2"]
        assert state["completed_task_ids"] == []
        assert state["total_tasks"] == 2
        assert state["completed_count"] == 0


@pytest.mark.unit
class TestProgressiveSaveManagerAddResult:
    """Tests for add_result() method."""

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_add_result_increments_count(self, _mock, tmp_path: Path) -> None:
        """add_result increases completed_count."""
        output = tmp_path / "results.json"
        mgr = ProgressiveSaveManager(output, _make_config(), "/bench.jsonld")
        mgr.initialize(["task1"])

        result = _make_result()
        mgr.add_result(result)

        assert mgr.completed_count == 1

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_add_result_saves_both_files(self, _mock, tmp_path: Path) -> None:
        """add_result updates both .state and .tmp files."""
        output = tmp_path / "results.json"
        mgr = ProgressiveSaveManager(output, _make_config(), "/bench.jsonld")
        mgr.initialize(["task1"])

        result = _make_result()
        mgr.add_result(result)

        # Files should have been updated with the new result
        state_content = json.loads(mgr.state_path.read_text())
        assert state_content["completed_count"] == 1
        assert len(state_content["completed_task_ids"]) == 1

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_get_result_set(self, _mock, tmp_path: Path) -> None:
        """get_result_set returns VerificationResultSet with all added results."""
        output = tmp_path / "results.json"
        mgr = ProgressiveSaveManager(output, _make_config(), "/bench.jsonld")
        mgr.initialize(["t1", "t2"])

        r1 = _make_result(question_id="q1")
        r2 = _make_result(question_id="q2")
        mgr.add_result(r1)
        mgr.add_result(r2)

        result_set = mgr.get_result_set()
        assert len(result_set.results) == 2

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_get_pending_task_ids(self, _mock, tmp_path: Path) -> None:
        """get_pending_task_ids returns tasks not yet completed."""
        output = tmp_path / "results.json"
        mgr = ProgressiveSaveManager(output, _make_config(), "/bench.jsonld")
        mgr.initialize(["task_a", "task_b", "task_c"])

        # Mark one as done by adding it to completed set directly
        mgr._completed_task_ids.add("task_a")

        pending = mgr.get_pending_task_ids()
        assert pending == {"task_b", "task_c"}


@pytest.mark.unit
class TestProgressiveSaveManagerResume:
    """Tests for load_for_resume() classmethod."""

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_roundtrip_save_and_resume(self, _mock, tmp_path: Path) -> None:
        """Saved state can be loaded back and continue from where it left off."""
        output = tmp_path / "results.json"
        config = _make_config()

        # Phase 1: Create and add results
        mgr = ProgressiveSaveManager(output, config, "/bench.jsonld")
        mgr.initialize(["t1", "t2", "t3"])

        r1 = _make_result(question_id="q1")
        mgr.add_result(r1)

        # Phase 2: Resume
        resumed = ProgressiveSaveManager.load_for_resume(mgr.state_path)

        assert resumed.completed_count == 1
        assert resumed.total_tasks == 3
        assert resumed.benchmark_path == "/bench.jsonld"
        assert len(resumed._results) == 1
        assert resumed._results[0].metadata.question_id == "q1"

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_resume_missing_state_file_raises(self, _mock, tmp_path: Path) -> None:
        """Missing state file raises FileNotFoundError."""
        fake_state = tmp_path / "nonexistent.json.state"
        with pytest.raises(FileNotFoundError, match="State file not found"):
            ProgressiveSaveManager.load_for_resume(fake_state)

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_resume_missing_tmp_file_raises(self, _mock, tmp_path: Path) -> None:
        """State file exists but .tmp missing raises FileNotFoundError."""
        output = tmp_path / "results.json"
        config = _make_config()
        mgr = ProgressiveSaveManager(output, config, "/bench.jsonld")
        mgr.initialize(["t1"])

        # Delete the tmp file
        mgr.tmp_path.unlink()

        with pytest.raises(FileNotFoundError, match="Results file not found"):
            ProgressiveSaveManager.load_for_resume(mgr.state_path)

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_resume_version_mismatch_raises(self, _mock, tmp_path: Path) -> None:
        """Incompatible format version raises ValueError."""
        output = tmp_path / "results.json"
        config = _make_config()
        mgr = ProgressiveSaveManager(output, config, "/bench.jsonld")
        mgr.initialize(["t1"])

        # Tamper with the state file version
        state = json.loads(mgr.state_path.read_text())
        state["format_version"] = "99.0"
        mgr.state_path.write_text(json.dumps(state))

        with pytest.raises(ValueError, match="Incompatible state format version"):
            ProgressiveSaveManager.load_for_resume(mgr.state_path)


@pytest.mark.unit
class TestProgressiveSaveManagerCompatibility:
    """Tests for is_compatible() method."""

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_compatible_same_config(self, _mock, tmp_path: Path) -> None:
        """Same config and benchmark is compatible."""
        output = tmp_path / "results.json"
        config = _make_config()
        mgr = ProgressiveSaveManager(output, config, "/bench.jsonld")
        mgr.initialize(["t1"])

        is_compat, reason = mgr.is_compatible(config, "/bench.jsonld")
        assert is_compat is True
        assert reason == ""

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_incompatible_benchmark_path(self, _mock, tmp_path: Path) -> None:
        """Different benchmark path is incompatible."""
        output = tmp_path / "results.json"
        config = _make_config()
        mgr = ProgressiveSaveManager(output, config, "/bench.jsonld")
        mgr.initialize(["t1"])

        is_compat, reason = mgr.is_compatible(config, "/different.jsonld")
        assert is_compat is False
        assert "Benchmark path changed" in reason


@pytest.mark.unit
class TestProgressiveSaveManagerFinalize:
    """Tests for finalize() method."""

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_finalize_removes_files(self, _mock, tmp_path: Path) -> None:
        """finalize() deletes .state and .tmp files."""
        output = tmp_path / "results.json"
        mgr = ProgressiveSaveManager(output, _make_config(), "/bench.jsonld")
        mgr.initialize(["t1"])

        assert mgr.state_path.exists()
        assert mgr.tmp_path.exists()

        mgr.finalize()

        assert not mgr.state_path.exists()
        assert not mgr.tmp_path.exists()


@pytest.mark.unit
class TestInspectStateFile:
    """Tests for inspect_state_file() standalone function."""

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_inspect_returns_status(self, _mock, tmp_path: Path) -> None:
        """inspect_state_file returns ProgressiveJobStatus with correct metadata."""
        output = tmp_path / "results.json"
        config = _make_config()
        # Use proper task keys that match what add_result generates
        task_keys = [_task_key_for("q1"), _task_key_for("q2"), _task_key_for("q3")]
        mgr = ProgressiveSaveManager(output, config, "/bench.jsonld")
        mgr.initialize(task_keys)

        # Add one result to simulate partial progress
        r1 = _make_result(question_id="q1")
        mgr.add_result(r1)

        status = inspect_state_file(mgr.state_path)

        assert isinstance(status, ProgressiveJobStatus)
        assert status.total_tasks == 3
        assert status.completed_count == 1
        assert status.pending_count == 2
        assert status.benchmark_path == "/bench.jsonld"
        assert status.tmp_file_exists is True
        assert status.tmp_file_size is not None
        assert status.tmp_file_size > 0

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_inspect_missing_file_raises(self, _mock, tmp_path: Path) -> None:
        """Missing state file raises FileNotFoundError."""
        fake = tmp_path / "no.json.state"
        with pytest.raises(FileNotFoundError, match="State file not found"):
            inspect_state_file(fake)

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_inspect_invalid_version_raises(self, _mock, tmp_path: Path) -> None:
        """Invalid format version raises ValueError."""
        state_file = tmp_path / "bad.json.state"
        state_file.write_text(json.dumps({"format_version": "999.0"}))

        with pytest.raises(ValueError, match="Incompatible state format version"):
            inspect_state_file(state_file)

    @patch("karenina.schemas.verification.config.os.getenv", return_value=None)
    def test_inspect_progress_percent(self, _mock, tmp_path: Path) -> None:
        """progress_percent property computes correctly."""
        output = tmp_path / "results.json"
        task_keys = [_task_key_for("q1"), _task_key_for("q2"), _task_key_for("q3"), _task_key_for("q4")]
        mgr = ProgressiveSaveManager(output, _make_config(), "/bench.jsonld")
        mgr.initialize(task_keys)

        r1 = _make_result(question_id="q1")
        mgr.add_result(r1)

        status = inspect_state_file(mgr.state_path)
        assert status.progress_percent == 25.0
