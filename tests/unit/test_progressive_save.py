"""Tests for progressive save functionality, specifically TaskIdentifier."""

from unittest.mock import MagicMock

from karenina.cli.progressive_save import TaskIdentifier
from karenina.schemas import VerificationConfig
from karenina.schemas.workflow import VerificationResult
from karenina.schemas.workflow.models import ModelConfig


class TestTaskIdentifierFromResult:
    """Tests for TaskIdentifier.from_result() method."""

    def _create_mock_result(
        self,
        question_id: str = "q1",
        answering_model: str = "anthropic/claude-haiku-4-5",
        parsing_model: str = "anthropic/claude-haiku-4-5",
        answering_replicate: int = 1,
        mcp_servers: list[str] | None = None,
    ) -> VerificationResult:
        """Create a mock VerificationResult for testing."""
        # Create nested mocks for metadata
        metadata = MagicMock()
        metadata.question_id = question_id
        metadata.answering_model = answering_model
        metadata.parsing_model = parsing_model
        metadata.answering_replicate = answering_replicate

        result = MagicMock()
        result.metadata = metadata
        result.answering_mcp_servers = mcp_servers
        return result

    def _create_config(
        self,
        answering_models: list[dict],
        parsing_models: list[dict] | None = None,
    ) -> VerificationConfig:
        """Create a VerificationConfig for testing."""
        if parsing_models is None:
            parsing_models = [
                {
                    "id": "parsing-1",
                    "model_provider": "anthropic",
                    "model_name": "claude-haiku-4-5",
                    "interface": "langchain",
                }
            ]

        return VerificationConfig(
            answering_models=[ModelConfig(**m) for m in answering_models],
            parsing_models=[ModelConfig(**m) for m in parsing_models],
        )

    def test_single_config_no_mcp(self):
        """Test with a single answering model without MCP."""
        config = self._create_config(
            answering_models=[
                {
                    "id": "answering-1",
                    "model_provider": "anthropic",
                    "model_name": "claude-haiku-4-5",
                    "interface": "langchain",
                }
            ]
        )

        result = self._create_mock_result(
            question_id="q1",
            answering_model="anthropic/claude-haiku-4-5",
            mcp_servers=None,
        )

        task_id = TaskIdentifier.from_result(result, config)

        assert task_id.question_id == "q1"
        assert task_id.answering_model_id == "answering-1"
        assert task_id.mcp_hash == ""
        assert task_id.parsing_model_id == "parsing-1"

    def test_single_config_with_mcp(self):
        """Test with a single answering model with MCP."""
        config = self._create_config(
            answering_models=[
                {
                    "id": "answering-1",
                    "model_provider": "anthropic",
                    "model_name": "claude-haiku-4-5",
                    "interface": "langchain",
                    "mcp_urls_dict": {"otar-local": "http://localhost:8080"},
                }
            ]
        )

        result = self._create_mock_result(
            question_id="q1",
            answering_model="anthropic/claude-haiku-4-5",
            mcp_servers=["otar-local"],
        )

        task_id = TaskIdentifier.from_result(result, config)

        assert task_id.question_id == "q1"
        assert task_id.answering_model_id == "answering-1"
        assert task_id.mcp_hash != ""  # Should have MCP hash
        assert task_id.parsing_model_id == "parsing-1"

    def test_multiple_mcp_configs_same_model_first_config(self):
        """Test with multiple MCP configs - result matches first config."""
        config = self._create_config(
            answering_models=[
                {
                    "id": "answering-1",
                    "model_provider": "anthropic",
                    "model_name": "claude-haiku-4-5",
                    "interface": "langchain",
                    "mcp_urls_dict": {"otar-local": "http://localhost:8080"},
                },
                {
                    "id": "answering-2",
                    "model_provider": "anthropic",
                    "model_name": "claude-haiku-4-5",
                    "interface": "langchain",
                    "mcp_urls_dict": {"otar-official": "http://official:8080"},
                },
            ]
        )

        # Result from first MCP config
        result = self._create_mock_result(
            question_id="q1",
            answering_model="anthropic/claude-haiku-4-5",
            mcp_servers=["otar-local"],
        )

        task_id = TaskIdentifier.from_result(result, config)

        assert task_id.answering_model_id == "answering-1"
        # Should have hash for otar-local config
        expected_hash = TaskIdentifier.compute_mcp_hash(config.answering_models[0])
        assert task_id.mcp_hash == expected_hash

    def test_multiple_mcp_configs_same_model_second_config(self):
        """Test with multiple MCP configs - result matches second config.

        This is the critical test for the bug fix. Previously, this would
        incorrectly match the first config because only model string was checked.
        """
        config = self._create_config(
            answering_models=[
                {
                    "id": "answering-1",
                    "model_provider": "anthropic",
                    "model_name": "claude-haiku-4-5",
                    "interface": "langchain",
                    "mcp_urls_dict": {"otar-local": "http://localhost:8080"},
                },
                {
                    "id": "answering-2",
                    "model_provider": "anthropic",
                    "model_name": "claude-haiku-4-5",
                    "interface": "langchain",
                    "mcp_urls_dict": {"otar-official": "http://official:8080"},
                },
            ]
        )

        # Result from SECOND MCP config
        result = self._create_mock_result(
            question_id="q1",
            answering_model="anthropic/claude-haiku-4-5",
            mcp_servers=["otar-official"],  # <-- Different MCP server
        )

        task_id = TaskIdentifier.from_result(result, config)

        # Should match second config, not first!
        assert task_id.answering_model_id == "answering-2"
        # Should have hash for otar-official config
        expected_hash = TaskIdentifier.compute_mcp_hash(config.answering_models[1])
        assert task_id.mcp_hash == expected_hash

    def test_multiple_mcp_configs_no_match_falls_back(self):
        """Test fallback when no MCP config matches."""
        config = self._create_config(
            answering_models=[
                {
                    "id": "answering-1",
                    "model_provider": "anthropic",
                    "model_name": "claude-haiku-4-5",
                    "interface": "langchain",
                    "mcp_urls_dict": {"otar-local": "http://localhost:8080"},
                },
            ]
        )

        # Result with different MCP server that doesn't match
        result = self._create_mock_result(
            question_id="q1",
            answering_model="anthropic/claude-haiku-4-5",
            mcp_servers=["unknown-server"],
        )

        task_id = TaskIdentifier.from_result(result, config)

        # Falls back to result value
        assert task_id.answering_model_id == "anthropic/claude-haiku-4-5"
        assert task_id.mcp_hash == ""

    def test_task_id_roundtrip_with_mcp(self):
        """Test that task IDs can be generated and parsed back correctly."""
        config = self._create_config(
            answering_models=[
                {
                    "id": "answering-1",
                    "model_provider": "anthropic",
                    "model_name": "claude-haiku-4-5",
                    "interface": "langchain",
                    "mcp_urls_dict": {"otar-official": "http://official:8080"},
                },
            ]
        )

        result = self._create_mock_result(
            question_id="urn:uuid:abc123",
            answering_model="anthropic/claude-haiku-4-5",
            answering_replicate=2,
            mcp_servers=["otar-official"],
        )

        task_id = TaskIdentifier.from_result(result, config)
        key = task_id.to_key()

        # Parse it back
        parsed = TaskIdentifier.from_key(key)

        assert parsed.question_id == "urn:uuid:abc123"
        assert parsed.answering_model_id == "answering-1"
        assert parsed.mcp_hash == task_id.mcp_hash
        assert parsed.parsing_model_id == "parsing-1"
        assert parsed.replicate == 2
