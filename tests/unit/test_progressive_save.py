"""Tests for progressive save functionality, specifically TaskIdentifier."""

from karenina.cli.progressive_save import TaskIdentifier
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.workflow.models import ModelConfig


class TestTaskIdentifierFromResult:
    """Tests for TaskIdentifier.from_result() method."""

    def _make_identity(
        self,
        interface: str = "langchain",
        model_name: str = "claude-haiku-4-5",
        tools: list[str] | None = None,
    ) -> ModelIdentity:
        return ModelIdentity(interface=interface, model_name=model_name, tools=tools or [])

    def _create_mock_result(
        self,
        question_id: str = "q1",
        answering_interface: str = "langchain",
        answering_model_name: str = "claude-haiku-4-5",
        answering_tools: list[str] | None = None,
        parsing_interface: str = "langchain",
        parsing_model_name: str = "claude-haiku-4-5",
        parsing_tools: list[str] | None = None,
        replicate: int = 1,
    ):
        """Create a mock-like object with proper ModelIdentity metadata."""
        answering = self._make_identity(answering_interface, answering_model_name, answering_tools)
        parsing = self._make_identity(parsing_interface, parsing_model_name, parsing_tools)

        class MockMetadata:
            pass

        metadata = MockMetadata()
        metadata.question_id = question_id
        metadata.answering = answering
        metadata.parsing = parsing
        metadata.replicate = replicate

        class MockResult:
            pass

        result = MockResult()
        result.metadata = metadata
        return result

    def test_single_config_no_mcp(self):
        """Test with a single answering model without MCP."""
        result = self._create_mock_result(question_id="q1")
        task_id = TaskIdentifier.from_result(result)

        assert task_id.question_id == "q1"
        assert task_id.answering_canonical_key == "langchain:claude-haiku-4-5:"
        assert task_id.parsing_canonical_key == "langchain:claude-haiku-4-5:"

    def test_single_config_with_mcp(self):
        """Test with a single answering model with MCP tools."""
        result = self._create_mock_result(
            question_id="q1",
            answering_tools=["otar-local"],
        )
        task_id = TaskIdentifier.from_result(result)

        assert task_id.question_id == "q1"
        assert task_id.answering_canonical_key == "langchain:claude-haiku-4-5:otar-local"

    def test_different_tools_produce_different_keys(self):
        """Test that different MCP tool sets produce different canonical keys."""
        result1 = self._create_mock_result(answering_tools=["otar-local"])
        result2 = self._create_mock_result(answering_tools=["otar-official"])

        task_id1 = TaskIdentifier.from_result(result1)
        task_id2 = TaskIdentifier.from_result(result2)

        assert task_id1.answering_canonical_key != task_id2.answering_canonical_key
        assert task_id1.to_key() != task_id2.to_key()

    def test_different_interfaces_produce_different_keys(self):
        """Test that different interfaces produce different canonical keys."""
        result1 = self._create_mock_result(answering_interface="langchain")
        result2 = self._create_mock_result(answering_interface="claude_agent_sdk")

        task_id1 = TaskIdentifier.from_result(result1)
        task_id2 = TaskIdentifier.from_result(result2)

        assert task_id1.answering_canonical_key != task_id2.answering_canonical_key

    def test_task_id_roundtrip(self):
        """Test that task IDs can be generated and parsed back correctly."""
        result = self._create_mock_result(
            question_id="urn:uuid:abc123",
            answering_tools=["otar-official"],
            replicate=2,
        )

        task_id = TaskIdentifier.from_result(result)
        key = task_id.to_key()

        # Parse it back
        parsed = TaskIdentifier.from_key(key)

        assert parsed.question_id == "urn:uuid:abc123"
        assert parsed.answering_canonical_key == "langchain:claude-haiku-4-5:otar-official"
        assert parsed.parsing_canonical_key == "langchain:claude-haiku-4-5:"
        assert parsed.replicate == 2

    def test_task_id_roundtrip_no_replicate(self):
        """Test roundtrip without replicate."""
        result = self._create_mock_result(replicate=None)

        task_id = TaskIdentifier.from_result(result)
        key = task_id.to_key()
        parsed = TaskIdentifier.from_key(key)

        assert parsed.replicate is None
        assert parsed.question_id == "q1"


class TestTaskIdentifierFromTaskDict:
    """Tests for TaskIdentifier.from_task_dict() method."""

    def test_basic_task_dict(self):
        """Test creating TaskIdentifier from a task dictionary."""
        answering_model = ModelConfig(
            id="answering-1",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            interface="langchain",
        )
        parsing_model = ModelConfig(
            id="parsing-1",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            interface="langchain",
        )

        task = {
            "question_id": "q1",
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "replicate": 1,
        }

        task_id = TaskIdentifier.from_task_dict(task)

        assert task_id.question_id == "q1"
        assert task_id.answering_canonical_key == "langchain:claude-haiku-4-5:"
        assert task_id.parsing_canonical_key == "langchain:claude-haiku-4-5:"
        assert task_id.replicate == 1

    def test_task_dict_with_mcp_tools(self):
        """Test creating TaskIdentifier from task dict with MCP tools."""
        answering_model = ModelConfig(
            id="answering-1",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            interface="langchain",
            mcp_urls_dict={"brave": "http://localhost:8080", "fs": "http://localhost:8081"},
        )
        parsing_model = ModelConfig(
            id="parsing-1",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            interface="langchain",
        )

        task = {
            "question_id": "q1",
            "answering_model": answering_model,
            "parsing_model": parsing_model,
        }

        task_id = TaskIdentifier.from_task_dict(task)

        # Tools are sorted in canonical_key
        assert task_id.answering_canonical_key == "langchain:claude-haiku-4-5:brave|fs"
        # Parsing model never has tools (role="parsing")
        assert task_id.parsing_canonical_key == "langchain:claude-haiku-4-5:"

    def test_from_task_dict_matches_from_result(self):
        """Test that from_task_dict and from_result produce the same key for matching inputs."""
        answering_model = ModelConfig(
            id="answering-1",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            interface="langchain",
            mcp_urls_dict={"otar-local": "http://localhost:8080"},
        )
        parsing_model = ModelConfig(
            id="parsing-1",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
            interface="langchain",
        )

        task = {
            "question_id": "q1",
            "answering_model": answering_model,
            "parsing_model": parsing_model,
            "replicate": 1,
        }

        task_key = TaskIdentifier.from_task_dict(task).to_key()

        # Simulate what a result would look like after pipeline execution
        answering_identity = ModelIdentity.from_model_config(answering_model, role="answering")
        parsing_identity = ModelIdentity.from_model_config(parsing_model, role="parsing")

        class MockMetadata:
            pass

        metadata = MockMetadata()
        metadata.question_id = "q1"
        metadata.answering = answering_identity
        metadata.parsing = parsing_identity
        metadata.replicate = 1

        class MockResult:
            pass

        result = MockResult()
        result.metadata = metadata

        result_key = TaskIdentifier.from_result(result).to_key()

        assert task_key == result_key
