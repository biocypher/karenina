"""Test FixtureBackedLLMClient for deterministic LLM test replay.

This module verifies that the fixture-backed LLM client:
- Correctly hashes messages for fixture lookup
- Raises ValueError when fixture not found
- Returns MockResponse with correct attributes
- Handles BaseMessage objects correctly
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from karenina.utils.testing import FixtureBackedLLMClient, MockResponse, MockUsage


@pytest.mark.unit
def test_fixture_client_raises_value_error_for_unknown_prompt(llm_fixtures_dir: Path) -> None:
    """Test that FixtureBackedLLMClient raises ValueError for unknown prompts."""
    client = FixtureBackedLLMClient(llm_fixtures_dir)

    with pytest.raises(ValueError, match="No fixture found for prompt hash"):
        client.invoke([HumanMessage("Unknown prompt that has no fixture")])


@pytest.mark.unit
def test_fixture_client_hash_consistency() -> None:
    """Test that message hashing is consistent across multiple calls."""
    client = FixtureBackedLLMClient(Path("/tmp"))

    messages = [HumanMessage("Test message")]
    hash1 = client._hash_messages(messages)
    hash2 = client._hash_messages(messages)

    assert hash1 == hash2, "Hash should be consistent for identical messages"


@pytest.mark.unit
def test_fixture_client_hash_normalizes_whitespace() -> None:
    """Test that hashing normalizes whitespace for consistency."""
    client = FixtureBackedLLMClient(Path("/tmp"))

    messages1 = [HumanMessage("Test message with spaces")]
    messages2 = [HumanMessage("Test    message    with    spaces")]

    hash1 = client._hash_messages(messages1)
    hash2 = client._hash_messages(messages2)

    assert hash1 == hash2, "Hash should normalize whitespace"


@pytest.mark.unit
def test_fixture_client_hash_handles_base_messages() -> None:
    """Test that hashing handles both BaseMessage objects and plain dicts."""
    client = FixtureBackedLLMClient(Path("/tmp"))

    messages_as_objects = [SystemMessage("System prompt"), HumanMessage("User message")]
    messages_as_dicts = [{"content": "System prompt"}, {"content": "User message"}]

    hash1 = client._hash_messages(messages_as_objects)
    hash2 = client._hash_messages(messages_as_dicts)

    # Hashes should be the same since content is identical
    assert hash1 == hash2, "Hash should handle both BaseMessage and dict inputs"


@pytest.mark.unit
def test_mock_response_attributes() -> None:
    """Test that MockResponse has expected attributes."""
    response = MockResponse(content="Test content", id="test-id", model="test-model")

    assert response.content == "Test content"
    assert response.id == "test-id"
    assert response.model == "test-model"
    assert str(response) == "Test content"


@pytest.mark.unit
def test_mock_response_usage() -> None:
    """Test that MockResponse has usage metadata."""
    usage = MockUsage(input_tokens=10, output_tokens=20, total_tokens=30)
    response = MockResponse(content="Test", usage=usage)

    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 20
    assert response.usage.total_tokens == 30
    assert response.usage["input_tokens"] == 10


@pytest.mark.unit
def test_fixture_client_with_real_fixture(tmp_path: Path) -> None:
    """Test that FixtureBackedLLMClient loads and returns real fixtures correctly."""
    # Create a temporary fixture file
    fixture_dir = tmp_path / "llm_responses"
    fixture_dir.mkdir()

    # Compute the hash for our test message
    client = FixtureBackedLLMClient(fixture_dir)
    test_message = [HumanMessage("What is 2+2?")]
    prompt_hash = client._hash_messages(test_message)

    # Create fixture file with expected structure
    fixture_data = {
        "metadata": {
            "model": "claude-haiku-4-5",
            "captured_at": "2025-01-11T00:00:00Z",
        },
        "request": {"messages": [{"content": "What is 2+2?"}]},
        "response": {
            "content": "4",
            "id": "msg-123",
            "model": "claude-haiku-4-5",
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        },
    }

    fixture_file = fixture_dir / f"{prompt_hash}.json"
    with fixture_file.open("w") as f:
        json.dump(fixture_data, f)

    # Now test that the client loads and returns the fixture
    response = client.invoke(test_message)

    assert response.content == "4"
    assert response.id == "msg-123"
    assert response.model == "claude-haiku-4-5"
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 1
    assert response.usage.total_tokens == 6
