"""Testing utilities for karenina.

This module provides utilities for deterministic testing of LLM-based code,
including a fixture-backed LLM client that replays captured responses.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage


@dataclass
class MockUsage:
    """Mock usage metadata for LLM responses.

    Mimics the usage information returned by real LLM APIs.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __getitem__(self, key: str) -> int:
        """Allow dict-style access for compatibility."""
        return getattr(self, key, 0)


@dataclass
class MockResponse:
    """Mock LLM response that matches real LLM client interface.

    Real LLM responses have .content, .id, .model, and optionally .usage attributes.
    This class provides the same interface for testing.

    Note: For LangGraph agents, use FixtureBackedLLMClient which returns AIMessage.
    """

    content: str
    id: str = "mock-response-id"
    model: str = "claude-haiku-4-5"
    usage: MockUsage = field(default_factory=MockUsage)

    def __str__(self) -> str:
        return self.content


class FixtureBackedLLMClient:
    """LLM client that returns captured fixture responses instead of calling real API.

    This enables deterministic, fast tests without live API calls. Fixtures are
    indexed by SHA256 hash of the request messages, ensuring the same prompt
    always returns the same captured response.

    Fixtures MUST be captured from real pipeline runs, not hand-crafted, to ensure
    production accuracy. Use scripts/capture_fixtures.py to generate fixtures.

    Example:
        ```python
        from karenina.utils.testing import FixtureBackedLLMClient
        from pathlib import Path

        client = FixtureBackedLLMClient(Path("tests/fixtures/llm_responses"))
        response = client.invoke([HumanMessage("What is 2+2?")])
        print(response.content)  # "4" (from captured fixture)
        ```

    Attributes:
        fixtures_dir: Root directory containing LLM response fixtures
    """

    def __init__(self, fixtures_dir: Path) -> None:
        """Initialize the fixture-backed client.

        Args:
            fixtures_dir: Root directory containing LLM response fixtures.
                Fixtures should be JSON files named by SHA256 hash of the prompt.
        """
        self._fixtures_dir = Path(fixtures_dir)
        self._cache: dict[str, dict[str, Any]] = {}  # prompt_hash -> fixture data

    def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:  # noqa: ARG002
        """Invoke LLM with messages, returning captured fixture response as AIMessage.

        Args:
            messages: List of BaseMessage objects (HumanMessage, SystemMessage, etc.)
            **kwargs: Additional arguments (ignored for fixture replay)

        Returns:
            AIMessage compatible with LangGraph agents

        Raises:
            ValueError: If no fixture exists for the given prompt hash
        """
        prompt_hash = self._hash_messages(messages)

        # Check cache first
        if prompt_hash not in self._cache:
            fixture_data = self._load_fixture(prompt_hash)
            if fixture_data is None:
                raise ValueError(
                    f"No fixture found for prompt hash {prompt_hash[:8]}...\n"
                    f"To regenerate: python scripts/capture_fixtures.py --all\n"
                    f"Messages: {[str(m) for m in messages]}"
                )
            self._cache[prompt_hash] = fixture_data

        fixture = self._cache[prompt_hash]
        response_data = fixture.get("response", {})

        # Build response from fixture data
        content = response_data.get("content", "")
        response_id = response_data.get("id", f"fixture-{prompt_hash[:8]}")

        # Extract usage metadata
        usage_data = response_data.get("usage", {})
        usage_metadata = {
            "input_tokens": usage_data.get("input_tokens", 0),
            "output_tokens": usage_data.get("output_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0),
        }

        # Return AIMessage for LangGraph compatibility
        return AIMessage(
            content=content,
            id=response_id,
            usage_metadata=usage_metadata,
        )

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        """Async version of invoke (returns same result synchronously).

        Args:
            messages: List of BaseMessage objects
            **kwargs: Additional arguments (ignored)

        Returns:
            AIMessage compatible with LangGraph agents
        """
        return self.invoke(messages, **kwargs)

    def _hash_messages(self, messages: list[Any]) -> str:
        """Generate SHA256 hash of messages for fixture lookup.

        Normalizes message content to ensure consistent hashing across runs.

        Args:
            messages: List of BaseMessage objects

        Returns:
            SHA256 hex digest
        """
        # Sort and normalize messages for consistent hashing
        normalized = []
        for msg in messages:
            # Handle both BaseMessage objects and plain dicts
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = str(msg)

            # Normalize whitespace for consistency
            normalized.append(" ".join(str(content).split()))

        # Create deterministic JSON string
        hash_input = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def _load_fixture(self, prompt_hash: str) -> dict[str, Any] | None:
        """Load fixture file by prompt hash.

        Searches recursively in the fixtures directory.

        Args:
            prompt_hash: SHA256 hash of the prompt

        Returns:
            Fixture data dict with 'metadata', 'request', 'response' keys,
            or None if not found
        """
        if not self._fixtures_dir.exists():
            return None

        # Search recursively for fixture file named by hash
        for fixture_path in self._fixtures_dir.rglob(f"{prompt_hash}.json"):
            try:
                with fixture_path.open("r") as f:
                    data: dict[str, Any] = json.load(f)
                    return data
            except (json.JSONDecodeError, OSError) as e:
                # Log warning but continue searching
                print(f"Warning: Failed to load fixture {fixture_path}: {e}")

        return None

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "FixtureBackedLLMClient":  # noqa: ARG002
        """Bind tools to the LLM and return a new fixture-backed wrapper.

        This method enables compatibility with LangGraph agents that use tool binding.
        The returned client shares the same fixture directory but can be extended
        for tool-aware fixture matching in the future.

        Args:
            tools: List of tools to bind (stored but not used in fixture lookup)
            **kwargs: Additional arguments for bind_tools (ignored)

        Returns:
            A new FixtureBackedLLMClient sharing the same fixtures directory
        """
        # Return a new client that shares our fixtures
        # Tools are not currently used in fixture lookup, but could be in the future
        client = FixtureBackedLLMClient(self._fixtures_dir)
        client._cache = self._cache  # Share the cache
        return client

    def __getattr__(self, name: str) -> Any:
        """Provide default values for common LLM attributes.

        This ensures compatibility with LangGraph and other frameworks that
        may access attributes like model_name on the LLM object.
        """
        # Common LLM attributes with sensible defaults
        defaults = {
            "model_name": "claude-haiku-4-5",
            "model": "claude-haiku-4-5",
            "temperature": 0.0,
        }
        if name in defaults:
            return defaults[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


__all__ = ["FixtureBackedLLMClient", "MockResponse", "MockUsage"]
