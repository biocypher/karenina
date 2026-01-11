# pytest configuration for karenina tests

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Pure logic tests - no I/O, no LLM calls")
    config.addinivalue_line("markers", "integration: Multiple components working together")
    config.addinivalue_line("markers", "e2e: End-to-end workflow tests")
    config.addinivalue_line("markers", "slow: Tests taking > 1 second")
    config.addinivalue_line("markers", "pipeline: Verification pipeline tests")
    config.addinivalue_line("markers", "rubric: Rubric evaluation tests")
    config.addinivalue_line("markers", "storage: Checkpoint I/O tests")
    config.addinivalue_line("markers", "cli: CLI command tests")


# =============================================================================
# FixtureBackedLLMClient - Deterministic LLM test replay
# =============================================================================


@dataclass
class MockUsage:
    """Mock usage metadata for LLM responses."""

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
        client = FixtureBackedLLMClient(Path("tests/fixtures/llm_responses"))
        response = client.invoke([HumanMessage("What is 2+2?")])
        print(response.content)  # "4" (from captured fixture)
    """

    def __init__(self, fixtures_dir: Path) -> None:
        """Initialize the fixture-backed client.

        Args:
            fixtures_dir: Root directory containing LLM response fixtures
        """
        self._fixtures_dir = Path(fixtures_dir)
        self._cache: dict[str, dict[str, Any]] = {}  # prompt_hash -> fixture data

    def invoke(self, messages: list[Any], **kwargs: Any) -> MockResponse:  # noqa: ARG002
        """Invoke LLM with messages, returning captured fixture response.

        Args:
            messages: List of BaseMessage objects (HumanMessage, SystemMessage, etc.)
            **kwargs: Additional arguments (ignored for fixture replay)

        Returns:
            MockResponse with content, id, model, usage attributes

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
        model = response_data.get("model", "claude-haiku-4-5")

        # Extract usage metadata
        usage_data = response_data.get("usage", {})
        usage = MockUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return MockResponse(content=content, id=response_id, model=model, usage=usage)

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

        Searches recursively in fixtures/llm_responses/ directory.

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
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                # Log warning but continue searching
                print(f"Warning: Failed to load fixture {fixture_path}: {e}")

        return None


# =============================================================================
# Shared pytest fixtures
# =============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the root fixtures directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def llm_fixtures_dir(fixtures_dir: Path) -> Path:
    """Return the LLM response fixtures directory path."""
    return fixtures_dir / "llm_responses"


@pytest.fixture
def llm_client(llm_fixtures_dir: Path) -> FixtureBackedLLMClient:
    """Return a FixtureBackedLLMClient for testing."""
    return FixtureBackedLLMClient(llm_fixtures_dir)


@pytest.fixture
def sample_trace() -> str:
    """Return a realistic LLM trace text for testing."""
    return """The answer is BCL2.

BCL2 is a proto-oncogene located on chromosome 18q21.33. It encodes
a protein that inhibits apoptosis, making it an important gene in
cancer research.

References:
[1] Chromosome location and gene function
[2] Apoptosis regulation studies"""


@pytest.fixture
def tmp_benchmark(tmp_path: Path) -> Any:
    """Create a minimal Benchmark with 1 question for testing.

    Returns a Benchmark instance that can be saved/loaded.
    """
    from karenina import Benchmark

    benchmark = Benchmark.create(
        name="test-benchmark",
        description="A minimal benchmark for testing",
        version="0.1.0",
    )

    # Add a single question
    benchmark.add_question(
        question="What is 2+2?",
        raw_answer="4",
        question_id="q001",
        finished=True,
    )

    return benchmark
