# pytest configuration for karenina tests

from pathlib import Path
from typing import Any

import pytest

# Import testing utilities from the main package
from karenina.utils.testing import FixtureBackedLLMClient


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
