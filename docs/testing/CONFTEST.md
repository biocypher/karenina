# Shared Pytest Fixtures

**Parent**: [README.md](./README.md)

---

## Overview

This document describes the shared fixtures defined in `tests/conftest.py` and layer-specific `conftest.py` files.

---

## Root conftest.py

Located at `tests/conftest.py`, provides fixtures available to all tests.

### Path Fixtures

```python
@pytest.fixture
def fixtures_dir():
    """Path to fixtures directory."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def llm_fixtures_dir(fixtures_dir):
    """Path to LLM response fixtures."""
    return fixtures_dir / "llm_responses" / "claude-haiku-4-5"
```

### FixtureBackedLLMClient

The core fixture client that replays captured LLM responses:

```python
class FixtureBackedLLMClient:
    """
    LLM client that replays captured fixtures instead of making live API calls.

    Matches requests by prompt hash. If no fixture exists for a prompt,
    raises ValueError with instructions to generate the missing fixture.
    """

    def __init__(self, fixture_dir: Path):
        self.fixture_dir = fixture_dir
        self._cache = {}

    def invoke(self, messages, **kwargs):
        """
        Return captured response for matching prompt.

        Matches the calling convention used in the codebase.
        """
        prompt_hash = self._hash_messages(messages)

        if prompt_hash not in self._cache:
            self._load_fixture(prompt_hash)

        if prompt_hash not in self._cache:
            raise ValueError(
                f"No fixture for prompt hash: {prompt_hash}\n"
                f"Run: python scripts/capture_pipeline_fixtures.py --scenario <name>"
            )

        return self._cache[prompt_hash]

    def _load_fixture(self, prompt_hash: str):
        """Search fixtures for matching hash."""
        for fixture_file in self.fixture_dir.rglob("*.json"):
            with open(fixture_file) as f:
                fixture = json.load(f)
            if fixture["metadata"]["prompt_hash"] == prompt_hash:
                self._cache[prompt_hash] = self._build_response(fixture["response"])
                break

    def _hash_messages(self, messages) -> str:
        """Generate hash from message content."""
        # Convert messages to serializable format
        serializable = []
        for msg in messages:
            if hasattr(msg, "content"):
                serializable.append({"role": type(msg).__name__, "content": msg.content})
            else:
                serializable.append(msg)
        content = json.dumps(serializable, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    def _build_response(self, response_data: dict):
        """Build response object matching expected interface."""
        # Return object with .content attribute
        class MockResponse:
            def __init__(self, data):
                self.content = data["content"][0]["text"]
                self.id = data.get("id", "mock_id")
                self.model = data.get("model", "claude-haiku-4-5")
                self.usage = data.get("usage", {})

        return MockResponse(response_data)


@pytest.fixture
def llm_client(llm_fixtures_dir):
    """Fixture-backed LLM client for tests."""
    return FixtureBackedLLMClient(llm_fixtures_dir)
```

### Sample Data Fixtures

```python
@pytest.fixture
def sample_template():
    """Simple answer template for testing."""
    from karenina.schemas.domain import BaseAnswer

    class Answer(BaseAnswer):
        value: str

        def verify(self) -> bool:
            return self.value.lower() == "correct"

    return Answer

@pytest.fixture
def sample_trace():
    """Sample LLM response trace."""
    return "The answer is: correct. This is based on the evidence provided."

@pytest.fixture
def sample_benchmark(tmp_path):
    """Minimal benchmark for testing."""
    from karenina.benchmark import Benchmark

    benchmark = Benchmark(name="Test Benchmark")
    benchmark.add_question(
        prompt="What is the answer?",
        expected="correct"
    )
    return benchmark
```

### Checkpoint Fixtures

```python
@pytest.fixture
def minimal_checkpoint(tmp_path, fixtures_dir):
    """Load minimal checkpoint fixture."""
    src = fixtures_dir / "checkpoints" / "minimal.jsonld"
    dst = tmp_path / "minimal.jsonld"
    dst.write_text(src.read_text())
    return dst

@pytest.fixture
def benchmark_with_results(fixtures_dir):
    """Benchmark checkpoint with existing results."""
    from karenina.benchmark import Benchmark
    return Benchmark.load(fixtures_dir / "checkpoints" / "with_results.jsonld")
```

---

## Integration conftest.py

Located at `tests/integration/conftest.py`, provides fixtures specific to integration tests.

```python
@pytest.fixture
def pipeline(llm_client, tmp_path):
    """
    Create verification pipeline with fixture-backed LLM client.

    The pipeline is configured to use the fixture client instead of
    making real API calls.
    """
    from karenina.benchmark.verification import VerificationPipeline

    return VerificationPipeline(
        llm_client=llm_client,
        working_dir=tmp_path
    )

@pytest.fixture
def evaluator(llm_client):
    """Rubric evaluator with fixture-backed client."""
    from karenina.benchmark.verification.evaluators import RubricEvaluator

    return RubricEvaluator(llm_client=llm_client)

@pytest.fixture
def template_with_rubric():
    """Template that includes rubric traits."""
    ...

@pytest.fixture
def trace_with_citations():
    """Trace containing citation patterns like [1], [2]."""
    return "According to the study [1], the treatment was effective. Further research [2] confirmed these findings [3]."

@pytest.fixture
def trace_without_citations():
    """Trace with no citation patterns."""
    return "The treatment was effective based on the available evidence."
```

---

## E2E conftest.py

Located at `tests/e2e/conftest.py`, provides fixtures specific to end-to-end tests.

```python
from click.testing import CliRunner

@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()

@pytest.fixture
def large_benchmark(tmp_path, fixtures_dir):
    """10-question benchmark for resume tests."""
    src = fixtures_dir / "checkpoints" / "complex_benchmark.jsonld"
    dst = tmp_path / "complex.jsonld"
    dst.write_text(src.read_text())
    return dst

@pytest.fixture
def preset_file(tmp_path):
    """Sample preset configuration file."""
    preset = {
        "name": "test_preset",
        "model": "claude-haiku-4-5",
        "max_retries": 2
    }
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(preset))
    return path
```

---

## Fixture Organization

```
tests/
├── conftest.py                 # Shared fixtures (paths, LLM client, samples)
├── fixtures/
│   ├── README.md               # → Links to docs/testing/FIXTURES.md
│   ├── MANIFEST.md             # Auto-generated fixture inventory
│   ├── llm_responses/          # Captured LLM responses
│   ├── checkpoints/            # Sample checkpoint files
│   └── templates/              # Sample answer templates
├── unit/
│   └── (no conftest needed)
├── integration/
│   └── conftest.py             # Pipeline, evaluator fixtures
└── e2e/
    └── conftest.py             # CLI runner, large benchmarks
```

---

*Last updated: 2025-01-11*
