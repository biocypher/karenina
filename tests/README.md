# Karenina Tests

Quick reference for running and writing tests for the karenina package.

## Running Tests

```bash
# Run all tests
uv run pytest

# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/

# Run only e2e tests
uv run pytest tests/e2e/

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=src/karenina --cov-report=html

# Run specific test file
uv run pytest tests/unit/schemas/test_verification_config.py

# Run tests matching pattern
uv run pytest -k "test_verification"
```

## Test Organization

```
tests/
├── README.md                        # This file
├── conftest.py                      # Shared pytest fixtures
├── unit/                            # Pure logic, no I/O, no LLM calls
│   ├── benchmark/                   # Benchmark class tests
│   ├── cli/                         # CLI utility function tests
│   ├── domain/                      # Domain logic tests
│   ├── infrastructure/              # LLM client, session management
│   ├── integrations/                # Third-party integrations (GEPA)
│   ├── schemas/                     # Pydantic model validation
│   └── storage/                     # Storage serialization
│
├── integration/                     # Component integration
│   ├── cli/                         # CLI command tests
│   ├── rubrics/                     # Rubric evaluation
│   ├── storage/                     # Checkpoint I/O
│   └── verification/                # Pipeline orchestration
│
├── e2e/                             # Full workflow tests
│
└── fixtures/                        # Test fixtures
    ├── checkpoints/                 # Sample checkpoint files
    ├── llm_responses/               # Captured LLM responses
    │   └── claude-haiku-4-5/
    │       ├── template_parsing/
    │       ├── rubric_evaluation/
    │       ├── abstention/
    │       └── generation/
    └── templates/                   # Answer template fixtures
```

## Test Markers

Tests use pytest markers for categorization:

- `@pytest.mark.unit` - Unit tests (no external dependencies)
- `@pytest.mark.integration` - Integration tests (multiple components)
- `@pytest.mark.e2e` - End-to-end tests (full workflows)

Run tests by marker:
```bash
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m e2e
```

## LLM Fixtures

LLM response fixtures are captured from real API calls, not hand-crafted.

**To regenerate fixtures**:
```bash
# Regenerate all fixtures
python scripts/capture_fixtures.py --all

# Regenerate specific scenario
python scripts/capture_fixtures.py --scenario template_parsing

# List available scenarios
python scripts/capture_fixtures.py --list
```

See [docs/testing/FIXTURES.md](../docs/testing/FIXTURES.md) for details.

## Writing Tests

### Unit Tests

- Test pure logic without external dependencies
- Use mocks for LLM clients (FixtureBackedLLMClient)
- Place in `tests/unit/<module>/`
- Focus on edge cases and validation

Example:
```python
import pytest
from karenina.schemas.verification import VerificationConfig

@pytest.mark.unit
def test_verification_config_defaults() -> None:
    config = VerificationConfig(
        parsing_models=[...],
        answering_models=[...],
    )
    assert config.rubric_enabled is False
```

### Integration Tests

- Test multiple components working together
- Use fixture-backed LLM client where needed
- Place in `tests/integration/<component>/`
- Reference which fixture file is used in docstring

Example:
```python
import pytest
from karenina.integrations.gepa.scoring import compute_objective_scores

@pytest.mark.integration
def test_compute_objective_scores_template_only() -> None:
    """Test template-only scoring using fixtures/template_parsing/basic_extraction.json"""
    result = load_verification_result(...)
    scores = compute_objective_scores(result, "gpt-4", config)
```

### E2E Tests

- Test complete CLI workflows
- Invoke main() directly, not subprocess
- Place in `tests/e2e/`

## Conftest Fixtures

Shared fixtures available in `conftest.py`:

- `benchmark_path` - Path to sample checkpoint fixture
- `sample_checkpoint` - Loaded Benchmark object
- `template_dir` - Path to template fixtures
- `mock_llm_client` - Fixture-backed LLM client

## Important Principles

1. **Real fixtures, not fake** - LLM fixtures must be captured from actual API calls
2. **Challenge the code** - Tests should find bugs, not just confirm happy paths
3. **Test failure modes** - Error handling is as important as success paths
4. **Avoid brittle tests** - Tests should be resilient to implementation changes

## Coverage Goals

Target: ≥80% line coverage for source modules.

```bash
uv run pytest --cov=src/karenina --cov-report=term
```

## More Documentation

Detailed testing documentation:

- [docs/testing/README.md](../docs/testing/README.md) - Overall testing strategy
- [docs/testing/FIXTURES.md](../docs/testing/FIXTURES.md) - Fixture system
- [docs/testing/UNIT_TESTS.md](../docs/testing/UNIT_TESTS.md) - Unit test guidelines
- [docs/testing/INTEGRATION_TESTS.md](../docs/testing/INTEGRATION_TESTS.md) - Integration test guidelines
- [docs/testing/E2E_TESTS.md](../docs/testing/E2E_TESTS.md) - E2E test guidelines
- [docs/testing/ROADMAP.md](../docs/testing/ROADMAP.md) - Implementation tasks
