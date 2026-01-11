# Karenina Testing Strategy

**Status**: Planning
**Scope**: `karenina/` package (core Python library)

---

## Executive Summary

This document defines a comprehensive testing strategy for the karenina package, designed to ensure thorough coverage, realistic test scenarios, and maintainable test infrastructure.

**Key Principles**:
1. Tests should challenge the code, not just confirm it works
2. LLM fixtures must be captured from real API calls, not hand-crafted
3. Integration tests are first-class citizens, not afterthoughts
4. Every source module has a corresponding test directory

---

## Documentation Map

| Document | Purpose |
|----------|---------|
| [FIXTURES.md](./FIXTURES.md) | LLM fixture capture system, format, and regeneration |
| [UNIT_TESTS.md](./UNIT_TESTS.md) | Unit test guidelines and module coverage |
| [INTEGRATION_TESTS.md](./INTEGRATION_TESTS.md) | Integration boundaries, pipeline stages, rubric tests |
| [E2E_TESTS.md](./E2E_TESTS.md) | End-to-end workflow tests |
| [CONFTEST.md](./CONFTEST.md) | Shared pytest fixtures and configuration |
| [ROADMAP.md](./ROADMAP.md) | Implementation phases and tasks |

---

## Current State

### Source Module Structure

```
src/karenina/
├── benchmark/        # Benchmark class, question management
├── cli/              # Command-line interface (verify, preset, etc.)
├── domain/           # Core domain logic
├── infrastructure/   # External service adapters (LLM clients, etc.)
├── integrations/     # Third-party integrations
├── schemas/          # Pydantic models, answer templates
├── storage/          # Checkpoint I/O, persistence
└── utils/            # Shared utilities
```

### Current Test Structure (Problems)

```
tests/
├── test_*.py              # ~30 flat files at root (inconsistent organization)
├── unit/                  # Partial unit test directory
├── integration/           # Partial integration directory
├── test_verification/     # Verification-specific tests
├── storage/               # Storage-specific tests
└── fixtures/              # Minimal (only mock_mcp_server.py)
```

**Issues Identified**:
- No clear mapping between source modules and test directories
- Inconsistent placement (some flat, some in subdirectories)
- Minimal LLM fixtures — most tests use hand-crafted mocks
- Integration tests are sparse and don't cover failure modes
- No documented fixture regeneration procedure

---

## Target Structure

```
tests/
├── README.md                        # Philosophy: real fixtures, not mocks
├── conftest.py                      # Shared fixtures, pytest configuration
├── unit/                            # Pure logic, no I/O, no LLM calls
│   ├── benchmark/
│   ├── schemas/
│   ├── domain/
│   ├── storage/
│   └── utils/
│
├── integration/                     # Multiple components working together
│   ├── conftest.py                  # Integration-specific fixtures
│   ├── verification/                # Pipeline stage combinations
│   ├── templates/                   # Template parsing + verification
│   ├── rubrics/                     # Rubric evaluation flows
│   │   ├── test_llm_rubric_trait.py
│   │   ├── test_regex_trait.py
│   │   ├── test_callable_trait.py
│   │   └── test_metric_rubric_trait.py
│   ├── storage/                     # Checkpoint I/O
│   └── cli/                         # CLI command integration
│
├── e2e/                             # Full pipeline runs
│   ├── test_full_verification_pipeline.py
│   ├── test_batch_verification.py
│   └── test_cli_workflows.py
│
└── fixtures/
    ├── README.md                    # Fixture philosophy and regeneration
    ├── MANIFEST.md                  # Documents all fixtures and their purpose
    ├── llm_responses/
    │   └── claude-haiku-4-5/
    │       ├── template_parsing/
    │       ├── rubric_evaluation/
    │       ├── deep_judgment/
    │       └── error_cases/
    ├── checkpoints/                 # Sample checkpoint files
    └── templates/                   # Sample answer templates
```

---

## Pytest Markers

Define these markers in `pytest.ini` or `pyproject.toml`:

```ini
[pytest]
markers =
    # By layer
    unit: Unit tests (no I/O, no LLM)
    integration: Integration tests (multiple components)
    e2e: End-to-end workflow tests

    # By speed
    slow: Tests taking > 1 second

    # By feature area
    pipeline: Verification pipeline tests
    rubric: Rubric evaluation tests
    storage: Checkpoint I/O tests
    cli: CLI command tests
```

**Usage**:
```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run pipeline-related tests across all layers
pytest -m pipeline
```

---

## Test Philosophy

### Core Principles

1. **Tests Should Challenge the Code**
   - Don't write tests that pass by construction
   - Include edge cases, boundary conditions, and failure modes
   - Ask: "What inputs could break this?"

2. **Tests as Specification**
   - Tests document expected behavior
   - Reading tests should explain how components work
   - Each test has a clear, descriptive name

3. **Regression Prevention**
   - Every bug fix includes a regression test
   - The test must fail before the fix and pass after
   - Document the original issue in the test docstring

4. **Realistic Scenarios**
   - Use captured LLM responses, not hand-crafted mocks
   - Test with real checkpoint files when possible
   - Simulate actual user workflows

### What to Test

**Always Test**:
- Happy path (baseline functionality)
- Empty/null inputs
- Boundary values (0, 1, max, max+1)
- Invalid inputs (wrong types, malformed data)
- Error propagation (exceptions bubble up correctly)
- State changes (before/after verification)

**For LLM-Dependent Code**:
- Successful extraction/evaluation
- Malformed JSON responses
- Partial/truncated responses
- Model refusals ("I cannot...")
- Rate limit errors
- Context length exceeded
- Unexpected response formats

### Test Independence

- Tests must not depend on execution order
- Each test sets up its own state
- Use fixtures for shared setup, not global state
- Clean up any created files/resources

---

## Decisions (Resolved)

### Fixture Management

| Decision | Resolution |
|----------|------------|
| Staleness detection | Manual regeneration via capture script; script updates MANIFEST.md |
| Non-determinism | Use `temperature=0`, accept minor variance |
| Fixture storage | Check into git directly (no LFS needed initially) |
| Error fixtures | Hybrid — capture real errors when possible, synthesize rare ones |
| Missing fixtures | Raise clear `ValueError` with exact regeneration command |

### Test Execution

| Decision | Resolution |
|----------|------------|
| Speed target | 5+ minutes acceptable; use markers for local subsets |
| E2E approach | Call CLI entry point directly (not subprocess) |
| Fixture paths | Shared `fixtures/` at root + layer-specific when needed |

### Test Organization

| Decision | Resolution |
|----------|------------|
| Rubric tests | One file per trait type |
| Module exclusions | None — all src/karenina modules require coverage |
| Embedding tests | Deferred (focus on core pipeline first) |
| Concurrency tests | Included (checkpoint race conditions are real risks) |

### Integration Priority

1. Verification Pipeline (stages)
2. Template + Parser
3. Rubric Evaluation
4. Storage/Checkpoint I/O

---

## Coverage Goals

### Quantitative Targets

| Category | Metric | Target |
|----------|--------|--------|
| Unit tests | Line coverage | ≥90% |
| Unit tests | Branch coverage | ≥80% |
| Integration tests | Boundary coverage | 100% of boundaries |
| Integration tests | Failure modes | ≥3 per boundary |
| E2E tests | Workflow coverage | 5 canonical scenarios |
| Fixtures | LLM call sites | 100% (no unmocked API calls) |

### Coverage Gaps (Acceptable)

| Gap | Reason | Mitigation |
|-----|--------|------------|
| Embedding checks | Deferred to later phase | Manual testing |
| UI rendering | No GUI in karenina package | Covered by karenina-gui tests |
| Rate limiting | Hard to reproduce reliably | Manual testing + monitoring |

---

*Document version: 2.0*
*Last updated: 2025-01-11*
