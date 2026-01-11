# Testing Implementation Roadmap

**Parent**: [README.md](./README.md)

---

## Phase 1: Foundation

- [ ] Create target directory structure
- [ ] Set up `conftest.py` with shared fixtures
- [ ] Implement `FixtureBackedLLMClient`
- [ ] Create fixture capture script reference
- [ ] Create `tests/README.md` explaining fixture philosophy
- [ ] Create `tests/fixtures/MANIFEST.md`

---

## Phase 2: Big Bang Migration

- [ ] Move existing ~30 flat test files to new structure
- [ ] Update imports and fixture references
- [ ] Verify all tests still pass
- [ ] Add missing unit tests for uncovered modules

---

## Phase 3: LLM Fixtures

- [ ] Identify all LLM call sites in codebase
- [ ] Capture fixtures for each scenario (on-demand as tests are written)
- [ ] Validate fixtures work with FixtureBackedClient
- [ ] Document in MANIFEST.md

---

## Phase 4: Integration Tests

- [ ] Create tests for each integration boundary (priority order):
  1. Verification Pipeline (stages)
  2. Template + Parser
  3. Rubric Evaluation
  4. Storage/Checkpoint I/O
- [ ] Cover all 12 pipeline stages
- [ ] Test failure modes and recovery
- [ ] Add cross-component scenarios
- [ ] Include concurrency/race condition tests

---

## Phase 5: E2E Tests

- [ ] Implement 5 canonical workflow tests
- [ ] Add error handling tests
- [ ] Verify CLI interface contracts

---

## Phase 6: Polish

- [ ] Review and fill coverage gaps
- [ ] Update documentation
- [ ] Create contributor guide

---

*Last updated: 2025-01-11*
