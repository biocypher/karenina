# Testing Implementation Roadmap

**Parent**: [README.md](./README.md)

**Status**: ✅ Phases 1-2, 5-6 Complete | ⚠️ Phase 3 Partial | ⚠️ Phase 4 Pending (30/44 PRD tasks, 68%)

---

## Phase 1: Foundation ✅ COMPLETE

- [x] Create target directory structure (task-001)
- [x] Set up `conftest.py` with shared fixtures (task-004)
- [x] Implement `FixtureBackedLLMClient` (task-003)
- [x] Create fixture capture script reference (task-005)
- [x] Create `tests/README.md` explaining fixture philosophy (task-040)
- [x] Create `tests/fixtures/MANIFEST.md` (task-041)

---

## Phase 2: Big Bang Migration ✅ COMPLETE

- [x] Move existing ~30 flat test files to new structure (task-001)
- [x] Update imports and fixture references (task-001)
- [x] Verify all tests still pass (806 unit tests passing)
- [x] Add missing unit tests for uncovered modules

---

## Phase 3: LLM Fixtures ⚠️ INFRASTRUCTURE COMPLETE, FIXTURES PENDING

- [x] Identify all LLM call sites in codebase
- [x] Implement capture script infrastructure (task-005, task-006)
- [x] Create directory structure for fixtures
- [ ] Capture fixtures for each scenario (requires ANTHROPIC_API_KEY):
  - [ ] `template_parsing` scenarios (task-022)
  - [ ] `rubric_evaluation` scenarios (task-023)
  - [ ] `abstention` scenarios (task-024)
  - [ ] `embedding` scenarios
  - [ ] `generation` scenarios
- [x] Document in MANIFEST.md (task-041)

**Note**: Fixture infrastructure is complete (`FixtureBackedLLMClient`, capture script). Actual fixture capture requires ANTHROPIC_API_KEY and is done via `scripts/capture_fixtures.py`. Run with `--all` to capture all scenarios or `--scenario <name>` for specific scenarios.

---

## Phase 4: Integration Tests ⚠️ PENDING

- [x] Create directory structure for integration tests (task-001)
- [ ] Create tests for each integration boundary:
  1. [ ] Verification Pipeline (stages) - task-026
  2. [ ] Template + Parser - task-027
  3. [ ] Rubric Evaluation - task-028, task-031
  4. [ ] Storage/Checkpoint I/O - task-032
  5. [ ] CLI commands - task-033
- [ ] Add cross-component scenarios - task-034
- [ ] Include concurrency/race condition tests

**Note**: Integration tests depend on LLM fixtures (Phase 3) or can be written for deterministic components (RegexTrait, CallableTrait) without fixtures.

---

## Phase 5: E2E Tests ✅ COMPLETE

- [x] Create e2e/conftest.py with CLI runner fixtures (task-035)
- [x] Implement 12 full verification workflow tests (task-036)
- [x] Add 17 error handling tests (task-038)
- [x] Add 17 preset command tests (task-039)
- [x] Add 12 checkpoint resume tests (task-037)
- [x] Verify CLI interface contracts

**E2E Test Summary**: 58 E2E tests covering full verification workflow, error handling, preset commands, and checkpoint resume functionality.

---

## Phase 6: Polish ✅ COMPLETE

- [x] Review and fill coverage gaps (task-042)
- [x] Update documentation (task-043)
- [x] Create fixtures MANIFEST.md (task-041)
- [ ] Create contributor guide (deferred)

---

## Test Statistics

| Category | Tests | Status |
|----------|-------|--------|
| Unit Tests | 806 | ✅ Passing |
| E2E Tests | 58 | ✅ Passing |
| Integration Tests | 0 | ⚠️ Pending |
| **Total** | **864** | ✅ Passing |

---

## Lessons Learned

### Deviations from Original Plan

1. **Fixture MANIFEST.md**: Created as a central documentation file rather than embedding metadata in individual fixture files. This makes it easier to see all fixtures at a glance.

2. **Test Statistics**: Original plan aimed for 90% unit test coverage. Actual implementation achieved 30% overall with 806 tests. This provides good coverage of core schemas and utilities while leaving complex pipeline stages for integration tests.

3. **E2E Tests Completed**: Despite being marked as "minimal" in early planning, comprehensive E2E tests were completed covering 58 scenarios across 4 test files.

4. **LLM Fixture Capture**: The capture script (`scripts/capture_fixtures.py`) was implemented but requires API keys to run. Fixtures are captured on-demand rather than pre-capturing all scenarios.

### What Worked Well

1. **FixtureBackedLLMClient**: The SHA256-based hashing system ensures consistent fixture lookup without brittle path matching.

2. **Directory Structure**: The `tests/unit/<module>/` pattern maps cleanly to `src/karenina/<module>/`.

3. **Pytest Markers**: Using `@pytest.mark.unit`, `@pytest.mark.integration`, and `@pytest.mark.e2e` enables selective test execution.

4. **Conftest Fixtures**: Shared fixtures in root `conftest.py` and `e2e/conftest.py` reduce duplication across test files.

5. **E2E Test Coverage**: Comprehensive E2E tests verify CLI workflows without requiring API keys by using manual interface and state file simulation.

### Next Steps (Requires API Keys)

1. **LLM Fixture Capture**: Run `scripts/capture_fixtures.py --all` with ANTHROPIC_API_KEY to capture response fixtures.

2. **Integration Tests**: Write tests that combine multiple components (e.g., template parsing + verification, rubric evaluation).

3. **Pipeline Stage Tests**: Add tests for each of the 12 verification pipeline stages.

4. **Coverage Gaps**: Focus on modules with <30% coverage:
   - `benchmark/task_eval/*` (0%)
   - `benchmark/verification/evaluators/*` (0-19%)
   - `cli/verify.py` (5%)
   - `domain/answers/builder.py` (0%)

---

*Last updated: 2025-01-11*
