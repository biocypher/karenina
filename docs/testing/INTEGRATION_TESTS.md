# Integration Test Strategy

**Parent**: [README.md](./README.md)

---

## Scope

Integration tests verify that multiple components work together correctly. They use captured LLM fixtures (see [FIXTURES.md](./FIXTURES.md)) to ensure realistic behavior without live API calls.

---

## Integration Boundaries

| Boundary | Components Involved | Priority |
|----------|---------------------|----------|
| **Verification Pipeline** | Stage orchestrator, all stages | P0 |
| **Template + Parser** | Template loader, LLM client, parser | P1 |
| **Rubric Evaluation** | Rubric loader, trait evaluators, aggregator | P2 |
| **Storage** | Checkpoint I/O, Benchmark class | P3 |

---

## Coverage Matrix

For each integration boundary, test:

| Scenario | Description | Priority |
|----------|-------------|----------|
| Happy path | Everything works as expected | P0 |
| First component fails | Error propagates correctly | P0 |
| Second component fails | Partial state handled | P0 |
| Boundary conditions | Empty inputs, max sizes | P1 |
| Retry/recovery | Transient failures recovered | P1 |
| Concurrent execution | Race conditions, locking | P1 |

---

## Verification Pipeline Stages

The verification pipeline has multiple stages. Each requires integration testing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    VERIFICATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│ Stage 1: Load Template                                           │
│   Tests: missing template, invalid Python, syntax errors         │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2: Validate Template                                       │
│   Tests: schema errors, missing verify(), wrong inheritance      │
├─────────────────────────────────────────────────────────────────┤
│ Stage 3: Parse Template (LLM)                                    │
│   Tests: malformed JSON, missing fields, type mismatches         │
│   Fixtures: template_parsing/*                                   │
├─────────────────────────────────────────────────────────────────┤
│ Stage 4: Generate Answer (LLM)                                   │
│   Tests: refusal, empty response, truncated output               │
│   Fixtures: generation/*                                         │
├─────────────────────────────────────────────────────────────────┤
│ Stage 5: Verify Answer                                           │
│   Tests: verify() throws, returns non-bool, timeout              │
├─────────────────────────────────────────────────────────────────┤
│ Stage 6: Rubric Evaluation (LLM)                                 │
│   Tests: trait failures, mixed results, scoring errors           │
│   Fixtures: rubric_evaluation/*                                  │
├─────────────────────────────────────────────────────────────────┤
│ Stage 7: Deep Judgment (LLM)                                     │
│   Tests: multi-turn reasoning, early exit, max turns             │
│   Fixtures: rubric_evaluation/reasoning/*                        │
├─────────────────────────────────────────────────────────────────┤
│ Stage 8: Embedding Check (DEFERRED)                              │
│   Tests: embedding service down, dimension mismatch              │
│   Note: Deferred to later phase                                  │
├─────────────────────────────────────────────────────────────────┤
│ Stage 9: Abstention Detection                                    │
│   Tests: model says "I don't know", hedging language             │
│   Fixtures: abstention/*                                         │
├─────────────────────────────────────────────────────────────────┤
│ Stage 10: Finalize Results                                       │
│   Tests: aggregation edge cases, conflicting results             │
├─────────────────────────────────────────────────────────────────┤
│ Stage 11: Export Results                                         │
│   Tests: serialization errors, encoding issues                   │
├─────────────────────────────────────────────────────────────────┤
│ Stage 12: Save Checkpoint                                        │
│   Tests: disk full, permission denied, concurrent writes         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Template + Parser Integration

```python
"""
Integration: Template Parsing + Verification

Tests the flow: load template → LLM parses answer → verify() runs

Fixtures used:
- fixtures/llm_responses/claude-haiku-4-5/template_parsing/basic_extraction.json
- fixtures/llm_responses/claude-haiku-4-5/template_parsing/malformed_json.json
"""

class TestTemplateVerificationIntegration:
    """Tests template loading through verification result."""

    def test_successful_extraction_and_verification(
        self, pipeline, sample_template, sample_trace
    ):
        """
        Happy path: LLM extracts correctly, verify() returns True.

        Flow:
        1. Load template with single string field
        2. LLM parses trace into template
        3. verify() compares extracted value to ground truth
        4. Result is PASS
        """
        result = pipeline.run(template=sample_template, trace=sample_trace)

        assert result.passed is True
        assert result.extracted_answer is not None
        assert result.error is None

    def test_llm_returns_malformed_json(
        self, pipeline, sample_template, malformed_trace
    ):
        """
        LLM response doesn't match schema → graceful error.

        Uses fixture: template_parsing/malformed_json.json

        Expected behavior:
        - Pipeline catches JSON parse error
        - Result is FAIL with error message
        - No exception propagates
        """
        result = pipeline.run(template=sample_template, trace=malformed_trace)

        assert result.passed is False
        assert "JSON" in result.error or "parse" in result.error.lower()

    def test_verify_throws_exception(self, pipeline, buggy_template, sample_trace):
        """
        verify() raises exception → captured, not propagated.

        Template has verify() that raises ValueError.
        Pipeline should catch this and return error result.
        """
        result = pipeline.run(template=buggy_template, trace=sample_trace)

        assert result.passed is False
        assert "exception" in result.error.lower() or "error" in result.error.lower()
```

---

## Rubric Integration Tests

One test file per trait type:

### `test_llm_rubric_trait.py`

```python
class TestLLMRubricTraitIntegration:
    """Tests for LLMRubricTrait evaluation."""

    def test_boolean_trait_pass(self, evaluator, llm_client, sample_trace):
        """
        LLMRubricTrait with boolean output → True.
        Fixture: rubric_evaluation/score/bool_true.json
        """
        ...

    def test_boolean_trait_fail(self, evaluator, llm_client, sample_trace):
        """
        LLMRubricTrait with boolean output → False.
        Fixture: rubric_evaluation/score/bool_false.json
        """
        ...

    def test_scored_trait_valid(self, evaluator, llm_client, sample_trace):
        """
        LLMRubricTrait with integer score within range.
        Fixture: rubric_evaluation/score/valid.json
        """
        ...

    def test_scored_trait_out_of_range(self, evaluator, llm_client, sample_trace):
        """
        LLMRubricTrait returns score outside defined range.
        Fixture: rubric_evaluation/score/out_of_range.json

        Expected: Score clamped or error returned
        """
        ...
```

### `test_regex_trait.py`

```python
class TestRegexTraitIntegration:
    """Tests for RegexTrait evaluation (no LLM needed)."""

    def test_pattern_found(self, evaluator, trace_with_citations):
        """Pattern matches in trace."""
        ...

    def test_pattern_not_found(self, evaluator, trace_without_citations):
        """Pattern does not match."""
        ...

    def test_multiple_matches(self, evaluator, trace_many_citations):
        """Pattern matches multiple times."""
        ...
```

### `test_callable_trait.py`

```python
class TestCallableTraitIntegration:
    """Tests for CallableTrait evaluation."""

    def test_callable_returns_true(self, evaluator, sample_trace):
        """Custom function returns True."""
        ...

    def test_callable_returns_false(self, evaluator, sample_trace):
        """Custom function returns False."""
        ...

    def test_callable_raises_exception(self, evaluator, sample_trace):
        """Custom function throws → captured gracefully."""
        ...
```

### `test_metric_rubric_trait.py`

```python
class TestMetricRubricTraitIntegration:
    """Tests for MetricRubricTrait (precision/recall/F1)."""

    def test_perfect_extraction(self, evaluator, llm_client, ideal_trace):
        """All expected entities found, no false positives."""
        ...

    def test_partial_extraction(self, evaluator, llm_client, partial_trace):
        """Some entities missing → recall < 1.0."""
        ...

    def test_false_positives(self, evaluator, llm_client, noisy_trace):
        """Extra entities found → precision < 1.0."""
        ...
```

---

## Storage Integration Tests

```python
class TestStorageIntegration:
    """Tests for checkpoint I/O and persistence."""

    def test_checkpoint_roundtrip(self, tmp_path, sample_benchmark):
        """Save checkpoint → load checkpoint → identical data."""
        checkpoint_path = tmp_path / "test.jsonld"
        sample_benchmark.save(checkpoint_path)
        loaded = Benchmark.load(checkpoint_path)

        assert len(loaded.questions) == len(sample_benchmark.questions)
        assert loaded.metadata == sample_benchmark.metadata

    def test_progressive_save_during_verification(
        self, pipeline, benchmark_with_questions, tmp_path
    ):
        """Results saved progressively during batch verification."""
        ...

    def test_concurrent_checkpoint_writes(self, tmp_path, sample_benchmark):
        """
        Multiple concurrent writes to same checkpoint.

        Tests race condition handling - should not corrupt data.
        """
        ...
```

---

## Cross-Component Scenarios

```python
class TestCrossComponentIntegration:
    """Tests spanning multiple integration boundaries."""

    def test_template_passes_rubric_fails(
        self, pipeline, template_with_rubric, trace_correct_but_verbose
    ):
        """
        Template verification passes but rubric fails.

        Scenario: Model gives correct answer but is too verbose.
        - Template: extracts correct value → PASS
        - Rubric: checks conciseness → FAIL

        Both results should be captured independently.
        """
        ...

    def test_pipeline_interrupted_recovery(
        self, pipeline, benchmark_with_questions, tmp_path
    ):
        """
        Pipeline interrupted mid-run → graceful recovery on restart.

        Simulate interruption after 2 of 5 questions.
        Restart should resume from question 3, not start over.
        """
        ...
```

---

## Directory Structure

```
tests/integration/
├── conftest.py                      # Integration-specific fixtures
├── verification/
│   ├── test_pipeline_orchestration.py
│   ├── test_stage_error_propagation.py
│   ├── test_stage_context_passing.py
│   └── test_retry_recovery.py
├── templates/
│   ├── test_template_extraction.py
│   ├── test_template_verification.py
│   └── test_template_error_handling.py
├── rubrics/
│   ├── test_llm_rubric_trait.py
│   ├── test_regex_trait.py
│   ├── test_callable_trait.py
│   └── test_metric_rubric_trait.py
├── storage/
│   ├── test_checkpoint_roundtrip.py
│   ├── test_progressive_save.py
│   └── test_result_persistence.py
└── cli/
    ├── test_verify_command.py
    ├── test_preset_command.py
    └── test_config_loading.py
```

---

*Last updated: 2025-01-11*
