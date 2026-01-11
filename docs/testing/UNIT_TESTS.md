# Unit Test Strategy

**Parent**: [README.md](./README.md)

---

## Scope

Unit tests verify isolated logic without I/O, network calls, or LLM dependencies. They test individual functions and classes in isolation.

---

## Guidelines

### One Concept Per Test

```python
# Good: focused test
def test_fuzzy_match_exact_match_returns_1():
    assert fuzzy_match("hello", "hello") == 1.0

# Bad: testing multiple things
def test_fuzzy_match():
    assert fuzzy_match("hello", "hello") == 1.0
    assert fuzzy_match("hello", "world") < 0.5
    assert fuzzy_match("", "") == 1.0
```

### Descriptive Names

Pattern: `test_{function}_{scenario}_{expected_outcome}`

```python
def test_verify_template_missing_verify_method_raises_validation_error():
    ...

def test_parse_checkpoint_empty_file_returns_empty_benchmark():
    ...

def test_aggregate_results_mixed_pass_fail_calculates_correct_ratio():
    ...
```

### Arrange-Act-Assert

```python
def test_question_filter_by_tag():
    # Arrange
    benchmark = Benchmark()
    benchmark.add_question(Question(tags=["math"]))
    benchmark.add_question(Question(tags=["science"]))

    # Act
    filtered = benchmark.filter_by_tag("math")

    # Assert
    assert len(filtered) == 1
    assert filtered[0].tags == ["math"]
```

### Docstrings for Context

Write detailed docstrings so that coding agents have context about the test purpose:

```python
def test_rubric_trait_score_out_of_range_clamps_to_bounds():
    """
    When an LLM returns a score outside the trait's defined range,
    the evaluator should clamp the score to the valid bounds.

    This prevents invalid scores from propagating through the pipeline
    and ensures consistent result aggregation.

    Regression: Issue #142 - out-of-range scores caused aggregation failures
    """
    trait = LLMRubricTrait(name="clarity", min_score=1, max_score=5)
    # ... test implementation
```

---

## Module Coverage

### `benchmark/`

| Area | What to Test |
|------|--------------|
| Question CRUD | Add, remove, update, get by ID |
| Filtering | By tag, by status, by date range |
| Result aggregation | Pass rate, score averages, weighted metrics |
| Metadata validation | Required fields, type checking |

### `schemas/`

| Area | What to Test |
|------|--------------|
| Pydantic validation | Required fields, type coercion, constraints |
| Serialization | To/from JSON, to/from dict |
| Default values | Correct defaults applied |
| Field constraints | Min/max, regex patterns, enums |

### `domain/`

| Area | What to Test |
|------|--------------|
| Verification logic | Pass/fail determination (without LLM) |
| Trait evaluation rules | Boolean, scored, regex matching |
| Template validation | Schema correctness, method signatures |
| Result calculations | Aggregation formulas, edge cases |

### `storage/`

| Area | What to Test |
|------|--------------|
| JSON-LD serialization | Correct structure, context handling |
| Checkpoint validation | Schema compliance, required fields |
| Data integrity | Round-trip consistency |

### `utils/`

| Area | What to Test |
|------|--------------|
| Fuzzy matching | Exact match, partial match, no match |
| Text processing | Normalization, tokenization, cleaning |
| Hash utilities | Determinism, collision handling |

---

## Directory Structure

```
tests/unit/
├── benchmark/
│   ├── test_question_management.py
│   ├── test_benchmark_filtering.py
│   └── test_result_aggregation.py
├── schemas/
│   ├── test_answer_template_validation.py
│   ├── test_rubric_schema.py
│   └── test_checkpoint_schema.py
├── domain/
│   ├── test_verification_logic.py
│   ├── test_trait_evaluation.py
│   └── test_template_parsing.py
├── storage/
│   ├── test_jsonld_serialization.py
│   └── test_checkpoint_validation.py
└── utils/
    ├── test_fuzzy_match.py
    ├── test_text_processing.py
    └── test_hash_utils.py
```

---

## Edge Cases to Always Cover

| Category | Examples |
|----------|----------|
| Empty inputs | `""`, `[]`, `{}`, `None` |
| Boundary values | `0`, `1`, `max_int`, `max+1` |
| Invalid types | String where int expected |
| Unicode | Non-ASCII characters, emoji, RTL text |
| Whitespace | Leading, trailing, multiple spaces |
| Special characters | Quotes, backslashes, newlines |

---

*Last updated: 2025-01-11*
