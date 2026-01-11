# LLM Fixture Capture System

**Parent**: [README.md](./README.md)

---

## Overview

LLM fixtures are captured responses from real API calls, stored as JSON files for deterministic test replay. This ensures tests reflect actual model behavior rather than idealized mocks.

**Critical Rule**: Fixtures MUST be captured from actual pipeline runs, NOT manually created.

---

## Identifying LLM Call Sites

Before capturing fixtures, identify all LLM invocation points in the codebase.

### Discovery Command

```bash
# Find all LLM invoke calls
grep -rn "\.invoke\(\|\.ainvoke\(" src/karenina --include="*.py"
```

### Current Call Site Inventory

Run the discovery command and map each call to its scenario category:

| File | Method | Scenario Category |
|------|--------|-------------------|
| `template_evaluator.py:601` | `_parse_standard` | `template_parsing/` |
| `template_evaluator.py:817` | `_retry_parse_with_null_feedback` | `template_parsing/retry/` |
| `template_evaluator.py:900` | `_retry_parse_with_format_feedback` | `template_parsing/retry/` |
| `template_parsing.py:99` | structured parsing | `template_parsing/` |
| `rubric_evaluator.py:1262` | excerpt extraction | `rubric_evaluation/excerpt/` |
| `rubric_evaluator.py:1537` | reasoning generation | `rubric_evaluation/reasoning/` |
| `rubric_evaluator.py:1609` | score extraction | `rubric_evaluation/score/` |
| `rubric_evaluator.py:1731` | (additional evaluation) | `rubric_evaluation/` |
| `rubric_parsing.py:128,147` | rubric parsing | `rubric_evaluation/parsing/` |
| `abstention_checker.py:165` | abstention detection | `abstention/` |
| `embedding_check.py:275` | semantic equivalence | `embedding/` |
| `verification_utils.py:346,525` | retry wrapper | `verification/retry/` |
| `generator.py:329` | answer generation | `generation/` |

### Keeping Inventory Updated

When adding new LLM calls:
1. Run the discovery command
2. Add new entries to this table
3. Create corresponding fixture scenarios
4. Update MANIFEST.md

---

## Primary Model

**Model**: `claude-haiku-4-5`

**Rationale**:
- Fast response times (suitable for fixture generation)
- Low cost (enables comprehensive fixture sets)
- Representative of production behavior

---

## Fixture Format

```json
{
  "metadata": {
    "model": "claude-haiku-4-5",
    "captured_at": "2025-01-11T14:30:00Z",
    "prompt_hash": "sha256:abc123...",
    "scenario": "template_parsing/basic_extraction",
    "source_file": "template_evaluator.py",
    "source_line": 601,
    "karenina_version": "0.5.0"
  },
  "request": {
    "messages": [...],
    "max_tokens": 1024,
    "temperature": 0
  },
  "response": {
    "id": "msg_...",
    "content": [...],
    "model": "claude-haiku-4-5-20241022",
    "usage": {...}
  }
}
```

---

## Fixture Scenarios by Call Site

### Template Parsing (`template_evaluator.py`, `template_parsing.py`)

| Scenario | Call Site | Description |
|----------|-----------|-------------|
| `basic_extraction` | `_parse_standard` | Single-field schema, clean extraction |
| `complex_schema` | `_parse_standard` | Nested fields, multiple types |
| `malformed_json` | `_parse_standard` | LLM returns invalid JSON |
| `partial_extraction` | `_parse_standard` | Some fields filled, others null |
| `refusal` | `_parse_standard` | LLM refuses ("I cannot...") |
| `null_retry_success` | `_retry_parse_with_null_feedback` | Retry fills null fields |
| `null_retry_still_null` | `_retry_parse_with_null_feedback` | Retry fails to fill |
| `format_retry_success` | `_retry_parse_with_format_feedback` | Format correction works |
| `format_retry_fail` | `_retry_parse_with_format_feedback` | Format still wrong |

### Rubric Evaluation - Excerpt Extraction (`rubric_evaluator.py:1262`)

| Scenario | Description |
|----------|-------------|
| `excerpt_found` | Relevant excerpts extracted |
| `excerpt_empty` | No relevant excerpts in answer |
| `excerpt_validation_fail` | Excerpts fail validation, triggers auto-fail |
| `excerpt_multiple` | Multiple distinct excerpts |

### Rubric Evaluation - Reasoning Generation (`rubric_evaluator.py:1537`)

| Scenario | Description |
|----------|-------------|
| `reasoning_clear` | Clear reasoning with conclusion |
| `reasoning_ambiguous` | Ambiguous reasoning |
| `reasoning_contradicts` | Reasoning contradicts final score |

### Rubric Evaluation - Score Extraction (`rubric_evaluator.py:1609,1731`)

| Scenario | Description |
|----------|-------------|
| `score_valid` | Score within expected range |
| `score_boundary_low` | Score at minimum (e.g., 1/5) |
| `score_boundary_high` | Score at maximum (e.g., 5/5) |
| `score_out_of_range` | Score outside range (0 or 6 on 1-5) |
| `score_bool_true` | Boolean trait → True |
| `score_bool_false` | Boolean trait → False |

### Rubric Parsing (`rubric_parsing.py`)

| Scenario | Description |
|----------|-------------|
| `rubric_parse_success` | Valid rubric definition parsed |
| `rubric_parse_invalid` | Invalid rubric structure |

### Abstention Detection (`abstention_checker.py`)

| Scenario | Description |
|----------|-------------|
| `abstention_detected` | Model abstained ("I don't know") |
| `abstention_not_detected` | Model gave definitive answer |
| `abstention_hedging` | Model hedged but answered |
| `abstention_parse_fail` | Response not parseable |

### Embedding Check (`embedding_check.py`)

| Scenario | Description |
|----------|-------------|
| `semantic_equivalent` | Answers semantically match |
| `semantic_different` | Answers semantically differ |
| `semantic_parse_fail` | Parsing for comparison failed |

### Answer Generation (`generator.py`)

| Scenario | Description |
|----------|-------------|
| `generation_success` | Model generates valid answer |
| `generation_refusal` | Model refuses to answer |
| `generation_truncated` | Response cut off |

### Error Cases (Cross-cutting)

| Scenario | Source | Description |
|----------|--------|-------------|
| `rate_limit` | Synthesized | 429 error response |
| `context_exceeded` | Synthesized | Token limit exceeded |
| `timeout` | Synthesized | Request timeout |
| `auth_error` | Synthesized | Invalid API key |

---

## Capture Procedure

### Step 1: Run Discovery

```bash
grep -rn "\.invoke\(\|\.ainvoke\(" src/karenina --include="*.py"
```

Compare output to the inventory table above. Add any new call sites.

### Step 2: Capture Fixtures

```bash
# Capture for specific scenario
python scripts/capture_pipeline_fixtures.py \
    --scenario template_parsing \
    --output tests/fixtures/llm_responses/

# Capture all
python scripts/capture_pipeline_fixtures.py --all

# List available scenarios
python scripts/capture_pipeline_fixtures.py --list
```

### Step 3: Validate and Update Manifest

The capture script automatically updates `tests/fixtures/MANIFEST.md`.

---

## Missing Fixture Behavior

When a test requires a missing fixture:

```python
ValueError: No fixture for prompt hash: sha256:abc123...
Source: template_evaluator.py:601 (_parse_standard)
Run: python scripts/capture_pipeline_fixtures.py --scenario template_parsing
```

---

## Directory Structure

```
tests/fixtures/
├── README.md
├── MANIFEST.md
├── llm_responses/
│   └── claude-haiku-4-5/
│       ├── template_parsing/
│       │   ├── basic_extraction.json
│       │   ├── complex_schema.json
│       │   ├── malformed_json.json
│       │   ├── partial_extraction.json
│       │   ├── refusal.json
│       │   └── retry/
│       │       ├── null_retry_success.json
│       │       ├── null_retry_still_null.json
│       │       ├── format_retry_success.json
│       │       └── format_retry_fail.json
│       ├── rubric_evaluation/
│       │   ├── excerpt/
│       │   │   ├── found.json
│       │   │   ├── empty.json
│       │   │   ├── validation_fail.json
│       │   │   └── multiple.json
│       │   ├── reasoning/
│       │   │   ├── clear.json
│       │   │   ├── ambiguous.json
│       │   │   └── contradicts.json
│       │   ├── score/
│       │   │   ├── valid.json
│       │   │   ├── boundary_low.json
│       │   │   ├── boundary_high.json
│       │   │   ├── out_of_range.json
│       │   │   ├── bool_true.json
│       │   │   └── bool_false.json
│       │   └── parsing/
│       │       ├── success.json
│       │       └── invalid.json
│       ├── abstention/
│       │   ├── detected.json
│       │   ├── not_detected.json
│       │   ├── hedging.json
│       │   └── parse_fail.json
│       ├── embedding/
│       │   ├── semantic_equivalent.json
│       │   ├── semantic_different.json
│       │   └── parse_fail.json
│       ├── generation/
│       │   ├── success.json
│       │   ├── refusal.json
│       │   └── truncated.json
│       └── error_cases/
│           ├── rate_limit.json
│           ├── context_exceeded.json
│           ├── timeout.json
│           └── auth_error.json
├── checkpoints/
│   ├── minimal.jsonld
│   ├── with_results.jsonld
│   └── complex_benchmark.jsonld
└── templates/
    ├── simple_extraction.py
    ├── multi_field.py
    └── with_regex_trait.py
```

---

## Regeneration

```bash
# Regenerate specific category
python scripts/capture_pipeline_fixtures.py --scenario rubric_evaluation --force

# Regenerate all
python scripts/capture_pipeline_fixtures.py --all --force
```

---

*Last updated: 2025-01-11*
