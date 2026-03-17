# Test Fixtures Manifest

This document catalogs all test fixtures in the `tests/fixtures/` directory.

**Status**: 🚧 Partial - Checkpoint and template fixtures are available. LLM response fixtures require API key capture.

## Fixtures Overview

| Category | Status | Count |
|----------|--------|-------|
| Checkpoint fixtures | ✅ Complete | 3 |
| Template fixtures | ✅ Complete | 3 |
| LLM response fixtures | ⚠️ Pending API keys | 0 |

---

## Checkpoint Fixtures (`tests/fixtures/checkpoints/`)

Checkpoint files are JSON-LD formatted benchmarks used for testing verification workflows.

### `minimal.jsonld`
- **Description**: Minimal checkpoint with a single arithmetic question
- **Questions**: 1
- **Content**: "What is 2+2?"
- **Template**: Single-field integer extraction
- **Use Case**: Basic CLI verification tests, quick smoke tests

### `multi_question.jsonld`
- **Description**: Benchmark with 5 diverse questions for comprehensive testing
- **Questions**: 5
- **Content**: Arithmetic, geography, science, literature, logic questions
- **Templates**: Various extraction patterns
- **Use Case**: Batch operations, filtering, progress reporting tests

### `with_results.jsonld`
- **Description**: Checkpoint with existing verification results
- **Questions**: 1
- **Content**: Capital city question with completed verification
- **Use Case**: Resume functionality, incremental verification testing

---

## Template Fixtures (`tests/fixtures/templates/`)

Template files define Pydantic `Answer` classes with `verify()` methods for testing answer evaluation.

### `simple_extraction.py`
- **Description**: Single-field string extraction template
- **Fields**: `value: str`
- **Ground Truth**: `"42"`
- **Pattern**: Basic field extraction with direct comparison
- **Use Case**: Testing simple template parsing and verification

### `multi_field.py`
- **Description**: Multi-field template with nested structures and complex types
- **Fields**:
  - `main_answer: str` - Primary answer text
  - `confidence: float` - Score from 0.0 to 1.0
  - `keywords: list[str]` - Relevant keywords
  - `entities: list[str]` - Named entities
  - `citation: Citation` - Nested citation object (identifier, page, url)
  - `disclaimer: str | None` - Optional caveat
- **Ground Truth**: Mitochondria/cell biology answer
- **Pattern**: Complex validation with nested objects and collections
- **Use Case**: Testing multi-field parsing, nested structures, optional fields

### `with_correct_dict.py`
- **Description**: Template using `correct_dict` pattern for ground truth management
- **Fields**:
  - `gene_name: str` - Gene symbol
  - `chromosome: str` - Chromosome location
  - `function: str` - Gene function description
  - `synonyms: list[str] | None` - Alternative symbols
  - `omim_id: int | None` - OMIM database ID
- **Ground Truth**: `BCL2` gene information
- **Pattern**: `model_post_init` populates `self.correct` dictionary
- **Use Case**: Testing flexible ground truth management, fuzzy matching

---

## LLM Response Fixtures (`tests/fixtures/llm_responses/`)

> **⚠️ Status**: Empty - Requires `ANTHROPIC_API_KEY` to capture.

LLM response fixtures contain captured API responses for deterministic test replay without making live API calls.

### Directory Structure
```
llm_responses/
└── claude-haiku-4-5/
    ├── template_parsing/    (empty - pending capture)
    ├── rubric_evaluation/   (empty - pending capture)
    ├── abstention/          (empty - pending capture)
    ├── generation/          (empty - pending capture)
    └── embedding/           (empty - pending capture)
```

### How to Capture Fixtures

Run the capture script with an API key:

```bash
# Capture all scenarios
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/capture_fixtures.py --all

# Capture specific scenario
python scripts/capture_fixtures.py --scenario template_parsing

# List available scenarios
python scripts/capture_fixtures.py --list
```

### Expected Fixture Format

Each fixture JSON file contains:
```json
{
  "metadata": {
    "scenario": "template_parsing",
    "model": "claude-haiku-4-5",
    "captured_at": "2025-01-11T12:00:00Z",
    "prompt_hash": "abc123..."
  },
  "request": {
    "messages": [...],
    "kwargs": {...}
  },
  "response": {
    "content": "...",
    "id": "...",
    "model": "claude-haiku-4-5",
    "usage": {...}
  }
}
```

---

## Usage in Tests

### Checkpoint Fixtures
```python
from tests.conftest import fixtures_dir

minimal_checkpoint = fixtures_dir / "checkpoints" / "minimal.jsonld"
benchmark = Benchmark.load(minimal_checkpoint)
```

### Template Fixtures
```python
from karenina.schemas.entities import load_template_module
from pathlib import Path

template_module = load_template_module(
    Path("tests/fixtures/templates/simple_extraction.py")
)
Answer = template_module.Answer
```

### LLM Response Fixtures (when captured)
```python
from tests.conftest import llm_client

response = llm_client.invoke(messages=[...])
# Returns captured response without API call
```

---

## Adding New Fixtures

### Checkpoint Fixtures
1. Create JSON-LD file in `tests/fixtures/checkpoints/`
2. Follow JSON-LD schema with `@type: DataFeed`
3. Add at least one question with template
4. Update this MANIFEST

### Template Fixtures
1. Create `.py` file in `tests/fixtures/templates/`
2. Define `Answer(BaseAnswer)` class
3. Implement `verify()` method
4. Set ground truth in `model_post_init()`
5. Update this MANIFEST

### LLM Response Fixtures
1. Run `scripts/capture_fixtures.py --scenario <name>`
2. Fixtures auto-save to appropriate subdirectory
3. Update this MANIFEST with captured files

---

## Keeping MANIFEST in Sync

When adding, modifying, or removing fixtures:
1. Update the relevant section in this file
2. Run tests to ensure fixtures work: `pytest tests/ -k fixture`
3. Commit both fixture files and this MANIFEST together

---

*Last updated: 2025-01-11*
