# Creating Benchmarks

This section walks through the complete workflow for authoring a benchmark — from creating an empty checkpoint to saving a fully populated benchmark with questions, templates, rubrics, and few-shot examples.

## Workflow Overview

```
Create checkpoint
    │
    ▼
Add questions (from dict or Question object)
    │
    ▼
Classify questions with Adele [optional]
    │
    ▼
Generate templates (automated) ─── or ─── Write custom templates
    │
    ▼
Define rubrics (global and/or question-specific)
    │
    ▼
Add few-shot examples [optional]
    │
    ▼
Save (JSON-LD checkpoint or database)
```

Each step has a dedicated page with detailed instructions and executable examples.

---

## Workflow Steps

### 1. Create a Checkpoint

Start by creating an empty benchmark with metadata (name, description, version, creator, keywords):

```python
from karenina import Benchmark

benchmark = Benchmark.create(
    name="My Benchmark",
    description="Evaluating LLM knowledge",
    version="1.0.0"
)
```

[Create a checkpoint from scratch →](creating-checkpoint.md)

### 2. Add Questions

Populate the benchmark with questions. You can add questions as simple dicts or as `Question` objects with rich metadata and sources:

```python
benchmark.add_question({
    "text": "What is the capital of France?",
    "acceptedAnswer": "Paris"
})
```

[Add questions to a benchmark →](adding-questions.md)

### 3. Classify Questions with Adele (Optional)

Use the Adele classification system to characterize questions across 18 dimensions (reasoning depth, domain specificity, answer format, etc.). Classifications guide template design and can be used for filtering.

[Classify questions with Adele →](classifying-with-adele.md)

### 4. Add Evaluation Criteria

Every question needs evaluation criteria. Karenina provides two complementary approaches:

**Generate templates automatically** — Use an LLM to produce answer templates from your questions. Review and refine the generated code.

[Generate templates automatically →](generating-templates.md)

**Write custom templates** — Define your own Pydantic models with complex verification logic, multiple attributes, and domain-specific comparisons.

[Write custom templates →](writing-templates.md)

For background on what templates are and how they work, see [Answer Templates](../04-core-concepts/answer-templates.md).

### 5. Define Rubrics

Add rubric traits to evaluate response quality beyond factual correctness. Traits can be applied globally (all questions) or per-question:

- **LLM traits** — Subjective assessment (boolean or score) via a Judge LLM
- **Literal traits** — Ordered categorical classification via a Judge LLM
- **Regex traits** — Pattern matching for format compliance
- **Callable traits** — Custom Python functions
- **Metric traits** — Precision/recall/F1 for extraction completeness

[Define rubrics for a benchmark →](defining-rubrics.md)

For background on rubric concepts, see [Rubrics](../04-core-concepts/rubrics/index.md).

### 6. Add Few-Shot Examples (Optional)

Inject example responses into the Judge LLM's parsing prompt to improve accuracy. Useful for complex or ambiguous response formats.

[Add few-shot examples →](few-shot-examples.md)

For background on few-shot configuration modes, see [Few-Shot](../04-core-concepts/few-shot.md).

### 7. Save the Benchmark

Persist the benchmark for sharing and future use:

- **JSON-LD checkpoint** — Portable file for sharing, version control, and inspection
- **Database** — SQLite storage for persistent management and querying

```python
# Save as checkpoint
benchmark.save("my_benchmark.jsonld")

# Save to database
benchmark.save_to_db(storage="sqlite:///benchmarks.db")
```

[Save benchmarks →](saving-benchmarks.md)

---

## What You Need

Not every step is required. Here's a guide based on your evaluation strategy:

| Evaluation Strategy | Required Steps | Optional Steps |
|---------------------|---------------|----------------|
| **Template-only** (correctness) | Create checkpoint, add questions, add templates, save | Adele, few-shot |
| **Template + rubric** (correctness + quality) | Create checkpoint, add questions, add templates, define rubrics, save | Adele, few-shot |
| **Rubric-only** (quality assessment) | Create checkpoint, add questions, define rubrics, save | Adele, few-shot |

See [Evaluation Modes](../04-core-concepts/evaluation-modes.md) for details on how these strategies map to pipeline behavior.

---

## Next Steps

Once your benchmark is built and saved, proceed to:

- [Running Verification](../06-running-verification/index.md) — Execute the benchmark against LLMs
- [Analyzing Results](../07-analyzing-results/index.md) — Inspect and compare verification outcomes
