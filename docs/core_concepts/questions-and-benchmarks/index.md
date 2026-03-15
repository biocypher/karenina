# Questions and Benchmarks

A **benchmark** is Karenina's self-contained evaluation package: questions, answer templates, rubric traits, and metadata bundled into a single portable unit. A **question** is the building block inside a benchmark, carrying the text sent to the LLM and a reference answer. This page provides a structural overview; the sub-pages cover each component in depth.

## 1. Benchmark Structure

A benchmark organizes its content in a tree. Understanding this tree is the key to understanding how Karenina's pieces fit together:

```
Benchmark
├── Metadata (name, version, description, creator, timestamps)
├── Custom Properties          ← arbitrary key-value pairs at the benchmark level
├── Global Rubric Traits       ← quality checks applied to every question
└── Questions[]
    ├── Question text          ← what to ask the LLM
    ├── Expected answer        ← raw_answer: human-readable reference answer
    ├── Answer notes           ← optional free-text notes for interpreting the answer
    ├── Answer template        ← correctness verification code (Pydantic model)
    ├── Question-specific traits ← quality checks for this question only
    ├── Few-shot examples      ← optional parsing guidance for the Judge LLM
    ├── Intrinsic metadata     ← keywords, author, sources, timestamps, custom fields
    └── Registry entry         ← finished flag, date_added (benchmark membership state)
```

The sub-pages cover each layer in depth:

- [**Benchmarks**](../../notebooks/core_concepts/questions-and-benchmarks/benchmarks.ipynb): the benchmark as a package, metadata, persistence (checkpoints and database)
- [**Questions**](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb): the Question schema, deterministic IDs, `raw_answer` vs `ground_truth`, the `finished` flag
- [**Checkpoints**](checkpoints.md): the JSON-LD file format used for portable benchmark persistence

## 2. Questions: Two Layers of Data

Each question stores data at two levels: the Question object itself (text, `raw_answer`, keywords, template, rubric traits, metadata) and a membership record tracking the question's state within this benchmark (`finished` flag, `date_added`). This split exists because the same question can belong to multiple benchmarks with different membership states. See [Questions](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb) for the full field reference.

## 3. How Questions, Templates, and Rubrics Connect

Each question can optionally have an **answer template** and **question-specific rubric traits**. Additionally, **global rubric traits** defined at the benchmark level apply to all questions. These components are independently attachable: a question can exist without a template, without a rubric, or without both.

```
                    ┌────────────────────┐
                    │     Benchmark      │
                    │                    │
                    │  Global Rubric ────┼──── applies to ALL questions
                    └────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │ Question 1│ │ Question 2│ │ Question 3│
        │           │ │           │ │           │
        │ Template ✓│ │ Template ✓│ │ No template│
        │ Q-Rubric ✓│ │ No Q-Rubric│ │ Q-Rubric ✓│
        └───────────┘ └───────────┘ └───────────┘
```

- **Question 1**: Evaluated with its template (correctness) + global rubric + question-specific rubric (quality)
- **Question 2**: Evaluated with its template + global rubric only
- **Question 3**: No template; can only be evaluated in [`rubric_only` mode](../../notebooks/core_concepts/evaluation-modes.ipynb)

For details on what templates and rubrics *do*, see [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb), [Rubrics](../rubrics/index.md), and [Templates vs Rubrics](../../notebooks/core_concepts/template-vs-rubric.ipynb).

## 4. The `finished` Flag

Only questions marked `finished=True` enter the verification pipeline. Defaults and troubleshooting are covered in [Questions](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb#the-finished-flag).

## 5. Evaluation Modes

The benchmark's composition (which questions have templates, which have rubrics) determines which evaluation mode to use:

| Mode | Templates | Rubrics | When to Use |
|------|-----------|---------|-------------|
| `template_only` | Yes | No | Pure correctness verification (default) |
| `template_and_rubric` | Yes | Yes | Correctness + quality assessment |
| `rubric_only` | No | Yes | Quality-only evaluation (open-ended questions) |

See [Evaluation Modes](../../notebooks/core_concepts/evaluation-modes.ipynb) for the complete stage matrix and configuration details.

## 6. Definition vs Execution

The benchmark defines *what* to evaluate: which questions to ask, how to verify correctness, and what quality traits to assess. Runtime settings (which models to use, how many replicates, timeouts, caching) are specified separately in [`VerificationConfig`](../../notebooks/core_concepts/evaluation-modes.ipynb). This separation means the same benchmark can be run against different models or configurations without modification. Results are stored in the database, not inside the benchmark.

## 7. Next Steps

- [Benchmarks deep dive](../../notebooks/core_concepts/questions-and-benchmarks/benchmarks.ipynb): the benchmark as a package, metadata, persistence
- [Questions deep dive](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb): the Question schema, deterministic IDs, `raw_answer` vs `ground_truth`, the `finished` flag
- [Checkpoints](checkpoints.md): how benchmarks are persisted as JSON-LD files
- [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb): how correctness verification works
- [Rubrics](../rubrics/index.md): how quality assessment works
- [Creating Benchmarks](../../workflows/creating-benchmarks/index.md): step-by-step authoring workflow
