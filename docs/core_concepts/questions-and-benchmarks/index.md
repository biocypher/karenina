# Questions and Benchmarks

A **benchmark** is Karenina's central object: think of it as a sealed envelope containing everything needed to reproduce an evaluation. It bundles questions, expected answers, verification logic, and quality checks into a single portable package. A **question** is the basic building block inside a benchmark, carrying the text sent to the LLM and a reference answer.

This page provides a conceptual overview and links to the deep dives on each component. If this is your first time here, we recommend reading [Benchmarks](../../notebooks/core_concepts/questions-and-benchmarks/benchmarks.ipynb) first (the container), then [Questions](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb) (what goes inside). For step-by-step authoring guides, see [Creating Benchmarks](../../workflows/creating-benchmarks/index.md).

## Benchmark Structure

A benchmark organizes its content in a tree. Understanding this tree is the key to understanding how Karenina's pieces fit together:

```
Benchmark
├── Metadata (name, version, description, creator, timestamps)
├── Custom Properties          ← arbitrary key-value pairs at the benchmark level
├── Global Rubric Traits       ← quality checks applied to every question
└── Questions[]
    ├── Question text          ← what to ask the LLM
    ├── Expected answer        ← raw_answer: human-readable reference answer
    ├── Answer template        ← correctness verification code (Pydantic model)
    ├── Question-specific traits ← quality checks for this question only
    ├── Few-shot examples      ← optional parsing guidance for the Judge LLM
    ├── Intrinsic metadata     ← keywords, author, sources, timestamps, custom fields
    └── Registry entry         ← finished flag, date_added (benchmark membership state)
```

The sub-pages cover each layer in depth:

- [**Benchmarks**](../../notebooks/core_concepts/questions-and-benchmarks/benchmarks.ipynb): the benchmark as a package, metadata, persistence (checkpoints and database)
- [**Questions**](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb): the Question schema, deterministic IDs, `raw_answer` vs `ground_truth`, the `finished` flag

## Questions: Two Layers of Data

Each question stores data at two levels:

- **The Question object** carries everything intrinsic to the question itself: the question text, `raw_answer`, keywords, optional template code, optional rubric traits, and metadata like author and sources.
- **The membership record** tracks the question's state within this benchmark: whether it is marked `finished` (ready for the [verification pipeline](../verification-pipeline.md)) and when it was added.

This split exists because the same question can belong to multiple benchmarks. It might be finalized in a published benchmark but still under review in a draft.

See [Questions](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb) for the full field reference, deterministic IDs, and the `finished` flag.

## How Questions, Templates, and Rubrics Connect

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
- **Question 3**: No template; can only be evaluated in [`rubric_only` mode](../evaluation-modes.md)

For details on what templates and rubrics *do*, see [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb), [Rubrics](../rubrics/index.md), and [Templates vs Rubrics](../template-vs-rubric.md).

## The `finished` Flag

Every question in a benchmark has a `finished` flag that gates pipeline entry: only finished questions are included when verification runs. The default varies by interface (the Python API defaults to `True`; the GUI defaults to `False`). See [Questions](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb) for the full details, defaults, and troubleshooting.

## Evaluation Modes

The benchmark's composition (which questions have templates, which have rubrics) determines which evaluation mode to use:

| Mode | Templates | Rubrics | When to Use |
|------|-----------|---------|-------------|
| `template_only` | Yes | No | Pure correctness verification (default) |
| `template_and_rubric` | Yes | Yes | Correctness + quality assessment |
| `rubric_only` | No | Yes | Quality-only evaluation (open-ended questions) |

See [Evaluation Modes](../evaluation-modes.md) for the complete stage matrix and configuration details.

## Next Steps

- [Benchmarks deep dive](../../notebooks/core_concepts/questions-and-benchmarks/benchmarks.ipynb): the benchmark as a package, metadata, persistence
- [Questions deep dive](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb): the Question schema, deterministic IDs, `raw_answer` vs `ground_truth`, the `finished` flag
- [Checkpoints](../checkpoints.md): how benchmarks are persisted as JSON-LD files
- [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb): how correctness verification works
- [Rubrics](../rubrics/index.md): how quality assessment works
- [Creating Benchmarks](../../workflows/creating-benchmarks/index.md): step-by-step authoring workflow
