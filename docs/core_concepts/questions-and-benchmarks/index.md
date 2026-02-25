# Questions and Benchmarks

A **benchmark** is the central object in Karenina: a self-contained evaluation unit that bundles questions, [answer templates](../answer-templates.md), [rubric traits](../rubrics/index.md), and metadata into a portable, versioned package. A **question** is the atomic unit within a benchmark, carrying the text sent to the LLM and the reference answer used for [verification](../verification-pipeline.md).

This page provides a conceptual overview and links to the deep dives on each component. For step-by-step authoring guides, see [Creating Benchmarks](../../workflows/creating-benchmarks/index.md).

## Benchmark Structure

A benchmark organizes its content in a tree. Understanding this tree is the key to understanding how Karenina's pieces fit together:

```
Benchmark
├── Metadata (name, version, description, creator, timestamps)
├── Custom Properties          ← arbitrary key-value pairs at the benchmark level
├── Global Rubric Traits       ← quality checks applied to every question
└── Questions[]
    ├── Question text          ← what to ask the LLM
    ├── Expected answer        ← raw_answer: human-readable ground truth
    ├── Answer template        ← correctness verification code (Pydantic model)
    ├── Question-specific traits ← quality checks for this question only
    ├── Few-shot examples      ← optional parsing guidance for the Judge LLM
    └── Question metadata      ← finished flag, author, sources, timestamps, custom fields
```

The sub-pages cover each layer in depth:

- [**Benchmarks**](../../notebooks/core_concepts/questions-and-benchmarks/benchmarks.ipynb): the facade, managers, readiness checking, default templates, filtering
- [**Questions**](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb): the Question schema, deterministic IDs, `raw_answer`, metadata layers, the `finished` flag

## The Question: Two Layers of Data

A question has a **lightweight Pydantic model** and a **rich benchmark cache entry**. This split is intentional:

| Layer | Fields | Purpose |
|-------|--------|---------|
| `Question` object | `question`, `raw_answer`, `tags`, [`few_shot_examples`](../few-shot.md), `id` (computed) | Portable, standalone; can exist outside a benchmark |
| Benchmark cache | `finished`, `author`, `sources`, `custom_metadata`, `date_created`, `date_modified`, `answer_template`, `question_rubric` | Rich metadata that only makes sense in a benchmark context |

When you call `benchmark.add_question(...)`, the benchmark creates a cache entry that wraps the Question data with these additional fields. To access the full picture, use `benchmark.get_question_metadata(question_id)`.

See [Questions](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb) for the full details on both layers.

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

Every question in a benchmark has a `finished` boolean that gates pipeline entry. The verification pipeline only processes finished questions.

The default value depends on the interface:

| Interface | Default `finished` | Rationale |
|-----------|-------------------|-----------|
| Python API (`add_question()`) | `True` | Power users adding questions programmatically are assumed to have complete data |
| GUI (karenina-gui) | `False` | Prompts the user to review and complete template authoring before running verification |

If verification produces zero results, check `finished` status first. See [Questions](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb) for examples of managing the `finished` flag.

## Evaluation Modes

The benchmark's composition (which questions have templates, which have rubrics) determines which evaluation mode to use:

| Mode | Templates | Rubrics | When to Use |
|------|-----------|---------|-------------|
| `template_only` | Yes | No | Pure correctness verification (default) |
| `template_and_rubric` | Yes | Yes | Correctness + quality assessment |
| `rubric_only` | No | Yes | Quality-only evaluation (open-ended questions) |

See [Evaluation Modes](../evaluation-modes.md) for the complete stage matrix and configuration details.

## Next Steps

- [Benchmarks deep dive](../../notebooks/core_concepts/questions-and-benchmarks/benchmarks.ipynb): the facade pattern, default templates, readiness checking, filtering
- [Questions deep dive](../../notebooks/core_concepts/questions-and-benchmarks/questions.ipynb): the Question schema, deterministic IDs, `raw_answer` vs `ground_truth`, metadata layers
- [Checkpoints](../checkpoints.md): how benchmarks are persisted as JSON-LD files
- [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb): how correctness verification works
- [Rubrics](../rubrics/index.md): how quality assessment works
- [Creating Benchmarks](../../workflows/creating-benchmarks/index.md): step-by-step authoring workflow
