# Checkpoints: The Memory of Evaluation

While a **Benchmark** is the logical package for your evaluation, a **Checkpoint** is its physical reality. It is the "Record of Truth": a single, portable file that captures the complete state of a benchmark so it can be shared, version-controlled, and reproduced exactly in any environment.

Think of a checkpoint as the **Memory** of your evaluation. It doesn't just store questions; it stores the precise logic, quality standards, and provenance that define *why* a result is a pass or a fail.

## 1. The "Record of Truth" Philosophy

Karenina uses checkpoints to solve the "it works on my machine" problem in LLM evaluation. A checkpoint is designed to be:

*   **Self-Contained**: It includes the actual Python source code of your [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb). You don't need a central repository to run a checkpoint; the logic travels with the data.
*   **Human-Readable**: Even though it's a machine-interpretable format, you can open a checkpoint in any text editor and understand exactly what is being evaluated.
*   **Semantically Rich**: By using **JSON-LD**, we anchor our evaluation data in the global [Schema.org](https://schema.org) standard, making your benchmarks interoperable with other AI safety and evaluation tools.

## 2. Anatomy of a Checkpoint

A checkpoint organizes your benchmark into a clear, nested hierarchy. When you look inside, you are seeing a snapshot of the **Four Pillars**:

1.  **Benchmark Metadata**: The identity (name, version, creator) and the timeline (when it was born and last modified).
2.  **The Global Standards**: [Rubric traits](../rubrics/index.md) that apply to every question in the set.
3.  **The Questions**: A collection of [Question objects](../questions.md), each wrapped in a unique identity.
4.  **The Local Logic**: The specific [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb) and question-specific rubrics attached to individual prompts.

```
┌───────────────────────────────────────────────────────────┐
│               DataFeed (The Benchmark Root)               │
│                                                           │
│   Identity Metadata             Global Rubric Traits      │
│   (Name, Version, Creator)     (Safety, Conciseness)      │
│                                                           │
│   ┌───────────────────────────────────────────────────┐   │
│   │           DataFeedItems (The Questions)           │   │
│   │                                                   │   │
│   │   ┌─────────────┐   ┌─────────────┐               │   │
│   │   │ Question 1  │   │ Question 2  │   ...         │   │
│   │   │ (and ID)    │   │ (and ID)    │               │   │
│   │   └──────┬──────┘   └──────┬──────┘               │   │
│   │          │                 │                      │   │
│   │   ┌──────▼─────────────────▼──────────────────┐   │   │
│   │   │           Inside each Question            │   │   │
│   │   │                                           │   │   │
│   │   │  - Answer Template (Python source)        │   │   │
│   │   │  - Question-Specific Rubrics              │   │   │
│   │   │  - Local Metadata (Author, Sources)       │   │   │
│   │   └───────────────────────────────────────────┘   │   │
│   └───────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

## 3. The Journey of a Checkpoint

### 3.1. Capturing State (`save`)
When you save a benchmark, Karenina serializes the in-memory Pydantic models into a clean, indented JSON-LD file. It automatically updates the "last modified" timestamp and ensures that all Python logic is safely converted to strings.

```python
from pathlib import Path

# Capture the current state of the benchmark
benchmark.save(Path("drug_target_v1.jsonld"))
```

### 3.2. Portability & Sharing
Because it's a single file, a checkpoint can be committed to Git, sent to a colleague, or archived as part of a research paper. It captures the **Definition** of the evaluation, not the results, keeping the file lightweight and focused.

### 3.3. Restoring Context (`load`)
Loading a checkpoint restores the complete evaluation context. Karenina validates the file structure, rebuilds the internal question cache, and prepares the Python templates for execution.

```python
from karenina import Benchmark
from pathlib import Path

# Restore the benchmark from a file
benchmark = Benchmark.load(Path("drug_target_v1.jsonld"))
```

## 4. Why JSON-LD?

Karenina chose **JSON-LD** (JSON for Linked Data) over plain JSON or CSV for three critical reasons:

| Benefit | Impact on Your Evaluation |
| :--- | :--- |
| **Semantic Clarity** | Explicitly defines what is a `Question`, an `Answer`, or a `Rating` using standard types. |
| **Interoperability** | Your benchmarks aren't locked into Karenina; they speak the language of the web (Schema.org). |
| **Stability** | The format versioning allows us to evolve the framework while ensuring your old benchmarks still load correctly. |

## 5. Detailed Reference: The Checkpoint Specification

For power users and tool developers, this section breaks down the technical mapping of a checkpoint file.

### 5.1. Schema.org Mapping

| Karenina Concept | Schema.org Type | Purpose |
| :--- | :--- | :--- |
| **Benchmark** | `DataFeed` | The root container for the evaluation set. |
| **Question Wrapper** | `DataFeedItem` | Holds the unique ID and membership timestamps. |
| **Prompt** | `Question` | The literal text and nested components. |
| **Reference Answer** | `Answer` | The human-readable `raw_answer`. |
| **Verification Logic** | `SoftwareSourceCode` | The Python code for the `answer_template`. |
| **Rubric Trait** | `Rating` | Qualitative assessments (global or local). |
| **Keywords** | `keywords` on `Question` | Topic labels for categorization (native schema.org property). |
| **Metadata** | `PropertyValue` | Arbitrary key-value pairs (notes, author, sources, etc.). |

### 5.2. Deterministic IDs
Question IDs in a checkpoint are content-addressable fingerprints. They are generated using an MD5 hash of the question text:
`urn:uuid:question-{readable-prefix}-{8-char-hash}`

This ensures that the same question text always produces the same identity across any checkpoint file.

### 5.3. The `@context` Block

The `@context` tells JSON-LD processors how to interpret property names. Karenina's canonical context:

```json
{
  "@context": {
    "@version": 1.1,
    "@vocab": "https://schema.org/",
    "karenina": "urn:karenina:vocab:",
    "dataFeedElement": { "@id": "dataFeedElement", "@container": "@set" },
    "item": { "@id": "item", "@type": "@id" },
    "acceptedAnswer": { "@id": "acceptedAnswer", "@type": "@id" },
    "rating": { "@id": "contentRating", "@container": "@set" },
    "additionalProperty": { "@id": "additionalProperty", "@container": "@set" },
    "keywords": { "@id": "keywords", "@container": "@set" }
  }
}
```

Key points:

*   **`@vocab`** maps all unqualified terms to `https://schema.org/`. Only entries that add semantic information (container types, ID references, or remappings) are included explicitly.
*   **`karenina`** defines a namespace prefix for Karenina-specific vocabulary. All `additionalType` values on `Rating` objects use this prefix (e.g., `karenina:GlobalRubricTrait`, `karenina:QuestionSpecificRegexTrait`).
*   **`rating` → `contentRating`** remaps the JSON key `rating` to schema.org's [`contentRating`](https://schema.org/contentRating) property, which is the valid property on `CreativeWork` for accepting `Rating` values.

### 5.4. The `karenina:` Vocabulary Namespace

Rubric traits are stored as `Rating` objects with an `additionalType` that identifies the trait kind and scope. All values use the `karenina:` namespace prefix:

| `additionalType` | Trait Type | Scope |
| :--- | :--- | :--- |
| `karenina:GlobalRubricTrait` | LLM (boolean/score) | Global |
| `karenina:GlobalLLMRubricTrait` | LLM (literal) | Global |
| `karenina:GlobalRegexTrait` | Regex | Global |
| `karenina:GlobalCallableTrait` | Callable | Global |
| `karenina:GlobalMetricRubricTrait` | Metric | Global |
| `karenina:QuestionSpecificRubricTrait` | LLM (boolean/score) | Per-question |
| `karenina:QuestionSpecificLLMRubricTrait` | LLM (literal) | Per-question |
| `karenina:QuestionSpecificRegexTrait` | Regex | Per-question |
| `karenina:QuestionSpecificCallableTrait` | Callable | Per-question |
| `karenina:QuestionSpecificMetricRubricTrait` | Metric | Per-question |

Old checkpoints without the `karenina:` prefix are normalized automatically on load.

### 5.5. Example Structure (Annotated JSON-LD)

```json
{
  "@context": { "..." : "see above" },
  "@type": "DataFeed",
  "name": "Documentation Test Benchmark",
  "version": "1.0.0",
  "rating": [
    {
      "@type": "Rating",
      "name": "safety",
      "description": "Response is safe and appropriate",
      "bestRating": 1.0,
      "worstRating": 0.0,
      "additionalType": "karenina:GlobalRubricTrait"
    }
  ],
  "dataFeedElement": [
    {
      "@type": "DataFeedItem",
      "@id": "urn:uuid:question-what-is-the-capital-of-france-cb0b4aaf",
      "item": {
        "@type": "Question",
        "text": "What is the capital of France?",
        "keywords": ["geography", "europe"],
        "acceptedAnswer": { "@type": "Answer", "text": "Paris" },
        "hasPart": {
          "@type": "SoftwareSourceCode",
          "text": "class Answer(BaseAnswer): ...",
          "programmingLanguage": "Python"
        }
      }
    }
  ]
}
```

## 6. Next Steps

*   [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb): Understanding how the code inside a checkpoint is executed.
*   [Rubrics](../rubrics/index.md): How different trait types are represented as `Rating` objects.
*   [Evaluation Modes](../../notebooks/core_concepts/evaluation-modes.ipynb): How to run the evaluation defined in your checkpoint.
*   [Creating Benchmarks](../../workflows/creating-benchmarks/index.md): Step-by-step guides for building your first checkpoint.
