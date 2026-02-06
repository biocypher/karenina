# Checkpoints

Checkpoints are the file format Karenina uses to persist benchmarks. A checkpoint captures the complete state of a benchmark — questions, answer templates, rubric traits, and metadata — in a single portable file.

## What Is a Checkpoint?

A checkpoint is a **JSON-LD file** (`.jsonld`) that stores an entire benchmark. JSON-LD (JSON for Linked Data) extends standard JSON with semantic annotations from [Schema.org](https://schema.org/), making checkpoint files both human-readable and machine-interpretable.

**What a checkpoint contains:**

- Benchmark metadata (name, description, version, creator, timestamps)
- All questions with their text and expected answers
- Answer templates (Python code stored as strings)
- Rubric traits (global and question-specific)
- Question metadata (author, sources, custom properties)

**What a checkpoint does not contain:**

- Verification results (stored separately in the database or exported)
- Model configurations (defined at runtime via `VerificationConfig`)

## Why JSON-LD?

Karenina uses JSON-LD rather than plain JSON for several reasons:

| Benefit | Description |
|---------|-------------|
| **Semantic clarity** | Schema.org types (`Question`, `Answer`, `Rating`) make data relationships explicit |
| **Human-readable** | Standard JSON — open in any text editor |
| **Portable** | Works across environments, no database required |
| **Version-controlled** | Diff-friendly for Git tracking |
| **Backward-compatible** | Format versioning allows graceful evolution |

## Anatomy of a Checkpoint

A checkpoint follows a nested structure using Schema.org types. Here is the hierarchy:

```
DataFeed (benchmark root)
├── name, description, version, creator
├── dateCreated, dateModified
├── rating[]                          ← Global rubric traits
├── additionalProperty[]              ← Benchmark metadata
│   └── benchmark_format_version
└── dataFeedElement[]                 ← Questions
    └── DataFeedItem
        ├── @id                       ← Deterministic question ID
        ├── dateCreated, dateModified
        └── item: Question
            ├── text                  ← Question text
            ├── acceptedAnswer: Answer
            │   └── text              ← Expected answer
            ├── hasPart: SoftwareSourceCode
            │   ├── text              ← Template Python code
            │   └── programmingLanguage: "Python"
            ├── rating[]              ← Question-specific rubric traits
            └── additionalProperty[]  ← Question metadata (finished, author, sources)
```

### Schema.org Types Used

| Type | Purpose | Location |
|------|---------|----------|
| `DataFeed` | Benchmark container (root) | Top level |
| `DataFeedItem` | Question wrapper with timestamps and ID | `dataFeedElement[]` |
| `Question` | Question text and nested components | `DataFeedItem.item` |
| `Answer` | Expected answer text | `Question.acceptedAnswer` |
| `SoftwareSourceCode` | Answer template Python code | `Question.hasPart` |
| `Rating` | Rubric trait (all types) | `DataFeed.rating[]` or `Question.rating[]` |
| `PropertyValue` | Key-value metadata | `additionalProperty[]` at any level |

## Example Checkpoint Structure

Here is a simplified checkpoint showing one question with a template and a rubric trait:

```json
{
  "@context": {
    "@version": 1.1,
    "@vocab": "http://schema.org/",
    "DataFeed": "DataFeed",
    "DataFeedItem": "DataFeedItem",
    "Question": "Question",
    "Answer": "Answer",
    "SoftwareSourceCode": "SoftwareSourceCode",
    "Rating": "Rating",
    "PropertyValue": "PropertyValue",
    "dataFeedElement": {"@id": "dataFeedElement", "@container": "@set"},
    "rating": {"@id": "rating", "@container": "@set"},
    "additionalProperty": {"@id": "additionalProperty", "@container": "@set"}
  },
  "@type": "DataFeed",
  "@id": "urn:uuid:karenina-checkpoint-1770332819.237824",
  "name": "Documentation Test Benchmark",
  "description": "Sample benchmark with 5 questions",
  "version": "1.0.0",
  "creator": "Karenina Documentation",
  "dateCreated": "2026-02-05T23:06:59.237819",
  "dateModified": "2026-02-05T23:06:59.238117",
  "dataFeedElement": [
    {
      "@type": "DataFeedItem",
      "@id": "urn:uuid:question-what-is-the-capital-of-france-cb0b4aaf",
      "dateCreated": "2026-02-05T23:06:59.237875",
      "dateModified": "2026-02-05T23:06:59.237926",
      "item": {
        "@type": "Question",
        "text": "What is the capital of France?",
        "acceptedAnswer": {
          "@type": "Answer",
          "text": "Paris"
        },
        "hasPart": {
          "@type": "SoftwareSourceCode",
          "name": "What is the capital of France?... Answer Template",
          "text": "from pydantic import Field\nfrom karenina.schemas.entities import BaseAnswer\n\nclass Answer(BaseAnswer):\n    capital: str = Field(description=\"The capital city\")\n\n    def model_post_init(self, __context):\n        self.correct = {\"capital\": \"Paris\"}\n\n    def verify(self) -> bool:\n        return self.capital.strip().lower() == self.correct[\"capital\"].lower()\n",
          "programmingLanguage": "Python",
          "codeRepository": "karenina-benchmarks"
        },
        "rating": [
          {
            "@type": "Rating",
            "name": "conciseness",
            "description": "Is the response concise and to the point?",
            "bestRating": 1.0,
            "worstRating": 0.0,
            "additionalType": "QuestionSpecificRubricTrait",
            "additionalProperty": [
              {"@type": "PropertyValue", "name": "higher_is_better", "value": true}
            ]
          }
        ],
        "additionalProperty": [
          {"@type": "PropertyValue", "name": "finished", "value": true}
        ]
      }
    }
  ],
  "additionalProperty": [
    {"@type": "PropertyValue", "name": "benchmark_format_version", "value": "3.0.0-jsonld"}
  ]
}
```

## How Questions Are Stored

Each question lives inside a `DataFeedItem` wrapper:

- **Question text** — `Question.text`: The question posed to the LLM
- **Expected answer** — `Question.acceptedAnswer.text`: The ground-truth answer
- **Template** — `Question.hasPart` (`SoftwareSourceCode`): The full Python class definition as a string, stored in the `text` field
- **Rubric traits** — `Question.rating[]`: Question-specific rubric traits as `Rating` objects
- **Metadata** — `Question.additionalProperty[]`: Key-value pairs like `finished` (bool), `author` (JSON string), `sources` (JSON string)

### Deterministic Question IDs

Question IDs are generated deterministically from the question text using MD5 hashing:

```
urn:uuid:question-{readable-prefix}-{8-char-hash}
```

For example, "What is the capital of France?" produces:

```
urn:uuid:question-what-is-the-capital-of-france-cb0b4aaf
```

This means the same question always gets the same ID, enabling reliable cross-referencing. If a collision occurs, a counter suffix is appended (`-1`, `-2`, etc.).

## How Templates Are Stored

Answer templates are stored as `SoftwareSourceCode` objects inside `Question.hasPart`:

```json
{
  "@type": "SoftwareSourceCode",
  "name": "What is the capital of France?... Answer Template",
  "text": "from pydantic import Field\nfrom karenina.schemas.entities ...",
  "programmingLanguage": "Python",
  "codeRepository": "karenina-benchmarks"
}
```

The `text` field contains the complete Python code as a string. At runtime, Karenina executes this code to instantiate the `Answer` class for verification. Questions without templates have no `hasPart` field.

For details on writing templates, see [Answer Templates](answer-templates.md).

## How Rubric Traits Are Stored

Rubric traits are stored as `Rating` objects. The `additionalType` field distinguishes between global and question-specific scope, and between trait types:

| `additionalType` | Scope | Trait Type |
|-------------------|-------|------------|
| `GlobalRubricTrait` | Benchmark-wide | LLM (boolean/score/literal) |
| `QuestionSpecificRubricTrait` | Per-question | LLM (boolean/score/literal) |
| `GlobalRegexTrait` | Benchmark-wide | Regex pattern |
| `QuestionSpecificRegexTrait` | Per-question | Regex pattern |
| `GlobalCallableTrait` | Benchmark-wide | Custom Python function |
| `QuestionSpecificCallableTrait` | Per-question | Custom Python function |
| `GlobalMetricRubricTrait` | Benchmark-wide | Metric (precision/recall/F1) |
| `QuestionSpecificMetricRubricTrait` | Per-question | Metric (precision/recall/F1) |

**Global traits** are stored in `DataFeed.rating[]` (top level). **Question-specific traits** are stored in `Question.rating[]` (nested inside each question).

Each `Rating` stores its configuration in `additionalProperty[]` entries. For example, a regex trait stores `pattern`, `case_sensitive`, and `invert_result`:

```json
{
  "@type": "Rating",
  "name": "symbol_format",
  "description": "Response includes a chemical symbol in parentheses like (O)",
  "bestRating": 1.0,
  "worstRating": 0.0,
  "additionalType": "QuestionSpecificRegexTrait",
  "additionalProperty": [
    {"@type": "PropertyValue", "name": "pattern", "value": "\\([A-Z][a-z]?\\)"},
    {"@type": "PropertyValue", "name": "case_sensitive", "value": true},
    {"@type": "PropertyValue", "name": "invert_result", "value": false},
    {"@type": "PropertyValue", "name": "higher_is_better", "value": true}
  ]
}
```

For details on rubric trait types, see [Rubrics](rubrics/index.md).

## Saving and Loading

### Saving a Checkpoint

```python
from pathlib import Path

# Save benchmark to JSON-LD
benchmark.save(Path("my_benchmark.jsonld"))
```

When saving, Karenina:

1. Updates the `dateModified` timestamp
2. Serializes the in-memory Pydantic model to a JSON dict (using `by_alias=True` for Schema.org field names)
3. Omits null values for a cleaner file
4. Writes with 2-space indentation and Unicode support

The `save_deep_judgment_config` parameter controls whether deep judgment settings are included (default: `False`, for backward compatibility with older versions).

### Loading a Checkpoint

```python
from karenina import Benchmark
from pathlib import Path

benchmark = Benchmark.load(Path("my_benchmark.jsonld"))
```

When loading, Karenina:

1. Reads the JSON file
2. Validates the data against the `JsonLdCheckpoint` Pydantic model
3. Runs structural validation (correct types, required fields)
4. Builds an internal question cache for fast access

If the file is malformed or missing required fields, a `ValueError` is raised with a descriptive message.

## Format Version

Checkpoints include a format version in `additionalProperty`:

```json
{"@type": "PropertyValue", "name": "benchmark_format_version", "value": "3.0.0-jsonld"}
```

The current format version is `3.0.0-jsonld`. Karenina maintains backward compatibility with older checkpoint formats.

## Next Steps

- [Answer Templates](answer-templates.md) — How templates evaluate correctness
- [Rubrics](rubrics/index.md) — How rubric traits evaluate quality
- [Evaluation Modes](evaluation-modes.md) — Choose between template, rubric, or combined evaluation
- [Creating Benchmarks](../05-creating-benchmarks/index.md) — Build checkpoints from scratch
- [Saving Benchmarks](../05-creating-benchmarks/saving-benchmarks.md) — Save and share your work
