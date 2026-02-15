# Questions and Benchmarks

A **benchmark** is the central object in Karenina. It bundles together everything needed to evaluate LLM performance: the questions you want answered, the templates that define correctness, the rubrics that assess quality, and the metadata that makes results reproducible.

## What Is a Benchmark?

A benchmark is a self-contained evaluation unit composed of:

- **Questions** — The prompts sent to LLMs
- **Answer templates** — Code that defines how to judge correctness
- **Rubric traits** — Evaluators that assess response quality
- **Metadata** — Name, description, version, creator, timestamps

```
Benchmark
├── Metadata (name, version, creator)
├── Global Rubric Traits        ← quality checks for every question
└── Questions[]
    ├── Question text            ← what to ask the LLM
    ├── Expected answer          ← ground truth
    ├── Answer template          ← correctness verification code
    ├── Question-specific traits ← quality checks for this question only
    ├── Few-shot examples        ← optional parsing guidance
    └── Question metadata        ← author, sources, tags, custom fields
```

## What Is a Question?

A question is the smallest unit of evaluation. It contains the text that will be sent to an LLM and the expected answer used for verification.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `question` | `str` | The question text sent to the LLM |
| `raw_answer` | `str` | The ground-truth expected answer |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `tags` | `list[str]` | Category tags for filtering and organization |
| `few_shot_examples` | `list[dict]` | Example question-answer pairs for parsing guidance |

### Automatic Question ID

Each question gets a deterministic ID computed as an MD5 hash of the question text. This means the same question always gets the same ID, enabling reliable cross-referencing between benchmarks, results, and traces.

```python
# The ID is auto-generated from question text
import hashlib
question_id = hashlib.md5("What is the capital of France?".encode("utf-8")).hexdigest()
# "cb0b4aaf..."
```

## How Questions Relate to Templates and Rubrics

Each question can optionally have an **answer template** and **question-specific rubric traits** attached to it. Additionally, **global rubric traits** defined at the benchmark level apply to all questions.

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
- **Question 3**: No template — can only be evaluated in `rubric_only` mode

## The Benchmark Lifecycle

A typical workflow follows this progression:

### 1. Create

```python
from karenina.benchmark import Benchmark

benchmark = Benchmark.create(
    name="Drug Target Evaluation",
    description="Evaluate LLM accuracy on drug target identification",
    version="1.0.0",
)
```

### 2. Add Questions

```python
benchmark.add_question(
    question="What is the putative target of venetoclax?",
    raw_answer="BCL2",
    author={"name": "Dr. Smith"},
)
```

### 3. Attach Templates

```python
template_code = '''
from pydantic import Field
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    target: str = Field(
        description=(
            "The protein target of the drug as stated in the response. Use the "
            "standard gene symbol in uppercase without hyphens (e.g., 'BCL2' not "
            "'Bcl-2'). If multiple targets are mentioned, extract the primary one."
        )
    )

    def model_post_init(self, __context):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper() == self.correct["target"].upper()
'''

benchmark.add_answer_template(question_id, template_code)
```

### 4. Add Rubric Traits (Optional)

```python
from karenina.schemas import LLMRubricTrait, Rubric

rubric = Rubric(llm_traits=[
    LLMRubricTrait(
        name="Conciseness",
        description="Is the response concise and to the point?",
        kind="boolean",
        higher_is_better=True,
    ),
])

benchmark.set_global_rubric(rubric)
```

### 5. Save as Checkpoint

```python
from pathlib import Path

benchmark.save(Path("drug_targets.jsonld"))
```

### 6. Run Verification

```python
from karenina.schemas import VerificationConfig, ModelConfig

config = VerificationConfig(
    answering_models=[ModelConfig(id="answering", model_name="claude-sonnet-4-5-20250929", model_provider="anthropic")],
    parsing_models=[ModelConfig(id="parsing", model_name="claude-haiku-4-5", model_provider="anthropic")],
)

results = benchmark.run_verification(config)
```

### 7. Analyze Results

Results are returned as `VerificationResult` objects — one per question per answering model. See [Results and Scoring](results-and-scoring.md) for details on what comes out.

## The Benchmark Object

The `Benchmark` class is a **facade** that delegates to specialized managers:

| Manager | Responsibility |
|---------|---------------|
| `MetadataManager` | Benchmark name, version, timestamps |
| `QuestionManager` | Add, retrieve, and manage questions |
| `TemplateManager` | Add and retrieve answer templates |
| `RubricManager` | Add and manage rubric traits |
| `ResultsManager` | Store and retrieve verification results |
| `VerificationManager` | Run verification pipelines |
| `ExportManager` | Serialize to JSON-LD checkpoint format |

You interact with benchmarks through the `Benchmark` class — the managers are internal.

## Next Steps

- [Checkpoints](checkpoints.md) — How benchmarks are persisted as JSON-LD files
- [Answer Templates](answer-templates.md) — How correctness verification works
- [Rubrics](rubrics/index.md) — How quality assessment works
- [Creating Benchmarks](../workflows/creating-benchmarks/index.md) — Step-by-step benchmark authoring workflow
- [Evaluation Modes](evaluation-modes.md) — Choosing between template-only, rubric-only, or combined evaluation
