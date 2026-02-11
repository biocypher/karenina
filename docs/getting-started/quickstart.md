---
jupyter:
  jupytext:
    formats: getting-started//md,notebooks//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Quick Start

Get started with Karenina in minutes. This guide walks you through creating a benchmark, adding questions, writing answer templates, defining rubric traits, running verification, and inspecting results.

By the end you will have a working benchmark that evaluates LLM responses for both **correctness** (via answer templates) and **quality** (via rubric traits).

---

## Prerequisites

- **Python 3.11+**
- **Karenina installed** (see [Installation](installation.md))
- **API keys** for the LLM providers you plan to use:

> ```bash
> export OPENAI_API_KEY="sk-..."
> export ANTHROPIC_API_KEY="sk-ant-..."
> export GOOGLE_API_KEY="AI..."
> ```

---

```python tags=["hide-cell"]
# Mock cell: replays captured LLM responses from docs/data/quickstart/ so the
# quickstart executes without live API keys. The full pipeline logic runs;
# only the raw model calls are mocked.
# This cell is hidden in the rendered documentation.
import hashlib
import json
import tempfile
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

# Resolve fixtures directory (works from notebook, markdown, and repo root CWDs)
_FIXTURES_DIR = None
for _candidate in [Path("data/quickstart"), Path("../data/quickstart"), Path("docs/data/quickstart")]:
    if _candidate.is_dir():
        _FIXTURES_DIR = _candidate
        break
assert _FIXTURES_DIR is not None, "Could not find data/quickstart fixtures directory"

# Load fixtures indexed by prompt hash for order-independent matching
_fixtures_by_hash: dict[str, dict] = {}
for _p in _FIXTURES_DIR.glob("*.json"):
    _data = json.loads(_p.read_text())
    _fixtures_by_hash[_data["prompt_hash"]] = _data


def _hash_messages(messages) -> str:
    """Compute the same hash used during capture for fixture matching."""
    normalized = []
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        msg_type = msg.type if hasattr(msg, "type") else "unknown"
        if isinstance(content, str):
            normalized.append(f"{msg_type}:{content}")
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    text_parts.append(str(block.get("text", block.get("input", ""))))
                else:
                    text_parts.append(str(block))
            normalized.append(f"{msg_type}:{'|'.join(text_parts)}")
    return hashlib.sha256("|".join(normalized).encode()).hexdigest()[:16]


# Save original ainvoke for restoration
_original_ainvoke = BaseChatModel.ainvoke


async def _replaying_ainvoke(self, input, config=None, **kwargs):
    """Return the captured LLM response matching this request's prompt hash."""
    messages = input if isinstance(input, list) else [input]
    prompt_hash = _hash_messages(messages)
    fixture = _fixtures_by_hash.get(prompt_hash)
    if fixture is None:
        raise ValueError(f"No fixture for prompt hash {prompt_hash}")
    resp = fixture["response"]
    return AIMessage(
        content=resp["content"],
        id=resp.get("id", "fixture"),
        tool_calls=resp.get("tool_calls", []),
        response_metadata=resp.get("response_metadata", {}),
        usage_metadata=resp.get("usage_metadata"),
    )


BaseChatModel.ainvoke = _replaying_ainvoke

# Temp directory for save/load examples
_tmpdir = tempfile.mkdtemp()
```

## Step 1: Create a Benchmark

A benchmark is the top-level container that holds questions, answer templates, rubric traits, and verification results.

```python
from karenina import Benchmark

benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics and molecular biology",
    version="1.0.0",
    creator="Your Name",
)

print(f"Created benchmark: {benchmark.name}")
```

> **Learn more**: [Creating Checkpoints](../05-creating-benchmarks/creating-checkpoint.md) · [Core Concepts](../core_concepts/index.md)

---

## Step 2: Add Questions

Each question has a text prompt and a reference answer (the ground truth).

```python
questions = [
    {
        "question": "How many chromosomes are in a human somatic cell?",
        "answer": "46",
    },
    {
        "question": "What is the approved drug target of Venetoclax?",
        "answer": "BCL2",
    },
    {
        "question": "How many protein subunits does hemoglobin A have?",
        "answer": "4",
    },
]

question_ids = []
for q in questions:
    qid = benchmark.add_question(
        question=q["question"],
        raw_answer=q["answer"],
        author={"name": "Bio Curator", "email": "curator@example.com"},
    )
    question_ids.append(qid)

print(f"Added {len(question_ids)} questions")
```

> **Learn more**: [Adding Questions](../05-creating-benchmarks/adding-questions.md) — including bulk import from Excel, CSV, and TSV files

---

## Step 3: Write Answer Templates

Answer templates are Pydantic models that define how a Judge LLM should parse and verify a model's response. Each template:

1. Declares **attributes** the judge must extract (typed fields)
2. Stores the **correct values** in `model_post_init`
3. Implements a **`verify()`** method that compares extracted values to ground truth

The class must always be named `Answer` and inherit from `BaseAnswer`.

### Automatic Generation

The fastest way to get started is to let Karenina generate templates for you using an LLM. This analyses each question and its reference answer, then produces a complete template:

```python
benchmark.generate_all_templates(
    model="claude-haiku-4-5",
    model_provider="anthropic",
    temperature=0.0,
)

print(f"Generated templates for {benchmark.question_count} questions")
```

You can review a generated template to see what the LLM produced:

```python
generated_code = benchmark.get_template(question_ids[0])
print(generated_code)
```

### Manual Definition (Class-Based)

When you need precise control over verification logic, define templates as Python classes and pass them directly. This is especially useful for domain-specific comparisons or multi-field extraction:

```python
from pydantic import Field
from karenina.schemas.entities import BaseAnswer


class Answer(BaseAnswer):
    is_bcl2: bool = Field(
        description="Whether the response identifies BCL2 as the putative target of the drug"
    )

    def model_post_init(self, __context):
        self.correct = {"is_bcl2": True}

    def verify(self) -> bool:
        return self.is_bcl2 == self.correct["is_bcl2"]


benchmark.update_template(question_ids[1], Answer)

print(f"Updated template for Venetoclax question with class-based definition")
```

> **Learn more**: [Writing Templates](../05-creating-benchmarks/writing-templates.md) · [Generating Templates](../05-creating-benchmarks/generating-templates.md) · [Answer Templates (Concepts)](../core_concepts/answer-templates.md)

---

## Step 4: Add Rubric Traits

While templates verify **correctness**, rubrics assess **quality** — properties of the raw response like conciseness, safety, or format compliance.

Karenina supports four trait types: LLM, regex, callable, and metric. Here we use two.

### Global Trait (evaluated for every question)

```python
from karenina.schemas import LLMRubricTrait

benchmark.add_global_rubric_trait(
    LLMRubricTrait(
        name="Conciseness",
        description="Rate how concise the answer is on a scale of 1-5, where 1 is very verbose and 5 is extremely concise.",
        kind="score",
    )
)
print("Added global rubric trait: Conciseness (score 1-5)")
```

### Question-Specific Trait (evaluated for one question)

This regex trait checks that the Venetoclax answer mentions the BCL2 protein:

```python
from karenina.schemas import RegexTrait

venetoclax_qid = question_ids[1]  # The Venetoclax question

benchmark.add_question_rubric_trait(
    venetoclax_qid,
    RegexTrait(
        name="Contains BCL2",
        description="The response must mention BCL2",
        pattern=r"\bBCL2\b",
        case_sensitive=True,
    ),
)
print(f"Added regex trait 'Contains BCL2' to question {venetoclax_qid}")
```

> **Learn more**: [Defining Rubrics](../05-creating-benchmarks/defining-rubrics.md) · [All Four Trait Types](../core_concepts/rubrics/index.md) — LLM, regex, callable, and metric traits

---

## Step 5: Run Verification

Configure the answering model (the model being evaluated) and the parsing model (the judge), then run verification.

```python
from karenina.schemas import ModelConfig, VerificationConfig

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="claude-haiku-4-5",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="langchain",
            temperature=0.7,
            system_prompt="You are a knowledgeable assistant. Answer accurately and concisely.",
        )
    ],
    parsing_models=[
        ModelConfig(
            id="claude-haiku-4-5",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="langchain",
            temperature=0.0,
        )
    ],
    evaluation_mode="template_and_rubric",
    rubric_enabled=True,
)

results = benchmark.run_verification(config)
print(f"Verification complete — {len(results.results)} results")
```

> **Learn more**: [Verification Config](../06-running-verification/verification-config.md) · [Multi-Model Evaluation](../06-running-verification/multi-model.md) · [Model Config Reference](../reference/configuration/model-config.md) · [CLI Verification](../reference/cli/verify.md)

---

## Step 6: Inspect Results

`VerificationResultSet` provides specialized accessors that convert results into pandas DataFrames for analysis.

### Template results

Use `get_template_results()` to access pass/fail data and field-level comparisons:

```python
template_results = results.get_template_results()
df_templates = template_results.to_dataframe()

df_templates[["question_id", "field_name", "gt_value", "llm_value", "field_match"]]
```

### Pass rate

```python
template_results.aggregate_pass_rate(by="question_id")
```

### Rubric results

Use `get_rubrics_results()` to access trait scores as a DataFrame:

```python
rubric_results = results.get_rubrics_results()
df_rubrics = rubric_results.to_dataframe()

df_rubrics[["question_id", "trait_name", "trait_score", "trait_type"]]
```

> **Learn more**: [DataFrame Analysis](../07-analyzing-results/dataframe-analysis.md) · [VerificationResult](../07-analyzing-results/verification-result.md) · [Exporting Results](../07-analyzing-results/exporting.md)

---

## Step 7: Save and Load

Save the benchmark — including questions, templates, rubrics, and results — as a JSON-LD checkpoint file.

```python
from pathlib import Path

checkpoint_path = Path(_tmpdir) / "genomics_benchmark.jsonld"
benchmark.save(checkpoint_path)
print(f"Saved to genomics_benchmark.jsonld")
```

Load it back later:

```python
loaded = Benchmark.load(checkpoint_path)
print(f"Loaded '{loaded.name}' with {loaded.question_count} questions")
```

> **Learn more**: [Checkpoints](../core_concepts/checkpoints.md) · [Saving Benchmarks](../05-creating-benchmarks/saving-benchmarks.md) · [Loading Benchmarks](../06-running-verification/loading-benchmark.md)

```python tags=["hide-cell"]
# Restore original LLM behavior and clean up temp directory
import shutil

BaseChatModel.ainvoke = _original_ainvoke
shutil.rmtree(_tmpdir, ignore_errors=True)
```
