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
# Mock cell: patches run_verification so the quickstart executes without live API keys.
# This cell is hidden in the rendered documentation.
import datetime
import tempfile
from unittest.mock import patch

from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

_MOCK_RESPONSES = {
    "chromosomes": "There are 46 chromosomes in a human somatic cell — 23 pairs in total.",
    "venetoclax": "Venetoclax targets BCL2 (B-cell lymphoma 2), an anti-apoptotic protein in the BH3 family.",
    "hemoglobin": "Hemoglobin A has 4 protein subunits: two alpha and two beta globin chains.",
}


def _mock_run_verification(self, config, question_ids=None, **kwargs):
    """Return realistic mock results for documentation examples."""
    qids = question_ids or self.get_question_ids()
    mock_results = []
    for qid in qids:
        q = self.get_question(qid)
        question_text = q["question"]
        # Match question to mock response
        response, verified = "Mock response", True
        for key, resp in _MOCK_RESPONSES.items():
            if key in question_text.lower():
                response = resp
                break
        answering = ModelIdentity(model_name="gpt-4.1-mini", interface="langchain")
        parsing = ModelIdentity(model_name="gpt-4.1-mini", interface="langchain")
        ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
        result_id = VerificationResultMetadata.compute_result_id(
            qid, answering, parsing, ts
        )
        template_result = VerificationResultTemplate(
            raw_llm_response=response,
            verify_result=verified,
            template_verification_performed=True,
        )
        rubric_result = None
        if config.rubric_enabled:
            scores = {"Conciseness": 4}
            regex_scores = {}
            if "venetoclax" in question_text.lower():
                regex_scores["Contains BCL2"] = True
            rubric_result = VerificationResultRubric(
                rubric_evaluation_performed=True,
                rubric_evaluation_strategy="batch",
                llm_trait_scores=scores,
                regex_trait_scores=regex_scores if regex_scores else None,
            )
        result = VerificationResult(
            metadata=VerificationResultMetadata(
                question_id=qid,
                template_id="mock_template",
                completed_without_errors=True,
                question_text=question_text,
                raw_answer=q.get("raw_answer"),
                answering=answering,
                parsing=parsing,
                execution_time=1.2,
                timestamp=ts,
                result_id=result_id,
            ),
            template=template_result,
            rubric=rubric_result,
        )
        mock_results.append(result)
    return VerificationResultSet(results=mock_results)


_patcher_run = patch(
    "karenina.benchmark.benchmark.Benchmark.run_verification",
    _mock_run_verification,
)
_patcher_validate = patch.object(
    VerificationConfig, "_validate_config", lambda self: None
)
_patcher_run.start()
_patcher_validate.start()

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

---

## Step 3: Write Answer Templates

Answer templates are Pydantic models that define how a Judge LLM should parse and verify a model's response. Each template:

1. Declares **attributes** the judge must extract (typed fields)
2. Stores the **correct values** in `model_post_init`
3. Implements a **`verify()`** method that compares extracted values to ground truth

The class must always be named `Answer` and inherit from `BaseAnswer`.

Here is a template for the Venetoclax question:

```python
venetoclax_template = '''
from pydantic import Field
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    target: str = Field(description="The protein target of the drug mentioned in the response")

    def model_post_init(self, __context):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.strip().upper() == self.correct["target"].upper()
'''
```

Add templates to each question using `update_template`:

```python
chromosomes_template = '''
from pydantic import Field
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    count: int = Field(description="The number of chromosomes mentioned in the response")

    def model_post_init(self, __context):
        self.correct = {"count": 46}

    def verify(self) -> bool:
        return self.count == self.correct["count"]
'''

hemoglobin_template = '''
from pydantic import Field
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    subunit_count: int = Field(description="The number of protein subunits mentioned in the response")

    def model_post_init(self, __context):
        self.correct = {"subunit_count": 4}

    def verify(self) -> bool:
        return self.subunit_count == self.correct["subunit_count"]
'''

templates = [chromosomes_template, venetoclax_template, hemoglobin_template]
for qid, code in zip(question_ids, templates):
    benchmark.update_template(qid, code)

print(f"Added templates to {len(templates)} questions")
```

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

---

## Step 5: Run Verification

Configure the answering model (the model being evaluated) and the parsing model (the judge), then run verification.

```python
from karenina.schemas import VerificationConfig, ModelConfig

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="gpt-4.1-mini",
            model_name="gpt-4.1-mini",
            interface="langchain",
            temperature=0.7,
            system_prompt="You are a knowledgeable assistant. Answer accurately and concisely.",
        )
    ],
    parsing_models=[
        ModelConfig(
            id="gpt-4.1-mini",
            model_name="gpt-4.1-mini",
            interface="langchain",
            temperature=0.0,
        )
    ],
    rubric_enabled=True,
)

results = benchmark.run_verification(config)
print(f"Verification complete — {len(results.results)} results")
```

---

## Step 6: Inspect Results

### Iterate over results

Each `VerificationResult` contains metadata, template verification, and rubric evaluation.

```python
for result in results.results:
    q_text = result.metadata.question_text[:60]

    # Template verification
    if result.template and result.template.verify_result is not None:
        status = "PASS" if result.template.verify_result else "FAIL"
    else:
        status = "N/A"

    # Rubric scores
    rubric_info = ""
    if result.rubric and result.rubric.llm_trait_scores:
        scores = ", ".join(f"{k}={v}" for k, v in result.rubric.llm_trait_scores.items())
        rubric_info = f"  rubric: {scores}"

    print(f"  [{status}] {q_text}{rubric_info}")
```

### Aggregate pass rate

```python
total = len(results.results)
passed = sum(
    1
    for r in results.results
    if r.template and r.template.verify_result
)
print(f"\nOverall pass rate: {passed}/{total} ({passed / total * 100:.0f}%)")
```

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

```python tags=["hide-cell"]
# Clean up mocks and temp directory
import shutil

_patcher_run.stop()
_patcher_validate.stop()
shutil.rmtree(_tmpdir, ignore_errors=True)
```

---

## Going Further

### Automatic Template Generation

Instead of writing templates by hand, Karenina can generate them using an LLM.
Call `benchmark.generate_all_templates()` with model parameters:

> ```python
> benchmark.generate_all_templates(
>     model="gpt-4.1-mini",
>     model_provider="openai",
>     temperature=0.1,
> )
> ```

See [Generating Templates](../05-creating-benchmarks/generating-templates.md) for details.

### Extracting Questions from Files

Import questions from Excel, CSV, or TSV files using `extract_questions_from_file`:

> ```python
> from karenina.benchmark.authoring.questions import extract_questions_from_file
>
> questions = extract_questions_from_file(
>     file_path="questions.xlsx",
>     question_column="Question",
>     answer_column="Answer",
>     keywords_columns=[{"column": "Keywords", "separator": ","}],
> )
> ```

See [Adding Questions](../05-creating-benchmarks/adding-questions.md) for details.

### Using Different LLM Providers

Karenina supports many backends. Pass different `interface` values to `ModelConfig`:

> ```python
> # Anthropic Claude (via LangChain)
> ModelConfig(id="claude", model_name="claude-sonnet-4-5-20250929", interface="langchain")
>
> # OpenRouter
> ModelConfig(id="sonnet", model_name="anthropic/claude-sonnet-4-5-20250929", interface="openrouter")
>
> # Local model (Ollama or any OpenAI-compatible endpoint)
> ModelConfig(id="local", model_name="llama3", interface="openai_endpoint",
>             endpoint_base_url="http://localhost:11434/v1")
> ```

---

## Next Steps

| Topic | Link |
|-------|------|
| Core concepts (checkpoints, templates, rubrics) | [Core Concepts](../04-core-concepts/index.md) |
| Writing custom templates in depth | [Writing Templates](../05-creating-benchmarks/writing-templates.md) |
| All four rubric trait types | [Rubrics](../04-core-concepts/rubrics/index.md) |
| Verification configuration options | [Verification Config](../06-running-verification/verification-config.md) |
| DataFrame-based result analysis | [DataFrame Analysis](../07-analyzing-results/dataframe-analysis.md) |
| CLI verification (no Python needed) | [CLI Reference](../09-cli-reference/verify.md) |
