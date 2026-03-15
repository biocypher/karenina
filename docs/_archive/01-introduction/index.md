---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# What is Karenina?

**Karenina** is a Python framework for defining, running, and sharing LLM benchmarks in a rigorous and reproducible way. It enables systematic evaluation of large language model performance through structured, verifiable testing.

## Key Capabilities

- **Create benchmarks** from scratch or from existing question sets
- **Define precise evaluation criteria** using code-based answer templates (Pydantic models)
- **Evaluate answers** using both rule-based verification and LLM-as-judge strategies
- **Support natural, unconstrained outputs** — no rigid response formats required
- **Assess response quality** with rubrics (LLM judgment, regex, callable, and metric traits)
- **Track performance** across multiple models and configurations
- **Share and reproduce** benchmark results via JSON-LD checkpoint files

## When to Use Karenina

Karenina is designed for data scientists and ML engineers who need to:

- **Compare models systematically** across consistent criteria, not ad-hoc prompting
- **Go beyond simple string matching** — evaluate free-form LLM outputs with structured logic
- **Combine correctness and quality checks** — verify factual accuracy *and* assess response qualities like clarity, safety, or format compliance
- **Automate evaluation at scale** — run hundreds of questions across multiple models with a single configuration
- **Reproduce results** — share benchmarks as portable JSON-LD files that anyone can re-run

## Ecosystem Overview

Karenina has three packages that work together:

| Package | Type | Purpose |
|---------|------|---------|
| **karenina** | Python library | Core benchmarking framework (this documentation) |
| **karenina-server** | FastAPI backend | REST API exposing karenina functionality |
| **karenina-gui** | React/TypeScript | No-code web interface for benchmark management |

This documentation covers the **karenina** Python library. The server and GUI have their own documentation.

## How It Works

Karenina uses a **two-unit evaluation approach**:

1. **Answer Templates** verify *correctness* — did the model give the right answer? A Judge LLM parses the model's free-text response into a structured Pydantic schema, then a programmatic `verify()` method checks it against ground truth.

2. **Rubrics** assess *quality* — how well did the model answer? Trait evaluators examine the raw response for qualities like safety, conciseness, format compliance, or extraction completeness.

These two units are complementary. A common pattern: use a template to verify the model extracted the correct answer, then use rubrics to check that the response was concise, cited sources, and avoided hallucination.

For a deeper discussion, see [Templates vs Rubrics](template-vs-rubric.md) and [Philosophy](philosophy.md).

---

## Quickstart

Here is a complete working example that loads a benchmark, configures verification, runs it, and inspects results.

```python tags=["hide-cell"]
# Mock cell: patches run_verification so the quickstart executes without live API keys.
# This cell is hidden in the rendered documentation.
import datetime
from unittest.mock import patch

from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
)


def _mock_run_verification(self, config, question_ids=None, **kwargs):
    """Return realistic mock results for documentation examples."""
    qids = question_ids or self.get_question_ids()
    mock_results = []
    answers = {
        "capital of France": ("Paris", True),
        "6 multiplied by 7": ("42", True),
        "atomic number 8": ("Oxygen (O)", True),
        "17 a prime": ("True", True),
        "machine learning": ("Machine learning is a subset of AI", None),
    }
    for qid in qids:
        q = self.get_question(qid)
        question_text = q["question"]
        # Match question to mock answer
        response, verified = ("Mock response", True)
        for key, (resp, ver) in answers.items():
            if key in question_text.lower():
                response, verified = resp, ver
                break
        answering = ModelIdentity(model_name="gpt-4o", interface="langchain")
        parsing = ModelIdentity(model_name="gpt-4o", interface="langchain")
        ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
        result_id = VerificationResultMetadata.compute_result_id(
            qid, answering, parsing, ts
        )
        template_result = None
        if verified is not None:
            template_result = VerificationResultTemplate(
                raw_llm_response=response,
                verify_result=verified,
                template_verification_performed=True,
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
```

### Load a Benchmark

```python
from karenina import Benchmark

benchmark = Benchmark.load("test_checkpoint.jsonld")
print(f"Loaded '{benchmark.name}' with {benchmark.question_count} questions")
```

### Configure and Run Verification

```python
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.config import ModelConfig

config = VerificationConfig(
    answering_models=[ModelConfig(id="gpt-4o", model_name="gpt-4o", interface="langchain")],
    parsing_models=[ModelConfig(id="gpt-4o", model_name="gpt-4o", interface="langchain")],
)

results = benchmark.run_verification(config)
print(f"Completed {len(results.results)} verifications")
```

### Inspect Results

```python
for result in results.results:
    q_text = result.metadata.question_text[:60]
    if result.template and result.template.verify_result is not None:
        status = "PASS" if result.template.verify_result else "FAIL"
    else:
        status = "N/A (no template)"
    print(f"  [{status}] {q_text}")
```

```python tags=["hide-cell"]
# Clean up the mocks
_ = _patcher_run.stop()
_ = _patcher_validate.stop()
```

---

## Next Steps

- [Philosophy](philosophy.md) — Why LLM-as-judge evaluation works
- [Templates vs Rubrics](template-vs-rubric.md) — Understanding the two evaluation units
- [Installation](../02-installation/index.md) — Install karenina and set up API keys
- [Core Concepts](../core_concepts/index.md) — Deep dive into checkpoints, templates, rubrics, and more
