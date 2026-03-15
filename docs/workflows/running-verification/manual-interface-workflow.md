---
jupyter:
  jupytext:
    formats: docs/workflows/running-verification//md,docs/notebooks/running-verification//ipynb
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

# Manual Interface

This scenario uses the manual interface for offline evaluation — you supply pre-recorded LLM responses instead of generating them live. This is useful for evaluating cached responses, comparing parsing models on the same answers, or iterating on templates without re-running expensive LLM calls.

**What you'll learn:**

- When to use manual (offline) verification
- Prepare and register pre-recorded traces
- Configure the manual interface with a parsing model
- Common patterns: template iteration, parsing model comparison
- CLI workflow for manual verification

```python tags=["hide-cell"]
# Setup cell: creates a mock benchmark and manual traces.
# This cell is hidden in the rendered documentation.
import datetime
import hashlib
import json
import tempfile
from pathlib import Path

from karenina import Benchmark
from karenina.schemas.config import ModelConfig
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
)

_benchmark = Benchmark.create(
    name="Biomedical Factual QA",
    description="Template-only evaluation with pre-recorded responses",
    version="1.0.0",
)
_questions = [
    ("What is the approved pharmacological target of venetoclax?", "BCL2"),
    ("How many chromosome pairs do humans have?", "23"),
    ("What is the primary neurotransmitter in the sympathetic nervous system?", "Norepinephrine"),
    ("What organ produces insulin?", "Pancreas"),
    ("What is the half-life of caffeine in healthy adults?", "5 hours"),
]
for q, a in _questions:
    _benchmark.add_question(question=q, raw_answer=a)

_tmp = Path(tempfile.mkdtemp()) / "benchmark.jsonld"
_benchmark.save(str(_tmp))
_qids = _benchmark.get_question_ids()

# Pre-recorded responses (simulating cached LLM output)
_traces = {
    _qids[0]: "Venetoclax is a selective BCL2 inhibitor. It targets the BCL2 protein, which is overexpressed in certain cancers.",
    _qids[1]: "Humans have 23 pairs of chromosomes, for a total of 46 chromosomes.",
    _qids[2]: "The primary neurotransmitter of the sympathetic nervous system is norepinephrine (noradrenaline).",
    _qids[3]: "Insulin is produced by the beta cells of the islets of Langerhans in the pancreas.",
    _qids[4]: "The half-life of caffeine in healthy adults is approximately 5 hours, though it can range from 3-7 hours.",
}

_answering = ModelIdentity(model_name="manual", interface="manual")
_parsing = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()

_pass_fail = [True, True, True, True, True]


def _make(qid, q_text, raw_ans, response, verified):
    rid = VerificationResultMetadata.compute_result_id(qid, _answering, _parsing, _ts)
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=qid, template_id="tmpl_" + qid[:8],
            completed_without_errors=True, question_text=q_text,
            raw_answer=raw_ans, answering=_answering, parsing=_parsing,
            execution_time=0.8, timestamp=_ts, result_id=rid,
        ),
        template=VerificationResultTemplate(
            raw_llm_response=response,
            verify_result=verified, template_verification_performed=True,
            parsed_gt_response={"answer": raw_ans},
            parsed_llm_response={"answer": raw_ans},
        ),
    )


_mock_results = [
    _make(qid, q, a, _traces[qid], v)
    for qid, (q, a), v in zip(_qids, _questions, _pass_fail)
]
_mock_result_set = VerificationResultSet(results=_mock_results)
_orig_run = Benchmark.run_verification
Benchmark.run_verification = lambda self, config, **kw: _mock_result_set

# Write traces to a temp file for the JSON loading example (keyed by question text)
_traces_by_text = {q: _traces[qid] for qid, (q, _) in zip(_qids, _questions)}
_traces_file = Path(tempfile.mkdtemp()) / "traces.json"
_traces_file.write_text(json.dumps(_traces_by_text, indent=2))
```

---

## When to Use

| Scenario | Why Manual? |
|----------|-------------|
| **Template iteration** | Refine templates against the same responses without re-generating |
| **Parsing model comparison** | Try different judge models on identical inputs |
| **Cost control** | Evaluate expensive model outputs without re-running them |
| **Reproducibility** | Guarantee identical inputs across evaluation runs |
| **Cached responses** | Evaluate responses saved from a previous run or external system |

---

## Workflow Diagram

```
Load benchmark                Pre-record responses
    │                               │
    ▼                               ▼
Configure manual interface    Register traces (dict or file)
    │                               │
    └───────────┬───────────────────┘
                │
                ▼
    Run verification (parsing only — no answer generation)
                │
                ▼
        Inspect results
```

---

## Prepare Traces

Pre-recorded traces map question IDs to response strings:

### As a Python Dictionary

```python
from karenina import Benchmark

benchmark = Benchmark.load(str(_tmp))
question_ids = benchmark.get_question_ids()

# Map each question ID to a pre-recorded response
question_ids = benchmark.get_question_ids()
traces = {}
for qid in question_ids:
    q = benchmark.get_question(qid)
    traces[q["question"]] = f"Response for: {q['question'][:30]}..."

# Override with realistic responses
all_q = [benchmark.get_question(qid) for qid in question_ids]
traces[all_q[0]["question"]] = "Venetoclax is a selective BCL2 inhibitor. It targets the BCL2 protein."
traces[all_q[1]["question"]] = "Humans have 23 pairs of chromosomes, for a total of 46."
traces[all_q[2]["question"]] = "The primary neurotransmitter is norepinephrine (noradrenaline)."
traces[all_q[3]["question"]] = "Insulin is produced by the beta cells in the pancreas."
traces[all_q[4]["question"]] = "The half-life of caffeine is approximately 5 hours."

print(f"Prepared traces for {len(traces)} questions")
```

### From a JSON File

```python
import json

# Load from a JSON file (question_id → response string)
with open(str(_traces_file)) as f:
    traces_from_file = json.load(f)

print(f"Loaded {len(traces_from_file)} traces from file")
```

---

## Register Traces

Use `ManualTraces` to register pre-recorded responses with the benchmark:

```python
from karenina.adapters.manual.traces import ManualTraces

# Register all traces at once (ManualTraces requires the benchmark)
# map_to_id=True converts question text keys to internal question hashes
manual_traces = ManualTraces(benchmark)
manual_traces.register_traces(traces, map_to_id=True)

print(f"Registered {len(traces)} traces")
```

### Register Individual Traces

```python
manual_traces_individual = ManualTraces(benchmark)
for question_text, response in traces.items():
    manual_traces_individual.register_trace(question_text, response, map_to_id=True)

print(f"Registered {len(traces)} traces individually")
```

### Load from File and Register

```python
import json

manual_traces_from_file = ManualTraces(benchmark)
with open(str(_traces_file)) as f:
    file_traces = json.load(f)
manual_traces_from_file.register_traces(file_traces, map_to_id=True)

print(f"Loaded and registered {len(file_traces)} traces from file")
```

---

## Configure and Run

Set `interface="manual"` on the answering model. The parsing model still uses a live LLM:

```python
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.config import ModelConfig

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="manual",
            model_name="manual",
            model_provider="manual",
            interface="manual",
            manual_traces=manual_traces,
        )
    ],
    parsing_models=[
        ModelConfig(
            id="haiku-parser",
            model_name="claude-haiku-4-5",
            model_provider="anthropic",
            interface="langchain",
            temperature=0.0,
        )
    ],
    evaluation_mode="template_only",
)

results = benchmark.run_verification(config)
print(f"Results: {len(results)}")
```

With the manual interface, no answer generation LLM calls are made — only parsing calls.

---

## Inspect Results

```python
for result in results:
    meta = result.metadata
    t = result.template
    status = "PASS" if (t and t.verify_result) else "FAIL"
    print(f"[{status}] {meta.question_text[:50]}")
    print(f"         Model: {meta.answering.interface}/{meta.answering.model_name}")
```

---

## Common Patterns

### Template Iteration

Refine a template and re-evaluate without re-generating responses:

```python
# Step 1: Run with initial template → inspect failures
# Step 2: Update the template code (e.g., adjust verify() logic)
# Step 3: Re-run with same traces — only parsing is repeated
#
# benchmark.update_template(question_id, new_template_code)
# results = benchmark.run_verification(config)
print("Iterate: update template → re-run → compare results")
```

### Parsing Model Comparison

Evaluate the same responses with different parsing models:

```python
parsing_models = [
    ModelConfig(id="haiku-parser", model_name="claude-haiku-4-5",
                model_provider="anthropic", interface="langchain", temperature=0.0),
    ModelConfig(id="sonnet-parser", model_name="claude-sonnet-4-5",
                model_provider="anthropic", interface="langchain", temperature=0.0),
]

for parser in parsing_models:
    parser_config = VerificationConfig(
        answering_models=[
            ModelConfig(id="manual", model_name="manual",
                        model_provider="manual", interface="manual",
                        manual_traces=manual_traces)
        ],
        parsing_models=[parser],
        evaluation_mode="template_only",
    )
    parser_results = benchmark.run_verification(parser_config)
    passed = sum(1 for r in parser_results if r.template and r.template.verify_result)
    print(f"Parser {parser.id}: {passed}/{len(parser_results)} passed")
```

### Late Trace Population

Register traces incrementally, running verification as traces become available:

```python
# Start with a subset
partial_traces = ManualTraces(benchmark)
subset_keys = list(traces.keys())[:3]
for question_text in subset_keys:
    partial_traces.register_trace(question_text, traces[question_text], map_to_id=True)

print(f"Phase 1: {len(subset_keys)} traces registered")

# Add more traces later
remaining_keys = list(traces.keys())[3:]
for question_text in remaining_keys:
    partial_traces.register_trace(question_text, traces[question_text], map_to_id=True)

print(f"Phase 2: {len(subset_keys) + len(remaining_keys)} traces registered")
```

---

## CLI Workflow

```python
# Manual verification via CLI:
# karenina verify benchmark.jsonld --preset base.json \
#   --interface manual --manual-traces traces.json

# Compare parsing models on the same traces:
# karenina verify benchmark.jsonld --interface manual \
#   --manual-traces traces.json --parsing-model claude-haiku-4-5
# karenina verify benchmark.jsonld --interface manual \
#   --manual-traces traces.json --parsing-model claude-sonnet-4-5

print("CLI: karenina verify ... --interface manual --manual-traces traces.json")
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `KeyError: question_id` | Trace not registered for a question | Register traces for all questions, or use `question_ids` to verify a subset |
| Empty `raw_llm_response` | Trace is an empty string | Check the trace content — empty strings are valid but will likely fail parsing |
| Parsing errors on manual traces | Response format doesn't match template expectations | Review the template's expected fields and adjust the response or template |

---

## Related Pages

- [Basic Verification](basic-verification.ipynb) — Live verification walkthrough
- [Full Evaluation](full-evaluation.ipynb) — Add rubrics to manual evaluation
- [Adapters](../../core_concepts/adapters.md) — Manual adapter details
- [CLI Reference: verify](../../reference/cli/verify.md) — `--interface` and `--manual-traces` options

```python tags=["hide-cell"]
# Cleanup
Benchmark.run_verification = _orig_run
```
