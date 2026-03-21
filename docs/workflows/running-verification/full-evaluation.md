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

# Full Evaluation

This scenario runs verification with both template and rubric evaluation — the most comprehensive single-model workflow. You enable rubrics alongside templates, configure quality checks (abstention, sufficiency, embedding), customize prompts with `PromptConfig`, and use presets for repeatable configurations.

**What you'll learn:**

- Configure template+rubric evaluation mode
- Enable abstention, sufficiency, and embedding checks
- Customize judge prompts with `PromptConfig`
- Use presets for repeatable configurations
- Inspect combined template and rubric results

```python tags=["hide-cell"]
# Setup cell: creates a mock benchmark and patches run_verification.
# This cell is hidden in the rendered documentation.
import datetime
import tempfile
from pathlib import Path

from karenina import Benchmark
from karenina.schemas.config import ModelConfig
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

_benchmark = Benchmark.create(
    name="Drug Safety QA",
    description="Template + rubric evaluation of drug safety knowledge",
    version="1.0.0",
)
_questions = [
    ("What are the common side effects of metformin?", "GI symptoms: nausea, diarrhea, abdominal pain"),
    ("What is the mechanism of action of ibuprofen?", "COX-1 and COX-2 inhibition"),
    ("What are the contraindications for warfarin?", "Pregnancy, active bleeding, severe liver disease"),
    ("What is the recommended dosage of amoxicillin for adults?", "500mg every 8 hours"),
    ("What drug interactions should be monitored with SSRIs?", "MAOIs, triptans, anticoagulants"),
]
for q, a in _questions:
    _benchmark.add_question(question=q, raw_answer=a)

_tmp = Path(tempfile.mkdtemp()) / "benchmark.jsonld"
_benchmark.save(str(_tmp))

_answering = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_parsing = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
_qids = _benchmark.get_question_ids()

# Mock scenarios: normal, normal, abstention, insufficient, normal+embedding
_scenarios = [
    {"verified": True, "rubric": {"accuracy": 4, "completeness": True, "citation_format": True}, "abstention": False, "sufficiency": True, "embedding": None},
    {"verified": True, "rubric": {"accuracy": 5, "completeness": True, "citation_format": True}, "abstention": False, "sufficiency": True, "embedding": None},
    {"verified": None, "rubric": None, "abstention": True, "sufficiency": None, "embedding": None},
    {"verified": None, "rubric": None, "abstention": False, "sufficiency": False, "embedding": None},
    {"verified": True, "rubric": {"accuracy": 3, "completeness": False, "citation_format": True}, "abstention": False, "sufficiency": True, "embedding": 0.87},
]


def _make(qid, q_text, raw_ans, scenario):
    rid = VerificationResultMetadata.compute_result_id(qid, _answering, _parsing, _ts)
    template = VerificationResultTemplate(
        raw_llm_response=f"Based on clinical evidence, {raw_ans.lower()}.",
        verify_result=scenario["verified"],
        template_verification_performed=scenario["verified"] is not None,
        parsed_gt_response={"answer": raw_ans},
        parsed_llm_response={"answer": raw_ans if scenario["verified"] else "I cannot answer this"},
        abstention_check_performed=True,
        abstention_detected=scenario["abstention"],
        abstention_override_applied=scenario["abstention"],
        abstention_reasoning="Model declined to provide medical advice" if scenario["abstention"] else None,
        sufficiency_check_performed=not scenario["abstention"],
        sufficiency_detected=scenario["sufficiency"],
        sufficiency_override_applied=scenario["sufficiency"] is False,
        sufficiency_reasoning="Response lacks specific details" if scenario["sufficiency"] is False else None,
        embedding_check_performed=scenario["embedding"] is not None,
        embedding_similarity_score=scenario["embedding"],
        embedding_model_used="all-MiniLM-L6-v2" if scenario["embedding"] else None,
    )
    rubric = None
    if scenario["rubric"]:
        rubric = VerificationResultRubric(
            rubric_evaluation_performed=True,
            rubric_evaluation_strategy="batch",
            llm_trait_scores={"accuracy": scenario["rubric"]["accuracy"], "completeness": scenario["rubric"]["completeness"]},
            regex_trait_scores={"citation_format": scenario["rubric"]["citation_format"]},
        )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=qid,
            template_id="tmpl_" + qid[:8],
            completed_without_errors=True,
            question_text=q_text,
            raw_answer=raw_ans,
            answering=_answering,
            parsing=_parsing,
            execution_time=2.1,
            timestamp=_ts,
            result_id=rid,
        ),
        template=template,
        rubric=rubric,
    )


_mock_results = [_make(qid, q, a, s) for qid, (q, a), s in zip(_qids, _questions, _scenarios)]
_mock_result_set = VerificationResultSet(results=_mock_results)
_orig_run = Benchmark.run_verification
Benchmark.run_verification = lambda self, config, **kw: _mock_result_set
```

---

## Load Benchmark

```python
from karenina import Benchmark

benchmark = Benchmark.load(str(_tmp))
print(f"{benchmark.name}: {benchmark.question_count} questions")
```

For details on loading and inspecting benchmarks, see [Basic Verification](basic-verification.ipynb).

---

## Configure Template+Rubric Mode

Enable rubric evaluation alongside templates, plus optional quality checks:

```python
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.config import ModelConfig

config = VerificationConfig(
    answering_models=[
        ModelConfig(id="haiku", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain")
    ],
    parsing_models=[
        ModelConfig(id="haiku-parser", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain",
                    temperature=0.0)
    ],
    # Evaluation mode
    evaluation_mode="template_and_rubric",
    # Quality checks
    abstention_enabled=True,
    sufficiency_enabled=True,
    # Embedding similarity
    embedding_check_enabled=True,
    embedding_check_threshold=0.85,
)

print(f"Mode:       {config.evaluation_mode}")
print(f"Abstention: {config.abstention_enabled}")
print(f"Sufficiency:{config.sufficiency_enabled}")
print(f"Embedding:  {config.embedding_check_enabled}")
```

When abstention is detected, parsing and rubric stages are skipped — the result is auto-failed. When sufficiency is insufficient, the same skipping applies. See [Evaluation Modes](../../notebooks/core_concepts/evaluation-modes.ipynb) for stage-skipping rules.

---

## Customize with PromptConfig

`PromptConfig` lets you inject custom instructions into the judge prompts:

```python
from karenina.schemas.verification import PromptConfig

prompt_config = PromptConfig(
    parsing="Focus on extracting exact values. If the response contains hedging language, extract the most definitive statement.",
    rubric_evaluation="Evaluate rubric traits strictly — partial compliance should score lower.",
)

config_with_prompts = VerificationConfig(
    answering_models=[
        ModelConfig(id="haiku", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain")
    ],
    parsing_models=[
        ModelConfig(id="haiku-parser", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain",
                    temperature=0.0)
    ],
    evaluation_mode="template_and_rubric",
    prompt_config=prompt_config,
)

print(f"Parsing prompt:  {config_with_prompts.prompt_config.parsing[:50]}...")
print(f"Rubric prompt:   {config_with_prompts.prompt_config.rubric_evaluation[:50]}...")
```

`PromptConfig` fields correspond to pipeline stages. See [PromptConfig Reference](../../reference/configuration/prompt-config.md) for all injection points.

---

## Use Presets

Presets save a full `VerificationConfig` as JSON for repeatable runs:

### Load from Preset File

```python
# config = VerificationConfig.from_preset(Path("presets/production.json"))
print("Load with: VerificationConfig.from_preset(Path('presets/production.json'))")
```

### Override Preset Values

```python
# Apply overrides on top of a preset base
config_override = VerificationConfig.from_overrides(
    answering_id="haiku", answering_model="claude-haiku-4-5",
    parsing_id="haiku-parser", parsing_model="claude-haiku-4-5",
    evaluation_mode="template_and_rubric",
    abstention=True,
)

print(f"Mode: {config_override.evaluation_mode}")
```

### CLI with Preset + Overrides

```python
# Combine preset with command-line overrides:
# karenina verify checkpoint.jsonld --preset production.json \
#   --evaluation-mode template_and_rubric --abstention --sufficiency
print("CLI: karenina verify ... --preset production.json --abstention --sufficiency")
```

See [Preset Schema Reference](../../reference/configuration/preset-schema.md) for the preset JSON format.

---

## Run Verification

```python
results = benchmark.run_verification(config)
print(f"Total results: {len(results)}")
```

---

## Inspect Combined Results

### Template Results

```python
for result in results:
    meta = result.metadata
    t = result.template
    if t and t.abstention_detected:
        print(f"[ABSTAINED] {meta.question_text[:50]}")
    elif t and t.sufficiency_detected is False:
        print(f"[INSUFFICIENT] {meta.question_text[:50]}")
    elif t and t.template_verification_performed:
        status = "PASS" if t.verify_result else "FAIL"
        print(f"[{status}] {meta.question_text[:50]}")
    else:
        print(f"[SKIPPED] {meta.question_text[:50]}")
```

### Rubric Scores

```python
for result in results:
    if result.rubric and result.rubric.rubric_evaluation_performed:
        scores = result.rubric.get_all_trait_scores()
        print(f"Q: {result.metadata.question_text[:40]}  Traits: {scores}")
```

### Embedding Scores

```python
for result in results:
    t = result.template
    if t and t.embedding_check_performed:
        print(f"Q: {result.metadata.question_text[:40]}  "
              f"Similarity: {t.embedding_similarity_score:.2f}  "
              f"Override: {t.embedding_override_applied}")
```

---

## Dynamic Rubric

A [DynamicRubric](../../core_concepts/rubrics/index.md#6-dynamic-rubric) allows conditional rubric evaluation: traits are only scored when their concept is detected in the response. Attach a `DynamicRubric` to individual questions so that traits irrelevant to a particular response are skipped rather than evaluated against unrelated content.

```python
from karenina.schemas.entities.rubric import DynamicRubric, LLMRubricTrait

dynamic = DynamicRubric(
    llm_traits=[
        LLMRubricTrait(
            name="interaction_safety",
            summary="drug interaction warnings",
            description="Answer True if the response includes drug interaction warnings.",
            kind="boolean",
            higher_is_better=True,
        ),
        LLMRubricTrait(
            name="dosing_clarity",
            summary="dosing instructions",
            description="Rate dosing clarity from 1 to 5.",
            kind="score",
            higher_is_better=True,
        ),
    ],
)

# Attach per-question
benchmark.add_question(
    question="What is the recommended treatment for condition X?",
    raw_answer="Drug A, 500mg twice daily",
    dynamic_rubric=dynamic,
)

# Run with rubric mode enabled
config = VerificationConfig(
    answering_models=[
        ModelConfig(id="haiku", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain")
    ],
    parsing_models=[
        ModelConfig(id="haiku-parser", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain")
    ],
    evaluation_mode="template_and_rubric",
)

results = benchmark.run_verification(config)

# Inspect dynamic rubric metadata
for result in results:
    if result.rubric:
        print(f"Promoted: {result.rubric.dynamic_rubric_promoted_traits}")
        print(f"Skipped:  {result.rubric.dynamic_rubric_skipped_traits}")
```

The presence check runs automatically at the start of Stage 11 (RubricEvaluation). Skipped traits do not incur evaluation cost. Static rubric traits (attached via `benchmark.set_global_rubric()`) are always evaluated; only dynamic rubric traits are gated by presence.

---

## Related Pages

- [Basic Verification](basic-verification.ipynb) — Simpler template-only path
- [Deep Judgment](deep-judgment.ipynb) — Add excerpt-based reasoning to template and rubric evaluation
- [VerificationConfig Reference](../../reference/configuration/verification-config.md) — All configuration fields
- [PromptConfig Reference](../../reference/configuration/prompt-config.md) — Prompt injection points
- [Preset Schema Reference](../../reference/configuration/preset-schema.md) — Preset JSON format

```python tags=["hide-cell"]
# Cleanup
Benchmark.run_verification = _orig_run
```
