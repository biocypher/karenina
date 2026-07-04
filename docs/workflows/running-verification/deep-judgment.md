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

# Deep Judgment

This scenario adds deep judgment to verification — a multi-stage evaluation process that extracts excerpts from responses, performs fuzzy matching, generates reasoning traces, and optionally validates claims against external search results. Deep judgment works for both template and rubric evaluation.

**What you'll learn:**

- Enable deep judgment for template verification
- Inspect extracted excerpts, reasoning, and hallucination risk
- Enable search-based validation for factual claims
- Configure deep judgment for rubric traits
- Tune deep judgment parameters

```python tags=["hide-cell"]
# Setup cell: creates mock results with deep judgment data.
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
    VerificationResultDeepJudgment,
    VerificationResultDeepJudgmentRubric,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

_benchmark = Benchmark.create(
    name="Biomedical Claims QA",
    description="Deep judgment evaluation of biomedical claims",
    version="1.0.0",
)
_questions = [
    (
        "What is the primary mechanism of action of metformin?",
        "AMPK activation and hepatic glucose production reduction",
    ),
    (
        "What are the FDA-approved indications for pembrolizumab?",
        "Multiple cancer types including melanoma, NSCLC, and head/neck cancer",
    ),
    (
        "What is the recommended first-line treatment for H. pylori?",
        "Triple therapy: PPI + clarithromycin + amoxicillin",
    ),
    ("What are the major adverse effects of statins?", "Myopathy, rhabdomyolysis, hepatotoxicity, new-onset diabetes"),
    ("What is the half-life of warfarin?", "36-42 hours"),
]
for q, a in _questions:
    _benchmark.add_question(question=q, raw_answer=a)

_tmp = Path(tempfile.mkdtemp()) / "benchmark.jsonld"
_benchmark.save(str(_tmp))

_answering = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_parsing = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
_qids = _benchmark.get_question_ids()

_dj_data = [
    {
        "verified": True,
        "excerpts": {
            "mechanism": [
                {"text": "Metformin primarily activates AMPK", "confidence": "high", "similarity_score": 0.92}
            ]
        },
        "reasoning": {
            "mechanism": "The response correctly identifies AMPK activation as the primary mechanism. The excerpt directly supports this claim with high confidence."
        },
        "hallucination_risk": None,
    },
    {
        "verified": True,
        "excerpts": {
            "indications": [
                {"text": "FDA approved for melanoma", "confidence": "high", "similarity_score": 0.88},
                {"text": "approved for non-small cell lung cancer", "confidence": "high", "similarity_score": 0.91},
                {
                    "text": "indicated for head and neck squamous cell carcinoma",
                    "confidence": "medium",
                    "similarity_score": 0.79,
                },
            ]
        },
        "reasoning": {
            "indications": "Multiple excerpts confirm the stated indications. All major categories are covered with supporting evidence from the response."
        },
        "hallucination_risk": None,
    },
    {
        "verified": True,
        "excerpts": {
            "treatment": [
                {
                    "text": "PPI combined with clarithromycin and amoxicillin",
                    "confidence": "high",
                    "similarity_score": 0.95,
                }
            ]
        },
        "reasoning": {"treatment": "The response accurately describes the standard triple therapy regimen."},
        "hallucination_risk": None,
    },
    {
        "verified": False,
        "excerpts": {
            "adverse_effects": [
                {"text": "muscle pain and weakness", "confidence": "medium", "similarity_score": 0.72},
            ]
        },
        "reasoning": {
            "adverse_effects": "The response mentions myopathy but omits rhabdomyolysis and hepatotoxicity. Incomplete coverage of major adverse effects."
        },
        "hallucination_risk": None,
    },
    {
        "verified": True,
        "excerpts": {
            "half_life": [
                {
                    "text": "elimination half-life of approximately 40 hours",
                    "confidence": "high",
                    "similarity_score": 0.89,
                    "search_results": "Warfarin half-life ranges from 20-60 hours (mean ~40 hours). Source: FDA prescribing information.",
                    "hallucination_risk": "none",
                    "hallucination_justification": "Claim matches FDA reference data.",
                },
            ]
        },
        "reasoning": {
            "half_life": "The stated half-life falls within the accepted range. Search validation confirms the claim."
        },
        "hallucination_risk": {"half_life": "none"},
    },
]


def _make(qid, q_text, raw_ans, dj):
    rid = VerificationResultMetadata.compute_result_id(qid, _answering, _parsing, _ts)
    search_enabled = dj["hallucination_risk"] is not None
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=qid,
            template_id="tmpl_" + qid[:8],
            failure=None,
            caveats=[],
            question_text=q_text,
            raw_answer=raw_ans,
            answering=_answering,
            parsing=_parsing,
            execution_time=4.5,
            timestamp=_ts,
            result_id=rid,
        ),
        template=VerificationResultTemplate(
            raw_llm_response=f"Based on current evidence, {raw_ans.lower()}.",
            verify_result=dj["verified"],
            template_verification_performed=True,
            parsed_gt_response={"answer": raw_ans},
            parsed_llm_response={"answer": raw_ans},
        ),
        deep_judgment=VerificationResultDeepJudgment(
            deep_judgment_mode="full",
            deep_judgment_performed=True,
            extracted_excerpts=dj["excerpts"],
            attribute_reasoning=dj["reasoning"],
            deep_judgment_stages_completed=["excerpts", "reasoning", "parameters"],
            deep_judgment_model_calls=3,
            deep_judgment_search_enabled=search_enabled,
            hallucination_risk_assessment=dj["hallucination_risk"],
        ),
    )


_mock_results = [_make(qid, q, a, dj) for qid, (q, a), dj in zip(_qids, _questions, _dj_data)]

# Also build rubric DJ results
_rubric_dj_results = []
for i, (qid, (q, a)) in enumerate(zip(_qids[:3], _questions[:3])):
    rid = VerificationResultMetadata.compute_result_id(qid, _answering, _parsing, _ts)
    _rubric_dj_results.append(
        VerificationResult(
            metadata=VerificationResultMetadata(
                question_id=qid,
                template_id="tmpl_" + qid[:8],
                failure=None,
                caveats=[],
                question_text=q,
                raw_answer=a,
                answering=_answering,
                parsing=_parsing,
                execution_time=6.0,
                timestamp=_ts,
                result_id=rid,
            ),
            template=VerificationResultTemplate(
                raw_llm_response=f"Based on evidence, {a.lower()}.",
                verify_result=True,
                template_verification_performed=True,
                parsed_gt_response={"answer": a},
                parsed_llm_response={"answer": a},
            ),
            rubric=VerificationResultRubric(
                rubric_evaluation_performed=True,
                rubric_evaluation_strategy="batch",
                llm_trait_scores={"accuracy": 4, "evidence_quality": 3},
            ),
            deep_judgment_rubric=VerificationResultDeepJudgmentRubric(
                deep_judgment_rubric_performed=True,
                extracted_rubric_excerpts={
                    "accuracy": [
                        {"text": "excerpt supporting accuracy", "confidence": "high", "similarity_score": 0.88}
                    ]
                },
                rubric_trait_reasoning={
                    "accuracy": "The response demonstrates accurate knowledge.",
                    "evidence_quality": "Sources are mentioned but not cited formally.",
                },
                deep_judgment_rubric_scores={"accuracy": 4, "evidence_quality": 3},
                total_deep_judgment_model_calls=4,
                total_traits_evaluated=2,
            ),
        )
    )

_mock_result_set = VerificationResultSet(results=_mock_results)
_rubric_dj_result_set = VerificationResultSet(results=_rubric_dj_results)
_call_count = 0
_orig_run = Benchmark.run_verification


def _patched_run(self, config, **kw):
    global _call_count
    _call_count += 1
    if config.deep_judgment_rubric_mode != "disabled":
        return _rubric_dj_result_set
    return _mock_result_set


Benchmark.run_verification = _patched_run
```

---

## What Deep Judgment Does

Deep judgment adds a multi-stage evaluation layer between parsing and final result:

```
Standard pipeline:                  With deep judgment:
Parse → Verify → Finalize          Parse → Extract Excerpts → Fuzzy Match
                                          → Reason → [Search] → Verify → Finalize
```

For each template attribute, deep judgment:
1. **Extracts excerpts** from the response that relate to the attribute
2. **Fuzzy matches** excerpts against ground truth (similarity scoring)
3. **Generates reasoning** explaining how the excerpt supports or contradicts the expected value
4. **Optionally searches** external sources to validate factual claims

---

## Enable DJ for Templates

```python
from karenina import Benchmark
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.config import ModelConfig

benchmark = Benchmark.load(str(_tmp))

config = VerificationConfig(
    answering_models=[
        ModelConfig(id="haiku", model_name="claude-haiku-4-5", model_provider="anthropic", interface="langchain")
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
    deep_judgment_mode="full",
)

results = benchmark.run_verification(config)
print(f"Results with DJ: {len(results)}")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deep_judgment_mode` | `Literal` | `"disabled"` | Template deep-judgment mode: `"disabled"`, `"reasoning_only"`, `"full"` |
| `deep_judgment_search_enabled` | `bool` | `False` | Enable external search validation |
| `deep_judgment_excerpt_retry_attempts` | `int` | `2` | Max retries for excerpt extraction |
| `deep_judgment_fuzzy_match_threshold` | `float` | `0.80` | Min similarity score for excerpt matching |

---

## Inspect DJ Template Results

### Extracted Excerpts

Each attribute in the template gets a list of supporting excerpts:

```python
for result in results:
    dj = result.deep_judgment
    if dj and dj.deep_judgment_performed:
        print(f"\nQ: {result.metadata.question_text[:50]}")
        for attr, excerpts in (dj.extracted_excerpts or {}).items():
            print(f"  Attribute: {attr}")
            for exc in excerpts:
                print(
                    f'    "{exc["text"][:60]}" (confidence: {exc["confidence"]}, similarity: {exc["similarity_score"]:.2f})'
                )
```

### Reasoning Traces

```python
for result in results[:2]:
    dj = result.deep_judgment
    if dj and dj.attribute_reasoning:
        print(f"\nQ: {result.metadata.question_text[:50]}")
        for attr, reasoning in dj.attribute_reasoning.items():
            print(f"  {attr}: {reasoning[:80]}...")
```

---

## DJ with Search Validation

Enable search to validate factual claims against external sources. Requires a search API key (e.g., Tavily):

```python
config_with_search = VerificationConfig(
    answering_models=[
        ModelConfig(id="haiku", model_name="claude-haiku-4-5", model_provider="anthropic", interface="langchain")
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
    deep_judgment_mode="full",
    deep_judgment_search_enabled=True,
)

print(f"DJ mode: {config_with_search.deep_judgment_mode}")
print(f"Search enabled: {config_with_search.deep_judgment_search_enabled}")
```

When search is enabled, each excerpt includes `hallucination_risk` (none/low/medium/high) and supporting search results:

```python
# Inspect search-validated results (question 5 has search data)
result = results[4]
dj = result.deep_judgment
if dj and dj.deep_judgment_search_enabled:
    print(f"Q: {result.metadata.question_text[:50]}")
    for attr, excerpts in (dj.extracted_excerpts or {}).items():
        for exc in excerpts:
            if "search_results" in exc:
                print(f'  Excerpt: "{exc["text"][:50]}"')
                print(f"  Hallucination risk: {exc.get('hallucination_risk', 'N/A')}")
                print(f"  Search: {exc['search_results'][:80]}...")
    if dj.hallucination_risk_assessment:
        print(f"  Risk assessment: {dj.hallucination_risk_assessment}")
```

---

## DJ for Rubrics

Deep judgment also works for rubric traits, providing per-trait excerpts and reasoning:

```python
config_rubric_dj = VerificationConfig(
    answering_models=[
        ModelConfig(id="haiku", model_name="claude-haiku-4-5", model_provider="anthropic", interface="langchain")
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
    evaluation_mode="template_and_rubric",
    deep_judgment_rubric_mode="enable_all",
)

rubric_dj_results = benchmark.run_verification(config_rubric_dj)
print(f"Rubric DJ results: {len(rubric_dj_results)}")
```

### Inspect Rubric DJ Results

```python
for result in rubric_dj_results:
    djr = result.deep_judgment_rubric
    if djr and djr.deep_judgment_rubric_performed:
        print(f"\nQ: {result.metadata.question_text[:40]}")
        # Per-trait reasoning
        for trait, reasoning in (djr.rubric_trait_reasoning or {}).items():
            score = (djr.deep_judgment_rubric_scores or {}).get(trait, "N/A")
            print(f"  {trait} (score={score}): {reasoning[:60]}...")
        # Per-trait excerpts
        for trait, excerpts in (djr.extracted_rubric_excerpts or {}).items():
            print(f"  {trait} excerpts: {len(excerpts)}")
```

| `deep_judgment_rubric_mode` | Behavior |
|-----------------------------|----------|
| `"disabled"` (default) | No rubric deep judgment |
| `"enable_all"` | Apply DJ to all LLM rubric traits |
| `"use_checkpoint"` | Use per-trait settings from checkpoint |
| `"custom"` | Use `deep_judgment_rubric_config` dict |

---

## CLI Equivalent

```python
# Enable template deep judgment via CLI:
# karenina verify benchmark.jsonld --preset base.json --deep-judgment

# With rubric deep judgment (search validation flag is rubric-only):
# karenina verify benchmark.jsonld --preset base.json --deep-judgment --deep-judgment-rubric-mode enable_all --deep-judgment-rubric-search

# Note: template-side search validation is preset-only (no CLI flag).

print("CLI: --deep-judgment, --deep-judgment-rubric-mode, --deep-judgment-rubric-search")
```

---

## Related Pages

- [Full Evaluation](full-evaluation.ipynb) — Template+rubric without deep judgment
- [Basic Verification](basic-verification.ipynb) — Simplest verification path
- [VerificationConfig Reference](../../reference/configuration/verification-config.md) — All DJ configuration fields
- [Advanced Pipeline](../../core_concepts/verification-pipeline.md) — Pipeline stage details

```python tags=["hide-cell"]
# Cleanup
Benchmark.run_verification = _orig_run
```
