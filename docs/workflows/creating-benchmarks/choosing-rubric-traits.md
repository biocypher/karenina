---
jupyter:
  jupytext:
    formats: docs/workflows/creating-benchmarks//md,docs/notebooks/creating-benchmarks//ipynb
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

# Choosing the Right Rubric Trait Type

Karenina offers six trait types (LLM boolean, LLM score, LLM literal, regex, callable, metric) that cover seven distinct evaluation needs. This tutorial is organized by **need**: "I need to check X. Which trait type solves it?" For each need, you will create a trait, attach it to a benchmark, and learn when to prefer one type over another.

This complements the build-focused tutorials ([Full Evaluation Benchmark](full-evaluation-benchmark.ipynb), [Quality Assessment](quality-assessment-benchmark.ipynb)) by providing a decision framework for trait selection.

**What you'll learn:**

- Seven evaluation needs and which trait type addresses each
- When to prefer score vs boolean, literal vs score, callable vs regex
- Mixing global and per-question scoping in a single benchmark
- A decision flowchart for rapid trait selection

```python tags=["hide-cell"]
# Setup cell: hidden in rendered documentation.
# No mocking needed — all examples create Benchmark objects locally.
import re
import tempfile
from pathlib import Path

from pydantic import Field

from karenina.schemas.entities import BaseAnswer
```

---

## The Scenario

You are evaluating LLM responses about **pembrolizumab (Keytruda)**, a PD-1 checkpoint inhibitor used in non-small cell lung cancer (NSCLC). Three questions cover mechanism, clinical evidence, and clinical decision-making; together they exercise all seven evaluation needs.

```python
from karenina import Benchmark

benchmark = Benchmark.create(
    name="Pembrolizumab Evaluation",
    description="Need-driven rubric trait selection for pembrolizumab NSCLC responses",
    version="1.0.0",
)
print(f"Created: {benchmark.name}")
```

### Question 1: Mechanism

```python
q1_id = benchmark.add_question(
    question="Explain how pembrolizumab works as a cancer treatment.",
    raw_answer=(
        "Pembrolizumab is a humanized monoclonal antibody that binds to the PD-1 "
        "receptor on T cells, blocking the interaction between PD-1 and its ligands "
        "PD-L1 and PD-L2. This releases the immune checkpoint brake, restoring T cell "
        "mediated anti-tumor immune responses. By preventing tumor cells from exploiting "
        "the PD-1 pathway to evade immune detection, pembrolizumab enables the immune "
        "system to recognize and destroy cancer cells."
    ),
)


class Answer(BaseAnswer):
    identifies_pd1_target: bool = Field(
        description=(
            "True if the response identifies PD-1 (programmed death-1, programmed "
            "cell death protein 1, or CD279) as the molecular target of pembrolizumab. "
            "False if PD-1 is mentioned only as background context or a different "
            "target is named."
        )
    )

    def verify(self) -> bool:
        return self.identifies_pd1_target


benchmark.update_template(q1_id, Answer)
print(f"Q1 added: {q1_id[:50]}...")
```

### Question 2: Clinical Evidence

```python
q2_id = benchmark.add_question(
    question="What clinical evidence supports pembrolizumab as first-line therapy in NSCLC?",
    raw_answer=(
        "The KEYNOTE-024 trial demonstrated that pembrolizumab monotherapy significantly "
        "improved progression-free survival compared to platinum-based chemotherapy in "
        "patients with PD-L1 TPS >= 50%. KEYNOTE-042 extended this to patients with "
        "TPS >= 1%. KEYNOTE-189 showed that pembrolizumab combined with chemotherapy "
        "improved overall survival regardless of PD-L1 expression. KEYNOTE-407 confirmed "
        "similar benefits in squamous NSCLC."
    ),
)


class Answer(BaseAnswer):
    mentions_keynote_trial: bool = Field(
        description=(
            "True if the response mentions at least one KEYNOTE trial by name "
            "(e.g., KEYNOTE-024, KEYNOTE-189). False if clinical evidence is "
            "discussed without naming specific trials."
        )
    )

    def verify(self) -> bool:
        return self.mentions_keynote_trial


benchmark.update_template(q2_id, Answer)
print(f"Q2 added: {q2_id[:50]}...")
```

### Question 3: Clinical Decision

```python
q3_id = benchmark.add_question(
    question=(
        "When should pembrolizumab be considered over chemotherapy for NSCLC, and what biomarker testing is required?"
    ),
    raw_answer=(
        "Pembrolizumab should be considered as first-line monotherapy when PD-L1 "
        "tumor proportion score (TPS) is >= 50%, based on KEYNOTE-024. For TPS 1-49%, "
        "pembrolizumab combined with platinum-based chemotherapy is preferred per "
        "KEYNOTE-189/407. PD-L1 testing by immunohistochemistry (IHC) using the 22C3 "
        "pharmDx assay is required before initiating therapy. EGFR and ALK testing "
        "should also be performed to exclude patients who would benefit more from "
        "targeted therapy."
    ),
)


class Answer(BaseAnswer):
    mentions_pdl1_testing: bool = Field(
        description=(
            "True if the response states that PD-L1 expression testing is required "
            "or recommended before pembrolizumab treatment. False if PD-L1 is not "
            "mentioned in the context of testing or biomarker assessment."
        )
    )

    def verify(self) -> bool:
        return self.mentions_pdl1_testing


benchmark.update_template(q3_id, Answer)
print(f"Q3 added: {q3_id[:50]}...")
```

Templates are kept minimal here because the tutorial focus is rubric traits. See [Factual QA Benchmark](factual-qa-benchmark.ipynb) for detailed template patterns.

---

## Need 1: Subjective Quality (LLM Score)

**The need**: Rate a continuous quality dimension (clarity, readability, organization) where a simple pass/fail is too coarse.

A score trait asks the judge LLM to rate the response on a numeric scale. Anchor the scale at multiple points with concrete, observable criteria so the judge has clear guidance.

```python
from karenina.schemas import LLMRubricTrait

clarity_trait = LLMRubricTrait(
    name="Explanation Clarity",
    description=(
        "Rate how clearly this response explains its content to a healthcare "
        "professional who is not an oncologist. "
        "1 = disorganized or incomprehensible, key terms undefined. "
        "2 = understandable but poorly structured. "
        "3 = adequate, follows a logical order, defines most terms. "
        "4 = clear and well-organized, accessible on first read. "
        "5 = exceptionally clear, uses effective structure and plain language "
        "while maintaining scientific accuracy."
    ),
    kind="score",
    min_score=1,
    max_score=5,
    higher_is_better=True,
)

benchmark.add_global_rubric_trait(clarity_trait)
print(f"Added global trait: {clarity_trait.name} (score 1-5)")
```

This trait is **global** because clarity matters for every question, not just one.

<div class="admonition tip">
<p class="admonition-title">When to prefer boolean over score</p>
<p>Use a boolean LLM trait when the quality has a sharp boundary: "Is this safe?" or "Does it include a disclaimer?" Use score when the quality is genuinely continuous and you want to distinguish degrees. If your scale anchors collapse into two meaningful levels, simplify to boolean.</p>
</div>

---

## Need 2: Categorical Classification (LLM Literal)

**The need**: Classify a response into one of several categories where each category is qualitatively distinct.

A literal trait asks the judge to pick a named category rather than a number. This works for both ordered tiers (poor → excellent) and **nominal** categories where no class is inherently better than another. Here, the categories describe the target audience of a response. The language register is a strong, observable signal: a patient-facing response avoids abbreviations and uses reassuring tone, while a clinician-facing response assumes terminology and jumps to decision points. A response cannot target two audiences simultaneously, making the classes mutually exclusive.

```python
audience_trait = LLMRubricTrait(
    name="Target Audience",
    description="Classify the target audience of this response based on its language register.",
    kind="literal",
    classes={
        "patient": (
            "Written for a patient or caregiver. Uses plain language, avoids or "
            "defines medical abbreviations (spells out 'PD-L1' or explains it), "
            "includes reassuring context ('this is a standard treatment')."
        ),
        "medical_student": (
            "Written for a learner. Defines technical terms on first use, "
            "explains mechanisms step by step, provides background context "
            "that a practicing clinician would already know."
        ),
        "clinician": (
            "Written for a practicing physician. Uses medical abbreviations "
            "without definition (PFS, OS, TPS, IHC), assumes familiarity with "
            "trial design, focuses on actionable decision points."
        ),
        "researcher": (
            "Written for a scientific audience. Emphasizes molecular mechanisms, "
            "statistical endpoints (hazard ratios, confidence intervals), or "
            "study methodology over clinical applicability."
        ),
    },
    higher_is_better=True,
)

benchmark.add_question_rubric_trait(q2_id, audience_trait)
print(f"Added to Q2: {audience_trait.name} (classes: {list(audience_trait.classes.keys())})")
```

This trait is **per-question** on Q2 because audience register is most revealing for the clinical evidence question, where the same facts can be communicated in very different registers.

<div class="admonition tip">
<p class="admonition-title">Literal for nominal (unordered) categories</p>
<p>Literal traits return a class index (0, 1, 2, ...) based on dict order, and <code>higher_is_better</code> tells analysis tools the direction. For <strong>nominal</strong> categories like these, the ordering is arbitrary: "patient" is not worse than "researcher." The trait is still useful for classification and for tracking which register a model defaults to across runs. Set <code>higher_is_better</code> to whichever direction is convenient for your analysis; it has no semantic meaning for unordered classes.</p>
</div>

<div class="admonition tip">
<p class="admonition-title">Literal vs score: when to use which</p>
<p>Use literal when you can define <strong>qualitatively distinct</strong> categories with observable criteria (whether ordered or not). Use score when the dimension is genuinely continuous and you cannot define meaningful category boundaries. If you find yourself writing score anchors that sound like category definitions, switch to literal.</p>
</div>

---

## Need 3: Exact Keyword/Format Validation (Regex)

**The need**: Check whether a mechanical pattern (keyword, format, disclaimer) is present. No judgment required.

Regex traits run instantly with no LLM call, are perfectly reproducible, and cost nothing. Use them for checks that can be expressed as a pattern match.

```python
from karenina.schemas import RegexRubricTrait

disclaimer_trait = RegexRubricTrait(
    name="Safety Disclaimer",
    description=(
        "Checks that the response includes a safety disclaimer directing patients to consult a healthcare professional."
    ),
    pattern=r"consult\s+(a\s+|your\s+)?(doctor|physician|healthcare\s+(professional|provider))",
    case_sensitive=False,
    higher_is_better=True,
)

benchmark.add_global_rubric_trait(disclaimer_trait)
print(f"Added global trait: {disclaimer_trait.name}")
```

This trait is **global** because every response about a cancer drug should include a safety disclaimer.

<div class="admonition note">
<p class="admonition-title">Regex limitations</p>
<p>Regex traits cannot understand context. The pattern <code>consult.*doctor</code> matches whether the response genuinely advises consulting a doctor or merely mentions the word in passing. For context-dependent checks, use an LLM boolean trait instead. Reserve regex for mechanical validation where the presence of specific text is sufficient.</p>
</div>

---

## Need 4: Complex Validation Logic (Callable)

**The need**: Validate something that requires precise computation: counting, arithmetic, or compound logic that an LLM cannot do reliably.

LLMs are notoriously imprecise at counting characters, words, or items. A callable trait runs a Python function on the response text, giving you exact, deterministic results for any computation.

```python
from karenina.schemas import CallableRubricTrait


def check_response_length(text: str) -> bool:
    """Verify that a clinical decision response meets length requirements.

    Clinical decision guidance should be substantive (at least 200 characters)
    but concise enough for quick reference (under 2000 characters).
    """
    char_count = len(text.strip())
    return 200 <= char_count <= 2000


length_trait = CallableRubricTrait.from_callable(
    name="Response Length",
    func=check_response_length,
    kind="boolean",
    description=(
        "True if the response is between 200 and 2000 characters. "
        "Clinical decision guidance must be substantive enough to be useful "
        "but concise enough for quick reference."
    ),
    higher_is_better=True,
)

benchmark.add_question_rubric_trait(q3_id, length_trait)
print(f"Added to Q3: {length_trait.name}")
```

This trait is **per-question** on Q3 because the length constraint is specific to clinical decision guidance; mechanism explanations (Q1) or evidence summaries (Q2) may have different length expectations.

<div class="admonition tip">
<p class="admonition-title">Callable vs regex: where to draw the line</p>
<p>If the check is pure pattern matching (presence or absence of text), use RegexRubricTrait. If it requires <strong>counting</strong>, <strong>arithmetic</strong>, or <strong>combining multiple conditions</strong>, use CallableRubricTrait. An LLM score trait that asks "rate the response length from 1-5" introduces variance on something you can measure exactly. When precision matters, compute; do not judge.</p>
</div>

<div class="admonition warning">
<p class="admonition-title">Security note</p>
<p>CallableRubricTrait serializes functions with cloudpickle. Only load benchmarks containing callable traits from trusted sources, as deserialization executes arbitrary Python. Callable traits are not available in the GUI for this reason.</p>
</div>

---

## Need 5: Precision/Recall/F1 (Metric Trait)

**The need**: Measure how completely a response covers a checklist of expected items, with quantitative precision and recall.

Metric traits define a set of atomic instructions and have the judge LLM check each one. The result is a confusion matrix with precision, recall, and F1.

```python
from karenina.schemas import MetricRubricTrait

trial_coverage_trait = MetricRubricTrait(
    name="KEYNOTE Trial Coverage",
    description="Evaluate coverage of the major KEYNOTE trials supporting pembrolizumab in NSCLC.",
    evaluation_mode="tp_only",
    metrics=["precision", "recall", "f1"],
    tp_instructions=[
        "Mentions KEYNOTE-024 (pembrolizumab monotherapy, PD-L1 TPS >= 50%)",
        "Mentions KEYNOTE-042 (pembrolizumab monotherapy, PD-L1 TPS >= 1%)",
        "Mentions KEYNOTE-189 (pembrolizumab + chemo, non-squamous NSCLC)",
        "Mentions KEYNOTE-407 (pembrolizumab + chemo, squamous NSCLC)",
    ],
)

benchmark.add_question_rubric_trait(q2_id, trial_coverage_trait)
print(f"Added to Q2: {trial_coverage_trait.name} ({len(trial_coverage_trait.tp_instructions)} instructions)")
```

This trait is **per-question** on Q2 because the trial checklist is specific to the clinical evidence question.

<div class="admonition tip">
<p class="admonition-title">tp_only vs full_matrix</p>
<p>Use <code>tp_only</code> when you have a checklist of things the response <strong>should</strong> mention and you want to measure coverage (recall) and relevance (precision). Use <code>full_matrix</code> when you also need to verify that certain claims are <strong>absent</strong> (true negatives), enabling specificity and accuracy metrics. Most use cases start with <code>tp_only</code>; add <code>tn_instructions</code> and switch to <code>full_matrix</code> only when you need to penalize false claims.</p>
</div>

---

## Need 6: Deterministic, Reproducible Check (Regex Inverted)

**The need**: Ensure a pattern is **absent** from the response, with zero cost, zero variance, and full reproducibility.

When reproducibility matters (regulatory contexts, automated pipelines, CI gates), prefer deterministic traits. An inverted regex trait checks that an undesirable pattern does not appear.

```python
no_hedging_trait = RegexRubricTrait(
    name="No Clinical Hedging",
    description=("Checks that the response does not use hedging language that could undermine clinical confidence."),
    pattern=r"\b(I think|I believe|I guess|probably|it seems like|I'm not sure)\b",
    case_sensitive=False,
    invert_result=True,
    higher_is_better=True,
)

benchmark.add_global_rubric_trait(no_hedging_trait)
print(f"Added global trait: {no_hedging_trait.name} (inverted regex)")
```

This trait is **global** because hedging language is undesirable in any clinical response.

<div class="admonition note">
<p class="admonition-title">invert_result vs higher_is_better</p>
<p>These fields serve different purposes. <code>invert_result</code> flips the regex match result: True becomes "pattern NOT found" (a positive outcome). <code>higher_is_better</code> tells analysis tools how to interpret the score. For absence checks, set both to <code>True</code>: the trait returns True when the undesirable pattern is absent, and True is a good outcome.</p>
</div>

---

## Need 7: Evidence-Based with Excerpts (LLM Boolean + Deep Judgment)

**The need**: Make an LLM judgment transparent and auditable by extracting verbatim excerpts that support the evaluation.

Deep judgment enhances an LLM trait by requesting the judge to quote specific passages from the response. This creates a traceable evidence chain: you can inspect exactly which text the judge relied on.

```python
mechanism_trait = LLMRubricTrait(
    name="Molecular Mechanism Depth",
    description=(
        "Answer True if the response explains the molecular mechanism of "
        "pembrolizumab at the receptor level: specifically, that it binds PD-1 "
        "on T cells and blocks PD-L1/PD-L2 interaction, thereby restoring "
        "anti-tumor immune response. Answer False if the response only states "
        "that pembrolizumab is an immunotherapy or checkpoint inhibitor without "
        "explaining the receptor-level mechanism."
    ),
    kind="boolean",
    higher_is_better=True,
    deep_judgment_enabled=True,
    deep_judgment_excerpt_enabled=True,
    deep_judgment_max_excerpts=3,
)

benchmark.add_question_rubric_trait(q1_id, mechanism_trait)
print(
    f"Added to Q1: {mechanism_trait.name} (deep judgment, up to {mechanism_trait.deep_judgment_max_excerpts} excerpts)"
)
```

This trait is **per-question** on Q1 because receptor-level mechanism depth only applies to the mechanism question.

<div class="admonition tip">
<p class="admonition-title">When deep judgment is overkill</p>
<p>Deep judgment adds an extra LLM call per trait per question. Use it when you need <strong>auditability</strong> (why did the judge decide True/False?), <strong>transparency</strong> (what text evidence supports the score?), or <strong>debugging</strong> (is the judge reading the response correctly?). For high-volume benchmarks where you trust the trait description and do not need per-response justification, skip deep judgment to save cost and latency.</p>
</div>

---

## Inspect the Complete Benchmark

Review all traits attached to the benchmark, both global and per-question:

```python
# Global traits
global_rubric = benchmark.get_global_rubric()
print("Global rubric traits (applied to every question):")
for name in global_rubric.get_trait_names():
    print(f"  - {name}")

# Per-question traits
print()
for qid in benchmark.get_question_ids():
    q = benchmark.get_question(qid)
    q_text = q["question"][:60]
    has_rubric = q.get("question_rubric")
    if has_rubric:
        print(f"'{q_text}...' has question-level rubric")
    else:
        print(f"'{q_text}...' uses global traits only")
```

---

## Decision Flowchart

See the [decision flowchart on the Rubrics concept page](../../../core_concepts/rubrics/#decision-flowchart) for a quick reference that maps evaluation needs to trait types.

---

## Save and Reload

Verify the complete benchmark survives round-trip serialization:

```python
tmpdir = tempfile.mkdtemp()
checkpoint_path = Path(tmpdir) / "pembrolizumab_eval.jsonld"
benchmark.save(checkpoint_path)

loaded = Benchmark.load(checkpoint_path)
print(f"Questions: {loaded.question_count}")
print(f"Templates: {len(loaded.get_finished_templates())}")

# Verify rubrics survived round-trip
loaded_rubric = loaded.get_global_rubric()
print(f"Global traits: {loaded_rubric.get_trait_names()}")
```

---

## Cleanup

```python
import shutil

shutil.rmtree(tmpdir, ignore_errors=True)
```

---

## Trait-to-Need Summary

| Need | Trait Type | Scope | Question |
|------|-----------|-------|---------|
| 1. Subjective quality | LLMRubricTrait (score) | Global | All |
| 2. Categorical classification | LLMRubricTrait (literal, nominal) | Per-question | Q2 |
| 3. Format validation | RegexRubricTrait | Global | All |
| 4. Complex validation (counting) | CallableRubricTrait | Per-question | Q3 |
| 5. Precision/recall/F1 | MetricRubricTrait (tp_only) | Per-question | Q2 |
| 6. Deterministic check | RegexRubricTrait (inverted) | Global | All |
| 7. Evidence-based | LLMRubricTrait + deep judgment | Per-question | Q1 |

## Next Steps

- [Full Evaluation Benchmark](full-evaluation-benchmark.ipynb): Build a complete template + rubric benchmark from scratch
- [Quality Assessment](quality-assessment-benchmark.ipynb): Rubric-only evaluation for tasks with no single correct answer
- [Rubrics Overview](../../core_concepts/rubrics/index.md): Deep dive into rubric concepts, the `higher_is_better` field, and trait type API details
- [LLM Traits](../../notebooks/core_concepts/rubrics/llm-traits.ipynb): Boolean, score, and literal kinds with deep judgment
- [Regex Traits](../../notebooks/core_concepts/rubrics/regex-traits.ipynb): Deterministic pattern matching
- [Callable Traits](../../notebooks/core_concepts/rubrics/callable-traits.ipynb): Custom Python functions
- [Metric Traits](../../notebooks/core_concepts/rubrics/metric-traits.ipynb): Precision, recall, F1 computation
