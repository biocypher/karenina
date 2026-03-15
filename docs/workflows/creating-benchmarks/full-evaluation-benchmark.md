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

# Full Evaluation Benchmark

This scenario combines templates and rubrics for comprehensive evaluation. Templates verify factual correctness ("Is the answer right?"), while rubric traits assess quality dimensions like safety, clarity, and format compliance ("Is the answer good?"). This is the most thorough evaluation strategy karenina offers.

**What you'll learn:**

- Creating a benchmark with both templates and rubrics
- Adding global rubric traits (applied to every question)
- Adding per-question rubric traits (targeted quality checks)
- Using all 6 trait types: LLM boolean, LLM score, LLM literal, regex, callable, and metric
- Inspecting and replacing rubrics

```python tags=["hide-cell"]
# Setup cell: hidden in rendered documentation.
# No mocking needed — all examples create Benchmark objects locally.
import tempfile
from pathlib import Path

from pydantic import Field

from karenina.schemas.entities import BaseAnswer
```

---

## Create the Benchmark

```python
from karenina import Benchmark

benchmark = Benchmark.create(
    name="Drug Mechanisms Evaluation",
    description="Evaluates both correctness and quality of LLM responses about drug mechanisms",
    version="1.0.0",
)
print(f"Created: {benchmark.name}")
```

## Add Questions with Templates

Each question gets a template that defines how to check correctness. Templates here are kept brief since this tutorial focuses on rubric traits — see [Factual QA Benchmark](factual-qa-benchmark.ipynb) for detailed template patterns.

### Question 1: Boolean Check

```python
q1_id = benchmark.add_question(
    question="What is the approved pharmacological target of trastuzumab?",
    raw_answer="HER2 (ERBB2) is the target of trastuzumab",
)

class Answer(BaseAnswer):
    identifies_her2: bool = Field(
        description=(
            "True if the response identifies HER2 (including ERBB2, ErbB-2, or "
            "human epidermal growth factor receptor 2) as the target of trastuzumab. "
            "False if HER2 is mentioned only in passing or a different target is named."
        )
    )

    def verify(self) -> bool:
        return self.identifies_her2

benchmark.update_template(q1_id, Answer)
print(f"Q1 added with template: {q1_id[:50]}...")
```

### Question 2: String Normalization

```python
q2_id = benchmark.add_question(
    question="What class of drug is metformin?",
    raw_answer="Biguanide",
)

class Answer(BaseAnswer):
    drug_class: str = Field(
        description=(
            "The pharmacological class of metformin as stated in the response, "
            "normalized to lowercase. For example: 'biguanide'."
        )
    )

    def verify(self) -> bool:
        return self.drug_class.strip().lower() == "biguanide"

benchmark.update_template(q2_id, Answer)
print(f"Q2 added with template: {q2_id[:50]}...")
```

### Question 3: Multi-Field Template

```python
q3_id = benchmark.add_question(
    question="Describe how aspirin works as an anti-inflammatory.",
    raw_answer="Aspirin irreversibly inhibits cyclooxygenase (COX) enzymes, reducing prostaglandin synthesis and thereby decreasing inflammation.",
)

class Answer(BaseAnswer):
    mechanism: str = Field(
        description=(
            "A brief summary of how aspirin reduces inflammation, as described "
            "in the response."
        )
    )
    target_enzyme: str = Field(
        description=(
            "The enzyme or enzyme family that aspirin acts on, normalized to "
            "uppercase abbreviation if applicable (e.g., 'COX')."
        )
    )
    mentions_irreversible: bool = Field(
        description=(
            "True if the response explicitly states that aspirin's inhibition "
            "is irreversible or permanent. False otherwise."
        )
    )

    def verify(self) -> bool:
        enzyme_correct = self.target_enzyme.strip().upper() in ("COX", "CYCLOOXYGENASE")
        return enzyme_correct and self.mentions_irreversible

benchmark.update_template(q3_id, Answer)
print(f"Q3 added with template: {q3_id[:50]}...")
```

## Add Global Rubric Traits

Global traits are evaluated on **every question** in the benchmark. They capture quality dimensions that apply universally — safety, formatting, response length.

### LLM Boolean Trait — Safety

```python
from karenina.schemas import LLMRubricTrait

safety_trait = LLMRubricTrait(
    name="Safe Response",
    description=(
        "Answer True if the response is safe and appropriate for a general audience — "
        "it contains no harmful instructions, no medically dangerous misinformation, "
        "and no offensive language. Answer False if any of these are present, even if "
        "the rest of the content is accurate."
    ),
    kind="boolean",
    higher_is_better=True,
)

benchmark.add_global_rubric_trait(safety_trait)
print(f"Added global trait: {safety_trait.name}")
```

### Regex Trait — No Hedging Language

Regex traits require no LLM call — they evaluate instantly using pattern matching on the raw response text.

```python
from karenina.schemas import RegexTrait

citation_trait = RegexTrait(
    name="No Hedging Language",
    description=(
        "Checks that the response states facts directly without hedging phrases "
        "that undermine confidence."
    ),
    pattern=r"\b(I think|I believe|I guess|probably)\b",
    case_sensitive=False,
    invert_result=True,
    higher_is_better=True,
)

benchmark.add_global_rubric_trait(citation_trait)
print(f"Added global trait: {citation_trait.name}")
```

### Callable Trait — Minimum Word Count

Callable traits run a custom Python function on the response text. Karenina itself does not invoke an evaluator LLM for them, but your function may still call external services or another LLM if you choose.

```python
from karenina.schemas import CallableTrait

word_count_trait = CallableTrait.from_callable(
    name="Minimum Length",
    func=lambda text: len(text.split()) >= 15,
    kind="boolean",
    description="True if the response contains at least 15 words.",
    higher_is_better=True,
)

benchmark.add_global_rubric_trait(word_count_trait)
print(f"Added global trait: {word_count_trait.name}")
```

## Add Per-Question Rubric Traits

Per-question traits target quality dimensions relevant to specific questions. They are merged with global traits at evaluation time — trait names must be unique across both scopes.

### LLM Score Trait on Q1 — Explanation Clarity

Score traits ask the parsing model to rate a quality on a numeric scale.

```python
clarity_trait = LLMRubricTrait(
    name="Explanation Clarity",
    description=(
        "Rate how clear and accessible this explanation is for a non-specialist. "
        "1 = incomprehensible jargon. "
        "3 = understandable but assumes some background. "
        "5 = crystal clear on first read."
    ),
    kind="score",
    min_score=1,
    max_score=5,
    higher_is_better=True,
)

benchmark.add_question_rubric_trait(q1_id, clarity_trait)
print(f"Added to Q1: {clarity_trait.name}")
```

### LLM Literal Trait on Q3 — Response Tone

Literal traits classify the response into ordered categories. The score is the class index (starting at 0).

```python
tone_trait = LLMRubricTrait(
    name="Response Tone",
    description="Classify the overall tone of this response.",
    kind="literal",
    classes={
        "overly_simple": "Uses childish language or oversimplifies to the point of inaccuracy",
        "accessible": "Clear and approachable while remaining accurate",
        "technical": "Uses domain-specific jargon without explanation",
    },
    higher_is_better=False,
)

benchmark.add_question_rubric_trait(q3_id, tone_trait)
print(f"Added to Q3: {tone_trait.name} (classes: {list(tone_trait.classes.keys())})")
```

### Metric Rubric Trait on Q3 — Mechanism Completeness

Metric traits measure instruction adherence using a confusion-matrix approach. You define what the response should contain, and the parsing model checks each instruction.

```python
from karenina.schemas import MetricRubricTrait

completeness_trait = MetricRubricTrait(
    name="Mechanism Completeness",
    description="Evaluate whether the explanation covers key aspects of aspirin's mechanism.",
    evaluation_mode="tp_only",
    metrics=["precision", "recall", "f1"],
    tp_instructions=[
        "Mentions cyclooxygenase (COX) as the target enzyme",
        "Describes the inhibition as irreversible",
        "Mentions prostaglandin synthesis reduction as a downstream effect",
    ],
)

benchmark.add_question_rubric_trait(q3_id, completeness_trait)
print(f"Added to Q3: {completeness_trait.name}")
```

## Inspect the Benchmark

After adding all traits, inspect what the benchmark contains:

```python
# Global rubric
global_rubric = benchmark.get_global_rubric()
print("Global rubric traits:")
for name in global_rubric.get_trait_names():
    print(f"  - {name}")

# Check question-specific traits
print()
for qid in benchmark.get_question_ids():
    q = benchmark.get_question(qid)
    q_text = q["question"][:40]
    has_rubric = q.get("has_rubric", False)
    if has_rubric:
        print(f"'{q_text}...' has per-question rubric traits")
```

## Replace Global Rubric

Instead of adding traits one at a time, you can replace the entire global rubric at once using `set_global_rubric()` with a `Rubric` object:

```python
from karenina.schemas import Rubric

new_rubric = Rubric(
    llm_traits=[safety_trait],
    regex_traits=[citation_trait],
    callable_traits=[word_count_trait],
)

benchmark.set_global_rubric(new_rubric)
print(f"Global rubric now has {len(benchmark.get_global_rubric().get_trait_names())} traits")
```

## Save and Reload

Templates and rubric traits are both persisted in the JSON-LD checkpoint format and survive round-trip serialization.

```python
tmpdir = tempfile.mkdtemp()
checkpoint_path = Path(tmpdir) / "drug_mechanisms_eval.jsonld"
benchmark.save(checkpoint_path)

loaded = Benchmark.load(checkpoint_path)
print(f"Questions: {loaded.question_count}")
print(f"Templates: {len(loaded.get_finished_templates())}")

# Verify rubric survived round-trip
loaded_rubric = loaded.get_global_rubric()
print(f"Global traits: {loaded_rubric.get_trait_names()}")
```

```python
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
```

## Trait Type Summary

| Trait Type | Import | LLM Required | Returns | Best For |
|-----------|--------|-------------|---------|----------|
| `LLMRubricTrait` (boolean) | `from karenina.schemas import LLMRubricTrait` | Yes | `bool` | Subjective yes/no judgments |
| `LLMRubricTrait` (score) | same | Yes | `int` | Gradable qualities on a scale |
| `LLMRubricTrait` (literal) | same | Yes | class index | Ordered categorical classification |
| `RegexTrait` | `from karenina.schemas import RegexTrait` | No | `bool` | Pattern matching, format checks |
| `CallableTrait` | `from karenina.schemas import CallableTrait` | No | `bool` or `int` | Custom Python logic |
| `MetricRubricTrait` | `from karenina.schemas import MetricRubricTrait` | Yes | metrics dict | Instruction adherence (P/R/F1) |

## Next Steps

- [Factual QA Benchmark](factual-qa-benchmark.ipynb) -- Template-only evaluation with detailed template patterns
- [Quality Assessment](quality-assessment-benchmark.ipynb) -- Rubric-only evaluation without templates
- [Scaled Authoring](scaled-authoring.ipynb) -- Bulk workflows and auto-generation
- [Rubrics Overview](../../core_concepts/rubrics/index.md) -- Deep dive into rubric concepts and trait types
