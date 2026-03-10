# Rubrics

Rubrics evaluate **how** a model responded by assessing observable properties of the raw response trace, properties that do not require ground truth. While [answer templates](../../notebooks/core_concepts/answer-templates.ipynb) verify *what* the model said (factual correctness against a known answer), rubrics assess qualities like safety, conciseness, tone, or the presence of specific elements (citations, disclaimers).

Rubrics come in four trait types (LLM, regex, callable, metric) that work differently: some require an LLM call, others run locally with no model involved. They can be applied **globally** across all questions or **per-question** for domain-specific checks.

## 1. What Are Rubrics?

A **rubric** is a collection of evaluation traits that assess observable properties of an LLM response without requiring a ground-truth answer:

- **No ground truth needed**: rubrics evaluate properties you can judge by reading the response alone (conciseness, safety, presence of citations)
- **Complement templates**: templates check factual correctness via `verify()`; rubrics assess qualities that characterize the answer style or structure
- **Multiple trait types**: four types (LLM, regex, callable, metric) with different execution models

Unlike templates, which operate on parsed structured data, rubrics evaluate the **raw response text** directly. See [templates vs rubrics](../template-vs-rubric.md) for a full comparison of the two evaluation building blocks.

A `Rubric` in Karenina is a collector object that gathers traits of different types into separate lists:

```python
from karenina.schemas.entities.rubric import Rubric, LLMRubricTrait, RegexTrait

rubric = Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="conciseness",
            description="Is the response concise and free of unnecessary repetition?",
            kind="boolean",
            higher_is_better=True,
        ),
    ],
    regex_traits=[
        RegexTrait(
            name="has_citations",
            description="The response includes at least one citation.",
            pattern=r"\[\d+\]",
            higher_is_better=True,
        ),
    ],
    # callable_traits and metric_traits default to empty lists
)
```

## 2. Where Rubrics Attach

Once created, a rubric needs to be attached to an evaluation object. In benchmarks, the object it attaches to determines scope; in TaskEval, rubrics can be global or step-specific.

| Object | How to attach | Applies to | Use when | Evaluation behavior |
|--------|---------------|------------|----------|---------------------|
| **Benchmark** | Attach a rubric at the benchmark level, or add single traits with `benchmark.add_global_rubric_trait()` | Every question in the benchmark | You want the same quality checks across all responses, such as conciseness, tone, safety, or tool-grounding | Only benchmark-level traits are evaluated |
| **Question** | Attach a rubric at the question level, or add single traits with `benchmark.add_question_rubric_trait()` | One question only | A check only makes sense for a particular prompt, such as drug safety details or citation presence | Only question-level traits are evaluated |
| **Benchmark + Question** | Attach rubrics at both levels | The current question | You need both shared benchmark-wide traits and prompt-specific checks | Karenina merges both trait sets for that question; trait names must be unique across scopes or a `ValueError` is raised |
| **TaskEval** | Attach a rubric with `task_eval.add_rubric()`; pass `step_id` for step-specific evaluation | All recorded text or one named step | You are evaluating free-text output outside the benchmark loop | Traits evaluate against the TaskEval global scope or the selected step scope |

See [Full Evaluation Benchmark](../../notebooks/creating-benchmarks/full-evaluation-benchmark.ipynb) for benchmark usage and [TaskEval](../task-eval.md) for free-text evaluation. Each trait type has its own sub-page with full API details.

## 3. Trait Type Overview

Given the question "Which is the putative target of venetoclax?", a [template](../../notebooks/core_concepts/answer-templates.ipynb) checks whether the response identifies `BCL2` as the target (ground truth verification), while rubric traits assess other properties of the response:

| Trait Type | Returns | LLM Required | Example | Note |
|------------|---------|--------------|---------|------|
| [**LLMRubricTrait**](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) (boolean) | `bool` | Yes | "Mentions safety profile of the drug" | Supports optional [deep judgment](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) for evidence-based evaluation |
| [**LLMRubricTrait**](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) (score) | `int` | Yes | "Rate clarity from 1-5" | Configurable range |
| [**LLMRubricTrait**](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) (literal) | `int` | Yes | "Classify tone as formal/casual/technical" | Returns index based on class order; `higher_is_better` controls direction |
| [**RegexTrait**](../../notebooks/core_concepts/rubrics/regex-traits.ipynb) | `bool` | No | "Has bracket citations `[N]`" | 100% reproducible; supports `case_sensitive` and `invert_result` options |
| [**CallableTrait**](../../notebooks/core_concepts/rubrics/callable-traits.ipynb) | `bool` or `int` | No | "Under 150 words" | Created via `from_callable()`; serialized with cloudpickle; not available in GUI. Only load from trusted sources (deserialization executes arbitrary Python) |
| [**MetricRubricTrait**](../../notebooks/core_concepts/rubrics/metric-traits.ipynb) | metrics dict | Yes | "Expected drug interactions mentioned" | Two modes: `tp_only` (precision/recall/F1) and `full_matrix` (adds specificity/accuracy) |

Trait descriptions are not questions sent to the model; they are evaluation criteria applied to the response after the fact. Each trait type's sub-page includes a [pipeline diagram](../verification-pipeline.md) showing how evaluation works (RubricEvaluation).

No ground truth does not mean no specification. Rubric traits work better when the description makes your standard explicit. If you care about conciseness, say what that means in context: for example, "answers the question directly, avoids repetition, and stays under 120 words unless the prompt asks for detail." Clear trait descriptions improve the quality and consistency of evaluation even when no single correct answer exists.

See [templates vs rubrics](../template-vs-rubric.md) for a full comparison, and [evaluation modes](../evaluation-modes.md) for how to combine them in a single benchmark.

## 4. Choosing the Right Trait Type

| Need | Trait Type | Tutorial Example |
|------|-----------|-----------------|
| Subjective quality (clarity, conciseness, tone) | LLMRubricTrait (boolean or score) | [LLM score trait](../../notebooks/creating-benchmarks/choosing-rubric-traits.ipynb#need-1-subjective-quality-llm-score) |
| Categorical classification (quality tiers, tone levels) | LLMRubricTrait (literal) | [LLM literal trait](../../notebooks/creating-benchmarks/choosing-rubric-traits.ipynb#need-2-categorical-classification-llm-literal) |
| Exact keyword or format validation | RegexTrait | [Regex trait](../../notebooks/creating-benchmarks/choosing-rubric-traits.ipynb#need-3-exact-keywordformat-validation-regex) |
| Complex validation logic (word counts, structure) | CallableTrait | [Callable trait](../../notebooks/creating-benchmarks/choosing-rubric-traits.ipynb#need-4-complex-validation-logic-callable) |
| Precision/recall/F1 measurement | MetricRubricTrait | [Metric trait](../../notebooks/creating-benchmarks/choosing-rubric-traits.ipynb#need-5-precisionrecallf1-metric-trait) |
| Deterministic, reproducible check | RegexTrait or CallableTrait | [Inverted regex](../../notebooks/creating-benchmarks/choosing-rubric-traits.ipynb#need-6-deterministic-reproducible-check-regex-inverted) |
| Evidence-based evaluation with excerpts | LLMRubricTrait with deep judgment | [Deep judgment](../../notebooks/creating-benchmarks/choosing-rubric-traits.ipynb#need-7-evidence-based-with-excerpts-llm-boolean--deep-judgment) |

For a hands-on tutorial that walks through each of these needs with a complete example, see [Choosing the Right Rubric Trait Type](../../notebooks/creating-benchmarks/choosing-rubric-traits.ipynb).

### Decision Flowchart

```
1. Does the check require language understanding?
   │
   ├─ NO: Can it be expressed as a single regex pattern?
   │   │
   │   ├─ YES → RegexTrait
   │   │        Check presence: higher_is_better=True
   │   │        Check absence:  invert_result=True
   │   │
   │   └─ NO (multiple patterns, numeric logic, conditionals)
   │       → CallableTrait
   │         Accepts one str, returns bool or int.
   │
   └─ YES: Is it a checklist of items the response should cover?
       │
       ├─ YES → MetricRubricTrait
       │        Coverage only: evaluation_mode="tp_only"
       │        Coverage + absence: evaluation_mode="full_matrix"
       │
       └─ NO: What kind of judgment?
           │
           ├─ Yes/no → LLMRubricTrait (kind="boolean")
           │           Need traceable evidence? Add deep_judgment_enabled=True
           │
           ├─ Named tiers with observable boundaries
           │   → LLMRubricTrait (kind="literal")
           │     Write mutually exclusive class descriptions.
           │
           └─ Continuous scale (no clear category boundaries)
               → LLMRubricTrait (kind="score")
                 Anchor the scale at 3+ points with concrete criteria.
```

**Priority heuristic**: prefer deterministic traits (regex, callable) over LLM traits when possible. They are faster, cheaper, and perfectly reproducible. Use LLM traits only when the evaluation genuinely requires language understanding.

## 5. The `higher_is_better` Field

All trait types (except MetricRubricTrait, where metrics are inherently "higher is better") include a `higher_is_better` field that controls directionality:

- **Boolean traits**: `True` means `True` is a positive outcome
- **Score traits**: `True` means higher scores indicate better performance
- **Literal traits**: `True` means later classes (higher indices) are better
- **Regex traits**: `True` means a match indicates a positive outcome

This field is used by analysis tools and DataFrame builders to correctly interpret and aggregate rubric results. It is also crucial for the GEPA optimization procedure, which relies on `higher_is_better` to determine the direction of improvement when optimizing prompts against rubric scores. GEPA documentation is forthcoming.

## 6. Next Steps

- [LLM traits](../../notebooks/core_concepts/rubrics/llm-traits.ipynb): boolean and score kinds with deep judgment
- [Literal traits](../../notebooks/core_concepts/rubrics/llm-traits.ipynb): ordered categorical classification (part of LLM traits)
- [Regex traits](../../notebooks/core_concepts/rubrics/regex-traits.ipynb): deterministic pattern matching
- [Callable traits](../../notebooks/core_concepts/rubrics/callable-traits.ipynb): custom Python functions
- [Metric traits](../../notebooks/core_concepts/rubrics/metric-traits.ipynb): precision, recall, F1 computation
- [Evaluation modes](../evaluation-modes.md): template_only, template_and_rubric, rubric_only
- [Full Evaluation Benchmark](../../notebooks/creating-benchmarks/full-evaluation-benchmark.ipynb): workflow guide for adding rubrics to benchmarks
