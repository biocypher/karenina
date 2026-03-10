# Rubrics

Rubrics evaluate **how** a model responded by assessing observable properties of the raw response trace, properties that do not require ground truth. While [answer templates](../../notebooks/core_concepts/answer-templates.ipynb) verify *what* the model said (factual correctness against a known answer), rubrics assess qualities like safety, conciseness, tone, or the presence of specific elements (citations, disclaimers).

Rubrics come in four trait types (LLM, regex, callable, metric) that work differently: some require an LLM call, others run locally with no model involved. They can be applied **globally** across all questions or **per-question** for domain-specific checks.

## 1. What Are Rubrics?

A **rubric** is a collection of evaluation traits that assess observable properties of an LLM response without requiring a ground-truth answer:

- **No ground truth needed**: rubrics evaluate properties you can judge by reading the response alone (conciseness, safety, presence of citations)
- **Complement templates**: templates check factual correctness via `verify()`; rubrics assess qualities that characterize the answer style or structure
- **Multiple trait types**: four types (LLM, regex, callable, metric) with different execution models

Unlike templates, which operate on parsed structured data, rubrics evaluate the **raw response text** directly.

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

## 3. What Rubrics Do Not Evaluate

Rubrics are not designed for checking factual correctness. If you need to verify that an answer matches a known ground truth, use an [answer template](../../notebooks/core_concepts/answer-templates.ipynb). Rubrics cannot tell you whether an answer is *right*; they tell you whether it has certain observable properties.

For example:

- Given the question "Which is the putative target of venetoclax?", a **template** checks whether the response identifies `BCL2` as the target (ground truth verification).
- For the same question, the user might attach an **[LLM trait](../../notebooks/core_concepts/rubrics/llm-traits.ipynb)** called "Generated answer mentions afety profile of the drug" to assess whether the response addresses safety. The trait is not a question sent to the model; it is an evaluation criterion applied to the response after the fact (subjective judgment on the raw text; no single correct answer).
- A **[regex trait](../../notebooks/core_concepts/rubrics/regex-traits.ipynb)** such as "has bracket citations" checks whether the response contains at least one `[N]`-style citation (deterministic pattern match).
- A **[callable trait](../../notebooks/core_concepts/rubrics/callable-traits.ipynb)** such as "under 150 words" runs custom Python logic against the response text.
- A **[metric trait](../../notebooks/core_concepts/rubrics/metric-traits.ipynb)** such as "expected drug interactions mentioned" measures how many of a predefined list of items appear in the response (partial-coverage scoring).

[MetricRubricTrait](../../notebooks/core_concepts/rubrics/metric-traits.ipynb) occupies a middle ground: it references a list of expected items, but unlike a template it produces a score on a continuum (precision, recall, F1) rather than a binary pass/fail. Mentioning 3 out of 5 expected drug interactions yields an F1 of 0.6, not a failure.

No ground truth does not mean no specification. Rubric traits work better when the description makes your standard explicit. If you care about conciseness, say what that means in context: for example, "answers the question directly, avoids repetition, and stays under 120 words unless the prompt asks for detail." Clear trait descriptions improve the quality and consistency of evaluation even when no single correct answer exists.

See [templates vs rubrics](../template-vs-rubric.md) for a full comparison, and [evaluation modes](../evaluation-modes.md) for how to combine them in a single benchmark.

## 4. Four Rubric Trait Types

Karenina supports four types of evaluation traits, each suited for different evaluation needs:

| Trait Type | Returns | LLM Required | Best For |
|------------|---------|--------------|----------|
| [**LLMRubricTrait**](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) | `bool` or `int` | Yes | Subjective quality assessment (clarity, safety, tone) |
| [**Literal traits**](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) | `int` (class index) | Yes | Ordered categorical classification (quality tiers, tone levels) |
| [**RegexTrait**](../../notebooks/core_concepts/rubrics/regex-traits.ipynb) | `bool` | No | Deterministic pattern matching (keywords, format compliance) |
| [**CallableTrait**](../../notebooks/core_concepts/rubrics/callable-traits.ipynb) | `bool` or `int` | No | Custom Python logic (word counts, structure checks) |
| [**MetricRubricTrait**](../../notebooks/core_concepts/rubrics/metric-traits.ipynb) | metrics dict | Yes | Extraction completeness (precision, recall, F1) |

Each trait type's sub-page includes a pipeline diagram showing how evaluation works during [stage 11 (RubricEvaluation)](../verification-pipeline.md), what data reaches the parsing model (for LLM-dependent traits), and how results flow to `VerificationResult.rubric`.

### 4.1. LLMRubricTrait

LLM-evaluated traits where the parsing model uses its judgment to assess subjective qualities. Supports two kinds:

- **Boolean**: true/false judgments (e.g., *"Is this response safe?"*)
- **Score**: numeric rating within a configurable range (e.g., *"Rate clarity from 1-5"*)

LLM traits also support optional **deep judgment** for evidence-based evaluation with excerpt extraction and verification. See [LLM traits](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) for details.

### 4.2. Literal Traits

A specialized kind of LLM trait for **ordered categorical classification**. The LLM classifies the response into one of several predefined classes (e.g., formal/casual/technical tone, or poor/acceptable/good/excellent quality). Returns an integer index based on class order, with `higher_is_better` controlling interpretation. See the [literal kind section](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) for details.

### 4.3. RegexTrait

Deterministic pattern matching on the raw response text. Provides 100% reproducible evaluation without any LLM variability. Key options:

- `pattern`: Python regex pattern to match
- `case_sensitive`: whether matching is case-sensitive (default: `True`)
- `invert_result`: invert the boolean result for negative matching (default: `False`)

See [regex traits](../../notebooks/core_concepts/rubrics/regex-traits.ipynb) for details and examples.

### 4.4. CallableTrait

Custom Python functions serialized via cloudpickle. Supports both boolean (pass/fail) and score return types. Must be created programmatically using `CallableTrait.from_callable()`; not available via the GUI for security reasons.

**Security note:** Deserializing callable code can execute arbitrary Python. Only load callable traits from trusted sources.

See [callable traits](../../notebooks/core_concepts/rubrics/callable-traits.ipynb) for details and examples.

### 4.5. MetricRubricTrait

Metric traits measure how well a response covers a set of expected items. You provide a list of items that should (and optionally should not) appear in the response, and the parsing model classifies each item as present or absent. Karenina then computes standard metrics (precision, recall, F1) from the resulting confusion matrix.

At first glance this resembles a template, since both check for expected content. The key difference is that a template treats a missing item as a verification failure, while a metric trait produces a score on a continuum: mentioning 3 out of 5 expected drug interactions yields an F1 of 0.6, not a binary fail. This makes metric traits the right choice when partial coverage is meaningful rather than wrong. In `full_matrix` mode they can also verify the absence of items that should not appear (e.g., deprecated terms, contraindicated drugs).

Two evaluation modes:

- **tp_only**: define what should be present; computes precision, recall, F1
- **full_matrix**: define both what should and should not be present; additionally computes specificity and accuracy

See [metric traits](../../notebooks/core_concepts/rubrics/metric-traits.ipynb) for details and examples.

## 5. Choosing the Right Trait Type

| Need | Trait Type |
|------|-----------|
| Subjective quality (clarity, conciseness, tone) | LLMRubricTrait (boolean or score) |
| Categorical classification (quality tiers, tone levels) | LLMRubricTrait (literal) |
| Exact keyword or format validation | RegexTrait |
| Complex validation logic (word counts, structure) | CallableTrait |
| Precision/recall/F1 measurement | MetricRubricTrait |
| Deterministic, reproducible check | RegexTrait or CallableTrait |
| Evidence-based evaluation with excerpts | LLMRubricTrait with deep judgment |

## 6. The `higher_is_better` Field

All trait types (except MetricRubricTrait, where metrics are inherently "higher is better") include a `higher_is_better` field that controls directionality:

- **Boolean traits**: `True` means `True` is a positive outcome
- **Score traits**: `True` means higher scores indicate better performance
- **Literal traits**: `True` means later classes (higher indices) are better
- **Regex traits**: `True` means a match indicates a positive outcome

This field is used by analysis tools and DataFrame builders to correctly interpret and aggregate rubric results. It is also crucial for the GEPA optimization procedure, which relies on `higher_is_better` to determine the direction of improvement when optimizing prompts against rubric scores. GEPA documentation is forthcoming.

## 7. Next Steps

- [LLM traits](../../notebooks/core_concepts/rubrics/llm-traits.ipynb): boolean and score kinds with deep judgment
- [Literal traits](../../notebooks/core_concepts/rubrics/llm-traits.ipynb): ordered categorical classification (part of LLM traits)
- [Regex traits](../../notebooks/core_concepts/rubrics/regex-traits.ipynb): deterministic pattern matching
- [Callable traits](../../notebooks/core_concepts/rubrics/callable-traits.ipynb): custom Python functions
- [Metric traits](../../notebooks/core_concepts/rubrics/metric-traits.ipynb): precision, recall, F1 computation
- [Evaluation modes](../evaluation-modes.md): template_only, template_and_rubric, rubric_only
- [Full Evaluation Benchmark](../../notebooks/creating-benchmarks/full-evaluation-benchmark.ipynb): workflow guide for adding rubrics to benchmarks
