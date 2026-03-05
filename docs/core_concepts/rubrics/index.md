# Rubrics

Rubrics evaluate **how** a model responded by assessing observable properties of the raw response trace, properties that do not require ground truth. While [answer templates](../../notebooks/core_concepts/answer-templates.ipynb) verify *what* the model said (factual correctness against a known answer), rubrics assess qualities like safety, conciseness, tone, or the presence of specific elements (citations, disclaimers).

Rubrics come in four trait types (LLM, regex, callable, metric) that work differently: some require an LLM call, others run locally with no model involved. They can be applied **globally** across all questions or **per-question** for domain-specific checks.

## What Are Rubrics?

A **rubric** is a collection of evaluation traits that assess observable properties of an LLM response without requiring a ground-truth answer:

- **No ground truth needed**: rubrics evaluate properties you can judge by reading the response alone (conciseness, safety, presence of citations)
- **Complement templates**: templates check factual correctness via `verify()`; rubrics assess qualities that characterize the answer style or structure
- **Multiple trait types**: four types (LLM, regex, callable, metric) with different execution models
- **Two scopes**: global traits apply to every question; question-specific traits apply to one question only

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

Once created, a rubric needs to be attached to an evaluation context. You can attach a full `Rubric` object or add individual traits one at a time:

- **Benchmark (global rubric)**: traits applied to every question. Attach a full rubric or add single traits with `benchmark.add_global_rubric_trait()`. See [Full Evaluation Benchmark](../../notebooks/creating-benchmarks/full-evaluation-benchmark.ipynb).
- **Question (question-specific rubric)**: traits applied to a single question. Attach a full rubric or add single traits with `benchmark.add_question_rubric_trait()`. See [Full Evaluation Benchmark](../../notebooks/creating-benchmarks/full-evaluation-benchmark.ipynb).
- **TaskEval**: rubrics attached to free-text evaluation outside the benchmark loop. See [TaskEval](../task-eval.md) for usage via `task_eval.add_rubric()`.

Each trait type has its own sub-page with full API details. The sections below provide an overview.

## What Rubrics Do Not Evaluate

Rubrics are not designed for checking factual correctness. If you need to verify that an answer matches a known ground truth, use an [answer template](../../notebooks/core_concepts/answer-templates.ipynb). Rubrics cannot tell you whether an answer is *right*; they tell you whether it has certain observable properties.

For example:

- "Did the model return the correct gene symbol?" is a **template** question (requires ground truth).
- "Did the model respond concisely?" is a **rubric** question (observable from the response alone).
- "Did the model include citations?" is a **rubric** question (you can see whether citations are present without knowing the correct answer).

See [templates vs rubrics](../template-vs-rubric.md) for a full comparison, and [evaluation modes](../evaluation-modes.md) for how to combine them in a single benchmark.

## Global vs Question-Specific Rubrics

Rubrics can be attached at two scopes:

```
Benchmark
├── Global Rubric              ← traits applied to every question
└── Questions[]
    └── Question-specific Rubric  ← traits for one question only
```

### Global Rubrics

Global rubrics are evaluated for **every question** in a benchmark. Use them for traits that should be assessed uniformly across all responses.

**Examples:**

- Is the response concise?
- Is the assistant's tone polite and professional?
- Does the response avoid harmful or unsafe content?
- For an agent that uses tools: is the final answer grounded in data retrieved through tool calls, rather than generated from the model's own knowledge?

### Question-Specific Rubrics

Question-specific rubrics are evaluated for **a single question only**. Use them for checks that only make sense in the context of a particular question.

**Examples:**

- For a question about a drug: does the response include safety information (contraindications, side effects)?
- For a document retrieval question: does the response provide citations (not whether they are correct, just whether they are present)?

### Combining Both

When a question has both a global rubric and a question-specific rubric, Karenina **merges** them. The question is evaluated against all traits from both rubrics. Trait names must be unique across global and question-specific rubrics; duplicates raise a `ValueError`.

- Questions with only a global rubric: evaluated against global traits only
- Questions with both: evaluated against global traits + question-specific traits
- Questions with only a question-specific rubric: evaluated against question-specific traits only

## Rubric Structure

A rubric organizes its traits in a tree. Understanding this tree shows how rubric evaluation fits into the broader [verification pipeline](../verification-pipeline.md):

```
Rubric
├── LLM Traits[]              ← assessed by the parsing model
│   ├── boolean traits         ← yes/no judgments (safety, citations)
│   ├── score traits           ← numeric rating on a scale (clarity 1-5)
│   └── literal traits         ← ordered categorical classification (quality tiers)
├── Regex Traits[]             ← evaluated locally, no LLM call
├── Callable Traits[]          ← evaluated locally, no LLM call
└── Metric Traits[]            ← instructions sent to parsing model, metrics computed programmatically
```

## A Trait's Journey Through the Pipeline

Rubric traits are evaluated during [stage 11 (RubricEvaluation)](../verification-pipeline.md) of the verification pipeline, after template parsing and verification are complete. The flow depends on the trait type:

1. **LLM traits and metric traits** send the original question, the model's raw response trace, and the trait description (or instructions) to the parsing model. The parsing model returns a structured judgment.
2. **Regex and callable traits** are evaluated locally with no LLM call. The trait receives the raw response text and applies its logic (pattern matching or function execution) deterministically.
3. If [deep judgment](../../advanced-pipeline/deep-judgment-rubrics.md) is enabled for any LLM trait, **stage 12 (DeepJudgmentRubric)** runs as a post-processing layer, extracting verbatim excerpts and validating them against the response.
4. All trait results are assembled into the `VerificationResult.rubric` sub-object during stage 13 (FinalizeResult).

## What the Parsing Model Sees

Not all trait types involve an LLM call. This distinction affects cost, latency, and reproducibility:

| Trait Type | LLM Call? | What the Parsing Model Receives |
|------------|-----------|--------------------------------|
| **LLM traits** (boolean, score, literal) | Yes | System prompt (evaluator role) + user prompt with question text, response trace, and trait descriptions |
| **Metric traits** | Yes | System prompt + user prompt with question text, response trace, and TP/TN instructions to categorize |
| **Regex traits** | No | Nothing; evaluated locally via `re.search()` on the raw response |
| **Callable traits** | No | Nothing; evaluated locally by calling the Python function with the response text |

For LLM-dependent traits, the [prompt assembly](../../advanced-pipeline/prompt-assembly.md) system builds the messages using the same tri-section pattern (task instructions + adapter instructions + user instructions) used throughout the pipeline.

## When to Use Rubrics vs Templates

| Use Case | Choose |
|----------|--------|
| Verify a factual answer is correct | [Answer template](../../notebooks/core_concepts/answer-templates.ipynb) |
| Assess observable qualities (conciseness, safety, tone) | Rubric |
| Verify correctness **and** assess quality | Both together |

Templates and rubrics are complementary. See [templates vs rubrics](../template-vs-rubric.md) for a detailed comparison of trade-offs, and [evaluation modes](../evaluation-modes.md) for how to configure them together.

## Four Rubric Trait Types

Karenina supports four types of evaluation traits, each suited for different evaluation needs:

| Trait Type | Returns | LLM Required | Best For |
|------------|---------|--------------|----------|
| [**LLMRubricTrait**](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) | `bool` or `int` | Yes | Subjective quality assessment (clarity, safety, tone) |
| [**Literal traits**](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) | `int` (class index) | Yes | Ordered categorical classification (quality tiers, tone levels) |
| [**RegexTrait**](../../notebooks/core_concepts/rubrics/regex-traits.ipynb) | `bool` | No | Deterministic pattern matching (keywords, format compliance) |
| [**CallableTrait**](../../notebooks/core_concepts/rubrics/callable-traits.ipynb) | `bool` or `int` | No | Custom Python logic (word counts, structure checks) |
| [**MetricRubricTrait**](../../notebooks/core_concepts/rubrics/metric-traits.ipynb) | metrics dict | Yes | Extraction completeness (precision, recall, F1) |

### LLMRubricTrait

LLM-evaluated traits where the parsing model uses its judgment to assess subjective qualities. Supports two kinds:

- **Boolean**: true/false judgments (e.g., *"Is this response safe?"*)
- **Score**: numeric rating within a configurable range (e.g., *"Rate clarity from 1-5"*)

LLM traits also support optional **deep judgment** for evidence-based evaluation with excerpt extraction and verification. See [LLM traits](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) for details.

### Literal Traits

A specialized kind of LLM trait for **ordered categorical classification**. The LLM classifies the response into one of several predefined classes (e.g., formal/casual/technical tone, or poor/acceptable/good/excellent quality). Returns an integer index based on class order, with `higher_is_better` controlling interpretation. See the [literal kind section](../../notebooks/core_concepts/rubrics/llm-traits.ipynb) for details.

### RegexTrait

Deterministic pattern matching on the raw response text. Provides 100% reproducible evaluation without any LLM variability. Key options:

- `pattern`: Python regex pattern to match
- `case_sensitive`: whether matching is case-sensitive (default: `True`)
- `invert_result`: invert the boolean result for negative matching (default: `False`)

See [regex traits](../../notebooks/core_concepts/rubrics/regex-traits.ipynb) for details and examples.

### CallableTrait

Custom Python functions serialized via cloudpickle. Supports both boolean (pass/fail) and score return types. Must be created programmatically using `CallableTrait.from_callable()`; not available via the GUI for security reasons.

**Security note:** Deserializing callable code can execute arbitrary Python. Only load callable traits from trusted sources.

See [callable traits](../../notebooks/core_concepts/rubrics/callable-traits.ipynb) for details and examples.

### MetricRubricTrait

Metric traits measure how well a response covers a set of expected items. You provide a list of items that should (and optionally should not) appear in the response, and the parsing model classifies each item as present or absent. Karenina then computes standard metrics (precision, recall, F1) from the resulting confusion matrix.

At first glance this resembles a template, since both check for expected content. The key difference is that a template treats a missing item as a verification failure, while a metric trait produces a score on a continuum: mentioning 3 out of 5 expected drug interactions yields an F1 of 0.6, not a binary fail. This makes metric traits the right choice when partial coverage is meaningful rather than wrong. In `full_matrix` mode they can also verify the absence of items that should not appear (e.g., deprecated terms, contraindicated drugs).

Two evaluation modes:

- **tp_only**: define what should be present; computes precision, recall, F1
- **full_matrix**: define both what should and should not be present; additionally computes specificity and accuracy

See [metric traits](../../notebooks/core_concepts/rubrics/metric-traits.ipynb) for details and examples.

## Choosing the Right Trait Type

| Need | Trait Type |
|------|-----------|
| Subjective quality (clarity, conciseness, tone) | LLMRubricTrait (boolean or score) |
| Categorical classification (quality tiers, tone levels) | LLMRubricTrait (literal) |
| Exact keyword or format validation | RegexTrait |
| Complex validation logic (word counts, structure) | CallableTrait |
| Precision/recall/F1 measurement | MetricRubricTrait |
| Deterministic, reproducible check | RegexTrait or CallableTrait |
| Evidence-based evaluation with excerpts | LLMRubricTrait with deep judgment |

## The `higher_is_better` Field

All trait types (except MetricRubricTrait, where metrics are inherently "higher is better") include a `higher_is_better` field that controls directionality:

- **Boolean traits**: `True` means `True` is a positive outcome
- **Score traits**: `True` means higher scores indicate better performance
- **Literal traits**: `True` means later classes (higher indices) are better
- **Regex traits**: `True` means a match indicates a positive outcome

This field is used by analysis tools and DataFrame builders to correctly interpret and aggregate rubric results. It is also crucial for the GEPA optimization procedure, which relies on `higher_is_better` to determine the direction of improvement when optimizing prompts against rubric scores. GEPA documentation is forthcoming.

## Next Steps

- [LLM traits](../../notebooks/core_concepts/rubrics/llm-traits.ipynb): boolean and score kinds with deep judgment
- [Literal traits](../../notebooks/core_concepts/rubrics/llm-traits.ipynb): ordered categorical classification (part of LLM traits)
- [Regex traits](../../notebooks/core_concepts/rubrics/regex-traits.ipynb): deterministic pattern matching
- [Callable traits](../../notebooks/core_concepts/rubrics/callable-traits.ipynb): custom Python functions
- [Metric traits](../../notebooks/core_concepts/rubrics/metric-traits.ipynb): precision, recall, F1 computation
- [Evaluation modes](../evaluation-modes.md): template_only, template_and_rubric, rubric_only
- [Full Evaluation Benchmark](../../notebooks/creating-benchmarks/full-evaluation-benchmark.ipynb): workflow guide for adding rubrics to benchmarks
