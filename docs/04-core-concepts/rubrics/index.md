# Rubrics

Rubrics assess **qualitative traits of the raw response trace** rather than parsed, structured data. While [answer templates](../answer-templates.md) focus on *what* the model said (correctness), rubrics evaluate *how* it said it (quality dimensions like safety, conciseness, or thoroughness).

## What Are Rubrics?

A **rubric** is a collection of evaluation traits that assess qualities of LLM responses beyond factual correctness:

- **Quality assessment** -- evaluate traits like clarity, completeness, and style
- **Complement templates** -- templates check factual correctness; rubrics assess qualities that typically lack a ground truth or characterize the answer style
- **Multiple trait types** -- LLM-based, regex-based, callable-based, and metric-based evaluation
- **Flexible scope** -- apply globally to all questions or to specific questions only

Unlike templates, which operate on parsed structured data via `verify()`, rubrics evaluate broader characteristics of the **raw response text** using specialized evaluators.

## When to Use Rubrics vs Templates

| Use Case | Choose |
|----------|--------|
| Verify a factual answer is correct | [Answer template](../answer-templates.md) |
| Assess subjective quality (clarity, conciseness) | Rubric (LLM trait) |
| Check for required keywords or format patterns | Rubric (regex trait) |
| Apply custom validation logic (word count, structure) | Rubric (callable trait) |
| Measure extraction completeness (precision, recall, F1) | Rubric (metric trait) |
| Verify correctness **and** assess quality | Both together |

Templates and rubrics are complementary. A common pattern is to use a template to verify the model extracted the correct answer, and a rubric to check whether the response was concise, cited sources, or avoided hallucination. See [evaluation modes](../evaluation-modes.md) for how to configure this.

## Global vs Question-Specific Rubrics

Rubrics can be applied at two scopes:

### Global Rubrics

Global rubrics are evaluated for **every question** in a benchmark. Use them for traits that should be assessed universally.

**Best for:**

- General quality traits (clarity, conciseness, completeness)
- Safety requirements that apply to all responses
- Style guidelines that should be consistent throughout

**Example:** Ensure all answers in a genomics benchmark are clear and concise, regardless of the specific question.

### Question-Specific Rubrics

Question-specific rubrics are evaluated for **a single question only**. Use them for domain validation or specialized requirements.

**Best for:**

- Domain-specific terminology checks
- Question-specific validation requirements
- Classification or categorization metrics

**Example:** Check that the answer to "What is the approved drug target of Venetoclax?" mentions BH3 proteins -- a check that only makes sense for that particular question.

### Combining Both

When a question has both a global rubric and a question-specific rubric, karenina **merges** them. The question is evaluated against all traits from both rubrics. Trait names must be unique across global and question-specific rubrics -- duplicates raise a `ValueError`.

**Result:**

- Questions with only a global rubric: evaluated against global traits only
- Questions with both: evaluated against global traits + question-specific traits
- Questions with only a question-specific rubric: evaluated against question-specific traits only

## Four Rubric Trait Types

Karenina supports four types of evaluation traits, each suited for different evaluation needs:

| Trait Type | Returns | LLM Required | Best For |
|------------|---------|--------------|----------|
| [**LLMRubricTrait**](llm-traits.md) | `bool` or `int` | Yes | Subjective quality assessment (clarity, safety, tone) |
| [**Literal traits**](literal-traits.md) | `int` (class index) | Yes | Ordered categorical classification (quality tiers, tone levels) |
| [**RegexTrait**](regex-traits.md) | `bool` | No | Deterministic pattern matching (keywords, format compliance) |
| [**CallableTrait**](callable-traits.md) | `bool` or `int` | No | Custom Python logic (word counts, structure checks) |
| [**MetricRubricTrait**](metric-traits.md) | metrics dict | Yes | Extraction completeness (precision, recall, F1) |

### LLMRubricTrait

LLM-evaluated traits where the parsing model uses its judgment to assess subjective qualities. Supports two kinds:

- **Boolean** -- true/false judgments (e.g., *"Is this response safe?"*)
- **Score** -- numeric rating within a configurable range (e.g., *"Rate clarity from 1-5"*)

LLM traits also support optional **deep judgment** for evidence-based evaluation with excerpt extraction and verification. See [LLM traits](llm-traits.md) for details.

### Literal Traits

A specialized kind of LLM trait for **ordered categorical classification**. The LLM classifies the response into one of several predefined classes (e.g., formal/casual/technical tone, or poor/acceptable/good/excellent quality). Returns an integer index based on class order, with `higher_is_better` controlling interpretation. See [literal traits](literal-traits.md) for details.

### RegexTrait

Deterministic pattern matching on the raw response text. Provides 100% reproducible evaluation without any LLM variability. Key options:

- `pattern` -- Python regex pattern to match
- `case_sensitive` -- whether matching is case-sensitive (default: `True`)
- `invert_result` -- invert the boolean result for negative matching (default: `False`)

See [regex traits](regex-traits.md) for details and examples.

### CallableTrait

Custom Python functions serialized via cloudpickle. Supports both boolean (pass/fail) and score return types. Must be created programmatically using `CallableTrait.from_callable()` -- not available via the GUI for security reasons.

**Security note:** Deserializing callable code can execute arbitrary Python. Only load callable traits from trusted sources.

See [callable traits](callable-traits.md) for details and examples.

### MetricRubricTrait

Confusion matrix-based traits for measuring extraction completeness. Two evaluation modes:

- **tp_only** -- define what should be present; computes precision, recall, F1
- **full_matrix** -- define both what should and should not be present; additionally computes specificity and accuracy

See [metric traits](metric-traits.md) for details and examples.

## Choosing the Right Trait Type

| Need | Trait Type |
|------|-----------|
| Subjective quality (clarity, completeness, tone) | LLMRubricTrait (boolean or score) |
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

This field is used by analysis tools and DataFrame builders to correctly interpret and aggregate rubric results.

## Next Steps

- [LLM traits](llm-traits.md) -- boolean and score kinds with deep judgment
- [Literal traits](literal-traits.md) -- ordered categorical classification
- [Regex traits](regex-traits.md) -- deterministic pattern matching
- [Callable traits](callable-traits.md) -- custom Python functions
- [Metric traits](metric-traits.md) -- precision, recall, F1 computation
- [Evaluation modes](../evaluation-modes.md) -- template_only, template_and_rubric, rubric_only
- [Defining rubrics](../../05-creating-benchmarks/defining-rubrics.md) -- workflow guide for adding rubrics to benchmarks
