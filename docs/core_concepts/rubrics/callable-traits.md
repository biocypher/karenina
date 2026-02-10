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

# Callable Traits

Callable traits evaluate LLM responses using **custom Python functions**. They give you full programmatic control over evaluation logic -- from simple word count checks to complex domain-specific scoring -- without requiring an LLM call.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
import warnings
warnings.filterwarnings("ignore", message="Deserializing callable")
```

## Overview

A `CallableTrait` wraps a Python function that takes the response text as input and returns either a boolean (pass/fail) or an integer score. The function is serialized using [cloudpickle](https://github.com/cloudpipe/cloudpickle) so it can be stored in checkpoint files.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Human-readable identifier |
| `description` | `str \| None` | `None` | What this trait evaluates |
| `kind` | `str` | *(required)* | `"boolean"` for pass/fail, `"score"` for numeric |
| `callable_code` | `bytes` | *(required)* | Serialized function (cloudpickle) |
| `min_score` | `int \| None` | `None` | Minimum score (required if `kind="score"`) |
| `max_score` | `int \| None` | `None` | Maximum score (required if `kind="score"`) |
| `invert_result` | `bool` | `False` | Invert the boolean result (only for `kind="boolean"`) |
| `higher_is_better` | `bool` | *(required)* | Whether higher return values indicate better performance |

**Key characteristics:**

- Returns **boolean** or **integer** depending on `kind`
- No LLM call required -- evaluated locally by running your function
- Function must accept exactly **one `str` parameter** (the response text)
- Function is serialized via cloudpickle -- lambda functions and module-level functions work best
- Use the `from_callable()` class method to create traits (handles serialization automatically)

## Creating with `from_callable()`

Always use `CallableTrait.from_callable()` rather than constructing directly -- it validates the function signature and handles serialization:

```python
from karenina.schemas import CallableTrait

# Boolean callable: check minimum word count
word_count_trait = CallableTrait.from_callable(
    name="Minimum Word Count",
    description="Response must contain at least 50 words",
    func=lambda text: len(text.split()) >= 50,
    kind="boolean",
    higher_is_better=True,  # Passing the check is good
)

# Evaluate against sample responses
short_response = "The answer is BCL2."
long_response = "The drug target is BCL2. " * 20

print(word_count_trait.evaluate(short_response))  # False (too short)
print(word_count_trait.evaluate(long_response))    # True (long enough)
```

## Score-Based Callables

Score callables return an integer within a defined range. You must specify `min_score` and `max_score`:

```python
def count_sentences(text: str) -> int:
    """Count the number of sentences in the text."""
    import re
    sentences = re.split(r'[.!?]+', text.strip())
    return len([s for s in sentences if s.strip()])

sentence_count_trait = CallableTrait.from_callable(
    name="Sentence Count",
    description="Count the number of sentences in the response",
    func=count_sentences,
    kind="score",
    min_score=0,
    max_score=100,
    higher_is_better=True,  # More sentences = more detailed
)

sample = "BCL2 is a proto-oncogene. It regulates apoptosis. It is located on chromosome 18."
print(f"Score: {sentence_count_trait.evaluate(sample)}")  # Score: 3
```

## Inverted Boolean Callables

For boolean traits, `invert_result=True` flips the function's return value. This is useful when your function detects something undesirable:

```python
# Detect excessive repetition (bad), so invert the result
repetition_trait = CallableTrait.from_callable(
    name="No Excessive Repetition",
    description="Response should not repeat the same sentence excessively",
    func=lambda text: len(set(text.split(". "))) < len(text.split(". ")) * 0.5,
    kind="boolean",
    invert_result=True,  # True from func means repetition detected -> invert to False
    higher_is_better=True,  # True (no repetition) is good
)

print(repetition_trait.evaluate("Unique sentence one. Unique sentence two. Unique sentence three."))  # True
```

## The `higher_is_better` Field

The `higher_is_better` field controls how results are interpreted in aggregate analysis:

| `higher_is_better` | Meaning | Example |
|---------------------|---------|---------|
| `True` | Higher values = better performance | Word count, detail score |
| `False` | Lower values = better performance | Error count, complexity penalty |

```python
# Error count: lower is better
error_count_trait = CallableTrait.from_callable(
    name="Grammar Error Count",
    description="Count potential grammar errors (lower is better)",
    func=lambda text: text.count("  "),  # Simple: count double spaces as errors
    kind="score",
    min_score=0,
    max_score=50,
    higher_is_better=False,  # Fewer errors is better
)

print(f"Errors: {error_count_trait.evaluate('Clean  text with  two double  spaces.')}")  # Errors: 3
```

## Deserializing and Inspecting

You can retrieve the stored function using `deserialize_callable()`:

```python
# Create a trait
length_trait = CallableTrait.from_callable(
    name="Response Length",
    func=lambda text: len(text),
    kind="score",
    min_score=0,
    max_score=10000,
    higher_is_better=True,
)

# Deserialize and call directly
func = length_trait.deserialize_callable()
print(f"Function result: {func('Hello world')}")  # Function result: 11
print(f"Trait evaluate:  {length_trait.evaluate('Hello world')}")  # Trait evaluate:  11
```

!!! warning "Security Warning"
    Deserializing callable code can execute arbitrary Python code. Only load `CallableTrait` instances from **trusted sources**. `CallableTrait` cannot be created via the web API for security reasons.

## Serialization Best Practices

The function is serialized with cloudpickle when you call `from_callable()`. Follow these guidelines:

- **Use lambda functions** for simple logic -- they serialize cleanly
- **Use module-level named functions** for complex logic -- they are more reliable than nested functions
- **Avoid closures over large objects** -- the entire closure state is serialized
- **Avoid unpicklable state** -- database connections, file handles, and thread locks cannot be serialized
- **Keep dependencies minimal** -- the deserializing environment must have the same packages installed

```python
# Good: lambda
trait_a = CallableTrait.from_callable(
    name="Has Keywords",
    func=lambda text: any(kw in text.lower() for kw in ["target", "mechanism", "pathway"]),
    kind="boolean",
    higher_is_better=True,
)

# Good: module-level function (defined above in this example)
trait_b = CallableTrait.from_callable(
    name="Sentence Count",
    func=count_sentences,
    kind="score",
    min_score=0,
    max_score=100,
    higher_is_better=True,
)

print(trait_a.evaluate("The drug's mechanism of action involves BCL2."))  # True
print(trait_a.evaluate("The answer is 42."))  # False
```

## Using Callable Traits in a Rubric

Callable traits combine with other trait types in a `Rubric`:

```python
from karenina.schemas import CallableTrait, Rubric

rubric = Rubric(callable_traits=[
    CallableTrait.from_callable(
        name="Minimum Length",
        description="Response must be at least 100 characters",
        func=lambda text: len(text) >= 100,
        kind="boolean",
        higher_is_better=True,
    ),
    CallableTrait.from_callable(
        name="Technical Depth",
        description="Count technical terms as a quality proxy",
        func=lambda text: sum(1 for term in ["gene", "protein", "pathway", "mechanism", "target"]
                              if term in text.lower()),
        kind="score",
        min_score=0,
        max_score=20,
        higher_is_better=True,
    ),
])

print(f"Rubric has {len(rubric.callable_traits)} callable traits")
for trait in rubric.callable_traits:
    print(f"  - {trait.name} (kind={trait.kind})")
```

## Next Steps

- [LLM Traits](llm-traits.md) -- Boolean and score traits evaluated by an LLM judge
- [Literal Traits](literal-traits.md) -- Ordered categorical classification via LLM
- [Regex Traits](regex-traits.md) -- Deterministic pattern matching
- [Metric Traits](metric-traits.md) -- Precision, recall, and F1 for extraction tasks
- [Defining Rubrics](../../05-creating-benchmarks/defining-rubrics.md) -- Adding traits to benchmarks
