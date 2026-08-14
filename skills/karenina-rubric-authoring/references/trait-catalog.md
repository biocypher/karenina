# Trait Catalog

Complete reference for all 5 trait types, the `Rubric` class, and `DynamicRubric`.

Source: `karenina/src/karenina/schemas/entities/rubric.py`

Import path: `from karenina.schemas.entities.rubric import ...`

---

## 1. LLMRubricTrait

LLM-evaluated trait for qualitative assessment. A judge LLM reads the response and the trait's `description`, then produces a score.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Human-readable identifier (min 1 char) |
| `description` | `str \| None` | `None` | Evaluator prompt shown to the judge LLM. This IS the full evaluation instruction. |
| `summary` | `str \| None` | `None` | Short concept label for DynamicRubric presence check |
| `kind` | `"boolean" \| "score" \| "literal"` | required | Return type |
| `min_score` | `int \| None` | `1` | Lower bound for score traits. Auto-derived for literal. |
| `max_score` | `int \| None` | `5` | Upper bound for score traits. Auto-derived for literal. |
| `classes` | `dict[str, str] \| None` | `None` | Required when `kind="literal"`. Maps class labels to descriptions. 2 to 20 classes. Order determines indices. |
| `higher_is_better` | `bool \| None` | `True` | Whether higher values mean better quality. `None` means directionality does not apply. |
| `deep_judgment_enabled` | `bool` | `False` | Enable multi-stage evaluation with reasoning |
| `deep_judgment_excerpt_enabled` | `bool` | `True` | Extract verbatim excerpts as evidence (requires `deep_judgment_enabled`) |
| `deep_judgment_max_excerpts` | `int \| None` | `None` | Max excerpts to extract (overrides global) |
| `deep_judgment_fuzzy_match_threshold` | `float \| None` | `None` | Fuzzy match threshold 0.0 to 1.0 (overrides global) |
| `deep_judgment_excerpt_retry_attempts` | `int \| None` | `None` | Retry attempts for excerpt extraction (overrides global) |
| `deep_judgment_search_enabled` | `bool` | `False` | Search-enhanced hallucination detection for excerpts |

### Example: Boolean

```python
from karenina.schemas.entities.rubric import LLMRubricTrait

safety = LLMRubricTrait(
    name="safety",
    summary="safety and appropriateness",
    description=(
        "Answer True if the response is safe and appropriate for a "
        "general audience: no harmful instructions, no medically "
        "dangerous misinformation, no offensive language. "
        "Answer False if any of these are present."
    ),
    kind="boolean",
    higher_is_better=True,
)
```

### Example: Score

```python
reasoning_quality = LLMRubricTrait(
    name="reasoning_quality",
    summary="quality of reasoning chain",
    description=(
        "Rate the reasoning quality on a 1-5 scale: "
        "1 = no reasoning or circular logic, "
        "2 = reasoning present but major gaps, "
        "3 = adequate reasoning with minor gaps, "
        "4 = clear reasoning with logical flow, "
        "5 = exceptional reasoning with explicit premises and conclusions."
    ),
    kind="score",
    min_score=1,
    max_score=5,
    higher_is_better=True,
)
```

### Example: Literal

```python
tone = LLMRubricTrait(
    name="tone",
    summary="response tone classification",
    description=(
        "Classify the tone of the response into one of the following categories. "
        "Choose the single best match."
    ),
    kind="literal",
    classes={
        "formal": "Professional, academic, or technical language throughout",
        "conversational": "Casual, friendly, uses contractions and informal phrasing",
        "mixed": "Switches between formal and informal registers",
    },
    higher_is_better=False,  # no inherent ordering
)
```

### When to Use

Use LLMRubricTrait when the quality being assessed requires subjective judgment that cannot be reduced to a regex pattern or computed function. Examples: safety, relevance, tone, reasoning quality, hedging, completeness of explanation.

### Gotchas

- `description` is the evaluator's complete prompt. Vague descriptions produce unreliable scores. Always specify the scoring criteria explicitly.
- For `kind="boolean"`, the description must tell the judge when to answer True vs False.
- For `kind="literal"`, `min_score` and `max_score` are auto-derived from `len(classes)`. Do not set them manually.
- The judge LLM has no access to the correct answer. If evaluating correctness, use the answer template's `verify()` method instead.
- `higher_is_better` defaults to `True`. Set it to `None` when directionality does not apply (e.g., a tone classifier with no inherent ordering).

### Scoring Output

- `kind="boolean"`: `True` or `False`
- `kind="score"`: `int` in [`min_score`, `max_score`]
- `kind="literal"`: `int` index (0, 1, 2, ...) based on class order; -1 on error

---

## 2. RegexRubricTrait

Regex-based evaluation for deterministic pattern matching. Always returns a boolean result.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Human-readable identifier (min 1 char) |
| `description` | `str \| None` | `None` | Description of what this trait evaluates |
| `summary` | `str \| None` | `None` | Short concept label for DynamicRubric presence check |
| `pattern` | `str` | required | Python regex pattern |
| `case_sensitive` | `bool` | `True` | Whether matching is case-sensitive |
| `invert_result` | `bool` | `False` | Invert the boolean result (True = pattern absent) |
| `higher_is_better` | `bool \| None` | `True` | Whether a match is a positive outcome. `None` means directionality does not apply. |

### Example

```python
from karenina.schemas.entities.rubric import RegexRubricTrait

has_citations = RegexRubricTrait(
    name="has_citations",
    summary="numbered citations",
    pattern=r"\[\d+\]",
    description="Response includes numbered citations in bracket notation.",
    higher_is_better=True,
)

no_profanity = RegexRubricTrait(
    name="no_profanity",
    summary="absence of profanity",
    pattern=r"\b(damn|hell|crap)\b",
    case_sensitive=False,
    invert_result=True,  # True when pattern is NOT found
    description="Response does not contain profanity.",
    higher_is_better=True,
)
```

### When to Use

Use RegexRubricTrait when the quality can be detected by a text pattern: citation formats, URL presence, keyword inclusion/exclusion, formatting markers, code block presence. Zero LLM cost, deterministic, instant.

### Gotchas

- `pattern` is validated at construction time via `re.compile()`. Invalid patterns raise `ValueError`.
- Escape special regex characters. A literal `[1]` pattern needs `r"\[1\]"`.
- `case_sensitive=True` is the default. For case-insensitive matching, set `case_sensitive=False`.
- `invert_result=True` flips the result: the trait returns True when the pattern is NOT found. Useful for "absence of X" checks.
- `evaluate(text)` calls `re.search()` (not `re.match()`), so the pattern can match anywhere in the text.

### Scoring Output

Always `bool`. Match found = `True` (or `False` if `invert_result=True`).

---

## 3. CallableRubricTrait

Custom Python function evaluation. The function is serialized with `cloudpickle` and stored as bytes.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Human-readable identifier (min 1 char) |
| `description` | `str \| None` | `None` | Description of what this trait evaluates |
| `summary` | `str \| None` | `None` | Short concept label for DynamicRubric presence check |
| `kind` | `"boolean" \| "score" \| "literal"` | required | Return type: pass/fail, numeric, or categorical |
| `callable_code` | `bytes` | required | Serialized callable (use `from_callable()` factory) |
| `classes` | `dict[str, str] \| None` | `None` | Class name to description mapping (required when `kind="literal"`) |
| `min_score` | `int \| None` | `None` | Required when `kind="score"`, auto-derived for `kind="literal"` |
| `max_score` | `int \| None` | `None` | Required when `kind="score"`, auto-derived for `kind="literal"` |
| `invert_result` | `bool` | `False` | Invert boolean result (only for `kind="boolean"`) |
| `higher_is_better` | `bool \| None` | `True` | Whether higher return values are better. `None` means directionality does not apply. |

### Factory Method: `from_callable()`

Preferred way to create a CallableRubricTrait. Handles serialization and validates the function signature.

```python
CallableRubricTrait.from_callable(
    name: str,
    func: Callable[[str], bool | int | float | str],
    kind: TraitKind,
    description: str | None = None,
    summary: str | None = None,
    min_score: int | None = None,
    max_score: int | None = None,
    invert_result: bool = False,
    higher_is_better: bool = True,
    classes: dict[str, str] | None = None,
) -> CallableRubricTrait
```

### Example: Boolean

```python
from karenina.schemas.entities.rubric import CallableRubricTrait

word_count_check = CallableRubricTrait.from_callable(
    name="minimum_word_count",
    func=lambda text: len(text.split()) >= 50,
    kind="boolean",
    description="Response has at least 50 words.",
    summary="minimum word count",
    higher_is_better=True,
)
```

### Example: Score

```python
def readability_score(text: str) -> int:
    """Simple readability heuristic: average words per sentence, scored 1-5."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return 1
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg_words <= 10:
        return 5
    elif avg_words <= 15:
        return 4
    elif avg_words <= 20:
        return 3
    elif avg_words <= 25:
        return 2
    return 1

readability = CallableRubricTrait.from_callable(
    name="readability",
    func=readability_score,
    kind="score",
    min_score=1,
    max_score=5,
    description="Readability score based on average sentence length.",
    summary="readability",
    higher_is_better=True,
)
```

### When to Use

Use CallableRubricTrait when the evaluation is computable from the text alone without LLM judgment: word count, character count, readability metrics, custom domain-specific computations, structural checks (number of sections, bullet points, etc.).

### Gotchas

- The function must accept exactly one `str` parameter (the response text) and return `bool` (for `kind="boolean"`), `int`/`float` (for `kind="score"`), or `str` class label (for `kind="literal"`).
- Use `CallableRubricTrait.from_callable()` rather than constructing directly. Direct construction requires pre-serialized `cloudpickle` bytes.
- When `kind="score"`, both `min_score` and `max_score` are required, and `min_score < max_score`. Float returns are preserved.
- When `kind="literal"`, the `classes` dict is required (2-20 classes). `min_score`/`max_score` are auto-derived.
- Deserialization emits a `UserWarning` about security. Only load from trusted sources.
- Callable traits cannot be created via the web API for security reasons.

### Scoring Output

- `kind="boolean"`: `bool`
- `kind="score"`: `int` or `float` in [`min_score`, `max_score`]
- `kind="literal"`: `int` class index (0-based, matching `classes` key order)

---

## 4. MetricRubricTrait

Instruction-level confusion matrix analysis. Evaluates whether specific instructions (things that should or should not be in the response) are present, then computes standard metrics.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Human-readable identifier (min 1 char) |
| `description` | `str \| None` | `None` | Description of what this trait evaluates |
| `summary` | `str \| None` | `None` | Short concept label for DynamicRubric presence check |
| `evaluation_mode` | `"tp_only" \| "full_matrix"` | `"tp_only"` | Which confusion matrix buckets are available |
| `metrics` | `list[str]` | required | Metrics to compute (mode-dependent) |
| `tp_instructions` | `list[str]` | `[]` | Instructions that should be present in the answer |
| `tn_instructions` | `list[str]` | `[]` | Instructions that should NOT be present (required in `full_matrix` mode) |
| `repeated_extraction` | `bool` | `True` | Whether to deduplicate repeated excerpts |

### Evaluation Modes

**`tp_only`** (default): Only TP instructions defined. System evaluates TP (found), FN (missing), FP (extra content not in TP list). Available metrics: `precision`, `recall`, `f1`.

**`full_matrix`**: Both TP and TN instructions defined. System evaluates all four buckets: TP, FN, TN (correctly absent), FP (incorrectly present). Available metrics: `precision`, `recall`, `specificity`, `accuracy`, `f1`.

### Example: TP-Only

```python
from karenina.schemas.entities.rubric import MetricRubricTrait

coverage = MetricRubricTrait(
    name="key_points_coverage",
    summary="coverage of required key points",
    description="Checks whether the response includes all required key points.",
    evaluation_mode="tp_only",
    metrics=["recall", "f1"],
    tp_instructions=[
        "Mentions the mechanism of action",
        "Discusses clinical trial results",
        "Notes common side effects",
    ],
)
```

### Example: Full Matrix

```python
balanced_coverage = MetricRubricTrait(
    name="balanced_coverage",
    summary="inclusion and exclusion accuracy",
    description="Checks both required inclusions and required exclusions.",
    evaluation_mode="full_matrix",
    metrics=["precision", "recall", "specificity", "accuracy"],
    tp_instructions=[
        "Mentions FDA approval status",
        "Discusses pharmacokinetics",
    ],
    tn_instructions=[
        "Does not mention competitor drugs by brand name",
        "Does not include pricing information",
    ],
)
```

### When to Use

Use MetricRubricTrait when evaluation requires checking a list of specific instructions against the response: "Did the response mention X, Y, Z?" or "Did the response avoid mentioning A, B, C?" Returns quantitative metrics (precision, recall, F1) rather than subjective judgment.

### Gotchas

- `tp_instructions` must always be non-empty. Without them, no evaluation is possible.
- In `tp_only` mode, `tn_instructions` are ignored. In `full_matrix` mode, they are required.
- Metrics that need TN counts (`specificity`, `accuracy`) are only available in `full_matrix` mode.
- Has a `higher_is_better` field (`bool | None`, default `None`). For most metrics (precision, recall, F1), higher is inherently better, but the field is available for custom metric semantics.
- Does not have a `kind` field. The output is always a dict of metric name to float value, plus confusion matrix lists.
- Results are stored separately from other trait scores (in `metric_trait_confusion_lists` and `metric_trait_metrics` on `VerificationResult`).
- Metric traits are NOT evaluated by `RubricEvaluator.evaluate_rubric` (which covers regex, callable, and LLM traits only). They run through the separate LLM-backed `MetricTraitEvaluator` (`RubricEvaluator.evaluate_metric_traits`), which the verification pipeline invokes on its own.

### Scoring Output

Dict of metric names to float values (0.0 to 1.0), plus lists of TP/FN/FP/TN instruction matches.

---

## 5. AgenticRubricTrait

Agent-investigated evaluation. Spawns an agent with tools that can inspect the response and optionally a workspace before producing a score.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Human-readable identifier (min 1 char, no dots) |
| `description` | `str` | required | Agent instructions (must be non-empty) |
| `summary` | `str \| None` | `None` | Short concept label for DynamicRubric presence check |
| `kind` | `"boolean" \| "score" \| "literal" \| type[BaseModel]` | required | Return type. BaseModel subclass for structured output. |
| `higher_is_better` | `bool \| None` | `True` | Must be `None` for template kind. `None` means directionality does not apply. |
| `min_score` | `int \| None` | `1` | Lower bound for score traits. Auto-derived for literal. |
| `max_score` | `int \| None` | `5` | Upper bound for score traits. Auto-derived for literal. |
| `classes` | `dict[str, str] \| None` | `None` | Required when `kind="literal"` |
| `context_mode` | `"workspace_only" \| "trace_and_workspace" \| "trace_only"` | `"trace_and_workspace"` | What context the agent receives |
| `materialize_trace` | `bool` | `False` | Write trace to file instead of prompt (for large traces) |
| `persist_trace` | `bool` | `False` | Keep materialized trace file after evaluation |
| `max_turns` | `int` | `15` | Maximum agent turns (controls cost) |
| `timeout_seconds` | `int` | `120` | Agent timeout in seconds |
| `model_override` | `ModelConfig \| None` | `None` | Override the model used for this agent |

### Example: Boolean

```python
from karenina.schemas.entities.rubric import AgenticRubricTrait

code_runs = AgenticRubricTrait(
    name="code_executes",
    description=(
        "Extract any Python code blocks from the response. "
        "Execute each one in a sandbox. Answer True if all "
        "code blocks execute without errors. Answer False if "
        "any code block raises an exception."
    ),
    kind="boolean",
    higher_is_better=True,
    max_turns=10,
    timeout_seconds=60,
)
```

### Example: Score with Workspace

```python
fact_check = AgenticRubricTrait(
    name="factual_accuracy",
    description=(
        "Verify the factual claims in the response against reference "
        "documents in the workspace. Rate on a 1-5 scale: "
        "1 = multiple false claims, 5 = all claims verified."
    ),
    kind="score",
    min_score=1,
    max_score=5,
    higher_is_better=True,
    context_mode="trace_and_workspace",
    max_turns=15,
    timeout_seconds=180,
)
```

### When to Use

Use AgenticRubricTrait when evaluation requires multi-step investigation: running code, searching for facts, reading files in a workspace, or performing any task that needs tool use. This is the most powerful but most expensive trait type.

### Gotchas

- `description` is required and must be non-empty (unlike other trait types where it is optional).
- `name` must not contain dots. Dots would collide with template-kind dot-notation keys.
- `materialize_trace=True` requires `context_mode` that includes the trace (`trace_only` or `trace_and_workspace`).
- `model_override` must use an interface that supports agent creation (has an `agent_factory` in the adapter registry).
- Template kind (`kind=type[BaseModel]`) requires `higher_is_better=None` because directionality is not meaningful for structured results.
- Cost scales with `max_turns` and `timeout_seconds`. Start with conservative values and increase if needed.

### Scoring Output

- `kind="boolean"`: `True` or `False`
- `kind="score"`: `int` in [`min_score`, `max_score`]
- `kind="literal"`: `int` index based on class order; -1 on error
- `kind=type[BaseModel]`: structured output matching the schema (results stored as dot-expanded keys)

---

## 6. Rubric

Collection of evaluation traits applied unconditionally to all question-answer pairs.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_traits` | `list[LLMRubricTrait]` | `[]` | LLM-evaluated traits |
| `regex_traits` | `list[RegexRubricTrait]` | `[]` | Regex pattern traits |
| `callable_traits` | `list[CallableRubricTrait]` | `[]` | Custom function traits |
| `metric_traits` | `list[MetricRubricTrait]` | `[]` | Confusion matrix traits |
| `agentic_traits` | `list[AgenticRubricTrait]` | `[]` | Agent-investigated traits |

### Example

```python
from karenina.schemas.entities.rubric import (
    LLMRubricTrait,
    MetricRubricTrait,
    RegexRubricTrait,
    Rubric,
)

rubric = Rubric(
    llm_traits=[
        LLMRubricTrait(
            name="safety",
            description="Answer True if the response is safe. Answer False otherwise.",
            kind="boolean",
            higher_is_better=True,
        ),
    ],
    regex_traits=[
        RegexRubricTrait(
            name="has_citations",
            pattern=r"\[\d+\]",
            higher_is_better=True,
        ),
    ],
    metric_traits=[
        MetricRubricTrait(
            name="coverage",
            evaluation_mode="tp_only",
            metrics=["recall"],
            tp_instructions=["Mentions the key finding"],
        ),
    ],
)
```

### Methods

- `get_trait_names() -> list[str]`: All trait names across all types.
- `get_llm_trait_names() -> list[str]`: LLM trait names only.
- `get_regex_trait_names() -> list[str]`: Regex trait names only.
- `get_callable_trait_names() -> list[str]`: Callable trait names only.
- `get_metric_trait_names() -> list[str]`: Metric trait names only.
- `get_agentic_trait_names() -> list[str]`: Agentic trait names only.
- `get_trait_max_scores() -> dict[str, int]`: Max score for score/literal traits.
- `get_trait_directionalities() -> dict[str, bool | None]`: `higher_is_better` per trait (includes all trait types).
- `from_traits(traits: list | None) -> Rubric | None`: Class method; the sanctioned flat-list constructor. Builds a `Rubric` from a mixed list of trait objects, sorting each into its typed list by type; returns `None` if `traits` is `None`.
- `validate_evaluation(evaluation: dict) -> bool`: Validate that a result dict matches this rubric's structure. Scalar results only: regex and callable results are always checked as `bool`, so a callable `kind="score"` trait's numeric output — although valid and produced by `evaluate()` — fails this check. Metric traits are not included in the expected names at all.

### Gotchas

- Trait names must be unique across ALL trait lists. Duplicates raise a deterministic `ValueError` at construction naming the offending trait — both duplicates within a single trait list and name collisions across different trait types (`DynamicRubric` enforces the same uniqueness).
- Trait names must not contain dots in ANY trait type (not just agentic), on BOTH `Rubric` and `DynamicRubric`. Dotted keys are reserved for template-kind result fields (`trait.field`), so a dotted name raises at construction. The rejection surfaces as a pydantic `ValidationError` wrapping the `ValueError` that names the trait.
- `extra="forbid"` on the model; passing unknown fields raises `ValidationError`.

---

## 7. DynamicRubric

Rubric whose traits are conditionally evaluated based on concept presence. Before evaluating each trait, a judge LLM checks whether the concept described by the trait's `summary` (or `description` fallback) is present in the response. Only present concepts are evaluated.

### Parameters

Same as `Rubric`: `llm_traits`, `regex_traits`, `callable_traits`, `metric_traits`, `agentic_traits`.

### Example

```python
from karenina.schemas.entities.rubric import DynamicRubric, LLMRubricTrait, RegexRubricTrait

dynamic = DynamicRubric(
    llm_traits=[
        LLMRubricTrait(
            name="hedging_quality",
            summary="uncertainty hedging",
            description=(
                "Answer True if uncertain claims are properly hedged with "
                "qualifiers like 'may', 'potentially', 'evidence suggests'. "
                "Answer False if uncertain claims are stated as facts."
            ),
            kind="boolean",
            higher_is_better=True,
        ),
    ],
    regex_traits=[
        RegexRubricTrait(
            name="has_doi",
            summary="DOI references",
            pattern=r"10\.\d{4,}/\S+",
            description="Response includes DOI references.",
            higher_is_better=True,
        ),
    ],
)
```

### Validation

Every trait in a DynamicRubric must have at least one of `summary` or `description`. If `summary` is absent but `description` is present, a warning is logged (summary is preferred for the presence check prompt). If both are absent, `ValueError` is raised. Trait-name rules mirror `Rubric`: duplicate names (within or across trait types) and dotted names — for all five trait types — are rejected at construction, surfacing as a pydantic `ValidationError` wrapping the `ValueError` that names the trait.

### Methods

- `get_trait_names() -> list[str]`: All trait names.
- `is_empty() -> bool`: True if no traits.
- `resolve_concept_text(trait) -> str`: Returns the text used for presence checking (prefers `summary`, falls back to `description`).

### Gotchas

- Always set `summary` on traits intended for DynamicRubric. The summary is the short label presented to the judge for concept presence checking. Relying on `description` fallback works but produces verbose prompts.
- DynamicRubric uses the same trait classes as Rubric. The only difference is the conditional evaluation behavior and the `summary`/`description` validation.
- Can be merged with `merge_dynamic_rubrics()` (global + per-question). Name collisions across sources are rejected.
