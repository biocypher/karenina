# Deep Judgment for Rubrics

Deep judgment for rubrics applies the same evidence-based verification approach as [template deep judgment](deep-judgment-templates.md), but at the rubric trait level. Instead of verifying that the parsing LLM correctly extracted attribute values, rubric deep judgment verifies that trait scores are grounded in specific passages from the response text.

This is useful when rubric trait evaluations seem unreliable — for example, when the judge LLM assigns high clarity scores to responses that are actually unclear, or claims safety issues that don't exist in the text.

## When to Use Rubric Deep Judgment

| Scenario | Recommendation |
|----------|----------------|
| High-stakes trait evaluations (safety, compliance) | Enable |
| Rubric traits produce inconsistent scores across runs | Enable |
| Complex traits where the judge tends to hallucinate assessments | Enable |
| Simple boolean traits with clear yes/no answers | Usually unnecessary |
| Cost-sensitive bulk evaluations | Disable (adds LLM calls per trait) |
| Debugging unexpected rubric scores | Enable temporarily |

## How It Works

Rubric deep judgment adds a multi-stage evaluation process for each LLM rubric trait:

```
Standard rubric evaluation:
  Response → LLM evaluates trait → Score

Deep judgment rubric evaluation:
  Response → Extract excerpts → [Search validation] → Generate reasoning → Extract score → Auto-fail check
```

The deep judgment stages run during the RubricEvaluation pipeline stage (Stage 11). The auto-fail check runs as a separate pipeline stage (Stage 12: DeepJudgmentRubricAutoFail).

## The Four Modes

Rubric deep judgment is controlled by `deep_judgment_rubric_mode`, which determines how traits are configured:

| Mode | Description | Use Case |
|------|-------------|----------|
| `"disabled"` | Deep judgment OFF for all traits (default) | Standard evaluation |
| `"enable_all"` | Apply deep judgment to ALL LLM traits | Quick enable for entire rubric |
| `"use_checkpoint"` | Read settings from trait objects themselves | Checkpoint-driven workflows |
| `"custom"` | Per-trait and per-question configuration via nested dict | Fine-grained control |

### Mode: `disabled`

The default mode. All LLM rubric traits are evaluated using standard single-pass LLM judgment. No excerpts, reasoning, or search validation.

```python
from karenina.schemas import VerificationConfig

config = VerificationConfig(
    deep_judgment_rubric_mode="disabled",  # This is the default
    answering_models=[...],
    parsing_models=[...],
)
```

### Mode: `enable_all`

Applies deep judgment to every LLM rubric trait in the benchmark. All traits use the same global default settings.

```python
config = VerificationConfig(
    deep_judgment_rubric_mode="enable_all",
    deep_judgment_rubric_global_excerpts=True,  # Extract excerpts (default)
    answering_models=[...],
    parsing_models=[...],
)
```

Set `deep_judgment_rubric_global_excerpts=False` to skip excerpt extraction while still getting multi-stage reasoning:

```python
config = VerificationConfig(
    deep_judgment_rubric_mode="enable_all",
    deep_judgment_rubric_global_excerpts=False,  # Reasoning only, no excerpts
    answering_models=[...],
    parsing_models=[...],
)
```

### Mode: `use_checkpoint`

Reads deep judgment settings from the trait objects loaded from the checkpoint. This is useful when traits have been pre-configured with deep judgment settings and saved to a `.jsonld` file.

Each `LLMRubricTrait` has these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_judgment_enabled` | `bool` | `False` | Master toggle for this trait |
| `deep_judgment_excerpt_enabled` | `bool` | `True` | Extract excerpts as evidence |
| `deep_judgment_max_excerpts` | `int \| None` | `None` | Max excerpts (None = use global) |
| `deep_judgment_fuzzy_match_threshold` | `float \| None` | `None` | Fuzzy match threshold (None = use global) |
| `deep_judgment_excerpt_retry_attempts` | `int \| None` | `None` | Retry count (None = use global) |
| `deep_judgment_search_enabled` | `bool` | `False` | Enable search validation |

When a per-trait field is `None`, the global default from `VerificationConfig` is used.

```python
config = VerificationConfig(
    deep_judgment_rubric_mode="use_checkpoint",
    answering_models=[...],
    parsing_models=[...],
)
```

### Mode: `custom`

Provides per-trait and per-question configuration through a nested dictionary. This is the most flexible mode, allowing different settings for different traits and questions.

The configuration dictionary has two levels:

```python
config = VerificationConfig(
    deep_judgment_rubric_mode="custom",
    deep_judgment_rubric_config={
        "global": {
            "safety_check": {
                "enabled": True,
                "excerpt_enabled": True,
                "max_excerpts": 5,
                "search_enabled": True,
            },
            "clarity_score": {
                "enabled": True,
                "excerpt_enabled": False,  # Reasoning only
            },
        },
        "question_specific": {
            "question-abc-123": {
                "safety_check": {
                    "enabled": True,
                    "excerpt_enabled": True,
                    "fuzzy_match_threshold": 0.90,  # Stricter for this question
                },
            },
        },
    },
    answering_models=[...],
    parsing_models=[...],
)
```

**Resolution order** (first match wins):

1. Question-specific config for this trait → `config["question_specific"][question_id][trait_name]`
2. Global trait config → `config["global"][trait_name]`
3. Not found → trait is disabled (no deep judgment)

Each trait config entry is validated against `DeepJudgmentTraitConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable deep judgment for this trait |
| `excerpt_enabled` | `bool` | `True` | Extract excerpts as evidence |
| `max_excerpts` | `int \| None` | `None` | Max excerpts (None = use global default) |
| `fuzzy_match_threshold` | `float \| None` | `None` | Similarity threshold (None = use global) |
| `excerpt_retry_attempts` | `int \| None` | `None` | Retry count (None = use global) |
| `search_enabled` | `bool` | `False` | Enable search validation |

## Per-Trait Evaluation Process

For each trait with deep judgment enabled, the evaluation follows the same multi-stage process as [template deep judgment](deep-judgment-templates.md):

### With Excerpts

When `excerpt_enabled=True`:

1. **Excerpt extraction**: The LLM extracts verbatim quotes from the response that are relevant to the trait
2. **Fuzzy match validation**: Each excerpt is validated against the response text using `difflib.SequenceMatcher` with the configured threshold
3. **Retry on failure**: If validation fails, the LLM retries with error feedback up to `excerpt_retry_attempts` times
4. **Hallucination assessment** (optional): If `search_enabled=True`, each excerpt is checked against web search results
5. **Reasoning generation**: The LLM generates reasoning explaining its assessment based on the excerpts
6. **Score extraction**: A final score is extracted from the reasoning

### Without Excerpts

When `excerpt_enabled=False`:

1. **Reasoning generation**: The LLM generates reasoning directly from the full response
2. **Score extraction**: A final score is extracted from the reasoning

This is faster and cheaper (2 LLM calls per trait) but provides less verifiable evidence.

## Auto-Fail (Stage 12)

After rubric evaluation completes, the DeepJudgmentRubricAutoFail stage checks the results:

1. If `deep_judgment_rubric_performed` is True and `traits_without_valid_excerpts` is non-empty → **auto-fail**
2. Sets `verify_result = False`
3. Logs a WARNING listing the problematic traits and their retry metadata

The auto-fail is skipped if:

- Deep judgment rubric was not performed
- No traits are missing excerpts
- Abstention was detected (abstention takes priority)

## Configuration

### Global Defaults

All deep judgment rubric settings are on `VerificationConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_judgment_rubric_mode` | `Literal` | `"disabled"` | Mode: `"disabled"`, `"enable_all"`, `"use_checkpoint"`, `"custom"` |
| `deep_judgment_rubric_global_excerpts` | `bool` | `True` | Enable excerpts in `enable_all` mode |
| `deep_judgment_rubric_max_excerpts_default` | `int` | `7` | Default max excerpts per trait |
| `deep_judgment_rubric_fuzzy_match_threshold_default` | `float` | `0.80` | Default fuzzy match threshold (0.0–1.0) |
| `deep_judgment_rubric_excerpt_retry_attempts_default` | `int` | `2` | Default retry attempts for excerpt extraction |
| `deep_judgment_rubric_search_tool` | `str \| Callable` | `"tavily"` | Search tool: `"tavily"` or custom callable |
| `deep_judgment_rubric_config` | `dict \| None` | `None` | Custom mode per-trait config dict |

### Via from_overrides

```python
config = VerificationConfig.from_overrides(
    deep_judgment_rubric_mode="enable_all",
    deep_judgment_rubric_excerpts=True,
    deep_judgment_rubric_max_excerpts=5,
    deep_judgment_rubric_fuzzy_threshold=0.90,
    deep_judgment_rubric_retry_attempts=3,
    deep_judgment_rubric_search=True,
    deep_judgment_rubric_search_tool="tavily",
    answering_model="gpt-4o",
    answering_id="answering",
    parsing_model="gpt-4o",
    parsing_id="parsing",
)
```

Override parameter to config field mapping:

| Override Parameter | Config Field |
|-------------------|--------------|
| `deep_judgment_rubric_mode` | `deep_judgment_rubric_mode` |
| `deep_judgment_rubric_excerpts` | `deep_judgment_rubric_global_excerpts` |
| `deep_judgment_rubric_max_excerpts` | `deep_judgment_rubric_max_excerpts_default` |
| `deep_judgment_rubric_fuzzy_threshold` | `deep_judgment_rubric_fuzzy_match_threshold_default` |
| `deep_judgment_rubric_retry_attempts` | `deep_judgment_rubric_excerpt_retry_attempts_default` |
| `deep_judgment_rubric_search` | `deep_judgment_rubric_search_enabled` |
| `deep_judgment_rubric_search_tool` | `deep_judgment_rubric_search_tool` |
| `deep_judgment_rubric_config` | `deep_judgment_rubric_config` |

### Via CLI

```bash
# Enable for all traits
karenina verify benchmark.jsonld --preset my_preset.json \
    --deep-judgment-rubric-mode enable_all

# Custom mode with config file
karenina verify benchmark.jsonld --preset my_preset.json \
    --deep-judgment-rubric-mode custom \
    --deep-judgment-rubric-config rubric_dj_config.json

# Tune global defaults
karenina verify benchmark.jsonld --preset my_preset.json \
    --deep-judgment-rubric-mode enable_all \
    --deep-judgment-rubric-max-excerpts 5 \
    --deep-judgment-rubric-fuzzy-threshold 0.90 \
    --deep-judgment-rubric-retry-attempts 3
```

## Result Fields

Deep judgment rubric results are stored in `result.deep_judgment_rubric`:

| Field | Type | Description |
|-------|------|-------------|
| `deep_judgment_rubric_performed` | `bool` | Whether deep judgment rubric was executed |
| `extracted_rubric_excerpts` | `dict[str, list[dict]]` | Excerpts per trait with text, confidence, similarity score |
| `rubric_trait_reasoning` | `dict[str, str]` | Reasoning per trait explaining the score determination |
| `deep_judgment_rubric_scores` | `dict[str, int \| bool]` | Scores for traits evaluated with deep judgment |
| `standard_rubric_scores` | `dict[str, int \| bool]` | Scores for traits evaluated without deep judgment |
| `trait_metadata` | `dict[str, dict]` | Per-trait metadata (stages completed, model calls, retry counts) |
| `traits_without_valid_excerpts` | `list[str]` | Traits that failed excerpt extraction (triggers auto-fail) |
| `rubric_hallucination_risk_assessment` | `dict[str, dict]` | Per-trait hallucination risk (if search enabled) |
| `total_deep_judgment_model_calls` | `int` | Total LLM calls across all deep-judgment traits |
| `total_traits_evaluated` | `int` | Number of traits evaluated with deep judgment |
| `total_excerpt_retries` | `int` | Total retry attempts across all traits |

### Inspecting Results

```python
for result in results:
    dj_rubric = result.deep_judgment_rubric
    if dj_rubric and dj_rubric.deep_judgment_rubric_performed:
        # Check for failed excerpt extraction
        if dj_rubric.traits_without_valid_excerpts:
            print(f"Failed excerpts for: {dj_rubric.traits_without_valid_excerpts}")

        # Compare deep judgment vs standard scores
        for trait, score in (dj_rubric.deep_judgment_rubric_scores or {}).items():
            print(f"  {trait} (deep judgment): {score}")
        for trait, score in (dj_rubric.standard_rubric_scores or {}).items():
            print(f"  {trait} (standard): {score}")

        # Inspect reasoning for a specific trait
        reasoning = (dj_rubric.rubric_trait_reasoning or {}).get("safety_check")
        if reasoning:
            print(f"Safety reasoning: {reasoning[:200]}...")

        # Check hallucination risk (if search was enabled)
        for trait, risk in (dj_rubric.rubric_hallucination_risk_assessment or {}).items():
            if risk.get("overall_risk") in ("medium", "high"):
                print(f"  Warning: {trait} has {risk['overall_risk']} hallucination risk")

        # Aggregate statistics
        print(f"Model calls: {dj_rubric.total_deep_judgment_model_calls}")
        print(f"Traits evaluated: {dj_rubric.total_traits_evaluated}")
        print(f"Total retries: {dj_rubric.total_excerpt_retries}")
```

### Per-Trait Metadata

Each trait's metadata in `trait_metadata` contains:

| Field | Type | Description |
|-------|------|-------------|
| `stages_completed` | `list[str]` | Completed stages: `"excerpt_extraction"`, `"hallucination_assessment"`, `"reasoning_generation"`, `"score_extraction"` |
| `model_calls` | `int` | LLM calls for this trait |
| `had_excerpts` | `bool` | Whether excerpts were extracted |
| `excerpt_retry_count` | `int` | Number of retries for this trait |
| `excerpt_validation_failed` | `bool` | Whether excerpt validation ultimately failed |

## Comparison with Template Deep Judgment

| Aspect | Template Deep Judgment | Rubric Deep Judgment |
|--------|----------------------|---------------------|
| Pipeline stage | Stage 7 (ParseTemplate) | Stage 11 (RubricEvaluation) |
| Auto-fail stage | Stage 10 (DeepJudgmentAutoFail) | Stage 12 (DeepJudgmentRubricAutoFail) |
| Scope | Per-attribute (template fields) | Per-trait (rubric LLM traits) |
| Configuration | Single toggle (`deep_judgment_enabled`) | Four modes with per-trait control |
| Default max excerpts | 3 | 7 |
| Result location | `result.deep_judgment` | `result.deep_judgment_rubric` |
| Mixed evaluation | N/A (all-or-nothing for template) | Yes — some traits deep judgment, others standard |

## Cost Considerations

Deep judgment adds LLM calls per trait during rubric evaluation:

| Configuration | Additional LLM Calls Per Trait |
|---------------|-------------------------------|
| With excerpts, no search | 2–3 (excerpts + reasoning + score) |
| With excerpts and search | 3–4 (adds hallucination assessment) |
| Without excerpts | 2 (reasoning + score) |
| Per retry | +1 per failed excerpt attempt |

For a rubric with 5 LLM traits, all with deep judgment and excerpts enabled, the typical cost is 10–15 additional LLM calls per question. Use `enable_all` with `deep_judgment_rubric_global_excerpts=False` for a lower-cost alternative that still provides multi-stage reasoning.

## Error Handling

Rubric deep judgment uses the same graceful degradation strategy as template deep judgment:

- **Search failure**: Returns empty results, continues without hallucination assessment
- **Fuzzy match failure after retries**: Marks trait as missing excerpts, continues with other traits
- **Reasoning generation failure**: Logs warning, continues with empty reasoning
- **Score extraction failure**: Falls back to parsing the response directly

Traits that exhaust all retries without valid excerpts are added to `traits_without_valid_excerpts`, which triggers auto-fail in Stage 12.

## Related

- [Advanced Pipeline Overview](index.md) — Stage ordering and evaluation mode matrix
- [13 Stages in Detail](stages.md) — Stage 11 (RubricEvaluation) and Stage 12 (DeepJudgmentRubricAutoFail)
- [Deep Judgment: Templates](deep-judgment-templates.md) — The parallel system for template attributes
- [VerificationConfig Reference](../reference/configuration/verification-config.md) — All configuration fields including deep judgment rubric settings
- [VerificationResult Structure](../07-analyzing-results/verification-result.md) — Complete result hierarchy including `deep_judgment_rubric`
