# Deep Judgment for Templates

Deep judgment adds a verification layer to template parsing by requiring the parsing LLM to extract verbatim excerpts from the response text. If the LLM claims an attribute value but cannot locate supporting text in the response, the result is auto-failed. This catches hallucinated parsing, where the judge LLM invents attribute values not present in the original response.

Deep judgment supports two modes: **full mode** (excerpt extraction + reasoning + parsing) and **reasoning-only mode** (reasoning + parsing, no excerpts). Use reasoning-only mode when you want per-attribute reasoning traces for transparency but do not need excerpt extraction or its associated auto-fail checks.

## When to Use Deep Judgment

| Scenario | Recommendation |
|----------|----------------|
| High-stakes evaluations where parsing accuracy is critical | Enable |
| Complex templates with many attributes | Enable |
| Parsing model sometimes hallucinates values | Enable |
| Simple yes/no or single-value templates | Usually unnecessary |
| Cost-sensitive bulk evaluations | Disable (adds LLM calls) |
| Debugging unexpected verification failures | Enable temporarily |
| Want reasoning traces without excerpt overhead | Enable reasoning-only |

### Mode Comparison

| Mode | Excerpts | Reasoning | Search | LLM Calls |
|------|----------|-----------|--------|-----------|
| Disabled | No | No | No | 0 |
| Reasoning-only | No | Yes | No | 2 |
| Full | Yes | Yes | No | 3 |
| Full + Search | Yes | Yes | Yes | 4+ |

## How It Works

Deep judgment adds a multi-stage process between answer generation and parameter extraction:

```
Standard parsing:
  Response → Parse to schema → Verify

Reasoning-only deep judgment:
  Response → Generate reasoning → Parse to schema → Verify

Full deep judgment:
  Response → Extract excerpts → [Search validation] → Generate reasoning → Parse to schema → Verify → Auto-fail check
```

The stages run during the ParseTemplate pipeline stage (Stage 7). In full mode, the auto-fail check runs as a separate pipeline stage (Stage 10: DeepJudgmentAutoFail). In reasoning-only mode, Stage 10 is skipped because there are no excerpts to validate.

## Reasoning-Only Mode

Reasoning-only mode generates per-attribute reasoning traces without excerpt extraction. The LLM reads the full response and produces a reasoning explanation for each template attribute, then the reasoning is fed to `ParserPort` for structured parameter extraction. This yields the same `BaseAnswer` output as standard parsing, with `attribute_reasoning` populated for transparency.

### When to Use Reasoning-Only

- You want interpretability (why the parser chose each value) without the cost of excerpt extraction and fuzzy match validation
- Your templates are moderately complex and you want a reasoning audit trail
- Cost is a concern: reasoning-only uses 2 LLM calls per question, compared to 3+ for full deep judgment
- You do not need the auto-fail safety net (since there are no excerpts, the DeepJudgmentAutoFail stage is skipped)

### Configuration

```python
config = VerificationConfig(
    deep_judgment_mode="reasoning_only",
    answering_models=[...],
    parsing_models=[...],
)
```

Or via `from_overrides`:

```python
config = VerificationConfig.from_overrides(
    deep_judgment_mode="reasoning_only",
    answering_model="claude-haiku-4-5",
    answering_id="answering",
    parsing_model="claude-haiku-4-5",
    parsing_id="parsing",
)
```

### Two-Stage Process

1. **Reasoning generation**: The parsing LLM receives the response and template schema, then produces `{"attribute_name": "reasoning text"}` for each attribute (1 LLM call)
2. **Parameter extraction**: The reasoning text is passed to `ParserPort.parse_to_pydantic()` for structured parsing (1 LLM call)

### Result Structure

In reasoning-only mode:

- `deep_judgment_performed` = `True`
- `attribute_reasoning` = populated with per-attribute reasoning
- `extracted_excerpts` = empty dict (`{}`)
- `attributes_without_excerpts` = empty list (`[]`)
- `deep_judgment_stages_completed` = `["reasoning", "parameters"]`
- `deep_judgment_model_calls` = `2`
- `hallucination_risk_assessment` = `None` (search is not applicable)

### Auto-Fail Behavior

The DeepJudgmentAutoFail stage (Stage 10) is skipped entirely when reasoning-only mode is active. Since no excerpts are extracted, there are no missing excerpts to trigger a failure. The reasoning-only flag is stored as a pipeline artifact, and Stage 10 checks for it before running.

## The Three-Stage Process (Full Mode)

### Stage 1: Excerpt Extraction

The parsing LLM receives the raw response and the template schema, then extracts verbatim text excerpts for each attribute.

For each attribute, the LLM produces up to `max_excerpts` excerpts, each with:

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Verbatim quote from the response |
| `confidence` | `str` | `"high"` (direct statement), `"medium"` (implied), `"low"` (weak signal), or `"none"` (no excerpt found) |
| `similarity_score` | `float` | Fuzzy match score against the original response (see below) |

#### Fuzzy Match Validation

Each extracted excerpt is validated against the original response using fuzzy matching:

1. Whitespace is normalized in both the excerpt and the response (multiple spaces collapsed)
2. `difflib.SequenceMatcher.find_longest_match()` finds the best substring match
3. **Similarity score** = longest match length / normalized excerpt length
4. The excerpt passes if the score meets the configured threshold

This catches cases where the LLM slightly paraphrases rather than quoting verbatim.

#### Retry Logic

If fuzzy matching fails for an excerpt:

1. The excerpt is retried up to `excerpt_retry_attempts` times
2. Each retry includes error feedback telling the LLM why the previous excerpt failed
3. After max retries, that excerpt is skipped (the attribute is marked as missing excerpts)
4. A single failed excerpt does not halt the entire pipeline; other attributes continue

Attributes with no valid excerpts after all retries are added to `attributes_without_excerpts`, which triggers the auto-fail in Stage 10.

### Stage 1.5: Hallucination Assessment (Optional)

When `deep_judgment_search_enabled=True`, each excerpt is checked against web search results:

1. A search query is generated from each excerpt
2. The configured search tool (default: Tavily) retrieves relevant results
3. The LLM reviews each excerpt against the search results
4. A per-excerpt hallucination risk is assigned: `"none"`, `"low"`, `"medium"`, or `"high"`
5. The attribute-level risk is the **maximum** risk across all its excerpts

Search results and risk assessments are stored in the result for inspection but do not directly cause auto-fail. The auto-fail is based only on missing excerpts (Stage 10).

### Stage 2: Reasoning Generation

The LLM generates reasoning explaining how the extracted excerpts support or refute each attribute value:

- Without search: simple `{"attribute": "reasoning text"}` format
- With search: nested format including hallucination risk per attribute

Reasoning is stored in the result for transparency and debugging.

### Stage 3: Parameter Extraction

The reasoning text and excerpts are passed to `ParserPort.parse_to_pydantic()` for standard structured parsing. This produces the final `BaseAnswer` instance with all attributes populated, the same output as standard (non-deep-judgment) parsing.

## Auto-Fail (Stage 10)

After parsing completes, the DeepJudgmentAutoFail stage checks the results:

1. If `deep_judgment_performed` is True and `attributes_without_excerpts` is non-empty → **auto-fail**
2. Sets `verify_result = False` and `field_verification_result = False`
3. Logs a WARNING listing the problematic attributes

The auto-fail is skipped if:

- Deep judgment was not performed
- Reasoning-only mode was used (no excerpts to validate)
- No attributes are missing excerpts
- Abstention was detected (abstention takes priority)

## Configuration

All deep judgment template settings are on `VerificationConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_judgment_mode` | `Literal["disabled", "reasoning_only", "full"]` | `"disabled"` | Template deep-judgment mode. `"disabled"`: off. `"reasoning_only"`: reasoning only (2 LLM calls). `"full"`: excerpts + reasoning (3+ LLM calls). |
| `deep_judgment_max_excerpts_per_attribute` | `int` | `3` | Maximum excerpts per attribute (ignored in reasoning-only mode) |
| `deep_judgment_fuzzy_match_threshold` | `float` | `0.80` | Fuzzy match similarity threshold (0.0–1.0) |
| `deep_judgment_excerpt_retry_attempts` | `int` | `2` | Retries on fuzzy match failure |
| `deep_judgment_search_enabled` | `bool` | `False` | Enable web search validation |
| `deep_judgment_search_tool` | `str \| Callable` | `"tavily"` | Search tool: `"tavily"` or custom callable |

### Enabling Deep Judgment

```python
from karenina.schemas import VerificationConfig

config = VerificationConfig(
    deep_judgment_mode="full",
    answering_models=[...],
    parsing_models=[...],
)
```

### With Search Validation

```python
config = VerificationConfig(
    deep_judgment_enabled=True,
    deep_judgment_search_enabled=True,
    deep_judgment_search_tool="tavily",  # Requires TAVILY_API_KEY env var
    answering_models=[...],
    parsing_models=[...],
)
```

### Tuning Excerpt Quality

```python
config = VerificationConfig(
    deep_judgment_enabled=True,
    deep_judgment_max_excerpts_per_attribute=5,    # More evidence per attribute
    deep_judgment_fuzzy_match_threshold=0.90,       # Stricter matching
    deep_judgment_excerpt_retry_attempts=3,         # More retries
    answering_models=[...],
    parsing_models=[...],
)
```

### Via CLI

```bash
karenina verify benchmark.jsonld --preset my_preset.json --deep-judgment
```

### Via from_overrides

```python
config = VerificationConfig.from_overrides(
    deep_judgment_mode="full",
    answering_model="claude-haiku-4-5",
    answering_id="answering",
    parsing_model="claude-haiku-4-5",
    parsing_id="parsing",
)
```

## Search Tool Configuration

### Built-in Tavily

The default search tool uses [Tavily](https://tavily.com/) for web search:

- Requires `TAVILY_API_KEY` environment variable
- Default: 3 results per query, basic search depth
- Graceful degradation: returns empty results on search failure

### Custom Search Tool

You can provide a custom callable matching the search tool signature:

```python
def my_search(query: str | list[str]) -> str | list[str]:
    # Single query returns single result string
    # List of queries returns list of result strings
    ...

config = VerificationConfig(
    deep_judgment_enabled=True,
    deep_judgment_search_enabled=True,
    deep_judgment_search_tool=my_search,
    answering_models=[...],
    parsing_models=[...],
)
```

## Result Fields

Deep judgment results are stored in `result.deep_judgment`:

| Field | Type | Description |
|-------|------|-------------|
| `deep_judgment_performed` | `bool` | Whether deep judgment was executed |
| `extracted_excerpts` | `dict[str, list[dict]]` | Excerpts per attribute with text, confidence, and similarity score |
| `attribute_reasoning` | `dict[str, str]` | Reasoning per attribute |
| `deep_judgment_stages_completed` | `list[str]` | Which stages completed (e.g., `["excerpts", "reasoning", "parameters"]`) |
| `deep_judgment_model_calls` | `int` | Number of LLM calls made during deep judgment |
| `deep_judgment_excerpt_retry_count` | `int` | Total retries across all attributes |
| `attributes_without_excerpts` | `list[str]` | Attributes that failed excerpt extraction |
| `hallucination_risk_assessment` | `dict[str, str]` | Per-attribute hallucination risk (only if search enabled) |

### Inspecting Results

```python
for result in results:
    dj = result.deep_judgment
    if dj and dj.deep_judgment_performed:
        # Check for missing excerpts
        if dj.attributes_without_excerpts:
            print(f"Missing excerpts for: {dj.attributes_without_excerpts}")

        # Inspect excerpts per attribute
        for attr, excerpts in (dj.extracted_excerpts or {}).items():
            for excerpt in excerpts:
                print(f"  {attr}: [{excerpt['confidence']}] {excerpt['text'][:50]}...")

        # Check hallucination risk (if search was enabled)
        if dj.hallucination_risk_assessment:
            for attr, risk in dj.hallucination_risk_assessment.items():
                if risk in ("medium", "high"):
                    print(f"  Warning: {attr} has {risk} hallucination risk")
```

## Cost Considerations

Deep judgment adds LLM calls to the parsing phase:

| Configuration | Additional Parsing LLM Calls |
|---------------|------------------------------|
| Reasoning-only | 2 per question (reasoning + parse) |
| Full deep judgment | 3 per question (excerpts + reasoning + parse) |
| Full + search | 4 per question (adds hallucination assessment) |
| With retries (full only) | +1 per failed excerpt per retry |

The total cost depends on the number of attributes and the retry rate. For a template with 5 attributes and 2 retry attempts, the worst case is 2 base calls + 10 retries = 12 calls per question. In practice, retries are rare with well-behaved parsing models.

## Error Handling

Deep judgment is designed for graceful degradation:

- **Search failure**: Returns empty results, continues without hallucination assessment
- **Fuzzy match failure after retries**: Marks attribute as missing, continues with others
- **Reasoning generation failure**: Logs warning, continues with empty reasoning
- **JSON parse failure**: Logs warning, continues with partial results

The only hard failure is in Stage 3 (parameter extraction via ParserPort), which uses the standard parsing retry mechanisms.

## Related

- [Advanced Pipeline Overview](index.md): Stage ordering and evaluation mode matrix
- [13 Stages in Detail](stages.md): Stage 10 (DeepJudgmentAutoFail) specifics
- [Deep Judgment: Rubrics](deep-judgment-rubrics.md): Per-trait deep judgment for rubric evaluation
- [VerificationConfig Reference](../reference/configuration/verification-config.md): All configuration fields
- [VerificationResult Structure](../workflows/analyzing-results/verification-result.md): Complete result hierarchy
