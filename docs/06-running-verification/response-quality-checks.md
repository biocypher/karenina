# Response Quality Checks

Karenina provides two optional **pre-parsing checks** that evaluate the raw LLM response
before attempting to parse it into a template schema. Both checks run early in the
pipeline (stages 5-6), allowing the system to detect problematic responses and skip
expensive downstream processing.

| Check | What It Detects | Default |
|-------|----------------|---------|
| **Abstention detection** | Model refusals, evasions, disclaimers | Disabled |
| **Sufficiency detection** | Incomplete responses missing required information | Disabled |

Both are disabled by default and must be explicitly enabled.

## Why Pre-Parsing Checks?

Template parsing (stage 7) requires an LLM call to extract structured data from the
response. If the model refused to answer or provided insufficient information, that
parsing call will fail or produce meaningless results — wasting time and cost.

Pre-parsing checks solve this by detecting these cases early:

    Response generated
          │
          ▼
    ┌─ Abstention check ──┐     ┌─ Sufficiency check ──┐
    │  Did model refuse?   │ ──► │  Enough info to parse?│
    └──────────────────────┘     └───────────────────────┘
          │                              │
          ▼                              ▼
    If detected: auto-fail         If insufficient: auto-fail
    Skip parsing & verify          Skip parsing & verify
          │                              │
          └──────────┬───────────────────┘
                     ▼
            Continue to parsing
            (only if both pass)

When a check triggers an auto-fail, the result records **why** it failed (abstention
or insufficiency) rather than recording a parsing failure — making results easier to
interpret.

## Abstention Detection

### What It Detects

Abstention detection identifies responses where the model **refused to answer** or
**evaded the question**. The check uses an LLM-as-judge approach — the parsing model
analyzes the response and determines whether it constitutes a substantive answer.

**Detected as abstention:**

- Explicit refusals: "I cannot answer", "I'm unable to provide"
- Lack of information claims: "I don't have information about"
- Capability disclaimers: "I'm not equipped to", "I'm not authorized to"
- Safety/policy refusals: "This violates my guidelines"
- Evasive responses: Providing only general information without addressing the question
- Uncertainty without substance: Expressing uncertainty without attempting an answer
- Deflection: Redirecting to other resources without answering

**Not detected as abstention:**

- Qualified answers (answers with caveats or uncertainty that still attempt a response)
- Partial answers (answering part of a multi-part question)
- Requests for clarification followed by an attempted answer
- Answers with disclaimers that still provide substantive information

### Configuration

Enable abstention detection on `VerificationConfig`:

```python
from karenina.schemas import VerificationConfig

config = VerificationConfig(
    abstention_enabled=True,
    # ... other settings
)
```

Or via `from_overrides()`:

```python
config = VerificationConfig.from_overrides(
    abstention=True,
    # ... other overrides
)
```

### Auto-Fail Behavior

When abstention is detected:

1. `verify_result` is set to `False` (auto-fail)
2. `abstention_override_applied` is set to `True`
3. Parsing and verification stages are **skipped** (no further LLM calls)
4. The result records the abstention metadata for inspection

### Result Fields

After verification, abstention results are available on `result.template`:

| Field | Type | Description |
|-------|------|-------------|
| `abstention_check_performed` | `bool` | Whether the check was executed |
| `abstention_detected` | `bool \| None` | Whether abstention was found (`None` if check failed) |
| `abstention_reasoning` | `str \| None` | LLM's explanation for the determination |
| `abstention_override_applied` | `bool` | Whether `verify_result` was overridden to `False` |

### When to Use

Abstention detection is most useful for:

- **Safety and compliance testing** — Verify models refuse harmful requests appropriately
- **Capability assessment** — Identify where models hit knowledge boundaries
- **Quality benchmarking** — Distinguish genuine refusals from incorrect answers
- **Domain-specific evaluation** — Medical, legal, or financial domains where models
  should defer to professionals

Consider skipping it for:

- Standard factual benchmarks where abstention is rare
- High-volume testing where the added cost matters (adds one parsing LLM call per question)

## Sufficiency Detection

### What It Detects

Sufficiency detection checks whether the response **contains enough information** to
populate all fields in the answer template schema. It uses an LLM-as-judge approach —
the parsing model compares the response content against the template's JSON schema.

**Detected as sufficient:**

- Every field in the schema has corresponding information in the response
- Information may be implicit but clearly derivable from the response
- Approximate or qualified values are acceptable (e.g., "around 50" for a numeric field)
- Hedged information still counts (e.g., "likely BCL2" for a gene name field)

**Detected as insufficient:**

- A required field has no corresponding information in the response
- Response discusses the topic but omits specific data needed for a field
- Response is vague where the schema requires specific values
- Key information stated as unknown, uncertain, or unavailable
- Response explicitly states it cannot provide certain information

### Configuration

Enable sufficiency detection on `VerificationConfig`:

```python
from karenina.schemas import VerificationConfig

config = VerificationConfig(
    sufficiency_enabled=True,
    # ... other settings
)
```

Or via `from_overrides()`:

```python
config = VerificationConfig.from_overrides(
    sufficiency=True,
    # ... other overrides
)
```

### Auto-Fail Behavior

When the response is insufficient:

1. `verify_result` is set to `False` (auto-fail)
2. `sufficiency_override_applied` is set to `True`
3. Parsing and verification stages are **skipped** (no further LLM calls)
4. The result records sufficiency metadata for inspection

### Result Fields

After verification, sufficiency results are available on `result.template`:

| Field | Type | Description |
|-------|------|-------------|
| `sufficiency_check_performed` | `bool` | Whether the check was executed |
| `sufficiency_detected` | `bool \| None` | Whether the response is sufficient (`None` if check failed) |
| `sufficiency_reasoning` | `str \| None` | LLM's per-field explanation and overall determination |
| `sufficiency_override_applied` | `bool` | Whether `verify_result` was overridden to `False` |

### When to Use

Sufficiency detection is most useful for:

- **Complex templates** — Templates with many fields where responses may be incomplete
- **Multi-attribute extraction** — Questions requiring multiple pieces of information
- **Cost-sensitive pipelines** — Avoid wasting parsing calls on clearly incomplete responses

Consider skipping it for:

- Simple single-field templates where parsing failures are rare
- Templates where partial information is still valuable
- Rubric-only evaluation (no template schema to check against)

!!! note "Requires a template"
    Sufficiency detection only runs when a template (Answer class) exists for the
    question. In `rubric_only` evaluation mode, the sufficiency check is skipped since
    there is no schema to validate against.

## Using Both Together

Both checks can be enabled simultaneously. When combined, they run in sequence:

1. **Abstention check** runs first
2. If abstention is detected, the sufficiency check is **skipped** (no point checking
   sufficiency of a refused response)
3. If no abstention, the **sufficiency check** runs
4. If either triggers, parsing is skipped

```python
config = VerificationConfig(
    abstention_enabled=True,
    sufficiency_enabled=True,
    # ... other settings
)
```

Or via `from_overrides()`:

```python
config = VerificationConfig.from_overrides(
    abstention=True,
    sufficiency=True,
)
```

## Customizing with PromptConfig

Both checks accept custom instructions via
[PromptConfig](prompt-config.md) to tune
the LLM's detection behavior:

```python
from karenina.schemas import VerificationConfig, PromptConfig

config = VerificationConfig(
    abstention_enabled=True,
    sufficiency_enabled=True,
    prompt_config=PromptConfig(
        abstention_detection="In this biomedical benchmark, a response that says "
            "'consult a physician' without providing the requested gene name "
            "should be considered an abstention.",
        sufficiency_detection="Focus on whether the drug target field can be "
            "populated. Other fields are optional.",
    ),
)
```

These custom instructions are appended to the built-in prompts as user instructions
in the [tri-section prompt assembly](../11-advanced-pipeline/prompt-assembly.md)
pattern.

## Pipeline Position

Both checks occupy stages 5-6 in the 13-stage pipeline, after answer generation
and auto-fail checks but before template parsing:

| Stage | Name | Notes |
|-------|------|-------|
| 1 | Validate template | Always runs first |
| 2 | Generate answer | LLM generates response |
| 3 | Recursion limit auto-fail | Check for agent truncation |
| 4 | Trace validation auto-fail | Check trace ends with AI message |
| **5** | **Abstention check** | **If `abstention_enabled=True`** |
| **6** | **Sufficiency check** | **If `sufficiency_enabled=True`** |
| 7 | Parse template | Skipped if stage 5 or 6 triggered |
| 8 | Verify template | Skipped if stage 5 or 6 triggered |
| 9-13 | Remaining stages | Embedding, deep judgment, rubric, finalize |

## Inspecting Results

After running verification with quality checks enabled, inspect results to
understand detection outcomes:

```python
for result in results:
    template = result.template
    if template is None:
        continue

    # Check abstention
    if template.abstention_check_performed:
        if template.abstention_detected:
            print(f"  Abstained: {template.abstention_reasoning}")

    # Check sufficiency
    if template.sufficiency_check_performed:
        if not template.sufficiency_detected:
            print(f"  Insufficient: {template.sufficiency_reasoning}")
```

### Analysis Patterns

Calculate abstention and insufficiency rates across a benchmark:

```python
# Count abstentions
abstained = sum(
    1 for r in results
    if r.template and r.template.abstention_detected
)
total_checked = sum(
    1 for r in results
    if r.template and r.template.abstention_check_performed
)
if total_checked > 0:
    print(f"Abstention rate: {abstained / total_checked:.1%}")
```

## Error Handling

Both checks are designed to **fail safely**:

- **Abstention check failure**: Defaults to "not abstained" — the pipeline continues
  to parsing. This avoids incorrectly blocking answerable responses.
- **Sufficiency check failure**: Defaults to "sufficient" — the pipeline continues
  to parsing. This avoids incorrectly blocking parseable responses.
- Both use 3-attempt retry with exponential backoff for transient errors.

## Cost Considerations

Each check adds **one parsing LLM call** per question:

| Configuration | Parsing LLM Calls per Question |
|---------------|-------------------------------|
| Neither enabled | 1 (template parsing only) |
| Abstention only | 2 (abstention + parsing) |
| Sufficiency only | 2 (sufficiency + parsing) |
| Both enabled | Up to 3 (abstention + sufficiency + parsing) |

When a check triggers an auto-fail, the parsing call is skipped — so the net cost
may be lower than the maximum for questions where the model refuses or provides
insufficient information.

## Next Steps

- [VerificationConfig Tutorial](verification-config.md) — Configure checks alongside
  other verification settings
- [PromptConfig Tutorial](prompt-config.md) — Customize check behavior with custom
  instructions
- [Python API](python-api.md) — Run verification with quality checks enabled
- [VerificationResult Structure](../07-analyzing-results/verification-result.md) —
  Full reference for abstention and sufficiency result fields
- [Advanced Pipeline](../11-advanced-pipeline/index.md) — Understand how checks fit
  in the 13-stage pipeline
