# SPEC-004: Configuration Layering

**Status:** Draft for review
**Charter:** [README.md](README.md) §3, SPEC-004
**Principles:** P3 (layered configuration), P4, P7
**Destination:** ADR (config restructure), `schemas/config/` and `schemas/verification/` (code), parity test (SPEC-007), config reference docs
**Decisions inherited:** Composable sub-configs.

---

## 1. The Core vs Escape-Hatch Criterion

A field is **core** if and only if the engine or pipeline reads it to alter behavior and its effect is assertable by a test. Core fields are parity-tested (§3) and documented in the config reference.

The **escape hatch** is exactly one surface: `ModelConfig.adapter_overrides: dict[str, Any]`, passed verbatim to the adapter at construction. It is best-effort by contract: unknown keys are the adapter's problem, no parity test, schema description and docs state this explicitly. No other best-effort field may exist (a typed field that is sometimes honored is a P3 violation).

The field-by-field disposition of every existing `VerificationConfig` and `ModelConfig` field (core, escape hatch, or delete) is produced by Phase 0 Track B applying this criterion, recorded in the audit backlog. Known dead-on-arrival candidates from the original audit (the F04/F05/F15/F29 silent-drop family) enter that table with their SPEC-009 §4 rule-3 disposition.

## 2. Structure: Composable Sub-Configs

`VerificationConfig` becomes a thin composition. Each sub-config maps to exactly one pipeline region, which is what makes the parity test and the projection (§4) natural:

```python
class VerificationConfig(BaseModel):
    generation: GenerationConfig       # answering models, system prompt, few-shot, MCP
    judging: JudgingConfig             # parsing model, evaluation mode, deep judgment, embedding check
    rubric: RubricConfig | None        # trait selection, agentic rubric, dynamic rubric
    guards: GuardsConfig               # abstention, sufficiency, recursion, trace validation, placeholder retry
    tuning: ExecutionTuning            # SPEC-003 §3: caps, backpressure
    replay: ReplayConfig               # capture on/off, store location (SPEC-006)
```

| Sub-config | Pipeline region it feeds |
|---|---|
| `generation` | `generate_answer` stage and the adapter gateway |
| `judging` | `parse_template`, `verify_template`, `embedding_check`, deep judgment stages |
| `rubric` | rubric evaluation stages |
| `guards` | the autofail/guard stages |
| `tuning` | the driver only (never read by stages) |
| `replay` | capture hooks only (never alters pipeline behavior) |

Tier 1 never constructs these: every sub-config has complete defaults, and `VerificationConfig()` is valid. Tier-2 ergonomics (flat kwarg conveniences, presets) are owned by SPEC-002. Migration is delete-over-deprecate (P7): the old flat field names are removed, not aliased.

## 3. The Parity Test ("provably live")

Mechanism, two parts:

1. **Completeness check.** A test introspects the `VerificationConfig` schema (recursively over sub-configs), collects every core field path (for example `judging.deep_judgment_enabled`), and fails if any path lacks a registered parity case. Adding a field without a parity case is a red CI.
2. **Per-field parity cases.** A registry maps each field path to a `ParityCase`: two configs differing only in that field, a fixture-backed run, and an assertion on the observable difference (selected stage list, prompt content, artifact value, or result field). The case fails if the two runs are indistinguishable, which is the mechanical meaning of "a knob that changes nothing is a test failure".

The registry lives next to the tests (`tests/parity/cases/`), one module per sub-config. Escape-hatch content (`adapter_overrides`) is excluded by construction since it is not a schema field path.

## 4. How Config Reaches the Pipeline

The four-site hand-maintained marshalling chain (config → kwargs → `VerificationContext` → stages, the F30 root cause) is replaced by direct projection:

- The projection (SPEC-003 §1.3) builds a `TaskSpec` per task: the full `VerificationConfig` plus per-task bindings (question, template, answering and parsing model, replicate).
- Stages receive the `TaskSpec` and read their own sub-config directly (`spec.config.guards.abstention_enabled`). No re-flattening into context fields, no per-field copying, nothing to forget.
- The context object shrinks to: the `TaskSpec`, the artifact bag, and the result builder. The ~76-field `VerificationContext` is deleted.
- `StageOrchestrator.from_config` selects stages by reading sub-configs through one function with exhaustive handling, unit-tested per sub-config.

## 5. Field Lifecycle

Adding a core config field requires, in one PR: the field with description, the wiring that makes it live, its parity case, and a config-reference docs entry. The completeness check (§3.1) enforces the parity case mechanically, the PR checklist (SPEC-008 development skill) covers the rest. Removing a field follows P7: field, wiring, parity case, and docs entry deleted together.

## 6. Open Questions

1. Per-field disposition table: produced by Phase 0 Track B (resolution path stated in §1).
2. Exact membership of each sub-config: finalized by the same Track B table, the §2 region mapping is the assignment rule.
3. Preset interaction (named presets serializing sub-config compositions): owned by SPEC-002.
