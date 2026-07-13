# Change Specification: Evidence-Backed Question Quality Control

| Field | Value |
| --- | --- |
| **Status** | Draft for review |
| **Target package** | `karenina` (Python library) |
| **Primary facade** | `Benchmark` |
| **Product name** | Question quality control (Question QC) |
| **Module** | `karenina.question_qc` |
| **Facade method** | `Benchmark.run_qc` |
| **Config / results** | `QcConfig`, `QcResult`, `QcResultSet` |
| **Related packages** | Optional later: `karenina-server` / `karenina-gui` |
| **Date** | 2026-07-13 |

---

## 1. Summary

Add **evidence-backed question quality control** to Karenina: evaluate **benchmark questions** (question text + expected answer), not model responses.

Using read-only **evidence tools** (existing MCP / `AgentPort`):

1. **Propose** independent evidence that the claimed answer is supported  
2. **Validate** that evidence without trusting the proposer  
3. **Classify** the question’s fitness as ground truth  

Output: **classifications**, **witness** artifacts, and an audit trail.

**API follows the existing verification pattern:**

```python
# Existing
benchmark.run_verification(
    config,                    # VerificationConfig
    question_ids=None,
    run_name=None,
    async_enabled=None,
    progress_callback=None,
) -> VerificationResultSet

# New (same call shape)
benchmark.run_qc(
    config,                    # QcConfig
    question_ids=None,
    run_name=None,
    async_enabled=None,
    progress_callback=None,
) -> QcResultSet
```

Implementation: thin `karenina.question_qc` module + facade on `Benchmark`.  
**Do not** extend the 13-stage model-verification pipeline.

Core remains **backend-agnostic**: no prescribed store, query language, or external application.

---

## 2. Product concept

### 2.1 Problem

Benchmark questions may be unsupported by any checkable system of record, ill-formed as ground truth, or lacking a reproducible **witness**. Scoring models on those questions confounds **dataset quality** with **model quality**.

### 2.2 Principle

**The proposer’s authority is never enough.** Evidence must be re-checked by a separate role. A further independent role classifies the **question**.

### 2.3 Process

```text
Input:  (question, expected_answer, metadata?)
        + read-only evidence tools
        + policy for “supported”

1. PROPOSE   Emit a minimal witness that the claimed answer is supported.
2. VALIDATE  Re-check the witness; accept only if supports_claim ∧ no quality_issues.
3. REVISE    On reject, propose again with structured feedback (bounded), or abandon.
4. REVIEW    Classify the question from attempt history.

Output: classification + witness (when applicable) + audit trail
```

### 2.4 Question outcomes

| Outcome | Meaning |
| --- | --- |
| `supported` | Claim holds; retain witness |
| `unsupported` | Question is clear; system of record does not back the expected answer |
| `ill_formed` | Q/A pair ambiguous, contradictory, or unusable as GT |
| `inconclusive` | Uncertainty / needs rerun or human review |
| `error` / `timed_out` | Operational failure |

**Core artifacts only:** classification + witness (+ traces/metadata). No secondary product-specific artifact fields.

### 2.5 Out of scope of the idea

Not prescribed by core: particular databases, query languages, agent protocols, human UI, or extra artifact kinds beyond witness.

### 2.6 Lifecycle vs model evaluation

```text
Model evaluation (existing)     Question QC (new)
─────────────────────────       ──────────────────────────────
Did the model match GT?         Is this GT good enough to use?
run_verification                run_qc
VerificationConfig / ResultSet  QcConfig / QcResultSet
```

```text
1. Author / import Q+A     → checkpoint (templates may be default stubs)
2. run_qc                  → classifications + witnesses
3. Template authoring      → generate/refine templates for kept questions
4. run_verification        → score models against accepted GT
```

### 2.7 Naming

| Avoid | Why |
| --- | --- |
| **Curation** | Editorial “select/polish set”; weak for multi-role evidence audit |
| **Verification** (for this feature) | Already means model-answer evaluation |
| **Validation** alone | Collides with validator role / template validation |

| Surface | Name |
| --- | --- |
| Docs | Question quality control (Question QC) |
| Module | `karenina.question_qc` |
| Method | `run_qc` |
| Types | `QcConfig`, `QcResult`, `QcResultSet`, `QcLoop` |
| CLI (later) | `karenina qc` |

---

## 3. Why this fits Karenina

| Exists today | Gap |
| --- | --- |
| `Benchmark` owns Q/A, templates, verification, checkpoints | No QC run over questions |
| `ModelConfig` + `get_agent` / `get_parser` | No multi-role propose → validate → review loop |
| MCP for agents | No first-class question-evidence workflow |
| Verification stages | score **answers**, not **questions** (as ground-truth quality) |

**Facade choice:** extend `Benchmark` the same way as `run_verification`—a procedure over the benchmark’s questions.

---

## 4. Goals and non-goals

### 4.1 Goals

1. `Benchmark.run_qc` with the **same argument pattern** as `run_verification`.  
2. Multi-role loop: propose → validate → (revise) → review; isolated sessions.  
3. Evidence via existing `AgentPort` + MCP on `ModelConfig`.  
4. Backend-agnostic domain model (generic witness / tools).  
5. Structured role outputs (Pydantic; optional `ParserPort`).  
6. Auditable `QcResult` / `QcResultSet` (not mixed into `VerificationResult`).  
7. No changes to verification stage semantics.  
8. Composable with later template generation and `run_verification`.

### 4.2 Non-goals (v1)

1. Any specific DB driver / query dialect / prior app stack in core.  
2. Human review UI.  
3. Auto-rewriting of ill-formed questions.  
4. Scenario graph or verification stages as the QC engine.  
5. CLI required in v1 (Python API first; `karenina qc` later).  
6. Artifact types beyond **witness**.

---

## 5. Architecture (minimal change)

### 5.1 Shape

```text
Benchmark.run_qc(config, question_ids, run_name, async_enabled, progress_callback)
        │
        ▼
karenina.question_qc/
  models.py, contracts.py, prompts.py, gates.py
  loop.py      # QcLoop — domain only (Agent protocol)
  runner.py    # batch; get_agent per role; async/progress
        │
        ▼
ports / adapters (unchanged)
  ModelConfig → get_agent / get_parser
  operator-supplied MCP evidence tools
```

Same delegation style as verification: thin `Benchmark` method → manager/runner (cf. `VerificationManager`).

**Hard rule:** `question_qc.loop` must not import adapters, storage engines, or backend clients. Unit-test with fake agents.

### 5.2 Rejected alternatives

| Option | Verdict |
| --- | --- |
| Scenario graph for A↔B revise | Reject — wrong abstraction |
| New stages on `StageOrchestrator` | Reject — answer eval ≠ question QC |
| Standalone product only | Reject — loses checkpoint cohesion |

### 5.3 Facade

```python
def run_qc(
    self,
    config: QcConfig,
    question_ids: list[str] | None = None,
    run_name: str | None = None,
    async_enabled: bool | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> QcResultSet:
    """Run question quality control on the benchmark's question–answer set."""
```

| Argument | Same role as `run_verification` |
| --- | --- |
| `config` | `QcConfig` ↔ `VerificationConfig` |
| `question_ids` | Optional subset |
| `run_name` | Optional run label |
| `async_enabled` | Parallel control (same env-override conventions where practical) |
| `progress_callback` | `(float, str) -> None` |

**Default `question_ids`:** all questions with a non-empty `raw_answer`, **including default stub templates**.  
Unlike verification (defaults to *finished* questions ready for templates), QC is meant to run **before** template authoring.

Facade responsibilities:

- Resolve IDs / eligibility  
- Supply `(question_id, question, raw_answer, metadata)` to the runner  
- Apply async / progress / run_name conventions  
- Return `QcResultSet` only  

### 5.4 Domain loop

```text
for attempt in 1..max_attempts:
  proposal = run_role(PROPOSER, structured_history)
  if proposal.decision == abandon: break
  validation = run_role(VALIDATOR, proposal)
  record attempt
  if validation.passes_evidence_gate: break

review = run_role(REVIEWER, all_attempts)   # always when possible
terminal = derive(review, operational errors)
```

**Evidence gate:**

```text
accept ⇔ (supports_claim is True) ∧ (quality_issues is empty)
```

**Role isolation:** new agent session per role per question (shared weights OK; shared conversation state forbidden).

### 5.5 Role contracts

| Role | Fields (conceptual) |
| --- | --- |
| Proposer | `decision` (`propose` \| `abandon`), `witness`, `explanation`, optional progress |
| Validator | `supports_claim` (`bool` \| `null`), `reasoning`, `quality_issues`, optional evidence summary |
| Reviewer | `classification`, `reasoning`, `quality_issues`, optional remarks |

Witness lives on the proposal and is retained on the result when relevant.

Parsing (v1): Pydantic contracts; prefer `ParserPort` / structured output; bounded invalid-output repair. Not classic template `verify()` against GT—use `gates.py` for programmatic checks (e.g. non-empty witness on propose).

### 5.6 Evidence tools

- Configured via `ModelConfig` MCP fields (and optional per-role allowlists).  
- Recommended capabilities (not hard-coded tool names in domain logic): retrieve/execute evidence; discover catalog; resolve identifiers.  
- Read-only enforcement belongs to the tool server; Karenina may allowlist tool name substrings.  
- Large/truncated tool results → refine witness, not always hard fail.

### 5.7 Prompts

Policy-oriented core prompts. Optional **profiles** only adjust vocabulary/examples for backend classes—no loop forks. Domain field names stay generic (`witness`, `supports_claim`, `classification`).

---

## 6. Data model

```python
class QcClassification(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    ILL_FORMED = "ill_formed"
    INCONCLUSIVE = "inconclusive"

class Proposal(BaseModel):
    decision: Literal["propose", "abandon"]
    witness: str = ""
    explanation: str = ""
    abandon_reason: str = ""
    progress_report: str = ""
    raw_response: str = ""

class Validation(BaseModel):
    supports_claim: bool | None
    reasoning: str
    quality_issues: str = ""
    evidence_summary: dict[str, Any] | None = None
    progress_report: str = ""
    raw_response: str = ""

    @property
    def passes_evidence_gate(self) -> bool: ...

class Review(BaseModel):
    classification: QcClassification | None
    reasoning: str
    quality_issues: str = ""
    remarks: str = ""
    progress_report: str = ""
    raw_response: str = ""

class QcAttempt(BaseModel):
    number: int
    proposal: Proposal
    validation: Validation | None

class QcResult(BaseModel):
    question_id: str
    attempts: list[QcAttempt]
    review: Review | None
    terminal_status: str
    error_stage: str = ""
    error_message: str = ""
    # model identities, timings, usage, optional redacted tool traces
    # convenience: last/accepted witness when supported

class QcResultSet:
    """Collection of QcResult with filter/export helpers (cf. VerificationResultSet)."""
    ...
```

**Do not** overload `VerificationResult`.

### 6.1 Checkpoints

| Concern | Behavior |
| --- | --- |
| Input | Questions on the checkpoint (`question`, `raw_answer`, metadata); default templates OK |
| Schema change (v1) | **None** — QC results are a separate run export |
| Optional later | Metadata write-back; filtered checkpoint of `supported` questions |

Checkpoint = definition. `run_qc` = run over that definition (same spirit as verification results vs definition).

---

## 7. Configuration and API

### 7.1 Split (mirror verification)

| Lives on | Contents |
| --- | --- |
| **`QcConfig`** | How the run works (roles, runtime, prompts, context) |
| **`run_qc(...)` args** | `question_ids`, `run_name`, `async_enabled`, `progress_callback` |

### 7.2 `QcConfig` (illustrative)

```python
class RoleModelConfig(BaseModel):
    model: ModelConfig
    system_prompt_override: str | None = None
    allowed_tool_substrings: list[str] = Field(default_factory=list)

class QcRuntimeConfig(BaseModel):
    max_attempts: int = 3
    invalid_output_retries: int = 1
    role_timeout_seconds: float | None = None
    async_max_workers: int | None = None  # align with verification workers when set

class QcConfig(BaseModel):
    """Question QC run configuration (analogue of VerificationConfig)."""
    proposer: RoleModelConfig
    validator: RoleModelConfig
    reviewer: RoleModelConfig
    runtime: QcRuntimeConfig = Field(default_factory=QcRuntimeConfig)
    prompt_profile: str | None = None
    evidence_context_path: Path | None = None
```

Reuse `ModelConfig` (including MCP). Do **not** put `question_ids` / `run_name` / `async_enabled` / `progress_callback` on `QcConfig`.

### 7.3 Example

```python
from karenina import Benchmark
from karenina.question_qc import QcConfig, RoleModelConfig
from karenina.schemas.config import ModelConfig

benchmark = Benchmark.load("questions.jsonld")

agent = ModelConfig(
    id="qc",
    model_name="...",
    model_provider="...",
    interface="langchain",
    mcp_urls_dict={...},
    mcp_tool_filter=[...],
)

config = QcConfig(
    proposer=RoleModelConfig(model=agent),
    validator=RoleModelConfig(model=agent),
    reviewer=RoleModelConfig(model=agent),
)

results = benchmark.run_qc(
    config,
    question_ids=None,
    run_name="qc-pass-1",
    async_enabled=True,
    progress_callback=None,
)

results = benchmark.run_qc(config, question_ids=[qid1, qid2])
supported = results.filter(terminal="supported")  # illustrative helper
```

### 7.4 Runtime controls

| Control | v1 | Later |
| --- | --- | --- |
| `max_attempts`, repair retries | Yes | — |
| Per-role models | Yes | — |
| Facade args (`question_ids`, `run_name`, `async_enabled`, `progress_callback`) | Yes | — |
| Per-role timeout | Yes | — |
| Steering / active-time budgets | Optional | Recommended |
| Progressive save / resume | Optional | Recommended |
| CLI `karenina qc` | Optional | Recommended |

---

## 8. Integration

| Feature | Integration |
| --- | --- |
| `Benchmark` | `run_qc` facade |
| Checkpoints | Input; no required format break |
| Template generation | Downstream of QC |
| `run_verification` | Separate pipeline; after QC + templates |
| MCP docs | Tools for answering **and** question evidence |
| Agentic evaluation | Shared ports only |
| TaskEval / scenarios | Not the primary QC path |
| Rubrics | Optional secondary; not the evidence gate |

---

## 9. Phased delivery

### Phase 0 — Spec

- [x] Abstract Question QC concept  
- [x] `run_qc` mirrors `run_verification`  
- [x] No secondary artifacts; no backend lock-in  
- [x] Naming: product “Question QC”, API `run_qc` / `Qc*`  

### Phase 1 — Domain core

1. `karenina.question_qc` models, contracts, gates, prompts  
2. `QcLoop` + fake-agent unit tests  
3. `QcResult` / `QcResultSet` + JSON export  

### Phase 2 — Ports + Benchmark

1. `AgentPort` turn adapter  
2. Runner (batch, async, progress)  
3. `Benchmark.run_qc`  
4. Mock-MCP integration test  

### Phase 3 — Operability

1. Timeouts / optional steering  
2. Progressive save / resume  
3. Optional metadata write-back / filtered export  
4. Optional `karenina qc`  
5. Docs  

### Phase 4 — Ecosystem (optional)

Human review, analytics views, presets for common evidence-tool setups.

---

## 10. Testing

| Layer | Focus |
| --- | --- |
| Unit | Loop, gates, parsing, terminals |
| Unit | Runner session isolation (mocks) |
| Integration | `get_agent` + mock evidence tools |
| Integration | `run_qc` on Q+A-only checkpoint (default templates) |
| Regression | Verification / authoring unchanged |

---

## 11. Documentation plan

| Doc | Action |
| --- | --- |
| `docs/core_concepts/question-quality-control.md` | Concept: Question QC vs model verification |
| `docs/workflows/question-quality-control/` | Using `run_qc` |
| `docs/core_concepts/evaluation-modes.md` | Mention Question QC |
| `docs/core_concepts/mcp-overview.md` | Answering tools + question evidence tools |
| Checkpoints docs | QC is a run over definition |
| README | Feature bullet |

---

## 12. Risks

| Risk | Mitigation |
| --- | --- |
| Multi-agent framework creep | Fixed three-role loop only |
| Backend leakage | Generic names; tools via config |
| Confused with verification | Separate types + `run_qc` name |
| Cost | Defaults, filters, docs |
| Checkpoint pollution | Results external in v1 |

---

## 13. Decisions (locked for this draft)

| Topic | Decision |
| --- | --- |
| Facade | `Benchmark.run_qc` |
| Arguments | Same as `run_verification`: `config`, `question_ids`, `run_name`, `async_enabled`, `progress_callback` |
| Config type | `QcConfig` (not on-config: question_ids / run control) |
| Default questions | All with `raw_answer` (incl. default templates) |
| Product language | Question quality control |
| Avoid | “Curation” as API; secondary inspection artifacts |
| Reviewer | Always run when possible |
| v1 persistence | JSON export; progressive save later |
| Witness write-back | Not required in v1 |

### Still open (non-blocking)

1. Exact `QcResultSet` helper API (filter/export names).  
2. Whether `async_max_workers` lives only on `QcRuntimeConfig` or also env vars like verification.  
3. CLI flag surface for `karenina qc` (mirror `verify` as much as practical).

---

## 14. Success metrics

1. Load/build `Benchmark` with Q+A only → attach evidence tools → `run_qc` → classifications + witnesses.  
2. Domain loop unit-tested with fake agents (no network/backend).  
3. Verification pipeline untouched; existing tests green.  
4. Docs separate question quality from model quality.  
5. Core has no dependency on a particular store or query language.

---

## 15. Acceptance checklist (Phase 1–2)

- [ ] `run_qc(config, question_ids=..., run_name=..., async_enabled=..., progress_callback=...)` on `Benchmark`  
- [ ] Delegates to `karenina.question_qc`; same arg roles as `run_verification`  
- [ ] Default question set = questions with `raw_answer`  
- [ ] Fake three-role suite: gate, revise, abandon, review, repair  
- [ ] `QcResult` / `QcResultSet` distinct from verification results  
- [ ] No backend-specific imports in domain loop  
- [ ] No artifact fields beyond witness in core contracts  
- [ ] Mock evidence-tool integration green  
- [ ] Q+A-only checkpoint (default templates) works as input  
- [ ] Verification pipeline untouched  

---

## 16. Decision log

| Date | Decision |
| --- | --- |
| 2026-07-13 | Initial draft (concrete prior pipeline references) |
| 2026-07-13 | Abstract multi-role Question QC; minimal `question_qc` module; no backend lock-in |
| 2026-07-13 | Drop secondary inspection artifacts; prefer “Question QC” over “curation” |
| 2026-07-13 | **`Benchmark.run_qc`** with the same arguments as **`run_verification`** |
| 2026-07-13 | Spec consolidated under this document |
| 2026-07-13 | Prefer **question** terminology over **item** throughout |

---

## 17. Appendix — Non-normative examples

Operators may realize evidence tools however they like. **Not** required core types:

| Abstract | Example realizations |
| --- | --- |
| System of record | Graph DB, warehouse, search index, documents, HTTP APIs |
| Witness | Query string, procedure call, doc ids + quotes, request plan |
| Execute evidence | Read-only fetch/query tool |
| Discover catalog | Schema / index listing tool |

