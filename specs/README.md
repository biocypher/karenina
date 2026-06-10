# Karenina Harmonization Spec Suite: Central Coordinator

**Date:** 2026-06-10
**Status:** Living index, updated as specs are drafted and approved
**Parent:** [harmonization-master-design.md](harmonization-master-design.md) (the master design)
**Location note:** This suite lives inside the karenina library (`karenina/specs/`) per the knowledge-layer principle: decisions are versioned with the code they govern. Approved content graduates from here into its final destination (`docs/adr/`, docs pages, contract tests, skills).

---

## 1. Purpose

The master design records intent, values, principles, and program phases. This suite turns each value and direction into a concrete, unambiguous specification. Each specialized spec owns one aspect, answers a defined set of questions, and is the single source of truth for that aspect. This document coordinates them: scope boundaries, dependency order, status, and the ledger of cross-spec tensions.

Rules of the suite:

- **One owner per question.** Every concrete question is answered in exactly one spec. Other specs link, never restate.
- **Concrete or absent.** A spec ships with real signatures, real schemas, real file paths, real test names. If an aspect cannot be made concrete yet, it is listed as an open question with what would resolve it, not prose-papered.
- **Master stays thin.** When a specialized spec is approved, the master's corresponding section becomes a summary plus a link.
- **Destination noted.** Each spec states where its content ultimately lives in the library (ADR, docs page, contract test, skill), per the knowledge-layer principle.

---

## 2. The Suite

| ID | Spec | Owns | Status |
|---|---|---|---|
| SPEC-000 | This document | Coordination, status, cross-spec tensions | Living |
| SPEC-001 | [Constitution](SPEC-001-constitution.md) | Final wording of the principles destined for `karenina/docs/` | Draft for review |
| SPEC-002 | Public API and tiered interface | Nouns, verbs, signatures, five-line path, disclosure rules | To write |
| SPEC-003 | [Execution model](SPEC-003-execution-model.md) | DAG task model, driver, concurrency, async core, sync facade | Draft for review |
| SPEC-004 | Configuration layering | Core vs escape hatch, field lifecycle, parity enforcement | To write |
| SPEC-005 | Failure and observability | Failure taxonomy, no-silent-degradation rules, error quality | To write |
| SPEC-006 | Results durability and interchange | Sinks, resume semantics, replay, format versioning, shareable benchmarks | To write |
| SPEC-007 | Testing and feedback harness | Three tiers, engine-health scoreboard, cull rubric, conformance suites | To write |
| SPEC-008 | Knowledge layer and agent tier | ADR process, extension guides, packaged skills, introspection API, example gamma | To write |
| SPEC-009 | [Audit methodology and backlog](SPEC-009-audit-methodology.md) | Phase 0 process, finding schema, disposition workflow | Draft for review |

Specs may be merged or split as drafting reveals their true size. Merges and splits are recorded here.

---

## 3. Per-Spec Charters

Each charter lists the concrete questions the spec must answer. A spec is approved when every charter question has either a concrete answer or an explicit open-question entry with a resolution path.

### SPEC-001: Constitution

Final, quotable wording of the eight principles and the future-workload design pressures, formatted for their destination in `karenina/docs/`. Must answer:

1. Exact text of each principle as it will appear in the library.
2. The destination file layout (single principles page vs page-per-principle, relationship to ADR index).
3. How principles are cited from ADRs, PRs, and reviews (naming/numbering convention).
4. The amendment process: what it takes to change a principle later.

### SPEC-002: Public API and Tiered Interface

The sklearn promise made concrete. Must answer:

1. The exact five-line happy path: real code, runnable, with the chosen defaults named (default judge model, default template behavior, default pipeline stages).
2. The core noun set and the uniform verb grammar: which objects exist at tier 1, their method names and signatures, and the consistency rules new API must obey.
3. Progressive disclosure mechanics: the precise ladder (defaults, presets, config objects, ports) with one worked example showing the same task at each rung, demonstrating "you only add, never migrate."
4. What is deliberately NOT in tier 1 (the surface that requires the engineer tier).
5. Naming and signature conventions as enforceable rules (the consistency contract a linter or review checklist can apply).
6. Backwards mapping: which current public API survives, which is renamed, which is deleted (delete-over-deprecate applied to the API surface).

### SPEC-003: Execution Model

The engine's concrete architecture. Must answer:

1. The Task abstraction: dataclass/protocol definition, identity (what replaces the result_key 4-tuple), dependency declaration, and how QA, scenario turns, TaskEval items, and future tournament tasks map onto it.
2. The driver: its public interface, scheduling semantics (ready-set execution over the DAG), and how the four entry points project into it.
3. Concurrency: where the global cap and per-model caps live, their config surface, and the guarantee statement (which calls they cover, including agent adapters).
4. Async-native core: the boundary line (what is async, what is sync facade), the bridging mechanism, adapter async requirements, and the fallback plan if the Phase 0 spike fails.
5. Lifecycle: resource ownership (httpx clients, MCP sessions, portals if any), startup/teardown contract, leak detection.
6. What survives from the current code (orchestrator, stages) vs what is replaced (executors, batch_runner), file by file.

### SPEC-004: Configuration Layering

Must answer:

1. The criterion deciding whether a field is core (provably live) or escape hatch (best effort), and the audit disposition for every existing `VerificationConfig` and `ModelConfig` field.
2. The parity test design: how "provably live" is mechanically enforced (the test that fails when a knob changes nothing).
3. The field lifecycle: what adding a config field requires (wiring, test, docs entry) before it can merge.
4. The escape-hatch contract: exact passthrough semantics, what is promised and what is not, how it is marked in schemas and docs.
5. How config reaches the pipeline: the `from_config` projection design that replaces the four-site marshalling chain.

### SPEC-005: Failure and Observability

Must answer:

1. The failure taxonomy: the closed set of failure and caveat categories, their schema, and the mapping from current `Failure`/`Caveats` types.
2. The raise-vs-record decision table: for each error site class (config error, adapter error, stage error, sink error, teardown error), whether it raises, fails the task, or records a caveat, and why.
3. The visibility guarantee: how a user (or agent) discovers every absorbed failure from the result object alone.
4. Error message quality rules: the required anatomy of an error (what, why, next action) and how it is reviewed/tested.
5. Logging policy: levels, what is logged where, the no-swallowed-exception rule for cleanup paths.

### SPEC-006: Results Durability and Interchange

Must answer:

1. Resume semantics: the exact contract (what is completed, what is retried, transient-vs-permanent failure handling fixing F01), at task granularity in the DAG model.
2. Sink protocol v2: the hook set, thread/async safety ownership, and the streaming-friendly lifecycle (no finite-task-set assumption).
3. Replay store contract: capture guarantees, keying (fixing F03's canonical-key mismatch), cross-replicate isolation, and the extend_* miss behavior.
4. The result artifact: the self-description requirement (config, model identity, failure metadata embedded) as a schema.
5. The interchange format: checkpoint/JSON-LD versioning scheme, the format version field, and the stability commitment boundary (what is stable for sharing, what remains internal).

### SPEC-007: Testing and Feedback Harness

Must answer:

1. Tier 1 contract tests: the named list (config parity, path parity, resume round-trip, teardown leak, conformance suites per port), each with its assertion strategy.
2. Tier 2 goldens: fixture capture process, which paths and configs get goldens, and the update policy when behavior legitimately changes.
3. Tier 3 live scoreboard: the exact case list (mapped to the example gamma), provider matrix (Codon vLLM, z.ai GLM, Anthropic, agentic path), cost budget, flake policy, and the `make engine-health` interface.
4. The cull rubric: the classification criteria (contract / bug-characterization / implementation-mirror) operationalized so subagents can apply them consistently, and the deletion workflow.
5. CI gating: which tiers run on PR, on merge, on demand.

### SPEC-008: Knowledge Layer and Agent Tier

Must answer:

1. ADR format and process: template, numbering, where they live (`karenina/docs/adr/`), when one is required, and the seed list (DAG model, async core, concurrency, sink v2, config projection, format stability).
2. Extension-point guides: the guide-per-seam list, the required sections (contract, files to touch, the test that catches mistakes), and their pairing with conformance suites.
3. Skill packaging: how the using-karenina skill family ships with the karenina package (mechanism, versioning, sync with the monorepo copies).
4. Introspection API: the concrete surface (functions/CLI for listing adapters, stages, traits, config schemas with docs) and its consumers.
5. The doctested example gamma: the case list abstracted from `paper_examples/`, where they live, how they run in CI, and their dual role as agent few-shot corpus and live-tier basis.
6. Backlog-as-record: where the audit backlog and dispositions live in-repo and their schema (shared with SPEC-009).

### SPEC-009: Audit Methodology and Backlog

Must answer:

1. Track definitions for Phase 0: engine re-verification, outer subtrees, test quality, docs/skills drift, async spike, each with scope, method, and output schema.
2. The finding schema: fields (id, severity, location, issue, evidence, effort, fix direction) and the disposition states (fix, delete, defer-with-reason, rejected).
3. The verification standard: what "code-verified" means (every claim has a file:line citation against the current tree).
4. Subagent orchestration: how audit tracks are parallelized, reviewed, and merged into the single backlog.
5. The async spike protocol: per-adapter compatibility checks, pass/fail criteria, and how the result selects between async-native and contained-portals.

---

## 4. Dependency Order

```
SPEC-001 (constitution)  ──────────────┐
SPEC-009 (audit method)  ── Phase 0 ───┤
                                       ▼
SPEC-004 (config)  ◄── informs ── SPEC-003 (execution model)
SPEC-005 (failure) ◄── informs ──┘        │
SPEC-006 (durability) ◄───────────────────┘
SPEC-002 (public API)  ◄── constrained by 003/004/006
SPEC-007 (testing) ◄── consumes contracts from 003/004/005/006
SPEC-008 (knowledge/agent tier) ◄── packages everything
```

Drafting order: 001 and 009 first (they unblock Phase 0), then 003 (the architectural keystone), then 004/005/006 (its facets), then 002, 007, 008. Specs can be drafted before their dependencies are final, with open questions pointing at the blocking spec.

---

## 5. Cross-Spec Tension Ledger

| # | Tension | Specs | Resolution status |
|---|---|---|---|
| T1 | Format stability vs delete-over-deprecate | 006, 001 | Resolved in master §8: breakage allowed until harmonized format ships with version field |
| T2 | DAG generality vs simplicity | 003 | Resolved by recorded workload pressure (master §3.1); ADR-001 records fallback conditions |
| T3 | Async-native vs migration risk | 003, 009 | Open: gated on the Phase 0 spike (SPEC-009 Q5 defines the gate) |
| T4 | Five-line happy path vs no-silent-degradation | 002, 005 | Open: defaults must be quiet enough for tier 1 yet hide nothing; needs a joint rule |
| T5 | Skills in-package vs monorepo skill ecosystem | 008 | Open: single source with sync mechanism to be chosen |
| T6 | Streaming workloads vs sink/resume design | 003, 006 | Open: sink v2 must not assume finite task sets; concrete hooks TBD in 006 |

New tensions discovered during drafting are added here, never resolved silently inside one spec.

---

## 6. Status Log

- 2026-06-10: Suite created with charters for SPEC-001 through SPEC-009. Master design approved.
- 2026-06-10: Suite relocated from the monorepo planning folder into `karenina/specs/`, versioned with the library.
- 2026-06-10: SPEC-001 and SPEC-009 drafted. Decisions: constitution as one page with P1-P8 numbering and ADR-gated amendments. Audit dispositions made autonomously by agents per decision rules, user spot-checks flagged items plus a 10% sample.
- 2026-06-10: SPEC-003 drafted. Decisions: driver public-but-experimental (`karenina.experimental.engine`), streaming-shaped core from the start, TaskEval fully projected onto the driver. Single `TaskId` identity, AdapterGateway as the one concurrency chokepoint, sync facade via a background-loop bridge module.
