# Karenina Harmonization and De-Slop Program: Design

**Date:** 2026-06-10
**Status:** Approved design, pending implementation plan
**Scope:** Full sweep of the `karenina/` core library, with karenina-server and karenina-gui co-evolving where core changes force it.
**Spec suite:** Specialized specs that make each aspect of this design concrete are coordinated in [README.md](README.md) (SPEC-000).

---

## 1. Purpose

Harmonize the `karenina/` library: remove slop (structural duplication, dead and inert surface, file-size and naming drift, docs and skills drift), fix the behavior bugs the 2026-05-31 verification engine audit confirmed, converge the divergent execution paths onto one engine, and leave behind a durable in-library knowledge layer so future extension (by humans and by coding agents) is guided by recorded intent rather than archaeology.

This document records the product intent, the guiding principles, the decisions taken during planning, the program phases, and the exit criteria. It is the input to the implementation plan.

---

## 2. Intent and Values

### 2.1 What karenina is for

Karenina is an evaluation platform, built library-first. Its two core jobs:

1. **Build a domain benchmark.** Turn domain expertise into a rigorous, reusable, versioned benchmark: questions, templates, rubrics, curation.
2. **Audit and understand failures.** Not just scores: why a model fails, with traces, classified failures, deep judgment, and error analysis as first-class outputs.

The hill it defends against existing harnesses (lm-eval-harness, Inspect, promptfoo, DeepEval): **domain-expert accessibility with quick setup**. The north star is to be the **scikit-learn of evaluation science**.

Distribution ambition: **open-source community tool**. The paper launches it, then the goal is external adoption, third-party adapters, and shared benchmarks.

### 2.2 Three interface tiers, one library

1. **Domain scientist tier.** A five-line happy path: questions and a model in, results out, zero config beyond credentials. A small set of core nouns (Benchmark, Template, Rubric, Results) with uniform verbs and signatures, so learning one object teaches the rest.
2. **Eval engineer tier.** Reached by progressive disclosure on the same objects: defaults, then named presets, then explicit config objects, then custom ports and adapters. There is no separate advanced API and no migration moment. You only add depth.
3. **Agent tier.** The library is self-documenting to coding agents, which makes agents a universal interface for both human tiers. Concretely:
   - **Skills ship with the package.** The using-karenina skill family is part of the karenina repo and its releases, not only this monorepo's `.claude/`.
   - **Introspectable API.** The library can describe itself at runtime: list adapters, pipeline stages, trait types, and config schemas with their documentation, so an agent discovers capabilities without reading source.
   - **Doctested example gamma.** A curated, executable example per use case (abstracted from `paper_examples/`), tested in CI, serving as the canonical few-shot corpus for agents and the basis of the live smoke tier.

### 2.3 Community surfaces for year one

- **Shareable benchmarks.** The checkpoint/JSON-LD format becomes a stable interchange artifact: benchmarks are published, imported, and remixed across groups.
- **Showcase and docs site.** Polished mkdocs with the example gamma, tutorials per persona tier, and the paper's case studies as flagship walkthroughs.

---

## 3. Guiding Principles (Constitution)

These principles are the constitution that ADRs, contracts, and every harmonization decision appeal to. They move into `karenina/docs/` during Phase 0 so they are versioned with the code.

1. **Identity.** Karenina is an evaluation platform, built library-first. The Python API leads, the GUI is tuned on top of it afterward.
2. **Extensibility: ports plus pluggable pipeline.** Every framework integration is an adapter behind a protocol, registered via the registry, never special-cased in the engine. Engine code never imports an LLM framework. Stages, sinks, and trait types are documented extension surfaces with contracts, not internals.
3. **Configuration: layered surfaces.** A core config that is provably live (a standing parity test makes a knob that changes nothing a test failure), plus an explicit best-effort escape-hatch layer (adapter kwargs passthrough) that is documented as such and not parity-tested.
4. **Failures: no silent degradation, ever.** Every absorbed failure is classified (Failure/Caveats) and visible in the result, or it raises. A dropped feature, a missed cache, a skipped stage must be observable in the output. Silence is the worst bug class.
5. **Execution: one engine, many entries.** A single driver owns scheduling, concurrency, sinks, retries, and resume. QA, Scenario, TaskEval, and the server are thin entry points that project into it. The task model is a **DAG**: a scenario is a graph of turn-tasks, QA is the flat degenerate case, and tournament or comparative workloads fit without reshaping the driver. Concurrency is one global cap plus optional per-model caps. The core is **async-native with a sync public facade**, gated on a compatibility validation against the paper's concrete use cases before commitment (see 6.1, Phase 0 spike).
6. **Reproducibility: results are durable artifacts.** Every run is resumable, every result self-describing (config, model identity, failure metadata embedded), every LLM interaction captured in the replay store. Interruption never loses completed work (turn-level granularity, enabled by the DAG model). Re-running downstream stages never requires regenerating answers.
7. **Simplicity: delete over deprecate.** No external users yet means no shims. Dead code, inert flags, and superseded paths are removed outright in the same PR that obsoletes them. One way to do each thing. YAGNI enforced in review.
8. **Agent-first, self-verifying.** Every extension seam has a machine-checkable contract (test or protocol) and a skill that teaches it, and the codebase actively verifies agent work: conformance suites per seam, parity tests, the live scoreboard, and alignment checkers run as gates. Wrong extensions cannot pass CI.

### 3.1 Future workload pressure (recorded design inputs)

The engine must accommodate, over the next 1 to 2 years:

- The current three kinds (single-turn QA, multi-turn scenarios, pre-collected TaskEval), done well.
- **Long-horizon agentic tasks** (BixBench-style: an agent works for minutes or hours in a sandbox, one verification at the end). Long timeouts, workspace capture, and per-task resources become engine concerns.
- **Comparative and tournament evaluations** (pairwise comparisons, ELO, best-of-N): tasks whose inputs depend on other tasks' outputs. This is the strongest argument for the DAG task model.
- **Online and streaming evaluation** (continuous evaluation of production traffic or live logs): results stream in rather than arriving batch-shaped. The sink protocol and driver must not assume a finite, known-upfront task set.

---

## 4. Decision Record

| Question | Decision |
|---|---|
| Scope | Full library sweep of `karenina/`, starting from the verification engine |
| Slop targets | Structural duplication, dead and inert surface, file-size and naming drift, docs and skills drift |
| Fix policy | Refactor and fix behavior bugs together. Every inert config flag gets a verdict: restore or delete |
| Paper relationship | Unblocked. Results are mostly obtained, the library moves forward freely |
| Compatibility | Break freely. karenina-server and karenina-gui co-evolve when core changes force it |
| Target architecture | The audit's clean-core redesign, amended: DAG task model and async-native core supersede the flat executor merge (ADR-001) |
| Backlog source | Fresh full audit first: re-verify the 47 engine findings, audit the untouched subtrees, merge into one verified, dispositioned backlog |
| Testing | Three tiers: contract/parity rails, recorded goldens, live smoke scoreboard. Audit-then-cull the existing suite |
| Live tier coverage | All adapters. Codon vLLM for volume, z.ai GLM, Anthropic, at least one agentic path (claude_agent_sdk or claude_tool). Abstract cases generalized from `paper_examples/` |
| Process | Phased plan, subagent-driven execution, incremental PRs |
| Knowledge layer audience | Claude agents first. Human docs derive from agent-facing material |
| Backlog home | In-repo and durable: backlog, dispositions, and exit criteria live inside `karenina/`, versioned with the code |

---

## 5. Knowledge Layer

Structuring information and decisions as an integral part of the library is a primary deliverable, not a by-product. Four forms, each implementing a principle:

1. **ADRs** in `karenina/docs/adr/`: numbered decision records (context, decision, alternatives rejected, consequences) for every significant harmonization decision: DAG task model, async-native core, concurrency model, sink protocol, config projection, format stability. The future-workload list (3.1) is recorded as explicit design pressure so decisions carry their justification.
2. **Extension-point guides**: one how-to-extend document per seam (add an adapter, a pipeline stage, a sink, a trait type, a config field). Each states the contract, the files to touch, and the parity or conformance test that will catch mistakes.
3. **Contracts as code**: invariants live as standing tests and Protocol definitions with rich docstrings. The config-parity test, the path-parity tests, and the port protocols are the spec. Docs link to them rather than restating them.
4. **Development skills**: Claude-facing skills for library development (extending the pipeline, touching the engine, adding config), kept aligned via the skill-alignment-checker, and shipped with the package per the agent tier.

The audit backlog and its dispositions become part of this record (in-repo, durable).

---

## 6. Program Phases

### Phase 0: Fresh audit and feasibility spikes

Three parallel audit tracks plus one spike:

- **Engine re-verification.** Re-verify the 47 findings of `verification_engine_audit.md` against the current tree (the report is from 2026-05-31 and the tree has moved).
- **Outer subtree audits.** Schemas, adapters, replay, storage, CLI, integrations, scenario internals, and the gaps the original audit flagged but never verified: replay-store cross-replicate contamination, `workspace_capture.py` filesystem races, MCP session teardown loop affinity, GEPA semaphore clobber.
- **Test-quality audit.** Classify every test file as contract (keep), bug-characterization (delete with the bug), or implementation-mirror (delete during the refactor that touches its target).
- **Async-compatibility spike.** Validate the async-native core choice against each paper use case's adapter surface (langchain, claude_agent_sdk, claude_tool, deep_agents, manual, taskeval) before Phase 3 commits. This is the gate on principle 5.
- **Docs and skills drift pass** (including the wrong stage count in CLAUDE.md: the pipeline is 16 stages, `placeholder_retry_autofail` is omitted).

Output: one merged backlog where every item has a disposition (fix, delete, defer with reason), plus ADR-001 (DAG task model and async core) drafted with spike evidence. The principles in section 3 land in `karenina/docs/` here.

### Phase 1: Feedback harness

Build the rails before touching structure.

- **Tier 1, contract and parity tests (fast, deterministic).** Config-to-pipeline parity (every `VerificationConfig` core field provably reaches its stage or the test fails, killing the F04 family permanently), cross-path parity (QA, Scenario, TaskEval produce equivalent results for equivalent inputs), sink write/kill/resume round-trip, adapter teardown leak check.
- **Tier 2, recorded goldens.** Small benchmarks through each path with fixtures captured from real runs via the existing `capture_fixtures.py` discipline, asserting on final `VerificationResult` content, never on intermediate calls.
- **Tier 3, live smoke scoreboard.** An opt-in marked suite (for example `make engine-health`) running real verification end-to-end: a tiny QA benchmark with known answers, a short scenario, a resume-after-kill drill, a concurrency-cap check. Every adapter covered. Providers: Codon vLLM (volume), z.ai GLM, Anthropic, at least one agentic path. Cases are abstract, generalized from the `paper_examples/` gamma, and double as the doctested example corpus (2.2).

### Phase 2: Quick wins

The effort-S bug fixes from the verified backlog (the F01, F03, F04, F14, F15, F21 family and peers) land as small independent PRs on the corrected baseline, each validated by the new rails.

### Phase 3: Engine redesign

The clean-core migration, amended by ADR-001. Indicative order, refined by the implementation plan:

1. Async lifecycle leaf (or its retirement, per the spike outcome).
2. The unified driver with the DAG task model (supersedes the flat executor-twin merge).
3. One concurrency model (global cap plus per-model caps), respected by every path and adapter.
4. Sink-everywhere: all entry points, including the server's, flow through the driver's sink lifecycle. Resume becomes turn-granular for scenarios.
5. `VerificationContext.from_config` projection, eliminating the four-site hand-maintained marshalling chain.

Implementation-mirror tests are deleted in the same PR that rebuilds their target subsystem.

### Phase 4: Outer subtrees and public surface

- Work the merged backlog across schemas, adapters, replay, storage, CLI. File-size and naming harmonization happens here, per subtree.
- **API-shape workstream**: the tier-1 promise (five-line happy path, uniform verbs across nouns) and progressive disclosure (same objects, deeper kwargs) are designed and applied to the public surface. Harmonizing internals without harmonizing the public API would miss the sklearn goal.

### Phase 5: Consumers, knowledge layer, exit

- karenina-server and karenina-gui co-evolution for whatever the core broke.
- Agent-tier deliverables as features: packaged skills, the introspection API, the doctested example gamma in CI.
- Docs and skills realignment, ADR completeness check, final exit-criteria verification.

---

## 7. Exit Criteria

1. Every core `VerificationConfig` field observably reaches the pipeline or has been deleted, enforced by a standing parity test.
2. One driver. No duplicated executor logic. All entry points (including the server path) flow through it with full sink lifecycle.
3. No module over 800 lines.
4. All backlog items dispositioned (fixed, deleted, or deferred with a recorded reason) in the in-repo record.
5. `engine-health` (live smoke scoreboard) green across all adapters and providers.
6. The five-line happy path works as documented, and the example gamma passes in CI.
7. Skills pass the alignment checker. CLAUDE.md, docs, and ADRs match the code.
8. ruff, mypy, and vulture clean across the library.

---

## 8. Recorded Tensions (ADR seeds)

- **Format stability vs delete-over-deprecate.** "Shareable benchmarks" makes the on-disk checkpoint/JSON-LD format a stability commitment, while the harmonization policy is "break freely." Resolution: format breakage is allowed during this effort, and the stability commitment starts when the harmonized format ships, with a format version field so future migration is possible.
- **DAG generality vs simplicity.** The DAG task model is justified by recorded future workloads (tournaments, streaming), not speculation. If those workloads are abandoned, the ADR records the conditions under which the flat model would have sufficed.
- **Async-native vs migration risk.** Committed conditionally. The Phase 0 spike either confirms compatibility with every showcased use case or the program falls back to contained portals (the audit's original plan e), and the ADR records which branch was taken and why.

---

## 9. Out of Scope

- karenina-server and karenina-gui beyond what core changes force.
- New evaluation features unrelated to harmonization (the future workloads in 3.1 shape the architecture but are not built in this program).
- Paper reproduction on the new code: the paper pins its commit, results are already obtained.
