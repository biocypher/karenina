# SPEC-001: Constitution

**Status:** Draft for review
**Charter:** [README.md](README.md) §3, SPEC-001
**Destination:** `docs/principles.md` in this repository, published on the mkdocs site

---

## 1. Destination and Format

The constitution is a single page, `docs/principles.md`, added to the mkdocs nav in the same section as `contributing.md` under the title "Design Principles". The page contains: a short preamble (identity and north star), the eight principles, the recorded design pressures, the citation convention, and an amendment changelog at the bottom.

Principles are numbered P1 through P8. Numbers are permanent: a retired principle keeps its number and is marked retired, numbers are never reused.

## 2. The Constitution (final wording)

The text below is the quotable, binding wording as it will appear in `docs/principles.md`.

---

### Preamble

Karenina is an evaluation platform, built library-first. Its two core jobs are turning domain expertise into rigorous, reusable benchmarks and making model failures understandable. Its north star is to be the scikit-learn of evaluation science: domain-expert accessible, quick to set up, uniform to learn, deep when needed. It serves three interface tiers on one library: a five-line happy path for domain scientists, progressive disclosure on the same objects for evaluation engineers, and a self-documenting surface that makes coding agents a universal interface for both.

These principles bind design decisions in this repository. Code review, ADRs, and specs cite them by number. If a change cannot be reconciled with a principle, either the change is wrong or the principle must be amended through the process in §4, never silently.

### P1: Library first

The Python API is the leading interface. Server and GUI are tuned on top of it and never drive core design. Anything the platform can do must be doable from a notebook.

### P2: Ports plus pluggable pipeline

Every framework integration is an adapter behind a port protocol, registered through the registry, never special-cased in the engine. Engine code never imports an LLM framework. Pipeline stages, result sinks, and rubric trait types are documented extension surfaces with contracts and conformance tests, not internals.

### P3: Layered configuration

Configuration has exactly two layers. Core fields are provably live: a standing parity test makes a knob that changes nothing a test failure. The escape hatch (adapter kwargs passthrough) is explicitly best-effort, marked as such in schemas and docs, and not parity-tested. No third category exists.

### P4: No silent degradation

Every absorbed failure is classified and visible in the result, or it raises. A dropped feature, a missed cache, a skipped stage must be observable in the output. Silence is the worst bug class.

### P5: One engine, many entries

A single driver owns scheduling, concurrency, sinks, retries, and resume. Entry points (QA, scenarios, TaskEval, server) are thin projections into it. The task model is a DAG: scenarios are graphs of turn-tasks, single-turn QA is the flat degenerate case. The core is async-native with a sync public facade.

### P6: Results are durable artifacts

Every run is resumable at task granularity, every result self-describing, every LLM interaction captured for replay. Interruption never loses completed work. Re-running downstream stages never requires regenerating answers.

### P7: Delete over deprecate

Dead code, inert flags, and superseded paths are removed in the PR that obsoletes them. No shims, no compatibility fallbacks, one way to do each thing. The exception is the published benchmark interchange format, which carries a version field and a stability commitment from the moment it ships.

### P8: Agent-first, self-verifying

Every extension seam has a machine-checkable contract and a skill that teaches it. The codebase verifies extension work mechanically: conformance suites per port, parity tests, the live scoreboard, and alignment checkers run as gates. An incorrect extension cannot pass CI.

### Recorded design pressures

The architecture must accommodate, without reshaping (recorded 2026-06, revisit yearly):

- The current workloads: single-turn QA, multi-turn scenarios, pre-collected TaskEval.
- Long-horizon agentic tasks: an agent working for minutes or hours in a sandbox, verified at the end. Long timeouts, workspace capture, and per-task resources are engine concerns.
- Comparative and tournament evaluations: tasks whose inputs depend on other tasks' outputs. This pressure justifies the DAG task model in P5.
- Online and streaming evaluation: results arriving continuously rather than batch-shaped. The driver and sink protocol must not assume a finite, known-upfront task set.

---

## 3. Citation Convention

- First mention in a document: number plus name, for example "P4 (no silent degradation)". Subsequent mentions: bare number.
- ADRs list the principles they appeal to in a `Principles:` header line.
- PR descriptions and reviews cite principles when invoking them ("this violates P7" is a complete review argument, the burden shifts to the author).
- Contract tests that enforce a principle name it in their module docstring, so a failing test points back to the rule it defends.

## 4. Amendment Process

Amending a principle requires an ADR that records: the current text, the proposed text, the concrete trigger (what decision could not be reconciled), and the contracts or tests affected. The `docs/principles.md` page is updated in the same PR as the accepted ADR, with a one-line entry in the page's changelog. Retired principles keep their number and a pointer to the retiring ADR.

## 5. Open Questions

None. All charter questions answered above.
