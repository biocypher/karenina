# SPEC-008: Knowledge Layer and Agent Tier

**Status:** Draft for review
**Charter:** [README.md](README.md) §3, SPEC-008
**Principles:** P8 (agent-first, self-verifying), P2, P6
**Destination:** `docs/adr/`, `docs/extending/`, `skills/` (new, packaged), the introspection API, `examples/`
**Decisions inherited:** Skills single-sourced in the karenina repo, monorepo consumes. Backlog in-repo and durable.

---

## 1. ADRs

- **Location and naming:** `docs/adr/NNN-<slug>.md`, numbered sequentially from 001, never reused. An `index.md` lists all with status.
- **Template (required sections):** Status (proposed / accepted / superseded-by-NNN), Principles (the P-numbers appealed to, per SPEC-001 §3), Context, Decision, Alternatives considered (with why rejected), Consequences (including what becomes harder).
- **When required:** changing an extension contract (port, stage, sink, trait), changing public API shape (new verb, removed noun), amending a principle (SPEC-001 §4), bumping an interchange format version, or resolving a tension from the suite ledger.
- **Seed list (drafted during the program):**
  - ADR-001: DAG task model and async-native engine (evidence: SPEC-009 §5 spike)
  - ADR-002: Composable sub-config restructure (SPEC-004)
  - ADR-003: Sink protocol v2 and resume semantics (SPEC-006)
  - ADR-004: Unified failure taxonomy across adapters and modalities (SPEC-005)
  - ADR-005: Interchange format versioning and stability boundary (SPEC-006 §5)
  - ADR-006: Skills packaging and the agent tier (this spec)
  - ADR-007: Public API verb grammar (SPEC-002)

## 2. Extension-Point Guides

`docs/extending/<seam>.md`, one per seam: **adapter**, **pipeline stage**, **sink**, **rubric trait type**, **config field**. Each guide has exactly four sections:

1. **The contract:** the Protocol or schema, linked to its source, with the invariants stated in prose.
2. **Files to touch:** the complete list for a minimal correct extension.
3. **The test that catches mistakes:** which conformance suite or parity check (SPEC-007 §1) the extension must pass, and how to run it locally.
4. **Worked example:** a link to the smallest real implementation in-tree.

A guide and its conformance suite ship together: an extension seam without both is not public (P2, P8).

## 3. Skill Packaging

- **Single source:** skills move into this repository under `skills/` (using-karenina and the leaf family, plus the development skills below). The monorepo's `.claude/skills/` entries for karenina become a sync target of this folder, never edited directly.
- **Shipping:** `skills/` is included as package data. `karenina skills install [--target DIR]` copies them into a project's `.claude/skills/` (default: nearest project root), so any user's coding agent gets them with one command after `pip install karenina`.
- **Alignment:** the skill-alignment check runs in this repo's CI (merge gate), auditing skills against the current code per the existing checker workflow.
- **Development skills** (new, for contributors and agents working on karenina itself): extending each seam (mirroring §2's guides), adding a config field (the SPEC-004 §5 lifecycle as a checklist), and the error-anatomy review checklist (SPEC-005 §4).

## 4. Introspection API

The library describes itself at runtime, machine-readably. Surface:

```python
import karenina

karenina.describe()                  # full capability document (dict)
karenina.describe("adapters")        # registered adapters + capabilities + granularity
karenina.describe("stages")          # pipeline stages, order, the config that enables each
karenina.describe("traits")          # trait types and their schemas
karenina.describe("config")          # VerificationConfig JSON schema with field docs
```

CLI twin: `karenina describe [section] [--json]`. Sources are the registries and Pydantic schemas themselves (never a hand-maintained list, so it cannot drift). Consumers: coding agents discovering capabilities, the GUI's config surfaces, and the docs build (the config reference page is generated from `describe("config")`).

## 5. The Doctested Example Gamma

`examples/<case>.py`: runnable, abstract-domain scripts (no paper-specific content), each under 80 lines with a docstring stating what it demonstrates. The case list is the SPEC-007 §3 table plus authoring cases: `template_authoring`, `rubric_authoring`, `scenario_authoring`, `manual_answers`, `extend_rerun`.

Triple duty, by construction the same files:

1. **CI-verified docs:** every example runs in CI in fixture mode (tier 2) on merge. An example that breaks fails the build, so documentation cannot rot.
2. **Agent few-shot corpus:** the skills reference these files as canonical patterns (single source for "how do I X").
3. **Live scoreboard bodies:** `make engine-health` runs the engine cases live (SPEC-007 §3).

## 6. Backlog as Durable Record

Per SPEC-009: findings and `BACKLOG.md` live in `specs/audit/`, updated in the PRs that resolve items. When the program completes, `specs/` is retained as the historical record (statuses finalized, a closing note added to this suite's README), and the load-bearing content has graduated: principles to `docs/principles.md`, decisions to `docs/adr/`, contracts to tests, guidance to `docs/extending/` and `skills/`.

## 7. Open Questions

1. Sync mechanism for the monorepo's `.claude/skills/` (symlink vs copy script): decided when skills move, leaning symlink locally and copy in CI.
2. Whether `describe()` output gets a stable schema promise (agents may depend on it): leaning yes at format-version time, recorded in ADR-006.
