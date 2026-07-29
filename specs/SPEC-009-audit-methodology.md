# SPEC-009: Audit Methodology and Backlog

**Status:** Draft for review
**Charter:** [README.md](README.md) §3, SPEC-009
**Destination:** Process executed in Phase 0. The backlog it produces lives at `specs/audit/` and is the durable in-repo record. The methodology itself graduates to an ADR appendix once Phase 0 completes.

---

## 1. Phase 0 Tracks

Five tracks, run as parallel subagent workstreams. All findings are recorded against a pinned commit of this repository, stated in the backlog header. Note that the `fix/async-consistency` branch already contains in-flight remediation (for example `benchmark/verification/async_lifecycle.py` exists), so re-verification must run against the current tree, not the audit-era tree.

| Track | Scope | Method | Output |
|---|---|---|---|
| A: Engine re-verification | The 47 findings of `verification_engine_audit.md` | Per-finding re-verification against the pinned tree: confirm, amend, or mark already-fixed, with fresh file:line evidence | `specs/audit/findings-engine.md` |
| B: Outer subtrees | `schemas/`, `adapters/`, `replay/`, `storage/`, `cli/`, `integrations/`, `scenario/`, `utils/`, plus the four flagged gaps (replay cross-replicate contamination, `workspace_capture.py` races, MCP teardown loop affinity, GEPA semaphore clobber) | Structured audit per subtree against the four slop categories (duplication, dead/inert surface, size and naming drift, docs drift) and P1-P8 | `specs/audit/findings-<subtree>.md`, one file per subtree |
| C: Test quality | Every test file under `tests/` | Classify each file (or coherent class within a file): contract, bug-characterization, or implementation-mirror, per the rubric in SPEC-007 | `specs/audit/findings-tests.md` |
| D: Docs and skills drift | `docs/`, `.claude/skills/` (monorepo), `CLAUDE.md` files | Diff documented behavior against code (known seed: pipeline is 16 stages, `placeholder_retry_autofail` missing from the monorepo CLAUDE.md list) | `specs/audit/findings-docs.md` |
| E: Async spike | All adapters, paper use-case surface | Protocol in §5 | `specs/audit/spike-async.md` |

## 2. Finding Schema

One finding is one record in a track's findings file, in this exact shape:

```markdown
### <ID>: <one-line title>

- severity: high | medium | low
- effort: S | M | L
- location: <path>:<line>[, <path>:<line>...]
- principle: <P1..P8, the principle violated, or "slop" if none applies>
- issue: <what is wrong, one paragraph>
- evidence: <how it was verified: code path walked, test run, reproduction>
- fix_direction: <one sentence>
- disposition: fix | delete | defer | reject
- disposition_rationale: <one sentence citing a decision rule from §4 or "flagged">
- flagged: yes | no  (yes routes it to user review)
```

ID convention: track prefix plus counter. `ENG-001` (Track A keeps a mapping line to the original F-number), `SCH/ADP/RPL/STO/CLI/INT/SCN/UTL-001` per subtree, `TST-001`, `DOC-001`.

Severity and effort keep the original audit's scale for continuity. Severity is judged by user impact on the platform's core jobs (benchmark building, failure auditing), not code aesthetics.

## 3. Verification Standard

A claim is admissible only with a file:line citation against the pinned commit, plus a stated verification method (code path walked end to end, behavior reproduced, or test executed). Findings without evidence are not merged into the backlog. Track A findings additionally state whether the original F-number was confirmed, amended (with what changed), or already fixed (with the fixing commit if identifiable).

## 4. Disposition Workflow

Dispositions are made autonomously by the auditing agents, applying these decision rules in order. The user spot-checks rather than ratifies.

Decision rules:

1. **Violates P4 (silent degradation) → fix.** Never delete a silence by documenting it.
2. **Dead code (unreachable, unused, superseded) → delete**, per P7.
3. **Inert config flag**: if a showcased use case (the example gamma, `paper_examples/`) or the GUI exercises the intent behind it → fix (wire it live). Otherwise → delete.
4. **Duplication → fix** by convergence when both copies are load-bearing, **delete** when one is superseded.
5. **Naming, size, docs drift → fix**, batched per subtree.
6. **Anything whose fix would reshape an extension surface or the public API → defer** to the owning spec (SPEC-002/003/004/005/006) with a pointer, not fixed ad hoc.
7. **Cannot decide with these rules → flag** for user review. Deleting any user-visible feature is always flagged regardless of rules.

User review: all flagged findings, plus a 10% random sample of unflagged dispositions per track, drawn after merge. A failed spot-check (user overturns the disposition) triggers re-review of that track's dispositions against the corrected rule.

Merge: after tracks complete, a dedup pass joins findings that share a root cause (the merged record lists all locations), and the result is written to `specs/audit/BACKLOG.md`: one table, all findings, sorted by severity then effort, with disposition and status (open, in-progress, done, rejected). `BACKLOG.md` is the single source of truth for Phases 2 through 4 and is updated in the PR that resolves each item.

## 5. Async Spike Protocol (the P5 gate)

Purpose: decide async-native core vs contained portals with evidence, per the master design §8 tension T3.

**Per-adapter checklist** (langchain, claude_agent_sdk, claude_tool, deep_agents, langchain_deep_agents, manual, taskeval): each adapter is checked for

1. A native async invocation path (`ainvoke` or equivalent) that does not internally round-trip through a thread or portal.
2. Resource lifecycle compatible with a single event loop: client creation, use, and `aclose` on one loop, no thread-affinity assumptions.
3. Concurrency cap enforceable at the async leaf: the adapter's LLM calls can be wrapped by an `asyncio.Semaphore` without bypass (this is where F06's agent-bypass died).
4. MCP session attach and teardown on the driver's loop, where the adapter supports MCP.

**Prototype:** a thin asyncio driver (TaskGroup over a ready-set of tasks, one global `asyncio.Semaphore`) running two concrete workloads end to end with live calls: a 3-question QA benchmark via the langchain adapter against the Codon vLLM server, and one short scenario via the claude_agent_sdk adapter. No sinks, no resume, answer generation and parsing stages only. The prototype is throwaway evidence, not the future driver.

**Pass criterion:** every adapter passes checks 1 through 3 (check 4 where applicable), and both prototype workloads complete with correct results and a demonstrably enforced cap (log the max concurrent in-flight calls). Adapters failing a check get an itemized fix-cost estimate.

**Gate:** pass → ADR-001 commits to the async-native core (P5 stands as written). Fail with fix-cost S or M on the failing adapters → still pass, the fixes join the backlog. Fail with fix-cost L on any showcased-use-case adapter → fall back to contained portals (the original audit plan e), and ADR-001 records the failing evidence and the conditions for revisiting. P5's wording is amended through SPEC-001 §4 in that case.

## 6. Subagent Orchestration

- Tracks A through D run as parallel subagent workstreams, one dispatch per track (Track B fans out per subtree). Track E runs alongside, it shares no state with the others.
- Every high-severity finding gets an independent adversarial re-verification by a second agent before merge (prompted to refute, not confirm). Medium and low findings are merged on the original evidence.
- The merge, dedup, and `BACKLOG.md` assembly is one synthesis agent run, reviewed by the user together with the flagged findings and the random sample (§4).
- Conventions for all audit agents: read-only with respect to `src/` (audits never fix in place), findings written only in the §2 schema, citations against the pinned commit.

## 7. Open Questions

None. All charter questions answered above. The cull rubric referenced by Track C is owned by SPEC-007 and applied here once that spec is approved (Track C can run last, or provisionally with the rubric draft).
