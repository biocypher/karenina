# Karenina Agent Skills

This directory holds karenina's **agent skills**: instruction packs that teach a
coding agent how to drive the framework. One family covers *using* karenina
(building benchmarks, authoring templates and rubrics, configuring the pipeline,
reading results), the other covers *extending* it with a new adapter so karenina
can talk to another LLM SDK or agent framework.

The skills are versioned with the library, so they always match the API of the
commit you have checked out. They follow the [Agent Skills](https://agentskills.io)
format, so any compatible agent can load them.

## Using karenina

| Skill | Purpose |
|-------|---------|
| `using-karenina` | Router: picks the right workflow skill for a request, and carries the cross-cutting gotchas plus the full reference docs |
| `karenina-qa` | Single-turn QA benchmarks: questions, templates, checkpoints, runs |
| `karenina-scenarios` | Multi-turn branching scenario graphs, outcome criteria, TurnChecks |
| `karenina-task-eval` | Scoring pre-recorded outputs such as chat logs, agent traces, JSON exports |
| `karenina-template-authoring` | `BaseAnswer` and `VerifiedField` templates, verification primitives |
| `karenina-rubric-authoring` | Rubric traits (LLM, regex, callable, metric, agentic) and `DynamicRubric` |
| `karenina-verification` | `VerificationConfig`, adapters, guards, presets, resume |
| `karenina-results` | Loading, filtering, DataFrames, model comparison, export |
| `karenina-cli` | Terminal workflows: `karenina verify`, presets, progressive save and resume |
| `karenina-manual` | Full pipeline with manually supplied answers (`ManualAdapter`) |
| `testing-gate` | Pre-merge test battery: offline unit gate plus opt-in live tests |

## Extending karenina with a new adapter

| Skill | Purpose |
|-------|---------|
| `karenina-adapter-create` | Entry point, routes through the five phases below |
| `karenina-adapter-gather-context` | Phase 1: collect SDK capabilities and map them onto karenina ports |
| `karenina-adapter-design` | Phase 2: concept mapping and registry integration spec |
| `karenina-adapter-implement` | Phase 3: file-by-file implementation with conventions enforced |
| `karenina-adapter-test` | Phase 4: conformance suite, cold (mocked) and hot (live) tests |
| `karenina-adapter-review` | Phase 5: quality, correctness and convention review before merge |

## Installing

From a karenina checkout:

```bash
./skills/install.sh                      # Claude Code, this project
./skills/install.sh --agent codex        # Codex, Pi, Gemini CLI, OpenCode, ...
./skills/install.sh --scope user         # available in every project
./skills/install.sh --workflow-only      # skip the adapter development skills
./skills/install.sh --link               # symlink instead of copy, tracks this checkout
./skills/install.sh --help               # all options, including --dest and --force
```

Already installed skills are left alone unless you pass `--force`.

To install by hand, copy the skill directories into the location your agent scans:

| Agent | Project scope | User scope |
|-------|---------------|------------|
| Claude Code | `.claude/skills/` | `~/.claude/skills/` |
| Codex | `.agents/skills/` | `~/.agents/skills/` (also reads `~/.codex/skills/`) |
| Pi | `.agents/skills/` | `~/.agents/skills/` |
| Gemini CLI, OpenCode, Goose, Amp, ... | `.agents/skills/` | `~/.agents/skills/` |

```bash
mkdir -p .agents/skills
cp -r skills/karenina-* skills/using-karenina .agents/skills/
```

`.agents/skills/` is the shared convention that most agents now scan, so skills
installed there are visible to all of them at once. Project scope wins over user
scope when both define the same skill name. Restart your agent if it does not pick
up new skills automatically.

Copy only the skills you need. Every skill is self-contained, except that the usage
skills point at `using-karenina/references/` for deep dives, so keep
`using-karenina` alongside them.

Working inside this repository there is nothing to install: `.claude/skills` is
already a symlink to this directory.

## Keeping the reference docs in sync

`using-karenina/references/` is generated from `docs/`. Never edit it by hand. After
changing the docs, run:

```bash
make sync-skill-docs
```

## Without an agent

Each `SKILL.md` is plain Markdown with verified snippets, gotcha lists and API
tables, so the packs double as a task-oriented quick reference even when no coding
agent is involved.
