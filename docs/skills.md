# Agent Skills

Karenina ships a set of **agent skills** — structured instruction packs for
coding agents (Claude Code and compatible harnesses) — that encode the
correct karenina workflows: benchmark construction, template and rubric
authoring, pipeline configuration, result analysis, and the CLI. They live
in [`skills/`](https://github.com/biocypher/karenina/tree/main/skills) at the
repository root and are versioned together with the library, so they always
match the API of the commit you have checked out.

## What is included

| Skill | Purpose |
|-------|---------|
| `using-karenina` | Router: picks the right workflow skill from a request; carries cross-cutting gotchas and the full reference docs under `references/` |
| `karenina-qa` | Single-turn QA benchmarks: questions, templates, checkpoints, runs |
| `karenina-scenarios` | Multi-turn branching scenario graphs, outcome criteria, TurnChecks |
| `karenina-task-eval` | Scoring pre-recorded outputs (logs, traces, JSON exports) |
| `karenina-template-authoring` | `BaseAnswer` / `VerifiedField` templates and verification primitives |
| `karenina-rubric-authoring` | Rubric traits (LLM, regex, callable, metric, agentic) and `DynamicRubric` |
| `karenina-verification` | `VerificationConfig`, adapters, guards, presets, resume |
| `karenina-results` | Loading, filtering, DataFrames, comparison, export, repair |
| `karenina-cli` | Terminal workflows: `karenina verify`, presets, progressive save/resume |
| `karenina-manual` | Full pipeline with manually supplied answers (`ManualAdapter`) |
| `testing-gate` | Pre-merge test battery: offline unit gate + opt-in live tests |
| `karenina-adapter-*` | Building new karenina adapters (gather → design → implement → test → review) |

## Installation

From a karenina checkout, the installer copies (or symlinks) the skills into the
directory your agent scans:

```bash
./skills/install.sh                      # Claude Code, this project
./skills/install.sh --agent codex        # Codex, Pi, Gemini CLI, OpenCode, ...
./skills/install.sh --scope user         # available in every project
./skills/install.sh --workflow-only      # skip the adapter development skills
./skills/install.sh --help               # all options
```

To install by hand, copy the skill directories yourself:

| Agent | Project scope | User scope |
|-------|---------------|------------|
| Claude Code | `.claude/skills/` | `~/.claude/skills/` |
| Codex | `.agents/skills/` | `~/.agents/skills/` (also reads `~/.codex/skills/`) |
| Pi | `.agents/skills/` | `~/.agents/skills/` |
| Gemini CLI, OpenCode, Goose, Amp, ... | `.agents/skills/` | `~/.agents/skills/` |

```bash
mkdir -p .agents/skills
cp -r /path/to/karenina/skills/karenina-* /path/to/karenina/skills/using-karenina .agents/skills/
```

`.agents/skills/` is the shared convention that most agents now scan, so skills
installed there are visible to all of them at once.

**Working inside this repository** there is nothing to do: `.claude/skills` is a
symlink to `skills/`, so Claude Code discovers them automatically.

Copy only the skills you need. Every skill is self-contained, except that the
workflow skills refer to `using-karenina/references/` for deep-dive documentation,
so keep `using-karenina` alongside them.

## Keeping the reference docs in sync

`skills/using-karenina/references/` is generated from `docs/` — never edit it
by hand. After changing the docs, run:

```bash
make sync-skill-docs
```

## Using the skills without an agent

Each `SKILL.md` is plain Markdown and reads as a task-oriented guide with
verified code snippets, gotcha lists, and API tables. Even without a coding
agent they are a practical quick-reference for the workflows they cover.
